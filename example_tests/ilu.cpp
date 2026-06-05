#include <vector>
#include <unordered_map>
#include <set>
#include <map>
#include <mpi.h>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cmath>
#include "ilu.h"


struct ILUFact {
    // MPI & Matrix info
    int world_size, rank;
    int N; // Matrix size

    // Local matrix data
    int local_start, local_end; // rank own rows [local_start, local_end)
    int local_n; // number of rows owned by this rank

    // Local Permutation (Interior first)
    std::vector<int> perm;      // perm[permuted_i] = original_local_i
    std::vector<int> inv_perm;  // inv_perm[original_local_i] = permuted_i
    int n_int;                  // number of interior rows
    int n_sep;                  // number of separator rows

    // CSR Storage for L + U
    // Row order: permuted local indices [0..local_n-1]
    // Column indices: three partitions
    //   - [0, local_n): permuted local for [local_start, local_end)
    //   - [local_n, local_n + ext_cols_fwd.size()): E-block positions (col < local_start)
    //   - [local_n + ext_cols_fwd.size(), ...): F-block positions (col >= local_end)
    std::vector<int> lu_rowptr; // size local_n + 1
    std::vector<int> lu_col; // column indices (translated or local permuted)
    std::vector<double> lu_val; // L values (below diagonal) and U values (on and above diagonal)

    // Forward Pahse: External Columns (E Block)
    std::vector<int> ext_cols_fwd;      // sorted global column indices < local_start
    std::unordered_map<int, int> ext_col_to_local_fwd; // global col -> local permuted col
    std::vector<double> recv_buf_fwd; // received values, size = ext_cols_fwd.size()

    // Backward Phase: External Columns (F Block)
    std::vector<int> ext_cols_bwd;      // sorted global column indices >= local_end
    std::unordered_map<int, int> ext_col_to_local_bwd; // global col -> local permuted col
    std::vector<double> recv_buf_bwd; // received values, size = ext_cols_bwd.size()

    // External U Rows (for Factorization Iterations)
    // Flat mini-CSR for upper-triangular parts of rows < local_start
    std::vector<int> ext_rows_needed;   // sorted global row indices < local_start
    std::unordered_map<int, int> ext_row_to_pos;  // global row -> position in ext_rows_needed
    std::vector<int> ext_U_rowptr;      // row pointers, size = ext_rows_needed.size() + 1
    std::vector<int> ext_U_col;         // column indices (translated to positions in ext_cols_fwd)
    std::vector<double> ext_U_val;      // values (updated each factorization iteration)
     
    // Communication Plans: Forward Phase (Factorization + Ly=b)
    std::vector<int> fwd_dst_ranks;           // destination ranks (higher)
    std::vector<std::vector<int>> fwd_send_rows;  // rows to send to each dst rank
    std::vector<int> fwd_src_ranks;           // source ranks (lower)
    std::vector<std::vector<int>> fwd_recv_rows;  // rows to receive from each src rank
    

    // Communication Plans: Backward Phase (Ux=y)
    std::vector<int> bwd_dst_ranks;           // destination ranks (lower)
    std::vector<std::vector<int>> bwd_send_cols;  // columns to send to each dst rank
    std::vector<int> bwd_src_ranks;           // source ranks (higher)
    std::vector<std::vector<int>> bwd_recv_cols;  // columns to receive from each src rank

    // MPI Request Storage
    std::vector<MPI_Request> mpi_requests_fwd;
    std::vector<MPI_Request> mpi_requests_bwd;
};

// Only rank 0 calls this.
void read_coo(const char* filename, int& N, int& nnz, 
              std::vector<int>& row, std::vector<int>& col, std::vector<double>& val) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Error opening file %s\n", filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    char line[256];
    int line_num = 0;
    int entry_count = 0;

    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '%') continue;
        
        if (entry_count == 0) {
            // First data line: N M nnz
            int M;
            sscanf(line, "%d %d %d", &N, &M, &nnz);
            if (M != N) {
                fprintf(stderr, "Matrix must be square: %d x %d\n", N, M);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            row.resize(nnz);
            col.resize(nnz);
            val.resize(nnz);
        } else {
            // Data line: row col value (1-indexed, convert to 0-indexed)
            int r, c;
            double v;
            sscanf(line, "%d %d %lf", &r, &c, &v);
            row[entry_count - 1] = r - 1;
            col[entry_count - 1] = c - 1;
            val[entry_count - 1] = v;
        }
        entry_count++;
    }
    
    fclose(f);
}

void sort_coo(std::vector<int>& row, std::vector<int>& col, std::vector<double>& val) {
    int nnz = row.size();
    std::vector<int> perm(nnz);
    for (int i = 0; i < nnz; i++) perm[i] = i;
    
    // Sort permutation by (row[i], col[i])
    std::sort(perm.begin(), perm.end(), 
              [&](int i, int j) {
                  if (row[i] != row[j]) return row[i] < row[j];
                  return col[i] < col[j];
              });
    
    std::vector<int> row_sorted(nnz), col_sorted(nnz);
    std::vector<double> val_sorted(nnz);
    for (int i = 0; i < nnz; i++) {
        row_sorted[i] = row[perm[i]];
        col_sorted[i] = col[perm[i]];
        val_sorted[i] = val[perm[i]];
    }
    
    row = std::move(row_sorted);
    col = std::move(col_sorted);
    val = std::move(val_sorted);
}

// Build local CSR from COO slice.
// row[], col[], val[] contain entries with row indices in [local_start, local_end)
void build_local_csr(const std::vector<int>& row, const std::vector<int>& col, 
                     const std::vector<double>& val, int local_start, int local_end,
                     std::vector<int>& lu_rowptr, std::vector<int>& lu_col, 
                     std::vector<double>& lu_val) {
    int local_n = local_end - local_start;
    lu_rowptr.assign(local_n + 1, 0);

    // Count nonzeros per local row
    for (size_t i = 0; i < (int)row.size(); i++) {
        int local_row = row[i] - local_start;
        if (local_row >= 0 && local_row < local_n) {
            lu_rowptr[local_row + 1]++;
        }
    }

    // Convert counts to pointers
    for (int i = 0; i < local_n; i++) {
        lu_rowptr[i + 1] += lu_rowptr[i];
    }

    int total_nnz = lu_rowptr[local_n];
    lu_col.resize(total_nnz);
    lu_val.resize(total_nnz);

    // Fill column indices and values, entries are already sorted by row and then column
    std::vector<int> current_pos = lu_rowptr; // track current fill position for each row
    for (size_t i = 0; i < (int)row.size(); i++) {
        int local_row = row[i] - local_start;
        if (local_row >= 0 && local_row < local_n) {
            int pos = current_pos[local_row]++;
            lu_col[pos] = col[i];
            lu_val[pos] = val[i];
        }
    }
}


// Helper: Classify rows and build permutation
void classify_and_permute(const std::vector<int>& lu_rowptr, 
                          const std::vector<int>& lu_col,
                          int local_start, int local_end, int local_n,
                          std::vector<int>& perm, std::vector<int>& inv_perm,
                          int& n_int, int& n_sep) {
    perm.resize(local_n);
    inv_perm.resize(local_n);

    std::vector<int> interior_rows, separator_rows;

    for( int i = 0; i < local_n; i++) {
        int global_row = local_start + i;
        bool is_interior = true;

        for (int k = lu_rowptr[i]; k < lu_rowptr[i + 1]; k++) {
            int col = lu_col[k];
            if (col < local_start) {
                is_interior = false;
                break;
            }
        }

        if (is_interior) {
            interior_rows.push_back(i);
        } else {
            separator_rows.push_back(i);
        }
    }

    n_int = interior_rows.size();
    n_sep = separator_rows.size();

    // Build permutation: interior rows first, then separator rows
    for (int i = 0; i < n_int; i++) {
        perm[i] = interior_rows[i];
        inv_perm[interior_rows[i]] = i;
    }
    for (int i = 0; i < n_sep; i++) {
        perm[n_int + i] = separator_rows[i];
        inv_perm[separator_rows[i]] = n_int + i;
    }
}

// Apply symmetric permutation to local CSR.
// Columns are kept as global indices here; identify_and_translate_external_cols
// handles all column translation (including local->permuted) in one pass to avoid
// the encoding overlap that arises when permuted local indices [0,local_n) coincide
// with E-block global values for rank > 0.
void apply_permutation_to_csr(std::vector<int>& lu_rowptr,
                              std::vector<int>& lu_col,
                              std::vector<double>& lu_val,
                              const std::vector<int>& perm,
                              int local_n) {
    std::vector<int> new_rowptr(local_n + 1, 0);

    for (int permuted_i = 0; permuted_i < local_n; permuted_i++) {
        int original_i = perm[permuted_i];
        new_rowptr[permuted_i + 1] = lu_rowptr[original_i + 1] - lu_rowptr[original_i];
    }
    for (int i = 0; i < local_n; i++)
        new_rowptr[i + 1] += new_rowptr[i];

    int total_nnz = new_rowptr[local_n];
    std::vector<int> new_col(total_nnz);
    std::vector<double> new_val(total_nnz);

    for (int permuted_i = 0; permuted_i < local_n; permuted_i++) {
        int original_i = perm[permuted_i];
        int base = new_rowptr[permuted_i];
        int row_nnz = new_rowptr[permuted_i + 1] - base;

        std::vector<std::pair<int, double>> entries;
        entries.reserve(row_nnz);
        for (int k = lu_rowptr[original_i]; k < lu_rowptr[original_i + 1]; k++)
            entries.push_back({lu_col[k], lu_val[k]});  // keep columns as global

        std::sort(entries.begin(), entries.end());

        for (int j = 0; j < row_nnz; j++) {
            new_col[base + j] = entries[j].first;
            new_val[base + j] = entries[j].second;
        }
    }

    lu_rowptr = std::move(new_rowptr);
    lu_col    = std::move(new_col);
    lu_val    = std::move(new_val);
}

// Helper: Identify external columns and translate ALL column indices to the
// three-partition encoding:
//   [0, local_n)                      — permuted local
//   [local_n, local_n + k_fwd)        — E-block (col < local_start)
//   [local_n + k_fwd, ...)            — F-block (col >= local_end)
//
// Must be called AFTER apply_permutation_to_csr which keeps columns as global.
// Translates local columns here (using inv_perm) to avoid the overlap between
// permuted local indices [0,local_n) and E-block global values for rank > 0.
// Re-sorts each row after translation since encoding changes relative order.
void identify_and_translate_external_cols(const std::vector<int>& lu_rowptr,
                                          std::vector<int>& lu_col,
                                          std::vector<double>& lu_val,
                                          int local_start, int local_end, int local_n,
                                          const std::vector<int>& inv_perm,
                                          std::vector<int>& ext_cols_fwd,
                                          std::vector<int>& ext_cols_bwd,
                                          std::unordered_map<int, int>& ext_col_to_local_fwd,
                                          std::unordered_map<int, int>& ext_col_to_local_bwd) {
    // Pass 1: collect unique external column globals
    std::set<int> ext_fwd_set, ext_bwd_set;
    for (int col : lu_col) {
        if (col >= local_start && col < local_end) continue;
        if (col < local_start)  ext_fwd_set.insert(col);
        else                    ext_bwd_set.insert(col);
    }

    ext_cols_fwd.assign(ext_fwd_set.begin(), ext_fwd_set.end());
    ext_cols_bwd.assign(ext_bwd_set.begin(), ext_bwd_set.end());

    for (int i = 0; i < (int)ext_cols_fwd.size(); i++)
        ext_col_to_local_fwd[ext_cols_fwd[i]] = i;
    for (int i = 0; i < (int)ext_cols_bwd.size(); i++)
        ext_col_to_local_bwd[ext_cols_bwd[i]] = i;

    const int k_fwd = (int)ext_cols_fwd.size();

    // Pass 2: translate every column and re-sort each row.
    // After translation the partition order is local(small) < E-block(mid) < F-block(large),
    // which differs from the global sort order, so an explicit per-row sort is required.
    for (int i = 0; i < local_n; i++) {
        int base = lu_rowptr[i];
        int end  = lu_rowptr[i + 1];
        int row_nnz = end - base;

        std::vector<std::pair<int, double>> entries;
        entries.reserve(row_nnz);

        for (int k = base; k < end; k++) {
            int c = lu_col[k];
            int new_c;
            if (c >= local_start && c < local_end)
                new_c = inv_perm[c - local_start];              // local → [0, local_n)
            else if (c < local_start)
                new_c = local_n + ext_col_to_local_fwd[c];     // E-block
            else
                new_c = local_n + k_fwd + ext_col_to_local_bwd[c]; // F-block
            entries.push_back({new_c, lu_val[k]});
        }

        std::sort(entries.begin(), entries.end());

        for (int j = 0; j < row_nnz; j++) {
            lu_col[base + j] = entries[j].first;
            lu_val[base + j] = entries[j].second;
        }
    }
}

// Helper: Safe rank calculation
int get_owner_rank(int global_index, int world_size, int N) {
    int rank = 0;
    while (rank < world_size - 1 && 
           (rank + 1) * N / world_size <= global_index) {
        rank++;
    }
    return rank;
}

// Setup Backward Communication
void setup_backward_communication(const std::vector<int>& ext_cols_bwd,
                                  int local_start, int local_end, int N,
                                  int world_size, int rank,
                                  std::vector<int>& bwd_src_ranks,
                                  std::vector<std::vector<int>>& bwd_recv_cols,
                                  std::vector<int>& bwd_dst_ranks,
                                  std::vector<std::vector<int>>& bwd_send_cols) {
    // Identify which columns we need form higher ranks
    std::map<int, std::vector<int>> cols_needed_from_rank;

    for (int col : ext_cols_bwd) {
        int owner_rank = get_owner_rank(col, world_size, N);
        cols_needed_from_rank[owner_rank].push_back(col);
    }

    // Build receive plans
    for (const auto& [src_rank, cols] : cols_needed_from_rank) {
        bwd_src_ranks.push_back(src_rank);
        bwd_recv_cols.push_back(cols);
    }

    // Exchange request counts
    std::vector<int> send_counts(world_size, 0);
    for (const auto& [src_rank, cols] : cols_needed_from_rank) {
        send_counts[src_rank] = cols.size();
    }

    std::vector<int> recv_counts(world_size, 0);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                 recv_counts.data(), 1, MPI_INT,
                 MPI_COMM_WORLD);

    // Prepare buffer for MPI_Alltoallv
    std::vector<int> send_displs(world_size + 1, 0);
    for (int r = 0; r < world_size; r++) {
        send_displs[r + 1] = send_displs[r] + send_counts[r];
    }

    std::vector<int> send_buffer(send_displs[world_size]);

    // For each rank, write the columns we need to receive into the send buffer of that rank
    std::map<int, int> rank_write_pos;
    for (const auto& [src_rank, cols] : cols_needed_from_rank) {
        for (int col : cols) {
            send_buffer[send_displs[src_rank] + rank_write_pos[src_rank]++] = col;
        }
    }

    // MPI_Alltoallv to exchange column requests    
    std::vector<int> recv_displs(world_size + 1, 0);
    for (int r = 0; r < world_size; r++) {
        recv_displs[r + 1] = recv_displs[r] + recv_counts[r];
    }
    std::vector<int> recv_buffer(recv_displs[world_size]);

    MPI_Alltoallv(send_buffer.data(), send_counts.data(), send_displs.data(), MPI_INT,
                  recv_buffer.data(), recv_counts.data(), recv_displs.data(), MPI_INT,
                  MPI_COMM_WORLD);

    // Unpack received buffer
    std::map<int, std::vector<int>> cols_needed_by_rank;
    for (int r = 0; r < world_size; r++) {
        if (recv_counts[r] > 0) {
            for (int i = recv_displs[r]; i < recv_displs[r + 1]; i++) {
                cols_needed_by_rank[r].push_back(recv_buffer[i]);
            }
        }
    }

    bwd_dst_ranks.clear();
    bwd_send_cols.clear();

    for (const auto& [dst_rank, cols] : cols_needed_by_rank) {
        // Sort and deduplicate 
        std::vector<int> sorted_cols = cols;
        std::sort(sorted_cols.begin(), sorted_cols.end());
        sorted_cols.erase(std::unique(sorted_cols.begin(), sorted_cols.end()), 
                          sorted_cols.end());
        
        bwd_dst_ranks.push_back(dst_rank);
        bwd_send_cols.push_back(sorted_cols);
    }
}

void setup_forward_communication(const std::vector<int>& lu_rowptr,
                                 const std::vector<int>& lu_col,
                                 const std::vector<int>& ext_cols_fwd,
                                 int local_start, int local_end, int N,
                                 int world_size, int rank, int n_int, int n_sep,
                                 std::vector<int>& ext_rows_needed,
                                 std::vector<int>& fwd_src_ranks,
                                 std::vector<std::vector<int>>& fwd_recv_rows,
                                 std::vector<int>& fwd_dst_ranks,
                                 std::vector<std::vector<int>>& fwd_send_rows) {
    // Identify which rows we need from lower ranks
    std::set<int> ext_rows_set;

    for (int perm_i = n_int; perm_i < n_int + n_sep; perm_i++) {
        for( int k = lu_rowptr[perm_i]; k < lu_rowptr[perm_i + 1]; k++) {
            int col_idx = lu_col[k];
            // E-block columns are in range [local_n, local_n + ext_cols_fwd.size())
            int local_n = lu_rowptr.size() - 1;
            if (col_idx >= local_n && col_idx < local_n + (int)ext_cols_fwd.size()) {
                int global_col = ext_cols_fwd[col_idx - local_n];
                ext_rows_set.insert(global_col);
            }
        }
    }

    ext_rows_needed.assign(ext_rows_set.begin(), ext_rows_set.end());

    // Group needed rows by source rank
    std::map<int, std::vector<int>> rows_needed_from_rank;
    for (int row : ext_rows_needed) {
        int owner_rank = get_owner_rank(row, world_size, N);
        rows_needed_from_rank[owner_rank].push_back(row);
    }

    // Build receive plans
    for (const auto& [src_rank, rows] : rows_needed_from_rank) {
        fwd_src_ranks.push_back(src_rank);
        fwd_recv_rows.push_back(rows);
    }

    // Exchange request counts
    std::vector<int> send_counts(world_size, 0);
    for (const auto& [src_rank, rows] : rows_needed_from_rank) {
        send_counts[src_rank] = rows.size();
    }

    std::vector<int> recv_counts(world_size, 0);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                 recv_counts.data(), 1, MPI_INT,
                 MPI_COMM_WORLD);

    // Prepare buffer for MPI_Alltoallv
    std::vector<int> send_displs(world_size + 1, 0);
    for (int r = 0; r < world_size; r++) {
        send_displs[r + 1] = send_displs[r] + send_counts[r];
    }

    std::vector<int> send_buffer(send_displs[world_size]);

    std::map<int, int> rank_write_pos;
    for (const auto& [src_rank, rows] : rows_needed_from_rank) {
        for (int row : rows) {
            send_buffer[send_displs[src_rank] + rank_write_pos[src_rank]++] = row;
        }
    }

    // MPI_Alltoallv to exchange row requests
    std::vector<int> recv_displs(world_size + 1, 0);
    for (int r = 0; r < world_size; r++) {
        recv_displs[r+1] = recv_displs[r] + recv_counts[r];
    }

    std::vector<int> recv_buffer(recv_displs[world_size]);

    MPI_Alltoallv(send_buffer.data(), send_counts.data(), send_displs.data(), MPI_INT,
                  recv_buffer.data(), recv_counts.data(), recv_displs.data(), MPI_INT,
                  MPI_COMM_WORLD);

    // Unpack received buffer
    std::map<int, std::vector<int>> rows_requested_by_rank;
    for (int r = 0; r < world_size; r++) {
        if (recv_counts[r] > 0) {
            for (int i = recv_displs[r]; i < recv_displs[r + 1]; i++) {
                rows_requested_by_rank[r].push_back(recv_buffer[i]);
            }
        }
    }

    fwd_dst_ranks.clear();
    fwd_send_rows.clear();

    for (const auto& [dst_rank, rows] : rows_requested_by_rank) {
        // Sort and deduplicate 
        std::vector<int> sorted_rows = rows;
        std::sort(sorted_rows.begin(), sorted_rows.end());
        sorted_rows.erase(std::unique(sorted_rows.begin(), sorted_rows.end()), 
                          sorted_rows.end());
        
        fwd_dst_ranks.push_back(dst_rank);
        fwd_send_rows.push_back(sorted_rows);
    }
}

void factorize_interior(std::vector<int>& lu_rowptr,
                        std::vector<int>& lu_col,
                        std::vector<double>& lu_val,
                        int local_n, int n_int) {
    // Pre-computed index of diagonal entry for each interior row
    std::vector<int> diag_ptr(n_int, -1);
    for (int i = 0; i < n_int; i++) {
        for (int k = lu_rowptr[i]; k < lu_rowptr[i + 1]; k++) {
            if (lu_col[k] == i) {
                diag_ptr[i] = k;
                break;
            }
        }
    }

    for (int perm_i = 0; perm_i < n_int; perm_i++) {
        for (int k = lu_rowptr[perm_i]; k < lu_rowptr[perm_i + 1]; k++) {
            int perm_col = lu_col[k];

            if (perm_col >= perm_i) break; // only process columns before diagonal

            int diag_idx = diag_ptr[perm_col];
            if (diag_idx == -1) continue;

            lu_val[k] /= lu_val[diag_idx];
            double l_val = lu_val[k];

            int m = lu_rowptr[perm_i];
            int m_end = lu_rowptr[perm_i + 1];

            for(int j = diag_idx + 1; j < lu_rowptr[perm_col + 1]; j++) {
                int col_j = lu_col[j];
                while (m < m_end && lu_col[m] < col_j) {
                    m++;
                }
                if (m < m_end && lu_col[m] == col_j) {
                    lu_val[m] -= l_val * lu_val[j];
                }
            }
        }
    }
}

void factorize_separators_pass(ILUFact* ilu, const std::vector<int>& diag_ptr) {
    const int local_n = ilu->local_n;
    const int num_ext_fwd = ilu->ext_cols_fwd.size();

    for (int perm_i = ilu->n_int; perm_i < local_n; perm_i++) {
        int diag_m = diag_ptr[perm_i];
        if (diag_m == -1) continue;

        // Iterate over entire row, classify entries into local L and E-block
        for (int k = ilu->lu_rowptr[perm_i]; k < ilu->lu_rowptr[perm_i + 1]; k++) {
            int col_idx = ilu->lu_col[k];
            
            bool is_local_lower = (col_idx < perm_i);
            bool is_e_block = (col_idx >= local_n && col_idx < local_n + num_ext_fwd);
            if (!is_local_lower && !is_e_block) continue;

            int pivot_start, pivot_end;
            const int* pivot_cols;
            const double* pivot_vals;
            double diag_val;

            if (is_e_block) {
                int ext_idx = col_idx - local_n;
                pivot_start = ilu->ext_U_rowptr[ext_idx];
                pivot_end = ilu->ext_U_rowptr[ext_idx + 1];
                pivot_cols = ilu->ext_U_col.data();
                pivot_vals = ilu->ext_U_val.data();
                diag_val = pivot_vals[pivot_start];
            } else {
                int pivot_diag = diag_ptr[col_idx];
                if (pivot_diag == -1) continue;
                pivot_start = pivot_diag;
                pivot_end = ilu->lu_rowptr[col_idx + 1];
                pivot_cols = ilu->lu_col.data();
                pivot_vals = ilu->lu_val.data();
                diag_val = pivot_vals[pivot_diag];
            }

            ilu->lu_val[k] /= diag_val;
            double l_val = ilu->lu_val[k];

            // For local pivots: start from row beginning so cross-updates between L entries work.
            // For E-block pivots: start from diagonal — ext_U_col can map to already-processed
            // local L positions (e.g. 3D Laplacian last boundary layer), starting before
            // diag_m would corrupt them; by this point all local lower-tri entries are frozen.
            int m = is_e_block ? diag_m : ilu->lu_rowptr[perm_i];
            int m_end = ilu->lu_rowptr[perm_i + 1];

            for (int j = pivot_start + 1; j < pivot_end; j++) {
                int col_j = pivot_cols[j];
                // Skip E-block of pivot row
                if (col_j >= local_n && col_j < local_n + num_ext_fwd) continue;
                while (m < m_end) {
                    int mc = ilu->lu_col[m];
                    if (mc >= local_n && mc < local_n + num_ext_fwd) { m++; continue; }
                    if (mc >= col_j) break;  // stop AT col_j, not past it
                    m++;
                }
                if (m < m_end && ilu->lu_col[m] == col_j) {
                    ilu->lu_val[m] -= l_val * pivot_vals[j];
                }
            }
        }
    }
}

void iterative_separator_factorization(ILUFact* ilu) {
    std::vector<int> diag_ptr(ilu->local_n, -1);
    for (int i = 0; i < ilu->local_n; i++) {
        for (int k = ilu->lu_rowptr[i]; k < ilu->lu_rowptr[i + 1]; k++) {
            if (ilu->lu_col[k] == i) {
                diag_ptr[i] = k;
                break;
            }
        }
    }

    // Prepare for communcation
    int num_fwd_src = ilu->fwd_src_ranks.size();
    int num_fwd_dst = ilu->fwd_dst_ranks.size();

    std::vector<int> recv_counts(num_fwd_src, 0);
    std::vector<int> recv_offsets(num_fwd_src + 1, 0);
    int current_ext_row = 0;

    for (int i = 0; i < num_fwd_src; i++) {
        int num_rows_from_rank = ilu->fwd_recv_rows[i].size();
        recv_counts[i] = ilu->ext_U_rowptr[current_ext_row + num_rows_from_rank] -
                         ilu->ext_U_rowptr[current_ext_row];
        current_ext_row += num_rows_from_rank;
    }
    // Prefix-sum to get byte offsets into ext_U_val for each source rank
    recv_offsets[0] = 0;
    for (int i = 0; i < num_fwd_src; i++)
        recv_offsets[i + 1] = recv_offsets[i] + recv_counts[i];

    const int local_n = ilu->local_n;
    const int num_ext_fwd = (int)ilu->ext_cols_fwd.size();

    std::vector<int> send_counts(num_fwd_dst, 0);
    std::vector<int> send_offsets(num_fwd_dst + 1, 0);
    int total_send = 0;

    for (int i = 0; i < num_fwd_dst; i++) {
        send_offsets[i] = total_send;
        int count = 0;
        for (int global_row : ilu->fwd_send_rows[i]) {
            int local_row = global_row - ilu->local_start;
            int perm_i = ilu->inv_perm[local_row];
            int diag_m = diag_ptr[perm_i];

            // Send diagonal and above, but skip E-block columns
            for (int k = diag_m; k < ilu->lu_rowptr[perm_i + 1]; k++) {
                int col_idx = ilu->lu_col[k];
                if (col_idx >= local_n && col_idx < local_n + num_ext_fwd) continue;
                count++;
            }
        }
        send_counts[i] = count;
        total_send += count;
    }

    std::vector<double> send_buffer(total_send, 0.0);
    std::vector<MPI_Request> mpi_requests;
    mpi_requests.reserve(num_fwd_src + num_fwd_dst);

    int max_iter = 50;
    double tol = 1e-8;
    int iter = 0;
    bool converged = false;

    // Snapshot of separator lu_val as it stands after factorize_interior (original A values
    // for those rows).  Must reset before every pass so the L-entry divisions and U-entry
    // subtractions always start from the same base — without this, repeated calls would
    // divide L entries by U_ii again each iteration, driving them to zero.
    const int sep_val_start = ilu->lu_rowptr[ilu->n_int];
    const int sep_val_end   = ilu->lu_rowptr[local_n];
    std::vector<double> orig_sep_lu_val(ilu->lu_val.begin() + sep_val_start,
                                        ilu->lu_val.begin() + sep_val_end);

    std::vector<double> prev_U_sep(ilu->lu_val.size(), 0.0);

    while (iter < max_iter && !converged) {
        mpi_requests.clear();

        // Receive U rows from lower-ranked sources into ext_U_val
        for (int i = 0; i < num_fwd_src; i++) {
            if (recv_counts[i] > 0) {
                MPI_Request req;
                MPI_Irecv(&ilu->ext_U_val[recv_offsets[i]], recv_counts[i], MPI_DOUBLE,
                          ilu->fwd_src_ranks[i], 0, MPI_COMM_WORLD, &req);
                mpi_requests.push_back(req);
            }
        }

        int buf_idx = 0;
        for (int i = 0; i < num_fwd_dst; i++) {
            for (int global_row : ilu->fwd_send_rows[i]) {
                int local_row = global_row - ilu->local_start;
                int perm_i = ilu->inv_perm[local_row];
                int diag_m = diag_ptr[perm_i];

                for (int k = diag_m; k < ilu->lu_rowptr[perm_i + 1]; k++) {
                    int col_idx = ilu->lu_col[k];
                    if (col_idx >= local_n && col_idx < local_n + num_ext_fwd) continue;

                    send_buffer[buf_idx++] = ilu->lu_val[k];
                }
            }
        }

        for (int i = 0; i < num_fwd_dst; i++) {
            if (send_counts[i] > 0) {
                MPI_Request req;
                MPI_Isend(&send_buffer[send_offsets[i]], send_counts[i], MPI_DOUBLE,
                          ilu->fwd_dst_ranks[i], 0, MPI_COMM_WORLD, &req);
                mpi_requests.push_back(req);
            }
        }

        if (!mpi_requests.empty()) {
            MPI_Waitall(mpi_requests.size(), mpi_requests.data(), MPI_STATUSES_IGNORE);
        }

        // Save current factorized U values for convergence comparison, then reset
        // separator rows to original A so factorize_separators_pass always starts clean.
        for (int perm_i = ilu->n_int; perm_i < local_n; perm_i++) {
            int diag_m = diag_ptr[perm_i];
            if (diag_m == -1) continue;

            for (int k = diag_m; k < ilu->lu_rowptr[perm_i + 1]; k++) {
                int col_idx = ilu->lu_col[k];
                // Skip E-block columns
                if (col_idx >= local_n && col_idx < local_n + num_ext_fwd) continue;

                prev_U_sep[k] = ilu->lu_val[k];
            }
        }

        // Reset all separator row entries (L and U) to original A values
        std::copy(orig_sep_lu_val.begin(), orig_sep_lu_val.end(),
                  ilu->lu_val.begin() + sep_val_start);

        factorize_separators_pass(ilu, diag_ptr);

        double local_diff_sq = 0.0;
        double local_norm_sq = 0.0;

        for (int perm_i = ilu->n_int; perm_i < local_n; perm_i++) {
            int diag_m = diag_ptr[perm_i];
            if (diag_m == -1) continue;
            
            for (int k = diag_m; k < ilu->lu_rowptr[perm_i + 1]; k++) {
                int col_idx = ilu->lu_col[k];
                // Skip E-block columns
                if (col_idx >= local_n && col_idx < local_n + num_ext_fwd) continue;

                double current_val = ilu->lu_val[k];
                double diff = current_val - prev_U_sep[k];
                
                local_diff_sq += diff * diff;
                local_norm_sq += current_val * current_val;
            }
        }
        double global_diff_sq = 0.0;
        double global_norm_sq = 0.0;

        MPI_Allreduce(&local_diff_sq, &global_diff_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_norm_sq, &global_norm_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double relative_error = sqrt(global_diff_sq) / (sqrt(global_norm_sq) + 1e-20);
        if (ilu->rank == 0) {
            printf("Iter %d: Relative change in U separators = %e\n", iter, relative_error);
        }

        if (relative_error < tol) {
            converged = true;
        }

        iter++;
    }
}

// One-time setup: exchange the sparsity structure of external U rows.
// Builds ext_row_to_pos, ext_U_rowptr, ext_U_col, ext_U_val so that
// iterative_separator_factorization can receive only values each iteration.
void setup_ext_U_structure(ILUFact* ilu) {
    const int local_n      = ilu->local_n;
    const int num_ext_fwd  = (int)ilu->ext_cols_fwd.size();
    const int world_size   = ilu->world_size;
    const int num_fwd_dst  = (int)ilu->fwd_dst_ranks.size();
    const int num_fwd_src  = (int)ilu->fwd_src_ranks.size();

    // Build ext_row_to_pos
    for (int j = 0; j < (int)ilu->ext_rows_needed.size(); j++)
        ilu->ext_row_to_pos[ilu->ext_rows_needed[j]] = j;

    // Precompute diagonal pointers
    std::vector<int> diag_ptr(local_n, -1);
    for (int i = 0; i < local_n; i++)
        for (int k = ilu->lu_rowptr[i]; k < ilu->lu_rowptr[i + 1]; k++)
            if (ilu->lu_col[k] == i) { diag_ptr[i] = k; break; }

    // Build per-row col counts and global col indices to send to each dst rank.
    std::vector<int> send_row_cnt_buf;
    std::vector<int> send_col_buf;
    std::vector<int> cnt_send_counts(world_size, 0);
    std::vector<int> col_send_counts(world_size, 0);

    for (int i = 0; i < num_fwd_dst; i++) {
        int dst = ilu->fwd_dst_ranks[i];
        for (int global_row : ilu->fwd_send_rows[i]) {
            int perm_i  = ilu->inv_perm[global_row - ilu->local_start];
            int diag_m  = diag_ptr[perm_i];
            int count   = 0;
            for (int k = diag_m; k < ilu->lu_rowptr[perm_i + 1]; k++) {
                int col_idx = ilu->lu_col[k];
                if (col_idx >= local_n && col_idx < local_n + num_ext_fwd) continue;
                int global_col = (col_idx < local_n)
                    ? ilu->local_start + ilu->perm[col_idx]
                    : ilu->ext_cols_bwd[col_idx - local_n - num_ext_fwd];
                send_col_buf.push_back(global_col);
                count++;
            }
            send_row_cnt_buf.push_back(count);
            cnt_send_counts[dst]++;
            col_send_counts[dst] += count;
        }
    }

    // Phase A: exchange per-row col counts
    std::vector<int> cnt_recv_counts(world_size, 0);
    MPI_Alltoall(cnt_send_counts.data(), 1, MPI_INT,
                 cnt_recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> cnt_send_displs(world_size + 1, 0), cnt_recv_displs(world_size + 1, 0);
    for (int r = 0; r < world_size; r++) {
        cnt_send_displs[r+1] = cnt_send_displs[r] + cnt_send_counts[r];
        cnt_recv_displs[r+1] = cnt_recv_displs[r] + cnt_recv_counts[r];
    }
    std::vector<int> recv_row_cnt_buf(cnt_recv_displs[world_size]);
    MPI_Alltoallv(send_row_cnt_buf.data(), cnt_send_counts.data(), cnt_send_displs.data(), MPI_INT,
                  recv_row_cnt_buf.data(), cnt_recv_counts.data(), cnt_recv_displs.data(), MPI_INT,
                  MPI_COMM_WORLD);

    // Phase B: exchange column indices (as globals)
    std::vector<int> col_recv_counts(world_size, 0);
    MPI_Alltoall(col_send_counts.data(), 1, MPI_INT,
                 col_recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> col_send_displs(world_size + 1, 0), col_recv_displs(world_size + 1, 0);
    for (int r = 0; r < world_size; r++) {
        col_send_displs[r+1] = col_send_displs[r] + col_send_counts[r];
        col_recv_displs[r+1] = col_recv_displs[r] + col_recv_counts[r];
    }
    std::vector<int> recv_col_buf(col_recv_displs[world_size]);
    MPI_Alltoallv(send_col_buf.data(), col_send_counts.data(), col_send_displs.data(), MPI_INT,
                  recv_col_buf.data(), col_recv_counts.data(), col_recv_displs.data(), MPI_INT,
                  MPI_COMM_WORLD);

    // Build ext_U_rowptr from received per-row counts
    const int num_ext_rows = (int)ilu->ext_rows_needed.size();
    ilu->ext_U_rowptr.assign(num_ext_rows + 1, 0);
    for (int i = 0; i < num_fwd_src; i++) {
        int src = ilu->fwd_src_ranks[i];
        for (int j = 0; j < (int)ilu->fwd_recv_rows[i].size(); j++) {
            int ext_idx = ilu->ext_row_to_pos[ilu->fwd_recv_rows[i][j]];
            ilu->ext_U_rowptr[ext_idx + 1] = recv_row_cnt_buf[cnt_recv_displs[src] + j];
        }
    }
    for (int j = 0; j < num_ext_rows; j++)
        ilu->ext_U_rowptr[j+1] += ilu->ext_U_rowptr[j];

    const int total_ext_nnz = ilu->ext_U_rowptr[num_ext_rows];
    ilu->ext_U_col.resize(total_ext_nnz);
    ilu->ext_U_val.assign(total_ext_nnz, 0.0);

    // Build ext_U_col: translate received global col indices to 3-partition encoding
    for (int i = 0; i < num_fwd_src; i++) {
        int src        = ilu->fwd_src_ranks[i];
        int col_pos    = col_recv_displs[src];
        for (int j = 0; j < (int)ilu->fwd_recv_rows[i].size(); j++) {
            int ext_idx  = ilu->ext_row_to_pos[ilu->fwd_recv_rows[i][j]];
            int row_start = ilu->ext_U_rowptr[ext_idx];
            int row_size  = recv_row_cnt_buf[cnt_recv_displs[src] + j];
            for (int jj = 0; jj < row_size; jj++) {
                int gc = recv_col_buf[col_pos++];
                int enc;
                if (gc < ilu->local_start) {
                    auto it = ilu->ext_col_to_local_fwd.find(gc);
                    enc = (it != ilu->ext_col_to_local_fwd.end())
                        ? local_n + it->second
                        : -1;  // not referenced by any local row; skip in factorize pass
                } else if (gc < ilu->local_end) {
                    enc = ilu->inv_perm[gc - ilu->local_start];
                } else {
                    auto it = ilu->ext_col_to_local_bwd.find(gc);
                    enc = (it != ilu->ext_col_to_local_bwd.end())
                        ? local_n + num_ext_fwd + it->second
                        : -1;  // not referenced by any local row; skip in factorize pass
                }
                ilu->ext_U_col[row_start + jj] = enc;
            }
        }
    }
}

struct ILUFact* ILU_factorize(int N, int nnz, int* row, int* col, double* val) {
    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ILUFact* ilu = new ILUFact();
    ilu->world_size = world_size;
    ilu->rank = rank;

    // Rank 0 read and sort COO data
    std::vector<int> row_global, col_global;
    std::vector<double> val_global;

    if (rank == 0) {
        row_global.assign(row, row + nnz);
        col_global.assign(col, col + nnz);
        val_global.assign(val, val + nnz);
        sort_coo(row_global, col_global, val_global);
    }

    // Broadcast matrix size to all ranks before storing in struct
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    ilu->N = N;

    // Compute row distribution
    ilu->local_start = rank * N / world_size;
    ilu->local_end = (rank + 1) * N / world_size;
    ilu->local_n = ilu->local_end - ilu->local_start;

    // Rank 0 partition COO data and send local slices to each rank
    std::vector<int> local_row_coo, local_col_coo;
    std::vector<double> local_val_coo;

    if(rank == 0) {
        std::vector<std::vector<int>> row_slices(world_size);
        std::vector<std::vector<int>> col_slices(world_size);
        std::vector<std::vector<double>> val_slices(world_size);

        int target_rank = 0;
        for (int i = 0; i < nnz; i++) {
            int r = row_global[i];
            
            while (target_rank < world_size - 1 &&
                   (target_rank + 1) * N / world_size <= r) {
                target_rank++;
            }
            row_slices[target_rank].push_back(r);
            col_slices[target_rank].push_back(col_global[i]);
            val_slices[target_rank].push_back(val_global[i]);
        }

        // Send slices to each rank
        for (int r = 0; r < world_size; r++) {
            int slice_nnz = row_slices[r].size(); 
            if (r == 0) {
                local_row_coo = row_slices[r];
                local_col_coo = col_slices[r];
                local_val_coo = val_slices[r];
            } else {
                MPI_Send(&slice_nnz, 1, MPI_INT, r, 0, MPI_COMM_WORLD);
                if (slice_nnz > 0) {
                    MPI_Send(row_slices[r].data(), slice_nnz, MPI_INT, r, 0, MPI_COMM_WORLD);
                    MPI_Send(col_slices[r].data(), slice_nnz, MPI_INT, r, 0, MPI_COMM_WORLD);
                    MPI_Send(val_slices[r].data(), slice_nnz, MPI_DOUBLE, r, 0, MPI_COMM_WORLD);
                }
            }
        }
    } else {
        int slice_nnz;
        MPI_Recv(&slice_nnz, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_row_coo.resize(slice_nnz);
        local_col_coo.resize(slice_nnz);
        local_val_coo.resize(slice_nnz);
        if (slice_nnz > 0) {
            MPI_Recv(local_row_coo.data(), slice_nnz, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(local_col_coo.data(), slice_nnz, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(local_val_coo.data(), slice_nnz, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } 
    }

    // Build local CSR
    build_local_csr(local_row_coo, local_col_coo, local_val_coo, 
                    ilu->local_start, ilu->local_end, 
                    ilu->lu_rowptr, ilu->lu_col, ilu->lu_val);
            

    // Classify rows and build permutation
    classify_and_permute(ilu->lu_rowptr, ilu->lu_col,
                        ilu->local_start, ilu->local_end, ilu->local_n,
                        ilu->perm, ilu->inv_perm,
                        ilu->n_int, ilu->n_sep);

    // Apply symmetric permutation to CSR 
    apply_permutation_to_csr(ilu->lu_rowptr, ilu->lu_col, ilu->lu_val,
                             ilu->perm, ilu->local_n);

    // Identify external columns and translate ALL columns to the three-partition encoding
    identify_and_translate_external_cols(ilu->lu_rowptr, ilu->lu_col, ilu->lu_val,
                                         ilu->local_start, ilu->local_end, ilu->local_n,
                                         ilu->inv_perm,
                                         ilu->ext_cols_fwd, ilu->ext_cols_bwd,
                                         ilu->ext_col_to_local_fwd,
                                         ilu->ext_col_to_local_bwd);
                                        
    // Setup communication plans
    setup_forward_communication(ilu->lu_rowptr, ilu->lu_col, ilu->ext_cols_fwd,
                                ilu->local_start, ilu->local_end, ilu->N,
                                world_size, rank, ilu->n_int, ilu->n_sep,
                                ilu->ext_rows_needed,
                                ilu->fwd_src_ranks, ilu->fwd_recv_rows,
                                ilu->fwd_dst_ranks, ilu->fwd_send_rows);

    setup_backward_communication(ilu->ext_cols_bwd,
                                 ilu->local_start, ilu->local_end, ilu->N,
                                 world_size, rank,
                                 ilu->bwd_src_ranks, ilu->bwd_recv_cols,
                                 ilu->bwd_dst_ranks, ilu->bwd_send_cols);

    // Setup ext_U structure for separator factorization
    setup_ext_U_structure(ilu);

    // Interior factorization
    factorize_interior(ilu->lu_rowptr, ilu->lu_col, ilu->lu_val,
                       ilu->local_n, ilu->n_int);

    // Separator factorization
    iterative_separator_factorization(ilu);

    return ilu;
}

void ILU_forward_sweep(ILUFact* ilu, const double* b_perm, double* y_perm) {
    int num_fwd_src = ilu->fwd_src_ranks.size();
    int num_fwd_dst = ilu->fwd_dst_ranks.size();
    int local_n = ilu->local_n;
    int num_ext_fwd = ilu->ext_cols_fwd.size();

    // Solve interior rows exactly once (no E-block lower-triangular dependencies)
    for (int perm_i = 0; perm_i < ilu->n_int; perm_i++) {
        double sum = 0.0;
        for (int k = ilu->lu_rowptr[perm_i]; k < ilu->lu_rowptr[perm_i + 1]; k++) {
            int col = ilu->lu_col[k];
            if (col >= perm_i) break;
            sum += ilu->lu_val[k] * y_perm[col];
        }
        y_perm[perm_i] = b_perm[perm_i] - sum;
    }

    // Initialize separator y and external receive buffer
    for (int perm_i = ilu->n_int; perm_i < local_n; perm_i++)
        y_perm[perm_i] = 0.0;
    ilu->recv_buf_fwd.assign(num_ext_fwd, 0.0);

    std::vector<int> recv_counts(num_fwd_src);
    std::vector<int> recv_offsets(num_fwd_src + 1, 0);
    int total_recv = 0;
    for (int i = 0; i < num_fwd_src; i++) {
        recv_counts[i] = ilu->fwd_recv_rows[i].size();
        recv_offsets[i] = total_recv;
        total_recv += recv_counts[i];
    }

    std::vector<int> send_counts(num_fwd_dst);
    std::vector<int> send_offsets(num_fwd_dst + 1, 0);
    int total_send = 0;
    for (int i = 0; i < num_fwd_dst; i++) {
        send_counts[i] = ilu->fwd_send_rows[i].size();
        send_offsets[i] = total_send;
        total_send += send_counts[i];
    }

    std::vector<double> flat_recv_buf(total_recv, 0.0);
    std::vector<double> flat_send_buf(total_send, 0.0);

    double tol = 1e-8;
    int max_iter = 50;

    for (int iter = 0; iter < max_iter; iter++) {
        // Pack current separator y values for higher-ranked neighbors
        int send_idx = 0;
        for (int i = 0; i < num_fwd_dst; i++) {
            for (int global_row : ilu->fwd_send_rows[i]) {
                int local_row = global_row - ilu->local_start;
                int perm_i = ilu->inv_perm[local_row];
                flat_send_buf[send_idx++] = y_perm[perm_i];
            }
        }

        std::vector<MPI_Request> recv_reqs;
        recv_reqs.reserve(num_fwd_src);
        for (int i = 0; i < num_fwd_src; i++) {
            if (recv_counts[i] > 0) {
                MPI_Request req;
                MPI_Irecv(&flat_recv_buf[recv_offsets[i]], recv_counts[i], MPI_DOUBLE,
                          ilu->fwd_src_ranks[i], 2, MPI_COMM_WORLD, &req);
                recv_reqs.push_back(req);
            }
        }

        std::vector<MPI_Request> send_reqs;
        send_reqs.reserve(num_fwd_dst);
        for (int i = 0; i < num_fwd_dst; i++) {
            if (send_counts[i] > 0) {
                MPI_Request req;
                MPI_Isend(&flat_send_buf[send_offsets[i]], send_counts[i], MPI_DOUBLE,
                          ilu->fwd_dst_ranks[i], 2, MPI_COMM_WORLD, &req);
                send_reqs.push_back(req);
            }
        }

        if (!recv_reqs.empty()) {
            MPI_Waitall(recv_reqs.size(), recv_reqs.data(), MPI_STATUSES_IGNORE);
        }

        // Scatter received E-block y values into recv_buf_fwd
        int idx = 0;
        for (int i = 0; i < num_fwd_src; i++) {
            for (int global_col : ilu->fwd_recv_rows[i]) {
                int local_ext_idx = ilu->ext_col_to_local_fwd[global_col];
                ilu->recv_buf_fwd[local_ext_idx] = flat_recv_buf[idx++];
            }
        }

        // Update separator rows: Gauss-Seidel for local, Jacobi for E-block
        double delta_sq = 0.0, norm_sq = 0.0;
        for (int perm_i = ilu->n_int; perm_i < local_n; perm_i++) {
            double sum = 0.0;
            for (int k = ilu->lu_rowptr[perm_i]; k < ilu->lu_rowptr[perm_i + 1]; k++) {
                int col = ilu->lu_col[k];
                if (col < perm_i) {
                    sum += ilu->lu_val[k] * y_perm[col];
                } else if (col >= local_n && col < local_n + num_ext_fwd) {
                    int ext_idx = col - local_n;
                    sum += ilu->lu_val[k] * ilu->recv_buf_fwd[ext_idx];
                }
            }
            double new_val = b_perm[perm_i] - sum;
            double delta = new_val - y_perm[perm_i];
            delta_sq += delta * delta;
            norm_sq  += new_val * new_val;
            y_perm[perm_i] = new_val;
        }

        double global_delta_sq = 0.0, global_norm_sq = 0.0;
        MPI_Allreduce(&delta_sq, &global_delta_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&norm_sq,  &global_norm_sq,  1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if (!send_reqs.empty()) {
            MPI_Waitall(send_reqs.size(), send_reqs.data(), MPI_STATUSES_IGNORE);
        }

        if (global_delta_sq <= tol * tol * global_norm_sq) break;
    }
}

void ILU_backward_sweep(ILUFact* ilu, const double* y_perm, double* x_perm) {
    int num_bwd_src = ilu->bwd_src_ranks.size();
    int num_bwd_dst = ilu->bwd_dst_ranks.size();
    int local_n = ilu->local_n;
    int num_ext_fwd = ilu->ext_cols_fwd.size();
    int num_ext_bwd = ilu->ext_cols_bwd.size();

    // Diagonal precomputation
    std::vector<int> diag_ptr(local_n, -1);
    for (int i = 0; i < local_n; i++) {
        for (int k = ilu->lu_rowptr[i]; k < ilu->lu_rowptr[i + 1]; k++) {
            if (ilu->lu_col[k] == i) {
                diag_ptr[i] = k;
                break;
            }
        }
    }

    // Initialize separator x and external receive buffer
    for (int perm_i = ilu->n_int; perm_i < local_n; perm_i++)
        x_perm[perm_i] = 0.0;
    ilu->recv_buf_bwd.assign(num_ext_bwd, 0.0);

    std::vector<int> recv_counts(num_bwd_src);
    std::vector<int> recv_offsets(num_bwd_src + 1, 0);
    int total_recv = 0;
    for (int i = 0; i < num_bwd_src; i++) {
        recv_counts[i] = ilu->bwd_recv_cols[i].size();
        recv_offsets[i] = total_recv;
        total_recv += recv_counts[i];
    }

    std::vector<int> send_counts(num_bwd_dst);
    std::vector<int> send_offsets(num_bwd_dst + 1, 0);
    int total_send = 0;
    for (int i = 0; i < num_bwd_dst; i++) {
        send_counts[i] = ilu->bwd_send_cols[i].size();
        send_offsets[i] = total_send;
        total_send += send_counts[i];
    }

    std::vector<double> flat_recv_buf(total_recv, 0.0);
    std::vector<double> flat_send_buf(total_send, 0.0);

    double tol = 1e-8;
    int max_iter = 50;

    for (int iter = 0; iter < max_iter; iter++) {
        // Pack current separator x values for lower-ranked neighbors
        int send_idx = 0;
        for (int i = 0; i < num_bwd_dst; i++) {
            for (int global_col : ilu->bwd_send_cols[i]) {
                int local_col = global_col - ilu->local_start;
                int perm_i = ilu->inv_perm[local_col];
                flat_send_buf[send_idx++] = x_perm[perm_i];
            }
        }

        std::vector<MPI_Request> recv_reqs;
        recv_reqs.reserve(num_bwd_src);
        for (int i = 0; i < num_bwd_src; i++) {
            if (recv_counts[i] > 0) {
                MPI_Request req;
                MPI_Irecv(&flat_recv_buf[recv_offsets[i]], recv_counts[i], MPI_DOUBLE,
                          ilu->bwd_src_ranks[i], 3, MPI_COMM_WORLD, &req);
                recv_reqs.push_back(req);
            }
        }

        std::vector<MPI_Request> send_reqs;
        send_reqs.reserve(num_bwd_dst);
        for (int i = 0; i < num_bwd_dst; i++) {
            if (send_counts[i] > 0) {
                MPI_Request req;
                MPI_Isend(&flat_send_buf[send_offsets[i]], send_counts[i], MPI_DOUBLE,
                          ilu->bwd_dst_ranks[i], 3, MPI_COMM_WORLD, &req);
                send_reqs.push_back(req);
            }
        }

        if (!recv_reqs.empty()) {
            MPI_Waitall(recv_reqs.size(), recv_reqs.data(), MPI_STATUSES_IGNORE);
        }

        // Scatter received F-block x values into recv_buf_bwd
        int idx = 0;
        for (int i = 0; i < num_bwd_src; i++) {
            for (int global_col : ilu->bwd_recv_cols[i]) {
                int local_ext_idx = ilu->ext_col_to_local_bwd[global_col];
                ilu->recv_buf_bwd[local_ext_idx] = flat_recv_buf[idx++];
            }
        }

        // Update separator rows (backward): Gauss-Seidel for local, Jacobi for F-block
        double delta_sq = 0.0, norm_sq = 0.0;
        for (int perm_i = local_n - 1; perm_i >= ilu->n_int; perm_i--) {
            double sum = 0.0;
            int diag_m = diag_ptr[perm_i];
            double diag_val = ilu->lu_val[diag_m];
            for (int k = diag_m + 1; k < ilu->lu_rowptr[perm_i + 1]; k++) {
                int col = ilu->lu_col[k];
                if (col < local_n) {
                    sum += ilu->lu_val[k] * x_perm[col];
                } else if (col >= local_n + num_ext_fwd) {
                    int ext_idx = col - (local_n + num_ext_fwd);
                    sum += ilu->lu_val[k] * ilu->recv_buf_bwd[ext_idx];
                }
            }
            double new_val = (y_perm[perm_i] - sum) / diag_val;
            double delta = new_val - x_perm[perm_i];
            delta_sq += delta * delta;
            norm_sq  += new_val * new_val;
            x_perm[perm_i] = new_val;
        }

        double global_delta_sq = 0.0, global_norm_sq = 0.0;
        MPI_Allreduce(&delta_sq, &global_delta_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&norm_sq,  &global_norm_sq,  1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if (!send_reqs.empty()) {
            MPI_Waitall(send_reqs.size(), send_reqs.data(), MPI_STATUSES_IGNORE);
        }

        if (global_delta_sq <= tol * tol * global_norm_sq) break;
    }

    // Solve interior rows backward once; interior rows can have F-block U entries
    // (cols >= local_end) pointing to higher-ranked ranks whose x values are now
    // in recv_buf_bwd from the last iteration of the separator convergence loop above.
    for (int perm_i = ilu->n_int - 1; perm_i >= 0; perm_i--) {
        double sum = 0.0;
        int diag_m = diag_ptr[perm_i];
        double diag_val = ilu->lu_val[diag_m];
        for (int k = diag_m + 1; k < ilu->lu_rowptr[perm_i + 1]; k++) {
            int col = ilu->lu_col[k];
            if (col < local_n) {
                sum += ilu->lu_val[k] * x_perm[col];
            } else if (col >= local_n + num_ext_fwd) {
                int ext_idx = col - (local_n + num_ext_fwd);
                sum += ilu->lu_val[k] * ilu->recv_buf_bwd[ext_idx];
            }
        }
        x_perm[perm_i] = (y_perm[perm_i] - sum) / diag_val;
    }
}

void ILU_solve(ILUFact* ilu, double* b, double* x) {
    int local_n = ilu->local_n;

    std::vector<double> b_perm(local_n);
    for (int i = 0; i < local_n; i++)
        b_perm[ilu->inv_perm[i]] = b[i];

    std::vector<double> y_perm(local_n, 0.0);
    ILU_forward_sweep(ilu, b_perm.data(), y_perm.data());

    std::vector<double> x_perm(local_n, 0.0);
    ILU_backward_sweep(ilu, y_perm.data(), x_perm.data());

    for (int i = 0; i < local_n; i++)
        x[i] = x_perm[ilu->inv_perm[i]];
}

void ILU_multiply(ILUFact* ilu, double* b, double* res) {
    int num_bwd_src = ilu->bwd_src_ranks.size();
    int num_bwd_dst = ilu->bwd_dst_ranks.size();
    int num_fwd_src = ilu->fwd_src_ranks.size();
    int num_fwd_dst = ilu->fwd_dst_ranks.size();
    int local_n = ilu->local_n;
    int num_ext_fwd = ilu->ext_cols_fwd.size();
    int num_ext_bwd = ilu->ext_cols_bwd.size();

    // Apply permutation to b
    std::vector<double> b_perm(local_n);
    for (int i = 0; i < local_n; i++)
        b_perm[ilu->inv_perm[i]] = b[i];

    // Diagonal precomputation
    std::vector<int> diag_ptr(local_n, -1);
    for (int i = 0; i < local_n; i++) {
        for (int k = ilu->lu_rowptr[i]; k < ilu->lu_rowptr[i + 1]; k++) {
            if (ilu->lu_col[k] == i) {
                diag_ptr[i] = k;
                break;
            }
        }
    }

    // Calculate z = U * b_perm
    // Receive F-block b values from higher ranks

    std::vector<int> recv_counts_bwd(num_bwd_src);
    std::vector<int> recv_offsets_bwd(num_bwd_src + 1, 0);
    int total_recv_bwd = 0;
    for (int i = 0; i < num_bwd_src; i++) {
        recv_counts_bwd[i] = ilu->bwd_recv_cols[i].size();
        recv_offsets_bwd[i] = total_recv_bwd;
        total_recv_bwd += recv_counts_bwd[i];
    }

    std::vector<double> flat_recv_bwd(total_recv_bwd, 0.0);
    std::vector<MPI_Request> recv_reqs_bwd;
    recv_reqs_bwd.reserve(num_bwd_src);

    for (int i = 0; i < num_bwd_src; i++) {
        if (recv_counts_bwd[i] > 0) {
            MPI_Request req;
            MPI_Irecv(&flat_recv_bwd[recv_offsets_bwd[i]], recv_counts_bwd[i], MPI_DOUBLE,
                      ilu->bwd_src_ranks[i], 4, MPI_COMM_WORLD, &req);
            recv_reqs_bwd.push_back(req);
        }
    }

    // Send our b_perm values to lower ranks that need them as F-block cols
    std::vector<int> send_counts_bwd(num_bwd_dst);
    std::vector<int> send_offsets_bwd(num_bwd_dst + 1, 0);
    int total_send_bwd = 0;
    for (int i = 0; i < num_bwd_dst; i++) {
        send_counts_bwd[i] = ilu->bwd_send_cols[i].size();
        send_offsets_bwd[i] = total_send_bwd;
        total_send_bwd += send_counts_bwd[i];
    }

    std::vector<double> flat_send_bwd(total_send_bwd, 0.0);
    int send_idx = 0;
    for (int i = 0; i < num_bwd_dst; i++) {
        for (int global_col : ilu->bwd_send_cols[i]) {
            int local_col = global_col - ilu->local_start;
            int perm_i = ilu->inv_perm[local_col];
            flat_send_bwd[send_idx++] = b_perm[perm_i];
        }
    }

    std::vector<MPI_Request> send_reqs_bwd;
    send_reqs_bwd.reserve(num_bwd_dst);
    for (int i = 0; i < num_bwd_dst; i++) {
        if (send_counts_bwd[i] > 0) {
            MPI_Request req;
            MPI_Isend(&flat_send_bwd[send_offsets_bwd[i]], send_counts_bwd[i], MPI_DOUBLE,
                      ilu->bwd_dst_ranks[i], 4, MPI_COMM_WORLD, &req);
            send_reqs_bwd.push_back(req);
        }
    }

    if (!recv_reqs_bwd.empty()) {
        MPI_Waitall(recv_reqs_bwd.size(), recv_reqs_bwd.data(), MPI_STATUSES_IGNORE);
    }

    ilu->recv_buf_bwd.resize(num_ext_bwd);
    int idx = 0;
    for (int i = 0; i < num_bwd_src; i++) {
        for (int global_col : ilu->bwd_recv_cols[i]) {
            int local_ext_idx = ilu->ext_col_to_local_bwd[global_col];
            ilu->recv_buf_bwd[local_ext_idx] = flat_recv_bwd[idx++];
        }
    }

    std::vector<double> z(local_n);
    for (int perm_i = 0; perm_i < local_n; perm_i++) {
        double sum = 0.0;
        int diag_m = diag_ptr[perm_i];
        for (int k = diag_m; k < ilu->lu_rowptr[perm_i + 1]; k++) {
            int col = ilu->lu_col[k];
            if (col < local_n) {
                sum += ilu->lu_val[k] * b_perm[col];
            } else if (col >= local_n + num_ext_fwd) {
                int ext_idx = col - (local_n + num_ext_fwd);
                sum += ilu->lu_val[k] * ilu->recv_buf_bwd[ext_idx];
            }
        }
        z[perm_i] = sum;
    }

    if (!send_reqs_bwd.empty()) {
        MPI_Waitall(send_reqs_bwd.size(), send_reqs_bwd.data(), MPI_STATUSES_IGNORE);
    }

    // Calculate res_perm = L * z
    // Receive E-block z values from lower ranks
    std::vector<int> recv_counts_fwd(num_fwd_src);
    std::vector<int> recv_offsets_fwd(num_fwd_src + 1, 0);
    int total_recv_fwd = 0;
    for (int i = 0; i < num_fwd_src; i++) {
        recv_counts_fwd[i] = ilu->fwd_recv_rows[i].size();
        recv_offsets_fwd[i] = total_recv_fwd;
        total_recv_fwd += recv_counts_fwd[i];
    }

    std::vector<double> flat_recv_fwd(total_recv_fwd, 0.0);
    std::vector<MPI_Request> recv_reqs_fwd;
    recv_reqs_fwd.reserve(num_fwd_src);

    for (int i = 0; i < num_fwd_src; i++) {
        if (recv_counts_fwd[i] > 0) {
            MPI_Request req;
            MPI_Irecv(&flat_recv_fwd[recv_offsets_fwd[i]], recv_counts_fwd[i], MPI_DOUBLE,
                      ilu->fwd_src_ranks[i], 5, MPI_COMM_WORLD, &req);
            recv_reqs_fwd.push_back(req);
        }
    }

    // Send our z values to higher ranks that need them as E-block
    std::vector<int> send_counts_fwd(num_fwd_dst);
    std::vector<int> send_offsets_fwd(num_fwd_dst + 1, 0);
    int total_send_fwd = 0;
    for (int i = 0; i < num_fwd_dst; i++) {
        send_counts_fwd[i] = ilu->fwd_send_rows[i].size();
        send_offsets_fwd[i] = total_send_fwd;
        total_send_fwd += send_counts_fwd[i];
    }

    std::vector<double> flat_send_fwd(total_send_fwd, 0.0);
    send_idx = 0;
    for (int i = 0; i < num_fwd_dst; i++) {
        for (int global_row : ilu->fwd_send_rows[i]) {
            int local_row = global_row - ilu->local_start;
            int perm_i = ilu->inv_perm[local_row];
            flat_send_fwd[send_idx++] = z[perm_i];
        }
    }

    std::vector<MPI_Request> send_reqs_fwd;
    send_reqs_fwd.reserve(num_fwd_dst);
    for (int i = 0; i < num_fwd_dst; i++) {
        if (send_counts_fwd[i] > 0) {
            MPI_Request req;
            MPI_Isend(&flat_send_fwd[send_offsets_fwd[i]], send_counts_fwd[i], MPI_DOUBLE,
                      ilu->fwd_dst_ranks[i], 5, MPI_COMM_WORLD, &req);
            send_reqs_fwd.push_back(req);
        }
    }

    if (!recv_reqs_fwd.empty()) {
        MPI_Waitall(recv_reqs_fwd.size(), recv_reqs_fwd.data(), MPI_STATUSES_IGNORE);
    }

    ilu->recv_buf_fwd.resize(num_ext_fwd);
    idx = 0;
    for (int i = 0; i < num_fwd_src; i++) {
        for (int global_col : ilu->fwd_recv_rows[i]) {
            int local_ext_idx = ilu->ext_col_to_local_fwd[global_col];
            ilu->recv_buf_fwd[local_ext_idx] = flat_recv_fwd[idx++];
        }
    }

    std::vector<double> res_perm(local_n);
    for (int perm_i = 0; perm_i < local_n; perm_i++) {
        double sum = z[perm_i]; // unit diagonal of L
        for (int k = ilu->lu_rowptr[perm_i]; k < ilu->lu_rowptr[perm_i + 1]; k++) {
            int col = ilu->lu_col[k];
            if (col < perm_i) {
                sum += ilu->lu_val[k] * z[col];
            } else if (col >= local_n && col < local_n + num_ext_fwd) {
                int ext_idx = col - local_n;
                sum += ilu->lu_val[k] * ilu->recv_buf_fwd[ext_idx];
            }
        }
        res_perm[perm_i] = sum;
    }

    for (int i = 0; i < local_n; i++)
        res[i] = res_perm[ilu->inv_perm[i]];

    if (!send_reqs_fwd.empty()) {
        MPI_Waitall(send_reqs_fwd.size(), send_reqs_fwd.data(), MPI_STATUSES_IGNORE);
    }
}

void ILU_free(ILUFact* ilu) {
    if (ilu != nullptr) {
        delete ilu;
    }
}