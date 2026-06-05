#include <cstdio>
#include <cassert>
#include <cstring>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <mpi.h>
#include "ilu.h"

using namespace std;

#define EPS 1e-4
// Entry-wise A≈LU check is O(N) ILU_multiply calls; skip for very large N
#define FACT_ENTRY_MAX_N 1000
// Random-vector check: done for all sizes (20 vectors)
#define FACT_RAND_VECTORS 20
#define SOLVE_VECTORS     8

// ---- Helpers ----------------------------------------------------------------

static int rank_first_row(int r, int N, int P) { return r * N / P; }

static void read_matrix(const char* path, int* N, int* nnz,
                        int** row, int** col, double** val) {
    FILE* fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", path); MPI_Abort(MPI_COMM_WORLD, 1); }

    char* line = NULL; size_t len = 0; ssize_t rd;
    int l = -1;
    while ((rd = getline(&line, &len, fp)) != -1) {
        if (line[0] == '%') continue;
        if (l == -1) {
            int M;
            sscanf(line, "%d %d %d", N, &M, nnz);
            assert(M == *N);
            *row = (int*)   malloc(*nnz * sizeof(int));
            *col = (int*)   malloc(*nnz * sizeof(int));
            *val = (double*)malloc(*nnz * sizeof(double));
        } else {
            sscanf(line, "%d %d %lf", *row + l, *col + l, *val + l);
            (*row)[l]--; (*col)[l]--;
        }
        l++;
    }
    fclose(fp);
    if (line) free(line);
}

// Gather each rank's local vector (length local_n) to a full N-vector on rank 0.
static void gather_to_root(const double* local, int local_n,
                           double* global, int N, int P) {
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    vector<int> counts(P), displs(P);
    for (int r = 0; r < P; r++) {
        displs[r] = rank_first_row(r, N, P);
        counts[r] = rank_first_row(r + 1, N, P) - displs[r];
    }
    MPI_Gatherv(local, local_n, MPI_DOUBLE,
                global, counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
}

// ---- Factorization quality check -------------------------------------------

// Compute (LU)*e_j for a set of column indices J.
// On rank 0, compare (LU)_{i,j} against A_{i,j} for every nonzero in those columns.
// Returns number of failed entries (broadcast to all ranks).
static int check_factorization_cols(ILUFact* ilu,
                                    int N, int nnz,
                                    const int* a_row, const int* a_col,
                                    const double* a_val,
                                    const vector<int>& cols,
                                    int P) {
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int first = rank_first_row(rank, N, P);
    int local_n = rank_first_row(rank + 1, N, P) - first;

    vector<double> e_local(local_n), col_local(local_n);
    vector<double> col_global(rank == 0 ? N : 0);

    int failed = 0;
    for (int j : cols) {
        for (int i = 0; i < local_n; i++)
            e_local[i] = (first + i == j) ? 1.0 : 0.0;

        ILU_multiply(ilu, e_local.data(), col_local.data());
        gather_to_root(col_local.data(), local_n, col_global.data(), N, P);

        if (rank == 0) {
            for (int k = 0; k < nnz; k++) {
                if (a_col[k] != j) continue;
                double diff = fabs(a_val[k] - col_global[a_row[k]]);
                if (diff > EPS) {
                    if (failed < 5)
                        printf("    FAIL A[%d,%d]: A=%.6g  LU=%.6g  diff=%.2e\n",
                               a_row[k], j, a_val[k], col_global[a_row[k]], diff);
                    failed++;
                }
            }
        }
    }
    MPI_Bcast(&failed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return failed;
}

// Full entry-wise check (all N columns). Feasible for N ≤ FACT_ENTRY_MAX_N.
static int check_factorization_entry(ILUFact* ilu,
                                     int N, int nnz,
                                     const int* a_row, const int* a_col,
                                     const double* a_val,
                                     int P) {
    vector<int> all_cols(N);
    for (int j = 0; j < N; j++) all_cols[j] = j;
    return check_factorization_cols(ilu, N, nnz, a_row, a_col, a_val, all_cols, P);
}

// Sampled check: pick FACT_RAND_VECTORS distinct random columns.
// Note: ILU_multiply computes L·U·v, which differs from A·v by the dropped fill-in
// of ILU(0). Comparing column-by-column at the original nonzero positions is the
// correct test; a global matrix-vector residual would incorrectly flag fill-in as error.
static int check_factorization_sampled(ILUFact* ilu,
                                       int N, int nnz,
                                       const int* a_row, const int* a_col,
                                       const double* a_val,
                                       int P) {
    int n_sample = min(N, FACT_RAND_VECTORS);
    mt19937 rng(1234);
    vector<int> cols(N); iota(cols.begin(), cols.end(), 0);
    shuffle(cols.begin(), cols.end(), rng);
    cols.resize(n_sample);
    return check_factorization_cols(ilu, N, nnz, a_row, a_col, a_val, cols, P);
}

// ---- Solve roundtrip check --------------------------------------------------

static bool check_solve(ILUFact* ilu, int N, const double* v_global, int P) {
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int first = rank_first_row(rank, N, P);
    int local_n = rank_first_row(rank + 1, N, P) - first;

    vector<double> b(local_n), x(local_n), res(local_n);
    memcpy(b.data(), v_global + first, local_n * sizeof(double));

    ILU_solve(ilu, b.data(), x.data());
    ILU_multiply(ilu, x.data(), res.data());

    int ok = 1;
    for (int i = 0; i < local_n; i++) {
        if (fabs(b[i] - res[i]) > EPS) { ok = 0; break; }
    }
    int global_ok;
    MPI_Reduce(&ok, &global_ok, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return global_ok == 1;
}

// ---- Per-matrix driver ------------------------------------------------------

static void test_matrix(const char* filename) {
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    int N = 0, nnz = 0;
    int   *row = NULL, *col = NULL;
    double *val = NULL;

    if (rank == 0) {
        read_matrix(filename, &N, &nnz, &row, &col, &val);
        printf("\n=== %s  (N=%d  nnz=%d  P=%d) ===\n", filename, N, nnz, P);
    }

    // --- Factorize ---
    double t0 = MPI_Wtime();
    ILUFact* ilu = ILU_factorize(N, nnz, row, col, val);
    double t_fact = MPI_Wtime() - t0;

    MPI_Bcast(&N,   1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast A to all ranks (needed for factorization checks below)
    vector<int>    a_row(nnz), a_col(nnz);
    vector<double> a_val(nnz);
    if (rank == 0) {
        memcpy(a_row.data(), row, nnz * sizeof(int));
        memcpy(a_col.data(), col, nnz * sizeof(int));
        memcpy(a_val.data(), val, nnz * sizeof(double));
    }
    MPI_Bcast(a_row.data(), nnz, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(a_col.data(), nnz, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(a_val.data(), nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) printf("  Factorization time: %.4f s\n", t_fact);

    // --- Test 1: Factorization quality (A_ij ≈ (LU)_ij for each nonzero) ---
    // For small N check all columns; for large N sample FACT_RAND_VECTORS columns.
    if (N <= FACT_ENTRY_MAX_N) {
        int fails = check_factorization_entry(ilu, N, nnz,
                                              a_row.data(), a_col.data(), a_val.data(), P);
        if (rank == 0)
            printf("  [1]  Entry-wise A≈LU  (all %d cols): %s\n",
                   N, fails == 0 ? "PASSED" : "FAILED");
    } else {
        int fails = check_factorization_sampled(ilu, N, nnz,
                                                a_row.data(), a_col.data(), a_val.data(), P);
        if (rank == 0)
            printf("  [1]  Sampled A≈LU  (%d of %d cols): %s\n",
                   min(N, FACT_RAND_VECTORS), N, fails == 0 ? "PASSED" : "FAILED");
    }

    // --- Test 2: Solve roundtrip ---
    int sv_pass = 0;
    vector<double> v(N);

    // Structured vectors
    for (int i = 0; i < N; i++) v[i] = 1.0;
    if (check_solve(ilu, N, v.data(), P)) sv_pass++;

    for (int i = 0; i < N; i++) v[i] = (double)i;
    if (check_solve(ilu, N, v.data(), P)) sv_pass++;

    for (int i = 0; i < N; i++) v[i] = (i % 2 == 0) ? 1.0 : -1.0;
    if (check_solve(ilu, N, v.data(), P)) sv_pass++;

    for (int i = 0; i < N; i++) v[i] = sin((double)i);
    if (check_solve(ilu, N, v.data(), P)) sv_pass++;

    // Random vectors
    mt19937 rng(999);
    uniform_real_distribution<double> dist(-10.0, 10.0);
    for (int t = 0; t < SOLVE_VECTORS - 4; t++) {
        if (rank == 0)
            for (int i = 0; i < N; i++) v[i] = dist(rng);
        MPI_Bcast(v.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (check_solve(ilu, N, v.data(), P)) sv_pass++;
    }

    if (rank == 0)
        printf("  [2]  Solve roundtrip  (%d vecs): %d/%d PASSED\n",
               SOLVE_VECTORS, sv_pass, SOLVE_VECTORS);

    // --- Timing: repeated solve ---
    int n_solves = 20;
    vector<double> b_local(rank_first_row(rank + 1, N, P) - rank_first_row(rank, N, P));
    for (int i = 0; i < (int)b_local.size(); i++) b_local[i] = 1.0;
    vector<double> x_local(b_local.size());

    MPI_Barrier(MPI_COMM_WORLD);
    double ts = MPI_Wtime();
    for (int t = 0; t < n_solves; t++)
        ILU_solve(ilu, b_local.data(), x_local.data());
    double t_solve = (MPI_Wtime() - ts) / n_solves;
    if (rank == 0)
        printf("  Avg solve time (%d runs): %.6f s\n", n_solves, t_solve);

    if (rank == 0) { free(row); free(col); free(val); }
    ILU_free(ilu);
}

// ---- main -------------------------------------------------------------------

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 2) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s matrix1.mtx [matrix2.mtx ...]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    for (int i = 1; i < argc; i++)
        test_matrix(argv[i]);

    if (rank == 0) printf("\nDone.\n");
    MPI_Finalize();
    return 0;
}
