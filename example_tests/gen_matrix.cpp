// gen_matrix.cpp — Generate sparse test matrices in MatrixMarket format.
//
//   ./gen_matrix tridiag  <N>          → tridiagonal N×N
//   ./gen_matrix laplacian2d <n>       → 2D 5-point Laplacian (n*n × n*n)
//   ./gen_matrix laplacian3d <n>       → 3D 7-point Laplacian (n³ × n³)
//   ./gen_matrix band <N> <bw>         → banded N×N with half-bandwidth bw
//
// All matrices have positive diagonal so LU decomposition exists without
// column pivoting.  Output goes to stdout in MatrixMarket coordinate format.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <utility>
#include <algorithm>

using namespace std;
using Entry = pair<pair<int,int>, double>;

static void emit_header(int N, int nnz) {
    printf("%% MatrixMarket matrix coordinate real general\n");
    printf("%d %d %d\n", N, N, nnz);
}

static void emit(vector<Entry>& es) {
    sort(es.begin(), es.end());
    for (auto& e : es)
        printf("%d %d %.15g\n", e.first.first + 1, e.first.second + 1, e.second);
}

// ---------------------------------------------------------------------------
// 1D tridiagonal: diagonal=2, off-diagonal=-1
// ---------------------------------------------------------------------------
static void gen_tridiag(int N) {
    vector<Entry> es;
    es.reserve(3 * N - 2);
    for (int i = 0; i < N; i++) {
        es.push_back({{i, i}, 2.0});
        if (i > 0)     es.push_back({{i, i-1}, -1.0});
        if (i < N - 1) es.push_back({{i, i+1}, -1.0});
    }
    emit_header(N, (int)es.size());
    emit(es);
}

// ---------------------------------------------------------------------------
// 2D 5-point Laplacian on n×n grid (row-major ordering)
//   diagonal=4, horizontal neighbors=-1, vertical neighbors=-1
// ---------------------------------------------------------------------------
static void gen_laplacian2d(int n) {
    int N = n * n;
    vector<Entry> es;
    es.reserve(5 * N);
    auto idx = [&](int r, int c) { return r * n + c; };
    for (int r = 0; r < n; r++) {
        for (int c = 0; c < n; c++) {
            int i = idx(r, c);
            es.push_back({{i, i}, 4.0});
            if (r > 0)     es.push_back({{i, idx(r-1,c)}, -1.0});
            if (r < n - 1) es.push_back({{i, idx(r+1,c)}, -1.0});
            if (c > 0)     es.push_back({{i, idx(r,c-1)}, -1.0});
            if (c < n - 1) es.push_back({{i, idx(r,c+1)}, -1.0});
        }
    }
    emit_header(N, (int)es.size());
    emit(es);
}

// ---------------------------------------------------------------------------
// 3D 7-point Laplacian on n×n×n grid (row-major ordering)
//   diagonal=6, six neighbors=-1
// ---------------------------------------------------------------------------
static void gen_laplacian3d(int n) {
    int N = n * n * n;
    vector<Entry> es;
    es.reserve(7 * N);
    auto idx = [&](int x, int y, int z) { return x*n*n + y*n + z; };
    for (int x = 0; x < n; x++)
    for (int y = 0; y < n; y++)
    for (int z = 0; z < n; z++) {
        int i = idx(x, y, z);
        es.push_back({{i, i}, 6.0});
        if (x > 0)     es.push_back({{i, idx(x-1,y,z)}, -1.0});
        if (x < n - 1) es.push_back({{i, idx(x+1,y,z)}, -1.0});
        if (y > 0)     es.push_back({{i, idx(x,y-1,z)}, -1.0});
        if (y < n - 1) es.push_back({{i, idx(x,y+1,z)}, -1.0});
        if (z > 0)     es.push_back({{i, idx(x,y,z-1)}, -1.0});
        if (z < n - 1) es.push_back({{i, idx(x,y,z+1)}, -1.0});
    }
    emit_header(N, (int)es.size());
    emit(es);
}

// ---------------------------------------------------------------------------
// Banded matrix: diagonal + bw upper/lower diagonals, positive definite
// ---------------------------------------------------------------------------
static void gen_band(int N, int bw) {
    vector<Entry> es;
    es.reserve((2*bw + 1) * N);
    for (int i = 0; i < N; i++) {
        double diag = 2.0 * bw + 1.0;  // ensure diagonal dominance
        es.push_back({{i, i}, diag});
        for (int d = 1; d <= bw; d++) {
            if (i + d < N) { es.push_back({{i, i+d}, -1.0}); }
            if (i - d >= 0){ es.push_back({{i, i-d}, -1.0}); }
        }
    }
    emit_header(N, (int)es.size());
    emit(es);
}

// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage:\n"
            "  %s tridiag    <N>\n"
            "  %s laplacian2d <n>      (N = n*n)\n"
            "  %s laplacian3d <n>      (N = n*n*n)\n"
            "  %s band        <N> <bw>\n",
            argv[0], argv[0], argv[0], argv[0]);
        return 1;
    }

    const char* type = argv[1];

    if (!__builtin_strcmp(type, "tridiag")) {
        gen_tridiag(atoi(argv[2]));
    } else if (!__builtin_strcmp(type, "laplacian2d")) {
        gen_laplacian2d(atoi(argv[2]));
    } else if (!__builtin_strcmp(type, "laplacian3d")) {
        gen_laplacian3d(atoi(argv[2]));
    } else if (!__builtin_strcmp(type, "band")) {
        if (argc < 4) { fprintf(stderr, "band needs <N> <bw>\n"); return 1; }
        gen_band(atoi(argv[2]), atoi(argv[3]));
    } else {
        fprintf(stderr, "Unknown type: %s\n", type);
        return 1;
    }
    return 0;
}
