#!/bin/bash
#SBATCH --job-name=dasilu_bench
#SBATCH --account=g105-2692
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=16
#SBATCH --time=01:30:00
#SBATCH --output=bench_%j.out
#SBATCH --error=bench_%j.err

# 16 nodes × 16 tasks = 256 tasks total.
# 2D Laplacians and band matrices scale well to P=256 (bandwidth << rows/rank).
# 3D Laplacians are capped at P=64: their bandwidth grows as n², quickly
# exceeding rows/rank and turning all local rows into separator rows.

set -e
make

# ── Generate matrices ─────────────────────────────────────────────────────────
./gen_matrix laplacian2d  316 > ss_lap2d_100k.mtx    # N=99856
./gen_matrix laplacian2d  632 > ss_lap2d_400k.mtx    # N=399424
./gen_matrix laplacian2d 1000 > ss_lap2d_1m.mtx      # N=1000000

./gen_matrix laplacian3d   22 > ss_lap3d_10k.mtx     # N=10648
./gen_matrix laplacian3d   46 > ss_lap3d_100k.mtx    # N=97336
./gen_matrix laplacian3d   71 > ss_lap3d_360k.mtx    # N=357911

./gen_matrix band   5000  10  > ss_band_5k.mtx       # N=5000,   bw=10
./gen_matrix band  50000  50  > ss_band_50k.mtx      # N=50000,  bw=50
./gen_matrix band 200000  50  > ss_band_200k.mtx     # N=200000, bw=50

echo ""
echo "################################################################"
echo "##                   STRONG SCALING                          ##"
echo "################################################################"

echo ""
echo "=== 2D Laplacians — scales to P=256 ==="
for MAT in ss_lap2d_100k.mtx ss_lap2d_400k.mtx ss_lap2d_1m.mtx; do
    echo ""
    echo "--- ${MAT} ---"
    for NP in 1 2 4 8 16 32 64 128 256; do
        srun -n ${NP} ./bigger_test ${MAT}
    done
done

echo ""
echo "=== 3D Laplacians — capped at P=64 (high bandwidth-to-block ratio) ==="
for MAT in ss_lap3d_10k.mtx ss_lap3d_100k.mtx ss_lap3d_360k.mtx; do
    echo ""
    echo "--- ${MAT} ---"
    for NP in 1 2 4 8 16 32 64; do
        srun -n ${NP} ./bigger_test ${MAT}
    done
done

echo ""
echo "=== Band matrices — scales to P=256 ==="
for MAT in ss_band_5k.mtx ss_band_50k.mtx ss_band_200k.mtx; do
    echo ""
    echo "--- ${MAT} ---"
    for NP in 1 2 4 8 16 32 64 128 256; do
        srun -n ${NP} ./bigger_test ${MAT}
    done
done

# ── Weak scaling: 2D Laplacian, ~10000 rows per rank ─────────────────────────
# n chosen so N = n² ≈ 10000 * P  →  n ≈ 100 * sqrt(P)
declare -A WS_N=([1]=100 [2]=141 [4]=200 [8]=283 [16]=400 [32]=566 [64]=800 [128]=1131 [256]=1600)

echo ""
echo "################################################################"
echo "##                    WEAK SCALING                           ##"
echo "##          lap2d, ~10000 rows per rank                      ##"
echo "################################################################"
for NP in 1 2 4 8 16 32 64 128 256; do
    N=${WS_N[$NP]}
    ./gen_matrix laplacian2d ${N} > ws_lap2d_${NP}.mtx
    echo ""
    echo "=== P=${NP}  n=${N}  N=$((N*N)) ==="
    srun -n ${NP} ./bigger_test ws_lap2d_${NP}.mtx
done
