#!/bin/bash
#SBATCH --job-name=dasilu_bench
#SBATCH --account=g105-2692
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --time=01:00:00
#SBATCH --output=bench_%j.out
#SBATCH --error=bench_%j.err

set -e

make

# ── Strong-scaling matrices (fixed size, vary P) ──────────────────────────────
./gen_matrix laplacian2d 316  > ss_lap2d_100k.mtx   # N=99856
./gen_matrix laplacian2d 632  > ss_lap2d_400k.mtx   # N=399424
./gen_matrix laplacian3d  22  > ss_lap3d_10k.mtx    # N=10648
./gen_matrix laplacian3d  46  > ss_lap3d_100k.mtx   # N=97336
./gen_matrix band        5000 10 > ss_band_5k.mtx   # N=5000, bw=10
./gen_matrix band       50000 50 > ss_band_50k.mtx  # N=50000, bw=50

echo ""
echo "################################################################"
echo "##                   STRONG SCALING                          ##"
echo "################################################################"
for MAT in ss_lap2d_100k.mtx ss_lap2d_400k.mtx ss_lap3d_10k.mtx ss_lap3d_100k.mtx ss_band_5k.mtx ss_band_50k.mtx; do
    echo ""
    echo "=== ${MAT} ==="
    for NP in 1 2 4 8 16 32 64; do
        srun -n ${NP} ./bigger_test ${MAT}
    done
done

# ── Weak-scaling (lap2d, target ~6250 rows/rank) ──────────────────────────────
# n chosen so N = n² ≈ 6250 * P
declare -A WS_N=( [1]=79 [2]=111 [4]=158 [8]=223 [16]=316 [32]=447 [64]=632 )

echo ""
echo "################################################################"
echo "##                    WEAK SCALING                           ##"
echo "##          lap2d, ~6250 rows per rank                       ##"
echo "################################################################"
for NP in 1 2 4 8 16 32 64; do
    N=${WS_N[$NP]}
    ./gen_matrix laplacian2d ${N} > ws_lap2d_${NP}.mtx
    echo ""
    echo "=== P=${NP}  n=${N}  N≈$((N*N)) ==="
    srun -n ${NP} ./bigger_test ws_lap2d_${NP}.mtx
done
