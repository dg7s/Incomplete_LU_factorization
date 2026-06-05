#!/bin/bash
# Okeanos job script for DAS-ILU performance evaluation.
# Edit #SBATCH lines to match your allocation and queue.
#
# Strong-scaling sweep: 1, 2, 4, 8, 16, 32 processes for each matrix.
#
# Usage (from example_tests/):
#   sbatch run_okeanos.sh
#
# Results are written to results_<N>proc_<matrix>.txt

#SBATCH --job-name=dasilu
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=2
#SBATCH --ntasks=32
#SBATCH --time=00:30:00
#SBATCH --output=dasilu_%j.out

set -e

# ---------- build (Cray wrapper, no CXX override needed) ----------
make clean
make

# ---------- generate matrices ----------
./gen_matrix tridiag    10000        > tridiag_10k.mtx
./gen_matrix laplacian2d   100       > lap2d_10k.mtx        # 10000 x 10000
./gen_matrix laplacian2d   316       > lap2d_100k.mtx       # ~99856 x 99856
./gen_matrix laplacian3d    22       > lap3d_10k.mtx        # 10648 x 10648
./gen_matrix band         10000   10 > band_10k_bw10.mtx
./gen_matrix band         10000   50 > band_10k_bw50.mtx

MATRICES="tridiag_10k.mtx lap2d_10k.mtx lap2d_100k.mtx lap3d_10k.mtx band_10k_bw10.mtx band_10k_bw50.mtx"

# ---------- strong-scaling sweep ----------
for NP in 1 2 4 8 16 32; do
    echo "===== P = ${NP} ====="
    for MAT in $MATRICES; do
        OUT="results_${NP}proc_${MAT%.mtx}.txt"
        srun -n ${NP} ./bigger_test ${MAT} 2>&1 | tee ${OUT}
    done
done

echo "All done."
