#!/bin/bash
#SBATCH --job-name=dasilu
#SBATCH --account=g105-2692
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=00:30:00
#SBATCH --output=dasilu_%j.out
#SBATCH --error=dasilu_%j.err

set -e

# Build
make

# Generate matrices
./gen_matrix laplacian2d  32  > lap2d_1024.mtx     # N=1024
./gen_matrix laplacian2d 100  > lap2d_10k.mtx      # N=10000
./gen_matrix laplacian2d 316  > lap2d_100k.mtx     # N≈100000
./gen_matrix laplacian3d  10  > lap3d_1k.mtx       # N=1000
./gen_matrix laplacian3d  22  > lap3d_10k.mtx      # N=10648
./gen_matrix band        5000 10 > band_5k.mtx

MATRICES="lap2d_1024.mtx lap2d_10k.mtx lap2d_100k.mtx lap3d_1k.mtx lap3d_10k.mtx band_5k.mtx"

# Strong scaling: fix matrix, vary process count
for NP in 1 2 4 8 16; do
    echo ""
    echo "########## P = ${NP} ##########"
    for MAT in $MATRICES; do
        srun -n ${NP} ./bigger_test ${MAT}
    done
done
