#!/bin/bash
#SBATCH --job-name=dasilu_real
#SBATCH --account=g105-2692
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=16
#SBATCH --time=01:00:00
#SBATCH --output=bench_real_%j.out
#SBATCH --error=bench_real_%j.err

# Strong-scaling benchmark of one real-world matrix from the SuiteSparse
# Matrix Collection (ecology2, McRae group).
#
# ecology2: N=999999, symmetric SPD, 2D ecology/diffusion discretisation
# (5-point stencil, same structure as lap2d_1m but from a real application).
#
# SuiteSparse page: https://sparse.tamu.edu/McRae/ecology2
# The MM file stores only the lower triangle (format=symmetric);
# expand_sym.py mirrors off-diagonal entries to produce a general MM file.

set -e

# ── Build ─────────────────────────────────────────────────────────────────────
make

# ── Download and prepare ecology2 ────────────────────────────────────────────
if [ ! -f ecology2_gen.mtx ]; then
    echo "=== Downloading ecology2 from SuiteSparse ==="
    wget -q "https://sparse.tamu.edu/MM/McRae/ecology2.tar.gz" -O ecology2.tar.gz
    tar -xzf ecology2.tar.gz ecology2/ecology2.mtx
    mv ecology2/ecology2.mtx ecology2.mtx
    rm -rf ecology2 ecology2.tar.gz

    echo "=== Expanding symmetric storage to general format ==="
    python3 expand_sym.py < ecology2.mtx > ecology2_gen.mtx
    echo "    Done. $(wc -l < ecology2_gen.mtx) lines."
fi

echo ""
echo "################################################################"
echo "##         STRONG SCALING  —  ecology2  (N=999999)           ##"
echo "################################################################"

for NP in 1 2 4 8 16 32 64 128 256; do
    echo ""
    srun -n ${NP} ./bigger_test ecology2_gen.mtx
done
