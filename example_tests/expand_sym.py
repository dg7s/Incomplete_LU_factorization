#!/usr/bin/env python3
"""
Convert a MatrixMarket symmetric matrix to general format by mirroring
the off-diagonal entries.  Reads from stdin, writes to stdout.

Usage:
    python3 expand_sym.py < ecology2.mtx > ecology2_gen.mtx
"""
import sys

lines = sys.stdin.read().splitlines()

header_lines = []
size_line_idx = None
entries = []

for i, line in enumerate(lines):
    if line.startswith('%'):
        # rewrite 'symmetric' -> 'general' in the type header
        header_lines.append(line.replace('symmetric', 'general'))
    elif size_line_idx is None:
        size_line_idx = i
        parts = line.split()
        N, M, nnz_stored = int(parts[0]), int(parts[1]), int(parts[2])
    else:
        parts = line.split()
        r, c, v = int(parts[0]), int(parts[1]), float(parts[2])
        entries.append((r, c, v))
        if r != c:
            entries.append((c, r, v))

print('\n'.join(header_lines))
print(f"{N} {M} {len(entries)}")
for r, c, v in entries:
    print(f"{r} {c} {v:.15g}")
