#!/usr/bin/env bash
# The momentum solver arms on ONE mesh, 20 iterations each, brae CUDA mirror: the U-solve wall per outer
# iteration (BRAE_PHASE_TIME=1), the BiCGStab / sweep counts summed over the solved components (uIters)
# and the residual trajectory at iterations 10 and 20. Every arm solves the SAME systems to the SAME
# tolerance / relTol / maxIter / minIter (the U entry is rewritten only in its solver / preconditioner
# words); what differs is the method and where each stops inside relTol.
#   jacobi    PBiCGStab, diagonal (what brae substituted for the tutorial's GAMG before colourGS)
#   dilu      PBiCGStab, DILU (the case's own preconditioner where it names one)
#   hostGS    smoothSolver GaussSeidel, OpenFOAM's index order on the host (today's default for that entry)
#   devGS     the same, level-scheduled on the device (BRAE_GS_HOST_SMOOTHER=0)
#   colourGS  the same stop rule, COLOUR order, fused components (the DEFAULT since 2026-09-08)
# Usage: momentum_arms.sh <case dir with a mesh> [iterations]
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="$1"; N="${2:-20}"
W="${WORK:-$(mktemp -d)}"; mkdir -p "$W"
printf "%-9s %10s %8s  %s\n" "arm" "UEqn ms/it" "uIters" "U residual at 10 / 20   (announce line)"
for arm in jacobi dilu hostGS devGS colourGS; do
    d="$W/arm_$arm"; rm -rf "$d"; cp -r "$SRC" "$d"; rm -rf "$d"/[1-9]* 2>/dev/null; [ -d "$d/0" ] || cp -r "$d/0.orig" "$d/0"
    python3 - "$d" "$N" "$arm" <<'PY'
import re, sys
d, n, arm = sys.argv[1], sys.argv[2], sys.argv[3]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime %s;' % n, s); s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval %s;' % n, s)
s = re.sub(r'functions\s*\{.*?\n\}', '', s, flags=re.S); open(c, 'w').write(s)
f = d + '/system/fvSolution'; s = open(f).read()
entry = {'jacobi': 'solver PBiCGStab; preconditioner diagonal;', 'dilu': 'solver PBiCGStab; preconditioner DILU;',
         'hostGS': 'solver smoothSolver; smoother GaussSeidel;', 'devGS': 'solver smoothSolver; smoother GaussSeidel;',
         'colourGS': 'solver smoothSolver; smoother GaussSeidel;'}[arm]
# keep the entry's own tolerance / relTol / maxIter / minIter: take them from the block U sits in
m = re.search(r'"?\(?U[|)"][^{]*\{([^{}]*)\}', s) or re.search(r'\bU\s*\{([^{}]*)\}', s)
body = m.group(1) if m else ''
keep = ' '.join(x.strip() + ';' for x in re.findall(r'((?:tolerance|relTol|maxIter|minIter|nSweeps)\s+[^;]+)', body))
s, k = re.subn(r'"\(U\|e\|k\|epsilon\)"', '"(e|k|epsilon)"', s)
s = re.sub(r'(solvers\s*\{)', r'\1\n    U { %s %s }' % (entry, keep), s, count=1)
open(f, 'w').write(s)
PY
    case $arm in devGS) env="BRAE_GS_HOST_SMOOTHER=0 BRAE_U_SOLVER=ofOrder";; colourGS) env="BRAE_U_SOLVER=colourGS";; *) env="BRAE_U_SOLVER=ofOrder";; esac
    ( cd "$d" && env $env BRAE_RHOSIMPLEFOAM_MIRROR=cuda BRAE_PHASE_TIME=1 "$BRAE" -case "$d" > log 2>&1 ) || { printf "%-9s FAILED: %s\n" "$arm" "$(grep -m1 -i "error" "$d/log" | cut -c1-120)"; continue; }
    ueqn=$(grep "\[phase\] over" "$d/log" | grep -oE "UEqn [0-9.]+ s \(([0-9.]+) ms/it\)" | grep -oE "\([0-9.]+" | tr -d '(')
    its=$(grep -oE "uIters [0-9]+" "$d/log" | awk '{s+=$2} END {print s}')
    r10=$(grep -E "^Time = 10 " "$d/log" | awk '{print $5}'); r20=$(grep -E "^Time = 20 " "$d/log" | awk '{print $5}')
    ann=$(grep -E "momentum:|symGaussSeidel:|smoothSolver:|solvers/U solver|solvers/U smoother" "$d/log" | grep -v "NOTICE \[unread\]" | head -1 | cut -c1-90)
    printf "%-9s %10s %8s  %s / %s   %s\n" "$arm" "$ueqn" "$its" "$r10" "$r20" "$ann"
done
echo "(work dirs under $W)"
