#!/usr/bin/env bash
# The fused Gauss-Seidel walk is the per-component walk, byte for byte.
#
# A vector fvMatrix's components share topology and the upper/lower coefficients; OpenFOAM solves them one
# after another (fvMatrixSolve.C solveSegregated: per component a psi copy, addBoundaryDiag into its own
# diagonal, its own source, the solve, the write-back, the diagonal restored; correctBoundaryConditions only
# after the loop). One component's sweep reads only its own psi, diagonal and source, so item 60a walks the
# dependency levels ONCE per sweep and updates every still-active component at each level, paying the
# per-level dependent-launch latency once instead of once per component. Each component keeps its own
# normFactor, residual, sweep count and stop (gsFusedCondK applies gsSetCondK's test per component), so the
# per-component solves' numbers must come out to the bit -- or it is a different smoother, which this
# project refuses.
#
#   ARM 1   validation/T3A (26,820 cells, 2D: Ux, Uy; symGaussSeidel, relTol 0.1, maxIter 10; the walk is
#           single-block): 50 iterations under the fused walk and under BRAE_GS_FUSED=0, every `Time =` and
#           `Solving for` line (Initial / Final / No Iterations per component) byte-identical, the written
#           fields identical files.
#   ARM 2   validation/cav3d_cf (13,824 cells, 3D: three components; `GaussSeidel`, the ascending-only
#           smoother; Uz starts at residual 0 and must sit out from the first sweep): 30 iterations, the
#           same identity.
#   ARM 3   validation/duct3d_cf (72,000 cells, 3D, GaussSeidel): 20 iterations, the same identity.
#   ARMS    each fused arm announces the fused walk; each BRAE_GS_FUSED=0 arm does not -- so "identical" is
#           not the per-component walk compared with itself.
#   CONTROL T3A at relTol 0.01 on U under BRAE_GS_FUSED=0 must DIFFER from ARM 1's unfused run in its
#           Solving-for lines, so the line comparison has resolution.
# Since item 68 the DEFAULT smoother runs on the host; BRAE_GS_HOST_SMOOTHER=0 pins the device loop this
# gate is about (its announce assertions would fail loudly otherwise -- the arms would compare the host
# smoother with itself).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
for f in T3A cav3d_cf duct3d_cf; do [ -d "$ROOT/validation/$f" ] || { echo "SKIP: fixture $f missing"; exit 77; }; done
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-78s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
prep() {   # prep <dir> <fixture> <endTime> [uRelTol]
    mkdir -p "$1"; cp -r "$ROOT/validation/$2/constant" "$ROOT/validation/$2/system" "$1/"
    if [ -d "$ROOT/validation/$2/0.orig" ]; then cp -r "$ROOT/validation/$2/0.orig" "$1/0"; else cp -r "$ROOT/validation/$2/0" "$1/0"; fi
    python3 - "$1" "$3" "${4:-}" <<'PY'
import re, sys
d, n, rt = sys.argv[1:4]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime %s;' % n, s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval %s;' % n, s)
s = re.sub(r'\bwritePrecision\s+[^;]*;', 'writePrecision 15;', s)
open(c, 'w').write(s)
if rt:
    # T3A groups U with the turbulence scalars in one regex entry; give U its own tighter one.
    f = d + '/system/fvSolution'; s = open(f).read()
    s = s.replace('solvers\n{', 'solvers\n{\n    U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol %s; maxIter 10; }\n' % rt, 1)
    open(f, 'w').write(s)
PY
}
lines() { grep -E "^Time = |Solving for" "$1/log"; }
run() { ( cd "$1" && env $2 "$BRAE" "$1" > log 2>&1 ) || { echo "FAIL: brae crashed in $1"; tail -5 "$1/log"; exit 1; }; }
arm() {   # arm <label> <fixture> <endTime> <minLines> <fields...>
    local label=$1 fx=$2 n=$3 minl=$4; shift 4
    prep "$W/${fx}_f" "$fx" "$n"; prep "$W/${fx}_u" "$fx" "$n"
    run "$W/${fx}_f" "BRAE_GS_HOST_SMOOTHER=0 BRAE_SIMPLEFOAM_V2=1"
    run "$W/${fx}_u" "BRAE_GS_HOST_SMOOTHER=0 BRAE_GS_FUSED=0 BRAE_SIMPLEFOAM_V2=1"
    local ann; ann=$(grep -m1 "symGaussSeidel: fused walk" "$W/${fx}_f/log" | sed 's/^ *//')
    [ -n "$ann" ] && say "$label  fused arm announces: ${ann:0:60}" ok || say "$label  the fused arm announces the fused walk" FAIL
    grep -q "symGaussSeidel: fused walk" "$W/${fx}_u/log" && say "$label  the BRAE_GS_FUSED=0 arm does not announce it" FAIL || say "$label  the BRAE_GS_FUSED=0 arm does not announce it" ok
    local nl; nl=$(lines "$W/${fx}_f" | wc -l); echo "  $label  $fx: $nl log lines compared"
    [ "$nl" -ge "$minl" ] || say "$label  the runs produced the expected number of report lines" FAIL
    if diff <(lines "$W/${fx}_f") <(lines "$W/${fx}_u") > /dev/null; then say "$label  $fx: fused and per-component walks, every Time=/Solving-for line identical" ok
    else say "$label  $fx: fused and per-component walks, every Time=/Solving-for line identical" FAIL; echo "  ($(diff <(lines "$W/${fx}_f") <(lines "$W/${fx}_u") | grep -c '^<') lines differ)"; diff <(lines "$W/${fx}_f") <(lines "$W/${fx}_u") | head -4; fi
    local same=1 f
    for f in "$@"; do [ -f "$W/${fx}_f/$n/$f" ] || { same=0; echo "  missing: $n/$f"; continue; }; cmp -s "$W/${fx}_f/$n/$f" "$W/${fx}_u/$n/$f" || { same=0; echo "  differs: $n/$f"; }; done
    [ $same -eq 1 ] && say "$label  ...and the written fields at $n are identical files" ok || say "$label  ...and the written fields at $n are identical files" FAIL
}
arm "ARM 1" T3A 50 200 U p k omega nut phi     # Time + Ux + Uy + p per iteration
arm "ARM 2" cav3d_cf 30 120 U p phi
arm "ARM 3" duct3d_cf 20 80 U p phi
# CONTROL
prep "$W/T3A_c" T3A 50 0.01; run "$W/T3A_c" "BRAE_GS_HOST_SMOOTHER=0 BRAE_GS_FUSED=0 BRAE_SIMPLEFOAM_V2=1"
if diff <(lines "$W/T3A_u") <(lines "$W/T3A_c") > /dev/null; then say "CONTROL  relTol 0.01 on U changes the Solving-for lines (so ARM 1 can fail)" FAIL
else say "CONTROL  relTol 0.01 on U changes the Solving-for lines (so ARM 1 can fail)" ok; fi
[ $fail -eq 0 ] && echo "PASS: the fused Gauss-Seidel walk is the per-component walk, byte for byte"
exit $fail
