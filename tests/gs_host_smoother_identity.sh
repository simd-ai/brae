#!/usr/bin/env bash
# The host smoother is the device smoothSolver loop, byte for byte.
#
# Item 68: OpenFOAM's Gauss-Seidel sweep runs on the CPU (one thread per component), and everything that
# DECIDES stays on the device -- after each batch of sweeps the field is uploaded and the residual is the
# same amul/copy/axpy/sumMag the device loop runs, the stop test is gsSetCondK's on the same doubles, the
# report the same numbers. tests/gs_ladder shows the host sweep's psi is the device walk's to the bit, so
# the whole solve must be: every solve line identical, every field an identical file.
#
#   ARM 1   validation/T3A under V2, 50 iterations: U (two components, fused on the device arm) and
#           k/omega/ReThetat/gammaInt (scalar solves) under the host smoother (the default) and under
#           BRAE_GS_HOST_SMOOTHER=0, the device loop -- lines and fields identical.
#   ARM 2   validation/cav3d_cf, 30 iterations: three components, the ascending-only GaussSeidel, Uz
#           starting converged (the host path must leave it alone from the first sweep).
#   ARM 3   validation/duct3d_cf (72,000 cells), 20 iterations.
#   ARMS    the host arm announces the host smoother; the device arm does not.
#   CONTROL T3A at relTol 0.01 on U under the device loop differs from ARM 1's device arm.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
for f in T3A cav3d_cf duct3d_cf; do [ -d "$ROOT/validation/$f" ] || { echo "SKIP: fixture $f missing"; exit 77; }; done
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-80s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
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
    f = d + '/system/fvSolution'; s = open(f).read()
    s = s.replace('solvers\n{', 'solvers\n{\n    U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol %s; maxIter 10; }\n' % rt, 1)
    open(f, 'w').write(s)
PY
}
lines() { grep -E "^Time = |Solving for" "$1/log"; }
run() { ( cd "$1" && env $2 "$BRAE" "$1" > log 2>&1 ) || { echo "FAIL: brae crashed in $1"; tail -5 "$1/log"; exit 1; }; }
arm() {   # arm <label> <fixture> <endTime> <minLines> <fields...>
    local label=$1 fx=$2 n=$3 minl=$4; shift 4
    local dh="$W/${fx}_h" dd="$W/${fx}_d"
    prep "$dh" "$fx" "$n"; prep "$dd" "$fx" "$n"
    run "$dh" "BRAE_SIMPLEFOAM_V2=1"                          # the default since item 68
    run "$dd" "BRAE_GS_HOST_SMOOTHER=0 BRAE_SIMPLEFOAM_V2=1"
    grep -q "smoothSolver: host smoother" "$dh/log" && say "$label  the host arm announces the host smoother" ok || say "$label  the host arm announces the host smoother" FAIL
    grep -q "smoothSolver: host smoother" "$dd/log" && say "$label  the device arm does not" FAIL || say "$label  the device arm does not" ok
    local nl; nl=$(lines "$dh" | wc -l); echo "  $label  $fx: $nl log lines compared"
    [ "$nl" -ge "$minl" ] || say "$label  the runs produced the expected number of report lines" FAIL
    if diff <(lines "$dh") <(lines "$dd") > /dev/null; then say "$label  $fx: host smoother and device loop, every Time=/Solving-for line identical" ok
    else say "$label  $fx: host smoother and device loop, every Time=/Solving-for line identical" FAIL; echo "  ($(diff <(lines "$dh") <(lines "$dd") | grep -c '^<') lines differ)"; diff <(lines "$dh") <(lines "$dd") | head -4; fi
    local same=1 f
    for f in "$@"; do [ -f "$dh/$n/$f" ] || { same=0; echo "  missing: $n/$f"; continue; }; cmp -s "$dh/$n/$f" "$dd/$n/$f" || { same=0; echo "  differs: $n/$f"; }; done
    [ $same -eq 1 ] && say "$label  ...and the written fields at $n are identical files" ok || say "$label  ...and the written fields at $n are identical files" FAIL
}
arm "ARM 1" T3A 50 200 U p k omega nut ReThetat gammaInt phi
arm "ARM 2" cav3d_cf 30 120 U p phi
arm "ARM 3" duct3d_cf 20 100 U p phi
prep "$W/T3A_c" T3A 50 0.01; run "$W/T3A_c" "BRAE_GS_HOST_SMOOTHER=0 BRAE_SIMPLEFOAM_V2=1"
if diff <(lines "$W/T3A_d") <(lines "$W/T3A_c") > /dev/null; then say "CONTROL  relTol 0.01 on U changes the Solving-for lines (so ARM 1 can fail)" FAIL
else say "CONTROL  relTol 0.01 on U changes the Solving-for lines (so ARM 1 can fail)" ok; fi
[ $fail -eq 0 ] && echo "PASS: the host smoother is the device smoothSolver loop, byte for byte"
exit $fail
