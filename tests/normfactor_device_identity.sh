#!/usr/bin/env bash
# The normFactor stays on the device -- and every number is the same, byte for byte.
#
# Item 66. OpenFOAM's normFactor scales every residual a solver reports and tests; brae computed it on
# the device and then READ it to the host once per solve, though every division by it already happens on
# the device (the graph loops upload it and scale there). That read was a queue drain per solve for a
# number the host never used: 15 of the ~36 drains per T3A outer iteration, 3 of them inside the scalar
# transport's host-side normFactor arithmetic. Now deviceNormFactorInto leaves it on the device and the
# solvers take a pointer: the same kernels, the same operations (the divide by n, the scale, the two sums,
# (n1 + n2) + 1e-20, and every residual/normFactor quotient, now as a device division), so every solve
# line and every field must be identical to the host-read path, which BRAE_NORMFACTOR_HOST=1 restores.
#
#   ARM 1   validation/T3A, V2, 50 iterations: U (host smoother, fused), p (PCG graph), the four scalar
#           transports -- lines and fields identical.
#   ARM 2   validation/pitzDailyTurb, V2, 30 iterations: U on BiCGStab (the graph's iteration-0 divisions).
#   ARM 3   validation/duct3d_cf, 20 iterations: the kEpsilon wrapper's solves.
#   ARM 4   validation/rhoBox through the rho mirror, 50 iterations: U/h on BiCGStab, p on PCG.
#   ARMS    the device arm announces the device-resident normFactor; the host arm announces the read.
#   CONTROL T3A at relTol 0.01 on U under the host arm differs from ARM 1's host arm.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
for f in T3A pitzDailyTurb duct3d_cf rhoBox; do [ -d "$ROOT/validation/$f" ] || { echo "SKIP: fixture $f missing"; exit 77; }; done
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
lines() { grep -E "^Time = |Solving for|residual" "$1/log" | grep -v "normFactor:"; }
run() { ( cd "$1" && env $2 "$BRAE" "$1" > log 2>&1 ) || { echo "FAIL: brae crashed in $1"; tail -5 "$1/log"; exit 1; }; }
arm() {   # arm <label> <fixture> <endTime> <envBase> <minLines> <fields...>
    local label=$1 fx=$2 n=$3 base=$4 minl=$5; shift 5
    local dd="$W/${fx}_d" dh="$W/${fx}_h"
    prep "$dd" "$fx" "$n"; prep "$dh" "$fx" "$n"
    run "$dd" "$base"
    run "$dh" "BRAE_NORMFACTOR_HOST=1 $base"
    grep -q "normFactor: device-resident" "$dd/log" && say "$label  the device arm announces the device-resident normFactor" ok || say "$label  the device arm announces the device-resident normFactor" FAIL
    grep -q "normFactor: read to the host" "$dh/log" && say "$label  the host arm announces the host read" ok || say "$label  the host arm announces the host read" FAIL
    local nl; nl=$(lines "$dd" | wc -l); echo "  $label  $fx: $nl log lines compared"
    [ "$nl" -ge "$minl" ] || say "$label  the runs produced the expected number of report lines" FAIL
    if diff <(lines "$dd") <(lines "$dh") > /dev/null; then say "$label  $fx: device-resident and host-read normFactor, every line identical" ok
    else say "$label  $fx: device-resident and host-read normFactor, every line identical" FAIL; echo "  ($(diff <(lines "$dd") <(lines "$dh") | grep -c '^<') lines differ)"; diff <(lines "$dd") <(lines "$dh") | head -4; fi
    local same=1 f t
    t=$(ls -d "$dd"/[1-9]* 2>/dev/null | xargs -n1 basename | sort -n | tail -1)
    for f in "$@"; do [ -f "$dd/$t/$f" ] || { same=0; echo "  missing: $t/$f"; continue; }; cmp -s "$dd/$t/$f" "$dh/$t/$f" || { same=0; echo "  differs: $t/$f"; }; done
    [ $same -eq 1 ] && say "$label  ...and the written fields at $t are identical files" ok || say "$label  ...and the written fields at $t are identical files" FAIL
}
arm "ARM 1" T3A 50 "BRAE_SIMPLEFOAM_V2=1" 200 U p k omega nut ReThetat gammaInt phi
arm "ARM 2" pitzDailyTurb 30 "BRAE_SIMPLEFOAM_V2=1" 120 U p k epsilon nut phi
arm "ARM 3" duct3d_cf 20 "BRAE_SIMPLEFOAM_V2=1" 100 U p k epsilon nut phi
arm "ARM 4" rhoBox 50 "BRAE_RHOSIMPLEFOAM_MIRROR=cuda" 50 U p T rho phi
prep "$W/T3A_c" T3A 50 0.01; run "$W/T3A_c" "BRAE_NORMFACTOR_HOST=1 BRAE_SIMPLEFOAM_V2=1"
if diff <(lines "$W/T3A_h") <(lines "$W/T3A_c") > /dev/null; then say "CONTROL  relTol 0.01 on U changes the Solving-for lines (so ARM 1 can fail)" FAIL
else say "CONTROL  relTol 0.01 on U changes the Solving-for lines (so ARM 1 can fail)" ok; fi
[ $fail -eq 0 ] && echo "PASS: the normFactor stays on the device and every number is the host-read path's, byte for byte"
exit $fail
