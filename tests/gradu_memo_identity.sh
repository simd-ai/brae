#!/usr/bin/env bash
# grad(U) computed once per U state and shared across the sites that see it -- byte for byte.
#
# Item 65. OpenFOAM computes fvc::grad(U) once per U state when the case caches it (gradScheme.C: the
# cached field is reused while U's event number is unchanged); brae's V2 iteration computed it at four
# sites, twice on the predictor's U and twice on the corrected U. Every site evaluates U's boundary values
# through the same deviceBCValue on the same dbU, so at one U state the four are the same bits; the memo
# (deviceGradUShared) shares them, keyed on a fingerprint of the three internal fields and the boundary
# arrays the value kernel reads. Same kernels, same inputs -- so every solve line and every field must be
# identical to recomputing at every site, or the memo returned a gradient of some other U.
#
#   ARM 1   validation/T3A (kOmegaSSTLM: the momentum sites, the SST production/strain site and the LM
#           site), V2, 50 iterations: memo against BRAE_GRADU_MEMO=0, lines and fields identical.
#   ARM 2   validation/duct3d_cf (kEpsilon, 3D, 72,000 cells), 20 iterations.
#   ARM 3   validation/cav3d_cf (laminar, 3D: the momentum sites only), 30 iterations.
#   ARMS    the memo arm announces the memo; the =0 arm does not.
#   CONTROL BRAE_GRADU_MEMO=stale (never recompute after the first gradient) must CHANGE T3A's solve lines
#           against the =0 arm: the fail-proof that a memo returning a stale gradient is caught here.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
for f in T3A duct3d_cf cav3d_cf; do [ -d "$ROOT/validation/$f" ] || { echo "SKIP: fixture $f missing"; exit 77; }; done
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-80s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
prep() {   # prep <dir> <fixture> <endTime>
    mkdir -p "$1"; cp -r "$ROOT/validation/$2/constant" "$ROOT/validation/$2/system" "$1/"
    if [ -d "$ROOT/validation/$2/0.orig" ]; then cp -r "$ROOT/validation/$2/0.orig" "$1/0"; else cp -r "$ROOT/validation/$2/0" "$1/0"; fi
    python3 - "$1" "$3" <<'PY'
import re, sys
d, n = sys.argv[1:3]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime %s;' % n, s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval %s;' % n, s)
s = re.sub(r'\bwritePrecision\s+[^;]*;', 'writePrecision 15;', s)
open(c, 'w').write(s)
PY
}
lines() { grep -E "^Time = |Solving for" "$1/log"; }
run() { ( cd "$1" && env $2 "$BRAE" "$1" > log 2>&1 ) || { echo "FAIL: brae crashed in $1"; tail -5 "$1/log"; exit 1; }; }
arm() {   # arm <label> <fixture> <endTime> <minLines> <fields...>
    local label=$1 fx=$2 n=$3 minl=$4; shift 4
    local dm="$W/${fx}_m" dr="$W/${fx}_r"
    prep "$dm" "$fx" "$n"; prep "$dr" "$fx" "$n"
    run "$dm" "BRAE_SIMPLEFOAM_V2=1"
    run "$dr" "BRAE_GRADU_MEMO=0 BRAE_SIMPLEFOAM_V2=1"
    grep -q "grad(U): shared across" "$dm/log" && say "$label  the memo arm announces the memo" ok || say "$label  the memo arm announces the memo" FAIL
    grep -q "grad(U): shared across" "$dr/log" && say "$label  the BRAE_GRADU_MEMO=0 arm does not" FAIL || say "$label  the BRAE_GRADU_MEMO=0 arm does not" ok
    local nl; nl=$(lines "$dm" | wc -l); echo "  $label  $fx: $nl log lines compared"
    [ "$nl" -ge "$minl" ] || say "$label  the runs produced the expected number of report lines" FAIL
    if diff <(lines "$dm") <(lines "$dr") > /dev/null; then say "$label  $fx: memo and recompute-everywhere, every Time=/Solving-for line identical" ok
    else say "$label  $fx: memo and recompute-everywhere, every Time=/Solving-for line identical" FAIL; echo "  ($(diff <(lines "$dm") <(lines "$dr") | grep -c '^<') lines differ)"; diff <(lines "$dm") <(lines "$dr") | head -4; fi
    local same=1 f
    for f in "$@"; do [ -f "$dm/$n/$f" ] || { same=0; echo "  missing: $n/$f"; continue; }; cmp -s "$dm/$n/$f" "$dr/$n/$f" || { same=0; echo "  differs: $n/$f"; }; done
    [ $same -eq 1 ] && say "$label  ...and the written fields at $n are identical files" ok || say "$label  ...and the written fields at $n are identical files" FAIL
}
arm "ARM 1" T3A 50 200 U p k omega nut ReThetat gammaInt phi
arm "ARM 2" duct3d_cf 20 100 U p k epsilon nut phi
arm "ARM 3" cav3d_cf 30 120 U p phi              # laminar: no turbulence fields
prep "$W/T3A_s" T3A 50; run "$W/T3A_s" "BRAE_GRADU_MEMO=stale BRAE_SIMPLEFOAM_V2=1"
if diff <(lines "$W/T3A_r") <(lines "$W/T3A_s") > /dev/null; then say "CONTROL  a memo that never recomputes changes the run (so ARM 1 can fail)" FAIL
else say "CONTROL  a memo that never recomputes changes the run (so ARM 1 can fail)" ok; fi
[ $fail -eq 0 ] && echo "PASS: grad(U) is computed once per U state and shared, byte for byte"
exit $fail
