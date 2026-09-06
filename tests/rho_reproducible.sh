#!/usr/bin/env bash
# The compressible mirror gives the same answer twice.
#
# Items 59 and 69. Two identical runs of validation/sbMatched used to differ: 1e-12 within five
# iterations, and on squareBend 188 iterations against 191 with U 1.8e-04 apart -- a fifth of that case's
# whole disagreement with OpenFOAM, and enough to make any tight gate on this arm unwritable (the
# BiCGStab identity gate's compressible arm is floor-relative for exactly this reason).
#
# THE CAUSE, localised with the stage dumps rather than guessed. The queue blamed the reductions; they
# were already deterministic (reductions.cu is two-stage with no atomicAdd, and says so). Dumping every
# stage of two runs showed iteration 1 identical in all 54 stages and iteration 2 differing FIRST in the
# epsilon SOURCE, with its inputs -- gradU, gByNu, divU, divPhi, G, eps0, G0, k, epsilon, nut -- and the
# assembled matrix all identical. That is fvMatrix::setValues (device_fvoptions.cu): it ran one thread
# per PINNED cell and atomicAdd'ed the transferred coefficient into each neighbour's source, so a cell
# beside two or more pinned cells summed in completion order. Under the epsilon wall constraint every
# wall-adjacent cell is pinned, so such cells are everywhere. It is now a per-cell gather in face order.
#
#   ARM 1   sbMatched (112,000 cells, kEpsilon), 5 iterations twice: every residual line identical and
#           every written field an identical file.
#   ARM 2   the same at 20 iterations -- long enough for a last-bit difference to have grown visible
#           (the old code differed at 1e-12 by iteration 5 and 1.8e-04 by convergence).
#   ARM 3   rhoBox, 50 iterations twice: the small case, which was already reproducible, still is.
#   CONTROL a THIRD run of ARM 1 with one input perturbed (the momentum relaxation 0.9 -> 0.7) must
#           differ -- so "identical" is not this gate comparing a run with itself. Note what does NOT
#           serve: loosening p's tolerance from 1e-12 to 1e-8 leaves the printed 5-digit residuals
#           unchanged, because it moves the answer by about 1e-8.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
for f in sbMatched rhoBox; do [ -d "$ROOT/validation/$f" ] || { echo "SKIP: fixture $f missing"; exit 77; }; done
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-74s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
prep() {   # prep <dir> <fixture> <endTime> [tightenP]
    rm -rf "$1"; mkdir -p "$1"
    cp -r "$ROOT/validation/$2/constant" "$ROOT/validation/$2/system" "$1/"
    if [ -d "$ROOT/validation/$2/0.orig" ]; then cp -r "$ROOT/validation/$2/0.orig" "$1/0"; else cp -r "$ROOT/validation/$2/0" "$1/0"; fi
    python3 - "$1" "$3" "${4:-}" <<'PY'
import re, sys
d, n, tight = sys.argv[1:4]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
s = re.sub(r'\bstartFrom\s+[^;]*;', 'startFrom startTime;', s)
s = re.sub(r'\bstartTime\s+[^;]*;', 'startTime 0;', s)
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime %s;' % n, s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval %s;' % n, s)
s = re.sub(r'\bwritePrecision\s+[^;]*;', 'writePrecision 15;', s)
open(c, 'w').write(s)
if tight:
    # The perturbation has to be one the printed residuals can SEE. sbMatched solves every field to
    # 1e-12 with relTol 0, so moving p's tolerance to 1e-8 changes the answer by about 1e-8 and the
    # 5-digit `Time =` line does not move at all (measured). The momentum relaxation does: 0.9 -> 0.7.
    f = d + '/system/fvSolution'; s = open(f).read()
    s2 = re.sub(r'(\n\s*U\s+)0\.9\s*;', r'\g<1>0.7;', s, count=1)
    assert s2 != s, 'the U relaxation factor is not the one this control was written against'
    open(f, 'w').write(s2)
PY
}
run() { ( cd "$1" && BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BRAE" -case "$1" > log 2>&1 ) || { echo "FAIL: brae crashed in $1"; tail -5 "$1/log"; exit 1; }; }
lines() { grep -E "^Time = " "$1/log"; }
pair() {   # pair <label> <fixture> <endTime> <fields...>
    local label=$1 fx=$2 n=$3; shift 3
    local a="$W/${label// /_}_a" b="$W/${label// /_}_b"
    prep "$a" "$fx" "$n"; prep "$b" "$fx" "$n"
    run "$a"; run "$b"
    local nl; nl=$(lines "$a" | wc -l); echo "  $label  $fx: $nl residual lines over $n iterations"
    [ "$nl" -ge "$n" ] || say "$label  the runs reached $n iterations" FAIL
    diff <(lines "$a") <(lines "$b") > /dev/null \
        && say "$label  $fx x2: every residual line identical" ok \
        || { say "$label  $fx x2: every residual line identical" FAIL; diff <(lines "$a") <(lines "$b") | head -4; }
    local same=1 f
    for f in "$@"; do [ -f "$a/$n/$f" ] || { same=0; echo "  missing: $n/$f"; continue; }; cmp -s "$a/$n/$f" "$b/$n/$f" || { same=0; echo "  differs: $n/$f"; }; done
    [ $same -eq 1 ] && say "$label  ...and every written field at $n is an identical file" ok \
                    || say "$label  ...and every written field at $n is an identical file" FAIL
}
pair "ARM 1" sbMatched 5  U p T rho phi k epsilon
pair "ARM 2" sbMatched 20 U p T rho phi k epsilon
pair "ARM 3" rhoBox    50 U p T rho phi
# CONTROL -- a different input must give a different run
prep "$W/ctl" sbMatched 5 tight; run "$W/ctl"
diff <(lines "$W/ARM_1_a") <(lines "$W/ctl") > /dev/null \
    && say "CONTROL  a changed momentum relaxation changes the run (so the arms can fail)" FAIL \
    || say "CONTROL  a changed momentum relaxation changes the run (so the arms can fail)" ok
[ $fail -eq 0 ] && echo "PASS: the compressible mirror is reproducible run to run"
exit $fail
