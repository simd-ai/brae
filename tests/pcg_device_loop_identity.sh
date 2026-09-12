#!/usr/bin/env bash
# The pressure solve is one graph launch and one host read -- and it is the host loop's answer.
#
# Item 72. The device-resident AMG-PCG already replayed its steady-state iterations from a conditional
# graph, but it read the host THREE times per solve: the initial residual, the end of iteration 0, and
# the report. Each read drains a launch queue the driver runs about a millisecond ahead of the GPU (item
# 65 measured that cost directly). Now the prologue and iteration 0 are inside the graph too:
#   * the initial early-out is the WHILE handle's start value, set on the device by pcgStartCondK from
#     OpenFOAM's own test (minIter > 0 || !converged(initialResidual));
#   * iteration 0 is the body's first execution, differing only in the search direction, which
#     pcgSearchDirK branches on the iteration counter -- p = w at iteration 0, beta*p + w after, with
#     beta computed by scalarDivK's expression and guard;
#   * the report is one read of three device scalars after the launch.
# Same kernels, same operands, same stopping rule, so every solve line and every field must be what the
# host loop produces. BRAE_PCG_DEVICE=0 IS that host loop -- an independently written per-iteration loop,
# not a switch inside this one -- which makes it the strongest available oracle for this change.
#
#   ARM 1   validation/T3A under the V2 driver, 30 iterations (kOmegaSSTLM; p on AMG-PCG at the case's
#           own relTol 0.1, 16-29 cycles per solve): every Time=/Solving-for line identical to the host
#           loop's, every written field an identical file.
#   ARM 2   validation/pitzDailyTurb, 30 iterations (kEpsilon, p at relTol 0.1).
#   ARM 3   validation/rhoBox through the rho mirror, 50 iterations: the compressible caller of the same
#           solver, where p is solved to 1e-10 and the early-out and the cap both bite.
#   CONTROL T3A with p's relTol tightened to 0.01 must CHANGE the solve lines, so the comparison can fail.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
for f in T3A pitzDailyTurb rhoBox; do [ -d "$ROOT/validation/$f" ] || { echo "SKIP: fixture $f missing"; exit 77; }; done
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-78s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
prep() {   # prep <dir> <fixture> <endTime> [pRelTol]
    rm -rf "$1"; mkdir -p "$1"
    cp -r "$ROOT/validation/$2/constant" "$ROOT/validation/$2/system" "$1/"
    if [ -d "$ROOT/validation/$2/0.orig" ]; then cp -r "$ROOT/validation/$2/0.orig" "$1/0"; else cp -r "$ROOT/validation/$2/0" "$1/0"; fi
    python3 - "$1" "$3" "${4:-}" <<'PY'
import re, sys
d, n, rt = sys.argv[1:4]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
s = re.sub(r'\bstartFrom\s+[^;]*;', 'startFrom startTime;', s)
s = re.sub(r'\bstartTime\s+[^;]*;', 'startTime 0;', s)
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime %s;' % n, s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval %s;' % n, s)
s = re.sub(r'\bwritePrecision\s+[^;]*;', 'writePrecision 15;', s)
open(c, 'w').write(s)
if rt:
    f = d + '/system/fvSolution'; s = open(f).read()
    s2 = re.sub(r'(\bp\s*\n?\s*\{[^}]*?relTol\s+)[0-9.eE+-]+;', r'\g<1>%s;' % rt, s, count=1)
    assert s2 != s, 'the p block is not the one this control was written against'
    open(f, 'w').write(s2)
PY
}
lines() { grep -E "^Time = |Solving for|residual" "$1/log"; }
run() { ( cd "$1" && env $2 "$BRAE" $3 "$1" > log 2>&1 ) || { echo "FAIL: brae crashed in $1"; tail -5 "$1/log"; exit 1; }; }
arm() {   # arm <label> <fixture> <endTime> <env> <flag> <minLines> <fields...>
    local label=$1 fx=$2 n=$3 envs=$4 flag=$5 minl=$6; shift 6
    local a="$W/${fx}_g" b="$W/${fx}_h"
    prep "$a" "$fx" "$n"; prep "$b" "$fx" "$n"
    run "$a" "$envs" "$flag"
    run "$b" "BRAE_PCG_DEVICE=0 $envs" "$flag"
    local nl; nl=$(lines "$a" | wc -l); echo "  $label  $fx: $nl lines compared"
    [ "$nl" -ge "$minl" ] || say "$label  the runs produced the expected report lines" FAIL
    diff <(lines "$a") <(lines "$b") > /dev/null \
        && say "$label  $fx: the graph solve and the host loop, every line identical" ok \
        || { say "$label  $fx: the graph solve and the host loop, every line identical" FAIL; diff <(lines "$a") <(lines "$b") | head -4; }
    local same=1 f t
    t=$(ls -d "$a"/[1-9]* 2>/dev/null | xargs -n1 basename | sort -n | tail -1)
    for f in "$@"; do [ -f "$a/$t/$f" ] || { same=0; echo "  missing: $t/$f"; continue; }; cmp -s "$a/$t/$f" "$b/$t/$f" || { same=0; echo "  differs: $t/$f"; }; done
    [ $same -eq 1 ] && say "$label  ...and every written field at $t is an identical file" ok \
                    || say "$label  ...and every written field at $t is an identical file" FAIL
}
arm "ARM 1" T3A           30 "BRAE_SIMPLEFOAM_V2=1"           ""      120 U p k omega nut ReThetat gammaInt phi
arm "ARM 2" pitzDailyTurb 30 "BRAE_SIMPLEFOAM_V2=1"           ""      120 U p k epsilon nut phi
arm "ARM 3" rhoBox        50 "BRAE_RHOSIMPLEFOAM_MIRROR=cuda" "-case" 50  U p T rho phi
prep "$W/ctl" T3A 30 0.01; run "$W/ctl" "BRAE_SIMPLEFOAM_V2=1" ""
diff <(lines "$W/T3A_g") <(lines "$W/ctl") > /dev/null \
    && say "CONTROL  a tightened p relTol changes the solve lines (so the arms can fail)" FAIL \
    || say "CONTROL  a tightened p relTol changes the solve lines (so the arms can fail)" ok
[ $fail -eq 0 ] && echo "PASS: the pressure solve runs entirely on the device and answers what the host loop answers"
exit $fail
