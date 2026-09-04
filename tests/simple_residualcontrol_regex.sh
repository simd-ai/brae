#!/usr/bin/env bash
# residualControl's TURBULENCE criteria are honoured when the case writes them as a REGEX, which is how
# every stock tutorial writes them -- pitzDaily ships `"(k|epsilon|omega|f|v2)" 1e-3;`, T3A ships
# `"(k|omega|gammaInt|ReThetat)" 1e-4;`.
#
# WHY IT EXISTS, and what it is NOT. It was written chasing a suspected defect that turned out not to be
# one: T3A prints `NOTICE [unread] SIMPLE/residualControl/(k|omega|gammaInt|ReThetat)`, and the obvious
# reading -- that the driver looks its turbulence targets up by exact name and misses the regex -- is
# wrong. FoamDict::find is exact-then-regex already (foam_dict.cuh) and marks the matched regex key
# queried, and routing the lookup through residual_control.cuh instead changes nothing: measured, the
# regex form stops at the same iteration either way. T3A's notice is a SYMPTOM of that case diverging
# (queue item 32): the turbulence criteria are only consulted once p and U have passed, and on T3A they
# never do, so the entry is never queried and the audit reports it unread.
#
# What remained was a real behaviour with no gate on it: that a regex-keyed turbulence criterion actually
# binds end to end. This asserts it.
#
#   ARM      the regex form must stop at the SAME iteration as the exact-name form. Not a bound but an
#            equality: the two dictionaries express the identical criterion and can only differ if one of
#            them is being ignored.
#   CONTROL  the same case with the turbulence criterion REMOVED must stop somewhere else, or the arm is
#            comparing two runs that were never gated on it and would pass however the lookup behaved.
#   CONTROL  the regex arm must raise no `unread` notice for the entry -- the audit that started this.
#
# Fail-proof, 2026-09-03: with the turbulence target forced to -1 (the criterion dropped, which is the
# regression this guards against) all three arms stop at 224 instead of 322, both controls fire and the
# gate exits 1. Note what does NOT work as a fail-proof: disabling FoamDict's regex arm globally makes
# brae lose the case's SOLVER settings too and the run goes nan, which fails this gate for a reason that
# has nothing to do with residualControl.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="$ROOT/validation/pitzDaily"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
# Tight enough that the turbulence residuals bind well after p and U, loose enough that they are reached.
TURB="${TURB:-1e-4}"

[ -x "$BRAE" ]     || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: OpenFOAM (blockMesh) not available"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v blockMesh > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-64s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

cp -r "$SRC" "$W/base"; rm -rf "$W"/base/[1-9]* "$W"/base/log.*
( cd "$W/base" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh"; tail -5 "$W/base/log.blockMesh"; exit 1; }
grep -q '"(k|epsilon' "$W/base/system/fvSolution" \
    || { echo "FAIL: the fixture no longer writes its turbulence residualControl as a regex"; exit 1; }

# $1 label, $2 residualControl body -> the iteration it stopped at, in $STOPPED
stopAt() {
    local d="$W/$1"
    rm -rf "$d"; cp -r "$W/base" "$d"; rm -rf "$d"/[1-9]* "$d/run.log"
    RC="$2" python3 - "$d" <<'PY'
import os, re, sys
f = os.path.join(sys.argv[1], 'system/fvSolution')
s = open(f).read()
s, n = re.subn(r'residualControl\s*\{[^{}]*\}', 'residualControl { %s }' % os.environ['RC'], s)
if not n:
    raise SystemExit('the fixture has no residualControl block to rewrite')
open(f, 'w').write(s)
PY
    ( cd "$d" && timeout 900 "$BRAE" . > run.log 2>&1 ) || { echo "FAIL: brae crashed on $1"; tail -5 "$d/run.log"; exit 1; }
    STOPPED=$(grep -oE 'converged in [0-9]+ iterations' "$d/run.log" | tail -1 | grep -oE '[0-9]+')
    [ -n "$STOPPED" ] || STOPPED="endTime"
}

stopAt regex "p 1e-2; U 1e-3; \"(k|epsilon|omega|f|v2)\" $TURB;" ; N_REGEX=$STOPPED
stopAt exact "p 1e-2; U 1e-3; k $TURB; epsilon $TURB;"            ; N_EXACT=$STOPPED
stopAt none  "p 1e-2; U 1e-3;"                                    ; N_NONE=$STOPPED

echo "  stopped at:  regex $N_REGEX   exact $N_EXACT   without the turbulence criterion $N_NONE"
[ "$N_REGEX" = "$N_EXACT" ] \
    && say "a regex-keyed turbulence criterion stops where the exact-name one does" ok \
    || say "a regex-keyed turbulence criterion stops where the exact-name one does (regex $N_REGEX, exact $N_EXACT)" FAIL
# The criterion must BIND, or both arms above converged on p and U and the equality proves nothing.
[ "$N_NONE" != "$N_EXACT" ] \
    && say "control: dropping the turbulence criterion changes where the run stops" ok \
    || say "control: dropping the turbulence criterion changes where the run stops (both $N_NONE)" FAIL
# ...and neither run may report the entry unread: that audit is what found this.
if grep -qiE "NOTICE.*unread.*residualControl" "$W/regex/run.log"; then
    say "control: the regex entry is not reported unread" FAIL
    grep -iE "NOTICE.*unread.*residualControl" "$W/regex/run.log" | head -2
else
    say "control: the regex entry is not reported unread" ok
fi

[ $fail = 0 ] && echo PASS || { echo FAIL; exit 1; }
