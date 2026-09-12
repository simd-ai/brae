#!/usr/bin/env bash
# Foam::bound from the MODEL CONSTRUCTOR, which brae ran only inside correct().
#
# OpenFOAM bounds the two transported scalars the instant the model has read them --
# kEpsilon.C:182-183, kOmegaSSTBase.C:438-439, realizableKE.C:211-212, kOmega.C:151-152, every one of
# them `bound(k_, kMin_)` then `bound(<second>_, <second>Min_)` in the constructor body. The solver then
# calls turbulence->validate() (simpleFoam.C:92, rhoSimpleFoam.C:64), so correctNut builds the FIRST
# momentum equation's nut from the BOUNDED fields. brae's first bound() ran after the first turbulence
# solve, so a case whose initial k or epsilon sits under the floor spent iteration 1 on the file's value.
#
# THE ORACLE IS REAL OpenFOAM, and the discriminator is ORDER, not presence: a brae that bounds only in
# correct() still prints `bounding k` -- one iteration late -- so a test that merely greps the log passes
# while the defect stands. The constructor bounds k FIRST and the second scalar after it (kEpsilon.C:182,
# then :183). correct() runs the other way round, because it solves and bounds epsilon before it solves
# and bounds k (kEpsilon.C:301 then :331). So the FIRST TWO bounding lines of a floored run are
# `k` then `epsilon` when the constructor bound is there, and `epsilon` then `k` when it is not.
#
# Splitting the log at "Time =" would NOT work and was the first version of this test: simpleFoamV2
# prints its "Time = N" line AFTER that iteration's turbulence solve, so iteration 1's own bounding
# lines land above it and the V2 arm passed with nothing under test. Order is a property of the lines
# themselves and is the same discriminator on every driver.
#
# The floors are what make the guard fire (min < lowerBound, bound.C:40, a strict less-than). They are
# set ABOVE the case's own INITIAL field values, and ARM 0 proves that with OpenFOAM itself: unfloored,
# OpenFOAM prints nothing at all, so any line the floored arms print is caused by the keys.
#
# min and max are compared to OpenFOAM's BYTE FOR BYTE; the average is not, and bound_report.cuh has
# why (OpenFOAM's gAverage is a sequential sum, brae's a tree reduction -- 1e-13 relative on a uniform
# field, which at this fixture's `writePrecision 16` is visible).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="$ROOT/validation/pitzDailyTurb"
OFBASH=/usr/lib/openfoam/openfoam2412/etc/bashrc
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$SRC" ]  || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASH" ] || { echo "SKIP: no OpenFOAM v2412 at $OFBASH"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
IT=2
KMIN=5
EMIN=5000

fail=0
say() { printf '  %-72s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

stage() {   # stage <dir> <yes|no: write the floors>
    rm -rf "$1"; mkdir -p "$1"
    cp -r "$SRC/constant" "$SRC/system" "$1/"
    if [ -d "$SRC/0.orig" ]; then cp -r "$SRC/0.orig" "$1/0"; else cp -r "$SRC/0" "$1/0"; fi
    python3 - "$1" "$IT" "$2" "$KMIN" "$EMIN" <<'PYEOF'
import re, sys
d, it, setkeys, kmin, emin = sys.argv[1:6]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*', '', s, flags=re.S)
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime %s;' % it, s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval %s;' % it, s)
open(c, 'w').write(s + '\n')
f = d + '/system/fvSolution'; s = open(f).read()
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { }', s)
open(f, 'w').write(s)
if setkeys == 'yes':
    t = d + '/constant/turbulenceProperties'; s = open(t).read()
    s2 = re.sub(r'(RAS\s*\n?\s*\{)', r'\1\n    kMin            %s;\n    epsilonMin      %s;' % (kmin, emin), s, count=1)
    assert s2 != s, 'no RAS sub-dict to add the keys to'
    open(t, 'w').write(s2)
PYEOF
}

bounds() { grep '^bounding ' "$1" 2> /dev/null || true; }            # every bounding line, in order
first2() { bounds "$1" | head -2; }                                   # the constructor's pair
order()  { first2 "$1" | sed -E 's/^bounding ([A-Za-z]+),.*/\1/' | tr '\n' ' '; }
# min and max only -- see the header on the average.
minmax() { first2 "$1" | sed -E 's/ average:.*//'; }
# The pair AFTER the constructor's: correct()'s, which must run the other way round. This is what makes
# the order test above a real discriminator rather than a coincidence of this case.
next2()  { bounds "$1" | sed -n '3,4p' | sed -E 's/^bounding ([A-Za-z]+),.*/\1/' | tr '\n' ' '; }

# ---- ARM 0: real OpenFOAM, UNFLOORED -- the control that makes the floors the cause ----------------
stage "$W/of0" no
( set +u; . "$OFBASH" > /dev/null 2>&1; set -u; cd "$W/of0" && simpleFoam > run.log 2>&1 ) || true
grep -q '^Time = ' "$W/of0/run.log" \
    && say "control: real OpenFOAM runs the unfloored fixture" ok \
    || { tail -3 "$W/of0/run.log"; say "control: real OpenFOAM runs the unfloored fixture" FAIL; }
[ -z "$(bounds "$W/of0/run.log")" ] \
    && say "control: unfloored, OpenFOAM bounds NOTHING -- so the floors are the cause" ok \
    || say "control: unfloored, OpenFOAM bounds NOTHING -- so the floors are the cause" FAIL

# ---- ARM 1: real OpenFOAM, FLOORED -- the ORACLE ---------------------------------------------------
stage "$W/of1" yes
( set +u; . "$OFBASH" > /dev/null 2>&1; set -u; cd "$W/of1" && simpleFoam > run.log 2>&1 ) || true
OFN="$(order "$W/of1/run.log")"
OFMM="$(minmax "$W/of1/run.log")"
[ "$OFN" = "k epsilon " ] \
    && say "ORACLE: OpenFOAM's first two bounding lines are k then epsilon (constructor)" ok \
    || say "ORACLE: OpenFOAM's first two bounding lines are k then epsilon (constructor)" FAIL
[ "$(next2 "$W/of1/run.log")" = "epsilon k " ] \
    && say "ORACLE: and the NEXT two are epsilon then k -- correct()'s opposite order" ok \
    || say "ORACLE: and the NEXT two are epsilon then k -- correct()'s opposite order" FAIL
printf '        (OpenFOAM order: %s| then %s)\n' "${OFN:-nothing }" "$(next2 "$W/of1/run.log")"
first2 "$W/of1/run.log" | sed 's/^/          /'

# ---- ARM 2: brae, legacy simpleFoam driver ---------------------------------------------------------
stage "$W/b1" yes
( cd "$W/b1" && "$BRAE" -case "$W/b1" > run.log 2>&1 ) || true
B1="$(order "$W/b1/run.log")"
[ "$B1" = "$OFN" ] \
    && say "legacy simpleFoam: same first-two order as OpenFOAM (k then epsilon)" ok \
    || say "legacy simpleFoam: same first-two order as OpenFOAM (k then epsilon)" FAIL
[ "$(minmax "$W/b1/run.log")" = "$OFMM" ] \
    && say "legacy simpleFoam: min and max match OpenFOAM's byte for byte" ok \
    || say "legacy simpleFoam: min and max match OpenFOAM's byte for byte" FAIL
printf '        (brae legacy: %s| OpenFOAM: %s)\n' "${B1:-nothing }" "${OFN:-nothing }"
diff <(minmax "$W/b1/run.log") <(printf '%s\n' "$OFMM") | sed 's/^/          /' || true

# ---- ARM 3: brae, simpleFoamV2 ---------------------------------------------------------------------
stage "$W/b2" yes
( cd "$W/b2" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" -case "$W/b2" > run.log 2>&1 ) || true
B2="$(order "$W/b2/run.log")"
[ "$B2" = "$OFN" ] \
    && say "simpleFoamV2: same first-two order as OpenFOAM (k then epsilon)" ok \
    || say "simpleFoamV2: same first-two order as OpenFOAM (k then epsilon)" FAIL
[ "$(minmax "$W/b2/run.log")" = "$OFMM" ] \
    && say "simpleFoamV2: min and max match OpenFOAM's byte for byte" ok \
    || say "simpleFoamV2: min and max match OpenFOAM's byte for byte" FAIL
[ "$(next2 "$W/b2/run.log")" = "epsilon k " ] \
    && say "simpleFoamV2: correct()'s pair follows in the opposite order (discriminator live)" ok \
    || say "simpleFoamV2: correct()'s pair follows in the opposite order (discriminator live)" FAIL
printf '        (brae V2: %s| then %s| OpenFOAM: %s)\n' "${B2:-nothing }" "$(next2 "$W/b2/run.log")" "${OFN:-nothing }"

# ---- ARM 4: brae UNFLOORED -- the fail-proof. If these arms bounded unconditionally the ------------
# arms above would pass with the constructor bound deleted, because correct() would supply the line.
# Unfloored, nothing may appear before Time = on either driver.
stage "$W/b0" no
( cd "$W/b0" && "$BRAE" -case "$W/b0" > run.log 2>&1 ) || true
stage "$W/b0v2" no
( cd "$W/b0v2" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" -case "$W/b0v2" > run.log 2>&1 ) || true
[ -z "$(bounds "$W/b0/run.log")" ] && [ -z "$(bounds "$W/b0v2/run.log")" ] \
    && say "fail-proof: unfloored, neither brae driver bounds at all" ok \
    || say "fail-proof: unfloored, neither brae driver bounds at all" FAIL

# ---- ARM 5: the rhoSimpleFoam mirror, both arms, on its own fixture --------------------------------
# Its constructor bound lives in the HOST createFields, which the CUDA arm also runs before uploading,
# so both arms are the same code and both are checked -- a CUDA arm that stopped calling it would show
# up here as a missing line.
RHO="$ROOT/validation/sbMatched"
if [ -d "$RHO" ]; then
    stageRho() {   # stageRho <dir> <yes|no>
        rm -rf "$1"; mkdir -p "$1"
        cp -r "$RHO/constant" "$RHO/system" "$1/"
        cp -r "$RHO/0.orig" "$1/0"
        python3 - "$1" "$2" "$KMIN" "$EMIN" <<'PYEOF'
import re, sys
d, setkeys, kmin, emin = sys.argv[1:5]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*', '', s, flags=re.S)
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime 2;', s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval 2;', s)
open(c, 'w').write(s + '\n')
if setkeys == 'yes':
    t = d + '/constant/turbulenceProperties'; s = open(t).read()
    s2 = re.sub(r'(RAS\s*\n?\s*\{)', r'\1\n    kMin            %s;\n    epsilonMin      %s;' % (kmin, emin), s, count=1)
    assert s2 != s, 'no RAS sub-dict in the rho fixture'
    open(t, 'w').write(s2)
PYEOF
    }
    for arm in 1 cuda; do
        stageRho "$W/rho$arm" yes
        ( cd "$W/rho$arm" && BRAE_RHOSIMPLEFOAM_MIRROR=$arm "$BRAE" -case "$W/rho$arm" > run.log 2>&1 ) || true
        R="$(order "$W/rho$arm/run.log")"
        [ "$R" = "k epsilon " ] \
            && say "rhoSimpleFoam mirror ($arm): first two are k then epsilon (constructor)" ok \
            || say "rhoSimpleFoam mirror ($arm): first two are k then epsilon (constructor)" FAIL
        printf '        (%s: %s)\n' "$arm" "${R:-nothing }"
        first2 "$W/rho$arm/run.log" | sed 's/^/          /'
    done
    stageRho "$W/rho0" no
    ( cd "$W/rho0" && BRAE_RHOSIMPLEFOAM_MIRROR=1 "$BRAE" -case "$W/rho0" > run.log 2>&1 ) || true
    [ -z "$(bounds "$W/rho0/run.log")" ] \
        && say "fail-proof: unfloored, the rho mirror bounds nothing at all" ok \
        || say "fail-proof: unfloored, the rho mirror bounds nothing at all" FAIL
else
    say "rhoSimpleFoam mirror (fixture missing, skipped)" ok
fi

# ---- ARM 6: THE FAIL-PROOF. BRAE_CTOR_BOUND=0 removes the constructor bound and nothing else, so ---
# the same floored case must now open with correct()'s pair in the opposite order. If this arm still
# read `k epsilon` the order discriminator would be measuring something other than what it names, and
# every arm above would be vacuous.
stage "$W/fp" yes
( cd "$W/fp" && BRAE_CTOR_BOUND=0 "$BRAE" -case "$W/fp" > run.log 2>&1 ) || true
stage "$W/fpv2" yes
( cd "$W/fpv2" && BRAE_SIMPLEFOAM_V2=1 BRAE_CTOR_BOUND=0 "$BRAE" -case "$W/fpv2" > run.log 2>&1 ) || true
FP1="$(order "$W/fp/run.log")"; FP2="$(order "$W/fpv2/run.log")"
[ "$FP1" = "epsilon k " ] && [ "$FP2" = "epsilon k " ] \
    && say "fail-proof: without the constructor bound both drivers open epsilon-then-k" ok \
    || say "fail-proof: without the constructor bound both drivers open epsilon-then-k" FAIL
printf '        (legacy: %s| V2: %s| with the bound: %s)\n' "${FP1:-nothing }" "${FP2:-nothing }" "$OFN"
if [ -d "$RHO" ]; then
    stageRho "$W/fprho" yes
    ( cd "$W/fprho" && BRAE_CTOR_BOUND=0 BRAE_RHOSIMPLEFOAM_MIRROR=1 "$BRAE" -case "$W/fprho" > run.log 2>&1 ) || true
    FP3="$(order "$W/fprho/run.log")"
    [ "$FP3" != "k epsilon " ] \
        && say "fail-proof: without it the rho mirror does not open k-then-epsilon either" ok \
        || say "fail-proof: without it the rho mirror does not open k-then-epsilon either" FAIL
    printf '        (rho mirror without the bound: %s)\n' "${FP3:-nothing }"
fi

[ "$fail" = 0 ] && echo "== PASSED ==" || echo "== FAILED =="
exit "$fail"
