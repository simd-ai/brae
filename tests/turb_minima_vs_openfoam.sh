#!/usr/bin/env bash
# kMin, epsilonMin and omegaMin, which brae hardcoded at 1e-15 and ignored.
#
# OpenFOAM reads all three from the TOP LEVEL of the RAS sub-dict -- not from a ...Coeffs sub-dict --
# via getOrAddToDict with a default of SMALL (RASModel.C:73-99, re-read at :180-182), and the LES sub-dict
# carries the same three keys for the DES arms (LESModel.C:82-111). They are the lower bound Foam::bound
# clamps to and, more importantly, the value its guard compares against: a case that raises kMin makes
# bound() fire where it otherwise would not. brae hardcoded 1e-15 at all eighteen of its bound sites, so
# a case naming any of the three was silently ignored -- the substitution class this project refuses.
#
# THE HARD PART IS THAT THREE PARSERS READ turbulenceProperties: the shared readTurbulenceModel, the V2
# simpleFoam driver's own, and the rhoSimpleFoam mirror's. A key added to one is a key silently dropped
# by the other two -- the shape the turbulence preconditioner policy was in until it was made one rule.
# So this runs the SAME case through all three and requires the same answer.
#
# THE ORACLE IS ALGEBRAIC, not a second code: after bound(field, floor), min(field) >= floor, by
# construction (bound.C:48-57). It is only a test if the floor BITES -- a floor below the case's own
# minimum is satisfied by a run that ignores it entirely, which is exactly how this would pass while
# doing nothing. So the floors here are set ABOVE the case's own minima, and the control measures those
# minima to prove it.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="$ROOT/validation/pitzDailyTurb"
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$SRC" ]  || { echo "SKIP: fixture $SRC missing"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
IT=10
KMIN=5
EMIN=5000

fail=0
say() { printf '  %-70s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

stage() {   # stage <dir> <set the keys? yes|no>
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
fmin() {    # fmin <case dir> <field> -> the written field's minimum
    python3 - "$1/$IT/$2" <<'PYEOF'
import re, sys
try: b = open(sys.argv[1], 'rb').read()
except OSError: print('nan'); raise SystemExit
m = re.search(rb'internalField\s+nonuniform\s+List<scalar>\s*\n?(\d+)\s*\n\(', b)
if not m:
    u = re.search(rb'internalField\s+uniform\s+([0-9.eE+-]+)', b)
    print(u.group(1).decode() if u else 'nan'); raise SystemExit
print('%.10g' % min(float(x) for x in b[m.end():].split(b')\n', 1)[0].split()))
PYEOF
}
ge() { python3 -c "import sys; a,b=float(sys.argv[1]),float(sys.argv[2]); print('yes' if a>=b else 'no')" "$1" "$2"; }
lt() { python3 -c "import sys; a,b=float(sys.argv[1]),float(sys.argv[2]); print('yes' if a<b else 'no')" "$1" "$2"; }

# ---- CONTROL FIRST: the case's OWN minima, so we know the floors below actually bite ---------------
stage "$W/base" no
( cd "$W/base" && "$BRAE" -case "$W/base" > run.log 2>&1 ) || { tail -3 "$W/base/run.log"; say "the fixture runs" FAIL; }
BK=$(fmin "$W/base" k); BE=$(fmin "$W/base" epsilon)
[ "$(lt "$BK" "$KMIN")" = yes ] && [ "$(lt "$BE" "$EMIN")" = yes ] \
    && say "the case's own minima are BELOW the floors used below (so they bite)" ok \
    || say "the case's own minima are BELOW the floors used below (so they bite)" FAIL
printf '        (unfloored: k min %s, epsilon min %s | floors %s and %s)\n' "$BK" "$BE" "$KMIN" "$EMIN"

# ---- the three parsers, each given the same case ---------------------------------------------------
# parser 1: the shared readTurbulenceModel, through the legacy simpleFoam driver
# parser 2: simpleFoamV2's own inline parse
# parser 3: the rhoSimpleFoam mirror's parse -- covered by its own fixture below
stage "$W/p1" yes
( cd "$W/p1" && "$BRAE" -case "$W/p1" > run.log 2>&1 ) || true
K1=$(fmin "$W/p1" k); E1=$(fmin "$W/p1" epsilon)
[ "$(ge "$K1" "$KMIN")" = yes ] && [ "$(ge "$E1" "$EMIN")" = yes ] \
    && say "shared parser (legacy simpleFoam): both floors honoured" ok \
    || say "shared parser (legacy simpleFoam): both floors honoured" FAIL
printf '        (k min %s >= %s, epsilon min %s >= %s)\n' "$K1" "$KMIN" "$E1" "$EMIN"

stage "$W/p2" yes
( cd "$W/p2" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" -case "$W/p2" > run.log 2>&1 ) || true
K2=$(fmin "$W/p2" k); E2=$(fmin "$W/p2" epsilon)
[ "$(ge "$K2" "$KMIN")" = yes ] && [ "$(ge "$E2" "$EMIN")" = yes ] \
    && say "V2 parser: both floors honoured (it parses the dict itself)" ok \
    || say "V2 parser: both floors honoured (it parses the dict itself)" FAIL
printf '        (k min %s >= %s, epsilon min %s >= %s)\n' "$K2" "$KMIN" "$E2" "$EMIN"

# ---- the floor is what MAKES bound() fire, so the message must appear ------------------------------
# The guard is min(field) < floor; raising the floor above the case's own minimum is precisely what
# turns bound() on. If the keys were ignored, nothing would bound and this would be silent.
grep -q '^bounding ' "$W/p1/run.log" \
    && say "raising the floor makes Foam::bound fire, and it says so" ok \
    || say "raising the floor makes Foam::bound fire, and it says so" FAIL
printf '        (%s)\n' "$(grep -m1 '^bounding ' "$W/p1/run.log" || echo 'no bounding line')"
grep -q '^bounding ' "$W/base/run.log" \
    && say "...and the unfloored control does NOT bound (the keys are the cause)" FAIL \
    || say "...and the unfloored control does NOT bound (the keys are the cause)" ok

# ---- parser 3: the rhoSimpleFoam mirror, on its own fixture ---------------------------------------
RHO="$ROOT/validation/sbMatched"
if [ -d "$RHO" ]; then
    rm -rf "$W/p3"; mkdir -p "$W/p3"
    cp -r "$RHO/constant" "$RHO/system" "$W/p3/"
    cp -r "$RHO/0.orig" "$W/p3/0"
    python3 - "$W/p3" "$KMIN" "$EMIN" <<'PYEOF'
import re, sys
d, kmin, emin = sys.argv[1:4]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*', '', s, flags=re.S)
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime 8;', s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval 8;', s)
open(c, 'w').write(s + '\n')
t = d + '/constant/turbulenceProperties'; s = open(t).read()
s2 = re.sub(r'(RAS\s*\n?\s*\{)', r'\1\n    kMin            %s;\n    epsilonMin      %s;' % (kmin, emin), s, count=1)
assert s2 != s, 'no RAS sub-dict in the rho fixture'
open(t, 'w').write(s2)
PYEOF
    ( cd "$W/p3" && BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BRAE" -case "$W/p3" > run.log 2>&1 ) || true
    K3=$(python3 -c "
import re,sys
b=open('$W/p3/8/k','rb').read(); m=re.search(rb'internalField\s+nonuniform\s+List<scalar>\s*\n?(\d+)\s*\n\(',b)
print('%.10g'%min(float(x) for x in b[m.end():].split(b')\n',1)[0].split()) if m else 'nan')" 2>/dev/null || echo nan)
    E3=$(python3 -c "
import re,sys
b=open('$W/p3/8/epsilon','rb').read(); m=re.search(rb'internalField\s+nonuniform\s+List<scalar>\s*\n?(\d+)\s*\n\(',b)
print('%.10g'%min(float(x) for x in b[m.end():].split(b')\n',1)[0].split()) if m else 'nan')" 2>/dev/null || echo nan)
    [ "$(ge "${K3:-0}" "$KMIN")" = yes ] && [ "$(ge "${E3:-0}" "$EMIN")" = yes ] \
        && say "rhoSimpleFoam mirror parser: both floors honoured" ok \
        || say "rhoSimpleFoam mirror parser: both floors honoured" FAIL
    printf '        (k min %s >= %s, epsilon min %s >= %s)\n' "${K3:-?}" "$KMIN" "${E3:-?}" "$EMIN"
else
    say "rhoSimpleFoam mirror parser (fixture missing, skipped)" ok
fi

# ---- realizableKE bounds rather than hard-clamps ---------------------------------------------------
# OpenFOAM calls bound(epsilon_, epsilonMin_) at realizableKE.C:307 and bound(k_, kMin_) at :329. brae
# ran `fmax(field, 1e-15)` over cells, AFTER evaluating the boundary -- the wrong operation on the wrong
# domain in the wrong order, announcing nothing. bound() gives a cell that solved NEGATIVE its
# neighbours' area-weighted average; a floor gives it 1e-15 and puts that in the next iteration's
# denominator. Upwind convection never produces the negative cell, so a limited scheme is what exposes it
# (bound_cpp.cuh).
RKE="$ROOT/validation/pitzDailyRKE"
if [ -d "$RKE" ]; then
    rm -rf "$W/rke"; mkdir -p "$W/rke"
    cp -r "$RKE/constant" "$RKE/system" "$W/rke/"
    if [ -d "$RKE/0.orig" ]; then cp -r "$RKE/0.orig" "$W/rke/0"; else cp -r "$RKE/0" "$W/rke/0"; fi
    python3 - "$W/rke" <<'PYEOF'
import re, sys
d = sys.argv[1]
f = d + '/system/fvSchemes'; s = open(f).read()
s = re.sub(r'div\(phi,k\)\s+[^;]+;', 'div(phi,k)      Gauss limitedLinear 1;', s)
s = re.sub(r'div\(phi,epsilon\)\s+[^;]+;', 'div(phi,epsilon) Gauss limitedLinear 1;', s)
open(f, 'w').write(s)
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*', '', s, flags=re.S)
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime 20;', s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval 20;', s)
open(c, 'w').write(s + '\n')
PYEOF
    ( cd "$W/rke" && "$BRAE" -case "$W/rke" > run.log 2>&1 ) || true
    grep -q "realizableKECoeffs" "$W/rke/run.log" \
        && say "the realizableKE arm really is running realizableKE (control)" ok \
        || say "the realizableKE arm really is running realizableKE (control)" FAIL
    grep -q '^bounding ' "$W/rke/run.log" \
        && say "realizableKE calls Foam::bound, which reports (it hard-clamped, silently)" ok \
        || { tail -2 "$W/rke/run.log"; say "realizableKE calls Foam::bound, which reports (it hard-clamped, silently)" FAIL; }
    printf '        (%s)\n' "$(grep -m1 '^bounding ' "$W/rke/run.log" || echo 'no bounding line')"
else
    say "realizableKE bounds rather than clamps (fixture missing, skipped)" ok
fi

[ "$fail" = 0 ] && echo "== PASSED ==" || echo "== FAILED =="
exit "$fail"
