#!/usr/bin/env bash
# fixedFluxPressure TOGETHER WITH MRF is refused on the rebuilt simpleFoam path, and neither of them
# alone is -- on a fixture that carries a REAL rotating cellZone.
#
# WHY THIS EXISTS. constrainPressure sets a fixedFluxPressure patch's gradient from the flux through it,
# and on a rotating case OpenFOAM makes that flux RELATIVE first: `if (MRF) MRF.makeRelative(...)`
# inside constrainPressure itself (constrainPressure.C:70). brae's deviceConstrainPressure has no such
# term, so on an MRF case the wall's relative flux does not cancel and the gradient would come from the
# absolute one -- a wrong Neumann value written under the boundary condition's own name, which is the
# silent-substitution failure this project keeps finding. pEqn.cu therefore refuses the COMBINATION.
#
# Nothing tested it. It was queue item 29(b), found while fixing 29(a): tests/simplefoam_v2_dispatch.sh
# had been asserting a fixedFluxPressure refusal that commit c2ff3e4 deliberately lifted, and the
# replacement written there refused for the WRONG REASON -- pitzDaily carries no cellZones, so any
# MRFProperties added to it is rejected for the missing zone (which that file already tests) and the
# block passed without ever reaching this code. Hence a fixture with a real zone: rotatingCylinders,
# whose MRF names cellZone `all` and turns 6400 cells at omega 100.
#
#   ARM      ffp wall + MRF -> must REFUSE, and the message must carry `constrainPressure` AND
#            `MRF.relative`. Matching a bare "MRF" is what let the pitzDaily attempt pass on the
#            missing-zone refusal instead of this one.
#   CONTROL 1  MRF alone (the fixture as it ships) -> RUNS, and its log must show the zone is live, or
#            the arm above could be refusing on an MRF that is not actually turning anything.
#   CONTROL 2  ffp wall alone, MRFProperties removed and the inner wall driven so the solve is not
#            trivial -> RUNS with a non-zero residual. fixedFluxPressure by itself is supported and
#            gated against real OpenFOAM by validation/ffpi_vs_openfoam.sh; what this asserts is that
#            the refusal above is the COMBINATION and not the boundary condition.
#
# Both mutations are CHECKED to have applied. A regex that silently matches nothing turns a refusal test
# into a test that the case runs, which is how 29(a) survived 65 commits.
#
# Fail-proof, 2026-09-03: with pEqn.cu's `if (in.mrf && dbP->nSnGradFaces > 0) throw` commented out, the
# arm runs to completion (exit 0) and this gate FAILS; the two controls are unaffected.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILDDIR="${BUILD:-$ROOT/build}"
BIN="$BUILDDIR/brae"
SRC="$ROOT/validation/rotatingCylinders"
N=${N:-3}

[ -x "$BIN" ] || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC/constant/polyMesh" ] || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$SRC/constant/polyMesh/cellZones" ] || { echo "SKIP: fixture ships no cellZones"; exit 77; }
[ -f "$SRC/constant/MRFProperties" ]      || { echo "SKIP: fixture ships no MRFProperties"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-70s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

stage()   # $1 dst
{
    rm -rf "$1"; cp -r "$SRC" "$1"; rm -rf "$1"/[1-9]* "$1"/log*
    sed -i 's/^endTime.*/endTime         '"$N"';/' "$1/system/controlDict"
}

# The fixedFluxPressure wall. The fixture writes both walls under one regex entry, so the mutation
# splits them: outerWall takes the condition under test, innerWall keeps zeroGradient.
ffp()   # $1 dir
{
    python3 - "$1" <<'PYFFP'
import sys
d = sys.argv[1]
f = d + '/0/p'
s = open(f).read()
old = '''    "innerWall|outerWall"
    {
        type            zeroGradient;
    }'''
new = '''    innerWall
    {
        type            zeroGradient;
    }

    outerWall
    {
        type            fixedFluxPressure;
        value           uniform 0;
    }'''
if old not in s:
    raise SystemExit("the fixture's 0/p no longer carries the combined wall entry this mutation edits")
open(f, 'w').write(s.replace(old, new))
PYFFP
}

# ---- ARM: the combination is refused ---------------------------------------------------------
stage "$W/arm"
ffp "$W/arm" || { echo "FAIL: could not apply the fixedFluxPressure mutation"; exit 1; }
grep -q "fixedFluxPressure" "$W/arm/0/p" || { echo "FAIL: the mutation left no fixedFluxPressure patch"; exit 1; }
grep -q "cellZone" "$W/arm/constant/MRFProperties" || { echo "FAIL: the fixture lost its MRF zone"; exit 1; }
( cd "$W/arm" && BRAE_SIMPLEFOAM_V2=1 "$BIN" > log 2>&1 )
rc=$?
if [ $rc -eq 0 ]; then
    say "fixedFluxPressure + MRF is refused" FAIL
    tail -3 "$W/arm/log"
else
    say "fixedFluxPressure + MRF is refused (exit $rc)" ok
    if grep -q "constrainPressure" "$W/arm/log" && grep -q "MRF.relative" "$W/arm/log"; then
        say "...and the refusal names constrainPressure and the MRF.relative term" ok
    else
        say "...and the refusal names constrainPressure and the MRF.relative term" FAIL
        sed -n '1,4p' "$W/arm/log"
    fi
fi

# ---- CONTROL 1: MRF alone runs, on a zone that is actually turning ----------------------------
stage "$W/mrfonly"
( cd "$W/mrfonly" && BRAE_SIMPLEFOAM_V2=1 "$BIN" > log 2>&1 )
[ $? -eq 0 ] && say "control: MRF without a fixedFluxPressure wall runs" ok \
             || { say "control: MRF without a fixedFluxPressure wall runs" FAIL; tail -3 "$W/mrfonly/log"; }
grep -qE "MRF: [1-9][0-9]* zone" "$W/mrfonly/log" \
    && say "control: the fixture's MRF zone is live (the arm is not refusing on a dead zone)" ok \
    || { say "control: the fixture's MRF zone is live (the arm is not refusing on a dead zone)" FAIL; grep -i mrf "$W/mrfonly/log" | head -2; }

# ---- CONTROL 2: the fixedFluxPressure wall alone runs, and does work --------------------------
stage "$W/ffponly"
ffp "$W/ffponly" || { echo "FAIL: could not apply the fixedFluxPressure mutation"; exit 1; }
rm -f "$W/ffponly/constant/MRFProperties"
# Without the rotation nothing drives this closed annulus, and a run whose residuals are identically
# zero would not show the pressure assembly touching the patch at all. Drive the inner wall instead.
python3 - "$W/ffponly" <<'PYDRIVE'
import sys
d = sys.argv[1]
f = d + '/0/U'
s = open(f).read()
old = '''    innerWall
    {
        type            noSlip;
    }'''
new = '''    innerWall
    {
        type            fixedValue;
        value           uniform (0 1 0);
    }'''
if old not in s:
    raise SystemExit("the fixture's 0/U no longer carries the innerWall entry this mutation edits")
open(f, 'w').write(s.replace(old, new))
PYDRIVE
( cd "$W/ffponly" && BRAE_SIMPLEFOAM_V2=1 "$BIN" > log 2>&1 )
[ $? -eq 0 ] && say "control: a fixedFluxPressure wall without MRF runs" ok \
             || { say "control: a fixedFluxPressure wall without MRF runs" FAIL; tail -3 "$W/ffponly/log"; }
# ...and it solved something: a residual line whose U value is not 0.000000e+00.
grep -qE "U initial residual = [1-9]" "$W/ffponly/log" \
    && say "control: ...and that run is not a trivial zero-residual solve" ok \
    || { say "control: ...and that run is not a trivial zero-residual solve" FAIL; grep "initial residual" "$W/ffponly/log" | tail -2; }

[ $fail = 0 ] && echo PASS || { echo FAIL; exit 1; }
