#!/usr/bin/env bash
# rhoSimpleFoam's UEqn.H against REAL OpenFOAM's own assembled momentum matrix.
#
# THE ORACLE is tools/dumpPEqn: OpenFOAM's rhoSimpleFoam carrying a stage harness that writes, at SIMPLE
# iteration BRAE_DUMP_STAGE_ITER, the momentum equation's observable content --
#
#   stage_rAU    1/UEqn.A()                        the diagonal, AFTER relax()
#   stage_UIC    UEqn.internalCoeffs() per patch
#   stage_UBC    UEqn.boundaryCoeffs()  per patch
#   stage_muEff  turbulence->muEff()                the DYNAMIC viscosity the assembly used
#
# Dumping at iteration 1 means the state assembled from is the START-TIME field set, which brae can
# reconstruct exactly through createFields -- so the comparison isolates UEqn.H and carries no accumulated
# trajectory difference.
#
# stage_muEff IS INJECTED INTO brae rather than recomputed. brae has no ported compressible turbulence
# model yet (a separate manifest component), and mixing an unported closure into this measurement would
# produce a number that cannot be attributed to either. With OpenFOAM's own muEff supplied, a failure here
# means the momentum ASSEMBLY is wrong and nothing else.
#
# THE CONTROL, which is the reason this gate exists at all: the binary also assembles with the KINEMATIC
# nu_eff -- what the incompressible divDevReff carries -- and requires that to DISAGREE with OpenFOAM.
# The factor of rho between the two is the entire difference between this solver's momentum equation and
# simpleFoam's, and it is a factor that looks plausible in every field plot, so a bound both forms passed
# would gate nothing.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_rho_ueqn_cpp"
SRC="$ROOT/validation/sbMatched"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
cp -r "$SRC" "$W/case" || exit 1
rm -rf "$W"/case/[1-9]* "$W"/case/0 "$W"/case/processor* "$W"/case/log.*

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u

DUMP="$(command -v dumpPEqn || true)"
[ -z "$DUMP" ] && [ -n "${FOAM_USER_APPBIN:-}" ] && [ -x "$FOAM_USER_APPBIN/dumpPEqn" ] \
    && DUMP="$FOAM_USER_APPBIN/dumpPEqn"
[ -n "$DUMP" ] || { echo "SKIP: dumpPEqn not built -- (cd tools/dumpPEqn && wmake)"; exit 77; }
[ -d "$W/case/constant/polyMesh" ] || { echo "SKIP: fixture ships no mesh"; exit 77; }

cp -r "$W/case/0.orig" "$W/case/0"

# THE INLET IS REPLACED BY A PLAIN fixedValue, and that is deliberate isolation, not convenience.
# THE FIXTURE'S OWN INLET, as it ships. This gate used to neutralise sbMatched's flowRateInletVelocity
# because brae's inlet disagreed with OpenFOAM by 2.4e-01 and would have dominated a number meant to be
# about the momentum assembly. That boundary condition is now ported: OpenFOAM recomputes its value in
# updateCoeffs() from the registered rho's PATCH values, at construction (createFields.H builds rho before
# U) and again at every momentum assembly. With that in place the inlet carries EXACTLY the prescribed
# massFlowRate -- sum(phi) = -0.5 against OpenFOAM's -0.5 -- and rAU went 4.58e-05 -> 6.13e-15,
# boundaryCoeffs 4.15e-01 -> 4.89e-16. Neutralising it now would only hide the coupling it exercises.

python3 - "$W/case" <<'PYEOF'
import os, re, sys
c = os.path.join(sys.argv[1], 'system/controlDict')
s = open(c).read()
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
s = re.sub(r'^writeFormat .*',    'writeFormat     ascii;',    s, flags=re.M)
s = re.sub(r'^writePrecision .*', 'writePrecision  15;',       s, flags=re.M)
s = re.sub(r'^endTime .*',        'endTime         1;',        s, flags=re.M)
s = re.sub(r'^writeInterval .*',  'writeInterval   1;',        s, flags=re.M)
s = re.sub(r'^writeControl .*',   'writeControl    timeStep;', s, flags=re.M)
open(c, 'w').write(s)
PYEOF

# One SIMPLE iteration, dumping the momentum stages from it.
( cd "$W/case" && BRAE_DUMP_STAGE_ITER=1 "$DUMP" > dump.log 2>&1 ) \
    || { echo "FAIL: dumpPEqn did not run"; tail -20 "$W/case/dump.log"; exit 1; }
for f in stage_rAU stage_UIC stage_UBC stage_muEff stage_Uass; do
    [ -f "$W/case/1/$f" ] \
        || { echo "FAIL: dumpPEqn wrote no 1/$f"; tail -20 "$W/case/dump.log"; exit 1; }
done

"$BIN" "$W/case" 0 1 || exit 1

# ---- ARM 2: naca0012, whose freestream U patch is the one the boundary gradient needs ------------
#
# WHY A SECOND FIXTURE. sbMatched's U patches are a flowRateInletVelocity (fixedValue) and an inletOutlet
# whose valueFraction is 0 or 1 per face, and on all three of those OF's snGrad() IS the base class's
# (value - patchInternalField)*deltaCoeffs -- so the arm above cannot tell that formula from the patch's
# own. naca0012's freestreamVelocity carries a CONTINUOUS valueFraction, where the two part company:
# OF's gaussGrad boundary correction asks the patch for snGrad() (gaussGrad.C), and a mixed patch answers
# with the CURRENT valueFraction while value() still holds the blend of the previous one. brae inlined the
# base formula and read a boundary grad(U) 6.4e-04 out on this inlet, the dev2 tensor with it, and
# fvc::div of that tensor -- the explicit half of divDevRhoReff -- 3.0e-07 out, with the diagonal,
# off-diagonals, internalCoeffs and boundaryCoeffs all exact to 1e-15 (queue item 25). The harness's own
# `div of that tensor` row is what fails there, so this arm needs no bound of its own.
#
# Fail-proof, 2026-09-03, measured through this gate: with fvc::gradUBoundary back on the inline formula
# the naca arm exits 1 -- `gradU BOUNDARY on inlet 6.4e-04`, `div of that tensor 3.0e-07 FAIL` -- while
# the sbMatched arm above stays green at 8.3e-15 and 1.1e-14, which is the point: that fixture cannot
# tell the two formulas apart.
[ -f "${FOAM_TUTORIALS:-}/resources/geometry/NACA0012.obj.gz" ] \
    || { echo "  (naca arm SKIPPED: the NACA0012 geometry is not in this OpenFOAM install)"; exit 0; }
[ -d "$ROOT/validation/naca0012" ] || { echo "  (naca arm SKIPPED: fixture missing)"; exit 0; }
cp -r "$ROOT/validation/naca0012" "$W/naca" || exit 1
rm -rf "$W"/naca/[1-9]* "$W"/naca/0 "$W"/naca/log.*
cp -r "$W/naca/0.orig" "$W/naca/0"
# brae refuses an fvOptions file rather than dropping the constraint; the naca gates remove it on both
# sides, and this arm compares the momentum assembly, which the limitTemperature constraint does not touch.
rm -f "$W/naca/system/fvOptions"
( cd "$W/naca" && mkdir -p constant/geometry \
    && cp -f "$FOAM_TUTORIALS/resources/geometry/NACA0012.obj.gz" constant/geometry/ \
    && blockMesh > log.blockMesh 2>&1 && transformPoints -scale '(1 0 1)' > log.transformPoints 2>&1 \
    && extrudeMesh > log.extrudeMesh 2>&1 && topoSet > log.topoSet 2>&1 ) \
    || { echo "FAIL: naca mesh"; exit 1; }
python3 - "$W/naca" <<'PYNACA'
import os, re, sys
c = os.path.join(sys.argv[1], 'system/controlDict')
s = open(c).read()
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
for k, v in [('writeFormat', 'ascii'), ('writePrecision', '15'), ('endTime', '2'),
             ('writeInterval', '1'), ('writeControl', 'timeStep'), ('startFrom', 'startTime'),
             ('startTime', '0'), ('deltaT', '1')]:
    s = re.sub(r'\b%s\s+[^;]*;' % k, '%s %s;' % (k, v), s)
open(c, 'w').write(s)
PYNACA
# Iteration 2, not 1: at iteration 1 the freestream valueFraction is still the 0.5 both sides seed, so
# value and valueFraction agree and the two snGrad formulas coincide -- the arm would pass either way.
( cd "$W/naca" && BRAE_DUMP_STAGE_ITER=2 "$DUMP" > dump.log 2>&1 ) \
    || { echo "FAIL: dumpPEqn did not run on naca"; tail -20 "$W/naca/dump.log"; exit 1; }
[ -f "$W/naca/2/stage_UgradU" ] || { echo "FAIL: dumpPEqn wrote no naca 2/stage_UgradU"; exit 1; }
echo "  ---- naca0012, iteration 2 (the freestreamVelocity boundary gradient) ----"
"$BIN" "$W/naca" 1 2
