#!/usr/bin/env bash
# brae's interFoam under kOmegaSST against REAL OpenFOAM's, on RAS/waterChannel AS SHIPPED, field by
# field and solve by solve.
#
# THE METHOD is tests/interfoam_ras_dambreak_vs_openfoam.sh's: exactly N identical FIXED steps, both
# codes read at one instant, because an adaptive step turns any difference into a clock. The tutorial's
# own deltaT (0.1) is kept; only adjustTimeStep is turned off and the write made ASCII at 15 digits.
#
# WHAT THE CASE EXERCISES that no other gated interFoam case does:
#   kOmegaSST          the ordinary incompressible model (no `density` line, so the uniform lineage):
#                      the mixture's nu as a FIELD in F1, F2 and both diffusivities, the volumetric phi,
#                      validate() at construction, omega solved before k
#   the wall distance  wallDist's cell y for F1 and F2, beside the near-wall distance the wall
#                      functions take -- two fields OpenFOAM both calls y
#   the walls          omegaWallFunction (binomial n = 2) and nutkWallFunction on 6600 wall faces, k a
#                      kqRWallFunction
#   a pattern key      `"div\(phi,(k|omega)\)" Gauss upwind;` -- fvSchemes is a dictionary, and brae's
#                      scheme parser searched for the literal key and refused the case
#   the rest           flowRateInletVelocity (volumetric), MULESCorr, GAMG on p_rgh with a GaussSeidel
#                      smoother, a 3-D extruded mesh non-orthogonal to 13.7 degrees under
#                      `Gauss linear corrected` with no non-orthogonal corrector
# The mesh is the tutorial's own: blockMesh and two extrudeMesh passes, 28000 cells.
#
# THE CONTROL is OpenFOAM's own answer with `simulationType laminar` at the same instant: 51% of U.
#
# MEASURED, ten steps of 0.1: all 20 p_rgh, 10 alpha, 10 omega and 10 k iteration counts OpenFOAM's,
# initial residuals within 9.3e-13; alpha 7.9e-13, p_rgh 7.6e-13, U 4.3e-12, k 2.4e-12, omega 3.9e-12,
# nut 3.5e-12, the wall's nut 9.2e-13 of its largest value.
#
# WHAT THE GATE FOUND. pressureInletOutletVelocity's snGrad() was (stored value - cell)*deltaCoeffs,
# where OpenFOAM's directionMixed builds it from the valueFraction and the cell and never reads the
# stored value. At construction the atmosphere holds the file's (0 0 0) over cells at (1 0 0), so
# kOmegaSST's correctNut read a shear of 1/d there and wrote nut 3.7e-05 where OpenFOAM writes
# k/omega = 3.33e-02: that patch 100% out, the FIRST p_rgh residual of the run 7.1e-03, and at t = 1
# U 3.5e-03, omega 4.0e-02, nut 2.3e-01. The laminar run of the same case was already exact (6e-13),
# which is what put the defect in the closure's construction and nowhere else.
#
# BROKEN ONCE EACH, at t = 1 (U, omega):
#   validate() skipped, the first UEqn on the file's nut          5.1e-01, 1.7e-01
#   fvm::ddt dropped from both equations                          3.8e-01, 2.2e+01, 0 of 10 omega counts
#   the wall distance doubled                                     4.1e-04, 5.1e-01
#   PBiCGStab for the case's symGaussSeidel                       1.7e-04, 3.7e-02
#   water's nu as a scalar for the mixture's field                3.5e-05, 8.5e-03
#   the laplacian's non-orthogonal correction dropped             3.7e-07, 7.2e-04
# NOT DISCRIMINATED, measured: the inline (value - cell)*deltaCoeffs on k's, omega's and the wall U's
# boundary gradient in place of the patch's snGrad() (identical to the last digit: those patches hold
# refreshed values whenever the closure reads them), and relax() not called at the case's factor of 1.
#
# NOT CLAIMED: `density variable` with kOmegaSST, F3, decayControl, a wall-function blending other than
# binomial n = 2, a moving mesh and the device -- all refused by name; the scalarTransport function
# object `s`, which brae does not run (it does not feed back into the flow).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_inter_waterchannel_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/RAS/waterChannel"
STEPS=${STEPS:-10}
DT=${DT:-0.1}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: RAS/waterChannel tutorial not found at $SRC"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

W=${KEEP_W:-$(mktemp -d)}
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v blockMesh > /dev/null 2>&1   || { echo "SKIP: blockMesh not on PATH"; exit 77; }
command -v extrudeMesh > /dev/null 2>&1 || { echo "SKIP: extrudeMesh not on PATH"; exit 77; }
command -v interFoam > /dev/null 2>&1   || { echo "SKIP: interFoam not on PATH"; exit 77; }

END=$(python3 -c "print('%.10g' % ($STEPS*float('$DT')))")

# stage <profile>: copy the tutorial, apply the profile, fix the step, mesh it as Allrun.pre does, run
stage()
{
    local profile="$1"
    local C="$W/$profile"
    rm -rf "$C"
    cp -r "$SRC" "$C" || return 1
    rm -rf "$C"/[1-9]* "$C"/0 "$C"/processor* "$C"/log.*
    cp -r "$C/0.orig" "$C/0"
    grep -q "RASModel  *kOmegaSST;" "$C/constant/turbulenceProperties" \
        || { echo "FAIL: the tutorial no longer names kOmegaSST"; return 1; }
    grep -q "^density " "$C/constant/turbulenceProperties" \
        && { echo "FAIL: the tutorial now carries a \`density\` line, so the lineage is another"; return 1; }
    grep -q '"div\\(phi,(k|omega)\\)"' "$C/system/fvSchemes" \
        || { echo "FAIL: the tutorial no longer names div(phi,k) through a pattern key"; return 1; }
    if [ "$profile" = laminar ]; then
        sed -i 's/^simulationType .*/simulationType laminar;/' "$C/constant/turbulenceProperties"
    fi

    STEPS="$STEPS" DT="$DT" python3 - "$C" <<'PYEOF' || { echo "FAIL: staging $profile"; return 1; }
import os, re, sys
d = sys.argv[1]
n = int(os.environ['STEPS'])
dt = os.environ['DT']
c = os.path.join(d, 'system/controlDict')
s = open(c).read()
# the function objects write nothing the gate reads, and `s` is a scalarTransport brae does not run
s = re.sub(r'\nfunctions\s*\{.*\n\}\s*\n', '\n', s, flags=re.S)
for key, val in [('adjustTimeStep', 'no'), ('deltaT', dt), ('endTime', '%.10g' % (n*float(dt))),
                 ('writeControl', 'timeStep'), ('writeInterval', str(n)), ('writeFormat', 'ascii'),
                 ('writePrecision', '15')]:
    s, k = re.subn(r'^%s\s.*' % key, '%s %s;' % (key.ljust(15), val), s, flags=re.M)
    assert k == 1, key
s = re.sub(r'^writeCompression\s.*', 'writeCompression off;', s, flags=re.M)
open(c, 'w').write(s)
PYEOF
    ( cd "$C" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh [$profile]"; tail -20 "$C/log.blockMesh"; return 1; }
    local i
    for i in 1 2; do
        cp "$C/system/extrudeMeshDict.$i" "$C/system/extrudeMeshDict"
        ( cd "$C" && extrudeMesh > "log.extrudeMesh.$i" 2>&1 ) \
            || { echo "FAIL: extrudeMesh $i [$profile]"; tail -20 "$C/log.extrudeMesh.$i"; return 1; }
    done
    ( cd "$C" && setFields > log.setFields 2>&1 ) || { echo "FAIL: setFields [$profile]"; tail -20 "$C/log.setFields"; return 1; }
    ( cd "$C" && interFoam > log.interFoam 2>&1 ) || { echo "FAIL: interFoam [$profile]"; tail -30 "$C/log.interFoam"; return 1; }
    [ -d "$C/$END" ] || { echo "FAIL: OpenFOAM wrote no $END directory [$profile]"; ls "$C"; return 1; }
    echo "OpenFOAM ran $STEPS steps of deltaT $DT to t = $END   [$profile]"
}

rc=0
for p in laminar sst; do
    stage "$p" || { rc=1; break; }
done
[ $rc = 0 ] || { echo "interfoam_waterchannel_vs_openfoam: staging failed"; exit 1; }

# the oracle took the path
grep -q "Selecting RAS turbulence model kOmegaSST" "$W/sst/log.interFoam" \
    || { echo "FAIL: OpenFOAM's log does not select kOmegaSST"; exit 1; }
grep -q "Solving for omega" "$W/sst/log.interFoam" \
    || { echo "FAIL: OpenFOAM's log never solves omega"; exit 1; }
grep -q "Solving for omega" "$W/laminar/log.interFoam" \
    && { echo "FAIL: OpenFOAM's laminar control solved omega"; exit 1; }

"$BIN" "$W/sst" "$W/sst/0" "$W/sst/$END" "$STEPS" "$W/sst/log.interFoam" "$W/laminar/$END" || rc=1

echo "interfoam_waterchannel_vs_openfoam: rc $rc"
exit $rc
