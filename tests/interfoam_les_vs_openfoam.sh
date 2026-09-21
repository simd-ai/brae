#!/usr/bin/env bash
# brae's interFoam with LES kEqn against REAL OpenFOAM's, on LES/nozzleFlow2D, field by field and solve by
# solve.
#
# THE CASE: a diesel-like nozzle, AXISYMMETRIC -- a wedge one cell thick, 20603 cells after Allrun's two
# topoSet/refineMesh passes (300 prisms on the axis, 225 split-hex polyhedra), non-orthogonal to 40
# degrees. Fuel enters at 460 m/s; LESModel kEqn with `delta smooth` around cubeRootVol and maxDeltaRatio
# 1.1; `div(phi,k) Gauss limitedLinear 1`; p_rgh by GAMG and PCG-GAMG; four alpha sub-cycles; two
# correctors with one non-orthogonal pass. The tutorial adjusts its step (its first is 1.2e-09); the gate
# runs exactly N FIXED steps of 1e-9 in both codes -- at 1e-8 OpenFOAM itself diverges by step two.
#
# WHAT kEqn DOES (kEqn.C:130-189), with alpha = rho = 1 in the uniform lineage:
#   ddt(k) + div(phi, k) - laplacian(nut + nu, k)
#       == nut*(grad(U) && devTwoSymm(grad(U))) - SuSp(2/3 div(phi), k) - Sp(Ce*sqrt(k)/delta, k)
#   then bound(k, kMin) and nut = Ck*sqrt(k)*delta. Ce and kMin come from the LES dictionary itself, Ck from
#   kEqnCoeffs. validate() is correctNut, before the first UEqn.
# THE FILTER WIDTH: cubeRootVol takes sqrt(V/thickness) on a 2-D mesh -- and a wedge IS 2-D to it
# (polyMesh::calcDirections knocks the wedge normal out of geometricD) -- with the thickness the mesh's
# bounding-box span in that direction; smooth then raises a cell's delta to a neighbour's over
# maxDeltaRatio by a FaceCellWave that takes a new value only when it is 1% larger, so the visit order is
# part of the answer. It raises 3837 of the 20603 cells here.
#
# THE CONTROL is OpenFOAM's own answer run laminar at the same instant: 19% of U.
#
# MEASURED, 100 steps: alpha 6.6e-12, p_rgh 8.2e-11, U 4.5e-12, k 2.1e-12, nut 1.1e-12; all 400 p_rgh and
# 100 k iteration counts OpenFOAM's, every k final residual OpenFOAM's; the filter width 1.1e-15 before and
# after smoothing against the delta OpenFOAM writes, and 8.2e-16 on the same mesh made 3-D (`delta3d`,
# the wedge planes made ordinary patches, where cubeRootVol takes cbrt(V)).
#
# WHAT THE GATE FOUND, and none of it was the model. The case run LAMINAR in both codes -- the twin that
# separates the model from everything around it -- was 6.7e-05 out in U after 20 steps:
#   1. THE HOST WEDGE HAD NO MATRIX COEFFICIENTS. WedgePatchField overrode evaluate() alone, so a vector
#      wedge assembled with the zeroGradient coefficients (valueInternalCoeffs 1, gradientInternalCoeffs 0)
#      where OpenFOAM's transformFvPatchField has 1 - d and -deltaCoeffs*d, d = 0.5*(1 - cellT_kk); its
#      snGrad was the base class's. No host gate had a wedge. UEqn.A() was 2e-05 low in every cell and
#      1.7e-04 low in the axis corner, against OpenFOAM's dumped A() (tools/dumpInterFoam).
#   2. THE GRADIENT'S WEDGE VALUE WAS NOT ROTATED. grad(U)'s patch field on a wedge is itself a wedge
#      (fvPatchField::New puts the constraint type in), so gaussGrad's correctBoundaryConditions starts
#      from faceT & G & faceT^T, not the cell gradient. The explicit viscous term read the unrotated one:
#      HbyA 2e-06 out in every cell at step two.
#   With both, the laminar twin is 6.1e-13 after 20 steps, every p_rgh count equal. Then kEqn itself:
#   3. THE INTERPOLATION'S ARITHMETIC. At step one k was 2.3e-03 out while every term but convection
#      matched tools/dumpKEqn (an instrumented copy of OpenFOAM's kEqn). k is uniform 1e-11 over most of the
#      mesh then, so limitedLinear's r sees phiN - phiP = 0 exactly and NVDTVD's 1000x guard decides the
#      limiter by the SIGN of the upwind cell's gradient -- the round-off of a gradient of a uniform field.
#      OpenFOAM interpolates as lambda*(P - N) + N (surfaceInterpolationScheme.C:270), which returns the
#      value exactly on such a face; brae's w*P + (1 - w)*N can miss by an ulp, and that ulp was the
#      gradient. With OpenFOAM's form in fvc::gaussGrad, every kEqn stage matches the dump to 1e-12 and k
#      is 3.9e-14 after one step.
#
# BROKEN ONCE EACH (U, k, p_rgh counts equal of 400):
#   the wedge's host coefficients as the base class's (found)   1.1e-04  1.6e-06  400
#   grad(U)'s wedge value unrotated (found)                     2.9e-04  4.5e-04  399
#   brae's interpolation arithmetic in gaussGrad (found)        3.7e-09  5.1e-08  400
#   the delta not smoothed                                      3.2e-06  1.7e-05  400
#   cbrt(V) on the wedge, the 3-D branch                        4.8e-01  1.3e+01  339
#   the smoothing wave without its 1% tolerance                 1.3e-09  3.4e-09  400   (delta 2.9e-03)
#   no production                                               1.9e-01  1.0e+00  388
#   no dissipation, Ce*sqrt(k)/delta                            9.4e-02  2.3e+00  395
#   no SuSp(2/3 divU)                                           7.7e-12  7.7e-10  400
#   DkEff without nu                                            1.4e-03  1.8e-02  400
#   upwind for limitedLinear 1                                  9.2e-03  9.7e-02  398
#   validate() skipped, nut from the file                       2.2e-04  4.7e-04  400
#   twoSymm for devTwoSymm in G                                 2.6e-04  1.8e-03  400
#   nut not recomputed after the k solve                        1.9e-01  1.0e+00  388
# NOT DISCRIMINATED: grad(alpha)'s wedge value rotated for the interface normal (also OpenFOAM's, also
# ported) -- provably invisible, since the normal correction removes the face-normal component and nHatf
# takes only that component; Ce and kMin read from the LES dictionary rather than kEqnCoeffs, which the
# case leaves at their defaults.
#
# NOT CLAIMED, each refused by name (tests/interfoam_refusals.sh): every LESModel but kEqn, every LESdelta
# but cubeRootVol and smooth around it, `density variable` with LES, a k convection scheme other than
# upwind and limitedLinear, a gradient scheme other than Gauss linear, nut wall functions under LES, a
# coupled patch under the smooth delta; LES kEqn on a mesh that MOVES, on the device (its filter width
# is taken once).
#
# THE DEVICE ARM runs the case AS SHIPPED: kEqn (device_les_keqn.cu, a transcription of the host file
# this gate holds), `delta smooth`, GAMG for p_rgh and the GAMG preconditioner for p_rghFinal, on a
# WEDGE. MEASURED against OpenFOAM, 100 steps: alpha 6.6213e-12, p_rgh 8.2519e-11, U 4.5231e-12, k
# 2.1028e-12, nut 1.0508e-12 -- the host arm's 6.6214e-12, 8.2335e-11, 4.5224e-12, 2.1046e-12 and
# 1.0518e-12, which is what bounds it.
#
# IT WAS REFUSED TWICE, and the second refusal hid three defects that were none of them the closure's.
# With kEqn ported the case read U 1.7e-01 from OpenFOAM after ONE step with alpha exact -- and
# interFoam.C runs turbulence->correct() AFTER the pressure corrector, so step one's momentum reads the
# nut the 0 directory holds. The LAMINAR twin this script already stages reproduced the gap with no
# closure at all, and the four p_rgh solves of step one localised it: the first corrector's two took
# OpenFOAM's residuals to every digit, the second corrector's started at 6.5e-03 for 3.1e-04. All three
# are WEDGE defects, invisible while U is at rest, and this is the only gated case on a wedge:
#   the wedge's refValue never refreshed -- every other driver calls deviceUpdateWedge with
#   deviceUpdateSymmetry as a pair, and interFoam's device step called the second alone
#                                                   BROKEN ONCE: alpha 5.7e-01, U 6.8e-01
#   grad(U)'s OWN patch value on a wedge, faceT & G & faceT^T, not rotated, and the wedge's snGrad
#   taken in the mixed slot's form rather than (cellT & pif - pif)*0.5*deltaCoeffs -- the fix the
#   HOST needed on this same case (fvc.cu:637-672)  BROKEN ONCE: alpha 4.1e-04, U 2.9e-04, k 4.5e-04
#   the wedge's laplacian gradientBoundaryCoeffs: the mixed slot's one refValue is spent reproducing
#   the VALUE (faceT), and OpenFOAM's gradient coefficient takes cellT and half the deltaCoeffs --
#   second order in the wedge angle            BROKEN ONCE: alpha 3.5e-05, U 4.2e-05, k 6.6e-06
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_inter_les_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/LES/nozzleFlow2D"
STEPS=${STEPS:-100}
DT=${DT:-1e-9}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: LES/nozzleFlow2D tutorial not found at $SRC"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

W=${KEEP_W:-$(mktemp -d)}
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
for t in blockMesh topoSet refineMesh interFoam; do
    command -v "$t" > /dev/null 2>&1 || { echo "SKIP: $t not on PATH"; exit 77; }
done

END=$(python3 -c "print('%.10g' % ($STEPS*float('$DT')))")

# stage <profile>: copy the tutorial, fix the step, mesh it as Allrun does, apply the profile, run
stage()
{
    local profile="$1"
    local C="$W/$profile"
    rm -rf "$C"
    cp -r "$SRC" "$C" || return 1
    rm -rf "$C"/[1-9]* "$C"/0 "$C"/processor* "$C"/log.*
    cp -r "$C/0.orig" "$C/0"
    grep -q "LESModel  *kEqn;" "$C/constant/turbulenceProperties" \
        || { echo "FAIL: the tutorial no longer runs LESModel kEqn"; return 1; }
    grep -q "^ *delta  *smooth;" "$C/constant/turbulenceProperties" \
        || { echo "FAIL: the tutorial's LES delta is no longer smooth"; return 1; }

    STEPS="$STEPS" DT="$DT" python3 - "$C" <<'PYEOF' || { echo "FAIL: staging $profile"; return 1; }
import os, re, sys
d = sys.argv[1]
n = int(os.environ['STEPS'])
dt = os.environ['DT']
c = os.path.join(d, 'system/controlDict')
s = open(c).read()
s = re.sub(r'\nfunctions\s*\{.*\n\}\s*\n', '\n', s, flags=re.S)
for key, val in [('startFrom', 'startTime'), ('adjustTimeStep', 'no'), ('deltaT', dt),
                 ('endTime', '%.10g' % (n*float(dt))), ('writeControl', 'timeStep'),
                 ('writeInterval', str(n)), ('writeFormat', 'ascii'), ('writePrecision', '15')]:
    s, k = re.subn(r'^%s\s.*' % key, '%s %s;' % (key.ljust(15), val), s, flags=re.M)
    assert k == 1, key
s = re.sub(r'^writeCompression\s.*', 'writeCompression off;', s, flags=re.M)
open(c, 'w').write(s)
PYEOF
    ( cd "$C" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh [$profile]"; tail -20 "$C/log.blockMesh"; return 1; }
    for i in 1 2; do
        ( cd "$C" && topoSet -dict system/topoSetDict.$i > log.topoSet.$i 2>&1 \
              && refineMesh -dict system/refineMeshDict -overwrite > log.refineMesh.$i 2>&1 ) \
            || { echo "FAIL: topoSet/refineMesh $i [$profile]"; return 1; }
    done
    if [ "$profile" = laminar ]; then
        # THE CONTROL: the same case with no turbulence model
        sed -i 's/^simulationType .*/simulationType      laminar;/' "$C/constant/turbulenceProperties"
    fi
    if [ "$profile" = delta3d ]; then
        # THE 3-D BRANCH of cubeRootVol, which the wedge never reaches: the same mesh with its two wedge
        # planes made ordinary patches, so no patch knocks a direction out and nGeometricD is 3
        python3 - "$C" <<'PYEOF' || { echo "FAIL: staging $profile"; return 1; }
import os, re, sys
d = sys.argv[1]
b = os.path.join(d, 'constant/polyMesh/boundary')
t = open(b).read()
t, k = re.subn(r'(\n\s*(front|back)\s*\{[^}]*?type\s+)wedge;', r'\1patch;', t)
assert k == 2, k
t = re.sub(r'inGroups\s+1\(wedge\);', 'inGroups        1(patch);', t)
open(b, 'w').write(t)
for f in os.listdir(os.path.join(d, '0')):
    p = os.path.join(d, '0', f)
    if not os.path.isfile(p):
        continue
    u = open(p).read()
    u = re.sub(r'type\s+wedge;', 'type zeroGradient;', u)
    open(p, 'w').write(u)
PYEOF
    fi
    if [ "$profile" = delta ] || [ "$profile" = delta3d ]; then
        # OpenFOAM's own filter width, before and after smoothing, written at construction
        ( cd "$C" && interFoam -postProcess -func "writeObjects(delta,geometricDelta)" -time 0 > log.delta 2>&1 ) \
            || { echo "FAIL: writeObjects [$profile]"; tail -20 "$C/log.delta"; return 1; }
        [ -f "$C/0/delta" ] && [ -f "$C/0/geometricDelta" ] || { echo "FAIL: OpenFOAM wrote no delta"; return 1; }
        echo "OpenFOAM wrote its delta and geometricDelta   [$profile]"
        return 0
    fi
    ( cd "$C" && interFoam > log.interFoam 2>&1 ) || { echo "FAIL: interFoam [$profile]"; tail -30 "$C/log.interFoam"; return 1; }
    [ -d "$C/$END" ] || { echo "FAIL: OpenFOAM wrote no $END directory [$profile]"; ls "$C"; return 1; }
    echo "OpenFOAM ran $STEPS steps of deltaT $DT to t = $END   [$profile]"
}

rc=0
for p in delta delta3d laminar les; do
    stage "$p" || { rc=1; break; }
done
[ $rc = 0 ] || { echo "interfoam_les_vs_openfoam: staging failed"; exit 1; }

# the oracle took the path
grep -q "Selecting LES turbulence model kEqn" "$W/les/log.interFoam" \
    || { echo "FAIL: OpenFOAM's log does not select kEqn"; exit 1; }
grep -q "Selecting LES delta type smooth" "$W/les/log.interFoam" \
    || { echo "FAIL: OpenFOAM's log does not select the smooth delta"; exit 1; }

"$BIN" "$W/les" "$W/les/0" "$W/les/$END" "$STEPS" "$W/les/log.interFoam" "$W/laminar/$END" "$W/delta/0" \
       "$W/delta3d" || rc=1

echo "interfoam_les_vs_openfoam: rc $rc"
exit $rc
