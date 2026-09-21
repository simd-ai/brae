#!/usr/bin/env bash
# brae's interFoam across a CODED cyclicACMI baffle against REAL OpenFOAM's, on RAS/damBreakLeakage,
# field by field and solve by solve.
#
# THE CASE: createBaffles turns 13 internal faces into a cyclicACMI pair whose non-overlap patches are
# SYMMETRY planes, and whose `scale` is a coded PatchFunction1 -- zero until t > 0.5, then one on the two
# faces with 0.07 < y < 0.1. The water column stands behind the shut baffle for 500 steps and leaks
# through the opening for 20. Meshed as Allrun does (blockMesh, setFields, createBaffles), 2268 cells,
# kEpsilon, MULESCorr, three outer correctors; fixed steps of 1e-3; every solve pinned (see below).
#
# MEASURED, 520 steps from t = 0 (the baffle opens at step 500, t = 0.50000000000000033 > 0.5):
#   alpha 3.4e-13, p_rgh 1.5e-13, U 4.9e-12, k 9.2e-13, epsilon 2.2e-13, nut 3.5e-13; the flux through
#   the baffle's four patches 3.2e-14 of its largest; every alpha, k and epsilon count OpenFOAM's, 41 of
#   1560 p_rgh counts one apart at an edge stop.
# ...and 30 steps RESTARTED from OpenFOAM's own written state at t = 0.49 (it opens at step 11):
#   alpha 2.6e-13, p_rgh 1.1e-13, U 5.2e-12, k 7.8e-13; all 90 p_rgh counts equal.
# CONTROLS, OpenFOAM against itself: the baffle never opened, U 100%; opened on every face, 179%.
#
# WHAT THE GATE FOUND, each localised against OpenFOAM's own instrumented alphaEqn and pEqn:
#   1. MULES' limiter is synced across a coupled patch by syncTools::syncFaceList, which exchanges across
#      processor and cyclicPolyPatch ONLY -- not an AMI pair. OpenFOAM's alpha flux across an ACMI is not
#      equal and opposite (the receiving side 75% of the giving side's on the opening step); brae's was.
#   2. The rescale is lazy, and alphaEqn.H forms phic = cAlpha*|phi/magSf| BEFORE it: on the opening step
#      phic on the new faces is |phi| over the CLOSED area (41.7), times nHatf on the open one. brae
#      rescaled at the top of the step and put 5e-10 of water where OpenFOAM leaves none.
#   3. grad(U) on a symmetry patch is the mirror-averaged cell gradient (the constraint type in a derived
#      field) -- open since the wedge fix, first reached here: HbyA 3% out beside the baffle.
#   4. The host loop's clock started at 0 whatever the start directory said; the restart arm holds it.
#
# FAIL-PROOFS, each decision broken once in a scratch copy, the gate red every time (U, leak / restart):
#   the MULES limiter synced across the AMI pair       1.7e-01 / 1.7e-01
#   phic formed after the rescale                      1.7e-09 / 1.8e-09
#   no symmetry reflection of grad(U)                  7.8e-02 / 7.5e-02
#   the clock starting at 0                            green   / 1.0     (only a restart can see it)
#   the neighbour side left unscaled                   2.7     / 2.0
#   the non-overlap patch keeping its full area        7.0e-01 / 6.8e-01
#   the areas never rescaled                           1.0     / 1.0
#   the patches' |Sf| not reset                        1.8     / 1.8
# THE DEVICE ARM runs both arms of this gate -- the run from 0 and the restart -- on the pair COUPLED,
# from its own fresh geometry, patches and ACMI state (the rescale is in place, so the host run's
# objects are at the END state). The harness still asserts that the pair handed over UNCOUPLED is
# refused by name. What the device loop needed, each grounded in the file it cites:
#   the pair in the interface list, OPT-IN (the legacy drivers couple an ACMI through DeviceAMI), with
#   no MULES limiter sync across it -- syncFaceList covers processor and cyclicPolyPatch only
#   the rescale through the alpha step's geometryUpdate, after phic and before the pre-solve, once per
#   TIME STEP, with phic formed ahead of it; then the mesh's areas and the pair's OWN areas re-uploaded
#   IN PLACE and nothing else -- OpenFOAM keeps the cached weights and deltaCoeffs across a rescale
#   the symmetry mirror of grad(U) on the non-overlap patches, keyed on the MESH patch type
#   (tests/test_device_grad_symmetry.cu holds the kernel on its own)
#   the clock started at the start directory, which only the restart arm can see
# MEASURED, device against OpenFOAM (leak / restart): alpha 3.4e-11 / 3.6e-11, p_rgh 1.9e-11 / 2.0e-11,
# U 1.3e-09 / 1.3e-09, k 8.9e-11 / 1.0e-10, the baffle's flux 9.2e-12 of itself; the clock and the
# opened faces the host loop's, bit for bit. The device is DETERMINISTIC here (a second run differs by
# exactly 0), so those are its arithmetic and not noise: the test file says where they enter (one
# near-dry cell under the jet, MULES' gather order, then the density ratio).
# FAIL-PROOFS ON THE DEVICE ARM, each broken once (U unless it says otherwise):
#   MULES' limiter synced across the pair              1.7e-01          (the host arm's own number)
#   the pair's areas not re-uploaded                   1.0
#   the mesh geometry not re-uploaded                  7.5e-01
#   the clock starting at 0                            1.0 on the restart, and the baffle never opens
#   no symmetry mirror in grad(U)                      7.8e-02          (the host arm's own number)
#   phic formed AFTER the rescale                      k 5.8e-10 and the baffle's flux 2.9e-11 -- RED
#                                                      by those two checks ONLY; U reads 1.6e-09, inside
#                                                      this arm's floor. The narrowest of the set.
# NOT DISCRIMINATED on the device arm, and claimed as nothing more than correct by construction:
#   the grad(U) memo's fingerprint carrying the boundary areas (no digit moves without it: the one stale
#   hit it prevents is a single outer corrector at rest-state velocities), and |Sf| for phig laid out
#   in the device's face order rather than the mesh's (the fifth digit moves; phig is zero on every
#   patch that follows the pair here).
#
# NOT DISCRIMINATED: recomputing the face cells' volumes and centres after the rescale -- the pair's two
# areas sum to the same face, and the recomputed cells come out bitwise the same on this mesh.
# NOT CLAIMED: an ACMI pair that is not coincident face for face (OpenFOAM's AMI weights are then
# fractions -- refused), a moving mesh, the explicit MULES path, icAlpha/scAlpha and alpha sub-cycling
# with a moving scale (each moves the rescale point -- refused), and the device loop (refused).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_inter_leakage_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/RAS/damBreakLeakage"
STEPS=${STEPS:-520}
DT=${DT:-1e-3}
# THE RESTART ARM starts both codes from the state OpenFOAM writes RESTART_STEPS before the end
RESTART_STEPS=${RESTART_STEPS:-30}
# ...so the runs write every gcd(restart index, end index) steps: OpenFOAM's time index carries on from
# the restart (read back from <start>/uniform/time), and a timeStep writeInterval counts that index
WRITE_EVERY=$(python3 -c "import math; print(math.gcd($STEPS - $RESTART_STEPS, $STEPS))")

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: damBreakLeakage tutorial not found at $SRC"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

W=${KEEP_W:-$(mktemp -d)}
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
for t in blockMesh setFields createBaffles interFoam; do
    command -v "$t" > /dev/null 2>&1 || { echo "SKIP: $t not on PATH"; exit 77; }
done

END=$(python3 -c "
t = 0.0
for i in range($STEPS):
    t += float('$DT')
print('%.10g' % t)")

# stage <profile>: copy the tutorial, fix the step, apply the profile to the baffle's coded scale, mesh
# it as Allrun does, run
stage()
{
    local profile="$1"
    local C="$W/$profile"
    rm -rf "$C"
    cp -r "$SRC" "$C" || return 1
    rm -rf "$C"/[1-9]* "$C"/0 "$C"/processor* "$C"/log.* "$C"/dynamicCode
    grep -q "if (tm > 0.5)" "$C/system/createBafflesDict" \
        || { echo "FAIL: the tutorial's coded scale no longer opens at t > 0.5"; return 1; }
    grep -q "if(Fy\[i\] > 0.07 && Fy\[i\] < 0.1)" "$C/system/createBafflesDict" \
        || { echo "FAIL: the tutorial's coded scale no longer selects 0.07 < y < 0.1"; return 1; }
    case "$profile" in
        closed)  sed -i 's/if (tm > 0.5)/if (tm > 1e9)/' "$C/system/createBafflesDict" ;;
        allOpen) sed -i 's/if(Fy\[i\] > 0.07 \&\& Fy\[i\] < 0.1)/if(true)/' "$C/system/createBafflesDict" ;;
    esac

    STEPS="$STEPS" DT="$DT" WRITE_EVERY="$WRITE_EVERY" python3 - "$C" <<'PYEOF' || { echo "FAIL: staging $profile"; return 1; }
import os, re, sys
d = sys.argv[1]
n = int(os.environ['STEPS'])
dt = os.environ['DT']
t = 0.0
for i in range(n):
    t += float(dt)
c = os.path.join(d, 'system/controlDict')
s = open(c).read()
# written every RESTART_EVERY steps as well as at the end, so the restart arm below has a start
for key, val in [('adjustTimeStep', 'no'), ('deltaT', dt), ('endTime', '%.10g' % t),
                 ('writeControl', 'timeStep'), ('writeInterval', os.environ['WRITE_EVERY']), ('writeFormat', 'ascii'),
                 ('writePrecision', '15')]:
    s, k = re.subn(r'^%s\s.*' % key, '%s %s;' % (key.ljust(15), val), s, flags=re.M)
    assert k == 1, key
open(c, 'w').write(s)
# THE SOLVES ARE PINNED, in both codes: p_rgh to 1e-13 with relTol 0, U/k/epsilon to 1e-13, alpha to 1e-14.
# For 500 steps the column stands at rest behind the shut baffle, where every residual is round-off: at
# the case's own tolerances (p_rgh relTol 0.05, k and epsilon 1e-6) a comparison measures where two
# solvers stopped on that round-off. MEASURED with p_rgh alone pinned: k 2.9e-04 and U 6.1e-08 apart
# after 520 steps and 43 p_rgh counts two or more apart; with all pinned, k 9.2e-13 and U 4.9e-12.
v = os.path.join(d, 'system/fvSolution')
s = open(v).read()
for pat, val in [(r'(\n    p_rgh\s*\{[^}]*?tolerance\s+)[^;]+;', '1e-13'), (r'(\n    p_rgh\s*\{[^}]*?relTol\s+)[^;]+;', '0'),
                 (r'("\(U\|k\|epsilon\)\.\*"\s*\{[^}]*?tolerance\s+)[^;]+;', '1e-13'),
                 (r'("alpha\.water\.\*"\s*\{[^}]*?tolerance\s+)[^;]+;', '1e-14')]:
    s, k = re.subn(pat, r'\g<1>' + val + ';', s)
    assert k == 1, pat
open(v, 'w').write(s)
PYEOF
    ( cd "$C" && cp -r 0.orig 0 && blockMesh > log.blockMesh 2>&1 && setFields > log.setFields 2>&1 \
          && createBaffles -overwrite > log.createBaffles 2>&1 ) \
        || { echo "FAIL: meshing [$profile]"; return 1; }
    ( cd "$C" && interFoam > log.interFoam 2>&1 ) || { echo "FAIL: interFoam [$profile]"; tail -30 "$C/log.interFoam"; return 1; }
    [ -d "$C/$END" ] || { echo "FAIL: OpenFOAM wrote no $END directory [$profile]"; ls "$C"; return 1; }
    echo "OpenFOAM ran $STEPS steps of deltaT $DT to t = $END   [$profile]"
}

rc=0
for p in closed allOpen leak; do
    stage "$p" || { rc=1; break; }
done
[ $rc = 0 ] || { echo "interfoam_leakage_vs_openfoam: staging failed"; exit 1; }

"$BIN" "$W/leak" "$W/leak/0" "$W/leak/$END" "$STEPS" "$W/leak/log.interFoam" "$W/closed/$END" "$W/allOpen/$END" || rc=1

# THE RESTART ARM: OpenFOAM restarted from its own written state at RESTART, brae from the same files.
# The coded scale reads this->time(); a loop whose clock starts at 0 whatever the start directory says
# never reaches t > 0.5 in these steps and never opens the baffle.
RESTART=$(python3 -c "
t = 0.0
for i in range($STEPS - $RESTART_STEPS):
    t += float('$DT')
print('%.10g' % t)")
R="$W/restart"
rm -rf "$R"
mkdir -p "$R"
cp -r "$W/leak/constant" "$W/leak/system" "$W/leak/$RESTART" "$R/" || { echo "FAIL: no $RESTART written by the leak run"; exit 1; }
sed -i "s/^startTime .*/startTime       $RESTART;/" "$R/system/controlDict"
( cd "$R" && interFoam > log.interFoam 2>&1 ) || { echo "FAIL: interFoam [restart]"; tail -30 "$R/log.interFoam"; exit 1; }
[ -d "$R/$END" ] || { echo "FAIL: the restarted OpenFOAM wrote no $END directory"; ls "$R"; exit 1; }
echo "OpenFOAM restarted at t = $RESTART and ran $RESTART_STEPS steps to t = $END   [restart]"
"$BIN" "$R" "$R/$RESTART" "$R/$END" "$RESTART_STEPS" "$R/log.interFoam" "$W/closed/$END" "$W/allOpen/$END" || rc=1

echo "interfoam_leakage_vs_openfoam: rc $rc"
exit $rc
