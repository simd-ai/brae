#!/usr/bin/env bash
# brae's interFoam on an ADAPTIVE mesh against REAL OpenFOAM's -- unit 8e, the last unit of the
# dynamicRefineFvMesh port and the first arm that runs the WHOLE chain through the solver:
# selection -> hexRef8 -> removeFaces -> changeMesh -> the six maps -> every field's autoMap -> the
# solver's own rebuild -> the next step's ddt terms.
#
# THE ORACLE is real OpenFOAM v2412's interFoam on laminar/damBreakWithObstacle, two fixed steps of
# deltaT 1e-3, written at `writePrecision 18` so the comparison floor is floating point and not the
# file. A refining case writes its own polyMesh into every time directory, so the comparison is on
# OpenFOAM's mesh, cell by cell -- and the mesh arms assert that numbering first, because a field
# difference on a differently-numbered mesh would be unreadable.
#
# WHAT THE TWO STEPS EXERCISE, measured from OpenFOAM's own log:
#   step 1   1,430 cells refined   32,256 -> 42,266 cells   pcorr: initial residual 0, 0 iterations
#   step 2   5,714 cells refined   42,266 -> 82,264 cells   pcorr: initial residual 1, 8 iterations
# So step 1 maps fields onto a refined mesh and needs no flux correction, and step 2 needs the
# correction -- 8 GAMG iterations of it. One step alone would have passed with the flux rebuild missing.
#
# WHAT IT MEASURED on 2026-09-27, every arm green. These are FLOATING-POINT FLOORS, not tolerances:
# on this case brae reproduces OpenFOAM's own arithmetic, and the bounds in the .cu sit just above them.
#   alpha    1.8848e-15 absolute on 82,264 cells      patch values: alpha 0.0e+00 (exact)
#   p_rgh    2.7285e-11 / 3.7065e-15 relative                       U     1.4268e-16
#   U        2.2053e-13 / 2.4051e-13 relative                       p_rgh 2.7285e-11
#   p        2.7626e-11 / 2.2797e-14 relative
#   rAU      1.5132e-15 / 1.5133e-12 relative
#   phi      1.3878e-17 / 2.6343e-13 relative
#   Uf       2.2738e-13 / 2.5089e-13 relative
#   the 6 p_rgh solves and the 3 pcorr solves: EVERY iteration count identical (1/3/7 2/5/5 and 0/0/8),
#   and every initial and final residual identical to the bit -- 0.0e+00 relative difference on all nine
#   sum local continuity error 8.138623e-09 against OpenFOAM's 8.138623e-09 (3.0e-11 relative)
# The residuals are the sharper statement of the two: a field at 1e-13 could still be a solver that
# stopped somewhere else, and an iteration count cannot be.
#
# AND THE DEVICE ARM, in the same test and against the same oracle (unit 9). The change is HOST work on
# either loop, so what the device arm measures is the ROUND TRIP -- every mesh-sized buffer down to the
# host, the change, and every one of them back up onto a DeviceMesh rebuilt from scratch, which is what
# invalidates the schedule caches keyed on its addressingId. MEASURED: alpha 2.2204e-15 from OpenFOAM,
# p_rgh 6.0540e-15 relative, U 3.2213e-13, phi 3.2929e-13 -- and from the HOST arm, alpha 2.1176e-15,
# p_rgh 5.1892e-15, U 3.1454e-13. The arm is also run TWICE IN ONE PROCESS and the two runs are
# bit-identical, which is the detector for a schedule cached against a recycled pointer rather than
# against the addressing id.
#   FOUND BY THAT ARM, two defects nothing else could see:
#     * Uf.oldTime() was rotated inside `if (dyn)`, so an ADAPTIVE case kept the Uf the run started with
#       for the whole run. alpha stayed exact (2.2e-15) while p_rgh read 1.4536e-02, U 2.1642e-01 and phi
#       3.5678e-01 -- the pressure half alone, which is what pointed at ddtCorr.
#     * `f.amr` was built by the HOST driver rather than by the shared case build, so the device arm had a
#       null one and its branch asked a null pointer whether the case was adaptive. It reached `End:`
#       having refined NOTHING, on 32,256 cells, printing Courant 0.044 against the host's 0.088.
#
# TWO PROFILES. `open` is the tutorial as shipped. `closed` walls the atmosphere off -- U fixedValue,
# p_rgh fixedFluxPressure, alpha zeroGradient -- which makes the case's OWN pressure reference live:
# damBreakWithObstacle already writes `pRefPoint (0.51 0.51 0.51); pRefValue 0;` in its PIMPLE dict, inert
# only because totalPressure fixes a value. THE PRESSURE REFERENCE THROUGH A TOPOLOGY CHANGE (unit 10):
# OpenFOAM locates it ONCE, in createFields.H:104-113, and no solver re-locates or renumbers it -- pRefCell
# is a plain label that every pEqn indexes -- so after a refinement OpenFOAM pins the SAME INDEX, which is
# a different cell on the refined mesh. brae keeps it, and refuses only the one thing OpenFOAM cannot
# survive either: an index past the end after an unrefinement. MEASURED on the closed profile: all 9
# solves on OpenFOAM's iteration counts, alpha 1.6209e-14, p 1.9964e-14 relative, the reference cell 16400
# of 82264, and p pinned at pRefValue there in BOTH codes. THE CONTROL: pin a cell the change added --
# p 1.8350e-03, eleven orders above the gate's own distance. AND ONE VACUOUS ARM, measured: renumbering
# the reference through the change's own reverseCellMap is the IDENTITY under pure refinement, because
# hexRef8 modifies the parent in place and adds the other seven children (16400 -> 16400, 0.0e+00 apart).
#
# TWO CONTROLS ON THE MAPPING, one per half of the unit, each caught:
#   BRAE_CONTROL_AMR_NO_CORRECTPHI     the flux is left as the MAPPER wrote it -- no `phi = Sf & Uf` and
#                                      no pcorr solve. A split face inherits its parent's WHOLE flux
#                                      through a quarter of the area (Field::map copies, it does not
#                                      divide), so the mapped flux is not divergence-free and interFoam's
#                                      alpha equation has nothing to absorb that. MEASURED: alpha
#                                      2.8570e-01 out, U 3.2980e-01 relative, and max(alpha.water)
#                                      1.009594 where OpenFOAM stays at 1.
#   BRAE_CONTROL_AMR_RESIZE_NOT_MAP    the carried CELL fields are resized to the new count instead of
#                                      mapped -- the old values kept by index, the new cells at zero.
#                                      That is the defect class this unit exists to prevent. MEASURED:
#                                      alpha 3.6032e-02 out, U 6.0920e-02 relative.
#
# ...AND ONE ARM WHOSE ANSWER DEPENDS ON THE SCHEME, which is the sharpest thing in this gate.
#   BRAE_CONTROL_AMR_NO_OLDTIME_MAP    re-captures the old-time levels from the fields after the change
#                                      instead of mapping them.
#                                      UNDER EULER IT CHANGES NOTHING -- 0.0e+00 on alpha and on U -- and
#                                      the reason is structural: at the top of a step every old-time level
#                                      IS a copy of its own field (OpenFOAM's storeOldTimes runs on the
#                                      step's first access; brae's loop assigns them at the end of the
#                                      previous step), and the change happens before anything has solved.
#                                      UNDER CrankNicolson IT IS A CONTROL: the scheme reads the old-OLD
#                                      levels, which are two steps back and are NOT copies of anything
#                                      current. MEASURED on the cn profile: alpha 9.3122e-03 and U
#                                      2.5152e-01 from the gate's own arm. So the same switch is a
#                                      statement about the fixture on two profiles and a test on the third,
#                                      and the gate asserts which.
#
# WHAT THIS FIXTURE CANNOT WITNESS, and each is named rather than left to be assumed:
#   * THE FLUX CORRECTION. damBreakWithObstacle's dynamicMeshDict maps every flux to `none`
#     (`(phi none) (nHatf none) (rhoPhi none) (alphaPhi0.water none) (ghf none) (alphaPhiUn none)`), so
#     mapFields' own re-interpolation never runs. It is gated on an injected (braePhi braeU) pair in
#     tests/refine_update_vs_openfoam.sh, and a case that names a velocity is REFUSED by name here.
#   * UNREFINEMENT. The tutorial selects 0 split points at both steps -- and at every step of a 60-step
#     run of the same case at the tutorial's own resolution, measured. hexRef8's unrefinement path is
#     gated on its own in tests/hex_ref8_vs_openfoam.sh (unrefine, unrefine3, unrefineTwice) and the
#     driver's in tests/refine_update_vs_openfoam.sh, whose sphere moves so cells behind it coarsen.
#   * THE OLD-TIME LEVELS' MAPPING -- see the third control above, which measures that it cannot.
#   * THE CHOICE OF REFERENCE CELL under refinement: the reference point of this fixture is in the AIR,
#     far from the interface, so its own cell is never selected for refinement -- a control pinning
#     another child of its parent found no child. The arm that discriminates pins a cell the change added.
#   * A MOVING mesh beside refinement, turbulence, MRF, fvOptions, CrankNicolson, `correctPhi no` and a
#     CN restart directory: each is REFUSED by name (tests/interfoam_refusals.sh), so no arm here can
#     silently stand in for one.
#   * UNREFINEMENT ON THE DEVICE: the device arm runs the same two refining steps, so its unrefinement
#     path is exercised by nothing here either.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_inter_amr_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/laminar/damBreakWithObstacle"

# shellcheck disable=SC1091
. "$(dirname "$0")/require_fresh_binary.sh"
# shellcheck disable=SC1091
. "$(dirname "$0")/of_oracle_cache.sh"
command -v oracleKey > /dev/null \
    || { echo "FAIL: of_oracle_cache.sh did not define oracleKey -- the gate would run uncached"; exit 1; }

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: damBreakWithObstacle tutorial not found at $SRC"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
requireFresh "$BIN" || exit 1

W=${KEEP_W:-$(mktemp -d)}
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "$W"' EXIT

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
for app in blockMesh topoSet subsetMesh setFields interFoam; do
    command -v "$app" > /dev/null 2>&1 || { echo "SKIP: $app not on PATH"; exit 77; }
done

DT=0.001

# gate <profile> <nSteps> <staging edit run inside the case dir>
gate()
{
local profile="$1" N="$2"; shift 2
local END
END=$(python3 -c "print('%.10g' % ($N*float('$DT')))")
local C="$W/$profile"
rm -rf "$C"
cp -r "$SRC" "$C" || return 1
rm -rf "$C"/0 "$C"/processor* "$C"/log.*
cp -r "$C/0.orig" "$C/0"
( cd "$C" && eval "$@" ) || { echo "FAIL: the staging edit for $profile"; return 1; }

# The case's own controlDict, with only what an oracle needs changed: fixed steps, every step written,
# and `writePrecision 18` -- at the tutorial's own 6 the comparison floor is the FILE. MEASURED: p_rgh
# reaches 7.3e+03 on this case, so six significant digits resolve it to 5e-03, and every field came out
# exactly at that floor before this line went in.
python3 - "$C/system/controlDict" "$DT" "$END" <<'PYEOF'
import re, sys
path, dt, end = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path).read()
s = re.sub(r'\nfunctions\s*\{.*?\n\}\n', '\n', s, flags=re.S)
for key, val in [('startFrom', 'startTime'), ('startTime', '0'), ('stopAt', 'endTime'),
                 ('adjustTimeStep', 'no'), ('deltaT', dt), ('endTime', end),
                 ('writeControl', 'timeStep'), ('writeInterval', '1'), ('writeFormat', 'ascii'),
                 ('writePrecision', '18'), ('writeCompression', 'off'), ('timePrecision', '12')]:
    s, k = re.subn(r'^%s\s.*' % key, '%s %s;' % (key.ljust(15), val), s, flags=re.M)
    assert k == 1, key
open(path, 'w').write(s)
PYEOF
[ $? -eq 0 ] || { echo "FAIL: rewriting controlDict"; return 1; }

grep -q "dynamicFvMesh   dynamicRefineFvMesh" "$C/constant/dynamicMeshDict" \
    || { echo "FAIL: the tutorial no longer selects dynamicRefineFvMesh, so this gate tests nothing"; return 1; }

local key
key=$(oracleKey "$C" "interfoam_amr" "$profile" "$DT" "$N")
if oracleRestore "$C" "$key" "$END"; then
    echo "[$profile] OpenFOAM's $N steps of deltaT $DT to t = $END reused from the oracle cache"
else
    ( cd "$C" && blockMesh > log.blockMesh 2>&1 \
              && topoSet > log.topoSet 2>&1 \
              && subsetMesh -overwrite c0 -patch walls > log.subsetMesh 2>&1 \
              && setFields > log.setFields 2>&1 ) \
        || { echo "FAIL: preparing the case"; tail -20 "$C"/log.*; return 1; }
    grep -q "Subset 32256 of 32768 cells" "$C/log.subsetMesh" \
        || { echo "FAIL: damBreakWithObstacle is no longer 32256 cells, so the counts above are stale"; return 1; }
    [ ! -e "$C/constant/polyMesh/cellLevel" ] \
        || { echo "FAIL: the fresh mesh already carries a cellLevel, so it does not start from level 0"; return 1; }
    ( cd "$C" && interFoam > log.interFoam 2>&1 ) \
        || { echo "FAIL: interFoam"; tail -30 "$C/log.interFoam"; return 1; }
    [ -d "$C/$END" ] || { echo "FAIL: OpenFOAM wrote no $END directory"; ls "$C"; return 1; }
    oracleStore "$C" "$key"
fi

# THE FIXTURE MUST BE ABLE TO WITNESS, asserted on OpenFOAM's own log rather than assumed: the first two
# refinements at these counts, and a pcorr solve that does real work at the second one.
grep -q "Refined from 32256 to 42266 cells." "$C/log.interFoam" \
    || { echo "FAIL: OpenFOAM's first refinement is not 32256 -> 42266, so the measurements above are stale"
         grep -n "Refined from" "$C/log.interFoam"; return 1; }
grep -q "Refined from 42266 to 82264 cells." "$C/log.interFoam" \
    || { echo "FAIL: OpenFOAM's second refinement is not 42266 -> 82264"
         grep -n "Refined from" "$C/log.interFoam"; return 1; }
grep -q "Solving for pcorr, Initial residual = 1," "$C/log.interFoam" \
    || { echo "FAIL: OpenFOAM's pcorr solve does no work on this fixture, so the flux correction cannot"
         echo "      be witnessed and the control below would be vacuous"
         grep -n "pcorr" "$C/log.interFoam"; return 1; }
[ -f "$C/$END/polyMesh/owner" ] \
    || { echo "FAIL: OpenFOAM wrote no polyMesh into $END, so there is no mesh to compare on"; return 1; }
[ -f "$C/$END/Uf" ] \
    || { echo "FAIL: OpenFOAM wrote no Uf, so a refining mesh is not dynamic after all"; return 1; }

"$BIN" "$C" "$C/0" "$C/$END" "$N" "$C/log.interFoam" "$profile"
}

rc=0

# AS SHIPPED: an open tank, the atmosphere a totalPressure top, no pressure reference.
gate open 2 true || rc=1

# ...AND CLOSED, which is what makes the case's OWN pressure reference live. damBreakWithObstacle already
# writes `pRefPoint (0.51 0.51 0.51); pRefValue 0;` in its PIMPLE dict, and it is inert only because the
# atmosphere's totalPressure FIXES A VALUE, so p_rgh.needReference() is false (GeometricField.C:1068-1085).
# Walling the atmosphere off -- U fixedValue, p_rgh fixedFluxPressure, alpha zeroGradient, the proven
# `closedDamBreak` recipe of tests/interfoam_moving_vs_openfoam.sh -- makes it live with no fvSolution edit.
#
# THE POINT IS UNAMBIGUOUS, which matters: findCell takes the nearer of two centres when a point lies on a
# face, and that cost a day on sloshingTank2D. Here the mesh is a unit cube at 32x32x32, so faces sit at
# multiples of 0.03125 and 0.51 is strictly inside the cell spanning [0.5, 0.53125] in all three
# directions; the obstacle stops at z = 0.25, so the cell is in the retained subset.
gate closed 2 "python3 - <<'PYCLOSE'
import re
s = open('0/U').read()
s = s.replace('type            pressureInletOutletVelocity;', 'type            fixedValue;')
open('0/U', 'w').write(s)
s = open('0/p_rgh').read()
i = s.find('    atmosphere')
j = s.find('    }', i)
s = s[:i] + '    atmosphere\n    {\n        type            fixedFluxPressure;\n        phi             phiAbs;\n        value           uniform 0;\n' + s[j:]
open('0/p_rgh', 'w').write(s)
s = open('0/alpha.water').read()
s = re.sub(r'    atmosphere\n    \{[^}]*\}', '    atmosphere\n    {\n        type            zeroGradient;\n    }', s, count=1)
open('0/alpha.water', 'w').write(s)
PYCLOSE" || rc=1

# ...AND CrankNicolson, THREE STEPS. The scheme's ddt0 levels are registered FIELDS in OpenFOAM
# (CrankNicolsonDdtScheme's DDt0Field, REGISTER + AUTO_WRITE), so MapGeometricFields autoMaps each one's
# internal field and every patch field with no help from the scheme -- there is no topoChanging branch in
# CrankNicolsonDdtScheme.C at all. brae keeps them as locals of its time loop and hands them to the change
# (InterAmrCn). WHICH ONES EXIST here: ddt0(rho,U) and ddtCorrDdt0(U) (cell vectors, and the first carries
# no patch lists at all -- the fvmDdt path creates it without them) and ddtCorrDdt0(Uf) (a surface vector,
# because ddtCorr(U, phi, Uf) routes on mesh.dynamic() and a refining mesh IS dynamic). ddtCorrDdt0(phi) is
# never created and the port THROWS if it is; ddt(alpha) creates none; meshPhiCN_0 needs a mesh flux this
# mesh has not got. `nAlphaSubCycles 1` is MANDATORY: alphaEqn.H:126-133 makes OpenFOAM FatalError on
# CrankNicolson with sub-cycling, so without it there is no oracle.
#
# WHAT IT MEASURED: alpha 7.7716e-15, p_rgh 7.3209e-15 relative, U 4.4768e-13, p 2.6527e-14, phi
# 1.1772e-13, rAU 6.9447e-12, every one of the 9 solves on OpenFOAM's iteration count.
# THE CONTROL: BRAE_CONTROL_AMR_NO_CN_MAP drops the mapped CrankNicolson state and lets the scheme
# re-create it at the new size -- the plausible wrong port, where every field is the right SIZE, nothing
# throws and the run completes. MEASURED: alpha 1.0688e-02, U 3.2622e-01 relative. It discriminates even
# though ddt0(rho,U) itself is still zero at the only change that maps it, because the state it drops also
# holds Uf's old-old level, phi's, and the alpha flux's two blend levels, and none of those is zero there.
# THE DEVICE ARM RUNS IT TOO (unit 13): its ddt0 levels live in device buffers where the host keeps them in
# locals, so the adaptive round trip grew a CN half -- each level down component by component, through the
# SAME mapper, and back up. MEASURED: alpha 2.1982e-14 from OpenFOAM and 2.2714e-14 from the host arm,
# p_rgh 2.0e-14, U 8.8e-13, phi 4.3e-13. What it found: `if (!dyn) phiOldRequested = true;` where the host
# asks f.meshIsDynamic -- the EIGHTH site of that rule in this port -- which made the alpha equation's
# off-centred blend live ONE STEP EARLY on an adaptive case (max(alpha) 1.00008757 against 1.00000006 at
# t = 0.002, measured before any momentum or pressure of that step).
#
# AND WHAT THIS PROFILE CANNOT WITNESS, measured: the mapping of a NON-ZERO ddt0 level. A level created at
# step k is zero for the whole of step k (DDt0Field::evaluate is false on the step it is born), and this
# case refines at steps 1 and 2 and then NEVER again -- 0 cells selected at steps 3, 4, 5 and 6 -- so the
# last change maps a level that is still zero. laminar/damBreak with maxRefinement 3 changes at every step
# and would witness it; it is 2-D, and a 2-D adaptive case is refused for its `empty` patch (see the
# measurement in inter_amr_cpp.cu). That refusal is what this profile's coverage stops at.
gate cn 3 "python3 - <<'PYCN'
import re
s = open('system/fvSchemes').read()
s = re.sub(r'(ddtSchemes\s*\{[^}]*?default\s+)Euler;', r'\1CrankNicolson 0.5;', s, count=1, flags=re.S)
assert 'CrankNicolson' in s, 'the ddtSchemes default was not Euler'
open('system/fvSchemes', 'w').write(s)
s = open('system/fvSolution').read()
s, n = re.subn(r'nAlphaSubCycles\s+3;', 'nAlphaSubCycles 1;', s, count=1)
assert n == 1, 'nAlphaSubCycles 3 was not there to replace'
open('system/fvSolution', 'w').write(s)
PYCN" || rc=1
echo "interfoam_amr_vs_openfoam: rc $rc"
exit $rc
