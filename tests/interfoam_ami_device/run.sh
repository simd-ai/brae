# Sourced by the files of this folder. brae's interFoam DEVICE loop across a cyclicAMI pair against REAL
# OpenFOAM's, on RAS/mixerVesselAMI -- with the mesh held STATIC (`static`) and with the tutorial's own
# rotating zone (`rotating`).
#
# THE CASE: tests/interfoam_ami_vs_openfoam.sh's fixture -- the tutorial meshed as Allrun.pre does with the
# background block coarsened to (22 22 44), 82,510 cells and 8,872 faces a side of the pair -- with
# constant/dynamicMeshDict REMOVED in both codes, so the pair couples a mesh that does not move. Everything
# else is the tutorial's own: kEpsilon, a momentum predictor, two outer correctors, explicit MULES in two
# sub-cycles, `grad(U) cellLimited Gauss linear 1`, `limited corrected 0.33`.
# STAGED, in both codes, as that gate stages them: p_rgh and pcorr GAMG -> PCG with DIC, every solve pinned
# to 1e-13, fixed steps of 2e-4.
#
# MEASURED, five steps, device against OpenFOAM (2026-10-02): worst field file 8.9e-13 (rAU). The bound is
# one decade above it.
# CONTROLS, each one decision of the device loop put back to what it was, the same five steps:
#   BRAE_CONTROL_DEVICE_ALPHA_TOP_EVALUATE=1         8.2e-03  alpha's boundary evaluated at the top of the step
#   BRAE_CONTROL_DEVICE_PAIR_RHOPHI_LAST=1           6.5e-07  the pair's mass flux the last sub-cycle's
#   BRAE_CONTROL_DEVICE_PAIR_NO_NONORTH=1            1.9e-02  no non-orthogonal correction on the pair's faces
#   BRAE_CONTROL_DEVICE_PREDICTOR_NO_PAIR_FORCE=1    3.8e-04  the predictor's force without the pair's faces
#   BRAE_CONTROL_DEVICE_NONORTH_GRADU_UNLIMITED=1    8.4e-02  the momentum's correction on the unlimited grad(U)
#   BRAE_CONTROL_DEVICE_AMI_DDTCORR=1                8.3e-03  ddtCorr live across the cyclicAMI
#
# `rotating`: dynamicMeshDict KEPT -- solidBodyMotionSolver turns the zone at omega -5, every step moves the
# points, recomputes the AMI and runs CorrectPhi. MEASURED, ten steps, device against OpenFOAM: worst field
# file 8.8e-13 (rAU); twenty steps 9.9e-13. CONTROLS, the same ten steps:
#   BRAE_CONTROL_DEVICE_PAIR_STALE=1                 4.7e-02  the device keeps the pair it was built with
#   BRAE_CONTROL_DEVICE_PAIR_PHI_ABSOLUTE=1          1.2e-01  the pair's flux not made relative to the mesh
#   BRAE_CONTROL_DEVICE_PAIR_NO_NONORTH=1            3.8e-02  no non-orthogonal correction on the pair's faces
# NOT DISCRIMINATED on this fixture: the closure's moving-mesh terms, V0 in the ddt and the mesh flux in divU
# on the pair's faces as elsewhere (BRAE_CONTROL_DEVICE_KEPS_STATIC=1 reads 8.1e-13: a rigid rotation changes
# no volume) -- the host gate records the same.
# NOT CLAIMED: GAMG across the pair (staged), CrankNicolson across it (refused), a plain cyclic or a cyclicACMI
# on a moving mesh (refused), the tutorial's own 895k-cell mesh, and everything the host gate does not claim.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="${BUILD:-$ROOT/build}/brae_interFoam"
CMP="$ROOT/tools/foam_time_compare.py"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/RAS/mixerVesselAMI"
KIND=static
STEPS=5
DT=2e-4
BLOCK="22 22 44"
BOUND=9e-12
# a control has to land above this to count as seen: three decades over the bound
CONTROL_FLOOR=1e-8

# shellcheck disable=SC1091
. "$ROOT/tests/of_oracle_cache.sh"
command -v oracleRun > /dev/null \
    || { echo "FAIL: of_oracle_cache.sh did not define oracleRun -- the gate would run uncached"; exit 1; }
[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: mixerVesselAMI tutorial not found at $SRC"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
set +u
# shellcheck disable=SC1090
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
for t in blockMesh surfaceFeatureExtract snappyHexMesh createBaffles mergeOrSplitBaffles setFields interFoam; do
    command -v "$t" > /dev/null 2>&1 || { echo "SKIP: $t not on PATH"; exit 77; }
done

W=${KEEP_W:-$(mktemp -d)}
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "${W:?}"' EXIT
mkdir -p "$W"

# kind <static|rotating>: which fixture, and its step count
kind()
{
    KIND="$1"
    case "$KIND" in
        static)   STEPS=5 ;;
        rotating) STEPS=10 ;;
        *) echo "FAIL: unknown fixture $KIND"; exit 1 ;;
    esac
    END=$(python3 -c "
t = 0.0
for i in range($STEPS):
    t += float('$DT')
print('%.10g' % t)")
}

meshMixer()
{
    ( cd "$M" && blockMesh > log.blockMesh 2>&1 && surfaceFeatureExtract > log.surfaceFeatureExtract 2>&1 \
          && snappyHexMesh -overwrite > log.snappyHexMesh 2>&1 && createBaffles -overwrite > log.createBaffles 2>&1 \
          && mergeOrSplitBaffles -split -overwrite > log.mergeOrSplitBaffles 2>&1 && cp -r 0.orig 0 \
          && setFields > log.setFields 2>&1 )
}

# the staged case in $W/of with OpenFOAM's run, once per file (both cached)
stage()
{
    M="$W/mesh"
    rm -rf "${M:?}"
    cp -r "$SRC" "$M" || exit 1
    rm -rf "${M:?}"/[1-9]* "${M:?}"/0 "${M:?}"/processor* "${M:?}"/log.*
    grep -q "(50 50 100)" "$M/system/blockMeshDict" || { echo "FAIL: the tutorial's block is no longer (50 50 100)"; exit 1; }
    sed -i "s/(50 50 100)/($BLOCK)/" "$M/system/blockMeshDict"
    cp -rf "$TUT/resources/geometry/mixerVesselAMI" "$M/constant/triSurface" || exit 1
    oracleMesh "$M" interfoam_ami_device meshMixer \
        "$(cat "$TUT"/resources/geometry/mixerVesselAMI/* | sha256sum | cut -c1-16)" \
        || { echo "FAIL: meshing"; exit 1; }
    C="$W/of"
    rm -rf "${C:?}"
    cp -r "$M" "$C" || exit 1
    if [ "$KIND" = static ]; then
        rm -f "$C/constant/dynamicMeshDict"
    else
        grep -q "solidBodyMotionSolver\|solidBody" "$C/constant/dynamicMeshDict" \
            || { echo "FAIL: the tutorial's dynamicMeshDict no longer names a solid-body motion"; exit 1; }
    fi
    N="$STEPS" DT="$DT" python3 "$ROOT/tests/interfoam_ami_device/stage.py" "$C" || { echo "FAIL: staging"; exit 1; }
    oracleRun "$C" interfoam_ami_device "$KIND" "$STEPS" "$DT" \
        || { echo "FAIL: interFoam"; tail -30 "$C/log.interFoam"; exit 1; }
    [ -d "$C/$END" ] || { echo "FAIL: OpenFOAM wrote no $END directory"; ls "$C"; exit 1; }
    grep -q "AMI1" "$C/constant/polyMesh/boundary" || { echo "FAIL: the staged mesh has no AMI1 patch"; exit 1; }
}

# device <dir> [ENV=1]: brae's device loop on the staged case. Prints `<structure> <worst> <file> <files>`
device()
{
    local d="$W/$1"
    shift
    rm -rf "${d:?}"
    mkdir -p "$d"
    cp -r "$W/of/0" "$W/of/constant" "$W/of/system" "$d/"
    ( cd "$d" && env "$@" "$BIN" -case . -device > log.brae 2>&1 ) \
        || { echo "FAIL: brae -device did not run"; tail -5 "$d/log.brae" | cut -c1-600; return 1; }
    python3 "$CMP" "$W/of" "$d" "$END" > "$d/cmp.txt" 2>&1
    python3 "$ROOT/tests/interfoam_ami_device/worst.py" "$d/cmp.txt"
}

# ami_gate <static|rotating>: the device loop as shipped, at the bound
ami_gate()
{
    kind "$1"
    stage
    local r
    r=$(device dev) || { echo "$r"; echo "interfoam_ami_device: rc 1"; exit 1; }
    # shellcheck disable=SC2086
    set -- $r
    local rc=0
    grep -q "\[device\]" "$W/dev/log.brae" || { echo "  FAIL: the run did not take the device loop"; rc=1; }
    [ "$1" = 0 ] || { echo "  FAIL: the written files differ in structure"; grep BAD "$W/dev/cmp.txt" | head -4; rc=1; }
    [ "$4" -ge 10 ] || { echo "  FAIL: only $4 field files were compared"; rc=1; }
    if python3 -c "import sys; sys.exit(0 if float('$2') <= $BOUND else 1)"; then
        echo "  ok:   device against OpenFOAM, $STEPS steps across the $KIND pair, $4 files: worst $2 ($3), bound $BOUND"
    else
        echo "  FAIL: device against OpenFOAM, worst $2 ($3) over the bound $BOUND"
        rc=1
    fi
    echo "interfoam_ami_device: rc $rc"
    exit $rc
}

# ami_control <static|rotating> <ENV=1> ...: each control alone, asserted to fail on a number
ami_control()
{
    kind "$1"
    shift
    stage
    local rc=0 ctl r
    for ctl in "$@"; do
        r=$(device ctl "$ctl") || { echo "$r"; rc=1; continue; }
        # shellcheck disable=SC2086
        set -- $r
        if python3 -c "import sys; sys.exit(0 if float('$2') > $CONTROL_FLOOR else 1)"; then
            echo "  ok:   control $ctl fails on a number: worst $2 ($3), floor $CONTROL_FLOOR"
        else
            echo "  FAIL: control $ctl passed the gate ($2) -- the gate cannot see what it breaks"
            rc=1
        fi
    done
    echo "interfoam_ami_device: rc $rc"
    exit $rc
}
