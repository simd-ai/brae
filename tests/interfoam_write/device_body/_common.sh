# Shared by the three files of this folder: A RIGID BODY ON THE DEVICE (RAS/DTCHullMoving; the tutorial row
# itself is tests/interfoam_write/tutorial.sh). Two things
# the device loop had to carry, each with a control here:
#   THE BODY'S LOAD reads the closure's nut and U's cells as they stand when the mesh is moved; on this arm
#   both are the device's, brought down right before the mesh update. Arm W CANNOT witness that: it writes
#   every step, and the write itself downloads the closure. So this arm writes at the SECOND step only.
#   MEASURED (2026-10-01) on the coarsened hull, 108,833 cells: rigidBodyMotionState 1.1e-15 and p_rgh
#   3.6e-13 against OpenFOAM; with BRAE_CONTROL_DEVICE_BODY_STALE=1 (the load from the stale host copies)
#   6.9e-09 and 7.4e-09. On the tutorial's own 848,022 cells the same four read 3.4e-16, 8.2e-13, 2.9e-09
#   and 2.9e-09 -- the fixture is coarsened because each of these files is one run, five minutes there.
#   THE ATMOSPHERE'S `tangentialVelocity` (pressureInletOutletVelocity's refValue, which the device kernel
#   now blends into the inflow value). With BRAE_CONTROL_DEVICE_PIOV_NOREF=1 (the refValue left off the
#   device, the inflow tangential part zero): U 2.4e-06, k 3.1e-04.
BOUND_X3_STATE=3e-15

# x3_case <variant>: OpenFOAM's DTCHullMoving (cached) and a copy of it for brae that writes at the SECOND
# step only. Leaves $x3t (the write time) and $X3 (brae's case). A machine without a GPU skips the folder.
x3_case()
{
    [ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
    wcase DTCHullMovingCoarse of
    [ -d "$W/w_of_DTCHullMovingCoarse" ] || { say "ARM X3 DTCHullMoving did not stage" FAIL; finish "arm X3"; }
    x3t=$(echo $(timedirs "$W/w_of_DTCHullMovingCoarse") | awk '{print $2}')
    X3="$W/x3_br_$1"
    mkdir -p "$X3"
    cp -r "$W/w_of_DTCHullMovingCoarse/0" "$W/w_of_DTCHullMovingCoarse/constant" "$W/w_of_DTCHullMovingCoarse/system" "$X3/"
    sed -i -E 's/^(writeInterval\s+)[^;]*;/\12;/' "$X3/system/controlDict"
    grep -qE '^writeInterval\s+2;' "$X3/system/controlDict" \
        || say "ARM X3 DTCHullMoving: the write interval was not staged to 2" FAIL
}

x3state()   # x3state <cmp file> -- the body state's gap at the one write, and p_rgh's
{
    python3 - "$1" "$x3t" "$BOUND_X3_STATE" <<'EOF_X3S'
import json, sys
r = json.loads([l for l in open(sys.argv[1]) if l.startswith('RESULT ')][-1][7:])
s = r['files'][sys.argv[2] + '/uniform/rigidBodyMotionState']['rel']
p = r['files'][sys.argv[2] + '/p_rgh']['rel']
print('      rigidBodyMotionState %.3e (bound %s), p_rgh %.3e' % (s, sys.argv[3], p))
sys.exit(0 if r['structure'] == 0 and s <= float(sys.argv[3]) else 1)
EOF_X3S
}
