#!/usr/bin/env bash
# The AMI weights' advancing front without its two repeated costs, on RAS/mixerVesselAMI (two patches of 41,828
# faces): the lowest unmapped source face that has a seed is the top of a queue of seeded faces, where the front
# walked the unmapped faces upward to find it at every stall; and each patch's face neighbours are kept from one
# update to the next, where they were rebuilt though a rotating AMI keeps its faces.
# MEASURED 2026-10-04 on the tutorial, ms a step: the walk for a seeded face 116.3 -> 1.8, the face neighbours
# 29.1 -> 1.4, the AMI update 292 -> 157, the step 1,668 -> 1,549.
# Neither changes a weight or the order the weights are recorded in. The checks are in the code and run at every
# stall and every update: BRAE_CONTROL_AMI_SEED_CHECK=1 walks as before as well and compares the face,
# BRAE_CONTROL_AMI_TOPOLOGY_CHECK=1 rebuilds the kept neighbours and compares every list. TWO CONTROLS: the queue
# with the mapped faces left on top (BRAE_CONTROL_AMI_SEED_STALE=1), and each side handed the other side's kept
# neighbours and taking them unasked (BRAE_CONTROL_AMI_TOPOLOGY_STALE=1).
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase mixerVesselAMI of > "$W/wb_stage.txt" 2>&1
o="$W/w_of_mixerVesselAMI"
[ -d "$o" ] || { say "mixerVesselAMI did not stage" FAIL; finish "AMI walk bookkeeping"; }
# arm <name> [env...]: brae on a copy of the staged case, both checks on; never ends the gate
arm()
{
    local e="$W/wb_$1"
    shift
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    ( cd "$e" && env BRAE_CONTROL_AMI_SEED_CHECK=1 BRAE_CONTROL_AMI_TOPOLOGY_CHECK=1 "$@" \
          "$BIN" -case . -device > log.brae 2>&1; echo $? > exit.txt )
}
arm checks BRAE_X=1
arm seed BRAE_CONTROL_AMI_SEED_STALE=1
arm topology BRAE_CONTROL_AMI_TOPOLOGY_STALE=1
what="[device] at every stall the queue's face is the walk's, at every update the kept neighbours the rebuilt ones"
[ "$(cat "$W/wb_checks/exit.txt")" = 0 ] && [ "$(timedirs "$W/wb_checks")" = "$(timedirs "$o")" ] \
    && say "$what" ok || say "$what" FAIL
what="CONTROL  mapped faces left on the queue: the check stops the run and names the two faces"
[ "$(cat "$W/wb_seed/exit.txt")" != 0 ] \
    && grep -aq "AMI_SEED_CHECK: the lowest unmapped face with a seed is" "$W/wb_seed/log.brae" \
    && say "$what" ok || say "$what" FAIL
what="CONTROL  a side handed the other side's kept neighbours: the check stops the run and names the face"
[ "$(cat "$W/wb_topology/exit.txt")" != 0 ] \
    && grep -aq "AMI_TOPOLOGY_CHECK: the kept neighbours of face" "$W/wb_topology/log.brae" \
    && say "$what" ok || say "$what" FAIL
finish "the AMI front's seeded-face queue and kept face neighbours are the walk's and the rebuilt ones"
