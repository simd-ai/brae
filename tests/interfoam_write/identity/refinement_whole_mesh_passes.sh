#!/usr/bin/env bash
# The write gate: three whole-mesh passes of a change of topology done for the part of the mesh the change is
# about, against doing them for every face (their old forms), on sixty steps of laminar/oscillatingBox -- a mesh
# that refines seven times and unrefines once.
#   addMesh               the mesh into the change's lists WRITTEN whole, where it was appended a point, a cell
#                         and a face at a time (489,000 calls on damBreakWithObstacle)
#   removeFaces           the face-region walk started from the faces on an edge that goes, where it was started
#                         from every face of the mesh, each with a stack of its own
#   the hull average      a surface field's injected faces averaged from the field itself and a flux sent out
#                         and back in one pass, where every field was copied flat and every flux written to two
#                         whole-field arrays
# (and hexRef8's face level reads a face's vertices where they stand, which has no old form to run: its oracle
# is hex_ref8_vs_openfoam, OpenFOAM's own dump.)
# MEASURED 2026-10-05 on damBreakWithObstacle, ms a call: addMesh 3.9 -> 1.8, removeFaces' decisions 4.7 -> 1.0,
# the fluxes' hull average 4.7 -> 2.0 on a refinement and 3.7 -> 1.3 on an unrefinement, hexRef8's points and
# cells 5.4 -> 2.4; the step 136.7 -> 128.2.
# Each gives what it gave, so nothing written may change. The new arm runs with the three in-code checks on (the
# change's whole state against an entry at a time, the regions against a walk from every face, every field
# against the flat copy, bit for bit). THREE CONTROLS, one a pass, each deliberately wrong.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
OLD="BRAE_CONTROL_REMOVE_FACES_WALK_EVERY_FACE=1 BRAE_CONTROL_TOPO_ADD_MESH_ONE_BY_ONE=1"
OLD="$OLD BRAE_CONTROL_HULL_AVERAGE_COPIES=1"
CHECKS="BRAE_CONTROL_REMOVE_FACES_REGION_CHECK=1 BRAE_CONTROL_TOPO_ADD_MESH_CHECK=1"
CHECKS="$CHECKS BRAE_CONTROL_HULL_AVERAGE_CHECK=1"
OB="$TUT/multiphase/interFoam/laminar/oscillatingBox"
stage_allrun "$OB" "$W/wm_stage" 0.001 > "$W/wm_stage.txt" 2>&1 \
    || { say "oscillatingBox did not stage" FAIL; finish "refinement whole-mesh passes identity"; }
sed -i -E 's/^(endTime\s+)[^;]*;/\10.06;/' "$W/wm_stage/system/controlDict"
for v in old new; do
    e="$W/wm_$v"
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$W/wm_stage/0" "$W/wm_stage/constant" "$W/wm_stage/system" "$e/"
    case $v in
        old) runbrae "$e" device $OLD ;;
        new) runbrae "$e" device $CHECKS ;;
    esac
done
nref=$(grep -a "cells refined" "$W/wm_new/log.brae" | grep -avc " 0 cells refined")
nunref=$(grep -a "cells refined" "$W/wm_new/log.brae" | grep -avc " 0 split points unrefined")
n=0
t=0
for d in $(timedirs "$W/wm_old"); do
    for f in $(cd "$W/wm_old/$d" && find . -type f | sort); do
        t=$((t + 1))
        cmp -s "$W/wm_old/$d/$f" "$W/wm_new/$d/$f" || n=$((n + 1))
    done
done
what="[device] $nref refinements and $nunref unrefinements, the three checks on: $t written files, $n differ"
[ "$t" -gt 0 ] && [ "$n" = 0 ] && [ "$nref" -gt 0 ] && [ "$nunref" -gt 0 ] \
    && [ "$(timedirs "$W/wm_new")" = "$(timedirs "$W/wm_old")" ] && say "$what" ok || say "$what" FAIL
# the controls are expected to stop: run directly, runbrae would end the gate
stopped=0
for c in TOPO_ADD_MESH_NO_REGION:"the faces' patches written at once" \
         REMOVE_FACES_NO_WALK:"a walk from every face puts it in" \
         HULL_AVERAGE_NO_ROUND_TRIP:"a flux sent out and back in one pass"; do
    e="$W/wmc_${c%%:*}"
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$W/wm_stage/0" "$W/wm_stage/constant" "$W/wm_stage/system" "$e/"
    ( cd "$e" && env $CHECKS "BRAE_CONTROL_${c%%:*}=1" "$BIN" -case . -device > log.brae 2>&1; echo $? > exit.txt )
    [ "$(cat "$e/exit.txt")" != 0 ] && grep -q "_CHECK: .*${c#*:}" "$e/log.brae" && stopped=$((stopped + 1))
done
what="CONTROLS a boundary face with no patch, no region walk, a flux not sent out and back: $stopped of 3 stopped"
[ "$stopped" = 3 ] && say "$what" ok || say "$what" FAIL
finish "a change's whole-mesh passes done for the part the change is about give what they gave"
