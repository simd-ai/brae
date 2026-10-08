#!/usr/bin/env bash
# The write gate: an unrefinement's INTERPOLATIVE face mapping held as three lists, against the list of lists a
# face it was held as (BRAE_CONTROL_FACE_MAPPING_NESTED=1), on sixty steps of laminar/oscillatingBox -- a mesh
# that refines, moves and unrefines (refine/unrefine.sh's fixture).
# An unrefinement merges four faces into one, which makes the face mapping interpolative; every OTHER face of
# the mesh then carries two heap lists of length one, built in faceMapping, copied in surfaceMapping and freed.
# MEASURED 2026-10-05, damBreakWithObstacle (290,168 rows of one source and 48 of four): an unrefinement's field
# mapping 57.0 -> 16.0 ms a call, the step 179.7 -> 168.0; RAS/motorBike 28.3 -> 8.0 a call, 166.1 -> 160.3.
# The rows, their order and their weights are the same, and so is the expression a face accumulates in, so
# nothing written may change. The flat arm runs with BRAE_CONTROL_FACE_MAPPING_CHECK=1: both mappings are built
# at every unrefinement, every carried surface field is mapped through each and compared bit for bit, and so
# are the patches' rows. The CONTROL weights a merged INTERNAL face's four sources one each
# (BRAE_CONTROL_FACE_MAPPING_UNWEIGHTED=1), so the face takes their sum: the check must stop the run on a field.
# (With the same control on the patches' rows as well, the check stopped on patch `walls` first -- measured
# once, 2026-10-05, which is what proves the rows' comparison; the control was then moved onto the fields.)
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
OB="$TUT/multiphase/interFoam/laminar/oscillatingBox"
stage_allrun "$OB" "$W/fm_stage" 0.001 > "$W/fm_stage.txt" 2>&1 \
    || { say "oscillatingBox did not stage" FAIL; finish "flat face mapping identity"; }
sed -i -E 's/^(endTime\s+)[^;]*;/\10.06;/' "$W/fm_stage/system/controlDict"
for v in nested flat unweighted; do
    e="$W/fm_$v"
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$W/fm_stage/0" "$W/fm_stage/constant" "$W/fm_stage/system" "$e/"
    case $v in
        nested)     runbrae "$e" device BRAE_CONTROL_FACE_MAPPING_NESTED=1 ;;
        flat)       runbrae "$e" device BRAE_CONTROL_FACE_MAPPING_CHECK=1 ;;
        # the control is expected to stop: run it directly, runbrae would end the gate
        unweighted) ( cd "$e" && BRAE_CONTROL_FACE_MAPPING_CHECK=1 BRAE_CONTROL_FACE_MAPPING_UNWEIGHTED=1 \
                          "$BIN" -case . -device > log.brae 2>&1
                      echo $? > exit.txt ) ;;
    esac
done
mark="an interpolative face mapping is held as three lists"
nunref=$(grep -a "cells refined" "$W/fm_flat/log.brae" | grep -avc " 0 split points unrefined")
what="the flat run takes the three lists through $nunref unrefinements, the other arm the lists of lists"
grep -q "$mark" "$W/fm_flat/log.brae" && ! grep -q "$mark" "$W/fm_nested/log.brae" \
    && [ "$nunref" -gt 0 ] && say "$what" ok || say "$what" FAIL
n=0
t=0
for d in $(timedirs "$W/fm_nested"); do
    for f in $(cd "$W/fm_nested/$d" && find . -type f | sort); do
        t=$((t + 1))
        cmp -s "$W/fm_nested/$d/$f" "$W/fm_flat/$d/$f" || n=$((n + 1))
    done
done
what="[device] sixty steps with the mapping flat and checked at every unrefinement: $t written files, $n differ"
[ "$t" -gt 0 ] && [ "$n" = 0 ] && [ "$(timedirs "$W/fm_flat")" = "$(timedirs "$W/fm_nested")" ] \
    && say "$what" ok || say "$what" FAIL
what="CONTROL  a merged internal face's sources weighted one each: the check stops the run and names the field"
[ "$(cat "$W/fm_unweighted/exit.txt")" != 0 ] \
    && grep -q "FACE_MAPPING_CHECK: a carried surface" "$W/fm_unweighted/log.brae" \
    && say "$what" ok || say "$what" FAIL
finish "an unrefinement's face mapping in three lists maps every field to the same bits"
