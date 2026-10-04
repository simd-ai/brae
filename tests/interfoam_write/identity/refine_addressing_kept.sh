#!/usr/bin/env bash
# The write gate: the refinement's mesh addressing KEPT from one step to the next, against building it at every
# step, on sixty steps of laminar/oscillatingBox -- a mesh that refines, MOVES and unrefines (refine/unrefine.sh's
# fixture), so the kept lists are reused after each kind of change and their geometry is built again every step.
# refineUpdate reads ten lists of the mesh (cells, points, edges and their inverses, the geometry). It built them
# at the start of every step for the mesh its last change had just built them for and freed. Kept with the mesh
# they are of, compared by content (StepAddressingKept, dynamic_refine_fv_mesh_cpp.cu).
# MEASURED 2026-10-04, ms a step: damBreakWithObstacle 517 -> 428, RAS/motorBike 318 -> 264.
# The default arm runs with BRAE_CONTROL_REFINE_ADDRESSING_CHECK=1: at every reuse the lists are built as well
# and compared byte for byte. The CONTROL leaves the kept geometry as it was when the points moved
# (BRAE_CONTROL_REFINE_ADDRESSING_STALE_GEOMETRY=1): the check must stop the run and name the geometry.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
OB="$TUT/multiphase/interFoam/laminar/oscillatingBox"
stage_allrun "$OB" "$W/ra_stage" 0.001 > "$W/ra_stage.txt" 2>&1 \
    || { say "oscillatingBox did not stage" FAIL; finish "refinement addressing kept identity"; }
sed -i -E 's/^(endTime\s+)[^;]*;/\10.06;/' "$W/ra_stage/system/controlDict"
for v in rebuild kept stale; do
    e="$W/ra_$v"
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$W/ra_stage/0" "$W/ra_stage/constant" "$W/ra_stage/system" "$e/"
    case $v in
        rebuild) runbrae "$e" device BRAE_CONTROL_REFINE_ADDRESSING_REBUILD=1 ;;
        kept)    runbrae "$e" device BRAE_CONTROL_REFINE_ADDRESSING_CHECK=1 ;;
        # the control is expected to stop: run it directly, runbrae would end the gate
        stale)   ( cd "$e" && BRAE_CONTROL_REFINE_ADDRESSING_STALE_GEOMETRY=1 BRAE_CONTROL_REFINE_ADDRESSING_CHECK=1 \
                       "$BIN" -case . -device > log.brae 2>&1; echo $? > exit.txt ) ;;
    esac
done
mark="the mesh's addressing is kept from the last change"
nref=$(grep -a "cells refined" "$W/ra_kept/log.brae" | grep -avc " 0 cells refined")
nunref=$(grep -a "cells refined" "$W/ra_kept/log.brae" | grep -avc " 0 split points unrefined")
what="the kept run takes the kept path through $nref refinements and $nunref unrefinements, the other arm builds"
grep -q "$mark" "$W/ra_kept/log.brae" && ! grep -q "$mark" "$W/ra_rebuild/log.brae" \
    && [ "$nref" -gt 0 ] && [ "$nunref" -gt 0 ] && say "$what" ok || say "$what" FAIL
n=0
t=0
for d in $(timedirs "$W/ra_rebuild"); do
    for f in $(cd "$W/ra_rebuild/$d" && find . -type f | sort); do
        t=$((t + 1))
        cmp -s "$W/ra_rebuild/$d/$f" "$W/ra_kept/$d/$f" || n=$((n + 1))
    done
done
what="[device] sixty steps with the addressing kept and checked at every reuse: $t written files, $n differ"
[ "$t" -gt 0 ] && [ "$n" = 0 ] && [ "$(timedirs "$W/ra_kept")" = "$(timedirs "$W/ra_rebuild")" ] \
    && say "$what" ok || say "$what" FAIL
what="CONTROL  with the kept geometry left as it was when the points moved, the check stops the run and names it"
[ "$(cat "$W/ra_stale/exit.txt")" != 0 ] \
    && grep -q "REFINE_ADDRESSING_CHECK: of the kept addressing, the geometry" "$W/ra_stale/log.brae" \
    && say "$what" ok || say "$what" FAIL
finish "the refinement's addressing kept between steps is the one built at every step"
