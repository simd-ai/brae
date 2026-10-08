#!/usr/bin/env bash
# The write gate: the refinement's connectivity lists held COMPACT (CompactListList, compact_list_list.cuh: an
# array of values and one of offsets a list) against the lists of lists they replaced, on sixty steps of
# laminar/oscillatingBox -- a mesh that refines, moves and unrefines (refine/unrefine.sh's fixture).
# cells, pointCells, cellPoints, pointEdges, faceEdges, edgeFaces, cellEdges, cellCells and pointFaces were each
# a std::vector<std::vector<label>>: a heap block a row, a million on a 90,000-cell mesh, built after every
# change of topology. MEASURED 2026-10-04, ms a step: damBreakWithObstacle 401 -> 269, RAS/motorBike 251 -> 199;
# one build of the lists 96 -> 45 ms at ~85,000 cells.
# THE ORACLE is the lists of lists themselves, from the builders the tests hold to OpenFOAM's own dumps
# (meshCells, pointCellsFromCells, buildFaceEdges, ...): BRAE_CONTROL_REFINE_ADDRESSING_CHECK=1 builds them at
# every build and every reuse of the compact ones and compares each list row for row.
# THE CONTROL gives every cell its faces in ascending order (BRAE_CONTROL_REFINE_ADDRESSING_CELLS_SORTED=1) where
# meshCells' rows are the owned faces and then the neighboured ones -- the order a compact builder written from
# the description alone would give. The check must name `cells`; without the check the written files must
# change (measured: 837 of 1,620), which is what says the ORDER of a row is content here.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
OB="$TUT/multiphase/interFoam/laminar/oscillatingBox"
stage_allrun "$OB" "$W/rc_stage" 0.001 > "$W/rc_stage.txt" 2>&1 \
    || { say "oscillatingBox did not stage" FAIL; finish "refinement addressing compact identity"; }
sed -i -E 's/^(endTime\s+)[^;]*;/\10.06;/' "$W/rc_stage/system/controlDict"
for v in compact sorted sortedCheck; do
    e="$W/rc_$v"
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$W/rc_stage/0" "$W/rc_stage/constant" "$W/rc_stage/system" "$e/"
    case $v in
        compact)     runbrae "$e" device BRAE_CONTROL_REFINE_ADDRESSING_CHECK=1 ;;
        sorted)      runbrae "$e" device BRAE_CONTROL_REFINE_ADDRESSING_CELLS_SORTED=1 ;;
        # this control is expected to stop: run it directly, runbrae would end the gate
        sortedCheck) ( cd "$e" && BRAE_CONTROL_REFINE_ADDRESSING_CELLS_SORTED=1 \
                           BRAE_CONTROL_REFINE_ADDRESSING_CHECK=1 "$BIN" -case . -device > log.brae 2>&1
                       echo $? > exit.txt ) ;;
    esac
done
nref=$(grep -a "cells refined" "$W/rc_compact/log.brae" | grep -avc " 0 cells refined")
nunref=$(grep -a "cells refined" "$W/rc_compact/log.brae" | grep -avc " 0 split points unrefined")
nsteps=$(grep -ac "^ *t = .* dt = " "$W/rc_compact/log.brae")
what="[device] $nsteps steps, $nref refinements, $nunref unrefinements: every compact list is its list of lists"
[ "$nsteps" = 60 ] && [ "$nref" -gt 0 ] && [ "$nunref" -gt 0 ] && say "$what" ok || say "$what" FAIL
what="CONTROL  with each cell's faces sorted, the check stops the run and names the cells list"
[ "$(cat "$W/rc_sortedCheck/exit.txt")" != 0 ] \
    && grep -q "REFINE_ADDRESSING_CHECK: of the kept addressing, cells is not" "$W/rc_sortedCheck/log.brae" \
    && say "$what" ok || say "$what" FAIL
n=0
t=0
for d in $(timedirs "$W/rc_compact"); do
    for f in $(cd "$W/rc_compact/$d" && find . -type f | sort); do
        t=$((t + 1))
        cmp -s "$W/rc_compact/$d/$f" "$W/rc_sorted/$d/$f" || n=$((n + 1))
    done
done
what="CONTROL  ...and without the check the sorted rows change $n of $t written files: a row's order is content"
[ "$t" -gt 0 ] && [ "$n" -gt 0 ] && say "$what" ok || say "$what" FAIL
finish "the refinement's compact connectivity lists are the lists of lists they replaced"
