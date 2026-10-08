#!/usr/bin/env bash
# The write gate: the refinement's addressing lists from their FASTER builders -- the edges by buckets, pointCells
# by inverting cellPoints, cellEdges by a mark an edge, faceEdges through each point's higher neighbours --
# against the builders they replace, on sixty steps of laminar/oscillatingBox (refines, moves and unrefines) and
# on RAS/motorBike's W row. Both are 3-D meshes whose points are UNORDERED, the edge list's one-block branch;
# its four-block branch (an ordered mesh -- every 2-D one) is held to OpenFOAM's own dump by
# mesh_edges_addressing_vs_openfoam, which builds through the same default.
# After every change of topology the ten lists are built for the whole mesh. MEASURED 2026-10-05, ms a build
# (damBreakWithObstacle, 92k cells | motorBike): the edges 13.5 -> 7.0 | 8.5 -> 3.9, pointCells 4.8 -> 0.9 |
# 2.6 -> 0.4, cellEdges 7.5 -> 4.4 | 4.8 -> 2.7, faceEdges 3.0 -> 2.5; the steps 153.0 -> 137.3 | 124.2 -> 116.0.
# Each list is the same list -- the edges' order is (low point, high point) inside each block, which is what
# OpenFOAM's last loop leaves -- so nothing written may change. The new arms run with
# BRAE_CONTROL_REFINE_ADDRESSING_CHECK=1 (every list against the lists of lists the tests hold to OpenFOAM's own
# dumps, row for row, at every use; its edges by OpenFOAM's chains) and BRAE_CONTROL_MESH_EDGES_CHECK=1.
# FOUR CONTROLS, one a builder, each deliberately wrong, and a check must stop each.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
OLD="BRAE_CONTROL_MESH_EDGES_CHAINS=1 BRAE_CONTROL_CELL_EDGES_SORT_UNIQUE=1"
OLD="$OLD BRAE_CONTROL_POINT_CELLS_TWO_PASSES=1 BRAE_CONTROL_FACE_EDGES_POINT_SCAN=1"
CHECKS="BRAE_CONTROL_REFINE_ADDRESSING_CHECK=1 BRAE_CONTROL_MESH_EDGES_CHECK=1"
# identical <src> <tag>: the old arm and the checked new arm of a staged case; echoes "<files> <differ>"
identical()
{
    local src="$1" tag="$2" v e n=0 t=0 d f
    for v in old new; do
        e="$W/${tag}_$v"
        rm -rf "${e:?}"
        mkdir -p "$e"
        cp -r "$src/0" "$src/constant" "$src/system" "$e/"
        case $v in
            old) runbrae "$e" device $OLD ;;
            new) runbrae "$e" device $CHECKS ;;
        esac
    done
    for d in $(timedirs "$W/${tag}_old"); do
        for f in $(cd "$W/${tag}_old/$d" && find . -type f | sort); do
            t=$((t + 1))
            cmp -s "$W/${tag}_old/$d/$f" "$W/${tag}_new/$d/$f" || n=$((n + 1))
        done
    done
    [ "$(timedirs "$W/${tag}_new")" = "$(timedirs "$W/${tag}_old")" ] || n=$((n + 1))
    echo "$t $n"
}
OB="$TUT/multiphase/interFoam/laminar/oscillatingBox"
stage_allrun "$OB" "$W/fb_stage" 0.001 > "$W/fb_stage.txt" 2>&1 \
    || { say "oscillatingBox did not stage" FAIL; finish "fast addressing builders identity"; }
sed -i -E 's/^(endTime\s+)[^;]*;/\10.06;/' "$W/fb_stage/system/controlDict"
read -r t n <<< "$(identical "$W/fb_stage" fbo)"
nref=$(grep -a "cells refined" "$W/fbo_new/log.brae" | grep -avc " 0 cells refined")
what="[device] oscillatingBox, $nref refinements, every list checked at every use: $t written files, $n differ"
[ "$t" -gt 0 ] && [ "$n" = 0 ] && [ "$nref" -gt 0 ] && say "$what" ok || say "$what" FAIL
wcase motorBike of > "$W/fb_stage_mb.txt" 2>&1
[ -d "$W/w_of_motorBike" ] || { say "motorBike did not stage" FAIL; finish "fast addressing builders identity"; }
read -r t n <<< "$(identical "$W/w_of_motorBike" fbm)"
nref=$(grep -a "cells refined" "$W/fbm_new/log.brae" | grep -avc " 0 cells refined")
what="[device] motorBike, $nref refinements, every list checked at every use: $t written files, $n differ"
[ "$t" -gt 0 ] && [ "$n" = 0 ] && [ "$nref" -gt 0 ] && say "$what" ok || say "$what" FAIL
# the controls are expected to stop: run directly, runbrae would end the gate
stopped=0
for c in MESH_EDGES_UNSORTED:"the edges from the buckets" CELL_EDGES_FIRST_FACE:"cellEdges is not" \
         POINT_CELLS_DESCENDING:"pointCells is not" FACE_EDGES_NEXT_ENTRY:"faceEdges is not"; do
    e="$W/fbc_${c%%:*}"
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$W/w_of_motorBike/0" "$W/w_of_motorBike/constant" "$W/w_of_motorBike/system" "$e/"
    ( cd "$e" && env $CHECKS "BRAE_CONTROL_${c%%:*}=1" "$BIN" -case . -device > log.brae 2>&1; echo $? > exit.txt )
    [ "$(cat "$e/exit.txt")" != 0 ] && grep -q "_CHECK: .*${c#*:}" "$e/log.brae" && stopped=$((stopped + 1))
done
what="CONTROLS the edges unsorted, a cell's shared edge twice, a point's cells descending, a face's edge"
what="$what off by one: $stopped of 4 stopped"
[ "$stopped" = 4 ] && say "$what" ok || say "$what" FAIL
finish "the addressing lists from their faster builders are the lists the old builders gave"
