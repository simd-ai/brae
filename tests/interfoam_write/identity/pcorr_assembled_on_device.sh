#!/usr/bin/env bash
# The write gate: CorrectPhi's pcorr equation assembled on the GPU, against the host assembling it and uploading
# it, on brae's DEFAULT pressure path (under BRAE_PRESSURE_CASE_SOLVER, what lib.sh exports, pcorr is the host's).
# The host built fvm::laplacian(rAUf, pcorr) == fvc::div(phi) at every mesh update, folded its boundary, uploaded
# four arrays, and took the flux from phi after the solve. MEASURED 2026-10-04, ms a step for div(phi), the
# matrix, the fold, the uploads and the flux together: 22.1 -> 3.4 on waveMakerPiston at 896,000 cells (the step
# 1,011 -> 996), 17.8 -> 6.6 on the 845,536-cell moving hull. The GPU's pass is the host's arithmetic operation
# for operation -- a cell's faces in ascending face number, as the host's face loops reach it -- so the matrix,
# the source and phi are the host's bit for bit.
# Two cases, because they take different branches: waveMakerFlap (a patch that fixes pcorr, no reference cell,
# the geometry the GPU's own after each move) and sloshingTank2D (a closed tank: the reference cell, the
# geometry uploaded for the start's CorrectPhi). The GPU arm runs with BRAE_CONTROL_PCORR_ASSEMBLY_CHECK=1: the
# host assembles its system as well, and every entry of the off-diagonal, the folded diagonal and source and of
# phi after the flux is compared.
# BRAE_PCORR_AMG=plain in every arm: on a mesh that moves pcorr takes a smoothed-aggregation hierarchy by
# default, whose coarse matrices did not reproduce run to run when this was written (waveMakerFlap, 19 of 34
# files between two runs; summed in a fixed order since 2026-10-06, amg_pcg/smoothed_fixed_order.sh).
# TWO CONTROLS on one defect, a cell's owned faces summed before the ones it neighbours: the written files
# change, and under the check the run stops at the first entry.
. "$(dirname "$0")/../lib.sh"
unset BRAE_PRESSURE_CASE_SOLVER
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
for k in waveMakerFlap sloshingTank2D; do
    wcase $k of > "$W/pa_stage_$k.txt" 2>&1
    [ -d "$W/w_of_$k" ] || { say "$k did not stage" FAIL; finish "pcorr assembled on the device identity"; }
done
# arm <case> <name> [env...]: brae on a copy of the staged case; never ends the gate
arm()
{
    local o="$W/w_of_$1" e="$W/pa_$1_$2"
    shift 2
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    ( cd "$e" && env BRAE_PCORR_AMG=plain "$@" "$BIN" -case . -device > log.brae 2>&1; echo $? > exit.txt )
}
# differs <case> <a> <b>: the number of written files that are not byte-identical
differs()
{
    local n=0 t f
    for t in $(timedirs "$W/w_of_$1"); do
        for f in $(cd "$W/pa_$1_$2/$t" && find . -type f | sort); do
            cmp -s "$W/pa_$1_$2/$t/$f" "$W/pa_$1_$3/$t/$f" || n=$((n + 1))
        done
    done
    echo $n
}
mark="its equation is assembled on the GPU"
n=0
good=1
for k in waveMakerFlap sloshingTank2D; do
    arm $k host BRAE_CONTROL_PCORR_ASSEMBLY_HOST=1
    arm $k gpu BRAE_CONTROL_PCORR_ASSEMBLY_CHECK=1
    n=$((n + $(differs $k host gpu)))
    grep -aq "$mark" "$W/pa_${k}_gpu/log.brae" && ! grep -aq "$mark" "$W/pa_${k}_host/log.brae" \
        && [ "$(cat "$W/pa_${k}_gpu/exit.txt")" = 0 ] && [ "$(cat "$W/pa_${k}_host/exit.txt")" = 0 ] \
        && [ "$(timedirs "$W/pa_${k}_gpu")" = "$(timedirs "$W/w_of_$k")" ] || good=0
done
what="[device] pcorr assembled on the GPU, every entry checked against the host's, two cases: $n written files differ"
[ $good = 1 ] && [ "$n" = 0 ] && say "$what" ok || say "$what" FAIL
arm waveMakerFlap order BRAE_CONTROL_PCORR_ASSEMBLY_OWNER_FIRST=1
n=$(differs waveMakerFlap host order)
what="CONTROL  a cell's owned faces summed before the ones it neighbours: the written files change ($n of them)"
[ "$(cat "$W/pa_waveMakerFlap_order/exit.txt")" = 0 ] && [ "$n" != 0 ] && say "$what" ok || say "$what" FAIL
arm waveMakerFlap orderCheck BRAE_CONTROL_PCORR_ASSEMBLY_OWNER_FIRST=1 BRAE_CONTROL_PCORR_ASSEMBLY_CHECK=1
what="CONTROL  ...and under the check the run stops and names the folded diagonal"
[ "$(cat "$W/pa_waveMakerFlap_orderCheck/exit.txt")" != 0 ] \
    && grep -aq "PCORR_ASSEMBLY_CHECK: the folded diagonal, entry" "$W/pa_waveMakerFlap_orderCheck/log.brae" \
    && say "$what" ok || say "$what" FAIL
finish "pcorr's equation assembled on the GPU is the one the host assembles"
