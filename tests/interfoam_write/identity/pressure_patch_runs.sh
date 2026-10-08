#!/usr/bin/env bash
# The write gate: p_rgh's boundary hooks copying the host's patches between the GPU's boundary arrays and the
# host a RUN at a time, against a patch at a time (BRAE_CONTROL_PRESSURE_PATCH_COPIES=1), on RAS/motorBike's W
# row (61 patches the host handles) and on waveMakerFlap's (a 2-D mesh, whose `empty` faces are the GPU's own
# and end a run).
# pressureCoeffs brought each patch's phiHbyA and rAUf down and sent its two laplacian coefficients up on their
# own, and boundaryValues sent each patch's stored values up: a synchronous copy each, whose cost does not
# depend on its length. MEASURED 2026-10-05 on RAS/motorBike (4,649 boundary faces, six assemblies a step):
# 1,860 copies a step; pressureCoeffs 8.3 -> 0.5 ms a step, boundaryValues 1.6 -> 0.1, the step 143.9 -> 134.8.
# Patches that follow one another in the device's numbering are one range of it, copied once and sliced on the
# host: the same numbers in the same places, so nothing written may change. The run arms go with
# BRAE_CONTROL_PRESSURE_EMPTY_CHECK=1 (every boundary face's coefficients and stored value on the device against
# what the host builds, bitwise, at every assembly). The CONTROL sends a run's coefficients up one face late
# (BRAE_CONTROL_PRESSURE_RUN_SHIFTED=1): the check must stop the run.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
mark="patches are copied as"
# identical <key> <tag>: the patch arm and the checked run arm of one W row; echoes "<files> <differ> <notice>"
identical()
{
    local key="$1" tag="$2" o="$W/w_of_$1" v e n=0 t=0 d f
    for v in patch run; do
        e="$W/${tag}_$v"
        rm -rf "${e:?}"
        mkdir -p "$e"
        cp -r "$o/0" "$o/constant" "$o/system" "$e/"
        case $v in
            patch) runbrae "$e" device BRAE_CONTROL_PRESSURE_PATCH_COPIES=1 ;;
            run)   runbrae "$e" device BRAE_CONTROL_PRESSURE_EMPTY_CHECK=1 ;;
        esac
    done
    for d in $(timedirs "$o"); do
        for f in $(cd "$W/${tag}_patch/$d" && find . -type f | sort); do
            t=$((t + 1))
            cmp -s "$W/${tag}_patch/$d/$f" "$W/${tag}_run/$d/$f" || n=$((n + 1))
        done
    done
    local took=no
    grep -q "$mark" "$W/${tag}_run/log.brae" && ! grep -q "$mark" "$W/${tag}_patch/log.brae" \
        && [ "$(timedirs "$W/${tag}_run")" = "$(timedirs "$o")" ] && took=yes
    echo "$t $n $took"
}
# runsof <tag>: "<n> patches in <m>", read off the run arm's own notice
runsof()
{
    grep -a "$mark" "$W/$1_run/log.brae" \
        | sed -E 's/.*host.s ([0-9]+) patches are copied as ([0-9]+) run.*/\1 patches in \2/'
}
wcase motorBike of > "$W/pr_stage_mb.txt" 2>&1
[ -d "$W/w_of_motorBike" ] || { say "motorBike did not stage" FAIL; finish "pressure patch runs identity"; }
read -r t n took <<< "$(identical motorBike prm)"
runs=$(runsof prm)
what="[device] motorBike, $runs run(s), every face checked at every assembly: $t written files, $n differ"
[ "$took" = yes ] && [ "$t" -gt 0 ] && [ "$n" = 0 ] && say "$what" ok || say "$what" FAIL
wcase waveMakerFlap of > "$W/pr_stage_wf.txt" 2>&1
[ -d "$W/w_of_waveMakerFlap" ] || { say "waveMakerFlap did not stage" FAIL; finish "pressure patch runs identity"; }
read -r t n took <<< "$(identical waveMakerFlap prw)"
runs=$(runsof prw)
what="[device] waveMakerFlap, $runs run(s) beside the GPU's empty faces: $t written files, $n differ"
[ "$took" = yes ] && [ "$t" -gt 0 ] && [ "$n" = 0 ] && say "$what" ok || say "$what" FAIL
# the control is expected to stop: run it directly, runbrae would end the gate
e="$W/prm_shifted"
rm -rf "${e:?}"
mkdir -p "$e"
cp -r "$W/w_of_motorBike/0" "$W/w_of_motorBike/constant" "$W/w_of_motorBike/system" "$e/"
( cd "$e" && BRAE_CONTROL_PRESSURE_EMPTY_CHECK=1 BRAE_CONTROL_PRESSURE_RUN_SHIFTED=1 \
      "$BIN" -case . -device > log.brae 2>&1; echo $? > exit.txt )
what="CONTROL  a run's coefficients sent up one face late: the check stops the run and names the face"
[ "$(cat "$e/exit.txt")" != 0 ] \
    && grep -q "PRESSURE_EMPTY_CHECK: boundary face .* laplacian coefficients" "$e/log.brae" \
    && say "$what" ok || say "$what" FAIL
finish "p_rgh's boundary hooks copy the host's patches a run at a time and write what they wrote a patch at a time"
