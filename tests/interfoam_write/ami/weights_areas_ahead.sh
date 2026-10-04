#!/usr/bin/env bash
# The AMI weights' overlap areas computed AHEAD of the advancing front by the host's threads, on
# RAS/mixerVesselAMI (two patches of 41,828 faces). The front is sequential and stays so -- the order a face's
# partners are recorded in is the order OpenFOAM sums its weights in -- but the area of a source face and a target
# face is a function of the two faces alone, so every candidate pair's is computed first, in parallel, by the
# same routine, and the front looks its areas up (a pair the candidates missed is computed when asked).
# MEASURED 2026-10-04 on the tutorial, ms a step: the AMI update 420.0 -> 292.2 (the front 352.8 -> 175.5, the
# areas ahead 46.6 on 16 threads and 250.5 on one), the step 1,791 -> 1,672.
# The check is in the code: BRAE_CONTROL_AMI_AREA_CHECK=1 computes the area at EVERY ask of the front as well
# and compares the bits -- a million asks an update. Two arms of it, the default thread count and three threads
# (ranges that divide the faces unevenly), and the CONTROL: the kept area looked up by the source face alone
# (BRAE_CONTROL_AMI_AREA_BY_SOURCE=1), which the check must stop at the first pair.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase mixerVesselAMI of > "$W/wa_stage.txt" 2>&1
o="$W/w_of_mixerVesselAMI"
[ -d "$o" ] || { say "mixerVesselAMI did not stage" FAIL; finish "AMI areas ahead"; }
# arm <name> [env...]: brae on a copy of the staged case; never ends the gate
arm()
{
    local e="$W/wa_$1"
    shift
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    ( cd "$e" && env "$@" "$BIN" -case . -device > log.brae 2>&1; echo $? > exit.txt )
}
arm check BRAE_CONTROL_AMI_AREA_CHECK=1
arm three BRAE_CONTROL_AMI_AREA_CHECK=1 BRAE_AMI_THREADS=3
arm bysource BRAE_CONTROL_AMI_AREA_CHECK=1 BRAE_CONTROL_AMI_AREA_BY_SOURCE=1
what="[device] every area the front looks up is the one computing it gives, the run to its end"
[ "$(cat "$W/wa_check/exit.txt")" = 0 ] && [ "$(timedirs "$W/wa_check")" = "$(timedirs "$o")" ] \
    && say "$what" ok || say "$what" FAIL
what="[device] ...and on three threads, whose ranges divide the faces unevenly"
[ "$(cat "$W/wa_three/exit.txt")" = 0 ] && [ "$(timedirs "$W/wa_three")" = "$(timedirs "$o")" ] \
    && say "$what" ok || say "$what" FAIL
what="CONTROL  the kept area looked up by the source face alone: the check stops the run and names the pair"
[ "$(cat "$W/wa_bysource/exit.txt")" != 0 ] \
    && grep -aq "AMI_AREA_CHECK: the kept area of source face" "$W/wa_bysource/log.brae" \
    && say "$what" ok || say "$what" FAIL
finish "the AMI weights' areas computed ahead are the ones the front computes"
