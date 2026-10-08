#!/usr/bin/env bash
# The write gate: which wall-distance wave the motion solver's diffusivity runs, the GPU's or the host's, is
# decided by the front's mean width -- on waveMakerFlap, whose front is 140 cells.
. "$(dirname "$0")/../lib.sh"
# The two waves are one FaceCellWave to the bit (patch_wave_device.sh). What differs is the cost: the GPU wave
# pays for a sweep whatever the front holds, the host's for a cell. MEASURED 2026-10-06, the diffusivity's row in
# ms a step, GPU / host: waveMakerSolitary (14,250 cells, 38 a sweep) 7.8 / 0.6; waveMakerFlap (56,000, 140)
# 12.3 / 2.9; refined to 224,000 (280) 35.8 / 12.2 and to 896,000 (560) 123.7 / 53.3; the 3-D
# waveMakerMultiPaddleFlap (448,000, 2,240) 29.0 / 33.5. So the first call runs on the GPU and counts its sweeps,
# and a front under 2,000 cells takes the host wave from the second call (inter_driver_device.cu). The steps:
# waveMakerSolitary 23.1 -> 15.8 ms, waveMakerFlap 55.0 -> 45.3, waveMakerPiston 54.4 -> 44.9.
# ARMS, four pinned steps each so that the wave is called after the decision: the rule as shipped; the GPU wave
# kept (BRAE_PATCH_WAVE_MIN_FRONT=0); the host wave from the start (BRAE_CONTROL_PATCH_WAVE_HOST=1); the number
# set UNDER this front (100), the control of check 1 -- the rule must then say nothing and keep the GPU wave;
# and the GPU wave's first call made wrong (BRAE_CONTROL_PATCH_WAVE_OWNER_ONLY=1) with the rule on, the control
# of check 2 -- one wrong call out of the run's has to show.
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase waveMakerFlap of > "$W/fr_stage.txt" 2>&1
o="$W/w_of_waveMakerFlap"
[ -d "$o" ] || { say "waveMakerFlap did not stage" FAIL; finish "patch wave front rule"; }
dt=$(grep -m1 '^deltaT' "$o/system/controlDict" | tr -d ';' | awk '{print $2}')
t4=$(python3 -c "print('%.12g' % (4*float('$dt')))")
for v in rule gpu host under wrong; do
    e="$W/fr_$v"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    sed -i -E "s/^endTime .*/endTime         $t4;/; s/^writeInterval .*/writeInterval   4;/" "$e/system/controlDict"
    case $v in
        rule)  runbrae "$e" device BRAE_INTER_PHASE_TIME=1 ;;
        gpu)   runbrae "$e" device BRAE_INTER_PHASE_TIME=1 BRAE_PATCH_WAVE_MIN_FRONT=0 ;;
        host)  runbrae "$e" device BRAE_CONTROL_PATCH_WAVE_HOST=1 ;;
        under) runbrae "$e" device BRAE_INTER_PHASE_TIME=1 BRAE_PATCH_WAVE_MIN_FRONT=100 ;;
        # this control is expected to refuse or to write other numbers: run it directly
        wrong) ( cd "$e" && BRAE_CONTROL_PATCH_WAVE_OWNER_ONLY=1 "$BIN" -case . -device > log.brae 2>&1
                 echo $? > exit.txt ) ;;
    esac
done
# gpuCalls <dir>: the GPU wave's calls in the four steps, from the phase table's row (calls a step x 4)
gpuCalls()
{
    grep -a "patchWave: the wave (device)" "$1/log.brae" | sed -E 's/.*\(([0-9.]+) calls\/step\).*/\1/' \
        | awk '{printf "%.0f", $1*4} END {if (NR == 0) printf "0"}'
}
differs()
{
    local n=0 t f
    for t in $(timedirs "$1"); do
        for f in $(cd "$1/$t" && find . -type f | sort); do
            cmp -s "$1/$t/$f" "$2/$t/$f" || n=$((n + 1))
        done
    done
    echo $n
}
mark="wave runs on the host from its next call"
cr=$(gpuCalls "$W/fr_rule")
cg=$(gpuCalls "$W/fr_gpu")
cu=$(gpuCalls "$W/fr_under")
front=$(grep -a "$mark" "$W/fr_rule/log.brae" | sed -E 's/.*is a front of ([0-9]+),.*/\1/')
what="the rule: a front of ${front:-?} cells takes the host wave after 1 GPU call (GPU calls: rule $cr, kept $cg)"
[ "${front:-0}" -ge 100 ] && [ "${front:-0}" -lt 2000 ] && [ "$cr" = 1 ] && [ "$cg" -ge 4 ] \
    && ! grep -aq "$mark" "$W/fr_gpu/log.brae" \
    && say "$what" ok || say "$what" FAIL
what="CONTROL  the number set under the front (100): nothing said, the GPU wave kept ($cu calls)"
! grep -aq "$mark" "$W/fr_under/log.brae" && [ "$cu" = "$cg" ] && say "$what" ok || say "$what" FAIL
n=$(( $(differs "$W/fr_rule" "$W/fr_host") + $(differs "$W/fr_gpu" "$W/fr_host") ))
nt=$(for t in $(timedirs "$W/fr_host"); do find "$W/fr_host/$t" -type f; done | wc -l)
nw=0
if [ "$(cat "$W/fr_wrong/exit.txt")" = 0 ]; then
    nw=$(differs "$W/fr_wrong" "$W/fr_host")
else
    nw=refused
fi
what="[device] rule and GPU wave write the host wave's $nt files ($n differ); CONTROL one wrong GPU call: $nw"
[ "$n" = 0 ] && [ "$nt" -gt 0 ] && [ "$nw" != 0 ] && [ "$(timedirs "$W/fr_rule")" = "$t4 " ] \
    && say "$what" ok || say "$what" FAIL
finish "the wave the front's width picks is the host's wave, and says so"
