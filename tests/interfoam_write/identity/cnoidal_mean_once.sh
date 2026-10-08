#!/usr/bin/env bash
# The write gate: the cnoidal wave's mean square elevation SUMMED ONCE and kept, against summing it at every
# face of every update (BRAE_CONTROL_CNOIDAL_MEAN_RECOMPUTED=1), on laminar/waves/cnoidal's W row -- on the
# device loop and on the host loop, which update the same model.
# cnoidalWaveModel's Uf asks etaMeanSq -- 1000 elevations over a period, each through the elliptic integrals and
# the Jacobi functions -- for every wet face of the paddle at every update of the model. It is a function of the
# wave's height, its m and its period alone. MEASURED 2026-10-05 (52,500 cells, device): the step 45.9 -> 14.8
# ms; 20-core OpenFOAM 37.6.
# The same sum of the same 1000 terms in the same order is the same number, so nothing written may change. The
# kept arms run with BRAE_CONTROL_CNOIDAL_MEAN_CHECK=1: the mean is summed at every call as well and the kept
# number held to it, bitwise (the number a run returns is the kept one with or without the check). The model
# says so at EVERY sum, so the kept arms' logs must hold that line exactly once -- one cnoidal patch, summed
# once. The CONTROL keeps the mean of 999 elevations (BRAE_CONTROL_CNOIDAL_MEAN_SHORT=1): the check must stop it.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase cnoidal of > "$W/cm_stage.txt" 2>&1
o="$W/w_of_cnoidal"
[ -d "$o" ] || { say "cnoidal did not stage" FAIL; finish "cnoidal mean once identity"; }
for v in recomputed kept hrecomputed hkept short; do
    e="$W/cm_$v"
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    case $v in
        recomputed)  runbrae "$e" device BRAE_CONTROL_CNOIDAL_MEAN_RECOMPUTED=1 ;;
        kept)        runbrae "$e" device BRAE_CONTROL_CNOIDAL_MEAN_CHECK=1 ;;
        hrecomputed) runbrae "$e" host BRAE_CONTROL_CNOIDAL_MEAN_RECOMPUTED=1 ;;
        hkept)       runbrae "$e" host BRAE_CONTROL_CNOIDAL_MEAN_CHECK=1 ;;
        # the control is expected to stop: run it directly, runbrae would end the gate
        short)       ( cd "$e" && BRAE_CONTROL_CNOIDAL_MEAN_CHECK=1 BRAE_CONTROL_CNOIDAL_MEAN_SHORT=1 \
                           "$BIN" -case . -device > log.brae 2>&1; echo $? > exit.txt ) ;;
    esac
done
# same <a> <b>: how many files of arm a are not arm b's, byte for byte; "-" unless both hold OpenFOAM's times
# and the same files at each
same()
{
    local a="$W/cm_$1" b="$W/cm_$2" n=0 d f
    [ "$(timedirs "$a")" = "$(timedirs "$o")" ] && [ "$(timedirs "$b")" = "$(timedirs "$o")" ] || { echo "-"; return; }
    for d in $(timedirs "$o"); do
        [ "$(filesets "$a" "$d")" = "$(filesets "$b" "$d")" ] || { echo "-"; return; }
        for f in $(cd "$a/$d" && find . -type f | sort); do
            cmp -s "$a/$d/$f" "$b/$d/$f" || n=$((n + 1))
        done
    done
    echo $n
}
mark="the wave's mean square elevation is summed once and kept"
cd_=$(grep -c "$mark" "$W/cm_kept/log.brae")
ch_=$(grep -c "$mark" "$W/cm_hkept/log.brae")
what="the mean is summed $cd_ time(s) in the device run and $ch_ in the host run; the other arms sum it at every face"
[ "$cd_" = 1 ] && [ "$ch_" = 1 ] && ! grep -q "$mark" "$W/cm_recomputed/log.brae" \
    && ! grep -q "$mark" "$W/cm_hrecomputed/log.brae" && say "$what" ok || say "$what" FAIL
t=$(for d in $(timedirs "$W/cm_recomputed"); do find "$W/cm_recomputed/$d" -type f; done | wc -l)
n1=$(same recomputed kept)
n2=$(same hrecomputed hkept)
what="[device] and [host] the mean kept, held to the sum at every call: $t files an arm; $n1 and $n2 differ"
[ "$t" -gt 0 ] && [ "$n1" = 0 ] && [ "$n2" = 0 ] && say "$what" ok || say "$what" FAIL
what="CONTROL  the mean of 999 elevations kept: the check stops the run and prints both numbers"
[ "$(cat "$W/cm_short/exit.txt")" != 0 ] \
    && grep -q "CNOIDAL_MEAN_CHECK: the kept mean square elevation is" "$W/cm_short/log.brae" \
    && say "$what" ok || say "$what" FAIL
finish "the cnoidal wave's mean square elevation summed once is the one summed at every face"
