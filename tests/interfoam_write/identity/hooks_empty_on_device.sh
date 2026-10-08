#!/usr/bin/env bash
# The write gate: the alpha and forces hooks with an `empty` patch's entries written ON THE GPU, against every
# boundary face built on the host, on laminar/waves/streamFunction -- 2-D, 320,000 of its 324,160 boundary
# faces an empty patch's, a wave alpha patch (so the modelled-boundary hook runs) and three alpha sub-cycles.
# At every call the hooks evaluated alpha on those faces, built the mixture's boundary, a zero normal, a zero
# snGrad(rho) and a zero surface-tension flux there on the host, and uploaded all of it. Now the host works on
# the other patches, uploads them run by run, and a kernel writes an empty face's entry with what the host's
# own arithmetic gave: alpha's and nu's the cell's, the zeros, and (sigma*K) times a zero for the flux.
# MEASURED 2026-10-06 on this case, ms a step: alpha.updateBoundary 3.2 -> 0.6, updateModelledBoundary
# 1.2 -> 0.3, storedBoundary 0.3 -> 0.1, interfaceForces 3.3 -> 0.7; the step 34.2 -> 27.7.
# THE ORACLE is brae's own host build (BRAE_CONTROL_HOOKS_EMPTY_FROM_HOST=1): the written files byte for byte
# AND BRAE_CONTROL_HOOKS_EMPTY_CHECK=1, under which the host builds every array whole as well and compares
# bytes -- the files alone cannot see most of these entries, a zero multiplies them. The run with no switch
# is held too. TWO CONTROLS: a wrong snGrad(rho) on the empty faces changes the written files; a wrong alpha
# does not have to, and the check stops on it and names the array.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase streamFunction of > "$W/he_stage.txt" 2>&1
o="$W/w_of_streamFunction"
[ -d "$o" ] || { say "streamFunction did not stage" FAIL; finish "hooks' empty faces on the device identity"; }
for v in host check plain wrong stop; do
    e="$W/he_$v"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    case $v in
        host)  runbrae "$e" device BRAE_CONTROL_HOOKS_EMPTY_FROM_HOST=1 ;;
        check) runbrae "$e" device BRAE_CONTROL_HOOKS_EMPTY_CHECK=1 ;;
        plain) runbrae "$e" device ;;
        wrong) runbrae "$e" device BRAE_CONTROL_HOOKS_EMPTY_WRONG=snGradRho ;;
        # this control is expected to stop: run it directly, runbrae would end the gate
        stop)  ( cd "$e" && BRAE_CONTROL_HOOKS_EMPTY_WRONG=alpha BRAE_CONTROL_HOOKS_EMPTY_CHECK=1 \
                     "$BIN" -case . -device > log.brae 2>&1; echo $? > exit.txt ) ;;
    esac
done
differs()
{
    local n=0 t f
    for t in $(timedirs "$o"); do
        for f in $(cd "$1/$t" && find . -type f | sort); do
            cmp -s "$1/$t/$f" "$2/$t/$f" || n=$((n + 1))
        done
    done
    echo $n
}
mark="an empty patch's entries of alpha, the mixture, the normal"
arrays=$(grep -a "hooks empty check:" "$W/he_check/log.brae" | sed -E 's/.*check: (.*), [0-9]+ boundary.*/\1/' \
    | sort -u | wc -l)
n=$(differs "$W/he_host" "$W/he_check")
m=$(differs "$W/he_host" "$W/he_plain")
what="[device] $arrays arrays the host's whole build at every call, bitwise; files differ: checked $n, plain $m"
grep -q "$mark" "$W/he_check/log.brae" && grep -q "$mark" "$W/he_plain/log.brae" \
    && ! grep -q "$mark" "$W/he_host/log.brae" && [ "$arrays" -ge 6 ] && [ "$n" = 0 ] && [ "$m" = 0 ] \
    && [ "$(timedirs "$W/he_plain")" = "$(timedirs "$o")" ] && say "$what" ok || say "$what" FAIL
n=$(differs "$W/he_host" "$W/he_wrong")
what="CONTROL  with snGrad(rho) written as 1 on the empty faces the written files change ($n of them)"
[ "$n" != 0 ] && say "$what" ok || say "$what" FAIL
what="CONTROL  with alpha written as 1 on the empty faces the check stops the run and names alpha's values"
[ "$(cat "$W/he_stop/exit.txt")" != 0 ] \
    && grep -q "HOOKS_EMPTY_CHECK: alpha's patch values, evaluated, boundary face" "$W/he_stop/log.brae" \
    && say "$what" ok || say "$what" FAIL
finish "the hooks' boundary arrays with the empty patches' entries on the GPU are the ones the host builds"
