#!/usr/bin/env bash
# The write gate: p_rgh's boundary hooks with an `empty` patch's entries written on the GPU, against every
# boundary face built on the host, on waveMakerFlap -- a 2-D mesh, where the two empty patches are a face a cell.
# The three hooks of the pressure corrector (the laplacian's patch coefficients, the stored patch values the
# corrected laplacian's gradient reads, the patches' evaluation after the solve) walked every boundary face on the
# host at every assembly. On waveMakerPiston at 896,000 cells 1,792,000 of the 1,796,320 are an empty patch's:
# zeros and copies of cell values. MEASURED 2026-10-04, ms a step in the three hooks: 41 -> 2.4, the step
# 1,053 -> 1,012. The empty faces' coefficients are written by a kernel (0 and -0, the host's own products),
# their stored values mirrored from the device's p_rgh; the host handles the other patches' faces only.
# The default arm runs with BRAE_CONTROL_PRESSURE_EMPTY_CHECK=1: the host builds every face's entry as well and
# compares bytes. TWO CONTROLS, because the two kinds of entry are caught differently: wrong coefficients change
# the written files; unmirrored values do not (a zero multiplies them) and only the check sees them.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase waveMakerFlap of > "$W/pe_stage.txt" 2>&1
o="$W/w_of_waveMakerFlap"
[ -d "$o" ] || { say "waveMakerFlap did not stage" FAIL; finish "pressure empty faces on the device identity"; }
for v in host gpu wrong nomirror; do
    e="$W/pe_$v"
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    case $v in
        host)     runbrae "$e" device BRAE_CONTROL_PRESSURE_EMPTY_FROM_HOST=1 ;;
        gpu)      runbrae "$e" device BRAE_CONTROL_PRESSURE_EMPTY_CHECK=1 ;;
        wrong)    runbrae "$e" device BRAE_CONTROL_PRESSURE_EMPTY_COEFFS_WRONG=1 ;;
        # this control is expected to stop: run it directly, runbrae would end the gate
        nomirror) ( cd "$e" && BRAE_CONTROL_PRESSURE_EMPTY_NO_MIRROR=1 BRAE_CONTROL_PRESSURE_EMPTY_CHECK=1 \
                        "$BIN" -case . -device > log.brae 2>&1; echo $? > exit.txt ) ;;
    esac
done
# differs <a> <b>: the number of written files that are not byte-identical
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
mark="an empty patch's entries are written on the GPU"
n=$(differs "$W/pe_host" "$W/pe_gpu")
what="[device] the empty faces' entries on the GPU, every entry checked against the host's: $n written files differ"
grep -q "$mark" "$W/pe_gpu/log.brae" && ! grep -q "$mark" "$W/pe_host/log.brae" \
    && [ "$n" = 0 ] && [ "$(timedirs "$W/pe_gpu")" = "$(timedirs "$o")" ] && say "$what" ok || say "$what" FAIL
n=$(differs "$W/pe_host" "$W/pe_wrong")
what="CONTROL  with the empty faces' coefficients set to 1 the written files change ($n of them)"
[ "$n" != 0 ] && say "$what" ok || say "$what" FAIL
what="CONTROL  with the empty faces' stored values not mirrored, the check stops the run and names p_rgh"
[ "$(cat "$W/pe_nomirror/exit.txt")" != 0 ] \
    && grep -q "PRESSURE_EMPTY_CHECK: boundary face .*stored p_rgh" "$W/pe_nomirror/log.brae" \
    && say "$what" ok || say "$what" FAIL
finish "p_rgh's boundary entries with the empty patches' on the GPU are the ones the host builds"
