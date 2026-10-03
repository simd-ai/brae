#!/usr/bin/env bash
# The write gate: setRDeltaT's smoothing wave on the device against the host wave, on DTCHull.
. "$(dirname "$0")/../lib.sh"
# Since 2026-10-03 the device loop runs fvc::smooth's FaceCellWave on the GPU (device_fvc_smooth.cu), each cell
# and face folding its own visits in the host's order -- 430 ms a step to 50 on RAS/DTCHull, the only
# interFoam tutorial under localEuler. MEASURED, GPU wave against host wave, every file written at iteration 25
# BYTE-IDENTICAL. The control folds each cell's visits in the reverse order: the wave's answer depends on its
# order (propagationTol), and this shows the gate sees it.
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase DTCHull of > "$W/wave_stage.txt" 2>&1
o="$W/w_of_DTCHull"
[ -d "$o" ] || { say "DTCHull did not stage" FAIL; finish "smoothing wave identity"; }
for v in device host reversed; do
    e="$W/wave_$v"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    case $v in
        device)   runbrae "$e" device ;;
        host)     runbrae "$e" device BRAE_CONTROL_WAVE_HOST=1 ;;
        reversed) runbrae "$e" device BRAE_CONTROL_WAVE_REVERSED=1 ;;
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
n=$(differs "$W/wave_device" "$W/wave_host")
[ "$n" = 0 ] \
    && say "[device] the GPU wave writes the host wave's files, byte for byte" ok \
    || say "[device] the GPU wave writes the host wave's files, byte for byte ($n differ)" FAIL
n=$(differs "$W/wave_reversed" "$W/wave_host")
[ "$n" != 0 ] \
    && say "CONTROL  the wave's visits folded in reverse change $n written files" ok \
    || say "CONTROL  the wave's visits folded in reverse change the written files" FAIL
finish "setRDeltaT's smoothing wave on the device is the host wave"
