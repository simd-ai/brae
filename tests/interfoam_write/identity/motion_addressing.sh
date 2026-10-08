#!/usr/bin/env bash
# The write gate: the displacement motion solver's kept addressing against a rebuild at every step, on
# waveMakerPiston.
. "$(dirname "$0")/../lib.sh"
# Since 2026-10-03 the displacementLaplacian solver keeps what its two per-step passes take from the mesh's
# ADDRESSING: the cell-to-face lists of the inverseDistance diffusivity's wall-distance wave (patchWave's
# `cells`), and volPointInterpolation's boundary patch, patch-point flags and weight lists -- its WEIGHTS are
# still remade on the moved mesh. MEASURED on waveMakerPiston refined to 224,000 cells, ms a step: cells to
# points 255 -> 11, the diffusivity 250 -> 186, the step 996 -> 654. The control keeps the weights too
# (BRAE_CONTROL_VPI_STALE=1), those of the mesh before it moved.
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase waveMakerPiston of > "$W/ma_stage.txt" 2>&1
o="$W/w_of_waveMakerPiston"
[ -d "$o" ] || { say "waveMakerPiston did not stage" FAIL; finish "motion addressing identity"; }
for v in kept rebuilt stale; do
    e="$W/ma_$v"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    case $v in
        kept)    runbrae "$e" device ;;
        rebuilt) runbrae "$e" device BRAE_CONTROL_VPI_REBUILD=1 BRAE_CONTROL_MOTION_CELLS_REBUILT=1 ;;
        stale)   runbrae "$e" device BRAE_CONTROL_VPI_STALE=1 ;;
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
n=$(differs "$W/ma_kept" "$W/ma_rebuilt")
[ "$n" = 0 ] \
    && say "[device] the kept addressing writes the rebuilt one's files, byte for byte" ok \
    || say "[device] the kept addressing writes the rebuilt one's files, byte for byte ($n differ)" FAIL
n=$(differs "$W/ma_stale" "$W/ma_rebuilt")
[ "$n" != 0 ] \
    && say "CONTROL  interpolation weights left from before the move change $n written files" ok \
    || say "CONTROL  interpolation weights left from before the move change the written files" FAIL
finish "the motion solver's kept addressing is the rebuilt one"
