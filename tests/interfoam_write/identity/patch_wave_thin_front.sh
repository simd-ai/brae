#!/usr/bin/env bash
# The write gate: the GPU wall-distance wave with a thin front swept inside one launch, against every half-sweep
# launched from the host, on waveMakerFlap.
. "$(dirname "$0")/../lib.sh"
# Since 2026-10-04 a front that fits one thread block (up to 1,792 changed cells or faces) is swept inside ONE
# launch for as long as it fits (device_patch_wave.cu, pwThinFrontKernel): the launched half-sweeps' own folds as
# phases of one block. And a changed BOUNDARY face is no longer put on the changed list: its only cell is the
# one it was just set from, and the visit back changes nothing.
# MEASURED, the wave, ms a step, waveMakerPiston at 56,000 / 224,000 / 896,000 cells: 31.9 -> 11.6, 71.8 -> 32.0,
# 159 -> 108; the step 86 -> 65, 281 -> 243, 1,087 -> 1,035. A wider front keeps the launches: on the 3-D
# waveMakerMultiPaddleFlap the block was 2.6 times SLOWER when allowed 7,168 entries, so it is not.
# ORACLE: BRAE_CONTROL_PATCH_WAVE_CHECK=1 runs the host's wave too, at every call, and compares every cell's and
# boundary face's squared distance to the bit. The block is also SHRUNK two ways so that this small case hands
# the wave back and forth: BRAE_CONTROL_PATCH_WAVE_BLOCK_FACES=200 (the faces outgrow it after the first sweep:
# 399 launched half-sweeps a step) and BRAE_CONTROL_PATCH_WAVE_BLOCK_CELLS=100 (the cells outgrow it at every
# sweep: 401 launches a step, the second half-sweep on the host's path each time).
# The control drops the neighbour side's visits (BRAE_CONTROL_PATCH_WAVE_OWNER_ONLY=1): measured, cell 1 unset.
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase waveMakerFlap of > "$W/tf_stage.txt" 2>&1
o="$W/w_of_waveMakerFlap"
[ -d "$o" ] || { say "waveMakerFlap did not stage" FAIL; finish "thin front identity"; }
for v in block faces cells launched control; do
    e="$W/tf_$v"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    case $v in
        block)    runbrae "$e" device BRAE_CONTROL_PATCH_WAVE_CHECK=1 ;;
        faces)    runbrae "$e" device BRAE_CONTROL_PATCH_WAVE_CHECK=1 BRAE_CONTROL_PATCH_WAVE_BLOCK_FACES=200 ;;
        cells)    runbrae "$e" device BRAE_CONTROL_PATCH_WAVE_CHECK=1 BRAE_CONTROL_PATCH_WAVE_BLOCK_CELLS=100 ;;
        launched) runbrae "$e" device BRAE_CONTROL_PATCH_WAVE_NO_BLOCK=1 BRAE_CONTROL_PATCH_WAVE_LIST_BOUNDARY=1 ;;
        control)  ( cd "$e" && BRAE_CONTROL_PATCH_WAVE_CHECK=1 BRAE_CONTROL_PATCH_WAVE_OWNER_ONLY=1 \
                        "$BIN" -case . -device > log.brae 2>&1 )
                  echo $? > "$W/tf_control.rc" ;;
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
mark="is swept inside one launch"
grep -q "$mark" "$W/tf_block/log.brae" && ! grep -q "$mark" "$W/tf_launched/log.brae" \
    && say "the default run sweeps the thin front inside one launch, the other arm launches every half-sweep" ok \
    || say "the default run sweeps the thin front inside one launch, the other arm launches every half-sweep" FAIL
n=$(( $(differs "$W/tf_block" "$W/tf_launched") + $(differs "$W/tf_faces" "$W/tf_launched") \
    + $(differs "$W/tf_cells" "$W/tf_launched") ))
what="[device] one launch, and both hand-overs, give the host's wave at every call and the launched path's files"
[ "$n" = 0 ] && [ "$(timedirs "$W/tf_block")" = "$(timedirs "$o")" ] \
    && say "$what" ok || say "$what ($n differ)" FAIL
what="CONTROL  without the neighbour side's visits the check names the cell and stops"
[ "$(cat "$W/tf_control.rc")" != 0 ] \
    && grep -q "the wave run elsewhere is not the host's" "$W/tf_control/log.brae" \
    && say "$what" ok || say "$what" FAIL
finish "the thin front swept inside one launch is the launched wave"
