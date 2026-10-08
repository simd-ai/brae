#!/usr/bin/env bash
# The write gate: brae's kept host memory against glibc's defaults, on waveMakerPiston.
. "$(dirname "$0")/../lib.sh"
# Since 2026-10-04 brae_interFoam tells glibc to keep the memory it frees (src/cuda/host_allocator.cuh): no mmap
# for a malloc, no trimming, so a step's large work arrays are not faulted in again at every step.
# MEASURED on waveMakerPiston refined to 896,000 cells: 2,499 -> 1,702 ms a step, peak resident memory +0.9%;
# thirteen write-gate cases byte-identical against glibc's defaults (BRAE_HOST_ALLOCATOR=system).
# The notice is a proof: after setting it brae allocates 64 MB and stops unless the block came from the heap.
# The control skips the setting and keeps that probe (BRAE_CONTROL_HOST_ALLOCATOR_UNSET=1): the run must stop.
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase waveMakerPiston of > "$W/ha_stage.txt" 2>&1
o="$W/w_of_waveMakerPiston"
[ -d "$o" ] || { say "waveMakerPiston did not stage" FAIL; finish "host allocator identity"; }
for v in keep system env control bogus; do
    e="$W/ha_$v"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    case $v in
        keep)    runbrae "$e" device ;;
        system)  runbrae "$e" device BRAE_HOST_ALLOCATOR=system ;;
        env)     runbrae "$e" device MALLOC_TOP_PAD_=1048576 ;;
        control) ( cd "$e" && BRAE_CONTROL_HOST_ALLOCATOR_UNSET=1 "$BIN" -case . -device > log.brae 2>&1 )
                 echo $? > "$W/ha_control.rc" ;;
        bogus)   ( cd "$e" && BRAE_HOST_ALLOCATOR=bogus "$BIN" -case . -device > log.brae 2>&1 )
                 echo $? > "$W/ha_bogus.rc" ;;
    esac
done
mark="host memory: freed blocks are kept"
what="the default run keeps its memory and says so, glibc's arm does not, and a setting in the environment is left"
grep -q "$mark" "$W/ha_keep/log.brae" && ! grep -q "host memory" "$W/ha_system/log.brae" \
    && grep -q "host memory: the environment sets the allocator itself" "$W/ha_env/log.brae" \
    && say "$what" ok || say "$what" FAIL
n=0
for t in $(timedirs "$o"); do
    for f in $(cd "$W/ha_system/$t" && find . -type f | sort); do
        cmp -s "$W/ha_system/$t/$f" "$W/ha_keep/$t/$f" || n=$((n + 1))
    done
done
[ "$n" = 0 ] && [ "$(timedirs "$W/ha_keep")" = "$(timedirs "$o")" ] \
    && say "[device] with its memory kept brae writes the files it writes under glibc's defaults" ok \
    || say "[device] with its memory kept brae writes the files it writes under glibc's defaults ($n differ)" FAIL
what="CONTROL  with the setting skipped the 64 MB probe stops the run, and an unknown value is refused by name"
[ "$(cat "$W/ha_control.rc")" != 0 ] && grep -q "host allocator setting did not take" "$W/ha_control/log.brae" \
    && [ "$(cat "$W/ha_bogus.rc")" != 0 ] && grep -q "BRAE_HOST_ALLOCATOR=bogus is not one of" "$W/ha_bogus/log.brae" \
    && say "$what" ok || say "$what" FAIL
finish "brae's kept host memory changes no written byte"
