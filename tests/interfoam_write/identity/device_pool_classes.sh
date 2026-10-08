#!/usr/bin/env bash
# The write gate: the device memory pool keyed on SIZE CLASSES and trimmed of what stays idle, against the pool
# keyed on exact sizes that kept every block for good (BRAE_CONTROL_POOL_EXACT=1), on sixty steps of
# laminar/oscillatingBox -- a mesh that refines, moves and unrefines, so its buffers change size.
# A freed block went to the next request of EXACTLY its byte count. On a mesh whose sizes change nothing was
# reused and nothing was freed. MEASURED 2026-10-05 after the first step, over the benchmark interval:
#   RAS/motorBike         839 -> 217 cudaMalloc a step, 10.6 -> 2.3 ms; idle list at the end 3,607 -> 245 MB;
#                         the step 134.6 -> 125.9 ms
#   damBreakWithObstacle  680 -> 139 a step, 15.1 -> 2.7 ms; idle 5,221 -> 193 MB; the step 170.4 -> 157.9
#   RAS/mixerVesselAMI    38 -> 4.5 a step; idle 2,392 -> 668 MB; in use 3,443 -> 3,567 MB (+3.6%)
#   RAS/DTCHullMoving     (fixed topology) unchanged; in use 3,428 -> 3,522 MB (+2.7%)
# Which block a buffer sits in reaches no number, so nothing written may change. What MUST change is the pool's
# own count, read off its line (BRAE_POOL_REPORT=1): fewer allocations after the first step and less left idle.
# The CONTROL rounds a request DOWN to its class (BRAE_CONTROL_POOL_SHORT_CLASS=1), so a block is shorter than
# its buffer: the pool must stop the run at the first one.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
OB="$TUT/multiphase/interFoam/laminar/oscillatingBox"
stage_allrun "$OB" "$W/dp_stage" 0.001 > "$W/dp_stage.txt" 2>&1 \
    || { say "oscillatingBox did not stage" FAIL; finish "device pool size classes identity"; }
sed -i -E 's/^(endTime\s+)[^;]*;/\10.06;/' "$W/dp_stage/system/controlDict"
for v in exact classes short; do
    e="$W/dp_$v"
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$W/dp_stage/0" "$W/dp_stage/constant" "$W/dp_stage/system" "$e/"
    case $v in
        exact)   runbrae "$e" device BRAE_POOL_REPORT=1 BRAE_CONTROL_POOL_EXACT=1 ;;
        classes) runbrae "$e" device BRAE_POOL_REPORT=1 ;;
        # the control is expected to stop: run it directly, runbrae would end the gate
        short)   ( cd "$e" && BRAE_CONTROL_POOL_SHORT_CLASS=1 "$BIN" -case . -device > log.brae 2>&1
                   echo $? > exit.txt ) ;;
    esac
done
n=0
t=0
for d in $(timedirs "$W/dp_exact"); do
    for f in $(cd "$W/dp_exact/$d" && find . -type f | sort); do
        t=$((t + 1))
        cmp -s "$W/dp_exact/$d/$f" "$W/dp_classes/$d/$f" || n=$((n + 1))
    done
done
what="[device] sixty steps of a refining, moving mesh on either pool: $t written files, $n differ"
[ "$t" -gt 0 ] && [ "$n" = 0 ] && [ "$(timedirs "$W/dp_classes")" = "$(timedirs "$W/dp_exact")" ] \
    && say "$what" ok || say "$what" FAIL
# poolnum <arm> <sed pattern>: a number off the pool's own line
poolnum()
{
    grep -a "device pool after the first step" "$W/dp_$1/log.brae" | sed -E "$2"
}
allocs() { poolnum "$1" 's/.*: ([0-9]+) allocations.*/\1/'; }
idle() { poolnum "$1" 's/.*and ([0-9.]+) MB at the end.*/\1/'; }
ae=$(allocs exact)
ac=$(allocs classes)
ie=$(idle exact)
ic=$(idle classes)
what="the pool's own count after the first step: $ae allocations and $ie MB idle exact, $ac and $ic MB in classes"
grep -aq "(size classes)" "$W/dp_classes/log.brae" && grep -aq "(exact-size keys" "$W/dp_exact/log.brae" \
    && python3 -c "import sys; ae,ac,ie,ic=map(float,sys.argv[1:]); sys.exit(0 if ac*2 < ae and ic*2 < ie else 1)" \
        "$ae" "$ac" "$ie" "$ic" && say "$what" ok || say "$what" FAIL
what="CONTROL  a request rounded DOWN to its class: the pool stops the run and names the short block"
[ "$(cat "$W/dp_short/exit.txt")" != 0 ] \
    && grep -q "device pool: a request of .* is shorter than the buffer" "$W/dp_short/log.brae" \
    && say "$what" ok || say "$what" FAIL
finish "the device pool in size classes writes what the exact-size pool wrote, and reuses and frees what it kept"
