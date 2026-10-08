#!/usr/bin/env bash
# The write gate: the alpha hooks' boundary normal with its first half on the GPU, against the host's, on
# capillaryRise.
. "$(dirname "$0")/../lib.sh"
# Since 2026-10-03 the device loop's alpha hooks form each boundary cell's internal-face sum of the nHat
# gradient on the GPU (nHatStencilInternalKernel, the host loop's terms and order) and the host continues from it
# (finishNHatBoundary) -- the alpha step 111 -> 104 ms a step on RAS/DTCHull. MEASURED, GPU half against host
# half, every written file BYTE-IDENTICAL on DTCHull (25 steps), capillaryRise, weirOverflow and
# damBreakPermeable. capillaryRise carries the contact angle, where the boundary normal sets alpha's wall
# gradient: the control (BRAE_CONTROL_NHAT_DEVICE_OWNER_ONLY=1, the neighbour side's terms dropped in the kernel)
# changes 14 of its files and none of weirOverflow's or damBreakPermeable's.
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase capillaryRise of > "$W/nd_stage.txt" 2>&1
o="$W/w_of_capillaryRise"
[ -d "$o" ] || { say "capillaryRise did not stage" FAIL; finish "nHat device identity"; }
for v in device host owner; do
    e="$W/nd_$v"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    case $v in
        device) runbrae "$e" device ;;
        host)   runbrae "$e" device BRAE_CONTROL_NHAT_HOST_GRADIENT=1 ;;
        owner)  runbrae "$e" device BRAE_CONTROL_NHAT_DEVICE_OWNER_ONLY=1 ;;
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
n=$(differs "$W/nd_device" "$W/nd_host")
[ "$n" = 0 ] \
    && say "[device] the GPU half of the boundary normal writes the host half's files, byte for byte" ok \
    || say "[device] the GPU half of the boundary normal writes the host half's files, byte for byte ($n differ)" FAIL
n=$(differs "$W/nd_owner" "$W/nd_host")
[ "$n" != 0 ] \
    && say "CONTROL  the kernel without the neighbour side's terms changes $n written files" ok \
    || say "CONTROL  the kernel without the neighbour side's terms changes the written files" FAIL
finish "the alpha hooks' boundary normal with its first half on the GPU is the host's"
