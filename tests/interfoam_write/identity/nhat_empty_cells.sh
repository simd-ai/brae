#!/usr/bin/env bash
# The write gate: the alpha hooks' boundary normal with the empty patches left out of its stencil, against the
# stencil that holds them, on capillaryRise.
. "$(dirname "$0")/../lib.sh"
# Since 2026-10-04 the stencil of the alpha hooks' boundary normal leaves out the cells that touch only an
# `empty` patch, and the normal on an empty patch's faces is zero (NHatBoundaryStencil::skipEmpty): OpenFOAM has
# no faces there, and what reads it on the device skips them (deviceDiv) or multiplies them by a zero flux. On a
# 2-D mesh the stencil was every cell.
# MEASURED on waveMakerPiston refined to 896,000 cells: the stencil 896,000 -> 4,316 cells, the boundary normal
# 71 -> 1.4 ms a step, the alpha hook 106 -> 36. Every written file BYTE-IDENTICAL with the empty patches' cells
# kept in (BRAE_CONTROL_NHAT_EMPTY_CELLS=1) on capillaryRise, weirOverflow, waveMakerPiston, stokesI,
# sloshingTank3D6DoF, damBreakWithObstacle and DTCHull. capillaryRise carries the contact angle, where the
# boundary normal sets alpha's wall gradient; the control drops the neighbour side's terms in the kernel
# (BRAE_CONTROL_NHAT_DEVICE_OWNER_ONLY=1) and the files must change.
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase capillaryRise of > "$W/ne_stage.txt" 2>&1
o="$W/w_of_capillaryRise"
[ -d "$o" ] || { say "capillaryRise did not stage" FAIL; finish "nHat empty cells identity"; }
for v in out in owner; do
    e="$W/ne_$v"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    case $v in
        out)   runbrae "$e" device ;;
        in)    runbrae "$e" device BRAE_CONTROL_NHAT_EMPTY_CELLS=1 ;;
        owner) runbrae "$e" device BRAE_CONTROL_NHAT_DEVICE_OWNER_ONLY=1 ;;
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
mark="an empty patch's left out"
nout=$(grep -a "nHat: the alpha hooks" "$W/ne_out/log.brae" | sed -E 's/.*\(([0-9]+) of ([0-9]+)\).*/\1/')
nin=$(grep -a "nHat: the alpha hooks" "$W/ne_in/log.brae" | sed -E 's/.*\(([0-9]+) of ([0-9]+)\).*/\1/')
what="the default stencil leaves the empty patches out ($nout cells against $nin) and says so"
grep -q "$mark" "$W/ne_out/log.brae" && ! grep -q "$mark" "$W/ne_in/log.brae" && [ "${nout:-0}" -gt 0 ] \
    && [ "${nout:-0}" -lt "${nin:-0}" ] && say "$what" ok || say "$what" FAIL
n=$(differs "$W/ne_out" "$W/ne_in")
[ "$n" = 0 ] \
    && say "[device] without the empty patches' cells the boundary normal writes the same files, byte for byte" ok \
    || say "[device] without the empty patches' cells the boundary normal writes the same files ($n differ)" FAIL
n=$(differs "$W/ne_owner" "$W/ne_in")
[ "$n" != 0 ] \
    && say "CONTROL  the kernel without the neighbour side's terms changes $n written files" ok \
    || say "CONTROL  the kernel without the neighbour side's terms changes the written files" FAIL
finish "the boundary normal with the empty patches left out is the one with them in"
