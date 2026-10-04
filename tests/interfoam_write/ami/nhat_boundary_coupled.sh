#!/usr/bin/env bash
# The alpha hooks' boundary normal from the boundary's own cells on a mesh with a COUPLED pair, on
# RAS/mixerVesselAMI (894,950 cells, 119,480 of them on the boundary). The boundary-only stencil declined any
# mesh with a coupled patch, so each of the six alpha hooks of a step took interfaceProperties' calculateK whole
# on the host: MEASURED 2026-10-04, 152.5 ms a step. A coupled face is now in the stencil like any other: its
# term of the gradient is the two cells' interpolated value (coupledLinear, fvc::gaussGrad's own) and its normal
# reads the gradient on both sides, whose cells are the partner patch's face cells.
# A coupled mesh's runs do not reproduce (the pair's products sum atomically), so the check is in the code:
# BRAE_CONTROL_NHAT_BOUNDARY_CHECK=1 takes calculateK whole as well at every call and compares every patch face.
# BIT FOR BIT where the host forms the gradient (BRAE_CONTROL_NHAT_HOST_GRADIENT=1); where the GPU kernel forms
# it, within 6e-7 of the largest normal -- the kernel's sums differ from the host's in the last bit of a term,
# and the widest gap measured is 6.2e-08 of the largest. The CONTROL leaves the patch faces' terms out of the
# gradient (BRAE_CONTROL_NHAT_NO_PATCH_TERMS=1).
. "$(dirname "$0")/../lib.sh"
unset BRAE_PRESSURE_CASE_SOLVER
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase mixerVesselAMI of > "$W/nc_stage.txt" 2>&1
o="$W/w_of_mixerVesselAMI"
[ -d "$o" ] || { say "mixerVesselAMI did not stage" FAIL; finish "boundary normal on a coupled mesh"; }
# arm <name> [env...]: brae on a copy of the staged case, the comparison on; never ends the gate
arm()
{
    local e="$W/nc_$1"
    shift
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    ( cd "$e" && env BRAE_CONTROL_NHAT_BOUNDARY_CHECK=1 "$@" "$BIN" -case . -device > log.brae 2>&1
      echo $? > exit.txt )
}
mark="nHat: the alpha hooks take the boundary normal from the boundary's own cells"
arm host BRAE_CONTROL_NHAT_HOST_GRADIENT=1
arm device BRAE_X=1
arm nopatch BRAE_CONTROL_NHAT_NO_PATCH_TERMS=1
what="[device] host gradient: every patch face's normal is calculateK's, bit for bit, at every call"
[ "$(cat "$W/nc_host/exit.txt")" = 0 ] && grep -aq "$mark" "$W/nc_host/log.brae" \
    && [ "$(timedirs "$W/nc_host")" = "$(timedirs "$o")" ] && say "$what" ok || say "$what" FAIL
gap=$(grep -a "nHat boundary check: the widest gap" "$W/nc_device/log.brae" | tail -1 \
          | sed -E 's/.*so far ([^ ]+) .*/\1/')
what="[device] GPU gradient: within 6e-7 of the largest normal at every call (widest gap ${gap:-none})"
[ "$(cat "$W/nc_device/exit.txt")" = 0 ] && grep -aq "$mark" "$W/nc_device/log.brae" \
    && [ "$(timedirs "$W/nc_device")" = "$(timedirs "$o")" ] && say "$what" ok || say "$what" FAIL
what="CONTROL  the patch faces' terms left out of the gradient: the check stops the run and names the face"
[ "$(cat "$W/nc_nopatch/exit.txt")" != 0 ] \
    && grep -aq "NHAT_BOUNDARY_CHECK: the boundary normal of patch" "$W/nc_nopatch/log.brae" \
    && say "$what" ok || say "$what" FAIL
finish "the boundary normal on a coupled mesh comes from the boundary's own cells"
