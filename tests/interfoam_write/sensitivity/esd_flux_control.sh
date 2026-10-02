#!/usr/bin/env bash
# The write gate: the device control behind RAS/electrostaticDeposition's device bound.
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
wcase electrostaticDeposition of
o="$W/w_of_electrostaticDeposition"
[ -d "$o" ] || { say "electrostaticDeposition did not stage" FAIL; finish "esd control"; }
# The tutorial row holds the device arm to 1.8e-06 (measured 1.8e-07, p_rgh). Until 2026-10-02 it read 1.7e-05:
# on a moving mesh the device evaluated U's flux-switched patches after the pressure corrector with the
# RELATIVE flux, where pEqn.H evaluates them while phi is still absolute (:61, makeRelative at :69). On
# `side-02` the two fluxes have opposite sign on all 225 faces. BRAE_CONTROL_DEVICE_U_PATCH_RELATIVE=1 puts
# the relative flux back: MEASURED U and Uf 1.7e-05.
e="$W/esd_ctl"
mkdir -p "$e"
cp -r "$o/0" "$o/constant" "$o/system" "$e/"
runbrae "$e" device BRAE_CONTROL_DEVICE_U_PATCH_RELATIVE=1
grep -q "CONTROL MODE" "$e/log.brae" \
    && say "CONTROL  [device] BRAE_CONTROL_DEVICE_U_PATCH_RELATIVE=1 ran as a control" ok \
    || say "CONTROL  [device] BRAE_CONTROL_DEVICE_U_PATCH_RELATIVE=1 ran as a control" FAIL
python3 "$CMP" "$o" "$e" $(timedirs "$o") > "$W/cmp_esd_ctl.txt" 2>&1
judge "control U patches on the relative flux" "$W/cmp_esd_ctl.txt" 1.8e-06 "$o/log.interFoam" > "$W/esd_ctl.txt" \
    && { say "CONTROL  [device] the relative flux puts electrostaticDeposition over its bound" FAIL; cat "$W/esd_ctl.txt"; } \
    || say "CONTROL  [device] the relative flux puts electrostaticDeposition over its bound" ok
grep -qE "over the bound: [0-9.e+-]+/U " "$W/esd_ctl.txt" \
    && say "CONTROL  ...and it is U that goes over" ok \
    || { say "CONTROL  ...and it is U that goes over" FAIL; cat "$W/esd_ctl.txt"; }
finish "electrostaticDeposition device control: the flux U's patches read"
