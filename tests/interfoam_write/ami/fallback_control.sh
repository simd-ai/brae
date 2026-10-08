#!/usr/bin/env bash
# The write gate on a cyclicAMI pair (RAS/mixerVesselAMI): the PCG fallback for GAMG across the pair, switched off.
. "$(dirname "$0")/../lib.sh"
# OpenFOAM's run alone (cached); the tutorial's own row is tests/interfoam_write/tutorial.sh
wcase mixerVesselAMI of
[ -d "$W/w_of_mixerVesselAMI" ] || { say "ARM Z  mixerVesselAMI did not stage" FAIL; finish "arm Z"; }
# with BRAE_CONTROL_NO_AMI_PCG_FALLBACK=1 the tutorial's GAMG entry is left as read, and brae stops on the
# pair by name with nothing written
e="$W/z_ctl_nofallback"
mkdir -p "$e"
cp -r "$W/w_of_mixerVesselAMI/0" "$W/w_of_mixerVesselAMI/constant" "$W/w_of_mixerVesselAMI/system" "$e/"
cp "$e/system/fvSolution.gamg" "$e/system/fvSolution"
( cd "$e" && BRAE_CONTROL_NO_AMI_PCG_FALLBACK=1 "$BIN" -case . > log.brae 2>&1 ); rc=$?
[ $rc -ne 0 ] && grep -q "is an AMI" "$e/log.brae" && [ -z "$(timedirs "$e")" ] \
    && say "CONTROL  BRAE_CONTROL_NO_AMI_PCG_FALLBACK=1 stops mixerVesselAMI on its GAMG entry, by name" ok \
    || say "CONTROL  BRAE_CONTROL_NO_AMI_PCG_FALLBACK=1 stops mixerVesselAMI on its GAMG entry, by name" FAIL
finish "arm Z control: without the fallback the pair is refused by name"
