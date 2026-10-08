#!/usr/bin/env bash
# The write gate on a cyclicAMI pair (RAS/mixerVesselAMI): a coupled patch's value after a field expression is the RESULT's cells evaluated.
. "$(dirname "$0")/../lib.sh"
# OpenFOAM's run alone (cached); the tutorial's own row is tests/interfoam_write/tutorial.sh
wcase mixerVesselAMI of
[ -d "$W/w_of_mixerVesselAMI" ] || { say "ARM Z  mixerVesselAMI did not stage" FAIL; finish "arm Z"; }
# Z: a coupled patch's value after a field expression is the RESULT's cells evaluated, w*f_P + (1 - w)*f_N
# (correctLocalBoundaryConditions; coupledFvPatchField.H:198-205), not the operands' patch arithmetic.
# CONTROL: BRAE_CONTROL_COUPLED_OPERAND_VALUES=1 writes 1/(A's patch value) for rAU and
# p_rgh_b + rho_b*gh_b for p -- MEASURED 1.6e-02 and 6.9e-05 off on the AMI pair, both over the bound.
d="$W/z_ctl_operands"
mkdir -p "$d"
cp -r "$W/w_of_mixerVesselAMI/0" "$W/w_of_mixerVesselAMI/constant" "$W/w_of_mixerVesselAMI/system" "$d/"
runbrae "$d" host BRAE_CONTROL_COUPLED_OPERAND_VALUES=1
python3 "$CMP" "$W/w_of_mixerVesselAMI" "$d" $(timedirs "$W/w_of_mixerVesselAMI") > "$W/cmp_z.txt" 2>&1
judge "control operand values" "$W/cmp_z.txt" 5e-11 "$W/w_of_mixerVesselAMI/log.interFoam" > "$W/z.txt"
grep -qE "over the bound: .*/rAU " "$W/z.txt" && grep -qE "over the bound: .*/p " "$W/z.txt" \
    && say "CONTROL  BRAE_CONTROL_COUPLED_OPERAND_VALUES=1 puts mixerVesselAMI's rAU and p over the bound" ok \
    || { say "CONTROL  BRAE_CONTROL_COUPLED_OPERAND_VALUES=1 puts mixerVesselAMI's rAU and p over the bound" FAIL; cat "$W/z.txt"; }
finish "arm Z control: the operands' patch arithmetic is seen"
