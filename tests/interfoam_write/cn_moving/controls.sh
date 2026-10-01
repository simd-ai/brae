#!/usr/bin/env bash
# The write gate: the three controls of the floatingObject tutorial row's CrankNicolson state.
. "$(dirname "$0")/../lib.sh"
wcase floatingObject of
[ -d "$W/w_of_floatingObject" ] || { say "ARM X  floatingObject did not stage" FAIL; finish "arm X"; }
# the bound the tutorial row is held to (lib.sh, BOUND_WFO), and its three controls: see cn_moving/released.sh
# for what each one breaks
# the three controls on the tutorial as shipped, against arm W's own OpenFOAM run and bound
ot=$(timedirs "$W/w_of_floatingObject")
for ctl in "BRAE_CONTROL_CN_PHIOLD_PREV:/U " "BRAE_CONTROL_CN_OLD_AT_ENTRY:/epsilon_0 " "BRAE_CONTROL_CN_DDT0_CELLS_ONLY:/ddt0\\(k\\) "; do
    var=${ctl%%:*}
    file=${ctl#*:}
    e="$W/x_ctl_$var"
    mkdir -p "$e"
    cp -r "$W/w_of_floatingObject/0" "$W/w_of_floatingObject/constant" "$W/w_of_floatingObject/system" "$e/"
    runbrae "$e" host "$var=1"
    python3 "$CMP" "$W/w_of_floatingObject" "$e" $ot > "$W/cmp_x_$var.txt" 2>&1
    judge "control $var" "$W/cmp_x_$var.txt" "$BOUND_WFO" "$W/w_of_floatingObject/log.interFoam" > "$W/x_$var.txt"
    grep -qE "over the bound: [0-9.e+-]+$file" "$W/x_$var.txt" \
        && say "CONTROL  $var=1 puts floatingObject's ${file% } over the bound" ok \
        || { say "CONTROL  $var=1 puts floatingObject's ${file% } over the bound" FAIL; cat "$W/x_$var.txt"; }
done
finish "arm X controls: each defect is seen"
