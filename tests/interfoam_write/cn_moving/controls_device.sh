#!/usr/bin/env bash
# The write gate: the device loop's controls of the floatingObject tutorial row's CrankNicolson state.
. "$(dirname "$0")/../lib.sh"
wcase floatingObject of
[ -d "$W/w_of_floatingObject" ] || { say "ARM X  floatingObject did not stage" FAIL; finish "arm X"; }
# the bound the tutorial row is held to (lib.sh, BOUND_WFO), and its three controls: see cn_moving/released.sh
# for what each one breaks
# the three controls on the tutorial as shipped, against arm W's own OpenFOAM run and bound
ot=$(timedirs "$W/w_of_floatingObject")
for ctl in "BRAE_CONTROL_CN_PHIOLD_PREV:/U " "BRAE_CONTROL_CN_OLD_AT_ENTRY:/epsilon_0 "; do
    var=${ctl%%:*}
    file=${ctl#*:}
    e="$W/xd_ctl_$var"
    mkdir -p "$e"
    cp -r "$W/w_of_floatingObject/0" "$W/w_of_floatingObject/constant" "$W/w_of_floatingObject/system" "$e/"
    runbrae "$e" device "$var=1"
    python3 "$CMP" "$W/w_of_floatingObject" "$e" $ot > "$W/cmp_xd_$var.txt" 2>&1
    judge "control $var" "$W/cmp_xd_$var.txt" "$BOUND_WFO_DEVICE" "$W/w_of_floatingObject/log.interFoam" > "$W/xd_$var.txt"
    grep -qE "over the bound: [0-9.e+-]+$file" "$W/xd_$var.txt" \
        && say "CONTROL  [device] $var=1 puts floatingObject's ${file% } over the bound" ok \
        || { say "CONTROL  [device] $var=1 puts floatingObject's ${file% } over the bound" FAIL; cat "$W/xd_$var.txt"; }
done
# ...and the refusal the device loop had, put back: nothing written, the files named
e="$W/xd_refused"
mkdir -p "$e"
cp -r "$W/w_of_floatingObject/0" "$W/w_of_floatingObject/constant" "$W/w_of_floatingObject/system" "$e/"
( cd "$e" && BRAE_CONTROL_DEVICE_CN_WRITE_REFUSED=1 "$BIN" -case . -device > log.brae 2>&1 )
grep -q "does not keep in its written form" "$e/log.brae" && [ -z "$(timedirs "$e")" ] \
    && say "CONTROL  [device] BRAE_CONTROL_DEVICE_CN_WRITE_REFUSED=1 refuses by name and writes nothing" ok \
    || { say "CONTROL  [device] BRAE_CONTROL_DEVICE_CN_WRITE_REFUSED=1 refuses by name and writes nothing" FAIL; tail -3 "$e/log.brae" | cut -c1-300; }
finish "arm X device controls: each defect is seen"
