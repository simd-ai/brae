#!/usr/bin/env bash
# The write gate's CONTROLS for hooks_empty_on_device.sh, three more of its six arrays: the boundary normal,
# the surface-tension flux and nu each written as 1 on the empty faces, with the bitwise check on. The check
# has to stop the run on each and name that array -- a comparison that passes on six arrays proves nothing for
# an array it cannot fail on. (alpha's and snGrad(rho)'s controls are in hooks_empty_on_device.sh.
# snGrad(p_rgh)'s is not run: it is built with a momentum predictor alone, which no 2-D row here has.)
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase streamFunction of > "$W/hc_stage.txt" 2>&1
o="$W/w_of_streamFunction"
[ -d "$o" ] || { say "streamFunction did not stage" FAIL; finish "hooks' empty faces: the controls"; }
stops()   # stops <array's switch name> <the array's name in the check's message> <what the gate says>
{
    local e="$W/hc_$1"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    ( cd "$e" && BRAE_CONTROL_HOOKS_EMPTY_WRONG="$1" BRAE_CONTROL_HOOKS_EMPTY_CHECK=1 \
          "$BIN" -case . -device > log.brae 2>&1; echo $? > exit.txt )
    local what="CONTROL  with $3 written as 1 on the empty faces the check stops the run and names it"
    [ "$(cat "$e/exit.txt")" != 0 ] && grep -q "CONTROL MODE: the empty faces' entries of \`$1\`" "$e/log.brae" \
        && grep -q "HOOKS_EMPTY_CHECK: $2, boundary face" "$e/log.brae" && say "$what" ok || say "$what" FAIL
}
stops normal "the boundary normal" "the boundary normal"
stops flux "the surface-tension flux's patch faces" "the surface-tension flux"
stops nu "nu's patch values" "nu"
finish "each of the hooks' arrays kept on the GPU fails its check when its empty faces are written wrong"
