#!/usr/bin/env bash
# The write gate, a rigid body's state file: floatingObject under Euler, a body at rest, with the list-form control.
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
q_of_case
d="$W/q_br"
mkdir -p "$d"
cp -r "$W/q_of/0" "$W/q_of/constant" "$W/q_of/system" "$d/"
runbrae "$d" host
ok=1
[ "$(timedirs "$d")" = "$(timedirs "$W/q_of")" ] || ok=0
for t in $(timedirs "$W/q_of"); do
    [ "$(filesets "$W/q_of" $t)" = "$(filesets "$d" $t)" ] || ok=0
done
[ $ok -eq 1 ] && say "ARM Q  [host] floatingObject (Euler): OpenFOAM's directories and file sets" ok \
              || say "ARM Q  [host] floatingObject (Euler): OpenFOAM's directories and file sets" FAIL
python3 "$CMP" "$W/q_of" "$d" $(timedirs "$W/q_of") > "$W/cmp_q.txt" 2>&1
judge "floatingObject host" "$W/cmp_q.txt" "$BOUND_FO" "$W/q_of/log.interFoam" \
    && say "ARM Q  [host] floatingObject (Euler): every file's structure is OpenFOAM's, every value within $BOUND_FO" ok \
    || { say "ARM Q  [host] floatingObject (Euler): every file's structure is OpenFOAM's, every value within $BOUND_FO" FAIL; grep -v RESULT "$W/cmp_q.txt" | grep -B1 "^      " | head -12; }
rbstate "$W/q_of" "$d" "$BOUND_RB_DTC" \
    && say "ARM Q  [host] floatingObject's rigidBodyMotionState at rest: OpenFOAM's text, \`2 { 0 }\` included" ok \
    || say "ARM Q  [host] floatingObject's rigidBodyMotionState at rest: OpenFOAM's text, \`2 { 0 }\` included" FAIL
d="$W/q_ctl_paren"
mkdir -p "$d"
cp -r "$W/q_of/0" "$W/q_of/constant" "$W/q_of/system" "$d/"
runbrae "$d" host BRAE_CONTROL_RBSTATE_PAREN=1
rbstate "$W/q_of" "$d" "$BOUND_RB_DTC" > "$W/rbparen.txt" \
    && { say "CONTROL  BRAE_CONTROL_RBSTATE_PAREN=1 fails floatingObject's state text" FAIL; cat "$W/rbparen.txt"; } \
    || say "CONTROL  BRAE_CONTROL_RBSTATE_PAREN=1 fails floatingObject's state text" ok
finish "arm Q: a body at rest writes OpenFOAM's uniform lists"
