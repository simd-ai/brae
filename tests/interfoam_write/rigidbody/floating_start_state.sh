#!/usr/bin/env bash
# The write gate, a rigid body's state file: a start state read back as OpenFOAM reads it (four forms).
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
q_of_case
for v in A B D E; do
    q='2 ( 0.01 0.02 )'
    qdd='2 { 0 }'
    hdr=yes
    [ $v = A ] && q='2 { 0.01 }'
    [ $v = D ] && q='2 { 0 }' && qdd='2 ( 1e-301 0 )'
    [ $v = E ] && hdr=no
    for side in of br; do
        d="$W/q_rs_${side}_$v"
        rscase "$d"
        rsfile "$d/0/uniform/rigidBodyMotionState" "$q" "$qdd" $hdr
        [ $v = B ] && gzip "$d/0/uniform/rigidBodyMotionState"
    done
    label="$v: q $q, qDdot $qdd, header $hdr$([ $v = B ] && echo ', .gz')"
    runof "$W/q_rs_of_$v"
    runbrae "$W/q_rs_br_$v" host
    python3 "$CMP" "$W/q_rs_of_$v" "$W/q_rs_br_$v" $(timedirs "$W/q_rs_of_$v") > "$W/cmp_q_rs_$v.txt" 2>&1
    judge "floatingObject start state $v" "$W/cmp_q_rs_$v.txt" "$BOUND_FO" "$W/q_rs_of_$v/log.interFoam" \
        && rbstate "$W/q_rs_of_$v" "$W/q_rs_br_$v" "$BOUND_RB_DTC" \
        && say "ARM Q  [host] a start state read back ($label): OpenFOAM's run" ok \
        || say "ARM Q  [host] a start state read back ($label): OpenFOAM's run" FAIL
done
finish "arm Q: a start state is read back as OpenFOAM reads it"
