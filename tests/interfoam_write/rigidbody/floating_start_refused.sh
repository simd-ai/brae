#!/usr/bin/env bash
# The write gate, a rigid body's state file: two start inputs OpenFOAM refuses or brae cannot follow.
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
q_of_case
for v in F G; do
    for side in of br; do
        d="$W/q_rs_${side}_$v"
        rscase "$d"
        if [ $v = F ]; then
            rsfile "$d/0/uniform/rigidBodyMotionState" '0 ( )' '2 { 0 }' yes
        else
            python3 - "$d/constant/dynamicMeshDict" <<'EOF_G'
import re, sys
p = sys.argv[1]
s = open(p).read()
s, n = re.subn(r'(\nrigidBodyMotionCoeffs\s*\n\{\n)', r'\1    q               (0.01 0.02);\n', s)
open(p, 'w').write(s)
sys.exit(0 if n == 1 else 1)
EOF_G
            [ $? -eq 0 ] || say "ARM Q  G: the coeffs' q was not staged in $d" FAIL
        fi
    done
    ( cd "$W/q_rs_of_$v" && interFoam > log.interFoam 2>&1 )
    orc=$?
    ( cd "$W/q_rs_br_$v" && "$BIN" -case . > log.brae 2>&1 )
    brc=$?
    if [ $v = F ]; then
        [ $orc -ne 0 ] && grep -q "do not have the same size" "$W/q_rs_of_F/log.interFoam" \
            && [ $brc -ne 0 ] && grep -q "holds a \`q\` of 0 where the chain has 2" "$W/q_rs_br_F/log.brae" \
            && say "ARM Q  F: an empty q stops OpenFOAM and brae alike, by name" ok \
            || { say "ARM Q  F: an empty q stops OpenFOAM and brae alike, by name [OF rc $orc, brae rc $brc]" FAIL; tail -2 "$W/q_rs_br_F/log.brae" | sed 's/^/      /'; }
    else
        last=$(timedirs "$W/q_rs_of_G" | awk '{print $NF}')
        [ $orc -eq 0 ] && [ -n "$last" ] && grep -qE "^q +2 \( ?0\.01" "$W/q_rs_of_G/$last/uniform/rigidBodyMotionState" \
            && [ $brc -ne 0 ] && grep -q "the motion coeffs set \`q\`" "$W/q_rs_br_G/log.brae" \
            && [ -z "$(timedirs "$W/q_rs_br_G")" ] \
            && say "ARM Q  G: OpenFOAM starts from the coeffs' q; brae refuses it by name, nothing written" ok \
            || { say "ARM Q  G: OpenFOAM starts from the coeffs' q; brae refuses it by name, nothing written [OF rc $orc, brae rc $brc]" FAIL; tail -2 "$W/q_rs_br_G/log.brae" | sed 's/^/      /'; }
    fi
done
finish "arm Q: an empty q and a coeffs' q are each handled by name"
