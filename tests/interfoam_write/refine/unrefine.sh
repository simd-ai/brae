#!/usr/bin/env bash
# The write gate: sixty steps of oscillatingBox through an unrefinement (Y3).
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
OB="$TUT/multiphase/interFoam/laminar/oscillatingBox"
if [ -d "$OB" ]; then
    stage_allrun "$OB" "$W/y3_of" 0.001 || say "ARM Y3 oscillatingBox: meshing failed" FAIL
    sed -i -E 's/^(endTime\s+)[^;]*;/\10.06;/' "$W/y3_of/system/controlDict"
    d="$W/y3_br"
    mkdir -p "$d"
    cp -r "$W/y3_of/0" "$W/y3_of/constant" "$W/y3_of/system" "$d/"
    runof "$W/y3_of"
    grep -q "Unrefined from" "$W/y3_of/log.interFoam" \
        && say "fixture witnesses: OpenFOAM unrefines within the sixty steps ($(grep -m1 'Unrefined from' "$W/y3_of/log.interFoam"))" ok \
        || say "fixture witnesses: OpenFOAM unrefines within the sixty steps" FAIL
    runbrae "$d" host
    [ "$(timedirs "$d")" = "$(timedirs "$W/y3_of")" ] \
        && say "ARM Y3 [host] sixty steps: exactly OpenFOAM's time directories" ok \
        || say "ARM Y3 [host] sixty steps: exactly OpenFOAM's time directories" FAIL
    python3 "$CMP" "$W/y3_of" "$d" $(timedirs "$W/y3_of") > "$W/cmp_y3.txt" 2>&1
    judge "oscillatingBox unrefine host" "$W/cmp_y3.txt" 2e-11 "$W/y3_of/log.interFoam" \
        && say "ARM Y3 [host] sixty steps through an unrefinement: every directory OpenFOAM's, every value within 2e-11" ok \
        || { say "ARM Y3 [host] sixty steps through an unrefinement: every directory OpenFOAM's, every value within 2e-11" FAIL; grep -v RESULT "$W/cmp_y3.txt" | grep BAD | head -6; }
fi
finish "arm Y3: brae unrefines where OpenFOAM does"
