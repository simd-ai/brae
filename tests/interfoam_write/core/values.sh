#!/usr/bin/env bash
# The write gate on damBreak: every value and every keyword (arm D), with its two controls.
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
core_base
for arm in $ARMS; do
    judge "$arm" "$W/cmp_$arm.txt" "$BOUND_FIELDS" \
        && say "ARM D  [$arm] every file's structure is OpenFOAM's, every value within $BOUND_FIELDS" ok \
        || { say "ARM D  [$arm] every file's structure is OpenFOAM's, every value within $BOUND_FIELDS" FAIL; grep -v RESULT "$W/cmp_$arm.txt" | grep -B1 "^      " | head -20; }
done

# CONTROL: 0/ handed in as brae's 0.05
mkdir -p "$W/ctl0"; cp -r "$W/of/0" "$W/ctl0/0.05"
mkdir -p "$W/ctl0/0.05/uniform"; cp -r "$W/br_host/0.05/uniform/." "$W/ctl0/0.05/uniform/"
sed -i 's/^location .*/location    "0.05";/' "$W"/ctl0/0.05/* 2>/dev/null
python3 "$CMP" "$W/of" "$W/ctl0" 0.05 > "$W/cmp_ctl0.txt" 2>&1
judge "control 0/" "$W/cmp_ctl0.txt" "$BOUND_FIELDS" > /dev/null \
    && say "CONTROL  0/ handed in as brae's 0.05 FAILS D" FAIL \
    || say "CONTROL  0/ handed in as brae's 0.05 FAILS D" ok

# CONTROL: a deleted keyword
mkdir -p "$W/ctlk"; cp -r "$W/br_host/0.05" "$W/ctlk/"
sed -i '/inletValue/d' "$W/ctlk/0.05/alpha.water"
python3 "$CMP" "$W/of" "$W/ctlk" 0.05 > "$W/cmp_ctlk.txt" 2>&1 \
    && say "CONTROL  brae's alpha.water without its inletValue FAILS D" FAIL \
    || say "CONTROL  brae's alpha.water without its inletValue FAILS D" ok
finish "arm D: every file's structure and values are OpenFOAM's"
