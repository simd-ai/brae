#!/usr/bin/env bash
# The write gate on damBreak: RAS/damBreak under kEpsilon (arm G).
. "$(dirname "$0")/../lib.sh"
# G: RAS/damBreak (kEpsilon), adjustable 0.05 to 0.06
stage "$RAS" "$W/of_r" 0.06 adjustable 0.05 0 || exit 1
runof "$W/of_r"
[ "$(timedirs "$W/of_r")" = "0.05 " ] || { say "premise: OpenFOAM RAS writes {0.05} [$(timedirs "$W/of_r")]" FAIL; }
for arm in $ARMS; do
    stage "$RAS" "$W/br_r_$arm" 0.06 adjustable 0.05 0 || exit 1
    runbrae "$W/br_r_$arm" "$arm"
    [ "$(timedirs "$W/br_r_$arm")" = "0.05 " ] && [ "$(filesets "$W/of_r" 0.05)" = "$(filesets "$W/br_r_$arm" 0.05)" ] \
        && say "ARM G  [$arm] RAS: OpenFOAM's directories and file set (k, epsilon, nut)" ok \
        || say "ARM G  [$arm] RAS: OpenFOAM's directories and file set (k, epsilon, nut)" FAIL
    python3 "$CMP" "$W/of_r" "$W/br_r_$arm" 0.05 > "$W/cmp_r_$arm.txt" 2>&1
    judge "RAS $arm" "$W/cmp_r_$arm.txt" "$BOUND_FIELDS" \
        && say "ARM G  [$arm] RAS: every file's structure is OpenFOAM's, every value within $BOUND_FIELDS" ok \
        || { say "ARM G  [$arm] RAS: every file's structure is OpenFOAM's, every value within $BOUND_FIELDS" FAIL; grep -v RESULT "$W/cmp_r_$arm.txt" | grep -B1 "^      " | head -20; }
done
finish "arm G: the closure's fields are written as OpenFOAM writes them"
