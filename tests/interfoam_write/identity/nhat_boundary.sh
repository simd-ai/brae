#!/usr/bin/env bash
# The write gate: the device loop's boundary-only nHat against calculateK whole, on a contact-angle case.
. "$(dirname "$0")/../lib.sh"
# The alpha hooks hand the device the boundary normal; since 2026-10-03 they take it from the boundary's own
# cells (interfaceProps::calculateNHatBoundary) where calculateK took the whole mesh -- 37 ms a call, four
# calls a step on RAS/DTCHull, where every written file stayed byte-identical. capillaryRise is the case that
# can see it: its contact angle writes alpha's wall gradient back from that normal. MEASURED there: one ulp
# in a few boundary cells' gradient (the compiler fuses the two loops' multiply-adds differently on this
# aarch64 build) and 2.3e-16 at most in the written fields. Held to 1e-15.
BOUND_NHAT_IDENTITY=1e-15
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase capillaryRise of > "$W/nhat_stage.txt" 2>&1
o="$W/w_of_capillaryRise"
[ -d "$o" ] || { say "capillaryRise did not stage" FAIL; finish "nHat identity"; }
for v in boundary full; do
    e="$W/nhat_$v"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    if [ $v = full ]; then
        runbrae "$e" device BRAE_CONTROL_NHAT_FULL=1
    else
        runbrae "$e" device
    fi
done
grep -q "nHat: the alpha hooks take the boundary normal" "$W/nhat_boundary/log.brae" \
    && ! grep -q "nHat: the alpha hooks take the boundary normal" "$W/nhat_full/log.brae" \
    && say "the default run took the boundary-only normal, the control run calculateK whole" ok \
    || say "the default run took the boundary-only normal, the control run calculateK whole" FAIL
python3 "$CMP" "$W/nhat_full" "$W/nhat_boundary" $(timedirs "$o") > "$W/cmp_nhat.txt" 2>&1
judge "nHat boundary-only against whole" "$W/cmp_nhat.txt" "$BOUND_NHAT_IDENTITY" "$o/log.interFoam" \
    && say "[device] every written file of the two runs within $BOUND_NHAT_IDENTITY" ok \
    || { say "[device] every written file of the two runs within $BOUND_NHAT_IDENTITY" FAIL; grep -v RESULT "$W/cmp_nhat.txt" | head -8; }
# THE CONTROL: the boundary-only gradient without its patch faces' terms (BRAE_CONTROL_NHAT_NO_PATCH_TERMS=1)
# must leave that bound
e="$W/nhat_broken"
mkdir -p "$e"
cp -r "$o/0" "$o/constant" "$o/system" "$e/"
runbrae "$e" device BRAE_CONTROL_NHAT_NO_PATCH_TERMS=1
python3 "$CMP" "$W/nhat_full" "$e" $(timedirs "$o") > "$W/cmp_nhat_broken.txt" 2>&1
judge "nHat without patch terms" "$W/cmp_nhat_broken.txt" "$BOUND_NHAT_IDENTITY" "$o/log.interFoam" > "$W/nhat_broken.txt" \
    && { say "CONTROL  a gradient without the patch terms leaves the bound" FAIL; cat "$W/nhat_broken.txt"; } \
    || say "CONTROL  a gradient without the patch terms leaves the bound" ok
finish "the boundary-only nHat is calculateK's boundary"
