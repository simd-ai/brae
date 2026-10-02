#!/usr/bin/env bash
# The write gate: what laminar/waves/mangroveInteraction's loose tutorial bound is made of.
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
wcase mangroveInteraction of
[ -d "$W/w_of_mangroveInteraction" ] || { say "mangroveInteraction did not stage" FAIL; finish "mangrove branch"; }
# The tutorial row holds k to 8e-02. MEASURED (2026-10-02): the whole of it is SIX faces of the `top` patch
# (inletOutlet on k and epsilon) whose flux is 1e-20 in both codes with opposite sign -- |phi| there agrees to
# 1.5e-19 on a field of 4e-06 -- so each code takes the inflow or the outflow branch by the sign of round-off,
# and 1509 cells of k follow. THE SAME CASE WITH THAT ONE SWITCH REMOVED (k and epsilon zeroGradient on `top`,
# in both codes) is held here at round-off: MEASURED host 5.2e-12 (phi), k 2.0e-13; device 6.1e-12.
BOUND_MG_HOST=5.2e-11
BOUND_MG_DEVICE=6.1e-11
d="$W/mg_of"
mkdir -p "$d"
cp -r "$W/w_of_mangroveInteraction/0" "$W/w_of_mangroveInteraction/constant" "$W/w_of_mangroveInteraction/system" "$d/"
python3 - "$d" <<'EOF_E' || say "mangroveInteraction: the zeroGradient staging did not apply" FAIL
import re, sys
d = sys.argv[1]
for f in ("k", "epsilon"):
    p = d + "/0/" + f
    s = open(p).read()
    assert re.search(r"\n\s+top\s*\{[^}]*inletOutlet", s), f + ": top is no longer inletOutlet"
    s, k = re.subn(r"(\n\s+top\s*\{)[^}]*\}", r"\1\n        type            zeroGradient;\n    }", s)
    assert k == 1, f
    open(p, "w").write(s)
EOF_E
runof "$d"
mt=$(timedirs "$d")
for arm in host device; do
    e="$W/mg_br_$arm"
    mkdir -p "$e"
    cp -r "$d/0" "$d/constant" "$d/system" "$e/"
    runbrae "$e" $arm
    python3 "$CMP" "$d" "$e" $mt > "$W/cmp_mg_$arm.txt" 2>&1
    b=$BOUND_MG_HOST
    [ $arm = device ] && b=$BOUND_MG_DEVICE
    judge "mangroveInteraction, top zeroGradient, $arm" "$W/cmp_mg_$arm.txt" "$b" "$d/log.interFoam" \
        && say "[$arm] mangroveInteraction without the round-off switch: every value within $b" ok \
        || { say "[$arm] mangroveInteraction without the round-off switch: every value within $b" FAIL; grep -v RESULT "$W/cmp_mg_$arm.txt" | grep -B1 "^      " | head -8; }
done
# ...and the switch IS what the tutorial row's distance is: as shipped, the host's k is orders above this
wcase mangroveInteraction host > "$W/mg_shipped.txt" 2>&1
ws=$(worstof "$W/cmp_w_mangroveInteraction_host.txt")
closer 1e-6 "$ws" \
    && say "CONTROL  as shipped the same arm reads $ws -- the switch, not the solve" ok \
    || say "CONTROL  as shipped the same arm reads $ws -- the switch, not the solve" FAIL
finish "mangroveInteraction: the loose bound is one round-off switch"
