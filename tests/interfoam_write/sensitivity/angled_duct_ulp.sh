#!/usr/bin/env bash
# The write gate: what RAS/angledDuct's bound is made of -- a pressure solve that never converges.
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
wcase angledDuct host device > "$W/ad.txt" 2>&1
o="$W/w_of_angledDuct"
[ -d "$o" ] || { say "angledDuct did not stage" FAIL; finish "angledDuct"; }
# The tutorial's p_rgh is GAMG with `maxIter 50`, and every solve of the run ends on that cap; the first one
# ENDS at a residual of 1.36 from 1.0. A tighter tolerance cannot move such a run (the twin of the other
# sensitivity files reads 2.1e-11 here), and round-off does.
[ "$(grep -c 'Solving for p_rgh,.*No Iterations 50' "$o/log.interFoam")" -ge 6 ] \
    && say "every p_rgh solve of OpenFOAM's run ends on the 50-cycle cap" ok \
    || say "every p_rgh solve of OpenFOAM's run ends on the 50-cycle cap" FAIL
# THE CONTROL: OpenFOAM itself with gravity moved by ONE ULP. MEASURED (2026-10-02): phi 7.4e-10, U 7.1e-10,
# against brae's 8.3e-10 host and 1.3e-09 device. brae is held to one decade of that.
u="$W/ad_ulp"
mkdir -p "$u"
cp -r "$o/0" "$o/constant" "$o/system" "$u/"
python3 - "$u/constant/g" <<'EOF_U' || say "angledDuct: gravity was not moved by one ulp" FAIL
import math, re, sys
s = open(sys.argv[1]).read()
m = re.search(r'value\s+\(\s*([-0-9.e]+)\s+([-0-9.e]+)\s+([-0-9.e]+)\s*\)', s)
v = [float(x) for x in m.groups()]
i = max(range(3), key=lambda k: abs(v[k]))
w = math.nextafter(v[i], 0.0)
assert w != v[i]
v[i] = w
open(sys.argv[1], 'w').write(s[:m.start()] + 'value           (%r %r %r)' % tuple(v) + s[m.end():])
EOF_U
runof "$u"
python3 "$CMP" "$o" "$u" $(timedirs "$o") > "$W/cmp_ad_ulp.txt" 2>&1
ul=$(worstof "$W/cmp_ad_ulp.txt")
bh=$(worstof "$W/cmp_w_angledDuct_host.txt")
bd=$(worstof "$W/cmp_w_angledDuct_device.txt")
closer 1e-10 "$ul" \
    && say "CONTROL  one ulp on gravity moves OpenFOAM's own files by $ul -- the case amplifies round-off" ok \
    || say "CONTROL  one ulp on gravity moves OpenFOAM's own files by $ul -- the case amplifies round-off" FAIL
ten=$(python3 -c "print('%.6e' % (10*float('$ul')))")
closer "$bh" "$ten" && closer "$bd" "$ten" \
    && say "brae (host $bh, device $bd) is within one decade of that" ok \
    || say "brae (host $bh, device $bd) is within one decade of that" FAIL
finish "angledDuct: the bound is a capped pressure solve's round-off"
