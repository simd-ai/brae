# Sourced by sensitivity/*.sh: `worstof <compare output>` prints the worst field file's relative distance
# (the uniform/ directory left out), for a check that compares two distances and not a distance to a bound.
worstof()
{
    python3 - "$1" <<'EOF_P'
import json, sys
for line in open(sys.argv[1]):
    if line.startswith("RESULT "):
        r = json.loads(line[7:])
        f = {k: v["rel"] for k, v in r["files"].items() if "/uniform/" not in k}
        print("%.6e" % max(f.values()) if f and r["structure"] == 0 else "nan")
        break
else:
    print("nan")
EOF_P
}
# closer <a> <b>: a is a number and strictly below b
closer()
{
    python3 -c "import sys; a, b = float('$1'), float('$2'); sys.exit(0 if a == a and b == b and a < b else 1)"
}
# twin_check <key>: brae's two arms against OpenFOAM, and OpenFOAM against ITS TWIN -- itself with every
# tolerance one decade tighter (1e-14 for 1e-13). The check is that brae is closer to OpenFOAM than the twin
# is, on both arms.
twin_check()
{
    local key="$1" o t tw bh bd
    wcase "$key" host device > "$W/rest_$key.txt" 2>&1
    o="$W/w_of_$key"
    [ -d "$o" ] || { say "$key did not stage" FAIL; return; }
    t="$W/twin_$key"
    mkdir -p "$t"
    cp -r "$o/0" "$o/constant" "$o/system" "$t/"
    sed -i -E 's/(tolerance\s+)1e-13;/\11e-14;/' "$t/system/fvSolution"
    grep -q "1e-14;" "$t/system/fvSolution" || { say "$key: the twin's tolerances did not tighten" FAIL; return; }
    runof "$t"
    python3 "$CMP" "$o" "$t" $(timedirs "$o") > "$W/cmp_twin_$key.txt" 2>&1
    tw=$(worstof "$W/cmp_twin_$key.txt")
    bh=$(worstof "$W/cmp_w_${key}_host.txt")
    bd=$(worstof "$W/cmp_w_${key}_device.txt")
    closer "$bh" "$tw" && closer "$bd" "$tw" \
        && say "$key: brae (host $bh, device $bd) is closer to OpenFOAM than OpenFOAM's own twin ($tw)" ok \
        || say "$key: brae (host $bh, device $bd) is closer to OpenFOAM than OpenFOAM's own twin ($tw)" FAIL
}
