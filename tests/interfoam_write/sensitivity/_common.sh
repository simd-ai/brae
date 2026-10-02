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
