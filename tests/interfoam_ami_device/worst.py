# worst.py <foam_time_compare output>: `<structure> <worst rel> <file> <files>` over the field files, the
# uniform/ directory excluded (its cumulativeContErr is a signed sum whose own value is no scale).
import json
import sys

for line in open(sys.argv[1]):
    if line.startswith("RESULT "):
        r = json.loads(line[7:])
        f = {k: v["rel"] for k, v in r["files"].items() if "/uniform/" not in k}
        k = max(f, key=f.get)
        print("%d %.3e %s %d" % (r["structure"], f[k], k, len(f)))
        break
else:
    print("1 nan none 0")
