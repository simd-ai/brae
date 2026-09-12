#!/usr/bin/env python3
# The wall nut in a written V2 time directory: which patches carry a wall function, and whether the
# values there are actually nonzero. Before simpleFoamV2 dispatched on the 0/nut BC it ran the k-based
# nutk under every one, and on a nutUBlendedWallFunction case that produced exactly 0.000e+00 on the
# wall -- so "max > 0 on a wall-function patch" is the defect's own signature, not a proxy for it.
import re
import sys

path, want_type = sys.argv[1], sys.argv[2]
s = open(path).read()
b = s[s.index('boundaryField'):]
ok, seen = True, 0
for m in re.finditer(r'^    ([A-Za-z"][^\s]*)\s*\n\s*\{(.*?)\n    \}', b, re.S | re.M):
    name, body = m.group(1), m.group(2)
    ty = re.search(r'type\s+(\w+)', body)
    if not ty or ty.group(1) != want_type:
        continue
    v = re.search(r'nonuniform List<scalar>\s*\n?(\d+)\s*\n\((.*?)\)', body, re.S)
    if not v:
        print('     %-22s %s: no solved value list   FAIL' % (name, want_type)); ok = False; continue
    vals = [float(x) for x in v.group(2).split()]
    seen += 1
    hi = max(vals)
    print('     %-22s %-26s n=%d max=%.4e   %s' % (name, want_type, len(vals), hi, 'ok' if hi > 0 else 'FAIL'))
    ok = ok and hi > 0
if seen == 0:
    print('     no %s patch found -- the arm proves nothing   FAIL' % want_type); ok = False
sys.exit(0 if ok else 1)
