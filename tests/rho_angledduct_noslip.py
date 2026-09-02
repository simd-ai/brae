#!/usr/bin/env python3
# porosityWall's U BC slip -> noSlip, so the CUDA arm can run the tutorial's fvOptions without meeting
# the tilted-plane refusal. One BC changes; the porosity and the two constraints are untouched.
import sys

p = sys.argv[1]
s = open(p).read()
i = s.index('porosityWall')
j = s.index('type', i)
s = s[:j] + s[j:].replace('type            slip;', 'type            noSlip;', 1)
open(p, 'w').write(s)
