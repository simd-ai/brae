#!/usr/bin/env bash
# The write gate, a rigid body's state file: DTCHullMoving, a body that moves.
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
wcase DTCHullMovingCoarse host
[ -d "$W/w_br_DTCHullMovingCoarse_host" ] || { say "ARM Q  DTCHullMoving did not run" FAIL; finish "arm Q"; }
rbstate "$W/w_of_DTCHullMovingCoarse" "$W/w_br_DTCHullMovingCoarse_host" "$BOUND_RB_DTC" \
    && say "ARM Q  [host] DTCHullMoving's rigidBodyMotionState: OpenFOAM's text, each entry within $BOUND_RB_DTC" ok \
    || say "ARM Q  [host] DTCHullMoving's rigidBodyMotionState: OpenFOAM's text, each entry within $BOUND_RB_DTC" FAIL
python3 - "$W/w_of_DTCHullMovingCoarse" <<'PY' && say "fixture witnesses: OpenFOAM's DTCHullMoving body moves (paren lists, pointDisplacement non-zero)" ok \
                                         || say "fixture witnesses: OpenFOAM's DTCHullMoving body moves (paren lists, pointDisplacement non-zero)" FAIL
import os, re, sys
d = sys.argv[1]
last = sorted([t for t in os.listdir(d) if re.match(r'^[0-9.e+-]+$', t) and t != '0'], key=float)[-1]
st = open('%s/%s/uniform/rigidBodyMotionState' % (d, last)).read()
pd = open('%s/%s/pointDisplacement' % (d, last)).read()
body = pd[pd.find('internalField'):pd.find('boundaryField')]
dmax = max(abs(float(x)) for x in re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', body.split('(', 1)[1]))
print('      %s: q lists %s, max|pointDisplacement| %.3e' % (last, 'paren' if re.search(r'^q\s+\d+ \(', st, re.M) else 'NOT paren', dmax))
sys.exit(0 if re.search(r'^q\s+\d+ \(', st, re.M) and dmax > 1e-7 else 1)
PY
finish "arm Q: DTCHullMoving's state file is OpenFOAM's, entry by entry"
