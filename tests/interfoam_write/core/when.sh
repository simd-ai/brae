#!/usr/bin/env bash
# The write gate on damBreak: WHEN brae writes and WHICH files (arms A, B, C).
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
core_base
# the fixture must witness: a developed flow, and an atmosphere with inflow AND outflow faces
python3 - "$W/of/0.1" <<'PY' && say "fixture witnesses: max|U| > 0.1, atmosphere phi both signs" ok \
                             || say "fixture witnesses: max|U| > 0.1, atmosphere phi both signs" FAIL
import re, sys
d = sys.argv[1]
U = open(d + '/U').read()
uint = U[U.find('internalField'):U.find('boundaryField')]
mags = [sum(float(c)**2 for c in m.split())**0.5 for m in re.findall(r'\(([^()]*)\)', uint)]
phi = open(d + '/phi').read()
a = phi.find('atmosphere', phi.find('boundaryField'))
seg = phi[phi.find('(', a) + 1:phi.find(')', a)]
vals = [float(x) for x in seg.split()]
print('      max|U| %.3f, atmosphere phi in [%.3e, %.3e]' % (max(mags), min(vals), max(vals)))
sys.exit(0 if max(mags) > 0.1 and min(vals) < 0 < max(vals) else 1)
PY

for arm in $ARMS; do
    b=$(timedirs "$W/br_$arm")
    [ "$b" = "$o" ] && say "ARM A  [$arm] brae writes exactly OpenFOAM's directories, none at endTime [$b]" ok \
                    || say "ARM A  [$arm] brae writes exactly OpenFOAM's directories, none at endTime [$b]" FAIL
    for t in 0.05 0.1; do
        [ "$(filesets "$W/of" $t)" = "$(filesets "$W/br_$arm" $t)" ] \
            && say "ARM B  [$arm] $t/ holds OpenFOAM's file set" ok \
            || { say "ARM B  [$arm] $t/ holds OpenFOAM's file set" FAIL; echo "      OF:   $(filesets "$W/of" $t)"; echo "      brae: $(filesets "$W/br_$arm" $t)"; }
    done
    python3 - "$W/cmp_$arm.txt" "$BOUND_TIME" <<'PY' && say "ARM C  [$arm] uniform/time: name, index exact; value, deltaT, deltaT0 within $BOUND_TIME" ok \
                                                   || say "ARM C  [$arm] uniform/time: name, index exact; value, deltaT, deltaT0 within $BOUND_TIME" FAIL
import json, sys
r = json.loads([l for l in open(sys.argv[1]) if l.startswith('RESULT ')][-1][7:])
bad = 0
for t in ('0.05', '0.1'):
    e = r['files'].get(t + '/uniform/time')
    print('      %s/uniform/time: structure %s, worst rel %.3e' % (t, e and e['structure'], e['rel'] if e else -1))
    bad += (not e) or (not e['structure']) or e['rel'] > float(sys.argv[2])
sys.exit(1 if bad else 0)
PY
done
finish "arms A-C: brae writes OpenFOAM's directories, file sets and uniform/time"
