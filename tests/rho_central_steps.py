#!/usr/bin/env python3
"""A solver log, step by step, against OpenFOAM's: rhoCentralFoam's three lines a step under adjustTimeStep.

  rho_central_steps.py compare <brae log> <OpenFOAM log>
      reads `deltaT = ...` (setDeltaT.H), `Mean and max Courant Numbers = ... ...` (centralCourantNo.H) and
      `Time = ...` from each log and prints one line per measurement:
        steps         n_brae n_openfoam
        deltaT        identical_steps worst_relative step_of_worst     (identical: the printed digits are the same)
        names         identical_steps
        maxCo         worst_relative step_of_worst
        meanCo        worst_relative step_of_worst
        firstdiff     the first step whose deltaT or time name differs (0: none), then the time both reached before it
  rho_central_steps.py summary <log>
        steps n, distinct deltaT values, min and max deltaT, first deltaT
  Both logs must be written at the same precision (controlDict writePrecision sets OpenFOAM's Info precision).
"""
import re
import sys


def steps(path):
    out, cur = [], {}
    for line in open(path, errors='replace'):
        m = re.match(r'^deltaT = (\S+)', line)
        if m:
            cur = {'dt': m.group(1)}
            continue
        m = re.match(r'^Mean and max Courant Numbers = (\S+) (\S+)', line)
        if m:
            cur['mean'], cur['max'] = float(m.group(1)), float(m.group(2))
            continue
        m = re.match(r'^Time = (\S+)', line)
        if m:
            cur['t'] = m.group(1)
            out.append(cur)
            cur = {}
    return out


def rel(x, y):
    return abs(x - y)/max(abs(y), 1e-300)


def compare(b, o):
    a, r = steps(b), steps(o)
    print('steps %d %d' % (len(a), len(r)))
    n = min(len(a), len(r))
    if n == 0:
        return
    same = sum(1 for x, y in zip(a, r) if x.get('dt') == y.get('dt'))
    wd = max(((rel(float(x['dt']), float(y['dt'])) if 'dt' in x and 'dt' in y else float('inf')), i + 1)
             for i, (x, y) in enumerate(zip(a, r)))
    print('deltaT %d %.3e %d' % (same, wd[0], wd[1]))
    print('names %d' % sum(1 for x, y in zip(a, r) if x['t'] == y['t']))
    for k in ('max', 'mean'):
        w = max((rel(x[k], y[k]), i + 1) for i, (x, y) in enumerate(zip(a, r)))
        print('%sCo %.3e %d' % (k, w[0], w[1]))
    first = next((i + 1 for i, (x, y) in enumerate(zip(a, r)) if x.get('dt') != y.get('dt') or x['t'] != y['t']), 0)
    print('firstdiff %d %s' % (first, a[first - 2]['t'] if first > 1 else '-'))


def summary(p):
    s = steps(p)
    dts = [float(x['dt']) for x in s if 'dt' in x]
    print('steps %d distinct %d min %.17g max %.17g first %.17g' % (len(s), len(set(dts)), min(dts), max(dts), dts[0]))


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == 'compare':
        compare(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3 and sys.argv[1] == 'summary':
        summary(sys.argv[2])
    else:
        sys.exit(__doc__)
