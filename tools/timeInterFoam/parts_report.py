# parts_report.py <label> <cells> <N> <K0>: the per-part times of parts.sh's runs, ms a step
import os, re, sys
B = '/home/ghost/cudafoam/runs/bench/parts'
label, cells, N, K0 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
PARTS = ['deltaT', 'mesh', 'alpha', 'momentum', 'pressure', 'turbulence', 'write']

def of_parts(arm):
    p = '%s/p_%s_%s/log.of' % (B, label, arm)
    rows = {}
    if not os.path.exists(p):
        return None
    for line in open(p, errors='replace'):
        if line.startswith('TIMED '):
            f = line.split()
            rows[int(f[1])] = {f[i]: float(f[i + 1]) for i in range(2, len(f) - 1, 2)}
    if not rows:
        return None
    last = max(rows) - 1
    if last - K0 < 5 or K0 not in rows:
        return None
    return {k: 1000*(rows[last][k] - rows[K0][k])/(last - K0) for k in PARTS}

def brae_table(arm):
    p = '%s/p_%s_brae_%s/log.brae' % (B, label, arm)
    if not os.path.exists(p):
        return None, 0, {}
    steps, top, nested, inside = 0, {}, {}, False
    for line in open(p, errors='replace'):
        m = re.search(r'phase time over (\d+) steps', line)
        if m:
            steps = int(m.group(1))
            continue
        if 'inside those phases' in line:
            inside = True
            continue
        m = re.match(r'\s+(.*?)\s+([0-9.]+) ms/step', line)
        if m and steps:
            name = m.group(1).strip()
            if name == 'ALL PHASES':
                top[name] = float(m.group(2))
            elif inside:
                nested[name] = float(m.group(2))
            else:
                top[name] = float(m.group(2))
    return top, steps, nested

def brae_parts():
    full, nf, nestF = brae_table('full')
    half, nh, nestH = brae_table('half')
    if not full or not half or nf <= nh:
        return None, None, 'brae: ' + ('no phase table' if not full or not half else 'run lengths %d and %d' % (nf, nh))
    per = {k: (full[k]*nf - half.get(k, 0.0)*nh)/(nf - nh) for k in full}
    nest = {k: (nestF[k]*nf - nestH.get(k, 0.0)*nh)/(nf - nh) for k in nestF}
    return per, nest, '%d and %d steps' % (nf, nh)

o1, o20 = of_parts('of1'), of_parts('of20')
bp, bn, note = brae_parts()
# brae's phases onto OpenFOAM's parts (interFoam.C): the surface-tension force and the mixture belong to UEqn/pEqn
MAP = [('deltaT', ['0 between steps', '0 before setRDeltaT', '0 setRDeltaT']),
       ('mesh', ['0 step preparation']),
       ('alpha', ['1 alpha step']),
       ('momentum', ['2 interface forces', '3 momentum matrix', '4 momentum predictor']),
       ('pressure', ['5 pressure correctors']),
       ('turbulence', ['7 turbulence closure', '8 grad(U)']),
       ('write', ['9 write', '6 after the step', '9 end of the outer'])]
print('%s  (%s cells)   ms a step' % (label, cells))
print('  %-12s %10s %10s %10s %9s %9s' % ('part', 'OF 1 core', 'OF 20', 'brae', 'vs 20', 'vs 1'))
tot = [0.0, 0.0, 0.0]
for part, keys in MAP:
    b = sum(v for k, v in (bp or {}).items() if any(k.startswith(x) for x in keys)) if bp else None
    a1 = o1[part] if o1 else None
    a20 = o20[part] if o20 else None
    for i, v in enumerate((a1, a20, b)):
        tot[i] += v or 0.0
    f = lambda v: '%10.1f' % v if v is not None else '         -'
    r = lambda a, c: '%8.2fx' % (a/c) if a and c and c > 0.05 else '        -'
    print('  %-12s %s %s %s %s %s' % (part, f(a1), f(a20), f(b), r(a20, b), r(a1, b)))
print('  %-12s %10.1f %10.1f %10.1f %8.2fx %8.2fx' % ('TOTAL', tot[0], tot[1], tot[2],
      tot[1]/tot[2] if tot[2] else 0, tot[0]/tot[2] if tot[2] else 0))
print('  (%s)' % note)
if bn:
    print('  inside brae, ms a step:')
    for k, v in sorted(bn.items(), key=lambda kv: -kv[1])[:14]:
        if v >= 0.3:
            print('    %-48s %8.1f' % (k, v))
