#!/usr/bin/env python3
"""Block-by-block table: OpenFOAM (tools/timeRhoSimpleFoam, `PHASE ...` lines, max over ranks) against
brae (`BRAE_PHASE_TIME=1`, the `[phase]` report lines). Usage: phase_table.py <of log> <brae log> [skip] [brae ms/it]
brae prints no per-run wall line, so its whole-iteration wall comes from two timed runs,
(wall(100 it) - wall(10 it)) / 90, passed as the 4th argument.
The first `skip` iterations (default 1) are dropped from the OpenFOAM mean: iteration 1 pays first-touch
allocations in both codes and brae's report is over the whole run, so the comparison is a little kind to
OpenFOAM, never to brae."""
import re, sys
of_log, brae_log = sys.argv[1], sys.argv[2]
skip = int(sys.argv[3]) if len(sys.argv) > 3 else 1
brae_ms = float(sys.argv[4]) if len(sys.argv) > 4 else None
rows = []
for line in open(of_log):
    if not line.startswith('PHASE'): continue
    v = dict(re.findall(r'(\w+) ([0-9.eE+-]+)', line))
    sol = re.findall(r'\(solve ([0-9.eE+-]+)\)', line)
    rows.append({'UEqn': float(v['UEqn']), 'Usol': float(sol[0]), 'EEqn': float(v['EEqn']), 'Esol': float(sol[1]),
                 'pEqn': float(v['pEqn']), 'psol': float(sol[2]), 'turbulence': float(v['turbulence']),
                 'rest': float(v['rest']), 'total': float(v['total'])})
rows = rows[skip:]
n = len(rows)
of = {k: 1e3 * sum(r[k] for r in rows) / n for k in rows[0]}
b = {}
txt = open(brae_log).read()
m = re.search(r'\[phase\] over (\d+) iterations: UEqn [0-9.]+ s \(([0-9.]+) ms/it\), EEqn [0-9.]+ s \(([0-9.]+) ms/it\), '
              r'pEqn [0-9.]+ s \(([0-9.]+) ms/it\), turbulence [0-9.]+ s \(([0-9.]+) ms/it\); the four total [0-9.]+ s \(([0-9.]+) ms/it\)', txt)
b['UEqn'], b['EEqn'], b['pEqn'], b['turbulence'], b['four'] = (float(x) for x in m.groups()[1:])
m2 = re.search(r'linear solves: U [0-9.]+ s \(([0-9.]+) ms/it\), he [0-9.]+ s \(([0-9.]+) ms/it\), p [0-9.]+ s \(([0-9.]+) ms/it\)', txt)
b['Usol'], b['Esol'], b['psol'] = (float(x) for x in m2.groups())
mt = re.search(r'ExecutionTime = ([0-9.]+) s', txt.strip().split('\n')[-1]) or re.search(r'ExecutionTime = ([0-9.]+) s(?!.*ExecutionTime)', txt, re.S)
nIt = int(m.group(1))
total_wall = None
for l in reversed(txt.split('\n')):
    mm = re.search(r'ExecutionTime = ([0-9.]+) s', l)
    if mm: total_wall = float(mm.group(1)); break
b['total'] = brae_ms if brae_ms is not None else (1e3 * total_wall / nIt if total_wall else float('nan'))
b['rest'] = b['total'] - b['four']
print('%-22s %10s %10s %8s' % ('block (ms/iteration)', 'OF-20c', 'brae', 'brae/OF'))
for name, ko, kb in (('UEqn (assemble+solve)', 'UEqn', 'UEqn'), ('  of which U solve', 'Usol', 'Usol'),
                     ('EEqn (assemble+solve)', 'EEqn', 'EEqn'), ('  of which he solve', 'Esol', 'Esol'),
                     ('pEqn (assemble+solve)', 'pEqn', 'pEqn'), ('  of which p solve', 'psol', 'psol'),
                     ('turbulence (k+eps)', 'turbulence', 'turbulence'), ('rest of the iteration', 'rest', 'rest'),
                     ('whole iteration', 'total', 'total')):
    print('%-22s %10.1f %10.1f %8.2f' % (name, of[ko], b[kb], b[kb] / of[ko] if of[ko] > 0 else float('nan')))
print('(OpenFOAM: mean over %d iterations after skipping %d, max over ranks; brae: over %d iterations)' % (n, skip, nIt))
