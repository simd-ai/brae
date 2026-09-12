#!/usr/bin/env python3
# THE CASE'S OWN fvSolution/solvers ENTRIES REACH EVERY EQUATION -- one mutation per entry, and the
# written fields must MOVE.
#
# What this measures: OF's lduMatrix::solver reads tolerance/relTol/maxIter/minIter for EACH field from
# that field's own sub-dictionary (lduMatrixSolver.C:196-205). The rho mirror drivers used to read the
# energy tolerance from the turbulence slot (which nothing filled, so 1e-8/0 whatever the case said),
# take every equation's maxIter from p's entry, and forward minIter nowhere. None of that is visible
# in a comparison against OpenFOAM: with tight tolerances both codes agree to 1e-10 regardless, and
# with loose ones two different linear solvers legitimately stop at different points. So the oracle is
# NOT OpenFOAM. It is that mutating ONE entry changes the answer at a fixed iteration count -- and the
# mutations are chosen so the SOLVE does different work whatever the solver's convergence rate:
#
#   zero    tolerance 1e30 relTol 1  -> the initial residual passes, the solve does NOTHING, the
#           field keeps its pre-solve value. (A loosened relTol is NOT enough: the host's DILU-BiCGStab
#           already meets relTol 0.1 in one sweep on this case, so relTol 0.9 changes nothing.)
#   cap     tolerance 1e-300 relTol 0 with maxIter 1 against maxIter 5 -> one sweep against five.
#   floor   tolerance 1e30 relTol 1 with minIter 4 against minIter 0 -> four sweeps against none.
#
# applied to the energy entry and the k|epsilon entry, plus `zero` on p as the CONTROL: p's entry was
# read all along, so it must move before and after the fix, proving the comparison can see a solver
# control at all (non-vacuity). Fail-proof (the old assignments restored): every he/turb row reads
# 0.000e+00 against its partner -- the entry never reached the solve -- while the p control still moves.
#
# usage: rho_solver_entries.py <stagedCaseDir> <binary> <mirrorSelector> <workRoot> <heName>
import os
import re
import shutil
import subprocess
import sys

import numpy as np

MOVES = 1e-9   # relL2; a change that does not clear this did not reach the solve


def read(path):
    s = open(path).read()
    m = re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n?(\d+)\s*\n\(\n(.*?)\n\)\s*;', s, re.S)
    if not m:
        return None
    if m.group(1) == 'scalar':
        return np.array([float(x) for x in m.group(3).split()])
    return np.array([[float(c) for c in v.split()] for v in re.findall(r'\(([^)]*)\)', m.group(3))])


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def mutate(case, entry, body):
    """Replace the entry's body IN PLACE when the shipped file has that exact key (a second literal
    `p` or a second `"(k|epsilon)"` would not win: FoamDict keeps the shipped one), else add it as a
    literal, which resolves ahead of any regex."""
    f = os.path.join(case, 'system/fvSolution')
    s = open(f).read()
    pat = r'(?m)^(\s*)%s\s*\{[^{}]*\}' % re.escape(entry)
    if re.search(pat, s):
        s = re.sub(pat, lambda m: '%s%s { %s }' % (m.group(1), entry, body), s, count=1)
    else:
        s = re.sub(r'solvers\s*\{', 'solvers\n{\n    %s { %s }\n' % (entry, body), s, count=1)
    open(f, 'w').write(s)


def run(base, binary, selector, root, name, mutations):
    d = os.path.join(root, name)
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(base, d)
    for entry, body in mutations:
        mutate(d, entry, body)
    env = dict(os.environ, BRAE_RHOSIMPLEFOAM_MIRROR=selector)
    with open(os.path.join(d, 'run.log'), 'w') as log:
        rc = subprocess.call([binary, '-case', d], cwd=d, stdout=log, stderr=subprocess.STDOUT, env=env)
    if rc != 0:
        print('     %-12s did not run (exit %d)' % (name, rc))
        print(open(os.path.join(d, 'run.log')).read()[-600:])
        return None
    return d


def main():
    base, binary, selector, root, he = sys.argv[1:6]
    endT = None
    for line in open(os.path.join(base, 'system/controlDict')):
        m = re.search(r'\bendTime\s+([0-9.]+)\s*;', line)
        if m:
            endT = m.group(1)
    assert endT, 'no endTime in the staged controlDict'
    PB = 'solver PBiCGStab; preconditioner DILU;'
    KE = '"(k|epsilon)"'
    variants = {
        'base':      [],
        'pZero':     [('p', 'solver GAMG; smoother DICGaussSeidel; tolerance 1e30; relTol 1;')],
        'heZero':    [(he, PB + ' tolerance 1e30; relTol 1;')],
        'heCap1':    [(he, PB + ' tolerance 1e-300; relTol 0; maxIter 1;')],
        'heCap5':    [(he, PB + ' tolerance 1e-300; relTol 0; maxIter 5;')],
        'heFloor4':  [(he, PB + ' tolerance 1e30; relTol 1; minIter 4;')],
        'heFloor0':  [(he, PB + ' tolerance 1e30; relTol 1; minIter 0;')],
        'turbZero':  [(KE, PB + ' tolerance 1e30; relTol 1;')],
        'turbCap1':  [(KE, PB + ' tolerance 1e-300; relTol 0; maxIter 1;')],
        'turbCap5':  [(KE, PB + ' tolerance 1e-300; relTol 0; maxIter 5;')],
        'turbFloor4':[(KE, PB + ' tolerance 1e30; relTol 1; minIter 4;')],
        'turbFloor0':[(KE, PB + ' tolerance 1e30; relTol 1; minIter 0;')],
    }
    dirs = {}
    for name, muts in variants.items():
        d = run(base, binary, selector, root, name, muts)
        if d is None:
            return 1
        dirs[name] = d
    fields = {}
    for name, d in dirs.items():
        fields[name] = {f: read(os.path.join(d, endT, f)) for f in ('T', 'U', 'p', 'k')}
        if any(v is None for v in fields[name].values()):
            print('     %-12s wrote no %s/ fields' % (name, endT))
            return 1

    ok = True
    def moved(label, a, b, watch):
        nonlocal ok
        r = max(rel(fields[a][f], fields[b][f]) for f in watch)
        good = r > MOVES
        print('     %-42s max relL2(%s) vs %-8s %.3e   %s' % (label, '/'.join(watch), b, r, 'ok' if good else 'FAIL'))
        ok = ok and good

    moved('p zero-iteration (CONTROL, read before the fix too)', 'pZero',    'base',      ('p', 'U'))
    moved('%s tolerance/relTol reach the solve' % he,           'heZero',   'base',      ('T', 'U'))
    moved('%s maxIter reaches the solve (1 vs 5)' % he,        'heCap1',   'heCap5',    ('T', 'U'))
    moved('%s minIter reaches the solve (4 vs 0)' % he,        'heFloor4', 'heFloor0',  ('T', 'U'))
    moved('k|epsilon tolerance/relTol reach the solve',         'turbZero', 'base',      ('k',))
    moved('k|epsilon maxIter reaches the solve (1 vs 5)',      'turbCap1', 'turbCap5',  ('k',))
    moved('k|epsilon minIter reaches the solve (4 vs 0)',      'turbFloor4', 'turbFloor0', ('k',))
    # The printed controls are the case's, per equation.
    log = open(os.path.join(dirs['heCap1'], 'run.log')).read()
    pat = r'linear solver \(from fvSolution\): %s tol 1\.0e-300 relTol 0 maxIter 1 minIter 0' % re.escape(he)
    good = re.search(pat, log) is not None
    print('     %-42s %s' % ('the driver prints %s\'s own entry' % he, 'ok' if good else 'FAIL'))
    ok = ok and good
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
