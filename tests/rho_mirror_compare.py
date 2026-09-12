#!/usr/bin/env python3
# The CUDA mirror's written time directory against OpenFOAM's and against the host mirror's.
#
# Its own file rather than a heredoc inside the gate script: the gate already nests two levels of
# heredoc for the case-dictionary rewrites, and a third that itself contains regex escapes is how a
# comparison silently becomes a no-op that reports ok.
#
# BOUNDS from measurement on rhoBox at 200 iterations, at ~30x:
#   CUDA vs OpenFOAM     p 3.00e-12  T 1.27e-09  U 4.45e-10  rho 1.19e-09
#   CUDA vs host mirror  p 2.83e-12  T 1.32e-09  U 4.62e-10  rho 1.24e-09
# Looser than the host arm's own 1e-11 by exactly the linear solvers between the two paths (AMG-PCG
# against BiCGStab). They tighten as the port improves; they do not loosen.
import os
import re
import sys

import numpy as np

# The rhoBox defaults. A fixture whose own offset is different overrides them per field with
# CUDA_<FIELD>_BOUND, and `--host-only` drops the OpenFOAM column for a case where the HOST mirror
# carries an offset of its own -- the device arm's job there is to reproduce the host, not to beat it,
# and a bound wide enough to cover the host's offset would stop measuring the device at all.
DEFAULT_BOUNDS = {'p': 1e-10, 'T': 4e-08, 'U': 1.5e-08, 'rho': 4e-08}
BOUNDS = {f: float(os.environ.get('CUDA_%s_BOUND' % f.upper(), b))
          for f, b in DEFAULT_BOUNDS.items()}


def read(path):
    try:
        s = open(path).read()
    except OSError:
        return None
    m = re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n?(\d+)\s*\n\(\n(.*?)\n\)\s*;',
                  s, re.S)
    if not m:
        return None
    if m.group(1) == 'scalar':
        return np.array([float(x) for x in m.group(3).split()])
    return np.array([[float(c) for c in v.split()] for v in re.findall(r'\(([^)]*)\)', m.group(3))])


def main():
    ok = True
    columns = [('OpenFOAM', 'OF_DIR'), ('the host mirror', 'HOST_DIR')]
    if '--host-only' in sys.argv:
        columns = [c for c in columns if c[1] != 'OF_DIR']
    for tag, env in columns:
        for field, bound in BOUNDS.items():
            a = read(os.path.join(os.environ['BRAE_DIR'], field))
            b = read(os.path.join(os.environ[env], field))
            if a is None or b is None or a.shape != b.shape:
                print('     cuda %-5s MISSING or shape mismatch against %s   FAIL' % (field, tag))
                ok = False
                continue
            r = float(np.linalg.norm(a - b) / np.linalg.norm(b))
            good = r < bound
            print('     cuda %-5s vs %-16s %.4e   (bound %.1e)   %s'
                  % (field, tag, r, bound, 'ok' if good else 'FAIL'))
            ok = ok and good
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
