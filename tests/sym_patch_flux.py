#!/usr/bin/env python3
# THE FLUX THROUGH A SYMMETRY (or slip) FACE MUST BE IDENTICALLY ZERO. Not small, not converged-small:
# the constraint is no penetration, so phi on that patch is zero at EVERY iteration by construction.
#
# That is why this is the oracle for the tilted-plane arms rather than a field comparison against
# OpenFOAM. A stale symmetry refValue is a TRANSIENT defect: brae's device arm leaked 7.13e-04 of flux
# through rhoBoxSym's `slant` at iteration 2 and still converged to the same fixed point as the fixed
# code (U 7.2e-11 against the host mirror either way, at 200 iterations). A converged comparison cannot
# see it. This can, at iteration 1, with 0 as the oracle -- see converged-not-iteration-count.
#
# Reference readings on rhoBoxSym (max |phi| over the patch, iteration 2):
#   real OpenFOAM            8.97e-17     its own round-off through the transform
#   brae host mirror         1.66e-19
#   brae CUDA mirror         1.96e-19
#   brae CUDA, refresh out   7.13e-04     the fail-proof
#
# usage: sym_patch_flux.py <caseDir> <patchName> <bound> <time> [<time> ...]
import os
import re
import sys


def patch_flux(path, patch):
    s = open(path).read()
    i = s.index(patch)
    j = s.index('nonuniform', i)
    k = s.index('(', j)
    e = s.index(')', k)
    return [float(x) for x in s[k + 1:e].split()]


def main():
    case, patch, bound = sys.argv[1], sys.argv[2], float(sys.argv[3])
    times = sys.argv[4:]
    ok = True
    for t in times:
        f = os.path.join(case, t, 'phi')
        if not os.path.exists(f):
            print('     t=%-4s phi not written   FAIL' % t)
            ok = False
            continue
        v = patch_flux(f, patch)
        worst = max(abs(x) for x in v)
        good = worst < bound
        print('     t=%-4s max|phi| on %-14s %.4e   (bound %.1e)   %s'
              % (t, patch, worst, bound, 'ok' if good else 'FAIL'))
        ok = ok and good
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
