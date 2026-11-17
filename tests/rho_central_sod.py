#!/usr/bin/env python3
"""The exact Riemann solution of a shock tube and the measurements tests/rho_central_sod_vs_openfoam.sh asserts.

  rho_central_sod.py analyse <case> <time dir name> [<C file>]
      reads the case's start fields (0/p, 0/T; the left- and rightmost cells are the two initial states), its
      thermophysicalProperties (Cp, molWeight -> gamma = Cp/(Cp - R)) and the time directory's rho, p, T, U, solves the
      Riemann problem exactly (Toro, Riemann Solvers and Numerical Methods for Fluid Dynamics, ch. 4: the pressure
      function, Newton on p*, then sampling), and prints one line per measurement:
        exact   p* u* rho*L rho*R shock-speed head tail                  the star region
        L1      rho p u                                                  mean |q - q_exact| over the tube, divided by
                                                                         the left state's rho, p and sound speed
        shock   x_brae x_exact dx                                        where rho crosses (rho*R + rho_R)/2, rightmost
        contact x_brae x_exact dx                                        where rho crosses (rho*L + rho*R)/2 between
                                                                         the rarefaction tail and the shock
        min     rho p T                                                  positivity
  The exact solution is sampled at the cell centres (C, from OpenFOAM's writeCellCentres, else a uniform 1-D
  blockMesh is assumed from the patch extents). The tube must lie along x with its diaphragm at x = 0.
"""
import math
import os
import re
import sys

import numpy

RR = 8314.46261815324   # 1e3*NA*k with OpenFOAM v2412's DimensionedConstants (NA 6.02214076e23, k 1.380649e-23)


def internal(path, width=1):
    s = open(path).read()
    m = re.search(r'internalField\s+(uniform|nonuniform)\s+', s)
    k = m.end()
    if m.group(1) == 'uniform':
        if s[k] == '(':
            e = s.index(')', k)
            return numpy.array([float(x) for x in s[k + 1:e].split()]), True
        return numpy.array([float(s[k:s.index(';', k)])]), True
    m = re.compile(r'List<\w+>\s*(\d+)\s*\(').match(s, k)
    n = int(m.group(1))
    body = s[m.end():]
    if width == 1:
        vals = body.split(None, n)[:n]
        return numpy.array([float(v) for v in vals]), False
    vals = re.findall(r'\(([^()]*)\)', body)[:n]
    return numpy.array([[float(x) for x in v.split()] for v in vals]), False


def thermo(case):
    s = open(os.path.join(case, 'constant/thermophysicalProperties')).read()
    W = float(re.search(r'molWeight\s+([0-9.eE+-]+)', s).group(1))
    Cp = float(re.search(r'\bCp\s+([0-9.eE+-]+)', s).group(1))
    R = RR/W
    return R, Cp/(Cp - R)


def exact_riemann(rL, uL, pL, rR, uR, pR, g):
    """Toro ch. 4: p*, u* and the star densities by Newton on the pressure function"""
    cL, cR = math.sqrt(g*pL/rL), math.sqrt(g*pR/rR)
    AL, AR = 2/((g + 1)*rL), 2/((g + 1)*rR)
    BL, BR = (g - 1)/(g + 1)*pL, (g - 1)/(g + 1)*pR

    def f(p, r, pk, ck, A, B):
        if p > pk:
            q = math.sqrt(A/(p + B))
            return (p - pk)*q, q*(1 - (p - pk)/(2*(B + p)))
        return 2*ck/(g - 1)*((p/pk)**((g - 1)/(2*g)) - 1), (p/pk)**(-(g + 1)/(2*g))/(r*ck)

    p = max(1e-8, 0.5*(pL + pR) - 0.125*(uR - uL)*(rL + rR)*(cL + cR))
    for _ in range(200):
        fL, dL = f(p, rL, pL, cL, AL, BL)
        fR, dR = f(p, rR, pR, cR, AR, BR)
        pn = p - (fL + fR + uR - uL)/(dL + dR)
        pn = max(pn, 1e-8)
        if abs(pn - p) <= 1e-15*0.5*(pn + p):
            p = pn
            break
        p = pn
    fL, _ = f(p, rL, pL, cL, AL, BL)
    fR, _ = f(p, rR, pR, cR, AR, BR)
    u = 0.5*(uL + uR) + 0.5*(fR - fL)
    gm = (g - 1)/(g + 1)
    # left: rarefaction when p* <= pL (the Sod case), right: shock when p* > pR
    if p <= pL:
        rsL = rL*(p/pL)**(1/g)
        head, tail = uL - cL, u - cL*(p/pL)**((g - 1)/(2*g))
    else:
        rsL = rL*((p/pL + gm)/(gm*p/pL + 1))
        head = tail = uL - cL*math.sqrt((g + 1)/(2*g)*p/pL + (g - 1)/(2*g))
    if p > pR:
        rsR = rR*((p/pR + gm)/(gm*p/pR + 1))
        S = uR + cR*math.sqrt((g + 1)/(2*g)*p/pR + (g - 1)/(2*g))
    else:
        rsR = rR*(p/pR)**(1/g)
        S = uR + cR
    return dict(p=p, u=u, rsL=rsL, rsR=rsR, S=S, head=head, tail=tail, cL=cL, cR=cR)


def sample(x, t, rL, uL, pL, rR, uR, pR, g, st):
    """the exact (rho, u, p) at x/t, for p* <= pL (left rarefaction) and p* > pR (right shock): the Sod pattern"""
    s = x/t
    rho, u, p = numpy.empty_like(x), numpy.empty_like(x), numpy.empty_like(x)
    cL = st['cL']
    for i, si in enumerate(s):
        if si < st['u']:
            if si < st['head']:
                rho[i], u[i], p[i] = rL, uL, pL
            elif si > st['tail']:
                rho[i], u[i], p[i] = st['rsL'], st['u'], st['p']
            else:
                c = 2/(g + 1)*(cL + (g - 1)/2*(uL - si))
                u[i] = 2/(g + 1)*(cL + (g - 1)/2*uL + si)
                rho[i] = rL*(c/cL)**(2/(g - 1))
                p[i] = pL*(c/cL)**(2*g/(g - 1))
        else:
            if si < st['S']:
                rho[i], u[i], p[i] = st['rsR'], st['u'], st['p']
            else:
                rho[i], u[i], p[i] = rR, uR, pR
    return rho, u, p


def crossing(x, q, level, lo, hi, rightmost=True):
    """x where q crosses level between x = lo and x = hi, linearly between cell centres"""
    idx = [i for i in range(len(x) - 1) if lo <= x[i] <= hi and (q[i] - level)*(q[i + 1] - level) <= 0
           and q[i] != q[i + 1]]
    if not idx:
        return float('nan')
    i = idx[-1] if rightmost else idx[0]
    return x[i] + (level - q[i])*(x[i + 1] - x[i])/(q[i + 1] - q[i])


def analyse(case, tname, cfile=None):
    t = float(tname)
    p0, _ = internal(os.path.join(case, '0/p'))
    T0, _ = internal(os.path.join(case, '0/T'))
    R, g = thermo(case)
    if cfile and os.path.exists(cfile):
        C, _ = internal(cfile, 3)
        x = C[:, 0]
    else:
        n = len(p0)
        pts = open(os.path.join(case, 'constant/polyMesh/points')).read()
        xs = [float(v.split()[0]) for v in re.findall(r'\(([^()]*)\)', pts.split('(', 1)[1])]
        a, b = min(xs), max(xs)
        x = a + (numpy.arange(n) + 0.5)*(b - a)/n
    order = numpy.argsort(x)
    x = x[order]
    pL, pR = p0[order][0], p0[order][-1]
    rL, rR = pL/(R*T0[order][0]), pR/(R*T0[order][-1])
    st = exact_riemann(rL, 0.0, pL, rR, 0.0, pR, g)
    d = os.path.join(case, tname)
    rho, _ = internal(os.path.join(d, 'rho'))
    p, _ = internal(os.path.join(d, 'p'))
    T, _ = internal(os.path.join(d, 'T'))
    U, _ = internal(os.path.join(d, 'U'), 3)
    rho, p, T, u = rho[order], p[order], T[order], U[order][:, 0]
    re_, ue, pe = sample(x, t, rL, 0.0, pL, rR, 0.0, pR, g, st)
    dx = numpy.gradient(x)
    L = numpy.sum(dx)
    l1 = lambda a, b, s: numpy.sum(numpy.abs(a - b)*dx)/L/s
    print('exact %.10e %.10e %.10e %.10e %.10e %.10e %.10e gamma %.12f' % (st['p'], st['u'], st['rsL'], st['rsR'],
          st['S'], st['head'], st['tail'], g))
    print('L1 %.10e %.10e %.10e' % (l1(rho, re_, rL), l1(p, pe, pL), l1(u, ue, st['cL'])))
    xs_exact = st['S']*t
    xc_exact = st['u']*t
    xs = crossing(x, rho, 0.5*(st['rsR'] + rR), st['u']*t, x[-1], rightmost=True)
    xc = crossing(x, rho, 0.5*(st['rsL'] + st['rsR']), st['tail']*t, xs_exact, rightmost=False)
    h = float(numpy.median(dx))
    print('shock %.10e %.10e %.10e' % (xs, xs_exact, h))
    print('contact %.10e %.10e %.10e' % (xc, xc_exact, h))
    print('min %.10e %.10e %.10e' % (rho.min(), p.min(), T.min()))


if __name__ == '__main__':
    if len(sys.argv) >= 4 and sys.argv[1] == 'analyse':
        analyse(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
    else:
        sys.exit(__doc__)
