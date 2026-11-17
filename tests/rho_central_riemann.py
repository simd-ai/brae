#!/usr/bin/env python3
"""Riemann problems for the rhoCentralFoam gates (milestone F), on OpenFOAM's shockTube tutorial (x in [-5, 5],
diaphragm at 0): the cases and their exact solutions (no OpenFOAM needed).

  rho_central_riemann.py setup <problem> <case>
      rewrites a staged copy of shockTube (before setFields) for one of Toro's problems (Riemann Solvers and Numerical
      Methods for Fluid Dynamics, 3rd ed., sec. 4.3.3, table 4.1), scaled to the tutorial's tube ten times as long
      (times ten times as long): a perfect gas with R = 1 and gamma = 1.4 (molWeight = RR, Cp = 3.5), the left and right
      (rho, u, p) as setFields' U, T = p/rho and p, endTime and adjustable writes every endTime/4
        sod     (1, 0, 1)            | (0.125, 0, 0.1)    t = 2.0    Toro's test 1
        lax     (0.445, 0.698, 3.528) | (0.5, 0, 0.571)   t = 1.3    Lax's problem
        123     (1, -2, 0.4)          | (1, 2, 0.4)       t = 1.5    Toro's test 2: two rarefactions, near vacuum
        strong  (1, 0, 1000)          | (1, 0, 0.01)      t = 0.12   Toro's test 3: a strong shock
  rho_central_riemann.py exact <case> <time dir>
      the exact solution at the cell centres (all four wave patterns, sec. 4.5) against the time directory's rho, U, p:
      `exact pstar <p*> ustar <u*>` then `L1 rho <..> u <..> p <..>`, mean |q - q_exact| over the cells divided by the
      largest |q_exact|
  rho_central_riemann.py problems
      the problems' names
"""
import math
import os
import re
import sys

import numpy as np

RR = 8314.46261815324   # OpenFOAM v2412's universal gas constant, J/(kmol K)
G = 1.4
PROBLEMS = {
    "sod": ((1.0, 0.0, 1.0), (0.125, 0.0, 0.1), 2.0),
    "lax": ((0.445, 0.698, 3.528), (0.5, 0.0, 0.571), 1.3),
    "123": ((1.0, -2.0, 0.4), (1.0, 2.0, 0.4), 1.5),
    "strong": ((1.0, 0.0, 1000.0), (1.0, 0.0, 0.01), 0.12),
}


def edit(path, pattern, repl, count=1):
    s = open(path).read()
    t, n = re.subn(pattern, repl, s, count=count, flags=re.S | re.M)
    if n != count:
        raise SystemExit(f"rho_central_riemann: {path}: expected {count} edit(s) of /{pattern}/, made {n}")
    open(path, "w").write(t)


def setup(problem, case):
    (rL, uL, pL), (rR, uR, pR), t = PROBLEMS[problem]
    tp = f"{case}/constant/thermophysicalProperties"
    edit(tp, r"(molWeight\s+)[^;]+;", rf"\g<1>{RR!r};")
    edit(tp, r"(Cp\s+)[^;]+;", r"\g<1>3.5;")
    edit(tp, r"(Hf\s+)[^;]+;", r"\g<1>0;")
    sf = f"{case}/system/setFieldsDict"
    s = open(sf).read()
    a = s.index("defaultFieldValues")
    b = s.index("regions")
    s = (s[:a] + "defaultFieldValues\n(\n    volVectorFieldValue U (%r 0 0)\n    volScalarFieldValue T %r\n"
         "    volScalarFieldValue p %r\n);\n\n" % (uL, pL / rL, pL) + s[b:])
    c = s.index("fieldValues", s.index("regions"))
    d = s.index(");", c)
    s = (s[:c] + "fieldValues\n        (\n            volVectorFieldValue U (%r 0 0)\n"
         "            volScalarFieldValue T %r\n            volScalarFieldValue p %r\n        " % (uR, pR / rR, pR) + s[d:])
    open(sf, "w").write(s)
    cd = f"{case}/system/controlDict"
    edit(cd, r"^(endTime\s+)\S+;", rf"\g<1>{t!r};")
    edit(cd, r"^(writeControl\s+)\S+;", r"\g<1>adjustable;")
    edit(cd, r"^(writeInterval\s+)\S+;", rf"\g<1>{t / 4!r};")
    edit(cd, r"^(deltaT\s+)\S+;", r"\g<1>1e-05;")


# ---- the exact solution (Toro ch. 4) -------------------------------------------------------------------------------
def star(rL, uL, pL, rR, uR, pR, g=G):
    cL, cR = math.sqrt(g * pL / rL), math.sqrt(g * pR / rR)
    if 2 * (cL + cR) / (g - 1) <= uR - uL:
        raise SystemExit("rho_central_riemann: the initial states make a vacuum")

    def f(p, r, pk, ck):
        if p > pk:   # shock
            A, B = 2 / ((g + 1) * r), (g - 1) / (g + 1) * pk
            q = math.sqrt(A / (p + B))
            return (p - pk) * q, q * (1 - 0.5 * (p - pk) / (p + B))
        return (2 * ck / (g - 1) * ((p / pk) ** ((g - 1) / (2 * g)) - 1),
                (p / pk) ** (-(g + 1) / (2 * g)) / (r * ck))

    p = max(1e-10, 0.5 * (pL + pR) - 0.125 * (uR - uL) * (rL + rR) * (cL + cR))   # PVRS guess
    for _ in range(200):
        fL, dL = f(p, rL, pL, cL)
        fR, dR = f(p, rR, pR, cR)
        pn = max(1e-12, p - (fL + fR + uR - uL) / (dL + dR))
        if abs(pn - p) / (0.5 * (pn + p)) < 1e-15:
            p = pn
            break
        p = pn
    fL, _ = f(p, rL, pL, cL)
    fR, _ = f(p, rR, pR, cR)
    return p, 0.5 * (uL + uR) + 0.5 * (fR - fL)


def sample(s, rL, uL, pL, rR, uR, pR, ps, us, g=G):
    """(rho, u, p) at x/t = s (Toro sec. 4.5, every wave pattern)"""
    cL, cR = math.sqrt(g * pL / rL), math.sqrt(g * pR / rR)
    gm, gp = (g - 1) / (g + 1), (g + 1) / (2 * g)
    if s <= us:
        if ps > pL:
            SL = uL - cL * math.sqrt(gp * ps / pL + (g - 1) / (2 * g))
            if s <= SL:
                return rL, uL, pL
            return rL * (ps / pL + gm) / (gm * ps / pL + 1), us, ps
        if s <= uL - cL:
            return rL, uL, pL
        if s > us - cL * (ps / pL) ** ((g - 1) / (2 * g)):
            return rL * (ps / pL) ** (1 / g), us, ps
        u = 2 / (g + 1) * (cL + (g - 1) / 2 * uL + s)
        c = 2 / (g + 1) * (cL + (g - 1) / 2 * (uL - s))
        return rL * (c / cL) ** (2 / (g - 1)), u, pL * (c / cL) ** (2 * g / (g - 1))
    if ps > pR:
        SR = uR + cR * math.sqrt(gp * ps / pR + (g - 1) / (2 * g))
        if s >= SR:
            return rR, uR, pR
        return rR * (ps / pR + gm) / (gm * ps / pR + 1), us, ps
    if s >= uR + cR:
        return rR, uR, pR
    if s <= us + cR * (ps / pR) ** ((g - 1) / (2 * g)):
        return rR * (ps / pR) ** (1 / g), us, ps
    u = 2 / (g + 1) * (-cR + (g - 1) / 2 * uR + s)
    c = 2 / (g + 1) * (cR - (g - 1) / 2 * (uR - s))
    return rR * (c / cR) ** (2 / (g - 1)), u, pR * (c / cR) ** (2 * g / (g - 1))


def exact(case, tname):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import rho_central_fields as rf
    t = float(tname)
    x = rf.cell_centres(case)[:, 0]
    rho = rf.internal(f"{case}/{tname}/rho")
    u = rf.internal(f"{case}/{tname}/U")[:, 0]
    p = rf.internal(f"{case}/{tname}/p")
    p0 = rf.internal(f"{case}/0/p")
    T0 = rf.internal(f"{case}/0/T")
    U0 = rf.internal(f"{case}/0/U")
    if U0.ndim == 1:   # `uniform (u 0 0)`: setFields wrote no list
        U0 = np.tile(U0, (len(x), 1))
    if p0.ndim == 0:
        p0 = np.full(len(x), float(p0))
    if T0.ndim == 0:
        T0 = np.full(len(x), float(T0))
    iL, iR = int(np.argmin(x)), int(np.argmax(x))
    R = RR / float(re.search(r"molWeight\s+([^;]+);", open(f"{case}/constant/thermophysicalProperties").read()).group(1))
    rL, pL, uL = p0[iL] / (R * T0[iL]), p0[iL], U0[iL][0]
    rR, pR, uR = p0[iR] / (R * T0[iR]), p0[iR], U0[iR][0]
    ps, us = star(rL, uL, pL, rR, uR, pR)
    ex = np.array([sample(xi / t, rL, uL, pL, rR, uR, pR, ps, us) for xi in x])
    L1 = [float(np.mean(np.abs(q - ex[:, k])) / np.max(np.abs(ex[:, k]))) for k, q in enumerate((rho, u, p))]
    print(f"exact pstar {ps:.10g} ustar {us:.10g}")
    print(f"L1 rho {L1[0]:.6e} u {L1[1]:.6e} p {L1[2]:.6e}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["setup"] and len(a) == 3 and a[1] in PROBLEMS:
        setup(a[1], a[2])
    elif a[:1] == ["exact"] and len(a) == 3:
        exact(a[1], a[2])
    elif a[:1] == ["problems"]:
        print(" ".join(PROBLEMS))
    else:
        raise SystemExit(__doc__)
