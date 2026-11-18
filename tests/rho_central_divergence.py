#!/usr/bin/env python3
"""Diagnostics for the rhoCentralFoam gates' divergent cases (reported, never a bound).

  rho_central_divergence.py trace <OpenFOAM serial case> <deltaT> <name>=<case> ...
      at every write time the cases share with serial: the largest difference over U p T rho, relative to serial's
      largest |value| of the field (rho_central_fields.py absdiff). One line per case:
        `trace <name> <step of the first write above 1e-12> <above 1e-8> <above 1e-4> <largest> <its step> <field>`
      (steps are write times over deltaT, rounded; `-` where never above), then `points <name> <t>:<d> ...`
  rho_central_divergence.py shock1d <case> <time>
      a 1-D Riemann problem's right-moving shock read off the cell-centre pressure: the crossing nearest the right end
      of p = (p_right + p_plateau)/2, the plateau being the median p over the 0.3 behind it; and the exact shock (Toro
      ch. 4, tests/rho_central_riemann.py) for the case's initial states:
        `shock1d x <..> p2 <..> rho2 <..> exact_x <..> exact_p2 <..> exact_rho2 <..>`
"""
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rho_central_fields as rf  # noqa: E402
import rho_central_riemann as rr  # noqa: E402

LEVELS = (1e-12, 1e-8, 1e-4)


def times(case):
    return sorted((t for t in os.listdir(case) if re.fullmatch(r"[0-9.e+-]+", t) and float(t) > 0
                   and os.path.isfile(f"{case}/{t}/U")), key=float)


def trace(serial, dt, runs):
    ts = times(serial)
    for spec in runs:
        name, case = spec.split("=", 1)
        pts = []
        for t in ts:
            if not os.path.isfile(f"{case}/{t}/U"):
                continue
            w = max(rf.absdiff(f"{case}/{t}", f"{serial}/{t}", ["U", "p", "T", "rho"]), key=lambda r: r[2])
            pts.append((t, w[2], w[0]))
        if not pts:
            print(f"trace {name} - - - nan - -")
            continue
        step = lambda t: str(int(round(float(t) / dt)))
        first = [next((step(t) for t, d, _ in pts if d > lv), "-") for lv in LEVELS]
        t, d, f = max(pts, key=lambda p: p[1])
        print(f"trace {name} {' '.join(first)} {d:.3e} {step(t)} {f}")
        print(f"points {name} " + " ".join(f"{t}:{d:.3e}" for t, d, _ in pts))


def shock1d(case, tname):
    t = float(tname)
    x = rf.cell_centres(case)[:, 0]
    o = np.argsort(x)
    x = x[o]
    p = np.atleast_1d(rf.internal(f"{case}/{tname}/p"))[o]
    rho = np.atleast_1d(rf.internal(f"{case}/{tname}/rho"))[o]
    pR = p[-1]
    # the plateau: scan from the right for the first cell well above p_right, take the median behind it
    i = len(p) - 1
    while i > 0 and p[i] < pR + 0.05 * (p.max() - pR):
        i -= 1
    behind = (x > x[i] - 0.3) & (x < x[i] - 0.05)
    p2, rho2 = float(np.median(p[behind])), float(np.median(rho[behind]))
    mid = 0.5 * (pR + p2)
    j = len(p) - 1
    while j > 0 and not (p[j - 1] >= mid > p[j]):
        j -= 1
    xs = x[j - 1] + (mid - p[j - 1]) * (x[j] - x[j - 1]) / (p[j] - p[j - 1])
    # the exact solution for the case's initial states
    p0 = np.atleast_1d(rf.internal(f"{case}/0/p"))
    T0 = np.atleast_1d(rf.internal(f"{case}/0/T"))
    U0 = rf.internal(f"{case}/0/U")
    U0 = np.tile(U0, (len(x), 1)) if U0.ndim == 1 else U0
    if p0.size == 1:
        p0 = np.full(len(x), float(p0[0]))
    if T0.size == 1:
        T0 = np.full(len(x), float(T0[0]))
    p0, T0, U0 = p0[o], T0[o], U0[o]
    R = rr.RR / float(re.search(r"molWeight\s+([^;]+);", open(f"{case}/constant/thermophysicalProperties").read()).group(1))
    rL, uL, pL = p0[0] / (R * T0[0]), U0[0][0], p0[0]
    rR_, uR, pR0 = p0[-1] / (R * T0[-1]), U0[-1][0], p0[-1]
    ps, us = rr.star(rL, uL, pL, rR_, uR, pR0)
    g = rr.G
    cR = math.sqrt(g * pR0 / rR_)
    SR = uR + cR * math.sqrt((g + 1) / (2 * g) * ps / pR0 + (g - 1) / (2 * g))
    gm = (g - 1) / (g + 1)
    rho2e = rR_ * (ps / pR0 + gm) / (gm * ps / pR0 + 1)
    print(f"shock1d x {xs:.6g} p2 {p2:.6g} rho2 {rho2:.6g} exact_x {SR * t:.6g} exact_p2 {ps:.6g} exact_rho2 {rho2e:.6g}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["trace"] and len(a) >= 4:
        trace(a[1], float(a[2]), a[3:])
    elif a[:1] == ["shock1d"] and len(a) == 3:
        shock1d(a[1], a[2])
    else:
        raise SystemExit(__doc__)
