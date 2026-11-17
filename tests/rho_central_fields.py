#!/usr/bin/env python3
"""Field tools for the rhoCentralFoam gates, on OpenFOAM ascii time directories (no OpenFOAM needed).

  rho_central_fields.py absdiff <dir A> <dir B> [fields]
      per field: the largest |A - B| over the cells (absolute, in the field's units; a vector's by component) and
      that over max|B| (the gates' relative measure), one line per field: `absdiff <field> <abs> <rel> <cell>`
  rho_central_fields.py maxrel <dir A> <dir B> [fields]
      the worst max|A - B|/max|B| over the fields: `maxrel <value> <field>`
  rho_central_fields.py shock <case dir> <time> <kind>
      the shock angle and position read off the cell-centre pressure (kind: wedge, cone, oblique); see shock()
  rho_central_fields.py theory <kind>
      the inviscid angle for the case (oblique-shock relations, Taylor-Maccoll for the cone); see theory()
  rho_central_fields.py stats <case dir> <time> [<earlier time>]
      min rho, p, T over the cells, NaN count, and with an earlier time the largest relative change of p between
      them (how steady the run has become): `stats <min rho> <min p> <min T> <nan> <steady>`

Cell centres for `shock` come from the case's own polyMesh (points, faces, owner), averaged the way
primitiveMeshCellCentresAndVols does for a hex or prism closely enough to locate a shock (the angle is fitted, not
compared to OpenFOAM's digits).
"""
import math
import os
import re
import sys

import numpy as np


# ---- OpenFOAM ascii ---------------------------------------------------------------------------------------------------
def _value(s, i):
    m = re.compile(r"\s*(uniform|nonuniform)\s+").match(s, i)
    j = m.end()
    if m.group(1) == "uniform":
        if s[j] == "(":
            k = s.index(")", j)
            return np.array([float(x) for x in s[j + 1:k].split()])
        return np.array(float(s[j:s.index(";", j)]))
    m2 = re.compile(r"List<(\w+)>\s*(\d+)\s*\(").match(s, j)
    n, j = int(m2.group(2)), m2.end()
    if m2.group(1) == "scalar":
        a = np.array(s[j:s.index(")", j)].split(), dtype=float)
    else:
        body = s[j:s.index(")\n)", j) + 1] if ")\n)" in s[j:] else s[j:]
        a = np.array(re.findall(r"\(([^()]*)\)", body)[:n], dtype=object)
        a = np.array([[float(x) for x in r.split()] for r in a])
    if len(a) != n:
        raise SystemExit(f"rho_central_fields: list of {len(a)}, header says {n}")
    return a


def internal(path):
    s = open(path).read()
    return _value(s, s.index("internalField") + len("internalField"))


def _list(path, cast):
    s = open(path).read()
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    s = re.sub(r"//[^\n]*", "", s)
    m = re.search(r"\n(\d+)\s*\(", s[s.index("}") + 1:])
    j = s.index("}") + 1 + m.end()
    n = int(m.group(1))
    return n, s, j


def cell_centres(case):
    pm = f"{case}/constant/polyMesh"
    n, s, j = _list(f"{pm}/points", float)
    pts = np.array([[float(x) for x in t.split()] for t in re.findall(r"\(([^()]*)\)", s[j:])[:n]])
    n, s, j = _list(f"{pm}/faces", int)
    faces = [[int(x) for x in t.split()] for t in re.findall(r"\d+\(([^()]*)\)", s[j:])[:n]]
    n, s, j = _list(f"{pm}/owner", int)
    own = np.array(s[j:s.index(")", j)].split(), dtype=int)
    n, s, j = _list(f"{pm}/neighbour", int)
    nei = np.array(s[j:s.index(")", j)].split(), dtype=int)
    nc = own.max() + 1
    fc = np.array([pts[f].mean(axis=0) for f in faces])
    acc = np.zeros((nc, 3))
    cnt = np.zeros(nc)
    np.add.at(acc, own, fc)
    np.add.at(cnt, own, 1)
    np.add.at(acc, nei, fc[:len(nei)])
    np.add.at(cnt, nei, 1)
    return acc / cnt[:, None]


# ---- comparisons -----------------------------------------------------------------------------------------------------
def absdiff(a, b, fields):
    out = []
    for f in fields:
        x, y = internal(f"{a}/{f}"), internal(f"{b}/{f}")
        d = np.abs(x - y)
        if d.ndim == 2:
            d = d.max(axis=1)
        out.append((f, float(d.max()), float(d.max() / np.abs(y).max()), int(d.argmax())))
    return out


# ---- shocks -----------------------------------------------------------------------------------------------------------
def theory(kind):
    g = 1.4
    if kind in ("wedge", "cone"):
        M, th = 5.0, math.radians(15)
    if kind == "wedge":   # theta-beta-M, weak root
        f = lambda b: math.tan(th) - 2 / math.tan(b) * (M**2 * math.sin(b)**2 - 1) / (M**2 * (g + math.cos(2 * b)) + 2)
        lo, hi = math.asin(1 / M) + 1e-9, math.radians(60)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if f(lo) * f(mid) > 0 else (lo, mid)
        return {"beta": math.degrees(0.5 * (lo + hi))}
    if kind == "cone":    # Taylor-Maccoll, shooting on the shock angle until the surface ray is at 15 degrees
        def surface(beta):
            Mn1 = M * math.sin(beta)
            Mn2 = math.sqrt((1 + 0.5 * (g - 1) * Mn1**2) / (g * Mn1**2 - 0.5 * (g - 1)))
            d = math.atan(2 / math.tan(beta) * (M**2 * math.sin(beta)**2 - 1) / (M**2 * (g + math.cos(2 * beta)) + 2))
            M2 = Mn2 / math.sin(beta - d)
            V = (2 / ((g - 1) * M2**2) + 1) ** -0.5
            vr, vt, t = V * math.cos(beta - d), -V * math.sin(beta - d), beta
            h = -1e-5
            def rhs(t, vr, vt):
                a = 0.5 * (g - 1) * (1 - vr**2 - vt**2)
                return vt, (vt**2 * vr - a * (2 * vr + vt / math.tan(t))) / (a - vt**2)
            while vt < 0 and t > 1e-3:
                k1 = rhs(t, vr, vt); k2 = rhs(t + h / 2, vr + h / 2 * k1[0], vt + h / 2 * k1[1])
                k3 = rhs(t + h / 2, vr + h / 2 * k2[0], vt + h / 2 * k2[1]); k4 = rhs(t + h, vr + h * k3[0], vt + h * k3[1])
                vr += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); vt += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]); t += h
            return t
        lo, hi = math.asin(1 / M) + 1e-3, math.radians(40)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if surface(mid) < th else (lo, mid)
        return {"beta": math.degrees(0.5 * (lo + hi))}
    if kind == "oblique":  # the tutorial's incident shock (M 2.9, its top state) and the regular reflection
        M1, b1 = 2.9, None
        d1 = math.atan(0.50632 / 2.61933)
        f = lambda b, M, th: math.tan(th) - 2 / math.tan(b) * (M**2 * math.sin(b)**2 - 1) / (M**2 * (g + math.cos(2 * b)) + 2)
        def weak(M, th):
            lo, hi = math.asin(1 / M) + 1e-9, math.radians(64)
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if f(lo, M, th) * f(mid, M, th) > 0 else (lo, mid)
            return 0.5 * (lo + hi)
        b1 = weak(M1, d1)
        Mn1 = M1 * math.sin(b1)
        Mn2 = math.sqrt((1 + 0.5 * (g - 1) * Mn1**2) / (g * Mn1**2 - 0.5 * (g - 1)))
        M2 = Mn2 / math.sin(b1 - d1)
        b2 = weak(M2, d1)
        p2 = 1 + 2 * g / (g + 1) * (Mn1**2 - 1)
        p3 = p2 * (1 + 2 * g / (g + 1) * ((M2 * math.sin(b2))**2 - 1))
        # the reflected shock turns region 2's flow (inclined d1 towards the wall) back parallel to it: its angle to
        # the WALL is its angle to that flow less d1
        return {"incident": math.degrees(b1), "reflected": math.degrees(b2 - d1), "impact_x": 1.0 / math.tan(b1),
                "p2": p2, "p3": p3}
    raise SystemExit(f"rho_central_fields: no theory for `{kind}`")


def _crossings(xs, ys, level):
    """positions along a line where ys crosses level going from below to above, linearly interpolated"""
    out = []
    for i in range(len(xs) - 1):
        if (ys[i] - level) * (ys[i + 1] - level) < 0:
            out.append(xs[i] + (level - ys[i]) / (ys[i + 1] - ys[i]) * (xs[i + 1] - xs[i]))
    return out


def shock(case, time, kind):
    """The shock read off the cell-centre pressure: on each column of cells (fixed x), the radius/height where p
    crosses the mean of the free-stream and the post-shock plateau, the crossing nearest the free stream; a line
    fitted through those points over the middle of the ramp gives the angle (degrees) and its height at the outlet.
    For `oblique`, the incident shock is read on rows (fixed y) above y = 0.5 and the reflected one below, and the
    impact point is where the two lines meet y = 0."""
    C = cell_centres(case)
    p = internal(f"{case}/{time}/p")
    x, y = C[:, 0], C[:, 1]
    xr = np.round(x, 9)
    res = {}
    if kind in ("wedge", "cone"):
        pinf = 1.0
        pts = []
        for xc in np.unique(xr):
            if not (0.05 < xc < 0.28):
                continue
            m = xr == xc
            order = np.argsort(-y[m])
            ys, ps = y[m][order], p[m][order]
            plateau = np.median(ps[-4:])
            cr = _crossings(ys, ps, 0.5 * (pinf + plateau))
            if cr:
                pts.append((xc, cr[0]))
        pts = np.array(pts)
        k, c0 = np.polyfit(pts[:, 0], pts[:, 1], 1)
        res = {"angle": math.degrees(math.atan(k)), "apex_x": -c0 / k, "y_at_0.25": k * 0.25 + c0,
               "points": len(pts), "rms": float(np.sqrt(np.mean((np.polyval([k, c0], pts[:, 0]) - pts[:, 1])**2)))}
    elif kind == "oblique":
        # rows of cells (fixed y): the incident shock is the first crossing of (p1 + p2)/2 going downstream, the
        # reflected one the first crossing of (p2 + p3)/2 after it, p2 and p3 the inviscid plateaus (theory())
        th = theory("oblique")
        yr = np.round(y, 9)
        inc, ref = [], []
        for yc in np.unique(yr):
            m = yr == yc
            order = np.argsort(x[m])
            xs, ps = x[m][order], p[m][order]
            ci = _crossings(xs, ps, 0.5 * (1.0 + th["p2"]))
            if ci and 0.2 < yc < 0.9:
                inc.append((ci[0], yc))
            if ci and 0.1 < yc < 0.6:
                cr = [c for c in _crossings(xs, ps, 0.5 * (th["p2"] + th["p3"])) if c > ci[0]]
                if cr:
                    ref.append((cr[0], yc))
        inc, ref = np.array(inc), np.array(ref)
        ki, ci0 = np.polyfit(inc[:, 0], inc[:, 1], 1)
        kr, cr0 = np.polyfit(ref[:, 0], ref[:, 1], 1)
        res = {"incident": math.degrees(math.atan(-ki)), "reflected": math.degrees(math.atan(kr)),
               "impact_x_incident": -ci0 / ki, "impact_x_reflected": -cr0 / kr,
               "points_incident": len(inc), "points_reflected": len(ref)}
    return res


def stats(case, time, earlier=None):
    r, p, T = (internal(f"{case}/{time}/{f}") for f in ("rho", "p", "T"))
    nan = int(np.isnan(r).sum() + np.isnan(p).sum() + np.isnan(T).sum())
    steady = float("nan")
    if earlier:
        p0 = internal(f"{case}/{earlier}/p")
        steady = float(np.abs(p - p0).max() / np.abs(p).max())
    return float(r.min()), float(p.min()), float(T.min()), nan, steady


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    if a[0] == "absdiff":
        for f, ab, rel, c in absdiff(a[1], a[2], a[3:] or ["U", "p", "T", "rho"]):
            print(f"absdiff {f} {ab:.3e} {rel:.3e} {c}")
    elif a[0] == "maxrel":
        w = max(absdiff(a[1], a[2], a[3:] or ["U", "p", "T", "rho"]), key=lambda r: r[2])
        print(f"maxrel {w[2]:.3e} {w[0]}")
    elif a[0] == "shock":
        print("shock", " ".join(f"{k} {v:.6g}" if isinstance(v, float) else f"{k} {v}" for k, v in shock(a[1], a[2], a[3]).items()))
    elif a[0] == "theory":
        print("theory", " ".join(f"{k} {v:.6g}" for k, v in theory(a[1]).items()))
    elif a[0] == "stats":
        print("stats %.6g %.6g %.6g %d %.3e" % stats(a[1], a[2], a[3] if len(a) > 3 else None))
    else:
        raise SystemExit(__doc__)
