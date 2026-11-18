#!/usr/bin/env python3
"""Viscous cases for the rhoCentralFoam gate of milestone G, and what it measures on them (no OpenFOAM needed).

  rho_central_viscous.py setup <kind> <case>
      rewrites a staged copy of OpenFOAM's shockTube (shear, conduction, sod) or obliqueShock (oblique), before its mesh,
      into a viscous case: constTransport mu and Pr below, fvSolution's `h` solver entry read for `e` too ("(e|h)": the
      tutorials name `h`, an energy rhoCentralFoam does not solve), writePrecision 17. The tube is shockTube's ten times
      as long (x in [-5, 5], 1000 cells), a perfect gas with R = 1 and gamma = 1.4 (as tests/rho_central_riemann.py):
        shear       momentum diffusion alone (Stokes' first problem): the tube one cell high with its y faces a patch
                    (`lowerUpper`, zeroGradient), so U_y is solved; p = 1, T = 1 (rho = 1), U = (0, 0.1, 0) for x < 0
                    and (0, -0.1, 0) for x > 0; mu 0.01, Pr 1; t = 1, writes every 0.25
        conduction  energy diffusion alone: the tube at rest, p = 1, T = 2 for x < 0 and 1 for x > 0; mu 0.01, Pr 1;
                    t = 1, writes every 0.25
        sod         Toro's test 1 (tests/rho_central_riemann.py setup sod) with mu 0.002, Pr 1; t = 2, writes every 0.5
        oblique     obliqueShock as shipped (Mach 2.9, 1800 cells, t = 10) with mu 0.01 and Pr 0.72
  rho_central_viscous.py exact <kind> <case> <time>
      shear: U_y against -0.1 erf(x/(2 sqrt(nu t))), nu = mu/rho; conduction: T against 1.5 - 0.5 erf(x/(2 sqrt(k t)))
      with k = alphaEff/rho (alphaEff = gamma mu/Pr, e = Cv T), the constant-density limit. `exact L1 <..> max <..>`
      relative to the largest exact |value| (diagnostic: the cases are compressible)
  rho_central_viscous.py fluxes <A> <B> <time> <mu> <Pr>
      the viscous fluxes, the same arithmetic for both cases: on every internal face the normal derivative
      (q_N - q_P)/|C_N - C_P| of U's components and of T, giving the shear rate's stress mu*dU/dn (each component) and
      the heat flux alphaEff*Cv*dT/dn; prints `fluxes stress <|A - B| max over faces / max |B|> heat <..>`
"""
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rho_central_fields as rf  # noqa: E402
import rho_central_riemann as rr  # noqa: E402

G = 1.4
CP = 3.5           # R = 1, gamma = 1.4
CV = CP - 1.0


def edit(path, pattern, repl, count=1):
    s = open(path).read()
    t, n = re.subn(pattern, repl, s, count=count, flags=re.S | re.M)
    if n != count:
        raise SystemExit(f"rho_central_viscous: {path}: expected {count} edit(s) of /{pattern}/, made {n}")
    open(path, "w").write(t)


def transport(case, mu, Pr):
    tp = f"{case}/constant/thermophysicalProperties"
    edit(tp, r"^(\s*mu\s+)[^;]+;", rf"\g<1>{mu!r};")
    edit(tp, r"^(\s*Pr\s+)[^;]+;", rf"\g<1>{Pr!r};")
    edit(f"{case}/system/fvSolution", r'^(\s*)h\s*$', r'\1"(e|h)"')
    edit(f"{case}/system/controlDict", r"^(writePrecision\s+)\S+;", r"\g<1>17;")


def tube_thermo(case):
    tp = f"{case}/constant/thermophysicalProperties"
    edit(tp, r"(molWeight\s+)[^;]+;", rf"\g<1>{rr.RR!r};")
    edit(tp, r"(Cp\s+)[^;]+;", rf"\g<1>{CP!r};")
    edit(tp, r"(Hf\s+)[^;]+;", r"\g<1>0;")


def control(case, t, every):
    cd = f"{case}/system/controlDict"
    edit(cd, r"^(endTime\s+)\S+;", rf"\g<1>{t!r};")
    edit(cd, r"^(writeControl\s+)\S+;", r"\g<1>adjustable;")
    edit(cd, r"^(writeInterval\s+)\S+;", rf"\g<1>{every!r};")
    edit(cd, r"^(deltaT\s+)\S+;", r"\g<1>1e-05;")


HEAD = ("FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       %s;\n    object      %s;\n}\n"
        "dimensions      %s;\n\n")


def field(case, name, cls, dims, values, patches):
    n = len(values)
    if cls == "volVectorField":
        body = "\n".join("(%r %r %r)" % tuple(v) for v in values)
        kind = "vector"
    else:
        body = "\n".join("%r" % v for v in values)
        kind = "scalar"
    s = HEAD % (cls, name, dims) + "internalField   nonuniform List<%s>\n%d\n(\n%s\n)\n;\n\nboundaryField\n{\n" % (kind, n, body)
    for p, t in patches:
        s += "    %s\n    {\n        type            %s;\n    }\n" % (p, t)
    open(f"{case}/0/{name}", "w").write(s + "}\n")


def tube_mesh(case, y_patch):
    bm = f"{case}/system/blockMeshDict"
    edit(bm, r"\(100 1 1\)", "(1000 1 1)")
    if y_patch:   # the y faces a patch of their own: U_y is then a solved component
        edit(bm, r"empty\s*\{\s*type empty;\s*faces\s*\(\s*\(0 1 5 4\)\s*\(5 6 7 4\)\s*\(3 7 6 2\)\s*\(0 3 2 1\)\s*\);\s*\}",
             "lowerUpper\n    {\n        type patch;\n        faces\n        (\n            (0 1 5 4)\n            (3 7 6 2)\n"
             "        );\n    }\n    frontAndBack\n    {\n        type empty;\n        faces\n        (\n"
             "            (5 6 7 4)\n            (0 3 2 1)\n        );\n    }")


def setup(kind, case):
    if kind in ("shear", "conduction"):
        tube_mesh(case, kind == "shear")
        tube_thermo(case)
        control(case, 1.0, 0.25)
        x = -5 + (np.arange(1000) + 0.5) * 0.01
        patches = [("sides", "zeroGradient")] + ([("lowerUpper", "zeroGradient"), ("frontAndBack", "empty")]
                                                  if kind == "shear" else [("empty", "empty")])
        if kind == "shear":
            U = [(0.0, 0.1 if xi < 0 else -0.1, 0.0) for xi in x]
            T = [1.0] * 1000
        else:
            U = [(0.0, 0.0, 0.0)] * 1000
            T = [2.0 if xi < 0 else 1.0 for xi in x]
        os.makedirs(f"{case}/0", exist_ok=True)
        field(case, "U", "volVectorField", "[0 1 -1 0 0 0 0]", U, patches)
        field(case, "T", "volScalarField", "[0 0 0 1 0 0 0]", T, patches)
        field(case, "p", "volScalarField", "[1 -1 -2 0 0 0 0]", [1.0] * 1000, patches)
        transport(case, 0.01, 1.0)
    elif kind == "sod":
        rr.setup("sod", case)
        edit(f"{case}/system/blockMeshDict", r"\(100 1 1\)", "(1000 1 1)")
        transport(case, 0.002, 1.0)
    elif kind == "oblique":
        transport(case, 0.01, 0.72)
    else:
        raise SystemExit(f"rho_central_viscous: no case kind {kind}")


def exact(kind, case, tname):
    t = float(tname)
    x = rf.cell_centres(case)[:, 0]
    tr = open(f"{case}/constant/thermophysicalProperties").read()
    mu = float(re.search(r"^\s*mu\s+([^;]+);", tr, re.M).group(1))
    Pr = float(re.search(r"^\s*Pr\s+([^;]+);", tr, re.M).group(1))
    rho = rf.internal(f"{case}/{tname}/rho")
    if kind == "shear":
        q = rf.internal(f"{case}/{tname}/U")[:, 1]
        ex = np.array([-0.1 * math.erf(xi / (2 * math.sqrt(mu / r * t))) for xi, r in zip(x, rho)])
    else:
        q = rf.internal(f"{case}/{tname}/T")
        ex = np.array([1.5 - 0.5 * math.erf(xi / (2 * math.sqrt(G * mu / Pr / r * t))) for xi, r in zip(x, rho)])
    m = float(np.max(np.abs(ex)))
    print(f"exact L1 {float(np.mean(np.abs(q - ex))) / m:.6e} max {float(np.max(np.abs(q - ex))) / m:.6e}")


def faces(case):
    """owner and neighbour of the internal faces (rho_central_fields.py's polyMesh reader)"""
    pm = f"{case}/constant/polyMesh"
    _, s, j = rf._list(f"{pm}/neighbour", int)
    nei = np.array(s[j:s.index(")", j)].split(), dtype=int)
    _, s, j = rf._list(f"{pm}/owner", int)
    own = np.array(s[j:s.index(")", j)].split(), dtype=int)
    return own[:len(nei)], nei


def fluxes(a, b, tname, mu, Pr):
    own, nei = faces(b)
    C = rf.cell_centres(b)
    dn = np.linalg.norm(C[nei] - C[own], axis=1)
    out = {}
    for name, case in (("A", a), ("B", b)):
        U = rf.internal(f"{case}/{tname}/U")
        T = rf.internal(f"{case}/{tname}/T")
        out[name] = (mu * (U[nei] - U[own]) / dn[:, None], G * mu / Pr * CV * (T[nei] - T[own]) / dn)
    s = float(np.max(np.abs(out["A"][0] - out["B"][0])) / max(np.max(np.abs(out["B"][0])), 1e-300))
    h = float(np.max(np.abs(out["A"][1] - out["B"][1])) / max(np.max(np.abs(out["B"][1])), 1e-300))
    print(f"fluxes stress {s:.3e} heat {h:.3e}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["setup"] and len(a) == 3:
        setup(a[1], a[2])
    elif a[:1] == ["exact"] and len(a) == 4:
        exact(a[1], a[2], a[3])
    elif a[:1] == ["fluxes"] and len(a) == 6:
        fluxes(a[1], a[2], a[3], float(a[4]), float(a[5]))
    else:
        raise SystemExit(__doc__)
