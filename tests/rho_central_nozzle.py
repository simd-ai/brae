#!/usr/bin/env python3
"""Nozzle-boundary cases for the rhoCentralFoam gate of milestone H, and what it measures on them (no OpenFOAM needed).

  rho_central_nozzle.py setup <kind> <case>
      rewrites a staged case before its mesh. `ladenburg` and `ladenburg_long` are OpenFOAM's LadenburgJet60psi (its
      0.orig unpacked into 0 by the gate): as shipped but for writePrecision 17, and the long one run five times as far
      (endTime 1e-4, writes every 2e-5). `ladenburg_uzseed` (milestone I) is the shipped case with U_z seeded +-1e-3 in
      cells 25-39, the first row off the axis around the Mach disk, run to 3e-5 with writes every 5e-6: OpenFOAM's
      out-of-plane mode there grows from a seed far above round-off, so both codes' growth of it is compared. The rest are shockTube staged as tests/rho_central_viscous.py stages it: the
      tube ten times as long (x in [-5, 5], 1000 cells), a perfect gas with R = 1 and gamma = 1.4, its `sides` patch
      split into `outlet` (x = 5) and `inlet` (x = -5) in that face order, so the mesh's faces, owner and neighbour are
      shockTube1000's and its frozen partition applies:
        reservoir       at rest, p = T = 1; inlet: p totalPressure (psi thermo:psi, gamma 1.4, p0 2), T totalTemperature
                        (T0 1.2), U zeroGradient; outlet: p waveTransmissive (fieldInf 1, lInf 2), U and T zeroGradient.
                        Inviscid; t = 10, writes every 1
        pulse           at rest, p = T = 1 but p = 1.5 for |x| < 0.5; both ends waveTransmissive on p (inlet: fieldInf 1,
                        lInf 1; outlet: no lInf), U and T zeroGradient. Inviscid; t = 6, writes every 0.5
        backflow        U = (-0.3, 0, 0), p = T = 1, leaving through the inlet: p totalPressure (p0 1.1), T
                        totalTemperature (T0 1.1), U zeroGradient; entering at the outlet: U fixedValue (-0.3 0 0), T
                        fixedValue 1, p zeroGradient. Inviscid; t = 4, writes every 0.5 (fixing p and T on an outflow
                        over-specifies it: OpenFOAM's own runs part by 4e-01 at t = 10, so the case stops at 4)
        reservoir_visc  reservoir with sutherland transport (As 0.02, Ts 1), U and e solved as LadenburgJet60psi
                        ships them (smoothSolver GaussSeidel, tolerance 1e-10, relTol 0)
        conduction_suth tests/rho_central_viscous.py's conduction (T = 2 | 1 at rest, both ends zeroGradient) with
                        sutherland transport (As 0.02, Ts 1), U and e solved as reservoir_visc; t = 1, writes every 0.25
      (the inviscid tube cases amplify round-off in OpenFOAM itself: its two- and four-way runs and its one-ulp run part
      from serial by 1e-9 at t = 2 and 1e-4 to 1e-2 by the end, 2026-11-18; their envelope is wide late in the run)
  rho_central_nozzle.py patches <A> <B> <time> <patch[,patch...]>
      the boundary profiles: max over the patches' faces of |A - B| over max |B| for p, T and U (each component),
      `patches p <..> T <..> U <..>`
  rho_central_nozzle.py branches <case> <time> <patch> <p0>
      a totalPressure patch's two branches: faces at p0 (outflow: no dynamic head) and below it (inflow),
      `branches outflow <n> inflow <n> head <largest (p0 - p_b)/p0>`
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rho_central_fields as rf      # noqa: E402
import rho_central_viscous as rv     # noqa: E402

N = 1000
X = -5 + (np.arange(N) + 0.5) * 0.01
PSI = "        psi             thermo:psi;\n        gamma           1.4;\n"


def total_p(p0, value):
    return ("        type            totalPressure;\n        rho             none;\n" + PSI +
            f"        p0              uniform {p0!r};\n        value           uniform {value!r};\n")


def total_T(T0, value):
    return ("        type            totalTemperature;\n" + PSI +
            f"        T0              uniform {T0!r};\n        value           uniform {value!r};\n")


def wave(value, inf=None, linf=None):
    s = "        type            waveTransmissive;\n        field           p;\n" + PSI
    if linf is not None:
        s += f"        fieldInf        {inf!r};\n        lInf            {linf!r};\n"
    return s + f"        value           uniform {value!r};\n"


def plain(t, value=None):
    s = f"        type            {t};\n"
    return s + (f"        value           uniform {value};\n" if value is not None else "")


def field(case, name, cls, dims, values, patches):
    kind = "vector" if cls == "volVectorField" else "scalar"
    body = "\n".join(("(%r %r %r)" % tuple(v)) if kind == "vector" else ("%r" % v) for v in values)
    s = rv.HEAD % (cls, name, dims) + "internalField   nonuniform List<%s>\n%d\n(\n%s\n)\n;\n\nboundaryField\n{\n" % (
        kind, len(values), body)
    for p, b in patches:
        s += "    %s\n    {\n%s    }\n" % (p, b)
    open(f"{case}/0/{name}", "w").write(s + "}\n")


def tube_mesh(case):
    bm = f"{case}/system/blockMeshDict"
    rv.edit(bm, r"\(100 1 1\)", "(1000 1 1)")
    rv.edit(bm, r"sides\s*\{\s*type patch;\s*faces\s*\(\s*\(1 2 6 5\)\s*\(0 4 7 3\)\s*\);\s*\}",
            "outlet\n    {\n        type patch;\n        faces\n        (\n            (1 2 6 5)\n        );\n    }\n"
            "    inlet\n    {\n        type patch;\n        faces\n        (\n            (0 4 7 3)\n        );\n    }")


def sutherland(case):
    tp = f"{case}/constant/thermophysicalProperties"
    rv.edit(tp, r"^(\s*transport\s+)const;", r"\g<1>sutherland;")
    rv.edit(tp, r"^(\s*)mu\s+[^;]+;\s*\n\s*Pr\s+[^;]+;", r"\g<1>As              0.02;\n\g<1>Ts              1;")
    fs = f"{case}/system/fvSolution"
    rv.edit(fs, r'^(\s*)h\s*$', r'\1"(e|h)"')
    rv.edit(fs, r"^(\s*tolerance\s+)[^;]+;", r"\g<1>1e-10;", count=2)
    rv.edit(fs, r"^(\s*relTol\s+)[^;]+;", r"\g<1>0;", count=2)


def tube(kind, case, t, every, U, p, T, patches):
    tube_mesh(case)
    rv.tube_thermo(case)
    rv.control(case, t, every)
    rv.edit(f"{case}/system/controlDict", r"^(writePrecision\s+)\S+;", r"\g<1>17;")
    os.makedirs(f"{case}/0", exist_ok=True)
    pp = [(n, b["p"]) for n, b in patches] + [("empty", plain("empty"))]
    TT = [(n, b["T"]) for n, b in patches] + [("empty", plain("empty"))]
    UU = [(n, b["U"]) for n, b in patches] + [("empty", plain("empty"))]
    field(case, "U", "volVectorField", "[0 1 -1 0 0 0 0]", U, UU)
    field(case, "p", "volScalarField", "[1 -1 -2 0 0 0 0]", p, pp)
    field(case, "T", "volScalarField", "[0 0 0 1 0 0 0]", T, TT)


def seed_uz(case, amplitude, cells=range(25, 40)):
    """U_z = +-amplitude (alternating, + on odd cells) in the given cells of 0/U: the out-of-plane mode at the Mach disk."""
    p = f"{case}/0/U"
    s = open(p).read()
    i = s.index("internalField")
    j = s.index("(", s.index("\n", s.index("List<vector>", i)) + 1)
    k = s.index("\n)\n", j)
    rows = s[j + 1:k].strip().split("\n")
    for c in cells:
        v = rows[c].strip("()").split()
        v[2] = repr(amplitude * (1 if c % 2 else -1))
        rows[c] = "(" + " ".join(v) + ")"
    open(p, "w").write(s[:j + 1] + "\n" + "\n".join(rows) + s[k:])


def setup(kind, case):
    if kind in ("ladenburg", "ladenburg_long", "ladenburg_uzseed"):
        cd = f"{case}/system/controlDict"
        rv.edit(cd, r"^(writePrecision\s+)\S+;", r"\g<1>17;")
        if kind == "ladenburg_long":
            rv.edit(cd, r"^(endTime\s+)\S+;", r"\g<1>1e-04;")
            rv.edit(cd, r"^(writeInterval\s+)\S+;", r"\g<1>2e-05;")
        if kind == "ladenburg_uzseed":
            rv.edit(cd, r"^(endTime\s+)\S+;", r"\g<1>3e-05;")
            rv.edit(cd, r"^(writeInterval\s+)\S+;", r"\g<1>5e-06;")
            seed_uz(case, 1e-3)
        return
    rest = [(0.0, 0.0, 0.0)] * N
    if kind in ("reservoir", "reservoir_visc"):
        tube(kind, case, 10.0, 1.0, rest, [1.0] * N, [1.0] * N,
             [("outlet", {"p": wave(1.0, 1.0, 2.0), "T": plain("zeroGradient"), "U": plain("zeroGradient")}),
              ("inlet", {"p": total_p(2.0, 1.0), "T": total_T(1.2, 1.0), "U": plain("zeroGradient")})])
        if kind == "reservoir_visc":
            sutherland(case)
    elif kind == "pulse":
        tube(kind, case, 6.0, 0.5, rest, [1.5 if abs(x) < 0.5 else 1.0 for x in X], [1.0] * N,
             [("outlet", {"p": wave(1.0), "T": plain("zeroGradient"), "U": plain("zeroGradient")}),
              ("inlet", {"p": wave(1.0, 1.0, 1.0), "T": plain("zeroGradient"), "U": plain("zeroGradient")})])
    elif kind == "backflow":
        tube(kind, case, 4.0, 0.5, [(-0.3, 0.0, 0.0)] * N, [1.0] * N, [1.0] * N,
             [("outlet", {"p": plain("zeroGradient"), "T": plain("fixedValue", "1"),
                          "U": plain("fixedValue", "(-0.3 0 0)")}),
              ("inlet", {"p": total_p(1.1, 1.0), "T": total_T(1.1, 1.0), "U": plain("zeroGradient")})])
    elif kind == "conduction_suth":
        rv.setup("conduction", case)
        sutherland_after_transport(case)
    else:
        raise SystemExit(f"rho_central_nozzle: no case kind {kind}")


def sutherland_after_transport(case):
    """conduction as milestone G stages it (its fvSolution already reads `(e|h)`), then sutherland"""
    tp = f"{case}/constant/thermophysicalProperties"
    rv.edit(tp, r"^(\s*transport\s+)const;", r"\g<1>sutherland;")
    rv.edit(tp, r"^(\s*)mu\s+[^;]+;\s*\n\s*Pr\s+[^;]+;", r"\g<1>As              0.02;\n\g<1>Ts              1;")
    fs = f"{case}/system/fvSolution"
    rv.edit(fs, r"^(\s*tolerance\s+)[^;]+;", r"\g<1>1e-10;", count=2)
    rv.edit(fs, r"^(\s*relTol\s+)[^;]+;", r"\g<1>0;", count=2)


def has_value(path, patch):
    s = open(path).read()
    m = re.search(r"\n\s*%s\s*\{([^{}]*)\}" % re.escape(patch), s)
    return bool(m) and re.search(r"^\s*value\s", m.group(1), re.M) is not None


def patches(a, b, tname, names):
    """a zeroGradient (or any patch OpenFOAM writes no `value` for) is skipped: its value is its cell's"""
    out, where = {}, {}
    for f in ("p", "T", "U"):
        worst, scale, used = 0.0, 0.0, []
        for n in names.split(","):
            if not (has_value(f"{a}/{tname}/{f}", n) and has_value(f"{b}/{tname}/{f}", n)):
                continue
            va = np.atleast_2d(np.asarray(rf.patch_value(f"{a}/{tname}/{f}", n), float).T).T
            vb = np.atleast_2d(np.asarray(rf.patch_value(f"{b}/{tname}/{f}", n), float).T).T
            worst = max(worst, float(np.abs(va - vb).max()))
            scale = max(scale, float(np.abs(vb).max()))
            used.append(n)
        out[f] = ("%.3e" % (worst / scale if scale > 0 else worst)) if used else "none"
        where[f] = "+".join(used) if used else "-"
    print("patches " + " ".join(f"{f} {v}" for f, v in out.items()) + " on " +
          " ".join(f"{f}:{w}" for f, w in where.items()))


def branches(case, tname, patch, p0):
    pb = np.asarray(rf.patch_value(f"{case}/{tname}/p", patch), float)
    p0 = float(p0)
    out = int(np.sum(pb == p0))
    head = float(((p0 - pb) / p0).max())
    print(f"branches outflow {out} inflow {int(np.sum(pb < p0))} head {head:.6e}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "setup":
        setup(sys.argv[2], sys.argv[3])
    elif cmd == "patches":
        patches(*sys.argv[2:6])
    elif cmd == "branches":
        branches(*sys.argv[2:6])
    else:
        raise SystemExit(__doc__)
