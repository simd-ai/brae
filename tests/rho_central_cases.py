#!/usr/bin/env python3
"""Derived rhoCentralFoam cases for the milestone-C gate (tests/rho_central_wedge_vs_openfoam.sh), each made from the
wedge15Ma5 tutorial as shipped by editing what the variant names and nothing else.

  rho_central_cases.py symmetry <case>   bottom, top and the 15-degree obstacle become mesh patches of type `symmetry`
                                         and U, p, T `symmetry` on them: the non-planar constraint on an inclined plane
  rho_central_cases.py slipPT <case>     the obstacle's p and T become `slip` too (shipped: zeroGradient): a transform
                                         on the scalars of a non-constraint patch
  rho_central_cases.py cone <case>       the same geometry turned about the x axis: a 15-degree CONE in the Mach 5 stream,
                                         one cell thick between two `wedge` planes at +-2.5 degrees, the axis an `empty`
                                         patch of collapsed faces (as the movingCone tutorial builds its axis). Top is a
                                         `patch` with U slip, p and T zeroGradient; the cone is the tutorial's obstacle.

<case> is a copy of the tutorial; the files are rewritten in place. Every edit is checked to have happened.
"""
import math
import re
import sys


def edit(path, pattern, repl, count=1, flags=re.S):
    s = open(path).read()
    t, n = re.subn(pattern, repl, s, count=count, flags=flags)
    if n != count and count != 0:
        raise SystemExit(f"rho_central_cases: {path}: expected {count} edit(s) of /{pattern}/, made {n}")
    open(path, 'w').write(t)


def patch_type(case, patch, typ):
    # blockMeshDict: `<patch> { type <old>;` -> `type <typ>;`
    edit(f"{case}/system/blockMeshDict", rf"(\n\s*{patch}\s*\{{\s*type\s+)\w+;", rf"\g<1>{typ};")


def field_type(case, field, patch, typ):
    # 0/<field>: the patch's whole entry becomes `{ type <typ>; }`
    edit(f"{case}/0/{field}", rf"(\n\s*{patch}\s*\{{)[^}}]*\}}", rf"\g<1>\n        type            {typ};\n    }}")


def symmetry(case):
    for p in ("bottom", "top", "obstacle"):
        patch_type(case, p, "symmetry")
        for f in ("U", "p", "T"):
            field_type(case, f, p, "symmetry")


def slipPT(case):
    for f in ("p", "T"):
        field_type(case, f, "obstacle", "slip")


def cone(case):
    t = math.tan(math.radians(2.5))
    x0, x1, x2 = -0.15242, 0.0, 0.3048
    r2, r3 = 0.081670913853, 0.1524      # the tutorial's ramp end (0.3048*tan 15 deg) and top
    pts = [
        (x0, 0, 0), (x1, 0, 0),                                      # 0 1: on the axis
        (x2, r2, -r2*t), (x0, r3, -r3*t), (x1, r3, -r3*t), (x2, r3, -r3*t),   # 2-5: back (z < 0)
        (x2, r2, r2*t), (x0, r3, r3*t), (x1, r3, r3*t), (x2, r3, r3*t),       # 6-9: front (z > 0)
    ]
    v = "\n".join(f"    ({p[0]:.12g} {p[1]:.12g} {p[2]:.12g})" for p in pts)
    bmd = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}

scale   1;

vertices
(
{v}
);

blocks
(
    hex (0 1 4 3 0 1 8 7) (40 40 1) simpleGrading (1 1 1)
    hex (1 2 5 4 1 6 9 8) (80 40 1) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    inlet    {{ type patch; faces ((0 0 7 3)); }}
    outlet   {{ type patch; faces ((2 5 9 6)); }}
    axis     {{ type empty; faces ((0 1 1 0)); }}
    top      {{ type patch; faces ((3 7 8 4) (4 8 9 5)); }}
    obstacle {{ type patch; faces ((1 2 6 1)); }}
    back     {{ type wedge; faces ((0 1 4 3) (1 2 5 4)); }}
    front    {{ type wedge; faces ((0 1 8 7) (1 6 9 8)); }}
);

mergePatchPairs
(
);
"""
    open(f"{case}/system/blockMeshDict", "w").write(bmd)
    for f in ("U", "p", "T"):
        path = f"{case}/0/{f}"
        s = open(path).read()
        # drop the planar case's bottom and defaultFaces, rename nothing else; add axis, front and back
        for p in ("bottom", "defaultFaces"):
            s, n = re.subn(rf"\n\s*{p}\s*\{{[^}}]*\}}", "", s, count=1, flags=re.S)
            if n != 1:
                raise SystemExit(f"rho_central_cases: {path}: no `{p}` entry to drop")
        add = ("\n    axis\n    {\n        type            empty;\n    }"
               "\n    front\n    {\n        type            wedge;\n    }"
               "\n    back\n    {\n        type            wedge;\n    }")
        s, n = re.subn(r"(boundaryField\s*\{)", lambda m: m.group(1) + add, s, count=1)
        if n != 1:
            raise SystemExit(f"rho_central_cases: {path}: no boundaryField")
        open(path, "w").write(s)
        top = "slip" if f == "U" else "zeroGradient"
        field_type(case, f, "top", top)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("symmetry", "slipPT", "cone"):
        raise SystemExit(__doc__)
    {"symmetry": symmetry, "slipPT": slipPT, "cone": cone}[sys.argv[1]](sys.argv[2])
