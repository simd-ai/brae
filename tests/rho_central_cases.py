#!/usr/bin/env python3
"""Derived rhoCentralFoam cases for the milestone-C and -D gates (tests/rho_central_wedge_vs_openfoam.sh,
tests/rho_central_forwardstep_vs_openfoam.sh), each made from a tutorial as shipped by editing what the variant names and
nothing else.

  rho_central_cases.py symmetry <case>   bottom, top and the 15-degree obstacle become mesh patches of type `symmetry`
                                         and U, p, T `symmetry` on them: the non-planar constraint on an inclined plane
  rho_central_cases.py slipPT <case>     the obstacle's p and T become `slip` too (shipped: zeroGradient): a transform
                                         on the scalars of a non-constraint patch
  rho_central_cases.py cone <case>       the same geometry turned about the x axis: a 15-degree CONE in the Mach 5 stream,
                                         one cell thick between two `wedge` planes at +-2.5 degrees, the axis an `empty`
                                         patch of collapsed faces (as the movingCone tutorial builds its axis). Top is a
                                         `patch` with U slip, p and T zeroGradient; the cone is the tutorial's obstacle.
  rho_central_cases.py inletIO <case>    forwardStep: the inlet's U, p and T become `inletOutlet` with the shipped
                                         fixedValue as inletValue and value. The flux enters on every face, every step:
                                         the switch's inflow branch, on all three fields
  rho_central_cases.py openTop <case>    forwardStep: the top becomes a mesh `patch` and its U, p and T `inletOutlet`
                                         on the free stream ((3 0 0), 1, 1): an open top. The flux there is exactly 0
                                         until the bow shock arrives (pos0(0): the outflow branch), then leaves behind
                                         the shock and enters where the flow turns down: the switch takes both branches
  rho_central_cases.py sodRevSet <case>  shockTube BEFORE setFields: the left state starts at U (-50 0 0), flowing out
                                         of the left end, and the right state at (50 0 0), out of the right end
  rho_central_cases.py sodRev <case>     ...and AFTER setFields (on 0/): the ends' U and T become `inletOutlet` on
                                         inletValue (50 0 0) and 348.4, with no `value`, so the solver's read
                                         extrapolates the cells (a `value` setFields wrote there would be its own read's
                                         placeholder). Both ends flow out until the expansion reaches the left end and
                                         turns the flow there inward -- the switch changes branch at a resolved velocity,
                                         not on round-off noise -- and the inflow then holds at 50 m/s
  rho_central_cases.py piovBox <case>    shockTube's thermo, schemes and solution in a 2-D box (1 m, 40 x 40 cells) at
                                         rest at 1e5 Pa and 348.432 K, with a 3e5 Pa block over (0.15..0.45)^2
                                         (setFields): walls `slip` on three sides and the TOP open, U
                                         `pressureInletOutletVelocity`, p fixedValue 1e5, T inletOutlet. The top's flux is
                                         exactly 0 until the blast arrives, then leaves, and the expansion behind it
                                         draws it back in on part of the patch: both branches at resolved velocities,
                                         with a tangential component on the faces. deltaT 2e-6 fixed to 4e-3, written
                                         every 2.5e-4 at 17 digits

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


def inlet_outlet(path, patch, inlet, value):
    # 0/<field>: the patch's whole entry becomes inletOutlet on `inlet` (and `value`, unless None)
    body = f"\n        type            inletOutlet;\n        inletValue      uniform {inlet};"
    if value is not None:
        body += f"\n        value           uniform {value};"
    edit(path, rf"(\n\s*{patch}\s*\{{)[^}}]*\}}", rf"\g<1>{body}\n    }}")


def inletIO(case):
    for f, v in (("U", "(3 0 0)"), ("p", "1"), ("T", "1")):
        edit(f"{case}/0/{f}", r"(\n\s*inlet\s*\{\s*type\s+)fixedValue;", r"\g<1>fixedValue;")   # the shipped entry
        inlet_outlet(f"{case}/0/{f}", "inlet", v, v)


def openTop(case):
    patch_type(case, "top", "patch")
    for f, v in (("U", "(3 0 0)"), ("p", "1"), ("T", "1")):
        inlet_outlet(f"{case}/0/{f}", "top", v, v)


def sodRevSet(case):
    d = f"{case}/system/setFieldsDict"
    edit(d, r"volVectorFieldValue U \(0 0 0\)", "volVectorFieldValue U (-50 0 0)")
    edit(d, r"(fieldValues\s*\(\s*)", r"\g<1>volVectorFieldValue U (50 0 0)\n            ")


def sodRev(case):
    for f, v in (("U", "(50 0 0)"), ("T", "348.4")):
        edit(f"{case}/0/{f}", r"(\n\s*sides\s*\{\s*type\s+)zeroGradient;", r"\g<1>zeroGradient;")
        inlet_outlet(f"{case}/0/{f}", "sides", v, None)


def foam_file(cls, obj, body):
    return ("FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
            f"    class       {cls};\n    object      {obj};\n}}\n\n{body}")


def piovBox(case):
    # shockTube's thermophysicalProperties, turbulenceProperties, fvSchemes and fvSolution as shipped; the mesh, the
    # fields and the run written here. The fields are written whole (0/ is a copy of 0.orig), so no edit can miss.
    open(f"{case}/system/blockMeshDict", "w").write(foam_file("dictionary", "blockMeshDict", """scale   1;

vertices
(
    (0 0 0) (1 0 0) (1 1 0) (0 1 0) (0 0 0.025) (1 0 0.025) (1 1 0.025) (0 1 0.025)
);

blocks
(
    hex (0 1 2 3 4 5 6 7) (40 40 1) simpleGrading (1 1 1)
);

boundary
(
    top          { type patch; faces ((3 7 6 2)); }
    walls        { type wall;  faces ((0 4 7 3) (2 6 5 1) (1 5 4 0)); }
    frontAndBack { type empty; faces ((0 3 2 1) (4 5 6 7)); }
);
"""))
    open(f"{case}/system/setFieldsDict", "w").write(foam_file("dictionary", "setFieldsDict", """defaultFieldValues
(
    volScalarFieldValue p 100000
);

regions
(
    boxToCell
    {
        box (0.15 0.15 -1) (0.45 0.45 1);
        fieldValues
        (
            volScalarFieldValue p 300000
        );
    }
);
"""))
    open(f"{case}/system/controlDict", "w").write(foam_file("dictionary", "controlDict", """application     rhoCentralFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         0.004;
deltaT          2e-06;
writeControl    runTime;
writeInterval   0.00025;
purgeWrite      0;
writeFormat     ascii;
writePrecision  17;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable false;
adjustTimeStep  no;
"""))

    def field(name, cls, dim, internal, top, walls):
        open(f"{case}/0/{name}", "w").write(foam_file(cls, name, f"""dimensions      {dim};

internalField   uniform {internal};

boundaryField
{{
    top
    {{
{top}
    }}
    walls
    {{
{walls}
    }}
    frontAndBack
    {{
        type            empty;
    }}
}}
"""))
    field("U", "volVectorField", "[0 1 -1 0 0 0 0]", "(0 0 0)",
          "        type            pressureInletOutletVelocity;\n        value           uniform (0 0 0);",
          "        type            slip;")
    field("p", "volScalarField", "[1 -1 -2 0 0 0 0]", "100000",
          "        type            fixedValue;\n        value           uniform 100000;",
          "        type            zeroGradient;")
    field("T", "volScalarField", "[0 0 0 1 0 0 0]", "348.432",
          "        type            inletOutlet;\n        inletValue      uniform 348.432;\n        value           uniform 348.432;",
          "        type            zeroGradient;")


if __name__ == "__main__":
    kinds = {"symmetry": symmetry, "slipPT": slipPT, "cone": cone, "inletIO": inletIO, "openTop": openTop,
             "sodRevSet": sodRevSet, "sodRev": sodRev, "piovBox": piovBox}
    if len(sys.argv) != 3 or sys.argv[1] not in kinds:
        raise SystemExit(__doc__)
    kinds[sys.argv[1]](sys.argv[2])
