#!/usr/bin/env python3
"""Cases for the rhoCentralFoam gate of milestone I -- local time stepping, janaf, the rarefied-gas walls -- and what it
measures on them (no OpenFOAM needed).

  rho_central_lts.py setup <kind> <case>
      rewrites a staged case before (or, for biconic, after) its mesh:
        biconic         OpenFOAM's biconic25-55Run35 as shipped (localEuler, janaf + sutherland, maxwellSlipU and
                        smoluchowskiJumpT on the cone; its mesh from blockMesh, datToFoam grid256.dat, collapseEdges and
                        the wedge rename, as its Allrun), but for writePrecision 17 and writeFormat ascii
        fs_lts          forwardStep as shipped with ddtSchemes localEuler, to t = 0.5, writes every 0.1 (inviscid hConst)
        ob_lts_visc     obliqueShock as milestone G stages it (mu 0.01, Pr 0.72) with ddtSchemes localEuler, to t = 0.5,
                        writes every 0.125
        sod_janaf       the 1000-cell tube (x in [-5, 5] m) of N2 on biconic's janaf coefficients, p 1e5 | 1e4 Pa,
                        T 1500 | 600 K: the shock and the contact cross Tcommon = 1000 K; sutherland as biconic's (OpenFOAM
                        builds janaf psiThermo with sutherland only) and its U and e solves; t = 3e-3, writes 1e-3
        sod_janaf_visc  the same with sutherland (As 1.458e-3, 1000 times air's, so diffusion spans cells; Ts 110) and the
                        U and e solves as biconic25-55Run35 declares them
        channel_slip    a rarefied channel on biconic25-55Run35's system/ and thermo (0.05 x 0.01 m, 50 x 20; p 10 Pa, T 200 K,
                        U 1000 m/s: Knudsen about 0.04), its floor maxwellSlipU (thermalCreep, curvature) and
                        smoluchowskiJumpT (Twall 300 K), its top a symmetryPlane; Euler; t = 1e-4, writes 2.5e-5; U and e
                        solved as LadenburgJet60psi ships them (tolerance 1e-10, relTol 0)
  rho_central_lts.py flow <log>
      setRDeltaT.H's "Flow time scale min/max" lines, one per step: `flow <step> <min> <max>`
  rho_central_lts.py flowcmp <log a> <log b>
      the largest relative difference of the flow time scales step by step: `flowcmp <steps a> <steps b> <worst> <step>`
  rho_central_lts.py janaf <case> <time>
      the T range of a written time and how many cells sit on each side of Tcommon (1000): `janaf <Tmin> <Tmax> <low> <high>`
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rho_central_fields as rf      # noqa: E402
import rho_central_viscous as rv     # noqa: E402

N = 1000
X = [-5 + (i + 0.5)*0.01 for i in range(N)]

# biconic25-55Run35's mixture, verbatim (nitrogen)
JANAF = """    specie
    {
        molWeight       28.01348;
    }
    thermodynamics
    {
        Tlow            100;
        Thigh           10000;
        Tcommon         1000;
        highCpCoeffs    ( 2.9525407 0.0013968838 -4.9262577e-07 7.8600091e-11 -4.6074978e-15 -923.93753 5.8718221 );
        lowCpCoeffs     ( 3.5309628 -0.0001236595 -5.0299339e-07 2.4352768e-09 -1.4087954e-12 -1046.9637 2.9674391 );
    }
"""


def thermo(case, transport):
    tr = {"const": "const", "sutherland": "sutherland"}[transport[0]]
    body = ("        mu              %r;\n        Pr              0.72;\n" % transport[1] if tr == "const" else
            "        As              %r;\n        Ts              %r;\n        Pr              0.72;\n" % transport[1:])
    s = ("FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       dictionary;\n"
         "    object      thermophysicalProperties;\n}\n\nthermoType\n{\n    type            hePsiThermo;\n"
         "    mixture         pureMixture;\n    transport       %s;\n    thermo          janaf;\n"
         "    equationOfState perfectGas;\n    specie          specie;\n    energy          sensibleInternalEnergy;\n}\n\n"
         "mixture\n{\n%s    transport\n    {\n%s    }\n}\n" % (tr, JANAF, body))
    open(f"{case}/constant/thermophysicalProperties", "w").write(s)


def solves(case, tol="1e-09", relU="0.01", relE="0.1"):
    """biconic25-55Run35's fvSolution; with tol 1e-10 and relTol 0, LadenburgJet60psi's U and e solves"""
    s = ("FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       dictionary;\n    object      fvSolution;\n}\n\n"
         "solvers\n{\n    \"(rho|rhoU|rhoE)\"\n    {\n        solver          diagonal;\n    }\n\n"
         "    U\n    {\n        solver          smoothSolver;\n        smoother        GaussSeidel;\n"
         f"        nSweeps         2;\n        tolerance       {tol};\n        relTol          {relU};\n    }}\n\n"
         f"    e\n    {{\n        $U;\n        relTol          {relE};\n    }}\n}}\n")
    open(f"{case}/system/fvSolution", "w").write(s)


def lts(case):
    rv.edit(f"{case}/system/fvSchemes", r"^(\s*default\s+)Euler;", r"\g<1>localEuler;")


def output(case):
    cd = f"{case}/system/controlDict"
    rv.edit(cd, r"^(writePrecision\s+)\S+;", r"\g<1>17;")
    rv.edit(cd, r"^(writeFormat\s+)\S+;", r"\g<1>ascii;")


def tube_janaf(case, transport):
    rv.tube_mesh(case, False)
    thermo(case, transport)
    cd = f"{case}/system/controlDict"
    rv.edit(cd, r"^(endTime\s+)\S+;", r"\g<1>0.003;")
    rv.edit(cd, r"^(writeControl\s+)\S+;", r"\g<1>adjustable;")
    rv.edit(cd, r"^(writeInterval\s+)\S+;", r"\g<1>0.001;")
    rv.edit(cd, r"^(deltaT\s+)\S+;", r"\g<1>1e-07;")
    rv.edit(cd, r"^(writePrecision\s+)\S+;", r"\g<1>17;")
    os.makedirs(f"{case}/0", exist_ok=True)
    patches = [("sides", "zeroGradient"), ("empty", "empty")]
    rv.field(case, "U", "volVectorField", "[0 1 -1 0 0 0 0]", [(0.0, 0.0, 0.0)]*N, patches)
    rv.field(case, "p", "volScalarField", "[1 -1 -2 0 0 0 0]", [1e5 if x < 0 else 1e4 for x in X], patches)
    rv.field(case, "T", "volScalarField", "[0 0 0 1 0 0 0]", [1500.0 if x < 0 else 600.0 for x in X], patches)
    solves(case)


CHANNEL_MESH = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}

scale 1;

vertices
(
    (0 0 0) (0.05 0 0) (0.05 0.01 0) (0 0.01 0)
    (0 0 0.001) (0.05 0 0.001) (0.05 0.01 0.001) (0 0.01 0.001)
);

blocks
(
    hex (0 1 2 3 4 5 6 7) (50 20 1) simpleGrading (1 1 1)
);

boundary
(
    inlet { type patch; faces ((0 4 7 3)); }
    outlet { type patch; faces ((1 2 6 5)); }
    wall { type wall; faces ((0 1 5 4)); }
    top { type symmetryPlane; faces ((3 7 6 2)); }
    frontAndBack { type empty; faces ((0 3 2 1) (4 5 6 7)); }
);
"""


def entry(name, body):
    return "    %s\n    {\n%s    }\n" % (name, "".join("        %s\n" % l for l in body))


def channel(case):
    """a rarefied channel for maxwellSlipU and smoluchowskiJumpT alone, staged on biconic25-55Run35's system/ and
    thermo (its schemes, solves and mixture), Euler in time"""
    open(f"{case}/system/blockMeshDict", "w").write(CHANNEL_MESH)
    # U and e solved as LadenburgJet60psi ships them (tolerance 1e-10, relTol 0), not biconic's relTol 0.01 | 0.1: those
    # stop at two sweeps, and OpenFOAM's GaussSeidel in parallel (processor faces explicit) then parts from its serial
    # by 1e-3 (the gate's first full run, 2026-11-18) -- a spread too wide to bound the walls or show a control red
    solves(case, "1e-10", "0", "0")
    rv.edit(f"{case}/system/fvSchemes", r"^(\s*default\s+)localEuler;", r"\g<1>Euler;")
    cd = f"{case}/system/controlDict"
    s = open(cd).read()
    for k, v in (("endTime", "0.0001"), ("deltaT", "1e-09"), ("writeControl", "adjustable"), ("writeInterval", "2.5e-05"),
                 ("writePrecision", "17"), ("writeFormat", "ascii"), ("adjustTimeStep", "yes"), ("maxCo", "0.3")):
        s = re.sub(r"^(%s\s+)\S+;" % k, r"\g<1>%s;" % v, s, count=1, flags=re.M)
    s = re.sub(r"^(maxDeltaT\s+)\S+;", r"\g<1>1;", s, count=1, flags=re.M)
    open(cd, "w").write(s)
    os.makedirs(f"{case}/0", exist_ok=True)
    head = ("FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       %s;\n    object      %s;\n}\n"
            "dimensions      %s;\n\ninternalField   uniform %s;\n\nboundaryField\n{\n")
    U = head % ("volVectorField", "U", "[0 1 -1 0 0 0 0]", "(1000 0 0)")
    U += entry("inlet", ["type            fixedValue;", "value           uniform (1000 0 0);"])
    U += entry("outlet", ["type            zeroGradient;"])
    U += entry("wall", ["type            maxwellSlipU;", "accommodationCoeff 0.85;", "Uwall           uniform (0 0 0);",
                        "thermalCreep    on;", "curvature       on;", "value           uniform (1000 0 0);"])
    U += entry("top", ["type            symmetryPlane;"]) + entry("frontAndBack", ["type            empty;"]) + "}\n"
    T = head % ("volScalarField", "T", "[0 0 0 1 0 0 0]", "200")
    T += entry("inlet", ["type            fixedValue;", "value           uniform 200;"])
    T += entry("outlet", ["type            zeroGradient;"])
    T += entry("wall", ["type            smoluchowskiJumpT;", "accommodationCoeff 0.85;", "Twall           uniform 300;",
                        "gamma           1.4;", "value           uniform 200;"])
    T += entry("top", ["type            symmetryPlane;"]) + entry("frontAndBack", ["type            empty;"]) + "}\n"
    p = head % ("volScalarField", "p", "[1 -1 -2 0 0 0 0]", "10")
    p += entry("inlet", ["type            fixedValue;", "value           uniform 10;"])
    p += entry("outlet", ["type            zeroGradient;"])
    p += entry("wall", ["type            zeroGradient;"])
    p += entry("top", ["type            symmetryPlane;"]) + entry("frontAndBack", ["type            empty;"]) + "}\n"
    for n, t in (("U", U), ("T", T), ("p", p)):
        open(f"{case}/0/{n}", "w").write(t)


def setup(kind, case):
    if kind == "biconic":
        output(case)
    elif kind == "fs_lts":
        lts(case)
        cd = f"{case}/system/controlDict"
        rv.edit(cd, r"^(endTime\s+)\S+;", r"\g<1>0.5;")
        rv.edit(cd, r"^(writeInterval\s+)\S+;", r"\g<1>0.1;")
        rv.edit(cd, r"^(writePrecision\s+)\S+;", r"\g<1>17;")
    elif kind == "ob_lts_visc":
        rv.setup("oblique", case)
        lts(case)
        cd = f"{case}/system/controlDict"
        rv.edit(cd, r"^(endTime\s+)\S+;", r"\g<1>0.5;")
        rv.edit(cd, r"^(writeInterval\s+)\S+;", r"\g<1>0.125;")
    elif kind == "sod_janaf":
        tube_janaf(case, ("sutherland", 1.458e-06, 110.0))
    elif kind == "sod_janaf_visc":
        tube_janaf(case, ("sutherland", 1.458e-03, 110.0))
    elif kind == "channel_slip":
        channel(case)
    else:
        raise SystemExit(f"rho_central_lts: no case kind {kind}")


def flow(log):
    out = []
    for line in open(log):
        m = re.match(r"^Flow time scale min/max = (\S+), (\S+)", line)
        if m:
            out.append((float(m.group(1)), float(m.group(2))))
    return out


def main(a):
    if a[0] == "setup":
        setup(a[1], a[2])
    elif a[0] == "flow":
        for i, (mn, mx) in enumerate(flow(a[1])):
            print(f"flow {i + 1} {mn!r} {mx!r}")
    elif a[0] == "flowcmp":
        x, y = flow(a[1]), flow(a[2])
        worst, at = 0.0, 0
        for i in range(min(len(x), len(y))):
            d = max(abs(x[i][0] - y[i][0])/abs(y[i][0]), abs(x[i][1] - y[i][1])/abs(y[i][1]))
            if d > worst:
                worst, at = d, i + 1
        print(f"flowcmp {len(x)} {len(y)} {worst:.3e} {at}")
    elif a[0] == "janaf":
        T = np.asarray(rf.internal(f"{a[1]}/{a[2]}/T"), float)
        print(f"janaf {T.min():.6g} {T.max():.6g} {int((T < 1000).sum())} {int((T >= 1000).sum())}")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
