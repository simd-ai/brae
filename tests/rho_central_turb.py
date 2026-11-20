#!/usr/bin/env python3
"""Cases for the rhoCentralFoam gate of milestone J -- compressible RAS (kOmegaSST) on the central path -- (no
OpenFOAM needed).

  rho_central_turb.py setup <kind> <case>
      writes a case (system/, constant/, 0/) into <case>, staged on OpenFOAM's LadenburgJet60psi system/ (Kurganov,
      vanLeer/vanLeerV, the U and e solves) with the turbulence entries a RAS model needs:
        plate_sst   a supersonic turbulent flat plate: air (hConst, sutherland), Mach 1.9 (U 650 m/s, T 290 K,
                    p 1e4 Pa), a symmetry plane ahead of the plate's leading edge, the plate a wall with OpenFOAM's
                    wall functions (nutkWallFunction, kqRWallFunction, omegaWallFunction,
                    compressible::alphatWallFunction), T 290 K; inflow k from 1% intensity, omega from a 1 mm length;
                    110 x 50 cells graded to the wall; Euler, maxCo 0.2, to 3e-3 s (about 3.5 plate lengths of flow)
        plate_lts   the same with ddtSchemes localEuler, 1500 steps
        jet_sst     LadenburgJet60psi (its 0.orig unpacked into 0) with simulationType RAS kOmegaSST, its system/ as
                    shipped plus the turbulence entries: an axisymmetric wedge with no wall-type patch (the wall distance
                    GREAT everywhere, F1 0: kOmegaSST's k-epsilon branch in the jet's shear layers); k from 1% intensity
                    at the nozzle (inlet) and 0.1% of the jet's velocity elsewhere, omega from a 0.5 mm length (a tenth
                    of the nozzle radius); inlet fixedValue, freestreamInlet fixedValue, freestream and outlet
                    inletOutlet; nut and alphat calculated; to 2e-5 s as shipped, writes every 5e-6
      maxCo 0.2, not the jet's 0.5 nor 0.4: at 0.4 on this mesh (near-wall cells 33 times longer than high) OpenFOAM's
      own run grows a grid-scale wave train from the leading edge, laminar and RAS alike (p roughness along the first
      row 1e-2 at 5e-5 s, 1e-1 at 5e-4 s; its two- and four-way runs part from serial by 10% by 5e-4 s); at 0.2 and
      0.1 the first rows stay smooth (roughness 2e-4, flat), 2026-11-18
  rho_central_turb.py premise <case> <time> [wall patch]
      what makes the case turbulent, from a written time: nut/nu (nu = mu/rho, sutherland mu(T) and rho = p/(R T) from
      the case's thermophysicalProperties) over the cells, and on the wall patch the faces whose nut is not 0 (the wall
      function above yPlusLam): `premise <max nut/nu> <median nut/nu> <faces nut_w > 0> <faces>`
"""
import os
import re
import shutil
import sys

HEAD = ("FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       %s;\n    object      %s;\n}\n"
        "dimensions      %s;\n\ninternalField   uniform %s;\n\nboundaryField\n{\n")


def entry(name, body):
    return "    %s\n    {\n%s    }\n" % (name, "".join("        %s\n" % l for l in body))


def write(case, name, cls, dims, internal, patches):
    s = HEAD % (cls, name, dims, internal) + "".join(entry(p, b) for p, b in patches) + "}\n"
    open(f"{case}/0/{name}", "w").write(s)


TURB_PROPS = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      turbulenceProperties;
}

simulationType  RAS;

RAS
{
    RASModel        kOmegaSST;
    turbulence      on;
    printCoeffs     on;
}
"""

AIR = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      thermophysicalProperties;
}

thermoType
{
    type            hePsiThermo;
    mixture         pureMixture;
    transport       sutherland;
    thermo          hConst;
    equationOfState perfectGas;
    specie          specie;
    energy          sensibleInternalEnergy;
}

mixture
{
    specie
    {
        molWeight       28.96;
    }
    thermodynamics
    {
        Cp              1004.5;
        Hf              0;
    }
    transport
    {
        As              1.458e-06;
        Ts              110.4;
    }
}
"""

PLATE_MESH = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}

scale 1;

vertices
(
    (-0.05 0 0) (0 0 0) (0.5 0 0) (0.5 0.08 0) (0 0.08 0) (-0.05 0.08 0)
    (-0.05 0 0.01) (0 0 0.01) (0.5 0 0.01) (0.5 0.08 0.01) (0 0.08 0.01) (-0.05 0.08 0.01)
);

blocks
(
    hex (0 1 4 5 6 7 10 11) (10 50 1) simpleGrading (0.5 40 1)
    hex (1 2 3 4 7 8 9 10) (100 50 1) simpleGrading (4 40 1)
);

boundary
(
    inlet { type patch; faces ((0 6 11 5)); }
    outlet { type patch; faces ((2 3 9 8)); }
    top { type patch; faces ((5 11 10 4) (4 10 9 3)); }
    symmetryAhead { type symmetryPlane; faces ((0 1 7 6)); }
    plate { type wall; faces ((1 2 8 7)); }
    frontAndBack { type empty; faces ((0 5 4 1) (1 4 3 2) (6 7 10 11) (7 8 9 10)); }
);
"""


def schemes(case):
    """the jet's fvSchemes plus the turbulence entries: div(phi,k|omega) bounded-free upwind-limited (Gauss
    limitedLinear 1, as OpenFOAM's compressible RAS tutorials write them), the wall distance meshWave"""
    p = f"{case}/system/fvSchemes"
    s = open(p).read()
    s = re.sub(r"(divSchemes\s*\{)", r"\1\n    div(phi,k)      Gauss limitedLinear 1;\n    div(phi,omega)  Gauss limitedLinear 1;",
               s, count=1)
    if "wallDist" not in s:
        s += "\nwallDist\n{\n    method meshWave;\n}\n"
    open(p, "w").write(s)
    p = f"{case}/system/fvSolution"
    s = open(p).read()
    s = s.replace("solvers\n{", "solvers\n{\n    \"(k|omega)\"\n    {\n        solver          smoothSolver;\n"
                  "        smoother        GaussSeidel;\n        nSweeps         2;\n        tolerance       1e-10;\n"
                  "        relTol          0;\n    }\n", 1)
    open(p, "w").write(s)


def control(case, end, every, maxco="0.2"):
    p = f"{case}/system/controlDict"
    s = open(p).read()
    for k, v in (("startFrom", "startTime"), ("startTime", "0"), ("endTime", end), ("deltaT", "1e-09"),
                 ("writeControl", "adjustable"), ("writeInterval", every), ("writePrecision", "17"),
                 ("adjustTimeStep", "yes"), ("maxCo", maxco), ("maxDeltaT", "1")):
        s = re.sub(r"^(%s\s+)\S+;" % k, r"\g<1>%s;" % v, s, count=1, flags=re.M)
    open(p, "w").write(s)


def plate(case, lts):
    open(f"{case}/system/blockMeshDict", "w").write(PLATE_MESH)
    open(f"{case}/constant/thermophysicalProperties", "w").write(AIR)
    open(f"{case}/constant/turbulenceProperties", "w").write(TURB_PROPS)
    schemes(case)
    if lts:
        p = f"{case}/system/fvSchemes"
        s = open(p).read()
        s = re.sub(r"^(\s*default\s+)Euler;", r"\g<1>localEuler;", s, count=1, flags=re.M)
        open(p, "w").write(s)
        control(case, "1.5e-06", "5e-07")   # 1500 steps of the controlDict deltaT 1e-9
    else:
        control(case, "0.003", "0.0005")
    shutil.rmtree(f"{case}/0", ignore_errors=True)
    os.makedirs(f"{case}/0")
    U0, T0, p0 = "(650 0 0)", "290", "10000"
    k0, w0 = 1.5*(0.01*650)**2, (1.5*(0.01*650)**2)**0.5/(0.09**0.25*0.001)
    inflow = lambda v: ["type            fixedValue;", f"value           uniform {v};"]
    zg = ["type            zeroGradient;"]
    sym = ["type            symmetryPlane;"]
    emp = ["type            empty;"]
    io = lambda v: ["type            inletOutlet;", f"inletValue      uniform {v};", f"value           uniform {v};"]
    write(case, "U", "volVectorField", "[0 1 -1 0 0 0 0]", U0,
          [("inlet", inflow(U0)), ("outlet", zg), ("top", zg), ("symmetryAhead", sym),
           ("plate", ["type            noSlip;"]), ("frontAndBack", emp)])
    write(case, "p", "volScalarField", "[1 -1 -2 0 0 0 0]", p0,
          [("inlet", inflow(p0)), ("outlet", zg), ("top", zg), ("symmetryAhead", sym), ("plate", zg),
           ("frontAndBack", emp)])
    write(case, "T", "volScalarField", "[0 0 0 1 0 0 0]", T0,
          [("inlet", inflow(T0)), ("outlet", zg), ("top", zg), ("symmetryAhead", sym),
           ("plate", ["type            fixedValue;", f"value           uniform {T0};"]), ("frontAndBack", emp)])
    write(case, "k", "volScalarField", "[0 2 -2 0 0 0 0]", repr(k0),
          [("inlet", inflow(repr(k0))), ("outlet", io(repr(k0))), ("top", io(repr(k0))), ("symmetryAhead", sym),
           ("plate", ["type            kqRWallFunction;", f"value           uniform {k0!r};"]), ("frontAndBack", emp)])
    write(case, "omega", "volScalarField", "[0 0 -1 0 0 0 0]", repr(w0),
          [("inlet", inflow(repr(w0))), ("outlet", io(repr(w0))), ("top", io(repr(w0))), ("symmetryAhead", sym),
           ("plate", ["type            omegaWallFunction;", f"value           uniform {w0!r};"]), ("frontAndBack", emp)])
    write(case, "nut", "volScalarField", "[0 2 -1 0 0 0 0]", "0",
          [("inlet", ["type            calculated;", "value           uniform 0;"]),
           ("outlet", ["type            calculated;", "value           uniform 0;"]),
           ("top", ["type            calculated;", "value           uniform 0;"]), ("symmetryAhead", sym),
           ("plate", ["type            nutkWallFunction;", "value           uniform 0;"]), ("frontAndBack", emp)])
    write(case, "alphat", "volScalarField", "[1 -1 -1 0 0 0 0]", "0",
          [("inlet", ["type            calculated;", "value           uniform 0;"]),
           ("outlet", ["type            calculated;", "value           uniform 0;"]),
           ("top", ["type            calculated;", "value           uniform 0;"]), ("symmetryAhead", sym),
           ("plate", ["type            compressible::alphatWallFunction;", "Prt             0.85;", "value           uniform 0;"]),
           ("frontAndBack", emp)])


def jet(case):
    """LadenburgJet60psi with kOmegaSST: the case as shipped (0 from its 0.orig) plus the RAS model's entries"""
    open(f"{case}/constant/turbulenceProperties", "w").write(TURB_PROPS)
    schemes(case)
    p = f"{case}/system/controlDict"
    s = open(p).read()
    for k, v in (("writeInterval", "5e-06"), ("writePrecision", "17")):
        s = re.sub(r"^(%s\s+)\S+;" % k, r"\g<1>%s;" % v, s, count=1, flags=re.M)
    open(p, "w").write(s)
    Ujet, L = 315.6, 0.0005
    kj, ka = 1.5*(0.01*Ujet)**2, 1.5*(0.001*Ujet)**2
    wj, wa = kj**0.5/(0.09**0.25*L), ka**0.5/(0.09**0.25*L)
    fixed = lambda v: ["type            fixedValue;", f"value           uniform {v!r};"]
    io = lambda v: ["type            inletOutlet;", f"inletValue      uniform {v!r};", f"value           uniform {v!r};"]
    calc = ["type            calculated;", "value           uniform 0;"]
    wedge = ["type            wedge;"]
    for name, dims, inside, jetv in (("k", "[0 2 -2 0 0 0 0]", ka, kj), ("omega", "[0 0 -1 0 0 0 0]", wa, wj)):
        write(case, name, "volScalarField", dims, repr(inside),
              [("inlet", fixed(jetv)), ("outlet", io(inside)), ("freestreamInlet", fixed(inside)),
               ("freestream", io(inside)), ("wedge1", wedge), ("wedge2", wedge)])
    for name, dims in (("nut", "[0 2 -1 0 0 0 0]"), ("alphat", "[1 -1 -1 0 0 0 0]")):
        write(case, name, "volScalarField", dims, "0",
              [("inlet", calc), ("outlet", calc), ("freestreamInlet", calc), ("freestream", calc),
               ("wedge1", wedge), ("wedge2", wedge)])


def setup(kind, case):
    if kind in ("plate_sst", "plate_lts"):
        plate(case, kind == "plate_lts")
    elif kind == "jet_sst":
        jet(case)
    else:
        raise SystemExit(f"rho_central_turb: no case kind {kind}")


def premise(case, time, wall=None):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import numpy as np
    import rho_central_fields as rf
    tp = open(f"{case}/constant/thermophysicalProperties").read()
    num = lambda k: float(re.search(r"\b%s\s+([0-9.eE+-]+)\s*;" % k, tp).group(1))
    As, Ts, W = num("As"), num("Ts"), num("molWeight")
    d = f"{case}/{time}"
    nut = np.asarray(rf.internal(f"{d}/nut"), float)
    T = np.asarray(rf.internal(f"{d}/T"), float)
    p = np.asarray(rf.internal(f"{d}/p"), float)
    rho = p/(8314.47/W*T)
    r = nut/(As*np.sqrt(T)/(1 + Ts/T)/rho)
    on, n = 0, 0
    if wall:
        w = np.asarray(rf.patch_value(f"{d}/nut", wall), float)
        on, n = int((w > 0).sum()), int(w.size)
    print("premise %.4g %.4g %d %d" % (r.max(), np.median(r), on, n))


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 3 and a[0] == "setup":
        setup(a[1], a[2])
    elif len(a) in (3, 4) and a[0] == "premise":
        premise(*a[1:])
    else:
        raise SystemExit(__doc__)
