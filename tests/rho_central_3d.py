#!/usr/bin/env python3
"""Three-dimensional cases for the rhoCentralFoam gate tests/rho_central_3d_vs_openfoam.sh, and the engineering
quantities measured on them (no OpenFOAM needed). v2412 ships no 3-D rhoCentralFoam tutorial; these are built here.

  rho_central_3d.py setup <kind> <case> [level] [lts]
      writes system/, constant/ and 0/ into <case>, staged on OpenFOAM's LadenburgJet60psi system/ (Kurganov,
      vanLeer/vanLeerV, Gauss linear, the U and e solves: smoothSolver GaussSeidel, tolerance 1e-10). Air throughout:
      perfectGas, hConst (Cp 1004.5, molWeight 28.96: R 287.1, gamma 1.4002), sensibleInternalEnergy.
        corner_inv   an axial corner: Mach 3 air (T 300 K, p 1e4 Pa, U 1041.6 m/s) along a floor (y = 0) and a side
                     wall (z = 0) that both turn into a 10-degree compression ramp at x = 0.3 m; x in [0, 1] m,
                     y and z in [0, 0.5] m. Inviscid (const, mu 0). Floor and side slip; inlet fixedValue; outlet and
                     top (y = 0.5) inletOutlet at the free stream; z = 0.5 a symmetryPlane. Two blocks, (12 + 24) x 16
                     x 16 = 9216 cells at level 0, each level doubling every direction. Euler, maxCo 0.5, to 2e-3 s
                     (two flow lengths), writes every 5e-4
        corner_lam   the same corner, laminar: p 1e3 Pa (Re 6.6e5 a metre), sutherland (As 1.458e-6, Ts 110.4,
                     Pr 0.72 through hConst's Pr), floor and side noSlip at T 300 K, cells graded 50:1 towards them.
                     Euler, maxCo 0.2, to 1e-3 s, writes every 2.5e-4
        corner_sst   the same corner, turbulent: p 1e5 Pa (Re 6.6e7 a metre), kOmegaSST with OpenFOAM's wall functions
                     on floor and side (nutkWallFunction, kqRWallFunction, omegaWallFunction,
                     compressible::alphatWallFunction), graded 200:1; inflow k from 1 % intensity, omega from a 1 mm
                     length. Euler, maxCo 0.2, to 1e-3 s, writes every 2.5e-4
        box_blast    a closed 1 m cube, 24^3 cells: a sphere of radius 0.2 m at (0.35, 0.4, 0.45) at p 1e5 Pa in air at
                     1e4 Pa, T 300 K, at rest. Inviscid. x = 0 a symmetryPlane, the other five faces slip walls: no
                     mass or energy crosses them. Euler, maxCo 0.5, to 3e-3 s, writes every 5e-4
        box_restart  box_blast's case, to be started from its 1.5e-3 (the gate copies the time in)
        duct_tp      a square duct 0.3 x 0.02 x 0.02 m, 30 x 10 x 10 cells: air at rest (1e5 Pa, 300 K) driven by a
                     totalPressure inlet (p0 1.5e5 Pa, psi form) with totalTemperature (T0 300 K) and U zeroGradient,
                     a waveTransmissive outlet (fieldInf 1e5 Pa, lInf 0.3 m), noSlip walls at y = 0, y = 0.02 and
                     z = 0 (T zeroGradient), a symmetryPlane at z = 0.02. Laminar, sutherland. Euler, maxCo 0.3, to
                     2e-3 s, writes every 5e-4
      level (corner kinds): 0 (default), 1, 2, 3: the cells of every direction times 2^level.
      lts: ddtSchemes localEuler (controlDict's maxCo 0.5 and maxDeltaT 1, rDeltaTSmoothingCoeff 0.02 by default), for
      the converged states of the refinement study; the end time then counts pseudo-steps of the controlDict's deltaT 1.

  rho_central_3d.py ramp <case> <time>
      the corner's engineering quantities from a written time: the floor pressure on the ramp, area-averaged over
      x in [0.55, 0.75] and z in [0.35, 0.5] (away from the corner), over the free stream's 1e4|1e3|1e5 Pa, and the
      oblique-shock theory for Mach 3 and 10 degrees (gamma from the thermo), `ramp <p/p_inf> <theory> <shock angle>`
      where the shock angle is where p falls through the mean of free stream and ramp, up the column of cells nearest
      x = 0.85 next to the symmetry plane, seen from the ramp's start (x = 0.3, y = 0)
  rho_central_3d.py wall <case> <time> <patch>
      the wall's shear stress and heat flux from the first cell (the patch's snGrad): area-averaged over the floor's
      flat part x in [0.1, 0.25], z in [0.3, 0.5], `wall <tau_w Pa> <q_w W/m2> <Cf> <Re_x at 0.175>`
"""
import math
import os
import re
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

R_AIR = 8314.47/28.96
CP = 1004.5
GAMMA = CP/(CP - R_AIR)
T_INF = 300.0
MACH = 3.0
U_INF = round(MACH*math.sqrt(GAMMA*R_AIR*T_INF), 1)   # 1041.6
X0, L, H, W, THETA = 0.3, 1.0, 0.5, 0.5, math.radians(10.0)

HEAD = ("FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       %s;\n    object      %s;\n}\n"
        "dimensions      %s;\n\ninternalField   %s;\n\nboundaryField\n{\n")
DICT = "FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       dictionary;\n    object      %s;\n}\n"


def entry(name, body):
    return "    %s\n    {\n%s    }\n" % (name, "".join("        %s\n" % l for l in body))


def write(case, name, cls, dims, internal, patches):
    s = HEAD % (cls, name, dims, internal) + "".join(entry(p, b) for p, b in patches) + "}\n"
    open(f"{case}/0/{name}", "w").write(s)


def thermo(case, viscous):
    transport = ("sutherland", "        As              1.458e-06;\n        Ts              110.4;\n") if viscous \
        else ("const", "        mu              0;\n        Pr              1;\n")
    pr = "        Pr              0.72;\n" if viscous else ""
    open(f"{case}/constant/thermophysicalProperties", "w").write(
        DICT % "thermophysicalProperties" +
        "\nthermoType\n{\n    type            hePsiThermo;\n    mixture         pureMixture;\n"
        f"    transport       {transport[0]};\n    thermo          hConst;\n    equationOfState perfectGas;\n"
        "    specie          specie;\n    energy          sensibleInternalEnergy;\n}\n\nmixture\n{\n"
        "    specie\n    {\n        molWeight       28.96;\n    }\n"
        f"    thermodynamics\n    {{\n        Cp              {CP};\n        Hf              0;\n    }}\n"
        f"    transport\n    {{\n{transport[1]}{pr}    }}\n}}\n")
    # sutherland's Pr: OpenFOAM's sutherlandTransport takes no Pr; the Prandtl number of hConst/sutherland is
    # Cp mu/kappa with kappa from the modified Eucken correlation -- the entry above is read by const only, and
    # written for sutherland so the dictionary carries the intended value; nothing reads it there.


def control(case, end, every, maxco, lts=False):
    p = f"{case}/system/controlDict"
    s = open(p).read()
    for k, v in (("startFrom", "startTime"), ("startTime", "0"), ("endTime", end),
                 ("deltaT", "1" if lts else "1e-09"), ("writeControl", "runTime" if lts else "adjustable"),
                 ("writeInterval", every), ("writePrecision", "17"), ("adjustTimeStep", "no" if lts else "yes"),
                 ("maxCo", maxco), ("maxDeltaT", "1")):
        s = re.sub(r"^(%s\s+)\S+;" % k, r"\g<1>%s;" % v, s, count=1, flags=re.M)
    open(p, "w").write(s)


def lts_schemes(case):
    p = f"{case}/system/fvSchemes"
    s = open(p).read()
    s = re.sub(r"^(\s*default\s+)Euler;", r"\g<1>localEuler;", s, count=1, flags=re.M)
    open(p, "w").write(s)
    # setRDeltaT.H reads maxCo, maxDeltaT and rDeltaTSmoothingCoeff (default 0.02) from controlDict: control() sets them


def turb_schemes(case):
    p = f"{case}/system/fvSchemes"
    s = open(p).read()
    s = re.sub(r"(divSchemes\s*\{)", r"\1\n    div(phi,k)      Gauss limitedLinear 1;\n"
               r"    div(phi,omega)  Gauss limitedLinear 1;", s, count=1)
    if "wallDist" not in s:
        s += "\nwallDist\n{\n    method meshWave;\n}\n"
    open(p, "w").write(s)
    p = f"{case}/system/fvSolution"
    s = open(p).read()
    s = s.replace("solvers\n{", "solvers\n{\n    \"(k|omega)\"\n    {\n        solver          smoothSolver;\n"
                  "        smoother        GaussSeidel;\n        nSweeps         2;\n        tolerance       1e-10;\n"
                  "        relTol          0;\n    }\n", 1)
    open(p, "w").write(s)
    open(f"{case}/constant/turbulenceProperties", "w").write(
        DICT % "turbulenceProperties" + "\nsimulationType  RAS;\n\nRAS\n{\n    RASModel        kOmegaSST;\n"
        "    turbulence      on;\n    printCoeffs     on;\n}\n")


def corner_mesh(case, level, grade):
    n = 2**level
    nA, nB, ny, nz = 12*n, 24*n, 16*n, 16*n
    h = (L - X0)*math.tan(THETA)
    v = [(0, 0, 0), (X0, 0, 0), (X0, H, 0), (0, H, 0), (0, 0, W), (X0, 0, W), (X0, H, W), (0, H, W),
         (L, h, h), (L, H, h), (L, h, W), (L, H, W)]
    g = f"simpleGrading (1 {grade} {grade})"
    s = DICT % "blockMeshDict" + "\nscale 1;\n\nvertices\n(\n" + \
        "".join("    (%r %r %r)\n" % p for p in v) + ");\n\nblocks\n(\n" + \
        f"    hex (0 1 2 3 4 5 6 7) ({nA} {ny} {nz}) {g}\n" + \
        f"    hex (1 8 9 2 5 10 11 6) ({nB} {ny} {nz}) {g}\n);\n\nedges\n(\n);\n\nboundary\n(\n" + \
        "    inlet { type patch; faces ((0 4 7 3)); }\n" + \
        "    outlet { type patch; faces ((8 9 11 10)); }\n" + \
        "    top { type patch; faces ((3 7 6 2) (2 6 11 9)); }\n" + \
        "    floor { type wall; faces ((0 1 5 4) (1 8 10 5)); }\n" + \
        "    side { type wall; faces ((0 3 2 1) (1 2 9 8)); }\n" + \
        "    symmetry { type symmetryPlane; faces ((4 5 6 7) (5 10 11 6)); }\n);\n\nmergePatchPairs\n(\n);\n"
    open(f"{case}/system/blockMeshDict", "w").write(s)


def corner(case, kind, level, lts):
    viscous = kind != "corner_inv"
    p_inf = {"corner_inv": 1e4, "corner_lam": 1e3, "corner_sst": 1e5}[kind]
    corner_mesh(case, level, {"corner_inv": 1, "corner_lam": 50, "corner_sst": 200}[kind])
    thermo(case, viscous)
    if kind == "corner_sst":
        turb_schemes(case)
    if lts:
        lts_schemes(case)
        control(case, "4000", "1000", "0.5", lts=True)
    elif kind == "corner_inv":
        control(case, "0.002", "0.0005", "0.5")
    else:
        control(case, "0.001", "0.00025", "0.2")
    shutil.rmtree(f"{case}/0", ignore_errors=True)
    os.makedirs(f"{case}/0")
    U0, T0, p0 = f"({U_INF} 0 0)", repr(T_INF), repr(p_inf)
    fixed = lambda v: ["type            fixedValue;", f"value           uniform {v};"]
    io = lambda v: ["type            inletOutlet;", f"inletValue      uniform {v};", f"value           uniform {v};"]
    zg = ["type            zeroGradient;"]
    sym = ["type            symmetryPlane;"]
    wallU = ["type            noSlip;"] if viscous else ["type            slip;"]
    wallT = fixed(T0) if viscous else zg
    wallp = zg

    def six(f_in, f_wall, f_free):
        return [("inlet", f_in), ("outlet", f_free), ("top", f_free), ("floor", f_wall), ("side", f_wall),
                ("symmetry", sym)]
    write(case, "U", "volVectorField", "[0 1 -1 0 0 0 0]", f"uniform {U0}", six(fixed(U0), wallU, io(U0)))
    write(case, "p", "volScalarField", "[1 -1 -2 0 0 0 0]", f"uniform {p0}", six(fixed(p0), wallp, io(p0)))
    write(case, "T", "volScalarField", "[0 0 0 1 0 0 0]", f"uniform {T0}", six(fixed(T0), wallT, io(T0)))
    if kind == "corner_sst":
        k0 = 1.5*(0.01*U_INF)**2
        w0 = k0**0.5/(0.09**0.25*0.001)
        calc = ["type            calculated;", "value           uniform 0;"]
        write(case, "k", "volScalarField", "[0 2 -2 0 0 0 0]", f"uniform {k0!r}",
              six(fixed(repr(k0)), ["type            kqRWallFunction;", f"value           uniform {k0!r};"],
                  io(repr(k0))))
        write(case, "omega", "volScalarField", "[0 0 -1 0 0 0 0]", f"uniform {w0!r}",
              six(fixed(repr(w0)), ["type            omegaWallFunction;", f"value           uniform {w0!r};"],
                  io(repr(w0))))
        write(case, "nut", "volScalarField", "[0 2 -1 0 0 0 0]", "uniform 0",
              six(calc, ["type            nutkWallFunction;", "value           uniform 0;"], calc))
        write(case, "alphat", "volScalarField", "[1 -1 -1 0 0 0 0]", "uniform 0",
              six(calc, ["type            compressible::alphatWallFunction;", "Prt             0.85;",
                         "value           uniform 0;"], calc))


def box_mesh(case, n):
    s = DICT % "blockMeshDict" + "\nscale 1;\n\nvertices\n(\n" + \
        "    (0 0 0) (1 0 0) (1 1 0) (0 1 0) (0 0 1) (1 0 1) (1 1 1) (0 1 1)\n);\n\nblocks\n(\n" + \
        f"    hex (0 1 2 3 4 5 6 7) ({n} {n} {n}) simpleGrading (1 1 1)\n);\n\nedges\n(\n);\n\nboundary\n(\n" + \
        "    symmetryX { type symmetryPlane; faces ((0 4 7 3)); }\n" + \
        "    walls { type wall; faces ((1 2 6 5) (0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7)); }\n);\n\n" + \
        "mergePatchPairs\n(\n);\n"
    open(f"{case}/system/blockMeshDict", "w").write(s)


def box(case, kind):
    n = 24
    box_mesh(case, n)
    thermo(case, False)
    control(case, "0.003", "0.0005", "0.5")
    shutil.rmtree(f"{case}/0", ignore_errors=True)
    os.makedirs(f"{case}/0")
    c = (np.arange(n) + 0.5)/n
    # blockMesh numbers a block's cells x fastest, then y, then z
    z, y, x = np.meshgrid(c, c, c, indexing="ij")
    r2 = (x - 0.35)**2 + (y - 0.4)**2 + (z - 0.45)**2
    pv = np.where(r2.ravel() < 0.2**2, 1e5, 1e4)
    p_int = "nonuniform List<scalar>\n%d\n(\n%s\n)\n" % (pv.size, "\n".join(repr(float(v)) for v in pv))
    sym = ["type            symmetryPlane;"]
    zg = ["type            zeroGradient;"]
    write(case, "U", "volVectorField", "[0 1 -1 0 0 0 0]", "uniform (0 0 0)",
          [("symmetryX", sym), ("walls", ["type            slip;"])])
    write(case, "p", "volScalarField", "[1 -1 -2 0 0 0 0]", p_int, [("symmetryX", sym), ("walls", zg)])
    write(case, "T", "volScalarField", "[0 0 0 1 0 0 0]", "uniform 300", [("symmetryX", sym), ("walls", zg)])


PSI = ["psi             thermo:psi;", "gamma           %r;" % GAMMA]


def duct(case):
    s = DICT % "blockMeshDict" + "\nscale 1;\n\nvertices\n(\n" + \
        "    (0 0 0) (0.3 0 0) (0.3 0.02 0) (0 0.02 0) (0 0 0.02) (0.3 0 0.02) (0.3 0.02 0.02) (0 0.02 0.02)\n" + \
        ");\n\nblocks\n(\n    hex (0 1 2 3 4 5 6 7) (30 10 10) simpleGrading (1 1 1)\n);\n\nedges\n(\n);\n\n" + \
        "boundary\n(\n    inlet { type patch; faces ((0 4 7 3)); }\n    outlet { type patch; faces ((1 2 6 5)); }\n" + \
        "    walls { type wall; faces ((0 1 5 4) (3 7 6 2) (0 3 2 1)); }\n" + \
        "    symmetry { type symmetryPlane; faces ((4 5 6 7)); }\n);\n\nmergePatchPairs\n(\n);\n"
    open(f"{case}/system/blockMeshDict", "w").write(s)
    thermo(case, True)
    control(case, "0.002", "0.0005", "0.3")
    shutil.rmtree(f"{case}/0", ignore_errors=True)
    os.makedirs(f"{case}/0")
    sym = ["type            symmetryPlane;"]
    zg = ["type            zeroGradient;"]
    write(case, "U", "volVectorField", "[0 1 -1 0 0 0 0]", "uniform (0 0 0)",
          [("inlet", zg), ("outlet", zg), ("walls", ["type            noSlip;"]), ("symmetry", sym)])
    write(case, "p", "volScalarField", "[1 -1 -2 0 0 0 0]", "uniform 100000",
          [("inlet", ["type            totalPressure;", "rho             none;"] + PSI +
            ["p0              uniform 150000;", "value           uniform 100000;"]),
           ("outlet", ["type            waveTransmissive;", "field           p;"] + PSI +
            ["fieldInf        100000;", "lInf            0.3;", "value           uniform 100000;"]),
           ("walls", zg), ("symmetry", sym)])
    write(case, "T", "volScalarField", "[0 0 0 1 0 0 0]", "uniform 300",
          [("inlet", ["type            totalTemperature;"] + PSI + ["T0              uniform 300;",
                                                                   "value           uniform 300;"]),
           ("outlet", zg), ("walls", zg), ("symmetry", sym)])


def setup(kind, case, level=0, lts=False):
    if kind.startswith("corner_"):
        corner(case, kind, level, lts)
    elif kind in ("box_blast", "box_restart"):
        box(case, kind)
    elif kind == "duct_tp":
        duct(case)
    else:
        raise SystemExit(f"rho_central_3d: no case kind {kind}")


# ---- engineering quantities -------------------------------------------------------------------------------------------
def oblique(M, theta, g):
    """the weak oblique shock's angle and pressure ratio for Mach M and deflection theta (radians)"""
    lo, hi = math.asin(1/M) + 1e-9, math.radians(64.0)
    f = lambda b: math.tan(theta) - 2/math.tan(b)*(M*M*math.sin(b)**2 - 1)/(M*M*(g + math.cos(2*b)) + 2)
    for _ in range(200):
        mid = 0.5*(lo + hi)
        if f(lo)*f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    b = 0.5*(lo + hi)
    return b, 1 + 2*g/(g + 1)*(M*M*math.sin(b)**2 - 1)


def read_internal(path):
    import rho_central_fields as rf
    return rf.read_internal(path) if hasattr(rf, "read_internal") else _read_internal(path)


def _read_internal(path):
    s = open(path).read()
    m = re.search(r"internalField\s+uniform\s+([^;]+);", s)
    if m:
        return None
    m = re.search(r"internalField\s+nonuniform\s+List<(\w+)>\s*(\d+)\s*\(", s)
    n = int(m.group(2))
    body = s[m.end():]
    if m.group(1) == "scalar":
        return np.array(body.split(None, n)[:n], dtype=float)
    vals = re.findall(r"\(([^()]*)\)", body[:body.index("\n)")+1])[:n]
    return np.array([list(map(float, v.split())) for v in vals])


def cell_centres(case):
    """cell centres and volumes from OpenFOAM's writeCellCentres output (C, V in the time 0 directory); the gate writes
    them with postProcess -func writeCellCentres and writeCellVolumes"""
    C = _read_internal(f"{case}/0/C")
    V = _read_internal(f"{case}/0/V")
    return C, V


def ramp(case, tname):
    C, V = cell_centres(case)
    p = _read_internal(f"{case}/{tname}/p")
    pinf = float(re.search(r"internalField\s+uniform\s+([^;]+);", open(f"{case}/0/p").read()).group(1))
    x, y, z = C[:, 0], C[:, 1], C[:, 2]
    yf = np.maximum(x - X0, 0)*math.tan(THETA)
    first = np.zeros(len(x), bool)
    # the floor's first cell row: the lowest cell of each (x, z) column of the ramp block
    key = np.round(x, 9)*1e6 + np.round(z, 9)
    order = np.lexsort((y, key))
    ks = key[order]
    first[order[np.r_[True, ks[1:] != ks[:-1]]]] = True
    sel = first & (x > 0.55) & (x < 0.75) & (z > 0.35)
    ratio = float(np.sum(p[sel]*V[sel])/np.sum(V[sel]))/pinf
    beta, pr = oblique(MACH, THETA, GAMMA)
    # the shock angle: along y at the column nearest x = 0.85, in the z = 0.45..0.5 cells, the first height above the
    # floor where p passes (p_inf + p_ramp)/2 from above (downstream of the shock, below it, p is the ramp's)
    zc = np.max(z)
    xc = x[np.abs(z - zc) < 1e-6][np.argmin(np.abs(x[np.abs(z - zc) < 1e-6] - 0.85))]
    col = (np.abs(z - zc) < 1e-6) & (np.abs(x - xc) < 1e-6)   # the column of cells nearest x = 0.85 at the symmetry plane
    yy, pp, xx = y[col], p[col], x[col]
    o = np.argsort(yy)
    yy, pp, xx = yy[o], pp[o], xx[o]
    half = pinf*(1 + ratio)/2
    k = np.where(pp < half)[0]
    ang = float("nan")
    if len(k) and k[0] > 0:
        i = k[0]
        ys = yy[i - 1] + (half - pp[i - 1])*(yy[i] - yy[i - 1])/(pp[i] - pp[i - 1])
        ang = math.degrees(math.atan2(ys, xx[i] - X0))
    print("ramp %.6e %.6e %.4f %.4f" % (ratio, pr, ang, math.degrees(beta)))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "setup":
        setup(a[1], a[2], int(a[3]) if len(a) > 3 else 0, len(a) > 4 and a[4] == "lts")
    elif a and a[0] == "ramp":
        ramp(a[1], a[2])
    elif a and a[0] == "theory":
        b, pr = oblique(MACH, THETA, GAMMA)
        print("oblique M %.4f theta 10 gamma %.6f: beta %.4f deg, p2/p1 %.6f" % (MACH, GAMMA, math.degrees(b), pr))
    else:
        raise SystemExit(__doc__)
