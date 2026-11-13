"""The Ahmed-like body of tests/simple_ahmed_limited_vs_openfoam.sh: a body 1.044 x 0.288 x 0.389 m (its side profile's front
corners rounded at 0.1 m, a 25 degree rear slant 0.222 m long, 0.05 m above the ground) in a walled tunnel, meshed by OpenFOAM's
own snappyHexMesh (split-hex polyhedra where the refinement levels meet, three prism layers on the body) from a blockMesh
background of cells `h` m on a side. Every laplacian is `limited corrected 0.33`, every wall under wall functions.

    ahmed_case.py mesh <case> <h>                     blockMeshDict, the body's STL, surfaceFeatureExtract and snappyHexMesh
    ahmed_case.py rans <case> <iterations> <model>    simpleFoam from uniform fields: kOmegaSST, kEpsilon or SpalartAllmaras
    ahmed_case.py pimple <case> <steps> <dt>          the case made transient: pimpleFoam, Euler, PISO (one outer corrector,
                                                      two pressure correctors), no relaxation, written every step
    ahmed_case.py ulp <case>                          OpenFOAM's one-ulp twin: the first cell's Ux one unit in the last place higher
    ahmed_case.py fromsst <case> <kOmegaSST time>     a kEpsilon case's 0/ from kOmegaSST's fields at that time: U, p, phi, nut
                                                      and k as they are, epsilon = Cmu k omega cell by cell
"""
import math
import os
import re
import sys

H = 'FoamFile { version 2.0; format ascii; class %s; object %s; }\n'

# the body: length, height, width, front radius, slant angle and length, ground clearance
L = 1.044
HB = 0.288
WB = 0.389
R = 0.1
SLANT = math.radians(25.0)
LS = 0.222
CLEAR = 0.05
# the tunnel: x from -2 to 5, the ground at y = 0, the roof at 1.2, the side walls at z = +-1
X0, X1 = -2.0, 5.0
Y1 = 1.2
Z1 = 1.0
UINF = 10.0
NU = 1.5e-5
K_IN = 0.015
OMEGA_IN = 22.4
# epsilon = Cmu k omega, the same turbulence as k and omega's
EPSILON_IN = 0.09 * K_IN * OMEGA_IN
# nuTilda three times nu, the freestream value OpenFOAM's SpalartAllmaras tutorials take
NUTILDA_IN = 3.0 * NU
MODELS = ('kOmegaSST', 'kEpsilon', 'SpalartAllmaras')


def write(case, name, cls, text):
    os.makedirs(os.path.dirname(case + '/' + name), exist_ok=True)
    with open(case + '/' + name, 'w') as f:
        f.write(H % (cls, os.path.basename(name)) + text)


def profile(nArc=12):
    """the side profile (x, y), counter-clockwise, convex: rear face, slant, roof, the two rounded front corners, floor"""
    y0, y1 = CLEAR, CLEAR + HB
    xs = L - LS * math.cos(SLANT)
    yr = y1 - LS * math.sin(SLANT)
    pts = [(L, y0), (L, yr), (xs, y1), (R, y1)]
    for i in range(1, nArc + 1):
        a = math.pi / 2 + (math.pi / 2) * i / nArc
        pts.append((R + R * math.cos(a), y1 - R + R * math.sin(a)))
    # the front face is the edge from the first arc's end (0, y1 - R) to this one's start (0, y0 + R)
    for i in range(0, nArc + 1):
        a = math.pi + (math.pi / 2) * i / nArc
        pts.append((R + R * math.cos(a), y0 + R + R * math.sin(a)))
    return pts


def stl(path):
    """the body's closed surface: the profile extruded across z = +-WB/2, its two sides fanned from the centroid"""
    p = profile()
    zl, zr = -WB / 2, WB / 2
    cx = sum(x for x, _ in p) / len(p)
    cy = sum(y for _, y in p) / len(p)
    tris = []
    n = len(p)
    for i in range(n):
        a, b = p[i], p[(i + 1) % n]
        # the strip between a and b, its normal outward (the profile is counter-clockwise in x-y)
        tris.append(((a[0], a[1], zl), (b[0], b[1], zl), (b[0], b[1], zr)))
        tris.append(((a[0], a[1], zl), (b[0], b[1], zr), (a[0], a[1], zr)))
        # the two sides: z = zr faces +z (counter-clockwise seen from +z), z = zl faces -z
        tris.append(((cx, cy, zr), (a[0], a[1], zr), (b[0], b[1], zr)))
        tris.append(((cx, cy, zl), (b[0], b[1], zl), (a[0], a[1], zl)))
    with open(path, 'w') as f:
        f.write('solid body\n')
        for t in tris:
            u = [t[1][k] - t[0][k] for k in range(3)]
            v = [t[2][k] - t[0][k] for k in range(3)]
            nn = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
            m = math.sqrt(sum(c * c for c in nn)) or 1.0
            f.write(' facet normal %.9g %.9g %.9g\n  outer loop\n' % tuple(c / m for c in nn))
            for q in t:
                f.write('   vertex %.12g %.12g %.12g\n' % q)
            f.write('  endloop\n endfacet\n')
        f.write('endsolid body\n')


def mesh(case, h):
    for d in ('', '/constant/triSurface', '/system'):
        os.makedirs(case + d, exist_ok=True)
    stl(case + '/constant/triSurface/body.stl')
    nx = max(1, round((X1 - X0) / h))
    ny = max(1, round(Y1 / h))
    nz = max(1, round(2 * Z1 / h))
    write(case, 'system/blockMeshDict', 'dictionary',
          'scale 1;\nvertices ( (%g 0 %g) (%g 0 %g) (%g %g %g) (%g %g %g) (%g 0 %g) (%g 0 %g) (%g %g %g) (%g %g %g) );\n'
          % (X0, -Z1, X1, -Z1, X1, Y1, -Z1, X0, Y1, -Z1, X0, Z1, X1, Z1, X1, Y1, Z1, X0, Y1, Z1)
          + 'blocks ( hex (0 1 2 3 4 5 6 7) (%d %d %d) simpleGrading (1 1 1) );\nedges ();\nboundary\n(\n' % (nx, ny, nz)
          + '    inlet  { type patch; faces ( (0 4 7 3) ); }\n'
          + '    outlet { type patch; faces ( (1 2 6 5) ); }\n'
          + '    ground { type wall;  faces ( (0 1 5 4) ); }\n'
          + '    tunnel { type wall;  faces ( (3 7 6 2) (0 3 2 1) (4 5 6 7) ); }\n);\nmergePatchPairs ();\n')
    write(case, 'system/controlDict', 'dictionary',
          'application snappyHexMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1;\n'
          'writeControl timeStep; writeInterval 1; writeFormat ascii; writePrecision 17; writeCompression off;\n'
          'timeFormat general; timePrecision 6; runTimeModifiable false;\n')
    write(case, 'system/fvSchemes', 'dictionary',
          'ddtSchemes { default steadyState; }\ngradSchemes { default Gauss linear; }\ndivSchemes { default none; }\n'
          'laplacianSchemes { default Gauss linear corrected; }\ninterpolationSchemes { default linear; }\n'
          'snGradSchemes { default corrected; }\n')
    write(case, 'system/fvSolution', 'dictionary', '\n')
    write(case, 'system/surfaceFeatureExtractDict', 'dictionary',
          'body.stl\n{\n    extractionMethod extractFromSurface;\n    includedAngle 150;\n    subsetFeatures { nonManifoldEdges no; openEdges yes; }\n    writeObj no;\n}\n')
    write(case, 'system/meshQualityDict', 'dictionary', '#includeEtc "caseDicts/meshQualityDict"\n')
    # the wake box behind and around the body at level 2, the body at level 3: the levels meet in split-hex polyhedra
    write(case, 'system/snappyHexMeshDict', 'dictionary', '''castellatedMesh true;
snap true;
addLayers true;
geometry
{
    body.stl { type triSurfaceMesh; name body; }
    wake { type box; min (-0.3 0 -0.45); max (2.6 0.6 0.45); }
}
castellatedMeshControls
{
    maxLocalCells 20000000;
    maxGlobalCells 200000000;
    minRefinementCells 10;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 3;
    features ( { file "body.eMesh"; level 3; } );
    refinementSurfaces { body { level (3 3); patchInfo { type wall; } } }
    resolveFeatureAngle 30;
    refinementRegions { wake { mode inside; levels ((1E15 2)); } }
    locationInMesh (3.0001 0.6001 0.5001);
    allowFreeStandingZoneFaces true;
}
snapControls
{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 30;
    nRelaxIter 5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}
addLayersControls
{
    relativeSizes true;
    layers { body { nSurfaceLayers 3; } }
    expansionRatio 1.2;
    finalLayerThickness 0.5;
    minThickness 0.1;
    nGrow 0;
    featureAngle 60;
    slipFeatureAngle 30;
    nRelaxIter 3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
}
meshQualityControls
{
    #include "meshQualityDict"
    nSmoothScale 4;
    errorReduction 0.75;
}
mergeTolerance 1e-6;
''')


def field(case, name, cls, dims, value, inlet, outlet, wall):
    write(case, '0/' + name, cls,
          'dimensions %s; internalField uniform %s;\nboundaryField {\n    inlet { %s }\n    outlet { %s }\n'
          '    ground { %s }\n    tunnel { %s }\n    body { %s }\n}\n' % (dims, value, inlet, outlet, wall, wall, wall))


def transported(model):
    return {'kOmegaSST': ('k', 'omega'), 'kEpsilon': ('k', 'epsilon'), 'SpalartAllmaras': ('nuTilda',)}[model]


def schemes(case, ddt, model):
    div = ''.join(' div(phi,%s) bounded Gauss limitedLinear 1;' % f for f in transported(model))
    write(case, 'system/fvSchemes', 'dictionary',
          'ddtSchemes { default %s; }\ngradSchemes { default Gauss linear; }\n'
          'divSchemes { default none; div(phi,U) bounded Gauss linearUpwind grad(U);\n'
          ' %s\n  div((nuEff*dev2(T(grad(U))))) Gauss linear; }\n'
          'laplacianSchemes { default Gauss linear limited corrected 0.33; }\ninterpolationSchemes { default linear; }\n'
          'snGradSchemes { default limited corrected 0.33; }\nwallDist { method meshWave; }\n' % (ddt, div))


def rans(case, iterations, model):
    """the start: simpleFoam from uniform fields, every wall under wall functions"""
    assert model in MODELS, model
    write(case, 'system/controlDict', 'dictionary',
          'application simpleFoam; startFrom startTime; startTime 0; stopAt endTime; endTime %d;\n'
          'deltaT 1; writeControl timeStep; writeInterval %d; writeFormat ascii; writePrecision 17;\n'
          'writeCompression off; timeFormat general; timePrecision 6; runTimeModifiable false;\n' % (iterations, iterations))
    schemes(case, 'steadyState', model)
    names = '|'.join(('U',) + transported(model))
    relax = ' '.join('%s 0.7;' % f for f in ('U',) + transported(model))
    write(case, 'system/fvSolution', 'dictionary',
          'solvers { p { solver GAMG; smoother GaussSeidel; tolerance 1e-8; relTol 0.01; maxIter 5000; }\n'
          '  "(%s)" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-10; relTol 0.1; maxIter 2000; } }\n'
          'SIMPLE { nNonOrthogonalCorrectors 0; consistent false; }\n'
          'relaxationFactors { fields { p 0.3; } equations { %s } }\n' % (names, relax))
    write(case, 'constant/transportProperties', 'dictionary', 'transportModel Newtonian; nu %g;\n' % NU)
    write(case, 'constant/turbulenceProperties', 'dictionary',
          'simulationType RAS; RAS { RASModel %s; turbulence on; printCoeffs on; }\n' % model)
    o = 'type inletOutlet; inletValue uniform (0 0 0); value uniform (%g 0 0);' % UINF
    field(case, 'U', 'volVectorField', '[0 1 -1 0 0 0 0]', '(%g 0 0)' % UINF,
          'type fixedValue; value uniform (%g 0 0);' % UINF, o, 'type noSlip;')
    field(case, 'p', 'volScalarField', '[0 2 -2 0 0 0 0]', '0', 'type zeroGradient;', 'type fixedValue; value uniform 0;',
          'type zeroGradient;')

    def scalar(name, dims, v, wall):
        field(case, name, 'volScalarField', dims, '%g' % v, 'type fixedValue; value uniform %g;' % v,
              'type inletOutlet; inletValue uniform %g; value uniform %g;' % (v, v), wall)
    if model == 'SpalartAllmaras':
        scalar('nuTilda', '[0 2 -1 0 0 0 0]', NUTILDA_IN, 'type fixedValue; value uniform 0;')
        field(case, 'nut', 'volScalarField', '[0 2 -1 0 0 0 0]', '0', 'type calculated; value uniform 0;',
              'type calculated; value uniform 0;', 'type nutUSpaldingWallFunction; value uniform 0;')
        return
    scalar('k', '[0 2 -2 0 0 0 0]', K_IN, 'type kqRWallFunction; value uniform %g;' % K_IN)
    if model == 'kOmegaSST':
        scalar('omega', '[0 0 -1 0 0 0 0]', OMEGA_IN, 'type omegaWallFunction; value uniform %g;' % OMEGA_IN)
    else:
        scalar('epsilon', '[0 2 -3 0 0 0 0]', EPSILON_IN, 'type epsilonWallFunction; value uniform %g;' % EPSILON_IN)
    field(case, 'nut', 'volScalarField', '[0 2 -1 0 0 0 0]', '0', 'type calculated; value uniform 0;',
          'type calculated; value uniform 0;', 'type nutkWallFunction; value uniform 0;')


def model_of(case):
    m = re.search(r'RASModel\s+(\w+)', open(case + '/constant/turbulenceProperties').read())
    return m.group(1)


def pimple(case, steps, dt):
    """the case made transient from the fields in its 0/: pimpleFoam, Euler, one outer corrector and two pressure correctors, no
    relaxation (neither code then has a Final relaxation entry to resolve), every solve to 1e-12 at relTol 0, written every step"""
    model = model_of(case)
    write(case, 'system/controlDict', 'dictionary',
          'application pimpleFoam; startFrom startTime; startTime 0; stopAt endTime; endTime %.10g;\n'
          'deltaT %.10g; writeControl timeStep; writeInterval 1; writeFormat ascii; writePrecision 17;\n'
          'writeCompression off; timeFormat general; timePrecision 6; runTimeModifiable false; adjustTimeStep no;\n'
          % (steps * dt, dt))
    schemes(case, 'Euler', model)
    names = '|'.join(('U',) + transported(model))
    tight = '{ solver smoothSolver; smoother symGaussSeidel; tolerance 1e-12; relTol 0; maxIter 2000; }'
    ptight = '{ solver GAMG; smoother GaussSeidel; tolerance 1e-12; relTol 0; maxIter 5000; }'
    write(case, 'system/fvSolution', 'dictionary',
          'solvers\n{\n    p %s\n    pFinal %s\n    "(%s)" %s\n    "(%s)Final" %s\n}\n'
          'PIMPLE\n{\n    nOuterCorrectors 1;\n    nCorrectors 2;\n    nNonOrthogonalCorrectors 0;\n}\n'
          % (ptight, ptight, names, tight, names, tight))


def fromsst(case, sst):
    """a kEpsilon start from kOmegaSST's: OpenFOAM's kEpsilon from the uniform start diverges on this body under the case's entries
    (k's maximum 26, 261, 1198 at iterations 1-3, NaN by iteration 62) and under relaxation 0.5, p 0.2 and upwind k and epsilon
    alike (Arrhenius job 5063071), so its start is kOmegaSST's developed flow: U, p, phi, nut and k copied (the two cases' boundary
    conditions on them are one), epsilon = betaStar k omega = Cmu k omega cell by cell (kOmegaSST's own relation, Cmu = betaStar =
    0.09) under the kEpsilon case's own boundary conditions, whose uniform values OpenFOAM's first iteration re-evaluates"""
    assert model_of(case) == 'kEpsilon', model_of(case)
    for f in ('U', 'p', 'phi', 'nut', 'k'):
        with open(sst + '/' + f) as src, open(case + '/0/' + f, 'w') as dst:
            dst.write(src.read())
    values = {}
    for f in ('k', 'omega'):
        s = open(sst + '/' + f).read()
        m = re.search(r'internalField\s+nonuniform\s+List<scalar>\s*(\d+)\s*\(([^)]*)\)', s)
        assert m, 'no nonuniform internalField in ' + sst + '/' + f
        v = [float(x) for x in m.group(2).split()]
        assert len(v) == int(m.group(1)), f
        values[f] = v
    assert len(values['k']) == len(values['omega'])
    eps = '\n'.join(repr(0.09 * k * w) for k, w in zip(values['k'], values['omega']))
    s = open(case + '/0/epsilon').read()
    s2 = re.sub(r'internalField\s+uniform\s+[^;]+;', lambda m: 'internalField nonuniform List<scalar> %d\n(\n%s\n);'
                % (len(values['k']), eps), s, count=1)
    assert s2 != s, 'no uniform internalField in ' + case + '/0/epsilon'
    open(case + '/0/epsilon', 'w').write(s2)


def ulp(case):
    """OpenFOAM's one-ulp twin: the first cell's Ux one unit in the last place higher"""
    p = case + '/0/U'
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform\s+List<vector>\s*\d+\s*\(\s*\(\s*(\S+)', s)
    assert m, 'no nonuniform start in ' + p
    s = s[:m.start(1)] + repr(math.nextafter(float(m.group(1)), math.inf)) + s[m.end(1):]
    open(p, 'w').write(s)


if __name__ == '__main__':
    what = sys.argv[1]
    if what == 'mesh':
        mesh(sys.argv[2], float(sys.argv[3]))
    elif what == 'rans':
        rans(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    elif what == 'pimple':
        pimple(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]))
    elif what == 'ulp':
        ulp(sys.argv[2])
    elif what == 'fromsst':
        fromsst(sys.argv[2], sys.argv[3])
    else:
        sys.exit('ahmed_case.py: mesh, rans, pimple, ulp or fromsst')
