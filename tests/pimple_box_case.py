#!/usr/bin/env python3
"""A small hexahedral box case for pimpleFoam's set-up tests, written without OpenFOAM.

    pimple_box_case.py <dir> [--n NX NY NZ] [--skew S] [--2d] [--cyclic-x] [--model laminar|kEpsilon|kOmegaSST|SA|LES]
                             [--zone all|half|none]

The mesh is NX x NY x NZ unit-aspect cells over (0,1)^3 in OpenFOAM's ordering: internal faces upper-triangular (by owner,
then neighbour), each face's normal from owner to neighbour, boundary faces after them patch by patch with outward normals.
Patches: inlet (x = 0), outlet (x = 1), walls (y = 0 and y = 1), and z = 0 / z = 1 as `sides` (patch), or as `frontAndBack`
(empty) with --2d (NZ is then 1). --cyclic-x makes x = 0 / x = 1 the cyclic pair `left` / `right` instead of inlet and outlet.
--skew S shears the mesh, x += S*y, so its cells are not boxes. --zone writes cellZones `rotor`: every cell, the cells with
x < 0.5, or none.

The start fields (0/) are those the model reads, uniform, with a boundary condition on every patch. The case dictionaries are
a fixed-deltaT pimpleFoam run of two steps; tests edit them.
"""
import argparse
import os


def head(cls, obj, loc=None):
    s = 'FoamFile\n{\n    version 2.0;\n    format ascii;\n    class %s;\n' % cls
    if loc:
        s += '    location "%s";\n' % loc
    return s + '    object %s;\n}\n' % obj


def mesh(d, nx, ny, nz, skew, two_d, cyclic_x):
    pm = os.path.join(d, 'constant', 'polyMesh')
    os.makedirs(pm, exist_ok=True)

    def P(i, j, k):
        return i + (nx + 1) * (j + (ny + 1) * k)

    def C(i, j, k):
        return i + nx * (j + ny * k)

    pts = []
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                x, y, z = i / nx, j / ny, k / nz
                pts.append((x + skew * y, y, z))
    faces, owner, neigh = [], [], []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                c = C(i, j, k)
                if i + 1 < nx:
                    faces.append((P(i + 1, j, k), P(i + 1, j + 1, k), P(i + 1, j + 1, k + 1), P(i + 1, j, k + 1)))
                    owner.append(c)
                    neigh.append(C(i + 1, j, k))
                if j + 1 < ny:
                    faces.append((P(i, j + 1, k), P(i, j + 1, k + 1), P(i + 1, j + 1, k + 1), P(i + 1, j + 1, k)))
                    owner.append(c)
                    neigh.append(C(i, j + 1, k))
                if k + 1 < nz:
                    faces.append((P(i, j, k + 1), P(i + 1, j, k + 1), P(i + 1, j + 1, k + 1), P(i, j + 1, k + 1)))
                    owner.append(c)
                    neigh.append(C(i, j, k + 1))
    nInternal = len(faces)
    patches = []

    def patch(name, typ, fl, extra=''):
        start = len(faces)
        for f, c in fl:
            faces.append(f)
            owner.append(c)
        patches.append((name, typ, len(fl), start, extra))

    xmin = [((P(0, j, k), P(0, j, k + 1), P(0, j + 1, k + 1), P(0, j + 1, k)), C(0, j, k))
            for k in range(nz) for j in range(ny)]
    xmax = [((P(nx, j, k), P(nx, j + 1, k), P(nx, j + 1, k + 1), P(nx, j, k + 1)), C(nx - 1, j, k))
            for k in range(nz) for j in range(ny)]
    if cyclic_x:
        patch('left', 'cyclic', xmin, '        neighbourPatch right;\n        transform translational;\n'
              '        separationVector (1 0 0);\n')
        patch('right', 'cyclic', xmax, '        neighbourPatch left;\n        transform translational;\n'
              '        separationVector (-1 0 0);\n')
    else:
        patch('inlet', 'patch', xmin)
        patch('outlet', 'patch', xmax)
    ymin = [((P(i, 0, k), P(i + 1, 0, k), P(i + 1, 0, k + 1), P(i, 0, k + 1)), C(i, 0, k))
            for k in range(nz) for i in range(nx)]
    ymax = [((P(i, ny, k), P(i, ny, k + 1), P(i + 1, ny, k + 1), P(i + 1, ny, k)), C(i, ny - 1, k))
            for k in range(nz) for i in range(nx)]
    patch('walls', 'wall', ymin + ymax, '        inGroups 1(wall);\n')
    zmin = [((P(i, j, 0), P(i, j + 1, 0), P(i + 1, j + 1, 0), P(i + 1, j, 0)), C(i, j, 0))
            for j in range(ny) for i in range(nx)]
    zmax = [((P(i, j, nz), P(i + 1, j, nz), P(i + 1, j + 1, nz), P(i, j + 1, nz)), C(i, j, nz - 1))
            for j in range(ny) for i in range(nx)]
    if two_d:
        patch('frontAndBack', 'empty', zmin + zmax, '        inGroups 1(empty);\n')
    else:
        patch('sides', 'patch', zmin + zmax)

    nCells = nx * ny * nz
    note = 'nPoints:%d  nCells:%d  nFaces:%d  nInternalFaces:%d' % (len(pts), nCells, len(faces), nInternal)
    with open(os.path.join(pm, 'points'), 'w') as f:
        f.write(head('vectorField', 'points', 'constant/polyMesh'))
        f.write('%d\n(\n' % len(pts))
        for p in pts:
            f.write('(%.17g %.17g %.17g)\n' % p)
        f.write(')\n')
    with open(os.path.join(pm, 'faces'), 'w') as f:
        f.write(head('faceList', 'faces', 'constant/polyMesh'))
        f.write('%d\n(\n' % len(faces))
        for fc in faces:
            f.write('4(%d %d %d %d)\n' % fc)
        f.write(')\n')
    for name, lst in (('owner', owner), ('neighbour', neigh)):
        with open(os.path.join(pm, name), 'w') as f:
            f.write('FoamFile\n{\n    version 2.0;\n    format ascii;\n    class labelList;\n    note "%s";\n'
                    '    location "constant/polyMesh";\n    object %s;\n}\n' % (note, name))
            f.write('%d\n(\n' % len(lst))
            for v in lst:
                f.write('%d\n' % v)
            f.write(')\n')
    with open(os.path.join(pm, 'boundary'), 'w') as f:
        f.write(head('polyBoundaryMesh', 'boundary', 'constant/polyMesh'))
        f.write('%d\n(\n' % len(patches))
        for name, typ, n, start, extra in patches:
            f.write('    %s\n    {\n        type %s;\n%s        nFaces %d;\n        startFace %d;\n    }\n'
                    % (name, typ, extra, n, start))
        f.write(')\n')
    return [(p[0], p[1]) for p in patches], nCells


def zones(d, nx, ny, nz, which):
    if which == 'none':
        return
    cells = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if which == 'all' or (i + 0.5) / nx < 0.5:
                    cells.append(i + nx * (j + ny * k))
    with open(os.path.join(d, 'constant', 'polyMesh', 'cellZones'), 'w') as f:
        f.write(head('regIOobject', 'cellZones', 'constant/polyMesh'))
        f.write('1\n(\nrotor\n{\n    type cellZone;\ncellLabels List<label> %d\n(\n' % len(cells))
        for c in cells:
            f.write('%d\n' % c)
        f.write(')\n;\n}\n)\n')


def field(d, name, cls, dims, internal, bcs):
    with open(os.path.join(d, '0', name), 'w') as f:
        f.write(head(cls, name, '0'))
        f.write('dimensions %s;\ninternalField uniform %s;\nboundaryField\n{\n' % (dims, internal))
        for pname, body in bcs:
            f.write('    %s\n    {\n%s    }\n' % (pname, body))
        f.write('}\n')


def fields(d, patches, model):
    os.makedirs(os.path.join(d, '0'), exist_ok=True)

    def bc(kind):
        out = []
        for name, typ in patches:
            if typ in ('cyclic', 'empty'):
                out.append((name, '        type %s;\n' % typ))
            else:
                out.append((name, kind(name, typ)))
        return out

    def U(name, typ):
        if name == 'inlet':
            return '        type fixedValue;\n        value uniform (1 0 0);\n'
        if name == 'walls':
            return '        type noSlip;\n'
        return '        type zeroGradient;\n'
    field(d, 'U', 'volVectorField', '[0 1 -1 0 0 0 0]', '(1 0 0)', bc(U))

    def p(name, typ):
        if name == 'outlet':
            return '        type fixedValue;\n        value uniform 0;\n'
        return '        type zeroGradient;\n'
    field(d, 'p', 'volScalarField', '[0 2 -2 0 0 0 0]', '0', bc(p))
    if model == 'laminar':
        return

    def nut(wall):
        def f(name, typ):
            if name == 'walls':
                return '        type %s;\n        value uniform 0;\n' % wall
            return '        type calculated;\n        value uniform 0;\n'
        return f
    if model in ('kEpsilon', 'kOmegaSST'):
        field(d, 'nut', 'volScalarField', '[0 2 -1 0 0 0 0]', '0', bc(nut('nutkWallFunction')))

        def k(name, typ):
            if name == 'inlet':
                return '        type fixedValue;\n        value uniform 0.01;\n'
            if name == 'walls':
                return '        type kqRWallFunction;\n        value uniform 0.01;\n'
            return '        type zeroGradient;\n'
        field(d, 'k', 'volScalarField', '[0 2 -2 0 0 0 0]', '0.01', bc(k))
        second = 'epsilon' if model == 'kEpsilon' else 'omega'
        dims = '[0 2 -3 0 0 0 0]' if model == 'kEpsilon' else '[0 0 -1 0 0 0 0]'
        val = '0.01' if model == 'kEpsilon' else '10'

        def s(name, typ):
            if name == 'inlet':
                return '        type fixedValue;\n        value uniform %s;\n' % val
            if name == 'walls':
                return '        type %sWallFunction;\n        value uniform %s;\n' % (second, val)
            return '        type zeroGradient;\n'
        field(d, second, 'volScalarField', dims, val, bc(s))
    if model == 'SAIDDES':
        field(d, 'nut', 'volScalarField', '[0 2 -1 0 0 0 0]', '0', bc(nut('nutUSpaldingWallFunction')))

        def nt(name, typ):
            if name == 'inlet':
                return '        type fixedValue;\n        value uniform 3e-5;\n'
            if name == 'walls':
                return '        type fixedValue;\n        value uniform 0;\n'
            return '        type zeroGradient;\n'
        field(d, 'nuTilda', 'volScalarField', '[0 2 -1 0 0 0 0]', '3e-5', bc(nt))
    if model == 'LES':
        field(d, 'nut', 'volScalarField', '[0 2 -1 0 0 0 0]', '0', bc(nut('zeroGradient')))


def dicts(d, model):
    os.makedirs(os.path.join(d, 'system'), exist_ok=True)
    with open(os.path.join(d, 'system', 'controlDict'), 'w') as f:
        f.write(head('dictionary', 'controlDict', 'system'))
        f.write('application pimpleFoam;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\nendTime 0.002;\n'
                'deltaT 0.001;\nwriteControl timeStep;\nwriteInterval 1;\nwritePrecision 17;\n')
    sec = {'kEpsilon': 'epsilon', 'kOmegaSST': 'omega'}.get(model)
    with open(os.path.join(d, 'system', 'fvSchemes'), 'w') as f:
        f.write(head('dictionary', 'fvSchemes', 'system'))
        f.write('ddtSchemes { default Euler; }\ngradSchemes { default Gauss linear; }\ndivSchemes\n{\n    default none;\n'
                '    div(phi,U) Gauss linear;\n')
        if sec:
            f.write('    div(phi,k) Gauss upwind;\n    div(phi,%s) Gauss upwind;\n' % sec)
        if model == 'SAIDDES':
            f.write('    div(phi,nuTilda) Gauss upwind;\n')
        f.write('    div((nuEff*dev2(T(grad(U))))) Gauss linear;\n}\nlaplacianSchemes { default Gauss linear corrected; }\n'
                'interpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\n')
    with open(os.path.join(d, 'system', 'fvSolution'), 'w') as f:
        f.write(head('dictionary', 'fvSolution', 'system'))
        f.write('solvers\n{\n    p { solver PCG; preconditioner DIC; tolerance 1e-8; relTol 0; }\n'
                '    pFinal { $p; }\n    "(U|k|epsilon|omega|nuTilda)" { solver smoothSolver; smoother symGaussSeidel; '
                'tolerance 1e-8; relTol 0; }\n    "(U|k|epsilon|omega|nuTilda)Final" { solver smoothSolver; '
                'smoother symGaussSeidel; tolerance 1e-8; relTol 0; }\n}\n'
                'PIMPLE\n{\n    nOuterCorrectors 1;\n    nCorrectors 2;\n    nNonOrthogonalCorrectors 0;\n}\n')
    os.makedirs(os.path.join(d, 'constant'), exist_ok=True)
    with open(os.path.join(d, 'constant', 'transportProperties'), 'w') as f:
        f.write(head('dictionary', 'transportProperties', 'constant'))
        f.write('transportModel Newtonian;\nnu 1e-05;\n')
    with open(os.path.join(d, 'constant', 'turbulenceProperties'), 'w') as f:
        f.write(head('dictionary', 'turbulenceProperties', 'constant'))
        if model == 'laminar':
            f.write('simulationType laminar;\n')
        elif model in ('kEpsilon', 'kOmegaSST'):
            f.write('simulationType RAS;\nRAS\n{\n    RASModel %s;\n    turbulence on;\n}\n' % model)
        elif model == 'SAIDDES':
            f.write('simulationType LES;\nLES\n{\n    LESModel SpalartAllmarasIDDES;\n    turbulence on;\n'
                    '    delta IDDESDelta;\n}\n')
        elif model == 'LES':
            f.write('simulationType LES;\nLES\n{\n    LESModel Smagorinsky;\n    turbulence on;\n'
                    '    delta cubeRootVol;\n    cubeRootVolCoeffs { deltaCoeff 1; }\n}\n')


def main():
    a = argparse.ArgumentParser()
    a.add_argument('dir')
    a.add_argument('--n', type=int, nargs=3, default=[8, 6, 4])
    a.add_argument('--skew', type=float, default=0.0)
    a.add_argument('--2d', dest='two_d', action='store_true')
    a.add_argument('--cyclic-x', dest='cyclic_x', action='store_true')
    a.add_argument('--model', default='laminar', choices=['laminar', 'kEpsilon', 'kOmegaSST', 'SAIDDES', 'LES'])
    a.add_argument('--zone', default='none', choices=['all', 'half', 'none'])
    o = a.parse_args()
    nx, ny, nz = o.n
    if o.two_d:
        nz = 1
    patches, _ = mesh(o.dir, nx, ny, nz, o.skew, o.two_d, o.cyclic_x)
    zones(o.dir, nx, ny, nz, o.zone)
    fields(o.dir, patches, o.model)
    dicts(o.dir, o.model)


if __name__ == '__main__':
    main()
