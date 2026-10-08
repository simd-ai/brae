# Shared by the three sa_iddes_*_vs_openfoam.sh gates: a constructed box run ONE step by real pimpleFoam with
# tools/dumpSAIDDES's model -- OpenFOAM's own SpalartAllmarasIDDES, renamed, writing dTilda()'s terms cell by
# cell -- and brae's test binary on that dump.
# THE BOX: 4 x 40 x 2 cells of 5e-3 x 5e-4 x 5e-3 over a wall at y = 0, so y/hmax runs from 0.05 to 3.95 and
# alpha = 0.25 - y/hmax takes both signs; nu 1e-6; chi = nuTilda/nu is 0.5, 3, 10 and 100 by column (psi 10,
# 3.1, 1.3 and 1), and U = (S y, 0, 0) with S 1000 in one layer and 30 in the other, so r = nu/(|gradU|
# (kappa y)^2) crosses the blending functions' ranges at different heights. No coefficient is set: the model
# runs at OpenFOAM's defaults, which the dump's header line carries.
# The run is one step of 1e-9 s; nothing is claimed of the flow. The dump is taken inside the step's
# turbulence correct(), with the gradient and nuTilda OpenFOAM hands dTilda() there.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_sa_iddes_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
W=${KEEP_W:-$(mktemp -d)}
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
for t in blockMesh pimpleFoam; do
    command -v "$t" > /dev/null 2>&1 || { echo "SKIP: $t not on PATH"; exit 77; }
done
[ -f "${FOAM_USER_LIBBIN:-/nonexistent}/libdumpSAIDDES.so" ] \
    || { echo "SKIP: libdumpSAIDDES.so not built (wmake in tools/dumpSAIDDES)"; exit 77; }

# stage_and_dump <dir>: the case, blockMesh, one pimpleFoam step; the dump lands in <dir>/saIddes_dump.txt
stage_and_dump()
{
    local C="$1"
    mkdir -p "$C/system" "$C/constant" "$C/0"
    python3 - "$C" <<'PY'
import sys
d = sys.argv[1]
nx, ny, nz = 4, 40, 2
dx, dy, dz = 5e-3, 5e-4, 5e-3
nu = 1e-6
def head(cls, obj):
    return 'FoamFile { version 2.0; format ascii; class %s; object %s; }\n' % (cls, obj)
open(d + '/system/blockMeshDict', 'w').write(head('dictionary', 'blockMeshDict') + '''scale 1;
vertices
(
    (0 0 0) (%g 0 0) (%g %g 0) (0 %g 0) (0 0 %g) (%g 0 %g) (%g %g %g) (0 %g %g)
);
blocks ( hex (0 1 2 3 4 5 6 7) (%d %d %d) simpleGrading (1 1 1) );
edges ();
boundary
(
    bottom { type wall; faces ((1 5 4 0)); }
    top { type patch; faces ((3 7 6 2)); }
    sides { type patch; faces ((0 4 7 3) (2 6 5 1) (0 3 2 1) (4 5 6 7)); }
);
mergePatchPairs ();
''' % (nx*dx, nx*dx, ny*dy, ny*dy, nz*dz, nx*dx, nz*dz, nx*dx, ny*dy, nz*dz, ny*dy, nz*dz, nx, ny, nz))
open(d + '/system/controlDict', 'w').write(head('dictionary', 'controlDict') + '''application pimpleFoam;
libs ("libdumpSAIDDES.so");
startFrom startTime; startTime 0; stopAt endTime; endTime 1e-9; deltaT 1e-9;
writeControl timeStep; writeInterval 1; writeFormat ascii; writePrecision 18; timePrecision 12;
runTimeModifiable false;
''')
open(d + '/system/fvSchemes', 'w').write(head('dictionary', 'fvSchemes') + '''ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes
{
    default none;
    div(phi,U) Gauss linear;
    div(phi,nuTilda) Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; nRequired yes; }
''')
open(d + '/system/fvSolution', 'w').write(head('dictionary', 'fvSolution') + '''solvers
{
    p { solver PCG; preconditioner DIC; tolerance 1e-10; relTol 0; }
    pFinal { $p; }
    "(U|nuTilda)" { solver PBiCGStab; preconditioner DILU; tolerance 1e-12; relTol 0; }
    "(U|nuTilda)Final" { $U; }
}
PIMPLE { nOuterCorrectors 1; nCorrectors 1; nNonOrthogonalCorrectors 0; }
''')
open(d + '/constant/transportProperties', 'w').write(head('dictionary', 'transportProperties')
    + 'transportModel Newtonian;\nnu %g;\n' % nu)
open(d + '/constant/turbulenceProperties', 'w').write(head('dictionary', 'turbulenceProperties')
    + '''simulationType LES;
LES
{
    LESModel SpalartAllmarasIDDESDump;
    printCoeffs on;
    turbulence on;
    delta IDDESDelta;
    IDDESDeltaCoeffs
    {
        hmax maxDeltaxyz;
        maxDeltaxyzCoeffs {}
    }
}
''')
chis = [0.5, 3.0, 10.0, 100.0]
shear = [1000.0, 30.0]
U = []
nt = []
for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            y = (j + 0.5)*dy
            U.append('(%.17g 0 0)' % (shear[k]*y))
            nt.append('%.17g' % (chis[i]*nu))
n = nx*ny*nz
def field(cls, obj, dims, internal, patches):
    return (head(cls, obj) + 'dimensions %s;\ninternalField %s;\nboundaryField\n{\n' % (dims, internal)
            + ''.join('    %s { %s }\n' % kv for kv in patches) + '}\n')
open(d + '/0/U', 'w').write(field('volVectorField', 'U', '[0 1 -1 0 0 0 0]',
    'nonuniform List<vector> %d\n(\n%s\n)' % (n, '\n'.join(U)),
    [('bottom', 'type noSlip;'), ('top', 'type zeroGradient;'), ('sides', 'type zeroGradient;')]))
open(d + '/0/p', 'w').write(field('volScalarField', 'p', '[0 2 -2 0 0 0 0]', 'uniform 0',
    [('bottom', 'type zeroGradient;'), ('top', 'type fixedValue; value uniform 0;'),
     ('sides', 'type zeroGradient;')]))
open(d + '/0/nuTilda', 'w').write(field('volScalarField', 'nuTilda', '[0 2 -1 0 0 0 0]',
    'nonuniform List<scalar> %d\n(\n%s\n)' % (n, '\n'.join(nt)),
    [('bottom', 'type fixedValue; value uniform 0;'), ('top', 'type zeroGradient;'),
     ('sides', 'type zeroGradient;')]))
open(d + '/0/nut', 'w').write(field('volScalarField', 'nut', '[0 2 -1 0 0 0 0]', 'uniform 0',
    [('bottom', 'type calculated; value uniform 0;'), ('top', 'type calculated; value uniform 0;'),
     ('sides', 'type calculated; value uniform 0;')]))
PY
    ( cd "$C" && blockMesh > log.blockMesh 2>&1 ) \
        || { echo "FAIL: blockMesh"; tail -20 "$C/log.blockMesh"; exit 1; }
    ( cd "$C" && BRAE_DUMP_ITER=1 pimpleFoam > log.pimpleFoam 2>&1 ) \
        || { echo "FAIL: pimpleFoam"; tail -20 "$C/log.pimpleFoam"; exit 1; }
    [ -s "$C/saIddes_dump.txt" ] || { echo "FAIL: OpenFOAM wrote no dump"; tail -20 "$C/log.pimpleFoam"; exit 1; }
}
