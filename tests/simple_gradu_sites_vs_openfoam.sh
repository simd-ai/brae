#!/usr/bin/env bash
# gradSchemes grad(U) at the two sites of the legacy simpleFoam driver that did not read it, against real OpenFOAM. correctedSnGrad<vector>::correction is fullGradCorrection(U), which takes
# mesh.gradScheme("grad(U)") (correctedSnGrad.C:50-55), so the non-orthogonal laplacian correction takes the case's grad(U) entry when it has one, and `default` only when it has none. kOmegaSST's S2 on a
# `calculated` nut face is the model's fvc::grad(U) as well. Both took the bare Gauss gradient.
# MEASURED (JUPITER jobs 2635201, 2636147), motorBike at 14.1 M cells, first Ux initial residual 0.041183 in brae and 0.036073 in OpenFOAM from identical bytes; on the tutorial's coarse mesh (job 2641827)
# 0.0890829 and 0.0873860, with 84,098 of 307,645 interior cells' sources apart and nuEff 3.1e-02 off on a slip patch beside the wall. Unlimited, or on an orthogonal mesh without the slip patch, nothing shows.
#
# THE CASE: a 40 x 16 x 16 box, leaning (non-orthogonal) and graded, a wall body cut out of it, a moving lower wall and slip sides, from potentialFoam's start: motorBike's boundary types on a small mesh.
# The outlet fixes k and omega: OpenFOAM's wall function overwrites the wall-and-outlet corner cell through updateCoeffs while an inletOutlet or zeroGradient face keeps its stored value, which grad(field)
# reads, and brae re-evaluates the face (jobs 2637505, 2638428). MEASURED (job 2641949) with inletOutlet there: omega 3.5e-06 and k 1.9e-08, wholly in outlet cells. That gap is its own item, not this gate's.
# ARMS, kOmegaSST, BRAE_SST_F1_PATCH=1 (without it the inlet's F1 is its own gap), one SIMPLE iteration, every solve tight (p 1e-12, the rest 1e-13), OpenFOAM's sweep order:
#   shared    grad(U) cellLimited Gauss linear 1, div(phi,U) linearUpwindV grad(U): one limited gradient serves both
#   own       grad(U) cellLimited Gauss linear 1, div(phi,U) upwind: no linearUpwind gradient to share
#   apart     grad(U) cellLimited Gauss linear 0.5, div(phi,U) linearUpwind limited, `limited` cellLimited Gauss linear 1: two limited gradients
#   default   grad(U) Gauss linear, div(phi,U) linearUpwind limited: the correction takes the unlimited one (squareBendLiq's arrangement)
# THE ORACLE: real OpenFOAM on the same bytes. On JUPITER use a build that passes brae-multigpu's scripts/jsc/of_oracle.sh.
# THE BOUNDS: every field within 1e-10 relative L2; Ux, Uy, Uz and p first residuals within 5e-6, brae printing six digits.
# CONTROLS, asserted red on `shared`: BRAE_CONTROL_LAP_CORR_DEFAULT_GRAD=1 (the correction unlimited) and BRAE_CONTROL_SST_NUT_UNLIMITED=1 (S2 on the calculated faces unlimited).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
W=$(mktemp -d)
[ -n "${KEEP_W:-}" ] && { W="$KEEP_W"; rm -rf "${W:?}"; mkdir -p "$W"; } || trap 'rm -rf "$W"' EXIT
of()
{
    bash -c "source '$OFBASHRC' > /dev/null 2>&1; set -e; $*"
}
cat > "$W/gate.py" <<'PY'
import os
import re
import sys

import numpy as np

H = 'FoamFile { version 2.0; format ascii; class %s; object %s; }\n'
ARMS = {
    'shared': ('grad(U) cellLimited Gauss linear 1;', 'bounded Gauss linearUpwindV grad(U)'),
    'own': ('grad(U) cellLimited Gauss linear 1;', 'bounded Gauss upwind'),
    'apart': ('grad(U) cellLimited Gauss linear 0.5; limited cellLimited Gauss linear 1;', 'bounded Gauss linearUpwind limited'),
    'default': ('grad(U) Gauss linear; limited cellLimited Gauss linear 1;', 'bounded Gauss linearUpwind limited'),
}


def write(case, name, cls, text):
    with open(case + '/' + name, 'w') as f:
        f.write(H % (cls, os.path.basename(name)) + text)


def mesh(case):
    for d in ('', '/constant', '/system'):
        os.mkdir(case + d)
    write(case, 'system/controlDict', 'dictionary', 'application simpleFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 1;\ndeltaT 1; writeControl timeStep; writeInterval 1; writeFormat ascii; writePrecision 17; timeFormat general; timePrecision 6; runTimeModifiable false;\n')
    write(case, 'system/blockMeshDict', 'dictionary', '''scale 1;
vertices ( (-5 -4 0)(15 -4 0)(15 4 0)(-5 4 0)(-2 -4 8)(18 -4 8)(18 4 8)(-2 4 8) );
blocks ( hex (0 1 2 3 4 5 6 7) (40 16 16) simpleGrading (1 1 3) );
edges ();
boundary
(
    frontAndBack { type patch; faces ((0 1 5 4)(2 3 7 6)); }
    inlet { type patch; faces ((0 4 7 3)); }
    outlet { type patch; faces ((1 2 6 5)); }
    lowerWall { type wall; faces ((0 3 2 1)); }
    upperWall { type patch; faces ((4 5 6 7)); }
);
''')
    write(case, 'system/topoSetDict', 'dictionary', '''actions
(
    { name keep; type cellSet; action new; source boxToCell; box (-100 -100 -100)(100 100 100); }
    { name keep; type cellSet; action subtract; source boxToCell; box (1 -0.8 0.5)(3 0.8 2.0); }
);
''')
    write(case, 'system/fvSolution', 'dictionary', '''solvers
{
    p { solver GAMG; smoother GaussSeidel; tolerance 1e-12; relTol 0; maxIter 5000; }
    Phi { solver GAMG; smoother GaussSeidel; tolerance 1e-10; relTol 0; }
    "(U|k|omega)" { solver smoothSolver; smoother GaussSeidel; nSweeps 1; tolerance 1e-13; relTol 0; maxIter 5000; }
}
SIMPLE { nNonOrthogonalCorrectors 0; consistent yes; }
potentialFlow { nNonOrthogonalCorrectors 5; }
relaxationFactors { equations { U 0.9; k 0.7; omega 0.7; } }
''')
    write(case, 'constant/transportProperties', 'dictionary', 'transportModel Newtonian; nu 1.5e-5;\n')
    write(case, 'constant/turbulenceProperties', 'dictionary', 'simulationType RAS; RAS { RASModel kOmegaSST; turbulence on; printCoeffs on; }\n')
    schemes(case, 'shared')


def schemes(case, arm):
    grad, div = ARMS[arm]
    write(case, 'system/fvSchemes', 'dictionary', '''ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; %s }
divSchemes { default none; div(phi,U) %s; div(phi,k) bounded Gauss upwind; div(phi,omega) bounded Gauss upwind; div((nuEff*dev2(T(grad(U))))) Gauss linear; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
''' % (grad, div))


def fields(case):
    os.mkdir(case + '/0')
    T = 'dimensions %s; internalField uniform %s;\nboundaryField {\n    frontAndBack { %s }\n    inlet { %s }\n    outlet { %s }\n    lowerWall { %s }\n    upperWall { %s }\n    body { %s }\n}\n'
    write(case, '0/U', 'volVectorField', T % ('[0 1 -1 0 0 0 0]', '(20 0 0)', 'type slip;', 'type fixedValue; value uniform (20 0 0);', 'type inletOutlet; inletValue uniform (0 0 0); value uniform (0 0 0);', 'type fixedValue; value uniform (20 0 0);', 'type slip;', 'type noSlip;'))
    write(case, '0/p', 'volScalarField', T % ('[0 2 -2 0 0 0 0]', '0', 'type slip;', 'type zeroGradient;', 'type fixedValue; value uniform 0;', 'type zeroGradient;', 'type slip;', 'type zeroGradient;'))
    kw = 'type kqRWallFunction; value uniform 0.24;'
    write(case, '0/k', 'volScalarField', T % ('[0 2 -2 0 0 0 0]', '0.24', 'type slip;', 'type fixedValue; value uniform 0.24;', 'type fixedValue; value uniform 0.24;', kw, 'type slip;', kw))
    ow = 'type omegaWallFunction; value uniform 1.78;'
    write(case, '0/omega', 'volScalarField', T % ('[0 0 -1 0 0 0 0]', '1.78', 'type slip;', 'type fixedValue; value uniform 1.78;', 'type fixedValue; value uniform 1.78;', ow, 'type slip;', ow))
    nw = 'type nutkWallFunction; value uniform 0;'
    c = 'type calculated; value uniform 0;'
    write(case, '0/nut', 'volScalarField', T % ('[0 2 -1 0 0 0 0]', '0', c, c, c, nw, c, nw))


def read(fn):
    b = open(fn).read()
    m = re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n?(\d+)\s*\n?\(', b)
    n, nc = int(m.group(2)), 3 if m.group(1) == 'vector' else 1
    body = b[m.end():]
    body = body[:body.index('\n)')]
    return np.array(body.replace('(', ' ').replace(')', ' ').split(), dtype=float).reshape(n, nc)


def first(log):
    t = open(log, errors='replace').read()
    return {f: float(re.search(r'Solving for %s, Initial residual = ([0-9.e+-]+)' % f, t).group(1)) for f in ('Ux', 'Uy', 'Uz', 'p')}


def compare(of, br, want):
    g = {}
    for f in ('U', 'p', 'k', 'omega', 'nut'):
        a, b = read('%s/1/%s' % (of, f)), read('%s/1/%s' % (br, f))
        g[f] = np.linalg.norm(b - a) / np.linalg.norm(a)
    ro, rb = first(of + '/log'), first(br + '/log')
    r = {f: abs(rb[f] - ro[f]) / ro[f] for f in ro}
    ok = max(g.values()) <= 1e-10 and max(r.values()) <= 5e-6
    print('%s %s | first residuals %s' % ('within' if ok else 'MISSES', ' '.join('%s %.2e' % kv for kv in g.items()), ' '.join('%s %.1e' % kv for kv in r.items())))
    return ok == (want != 'miss')


what = sys.argv[1]
if what == 'mesh':
    mesh(sys.argv[2])
elif what == 'fields':
    fields(sys.argv[2])
elif what == 'schemes':
    schemes(sys.argv[2], sys.argv[3])
elif what == 'compare':
    sys.exit(0 if compare(*sys.argv[2:5]) else 1)
PY
g()
{
    python3 -B "$W/gate.py" "$@"
}
brae()   # <dir> [VAR=value ...]
{
    local c=$1
    shift
    ( cd "$c" && env BRAE_SIMPLE_CASE_SOLVER=1 BRAE_SST_F1_PATCH=1 BRAE_DICT_AUDIT=0 "$@" "$BRAE" -case . > log 2> err < /dev/null )
}
M=$W/mesh
g mesh "$M" || { echo "FAIL mesh"; exit 1; }
of "cd '$M' && blockMesh > log.blockMesh 2>&1 && topoSet > log.topoSet 2>&1 && subsetMesh keep -patch body -overwrite > log.subsetMesh 2>&1" < /dev/null || { echo "FAIL mesh"; exit 1; }
# subsetMesh's new patch is `empty`, which takes every direction from the solver
sed -i -E '/^ +body$/,/^ +\}$/ s/type +empty;/type wall;/' "$M/constant/polyMesh/boundary"
g fields "$M"
of "cd '$M' && potentialFoam -writephi > log.potentialFoam 2>&1" < /dev/null || { echo "FAIL potentialFoam"; exit 1; }
fail=0
for arm in shared own apart default
do
    a=$W/$arm
    mkdir "$a"
    for who in of br ctlA ctlB
    do
        cp -r "$M" "$a/$who"
        g schemes "$a/$who" "$arm"
    done
    of "cd '$a/of' && simpleFoam > log 2>&1" < /dev/null || { echo "FAIL OpenFOAM $arm"; fail=1; continue; }
    brae "$a/br" || { echo "FAIL brae $arm: $(tail -n 2 "$a/br/err" | cut -c1-200)"; fail=1; continue; }
    printf '  %-8s ' "$arm"
    g compare "$a/of" "$a/br" hold || fail=1
    if [ "$arm" = shared ]
    then
        brae "$a/ctlA" BRAE_CONTROL_LAP_CORR_DEFAULT_GRAD=1
        printf '  CONTROL correction unlimited  '
        g compare "$a/of" "$a/ctlA" miss || fail=1
        brae "$a/ctlB" BRAE_CONTROL_SST_NUT_UNLIMITED=1
        printf '  CONTROL patch S2 unlimited    '
        g compare "$a/of" "$a/ctlB" miss || fail=1
    fi
done
[ "$fail" = 0 ] && echo "PASS: the non-orthogonal correction and kOmegaSST's patch S2 take the case's grad(U)" || { echo FAIL; exit 1; }
