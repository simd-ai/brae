#!/usr/bin/env bash
# Foam::bound's average in the legacy simpleFoam driver, against real OpenFOAM. bound.C:48-56 gives a cell at or below zero fvc::average(max(field, floor)), surfaceSum(magSf*ssf)/surfaceSum(magSf) with
# each patch's own value (fvcAverage.C). deviceBoundField took the unweighted mean of a cell's faces with the cell's value on every boundary face.
# MEASURED (JUPITER job 2638365), pitzDailyTurb, epsilon driven to -41 in both codes: bounded cell 10874 OpenFOAM 31.58, brae 12.02; cell 11599 6.46 and 0.44; nut 48.5% relative L2 after one iteration.
#
# THE CASE: a 40 x 16 x 16 box, leaning (non-orthogonal) and graded (faces of unequal area), a wall body cut out of it, carrying slip, symmetry, fixedValue, inletOutlet, zeroGradient and wall function
# patches, run on kEpsilon and on kOmegaSST. The wall functions' files hold a value unlike any cell's: the dictionary constructors extrapolate the cell (omegaWallFunctionFvPatchScalarField.C), so the file's
# value must NOT reach the average, and brae's patch reads the cell.
# bound.C:59 is the other half: a zeroGradient face beside a bounded cell holds the floor until its next evaluate, where its cell holds the average, and `nut_` on a `calculated` patch and the next gradient
# read it. MEASURED (job 2645639), the solve arm's fields at iteration 1 exact in the cells, k 1e-15 in OpenFOAM and 2.58 in brae on those faces, nut 7e12 on `calculated` ones, U 4e-04 and p 1e-02 at iteration 2.
# kOmegaSST runs with BRAE_SST_F1_PATCH=1: without it the inlet's F1 is its own gap, k 5.3e-04 on the solve arm (job 2641949).
# ARMS, each on both models, OpenFOAM's sweep order and every solve tight (p 1e-12, the rest 1e-13):
#   start     Negative cells in the start field on every patch type and inside. `turbulence off`, so correct() returns at once (RASModel.C) and the fields written at iteration 1 are the model
#             constructor's bound (kEpsilon.C:182-183, kOmegaSSTBase.C:438-439). All fields, and k and the second field over the bounded cells alone. U starts from potentialFoam: on a uniform U,
#             S2 is round-off and SST's nut on a face beside a bounded cell, a1*kMin/(b1*sqrt(S2)), a ratio of round-offs (MEASURED, job 2645819: 1 in OpenFOAM and 0.0216 in brae on one face).
#   interior  The same with negative cells only where no face is on a patch.
#   solve     One iteration from a nonuniform positive start on limitedLinear, the outlet fixed far above the field, which drives a tight solve negative and correct() bounds it. All fields.
#   history   The solve arm for 30 iterations: every iteration's initial residuals and the count of bounding lines. At these tight entries: at p 1e-9 OpenFOAM's GAMG stops after 16 sweeps and brae's after 24,
#             a near-zero omega amplifies the difference through 1/omega in CDkOmega (cell 8749, CDkOmega 51,258 and 104,988), and kOmegaSST breaks at iteration 24 (job 2648101). OpenFOAM against itself with
#             only p's tolerance moved 1e-9 to 0.9e-9 passes this arm's k bound at iteration 28, where tight brae holds it throughout (job 2648136).
# THE ORACLE: real OpenFOAM on the same bytes. On JUPITER use a build that passes brae-multigpu's scripts/jsc/of_oracle.sh.
# THE BOUNDS: start, interior and solve, every held field within 1e-10 relative L2. history, U, p and k residuals within 2e-5 of OpenFOAM's and the second field's within 2e-4. It was 1e-2 for the wall rows'
# diagonal, which left the norm factor 0.42% out on omega and 0.27% on epsilon with no field to show it (device_scalar_transport.cuh, wfRelaxICKernel). MEASURED with that fixed: omega 7.6e-05, epsilon 3.9e-06.
# PREMISES, asserted: both codes print `bounding`, and the solve arm's OpenFOAM really bounds.
# CONTROLS, asserted red: BRAE_CONTROL_BOUND_UNWEIGHTED=1, the old average, on the start, interior and solve arms; BRAE_CONTROL_BOUND_PATCH_LIVE=1, the patch half gone, on the start and history arms;
# on kOmegaSST's solve arm, BRAE_CONTROL_SST_OMEGA_LIVE=1, grad(omega) reading the wall cell's new value on a neighbouring face.
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


def write(case, name, cls, text):
    with open(case + '/' + name, 'w') as f:
        f.write(H % (cls, os.path.basename(name)) + text)


def mesh(case, model):
    for d in ('', '/constant', '/system'):
        os.mkdir(case + d)
    second = 'epsilon' if model == 'kEpsilon' else 'omega'
    write(case, 'system/controlDict', 'dictionary', 'application simpleFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 1;\ndeltaT 1; writeControl timeStep; writeInterval 1; writeFormat ascii; writePrecision 17; timeFormat general; timePrecision 6; runTimeModifiable false;\n')
    write(case, 'system/blockMeshDict', 'dictionary', '''scale 1;
vertices ( (-5 -4 0)(15 -4 0)(15 4 0)(-5 4 0)(-2 -4 8)(18 -4 8)(18 4 8)(-2 4 8) );
blocks ( hex (0 1 2 3 4 5 6 7) (40 16 16) simpleGrading (2 1 3) );
edges ();
boundary
(
    sideA { type symmetry; faces ((0 1 5 4)); }
    sideB { type patch; faces ((2 3 7 6)); }
    inlet { type patch; faces ((0 4 7 3)); }
    outlet { type patch; faces ((1 2 6 5)); }
    lowerWall { type wall; faces ((0 3 2 1)); }
    top { type patch; faces ((4 5 6 7)); }
);
''')
    write(case, 'system/topoSetDict', 'dictionary', '''actions
(
    { name keep; type cellSet; action new; source boxToCell; box (-100 -100 -100)(100 100 100); }
    { name keep; type cellSet; action subtract; source boxToCell; box (1 -0.8 0.5)(4 0.8 2.5); }
);
''')
    write(case, 'system/fvSchemes', 'dictionary', '''ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; div(phi,U) bounded Gauss upwind; div(phi,k) bounded Gauss limitedLinear 1; div(phi,%s) bounded Gauss limitedLinear 1; div((nuEff*dev2(T(grad(U))))) Gauss linear; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
''' % second)
    solution(case, second, 1e-12, 1e-13)
    write(case, 'constant/transportProperties', 'dictionary', 'transportModel Newtonian; nu 1.5e-5;\n')
    turbulence(case, model, 'on')


def turbulence(case, model, on):
    write(case, 'constant/turbulenceProperties', 'dictionary', 'simulationType RAS; RAS { RASModel %s; turbulence %s; printCoeffs on; }\n' % (model, on))


def solution(case, second, ptol, tol):
    write(case, 'system/fvSolution', 'dictionary', '''solvers
{
    p { solver GAMG; smoother GaussSeidel; tolerance %g; relTol 0; maxIter 5000; }
    Phi { solver GAMG; smoother GaussSeidel; tolerance 1e-10; relTol 0; }
    "(U|k|%s)" { solver smoothSolver; smoother GaussSeidel; nSweeps 1; tolerance %g; relTol 0; maxIter 5000; }
}
SIMPLE { nNonOrthogonalCorrectors 0; }
potentialFlow { nNonOrthogonalCorrectors 5; }
relaxationFactors { equations { U 0.9; k 0.7; %s 0.7; } fields { p 0.3; } }
''' % (ptol, second, tol, second))


def fields(case, model, drive):
    second = 'epsilon' if model == 'kEpsilon' else 'omega'
    os.makedirs(case + '/0', exist_ok=True)
    T = 'dimensions %s; internalField uniform %s;\nboundaryField {\n    sideA { type symmetry; }\n    sideB { %s }\n    inlet { %s }\n    outlet { %s }\n    lowerWall { %s }\n    top { %s }\n    body { %s }\n}\n'
    write(case, '0/U', 'volVectorField', T % ('[0 1 -1 0 0 0 0]', '(20 0 0)', 'type slip;', 'type fixedValue; value uniform (20 0 0);', 'type inletOutlet; inletValue uniform (0 0 0); value uniform (20 0 0);', 'type fixedValue; value uniform (20 0 0);', 'type slip;', 'type noSlip;'))
    write(case, '0/p', 'volScalarField', T % ('[0 2 -2 0 0 0 0]', '0', 'type slip;', 'type zeroGradient;', 'type fixedValue; value uniform 0;', 'type zeroGradient;', 'type slip;', 'type zeroGradient;'))
    K, S = (0.24, 1.78) if model != 'kEpsilon' else (0.24, 0.05)
    hi = 300 if drive else 1
    kw = 'type kqRWallFunction; value uniform %g;' % K
    # the outlet fixed far above the field: limitedLinear's face values next to it overshoot, which drives a tight solve negative
    write(case, '0/k', 'volScalarField', T % ('[0 2 -2 0 0 0 0]', K, 'type zeroGradient;', 'type fixedValue; value uniform %g;' % K, 'type fixedValue; value uniform %g;' % (K * hi), kw, 'type zeroGradient;', kw))
    # the wall function's file value differs from every cell: OpenFOAM's constructor replaces it with the cell's
    sw = 'type %sWallFunction; value uniform %g;' % (second, 7 * S)
    write(case, '0/' + second, 'volScalarField', T % ('[0 0 -1 0 0 0 0]' if model != 'kEpsilon' else '[0 2 -3 0 0 0 0]', S, 'type zeroGradient;', 'type fixedValue; value uniform %g;' % S, 'type fixedValue; value uniform %g;' % (S * hi), sw, 'type zeroGradient;', sw))
    nw = 'type nutkWallFunction; value uniform 0;'
    c = 'type calculated; value uniform 0;'
    write(case, '0/nut', 'volScalarField', T % ('[0 2 -1 0 0 0 0]', 0, c, c, c, nw, c, nw))


def owners(case):
    def lst(name):
        b = open(case + '/constant/polyMesh/' + name).read()
        m = re.search(r'\n(\d+)\s*\n\(', b)
        rest = b[m.end():]
        return np.array(rest[:rest.index('\n)')].split(), dtype=int)
    own = lst('owner')
    b = open(case + '/constant/polyMesh/boundary').read()
    patches = {}
    for m in re.finditer(r'\n\s+(\w+)\n\s+\{[^}]*?nFaces\s+(\d+);[^}]*?startFace\s+(\d+);', b):
        patches[m.group(1)] = own[int(m.group(3)):int(m.group(3)) + int(m.group(2))]
    return own, patches


def pattern(case, model, how):
    """A smooth positive field spanning a decade, so the average is not the field's own mean; `every` makes every 7th cell and cells on every patch negative, `inside` only cells with no boundary face."""
    second = 'epsilon' if model == 'kEpsilon' else 'omega'
    own, patches = owners(case)
    n = own.max() + 1
    touched = np.zeros(n, dtype=bool)
    for cells in patches.values():
        touched[cells] = True
    chosen = np.zeros(n, dtype=bool)
    if how != 'positive':
        chosen[::7] = True
    if how == 'every':
        for cells in patches.values():
            chosen[cells[::3]] = True
    if how == 'inside':
        chosen[touched] = False
    for f, base in (('k', 0.24), (second, 1.78 if second == 'omega' else 0.05)):
        vals = base * (1.0 + 9.0 * np.abs(np.sin(0.37 * np.arange(n))))
        vals[chosen] = -vals[chosen]
        p = case + '/0/' + f
        s = open(p).read()
        s, k = re.subn(r'internalField\s+uniform\s+[^;]*;', 'internalField nonuniform List<scalar> %d(%s);' % (n, ' '.join('%.17g' % v for v in vals)), s, count=1)
        assert k == 1
        open(p, 'w').write(s)
    np.save(case + '/chosen.npy', chosen)
    return {name: int(chosen[cells].sum()) for name, cells in patches.items()}, int(chosen.sum())


def read(fn):
    b = open(fn).read()
    m = re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n?(\d+)\s*\n?\(', b)
    if not m:
        return None
    n, nc = int(m.group(2)), 3 if m.group(1) == 'vector' else 1
    body = b[m.end():]
    body = body[:body.index('\n)')]
    return np.array(body.replace('(', ' ').replace(')', ' ').split(), dtype=float).reshape(n, nc)


def gaps(of, br, names, t):
    out = {}
    for f in names:
        a = read('%s/%s/%s' % (of, t, f))
        b = read('%s/%s/%s' % (br, t, f))
        out[f] = np.linalg.norm(b - a) / np.linalg.norm(a)
    return out


def compare(of, br, model, t, want, held, chosen=None):
    second = 'epsilon' if model == 'kEpsilon' else 'omega'
    g = gaps(of, br, ('U', 'p', 'k', second, 'nut'), t)
    names = ('k', second, 'nut') if held == 'scalars' else tuple(g)
    ok = max(g[f] for f in names) <= 1e-10
    line = ' '.join('%s %.2e' % kv for kv in g.items())
    if held == 'scalars':
        line += ' (U, p printed)'
    if chosen is not None:
        c = np.load(chosen)
        for f in ('k', second):
            a = read('%s/%s/%s' % (of, t, f))[c]
            b = read('%s/%s/%s' % (br, t, f))[c]
            line += ' | %s bounded cells %.2e' % (f, np.linalg.norm(b - a) / np.linalg.norm(a))
    print('%s %s' % ('within' if ok else 'MISSES', line))
    return ok == (want != 'miss')


def history(of, br, second):
    def rows(fn):
        out = []
        for block in open(fn, errors='replace').read().split('\nTime = ')[1:]:
            out.append({m.group(1): float(m.group(2)) for m in re.finditer(r'Solving for (\w+), Initial residual = ([0-9.e+-]+)', block)})
        return out
    def bounds(fn):
        return len(re.findall(r'(?m)^bounding ', open(fn, errors='replace').read()))
    a, b = rows(of + '/log'), rows(br + '/log')
    worst = {}
    for x, y in zip(a, b):
        for f in x:
            worst[f] = max(worst.get(f, 0.0), abs(y[f] - x[f]) / max(abs(x[f]), 1e-300))
    ok = len(a) == len(b) == 30 and bounds(of + '/log') > 0 and bounds(of + '/log') == bounds(br + '/log')
    for f, w in worst.items():
        ok = ok and w <= (2e-4 if f == second else 2e-5)
    print('%s iterations %d %d, bounding lines %d %d; worst relative residual gap %s' % ('within' if ok else 'MISSES', len(a), len(b), bounds(of + '/log'), bounds(br + '/log'), ' '.join('%s %.2e' % kv for kv in worst.items())))
    return ok


what = sys.argv[1]
if what == 'mesh':
    mesh(sys.argv[2], sys.argv[3])
elif what == 'fields':
    fields(sys.argv[2], sys.argv[3], sys.argv[4] == 'drive')
elif what == 'pattern':
    print(pattern(sys.argv[2], sys.argv[3], sys.argv[4]))
elif what == 'turbulence':
    turbulence(sys.argv[2], sys.argv[3], sys.argv[4])
elif what == 'solution':
    solution(sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]))
elif what == 'end':
    p = sys.argv[2] + '/system/controlDict'
    s = open(p).read()
    s = re.sub(r'endTime\s+[^;]*;', 'endTime %s;' % sys.argv[3], s)
    s = re.sub(r'writeInterval\s+[^;]*;', 'writeInterval %s;' % sys.argv[3], s)
    open(p, 'w').write(s)
elif what == 'compare':
    sys.exit(0 if compare(*sys.argv[2:8], *(sys.argv[8:9])) else 1)
elif what == 'history':
    sys.exit(0 if history(*sys.argv[2:5]) else 1)
PY
g()
{
    python3 -B "$W/gate.py" "$@"
}
brae()   # <dir> [VAR=value ...]
{
    local c=$1
    shift
    ( cd "$c" && env BRAE_SIMPLE_CASE_SOLVER=1 BRAE_DICT_AUDIT=0 $F1 "$@" "$BRAE" -case . > log 2> err < /dev/null )
}
copy()
{
    mkdir "$2"
    cp -r "$1/0" "$1/constant" "$1/system" "$2/"
}
fail=0
for model in kEpsilon kOmegaSST
do
    second=epsilon
    F1=""
    if [ "$model" = kOmegaSST ]
    then
        second=omega
        F1=BRAE_SST_F1_PATCH=1
    fi
    M=$W/mesh-$model
    g mesh "$M" "$model" || { echo "FAIL mesh"; exit 1; }
    of "cd '$M' && blockMesh > log.blockMesh 2>&1 && topoSet > log.topoSet 2>&1 && subsetMesh keep -patch body -overwrite > log.subsetMesh 2>&1" < /dev/null || { echo "FAIL mesh $model"; exit 1; }
    # subsetMesh's new patch is `empty`, which takes every direction from the solver
    sed -i -E '/^ +body$/,/^ +\}$/ s/type +empty;/type wall;/' "$M/constant/polyMesh/boundary"
    grep -q 'type *wall' "$M/constant/polyMesh/boundary" || { echo "FAIL the body patch"; exit 1; }
    # START and INTERIOR: the constructor's bound
    for arm in "start every all" "interior inside all"
    do
        read -r name how held <<< "$arm"
        S=$W/$name-$model
        cp -r "$M" "$S"
        g fields "$S" "$model" plain
        of "cd '$S' && potentialFoam -writephi > log.potentialFoam 2>&1" < /dev/null || { echo "FAIL potentialFoam $name $model"; fail=1; continue; }
        echo "  $model $name, bounded cells: $(g pattern "$S" "$model" "$how")"
        g turbulence "$S" "$model" off
        for who in of br ctl lve
        do
            cp -r "$S" "$S-$who"
        done
        of "cd '$S-of' && simpleFoam > log 2>&1" < /dev/null || { echo "FAIL OpenFOAM $name $model"; fail=1; continue; }
        grep -q '^bounding k' "$S-of/log" && grep -q "^bounding $second" "$S-of/log" || { echo "FAIL premise: OpenFOAM did not bound"; fail=1; }
        brae "$S-br" || { echo "FAIL brae $name $model"; fail=1; continue; }
        grep -q '^bounding k' "$S-br/log" && grep -q "^bounding $second" "$S-br/log" || { echo "FAIL premise: brae did not bound"; fail=1; }
        printf '  %-10s %-8s ' "$model" "$name"
        g compare "$S-of" "$S-br" "$model" 1 hold "$held" "$S/chosen.npy" || fail=1
        brae "$S-ctl" BRAE_CONTROL_BOUND_UNWEIGHTED=1
        printf '  CONTROL old average    '
        g compare "$S-of" "$S-ctl" "$model" 1 miss "$held" "$S/chosen.npy" || fail=1
        if [ "$name" = start ]
        then
            brae "$S-lve" BRAE_CONTROL_BOUND_PATCH_LIVE=1
            printf '  CONTROL no patch half  '
            g compare "$S-of" "$S-lve" "$model" 1 miss "$held" "$S/chosen.npy" || fail=1
        fi
    done
    # SOLVE: correct()'s bound, one tight iteration
    V=$W/solve-$model
    cp -r "$M" "$V"
    g fields "$V" "$model" drive
    g pattern "$V" "$model" positive > /dev/null
    of "cd '$V' && potentialFoam -writephi > log.potentialFoam 2>&1" < /dev/null || { echo "FAIL potentialFoam"; fail=1; continue; }
    for who in of br ctl
    do
        cp -r "$V" "$V-$who"
    done
    of "cd '$V-of' && simpleFoam > log 2>&1" < /dev/null || { echo "FAIL OpenFOAM solve $model"; fail=1; continue; }
    grep -q '^bounding ' "$V-of/log" || { echo "FAIL premise: OpenFOAM's solve did not bound"; fail=1; }
    brae "$V-br" || { echo "FAIL brae solve $model"; fail=1; continue; }
    printf '  %-10s solve    ' "$model"
    g compare "$V-of" "$V-br" "$model" 1 hold all || fail=1
    brae "$V-ctl" BRAE_CONTROL_BOUND_UNWEIGHTED=1
    printf '  CONTROL old average    '
    g compare "$V-of" "$V-ctl" "$model" 1 miss all || fail=1
    if [ "$model" = kOmegaSST ]
    then
        cp -r "$V" "$V-lve"
        brae "$V-lve" BRAE_CONTROL_SST_OMEGA_LIVE=1
        printf '  CONTROL omega live     '
        g compare "$V-of" "$V-lve" "$model" 1 miss all || fail=1
    fi
    # HISTORY: thirty iterations at looser entries, the residuals each iteration
    Y=$W/history-$model
    cp -r "$V" "$Y"
    g end "$Y" 30
    for who in of br lve nrm
    do
        cp -r "$Y" "$Y-$who"
    done
    of "cd '$Y-of' && simpleFoam > log 2>&1" < /dev/null || { echo "FAIL OpenFOAM history $model"; fail=1; continue; }
    brae "$Y-br" || { echo "FAIL brae history $model"; fail=1; continue; }
    printf '  %-10s history  ' "$model"
    g history "$Y-of" "$Y-br" "$second" || fail=1
    brae "$Y-lve" BRAE_CONTROL_BOUND_PATCH_LIVE=1
    printf '  CONTROL no patch half  '
    if g history "$Y-of" "$Y-lve" "$second"
    then
        echo "FAIL CONTROL the patch half's history stayed green"
        fail=1
    fi
    # Red on kEpsilon alone: omega's wall rows are 8.3e-05 here with it and 7.6e-05 without, and tests/simple_linearupwind_turbulence_vs_openfoam.sh holds them where they show.
    if [ "$model" = kEpsilon ]
    then
        brae "$Y-nrm" BRAE_CONTROL_WALL_ROW_NORM=1
        printf '  CONTROL wall rows      '
        if g history "$Y-of" "$Y-nrm" "$second"
        then
            echo "FAIL CONTROL the wall rows' norm factor stayed green"
            fail=1
        fi
    fi
done
[ "$fail" = 0 ] && echo "PASS: a bounded cell takes OpenFOAM's area-weighted average, and a bounded face OpenFOAM's floor until its next evaluate" || { echo FAIL; exit 1; }
