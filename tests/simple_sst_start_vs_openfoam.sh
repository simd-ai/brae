#!/usr/bin/env bash
# kOmegaSST's first correctNut against real OpenFOAM on a 3-D duct (40 x 15 x 15, a wall all round, kOmegaSST). kOmegaSST::validate() takes fvc::grad(U) with U's patch values as read; brae took an
# inletOutlet patch at its inletValue, since nothing had resolved its flux yet. MEASURED on JUPITER, restart from OpenFOAM's iteration 50, outlet inletOutlet with inletValue 0: nut 19x low in the 225
# outlet-column cells for iteration 1, U 3.8e-04 from OpenFOAM after it, and with the outlet zeroGradient, the same code, U 7.2e-15. pitzDaily is blind: its momentum has no z, and a first iteration
# from rest there cannot tell.
#
# ARMS, one SIMPLE iteration each, every solve tight (p 1e-12, the rest 1e-13), OpenFOAM's sweep order: {outlet inletOutlet, outlet zeroGradient} x {cold from 0, restart from OpenFOAM's iteration 20}.
# THE ORACLE: real OpenFOAM on the same bytes. On JUPITER use a build that passes scripts/jsc/of_oracle.sh of brae-multigpu: its installed v2506 returns the wrong xz from T(tmp<volTensorField>).
# THE BOUNDS, each arm: U within 1e-12, p within 1e-10 after the iteration; Ux, Uy, Uz and p first residuals within 5e-6 relative, brae printing six digits. k, omega and nut are printed and not bound on
# the plain arm: they carry the inlet's F1 from the adjacent cell. The arm with BRAE_SST_F1_PATCH=1 holds them too, each within 1e-12. MEASURED restarted (JUPITER job 2622756): k 1.9e-14 omega 8.2e-15
# nut 5.4e-15, and 1.55e-06, 7.51e-06, 3.32e-05 without it. COLD, that bound is on upwind k and omega only (the coldup arm): limitedLinear from a uniform start breaks its round-off ties anyhow. MEASURED
# (job 2629994), OpenFOAM against itself with every k and omega moved a coin's 1 ulp: limitedLinear k 2.1e-02 omega 3.5e-02 nut 8.7e-02, upwind 4.8e-16, 3.9e-16, 6.2e-16.
# CONTROL, asserted red: BRAE_CONTROL_SST_START_EVALUATE=1 on the inletOutlet arms, the old value: U or a first residual must miss.
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


def field(case, name, cls, dims, value, inlet, outlet, walls):
    write(case, '0/' + name, cls, 'dimensions %s; internalField uniform %s;\nboundaryField {\n    inlet { %s }\n    outlet { %s }\n    walls { %s }\n}\n' % (dims, value, inlet, outlet, walls))


def stage(case, outlet, turbulence='limitedLinear'):
    for d in ('', '/0', '/constant', '/system'):
        os.mkdir(case + d)
    write(case, 'system/controlDict', 'dictionary', 'application simpleFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 1;\ndeltaT 1; writeControl timeStep; writeInterval 1; writeFormat ascii; writePrecision 17;\nwriteCompression off; timeFormat general; timePrecision 6; runTimeModifiable false;\n')
    write(case, 'system/blockMeshDict', 'dictionary', 'scale 1;\nvertices ( (0 0 0)(2 0 0)(2 0.2 0)(0 0.2 0)(0 0 0.2)(2 0 0.2)(2 0.2 0.2)(0 0.2 0.2) );\nblocks ( hex (0 1 2 3 4 5 6 7) (40 15 15)\n  simpleGrading (1 ((0.5 0.5 6)(0.5 0.5 0.1667)) ((0.5 0.5 6)(0.5 0.5 0.1667))) );\nedges ();\nboundary\n(\n    inlet  { type patch; faces ( (0 4 7 3) ); }\n    outlet { type patch; faces ( (1 2 6 5) ); }\n    walls  { type wall;  faces ( (0 1 5 4)(3 2 6 7)(0 3 2 1)(4 5 6 7) ); }\n);\n')
    write(case, 'system/fvSchemes', 'dictionary', 'ddtSchemes { default steadyState; }\ngradSchemes { default Gauss linear; }\ndivSchemes { default none; div(phi,U) bounded Gauss linearUpwind grad(U);\n  div(phi,k) bounded Gauss TURB; div(phi,omega) bounded Gauss TURB;\n  div((nuEff*dev2(T(grad(U))))) Gauss linear; }\nlaplacianSchemes { default Gauss linear corrected; }\ninterpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\nwallDist { method meshWave; }\n'.replace('TURB', 'upwind' if turbulence == 'upwind' else 'limitedLinear 1'))
    write(case, 'system/fvSolution', 'dictionary', 'solvers { p { solver GAMG; smoother GaussSeidel; tolerance 1e-12; relTol 0; maxIter 5000; }\n  "(U|k|omega)" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-13; relTol 0; maxIter 2000; } }\nSIMPLE { nNonOrthogonalCorrectors 0; }\nrelaxationFactors { fields { p 0.3; } equations { U 0.7; k 0.7; omega 0.7; } }\n')
    write(case, 'constant/transportProperties', 'dictionary', 'transportModel Newtonian; nu 1e-4;\n')
    write(case, 'constant/turbulenceProperties', 'dictionary', 'simulationType RAS; RAS { RASModel kOmegaSST; turbulence on; printCoeffs on; }\n')
    o = 'type inletOutlet; inletValue uniform (0 0 0); value uniform (10 0 0);' if outlet == 'io' else 'type zeroGradient;'
    field(case, 'U', 'volVectorField', '[0 1 -1 0 0 0 0]', '(10 0 0)', 'type fixedValue; value uniform (10 0 0);', o, 'type noSlip;')
    field(case, 'p', 'volScalarField', '[0 2 -2 0 0 0 0]', '0', 'type zeroGradient;', 'type fixedValue; value uniform 0;', 'type zeroGradient;')
    field(case, 'k', 'volScalarField', '[0 2 -2 0 0 0 0]', '0.375', 'type fixedValue; value uniform 0.375;', 'type inletOutlet; inletValue uniform 0.375; value uniform 0.375;', 'type kqRWallFunction; value uniform 0.375;')
    field(case, 'omega', 'volScalarField', '[0 0 -1 0 0 0 0]', '56.2963', 'type fixedValue; value uniform 56.2963;', 'type inletOutlet; inletValue uniform 56.2963; value uniform 56.2963;', 'type omegaWallFunction; value uniform 56.2963;')
    field(case, 'nut', 'volScalarField', '[0 2 -1 0 0 0 0]', '0', 'type calculated; value uniform 0;', 'type calculated; value uniform 0;', 'type nutkWallFunction; value uniform 0;')


def restart(case, start, end):
    c = open(case + '/system/controlDict').read()
    c = re.sub(r'startTime\s+[^;]*;', 'startTime %s;' % start, c)
    c = re.sub(r'endTime\s+[^;]*;', 'endTime %s;' % end, c)
    open(case + '/system/controlDict', 'w').write(c)


def read(fn):
    b = open(fn).read()
    m = re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n?(\d+)\s*\n?\(', b)
    n, nc = int(m.group(2)), 3 if m.group(1) == 'vector' else 1
    body = b[m.end():]
    body = body[:body.index('\n)')]
    return np.array(body.replace('(', ' ').replace(')', ' ').split(), dtype=float).reshape(n, nc)


def first(log):
    found = {}
    for line in open(log, errors='replace'):
        m = re.search(r'Solving for (Ux|Uy|Uz|p), Initial residual = ([0-9.e+-]+)', line)
        if m and m.group(1) not in found:
            found[m.group(1)] = float(m.group(2))
    return found


def compare(ofcase, brcase, t):
    gaps = {f: np.linalg.norm(read('%s/%s/%s' % (brcase, t, f)) - read('%s/%s/%s' % (ofcase, t, f))) / np.linalg.norm(read('%s/%s/%s' % (ofcase, t, f))) for f in ('U', 'p', 'k', 'omega', 'nut')}
    ro, rb = first(ofcase + '/log'), first(brcase + '/log')
    res = max(abs(rb[k] - ro[k]) / ro[k] for k in ro)
    return gaps, res, ro, rb


what = sys.argv[1]
if what == 'stage':
    stage(*sys.argv[2:])
elif what == 'restart':
    restart(sys.argv[2], sys.argv[3], sys.argv[4])
else:
    gaps, res, ro, rb = compare(sys.argv[2], sys.argv[3], sys.argv[4])
    want = sys.argv[5]
    ok = gaps['U'] <= 1e-12 and gaps['p'] <= 1e-10 and res <= 5e-6
    if want == 'all':
        ok = ok and gaps['k'] <= 1e-12 and gaps['omega'] <= 1e-12 and gaps['nut'] <= 1e-12
    print('%s U %.2e p %.2e k %.2e omega %.2e nut %.2e; first residuals worst %.2e (OpenFOAM Ux %.4e Uz %.4e p %.4e; brae Ux %.4e Uz %.4e p %.4e)' % ('within' if ok else 'MISSES', gaps['U'], gaps['p'], gaps['k'], gaps['omega'], gaps['nut'], res, ro['Ux'], ro['Uz'], ro['p'], rb['Ux'], rb['Uz'], rb['p']))
    sys.exit(0 if ok == (want != 'miss') else 1)
PY
fail=0
BASE="$W/mesh"
python3 -B "$W/gate.py" stage "$BASE" io
of "cd '$BASE' && blockMesh > log.blockMesh 2>&1" < /dev/null || { echo "FAIL blockMesh"; exit 1; }
for outlet in io zg
do
    # cold: both from 0
    cold="$W/cold-$outlet"
    python3 -B "$W/gate.py" stage "$cold" "$outlet"
    cp -r "$BASE/constant/polyMesh" "$cold/constant/"
    coldup="$W/coldup-$outlet"
    python3 -B "$W/gate.py" stage "$coldup" "$outlet" upwind
    cp -r "$BASE/constant/polyMesh" "$coldup/constant/"
    # restart: OpenFOAM to iteration 20, then each from its bytes
    dev="$W/dev-$outlet"
    python3 -B "$W/gate.py" stage "$dev" "$outlet"
    cp -r "$BASE/constant/polyMesh" "$dev/constant/"
    python3 -B "$W/gate.py" restart "$dev" 0 20
    of "cd '$dev' && simpleFoam > log 2>&1" < /dev/null || { echo "FAIL OpenFOAM development run, outlet $outlet"; fail=1; continue; }
    for kind in cold coldup restart
    do
        for who in of brae ctl f1
        do
            c="$W/$kind-$outlet-$who"
            mkdir "$c"
            if [ "$kind" = cold ]
            then
                cp -r "$cold/0" "$cold/constant" "$cold/system" "$c/"
                t=1
            elif [ "$kind" = coldup ]
            then
                cp -r "$coldup/0" "$coldup/constant" "$coldup/system" "$c/"
                t=1
            else
                cp -r "$dev/constant" "$dev/system" "$dev/20" "$c/"
                rm -rf "${c:?}/20/uniform"
                python3 -B "$W/gate.py" restart "$c" 20 21
                t=21
            fi
        done
        of "cd '$W/$kind-$outlet-of' && simpleFoam > log 2>&1" < /dev/null || { echo "FAIL OpenFOAM $kind $outlet"; fail=1; continue; }
        ( cd "$W/$kind-$outlet-brae" && BRAE_SIMPLE_CASE_SOLVER=1 "$BRAE" -case . > log 2> err < /dev/null ) || { echo "FAIL brae $kind $outlet exited $?"; fail=1; continue; }
        printf '  %-8s %-3s ' "$kind" "$outlet"
        python3 -B "$W/gate.py" compare "$W/$kind-$outlet-of" "$W/$kind-$outlet-brae" "$t" hold || fail=1
        ( cd "$W/$kind-$outlet-f1" && BRAE_SIMPLE_CASE_SOLVER=1 BRAE_SST_F1_PATCH=1 "$BRAE" -case . > log 2> err < /dev/null ) || { echo "FAIL brae F1 $kind $outlet exited $?"; fail=1; continue; }
        printf '  F1 %-8s %-3s ' "$kind" "$outlet"
        if [ "$kind" = cold ]
        then
            python3 -B "$W/gate.py" compare "$W/$kind-$outlet-of" "$W/$kind-$outlet-f1" "$t" hold || fail=1
        else
            python3 -B "$W/gate.py" compare "$W/$kind-$outlet-of" "$W/$kind-$outlet-f1" "$t" all || fail=1
        fi
        if [ "$outlet" = io ] && [ "$kind" != coldup ]
        then
            ( cd "$W/$kind-$outlet-ctl" && BRAE_SIMPLE_CASE_SOLVER=1 BRAE_CONTROL_SST_START_EVALUATE=1 "$BRAE" -case . > log 2> err < /dev/null ) || { echo "FAIL control $kind exited $?"; fail=1; continue; }
            printf '  CONTROL %-8s ' "$kind"
            python3 -B "$W/gate.py" compare "$W/$kind-$outlet-of" "$W/$kind-$outlet-ctl" "$t" miss || fail=1
        fi
    done
done
[ "$fail" = 0 ] && echo "PASS: kOmegaSST's first iteration is OpenFOAM's on both outlets, cold and restarted" || { echo FAIL; exit 1; }
