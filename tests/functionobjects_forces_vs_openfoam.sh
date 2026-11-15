#!/usr/bin/env bash
# functionObjects::forces and ::forceCoeffs (src/functionObjects/forces_fo.cuh) against OpenFOAM v2412 ON THE SAME FIELDS: brae
# runs 3 iterations with its functionObjects every step; OpenFOAM's simpleFoam -postProcess -time 3 then runs the same
# functionObjects on the fields brae wrote at 3. Any difference is the functionObject's, not the solver's.
#
# CASES (each 3 iterations, written at 3, ascii at 17 digits, OpenFOAM's sweep order):
#   sst   validation/pitzDailySST, kOmegaSST, grad(U) cellLimited Gauss linear 1 (the official DrivAer's scheme)
#   ke    validation/pitzDaily, kEpsilon, nutkWallFunction walls
#   cav   validation/cav3d_cf, laminar, 3-D: forces and moments in all three directions
# OBJECTS on each case:
#   fcA   forceCoeffs on a regular expression of the walls; rhoInf 1.2, pRef 3.5, an off-centre CofR, drag and lift off the
#         axes (15 degrees about z; in 3-D a general direction), every coefficient, front and rear parts included
#   fcB   forceCoeffs on one literal patch, an axisAngle coordinate system, a subset of coefficients
#   fA    forces on a literal patch list, CofR
#   fB    forces on a regular expression, origin and e1/e2 axes of their own
# BOUND   every column of the row at 3 within 1e-12 of OpenFOAM's, relative to the largest magnitude in that row (a component
#         that is zero in 2-D is held to the row's scale); every header line byte for byte OpenFOAM's
# CONTROLS (must be red, sst):
#   BRAE_CONTROL_FORCES_GRAD_CELL    the wall gradient without gaussGrad's boundary correction (the viscous force)
#   BRAE_CONTROL_FORCES_GLOBAL_AXES  the global components whatever the axes
# LIFECYCLE (sst, 4 iterations, the case writing at 3; OpenFOAM's own run of the same case gives the times):
#   L2 writeInterval 2; Lstart timeStart 2 timeEnd 3; Lwrite writeControl writeTime; Lexec executeInterval 2 and writeInterval 2;
#   Loff enabled false: each object's rows at the times OpenFOAM's are, and no file where OpenFOAM writes none
# RESTART (sst): 2 iterations, then on to 4 from latestTime: postProcessing/<name>/0 holds 1 and 2, postProcessing/<name>/2
#   holds 3 and 4, as OpenFOAM's restart lays them out
# REFUSALS (sst): directForceDensity, rho rho, writeFields, an unknown coefficient, a cylindrical coordinate system: each named
#   NOT run, and no file; binData and a pitchAxis that is not e2: each said to be unread
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
[ -x "$BRAE" ]                         || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -f "$OFBASHRC" ]                     || { echo "SKIP: real OpenFOAM not available"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
for d in pitzDailySST pitzDaily cav3d_cf
do
    [ -f "$ROOT/validation/$d/constant/polyMesh/points" ] || { echo "SKIP: fixture validation/$d missing"; exit 77; }
done
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v simpleFoam > /dev/null 2>&1 || { echo "SKIP: simpleFoam not on PATH"; exit 77; }
W=$(mktemp -d)
[ -n "${KEEP_W:-}" ] && { W="$KEEP_W"; rm -rf "${W:?}"; mkdir -p "$W"; } || trap 'rm -rf "$W"' EXIT
fail=0
say()
{
    printf '  %-110s %s\n' "$1" "$2"
    [ "$2" = FAIL ] && fail=1 || true
}
cat > "$W/fo.py" << 'PY'
import math, os, re, sys

WALLS = {'sst': ('(upper|lower)Wall', 'lowerWall', 'upperWall lowerWall', 'lower.*'),
         'ke': ('(upper|lower)Wall', 'upperWall', 'lowerWall upperWall', 'upper.*'),
         'cav': ('(movingWall|fixedWalls)', 'fixedWalls', 'movingWall fixedWalls', 'moving.*')}


def objects(case):
    rx, one, lits, rx2 = WALLS[case]
    if case == 'cav':
        drag, lift = '(0.8 0.36 0.48)', '(-0.36 0.928 -0.096)'
        cofr, origin = '(0.03 0.07 0.04)', '(0.05 0.05 0.05)'
    else:
        drag, lift = '(0.9659258262890683 0.25881904510252074 0)', '(-0.25881904510252074 0.9659258262890683 0)'
        cofr, origin = '(0.1 0.01 0.0005)', '(0 0.01 0)'
    return """
    fcA
    {
        type            forceCoeffs;
        libs            (forces);
        patches         ("%s");
        rho             rhoInf;
        rhoInf          1.2;
        pRef            3.5;
        magUInf         10;
        lRef            0.1;
        Aref            0.05;
        CofR            %s;
        dragDir         %s;
        liftDir         %s;
        log             false;
    }
    fcB
    {
        type            forceCoeffs;
        libs            (forces);
        patches         (%s);
        rho             rhoInf;
        rhoInf          1;
        magUInf         10;
        lRef            0.2;
        Aref            0.03;
        coordinateSystem
        {
            origin      (0.05 -0.01 0);
            rotation
            {
                type    axisAngle;
                axis    (1 2 3);
                angle   37;
            }
        }
        coefficients    (Cd Cl CmPitch Cd(f) Cl(r) CmYaw Cs);
        log             false;
    }
    fA
    {
        type            forces;
        libs            (forces);
        patches         (%s);
        rho             rhoInf;
        rhoInf          1;
        CofR            (0.2 0 0);
        log             false;
    }
    fB
    {
        type            forces;
        libs            (forces);
        patches         ("%s");
        rho             rhoInf;
        rhoInf          1;
        origin          %s;
        e1              (0.8 0.6 0);
        e2              (-0.6 0.8 0);
        log             false;
    }
""" % (rx, cofr, drag, lift, one, lits, rx2, origin)


def lifecycle():
    base = """
        type            forceCoeffs;
        libs            (forces);
        patches         (lowerWall);
        rho             rhoInf;
        rhoInf          1;
        magUInf         10;
        lRef            0.1;
        Aref            0.05;
        CofR            (0 0 0);
        dragDir         (1 0 0);
        liftDir         (0 1 0);
        coefficients    (Cd Cl);
        log             false;
"""
    return ''.join('    %s\n    {\n%s%s    }\n' % (n, base, extra) for n, extra in (
        ('L2', '        writeControl    timeStep;\n        writeInterval   2;\n'),
        ('Lstart', '        timeStart       2;\n        timeEnd         3;\n'),
        ('Lwrite', '        writeControl    writeTime;\n'),
        ('Lexec', '        executeControl  timeStep;\n        executeInterval 2;\n        writeControl    timeStep;\n        writeInterval   2;\n'),
        ('Loff', '        enabled         false;\n')))


def refusals():
    base = """
        type            forceCoeffs;
        libs            (forces);
        patches         (lowerWall);
        rhoInf          1;
        magUInf         10;
        lRef            0.1;
        Aref            0.05;
        CofR            (0 0 0);
        dragDir         (1 0 0);
        liftDir         (0 1 0);
        log             false;
"""
    out = ''
    for n, extra in (('Rdfd', '        rho rhoInf;\n        directForceDensity yes;\n'),
                     ('Rrho', '        rho rho;\n'),
                     ('Rwf', '        rho rhoInf;\n        writeFields yes;\n'),
                     ('Rcoef', '        rho rhoInf;\n        coefficients (Cd Cq);\n'),
                     ('Rcyl', '        rho rhoInf;\n        coordinateSystem { type cylindrical; origin (0 0 0); rotation { type axes; e3 (0 0 1); e1 (1 0 0); } }\n'),
                     ('Ubin', '        rho rhoInf;\n        binData { nBin 10; direction (1 0 0); cumulative yes; }\n'),
                     ('Upitch', '        rho rhoInf;\n        pitchAxis (0 1 0);\n')):
        b = base.replace('        CofR            (0 0 0);\n', '') if n == 'Rcyl' else base
        out += '    %s\n    {\n%s%s    }\n' % (n, b, extra)
    return out


def stage(src, dst, case, end, write, funcs):
    os.system('rm -rf "%s" && mkdir -p "%s" && cp -r "%s/0" "%s/constant" "%s/system" "%s/"' % (dst, dst, src, src, src, dst))
    for f in os.listdir(dst + '/constant/polyMesh'):
        if f.startswith('.brae_'):
            os.remove(dst + '/constant/polyMesh/' + f)
    p = dst + '/system/controlDict'
    s = open(p).read()
    s = re.sub(r'\nfunctions\s*\{.*?\n\}\n', '\n', s, flags=re.S)
    for k, v in (('startFrom', 'latestTime'), ('endTime', str(end)), ('writeControl', 'timeStep'), ('writeInterval', str(write)),
                 ('writeFormat', 'ascii'), ('writePrecision', '17'), ('purgeWrite', '0')):
        if re.search(r'(?m)^\s*%s\s' % k, s):
            s = re.sub(r'(?m)^(\s*%s\s+)[^;]*;' % k, lambda m: m.group(1) + v + ';', s)
        else:
            s = s.rstrip() + '\n%s %s;\n' % (k, v)
    s = s.rstrip() + '\n\nfunctions\n{\n' + funcs + '}\n'
    open(p, 'w').write(s)
    if case == 'sst':
        p = dst + '/system/fvSchemes'
        s = open(p).read()
        s = re.sub(r'(gradSchemes\s*\{[^}]*?default\s+Gauss linear;)', r'\1\n    grad(U)         cellLimited Gauss linear 1;', s, count=1)
        open(p, 'w').write(s)


def rows(path):
    head, out = [], {}
    for l in open(path):
        if l.startswith('#'):
            head.append(l.rstrip('\n'))
            continue
        w = l.split()
        if w:
            out[w[0]] = [float(x) for x in w[1:]]
    return head, out


def compare(br, of, t, bound):
    """every file OpenFOAM wrote under of/postProcessing against brae's under br/postProcessing/<obj>/0, row t"""
    worst, bad, n = 0.0, [], 0
    for obj in sorted(os.listdir(of + '/postProcessing')):
        for d in os.listdir(of + '/postProcessing/' + obj):
            for f in sorted(os.listdir(of + '/postProcessing/%s/%s' % (obj, d))):
                ho, ro = rows(of + '/postProcessing/%s/%s/%s' % (obj, d, f))
                pb = br + '/postProcessing/%s/0/%s' % (obj, f)
                if not os.path.exists(pb):
                    bad.append('%s/%s: brae wrote none' % (obj, f))
                    continue
                hb, rb = rows(pb)
                if hb != ho:
                    k = next((i for i, (x, y) in enumerate(zip(hb, ho)) if x != y), min(len(hb), len(ho)))
                    bad.append('%s/%s header line %d: brae %r OpenFOAM %r' % (obj, f, k, hb[k] if k < len(hb) else None, ho[k] if k < len(ho) else None))
                a, b = rb.get(t), ro.get(t)
                if a is None or b is None or len(a) != len(b):
                    bad.append('%s/%s row %s: brae %s OpenFOAM %s' % (obj, f, t, a, b))
                    continue
                scale = max(abs(y) for y in b) or 1.0
                e = max(abs(x - y) for x, y in zip(a, b)) / scale
                worst = max(worst, e)
                n += len(a)
                if e > bound:
                    bad.append('%s/%s row %s: %.2e of the row scale' % (obj, f, t, e))
    return worst, n, bad


def times(br):
    out = {}
    pp = br + '/postProcessing'
    if not os.path.isdir(pp):
        return out
    for obj in sorted(os.listdir(pp)):
        for d in sorted(os.listdir(pp + '/' + obj), key=float):
            for f in sorted(os.listdir(pp + '/%s/%s' % (obj, d))):
                out['%s/%s/%s' % (obj, d, f)] = ' '.join(sorted(rows(pp + '/%s/%s/%s' % (obj, d, f))[1], key=float))
    return out


what = sys.argv[1]
if what == 'stage':
    src, dst, case, end, write, kind = sys.argv[2:8]
    funcs = objects(case) if kind == 'objects' else (lifecycle() if kind == 'lifecycle' else (refusals() if kind == 'refusals' else ''))
    stage(src, dst, case, int(end), int(write), funcs)
elif what == 'compare':
    worst, n, bad = compare(sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]))
    print('%.3e %d %s' % (worst, n, ' | '.join(bad[:3])))
elif what == 'times':
    a, b = times(sys.argv[2]), times(sys.argv[3])
    keys = sorted(set(a) | set(b))
    for k in keys:
        print('%-40s brae [%s] OpenFOAM [%s] %s' % (k, a.get(k, '-'), b.get(k, '-'), 'same' if a.get(k) == b.get(k) else 'DIFFER'))
PY
FO="python3 -I -B $W/fo.py"
BOUND=1e-12
run_brae()   # run_brae <dir> [env...]
{
    local d=$1
    shift
    ( cd "$d" && env BRAE_SIMPLE_CASE_SOLVER=1 "$@" "$BRAE" -case . > log 2>&1; echo $? > rc )
}
oracle()   # oracle <brae dir> <OpenFOAM dir> <time>: OpenFOAM's functionObjects on brae's fields at <time>
{
    rm -rf "$2"
    cp -r "$1" "$2"
    rm -rf "$2/postProcessing"
    ( cd "$2" && simpleFoam -postProcess -time "$3" > log.post 2>&1; echo $? > rc.post )
}
echo "== the same fields: brae's objects every step, OpenFOAM's on brae's written fields at 3"
for c in sst ke cav
do
    case $c in
        sst) src=$ROOT/validation/pitzDailySST ;;
        ke) src=$ROOT/validation/pitzDaily ;;
        cav) src=$ROOT/validation/cav3d_cf ;;
    esac
    $FO stage "$src" "$W/$c" $c 3 3 objects
    run_brae "$W/$c"
    [ "$(cat "$W/$c/rc")" = 0 ] || { say "$c: brae ran" FAIL; tail -n 3 "$W/$c/log"; continue; }
    oracle "$W/$c" "$W/$c-of" 3
    [ "$(cat "$W/$c-of/rc.post")" = 0 ] || { say "$c: OpenFOAM's postProcess ran" FAIL; grep -m3 -A3 "FOAM FATAL" "$W/$c-of/log.post"; continue; }
    read -r worst n rest <<< "$($FO compare "$W/$c" "$W/$c-of" 3 $BOUND)"
    if [ -z "$rest" ] && [ "$n" -gt 0 ]
    then
        say "$c: forces/forceCoeffs at 3, $n columns of 4 objects' files: worst $worst of the row scale (bound $BOUND), headers OpenFOAM's" ok
    else
        say "$c: $n columns, worst $worst: $rest" FAIL
    fi
    # premise: brae ran the objects every step (rows 1 2 3), and OpenFOAM's postProcess wrote all six files
    nfiles=$(find "$W/$c-of/postProcessing" -type f | wc -l)
    [ "$nfiles" = 6 ] && say "$c: premise: OpenFOAM wrote the 6 files (coefficient x2, force and moment x2)" ok || say "$c: premise: OpenFOAM wrote $nfiles files, not 6" FAIL
    r=$(grep -v '^#' "$W/$c/postProcessing/fcA/0/coefficient.dat" | awk '{printf "%s ", $1}')
    [ "$r" = "1 2 3 " ] && say "$c: premise: brae's fcA rows at 1 2 3 (every step)" ok || say "$c: premise: brae's fcA rows at [$r]" FAIL
done
echo "== controls (must be red): sst"
for ctl in BRAE_CONTROL_FORCES_GRAD_CELL BRAE_CONTROL_FORCES_GLOBAL_AXES
do
    rm -rf "$W/ctl"
    cp -r "$W/sst" "$W/ctl"
    rm -rf "$W/ctl/postProcessing" "$W/ctl/1" "$W/ctl/2" "$W/ctl/3"
    run_brae "$W/ctl" "$ctl=1"
    grep -q "CONTROL $ctl" "$W/ctl/log" || say "control $ctl: injected (its line in the log)" FAIL
    read -r worst n rest <<< "$($FO compare "$W/ctl" "$W/sst-of" 3 $BOUND)"
    [ -n "$rest" ] && say "control $ctl: red, worst $worst of the row scale" ok || say "control $ctl: NOT red (worst $worst): the gate cannot see it" FAIL
done
echo "== lifecycle: brae's rows at OpenFOAM's times (sst, 4 iterations, fields written at 3)"
$FO stage "$ROOT/validation/pitzDailySST" "$W/life" sst 4 3 lifecycle
cp -r "$W/life" "$W/life-of"
run_brae "$W/life"
( cd "$W/life-of" && simpleFoam > log 2>&1; echo $? > rc )
if [ "$(cat "$W/life/rc")" = 0 ] && [ "$(cat "$W/life-of/rc")" = 0 ]
then
    $FO times "$W/life" "$W/life-of" > "$W/life.times"
    sed 's/^/     /' "$W/life.times"
    grep -q DIFFER "$W/life.times" && say "lifecycle: every object's files and rows where OpenFOAM's are" FAIL \
        || say "lifecycle: every object's files and rows where OpenFOAM's are ($(wc -l < "$W/life.times") files)" ok
    grep -q "Loff" "$W/life.times" && say "lifecycle: enabled false wrote a file" FAIL || say "lifecycle: enabled false: no file, in either code" ok
else
    say "lifecycle: brae exit $(cat "$W/life/rc"), OpenFOAM exit $(cat "$W/life-of/rc")" FAIL
fi
echo "== restart: 2 iterations, then on to 4 from latestTime (sst)"
$FO stage "$ROOT/validation/pitzDailySST" "$W/rs" sst 2 1 objects
cp -r "$W/rs" "$W/rs-of"
run_brae "$W/rs"
( cd "$W/rs-of" && simpleFoam > log 2>&1 )
for d in rs rs-of
do
    sed -i -E 's/^(endTime\s+)\S+;/\14;/' "$W/$d/system/controlDict"
done
run_brae "$W/rs"
( cd "$W/rs-of" && simpleFoam > log2 2>&1 )
$FO times "$W/rs" "$W/rs-of" | grep "fcA/" > "$W/rs.times"
sed 's/^/     /' "$W/rs.times"
if ! grep -q DIFFER "$W/rs.times" && grep -q "fcA/2/coefficient.dat *brae \[3 4\]" "$W/rs.times" && grep -q "fcA/0/coefficient.dat *brae \[1 2\]" "$W/rs.times"
then
    say "restart: postProcessing/fcA/0 holds 1 2 and fcA/2 holds 3 4, as OpenFOAM's restart lays them out" ok
else
    say "restart: brae's layout is not OpenFOAM's" FAIL
fi
echo "== refusals and unread entries (sst)"
$FO stage "$ROOT/validation/pitzDailySST" "$W/ref" sst 1 1 refusals
run_brae "$W/ref"
[ "$(cat "$W/ref/rc")" = 0 ] && say "refusals: the run goes on past the objects it does not run (OpenFOAM continues past one it cannot construct)" ok \
    || say "refusals: the run stopped (exit $(cat "$W/ref/rc"))" FAIL
for pair in "Rdfd:directForceDensity" "Rrho:Could not find rho" "Rwf:writeFields" "Rcoef:unknown coefficient type Cq" "Rcyl:coordinateSystem type cylindrical"
do
    n=${pair%%:*}
    why=${pair#*:}
    if grep -F "functions/$n" "$W/ref/log" | grep -F "NOT run" | grep -qF "$why" && [ ! -d "$W/ref/postProcessing/$n" ]
    then
        say "refusal $n: named NOT run ($why), no file" ok
    else
        say "refusal $n: not named NOT run for '$why', or a file was written" FAIL
    fi
done
grep -F "functions/Ubin/binData" "$W/ref/log" | grep -qF "binField" && say "unread: binData said to be unread (v2412 bins in binField)" ok || say "unread: binData not said" FAIL
grep -F "functions/Upitch/pitchAxis" "$W/ref/log" | grep -qF "IS NOT the axis" && say "unread: a pitchAxis that is not e2 said to differ from OpenFOAM's axis" ok \
    || say "unread: pitchAxis not said" FAIL
[ $fail = 0 ] && echo "PASS" || echo "FAIL"
exit $fail
