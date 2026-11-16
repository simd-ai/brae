#!/usr/bin/env bash
# U STOPS ON ITS OWN fvSolution ENTRY, minIter INCLUDED, in the legacy simpleFoam driver (steady and PIMPLE).
#
# OpenFOAM's smoothSolver enters its loop when `minIter_ > 0 || !converged` and goes on while
# `(n < maxIter && !converged) || n < minIter` (smoothSolver.C:159-208; maxIter 1000 and minIter 0 when the entry names
# none, lduMatrixSolver.C:204-205): with `minIter 1` it sweeps U once in every solve, even when the initial residual is
# already under the tolerance. brae's sweep in OpenFOAM's order took a literal 5000 for maxIter and 0 for minIter,
# whatever the U entry said, and skipped such a solve; the Krylov branch and the colour rule took the entry's own. The
# same call on the multi-GPU driver grew a mode at the tyre contact patches of a 123.4 M-cell DrivAer DDES (|U| 1,397
# m/s at step 20, 683,000 m/s at step 40, where OpenFOAM's run stayed flat). OpenFOAM's pimpleFoam LES tutorials
# (NACA4412, wallMountedHump, decayIsoTurb) name minIter on U, and brae's pimpleFoam runs this driver.
#
# THE ORACLE is real serial OpenFOAM (v2412) on the same staged case: its log's `No Iterations` for every Ux and Uy
# solve, and its U at the end. Every row starts from OpenFOAM's own iteration DEV of pitzDailySST (written at 17
# digits), stages p (GAMG) and k and omega (smoothSolver / GaussSeidel) at 1e-12 relTol 0, and U as smoothSolver /
# GaussSeidel / nSweeps 1 -- the legacy driver's own sweep count -- with the entries below:
#
#   row        run                    U                                      UFinal
#   u-min      simpleFoam, N its      tolerance 10, minIter 1
#   u-cap      simpleFoam, N its      tolerance 1e-15, maxIter 7
#   u-minmax   simpleFoam, N its      tolerance 10, minIter 5, maxIter 2
#   p-min      pimpleFoam, NP steps   tolerance 10, minIter 1                tolerance 10, minIter 2
#
# At tolerance 10 every solve starts converged and minIter alone sets the count; at 1e-15 maxIter alone. u-minmax is
# OpenFOAM's precedence: minIter outranks maxIter (5 sweeps, not 2). These counts are set by the entries, not by
# residuals, so they hold in any cell order: the colour rule's arm must give them too.
#
# CHECKS
#   COUNT    the steady rows, brae in OpenFOAM's order (BRAE_SIMPLE_CASE_SOLVER=1) and under the colour rule: every Ux and
#            Uy solve's sweep count is OpenFOAM's, solve for solve. OpenFOAM's order has three stop tests of its own, each
#            held: the host smoother (the default), the device's graph (BRAE_GS_HOST_SMOOTHER=0) and the device's
#            host-driven loop (BRAE_GS_HOST_SMOOTHER=0 BRAE_GS_HOST_LOOP=1). Not the PIMPLE row: brae_pimpleFoam prints
#            no line a solve, so it is held by its field alone, in the three arms of OpenFOAM's order (the transient
#            driver has no colour rule).
#   FIELD    brae in OpenFOAM's order, on the minIter rows: U at the end within the row's bound of OpenFOAM's (relative L2
#            over the cells), where the CONTROL is not. MEASURED (Arrhenius job 5012676, GH200, OpenFOAM v2412 in its
#            container): u-min 1.09e-06 against the control's 4.44e-02, and main b6953a86's binary (the defect itself)
#            4.44e-02: UBOUND 1e-05. p-min over its 20 steps (job 5013937): 2.53e-07 in each of the three arms against
#            the control's 1.03e-05: UBOUNDP 2e-06.
#   CONTROL  BRAE_CONTROL_U_FIXED_CAP=1, the sweep in OpenFOAM's order at 5000 sweeps and no minIter as before
#            2026-11-08: the COUNT check must fail on every steady row, and the FIELD check on the minIter rows.
#   FAIL-PROOFS (job 5012676) main b6953a86's binary: u-min 0 sweeps against OpenFOAM's 1, u-cap 160 against 7, u-minmax 0
#            against 5 in all three of OpenFOAM's-order arms. The candidate with main's device_amg_gauss_seidel.cu (maxIter
#            outranking minIter): u-minmax 2 sweeps against 5 in the three, every other check passed.
#
# What this does NOT claim: the colour rule's U field (another cell order, another iterate after a fixed number of
# sweeps), the Krylov branch (it took maxIter and minIter already), and pcorr's minIter (read by the same reader:
# tests/test_linear_solver_setup.cu).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
V="$ROOT/validation"
FX=pitzDailySST
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
N=5                      # the steady rows' iterations
NP=${NP:-20}             # the PIMPLE row's steps
DEV=100                  # OpenFOAM's iteration every row starts from
DT=1e-5                  # the PIMPLE row's step: Courant ~0.2 at pitzDaily's inlet
UBOUND=${UBOUND:-1e-05}  # the steady FIELD check's bound, relative L2 of U (MEASURED above)
UBOUNDP=${UBOUNDP:-2e-06}  # the PIMPLE row's (MEASURED above)
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
[ -d "$V/$FX" ] || { echo "SKIP: fixture $FX missing"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
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
    printf '  %-78s %s\n' "$1" "$2"
    [ "$2" = FAIL ] && fail=1
    return 0
}

# develop: the fixture as shipped to OpenFOAM's iteration DEV, ascii at 17 digits, in $W/dev
develop()
{
    local d=$W/dev
    [ -d "$d/$DEV" ] && return 0
    mkdir -p "$d"
    cp -r "$V/$FX/constant" "$V/$FX/system" "$V/$FX/0" "$d/"
    python3 - "$d/system/controlDict" "$DEV" <<'PY'
import re
import sys

c, dev = sys.argv[1], sys.argv[2]
s = open(c).read()
s = re.sub(r'functions\s*\{.*\n\}', 'functions\n{\n}', s, flags=re.S)
for k, v in (('startFrom', 'startTime'), ('endTime', dev), ('writeInterval', dev), ('writeControl', 'timeStep'),
             ('writeFormat', 'ascii'), ('writePrecision', '17'), ('writeCompression', 'off')):
    s, m = re.subn(r'(?m)^%s\s+\S+;' % k, '%s %s;' % (k.ljust(15), v), s)
    if m == 0:
        s = s.rstrip() + '\n%s %s;\n' % (k.ljust(15), v)
open(c, 'w').write(s)
PY
    ( cd "$d" && simpleFoam > log 2>&1 ) && [ -d "$d/$DEV" ]
}

# stage <dir> <simple|pimple> <U spec> [<UFinal spec>]: from OpenFOAM's iteration DEV, N iterations (steady) or N
# steps of DT from time 0 (PIMPLE: DEV's fields as the start)
stage()
{
    local d=$1 run=$2
    mkdir -p "$d"
    cp -r "$W/dev/constant" "$W/dev/system" "$d/"
    if [ "$run" = pimple ]
    then
        cp -r "$W/dev/$DEV" "$d/0"
        rm -rf "$d/0/uniform"
    else
        cp -r "$W/dev/$DEV" "$d/"
    fi
    local n=$N
    [ "$run" = pimple ] && n=$NP
    python3 - "$d" "$n" "$DEV" "$DT" "$run" "${3:-}" "${4:-}" <<'PY'
import re
import sys

d, n, dev, dt, run, uspec, ufspec = sys.argv[1:8]
n = int(n)
c = d + '/system/controlDict'
s = open(c).read()
s = re.sub(r'functions\s*\{.*\n\}', 'functions\n{\n}', s, flags=re.S)
if run == 'pimple':
    keys = (('application', 'pimpleFoam'), ('startFrom', 'startTime'), ('startTime', '0'),
            ('endTime', repr(n * float(dt))), ('deltaT', dt), ('writeControl', 'timeStep'), ('writeInterval', str(n)),
            ('writeFormat', 'ascii'), ('writePrecision', '15'), ('writeCompression', 'off'), ('adjustTimeStep', 'no'))
else:
    keys = (('startFrom', 'latestTime'), ('endTime', str(int(dev) + n)), ('writeInterval', str(n)),
            ('writeControl', 'timeStep'), ('writeFormat', 'ascii'), ('writePrecision', '15'), ('writeCompression', 'off'))
for k, v in keys:
    s, m = re.subn(r'(?m)^%s\s+\S+;' % k, '%s %s;' % (k.ljust(15), v), s)
    if m == 0:
        s = s.rstrip() + '\n%s %s;\n' % (k.ljust(15), v)
open(c, 'w').write(s)


def block(keys):
    body = ['        solver          smoothSolver;', '        smoother        GaussSeidel;', '        nSweeps         1;']
    body += ['        %s %s;' % (k.ljust(15), v) for k, v in keys]
    return '    {\n' + '\n'.join(body) + '\n    }'


def spec(t):
    return [tuple(kv.split(':', 1)) for kv in t.split(',')]


tight = [('tolerance', '1e-12'), ('relTol', '0')]
p = '        solver          GAMG;\n        smoother        GaussSeidel;\n' \
    '        tolerance       1e-12;\n        relTol          0;\n'
out = ['    p\n    {\n' + p + '    }', '    U\n' + block(spec(uspec))]
out += ['    %s\n%s' % (f, block(tight)) for f in ('k', 'omega')]
if run == 'pimple':
    out += ['    pFinal\n    {\n' + p + '    }', '    UFinal\n' + block(spec(ufspec or uspec))]
    out += ['    %sFinal\n%s' % (f, block(tight)) for f in ('k', 'omega')]
f = d + '/system/fvSolution'
s = open(f).read()
i = s.index('\nsolvers')
j = s.index('\n}', s.index('{', i))
s = s[:i] + '\nsolvers\n{\n' + '\n\n'.join(out) + s[j:]
s, m = re.subn(r'residualControl\s*\{[^}]*\}', 'residualControl\n    {\n    }', s)
if run == 'pimple':
    # no under-relaxation: the transient's outer correctors converge on their own at this step, and neither code
    # then has a Final relaxation entry to resolve
    s = re.sub(r'\nrelaxationFactors\s*\{.*\n\}', '', s, flags=re.S)
    s += '\nPIMPLE\n{\n    nOuterCorrectors 2;\n    nCorrectors 2;\n    nNonOrthogonalCorrectors 0;\n}\n'
open(f, 'w').write(s)
if run == 'pimple':
    f = d + '/system/fvSchemes'
    s = open(f).read()
    s = re.sub(r'ddtSchemes\s*\{[^}]*\}', 'ddtSchemes\n{\n    default         Euler;\n}', s, count=1)
    open(f, 'w').write(s)
PY
}

# counts <log>: one line a U component solve, "<field> <initial> <final residual> <sweeps>", in the log's order
counts()
{
    grep -E "Solving for (Ux|Uy|Uz)," "$1" \
        | sed -E 's/.*Solving for ([A-Za-z]+), Initial residual = ([^,]+), Final residual = ([^,]+), No Iterations ([0-9]+).*/\1 \2 \3 \4/'
}

# same <oracle counts> <brae counts>: "ok" when the solves are the same fields in the same order and every sweep
# count is the oracle's; else the first that is not
same()
{
    python3 - "$1" "$2" <<'PY'
import sys

a = [l.split() for l in open(sys.argv[1]) if l.strip()]
b = [l.split() for l in open(sys.argv[2]) if l.strip()]
if not a or len(a) != len(b):
    print('%d solves against the oracle\'s %d' % (len(b), len(a)))
    sys.exit(1)
for i, (x, y) in enumerate(zip(a, b)):
    if x[0] != y[0]:
        print('solve %d is %s, the oracle\'s %s' % (i + 1, y[0], x[0]))
        sys.exit(1)
    if x[3] != y[3]:
        print('solve %d (%s): %s sweeps, the oracle %s' % (i + 1, y[0], y[3], x[3]))
        sys.exit(1)
print('OpenFOAM\'s counts, %d solves (%s sweeps)' % (len(a), '/'.join(sorted(set(x[3] for x in a)))))
PY
}

# udiff <oracle U> <brae U>: the relative L2 of brae's internal U against OpenFOAM's, over the cells
udiff()
{
    python3 - "$1" "$2" <<'PY'
import math
import re
import sys


def cells(f):
    s = open(f).read()
    m = re.search(r'internalField\s+nonuniform\s+List<vector>\s*\n?(\d+)\s*\n?\(', s)
    if not m:
        sys.exit('%s holds no ascii nonuniform internalField' % f)
    n = int(m.group(1))
    v = re.findall(r'\(([^()]*)\)', s[m.end():])[:n]
    if len(v) != n:
        sys.exit('%s lists %d of %d values' % (f, len(v), n))
    return [tuple(float(t) for t in x.split()) for x in v]


a = cells(sys.argv[1])
b = cells(sys.argv[2])
if len(a) != len(b):
    sys.exit('%d cells against the oracle\'s %d' % (len(b), len(a)))
num = sum(sum((p - q) ** 2 for p, q in zip(x, y)) for x, y in zip(a, b))
den = sum(sum(p * p for p in x) for x in a)
print('%.3e' % math.sqrt(num / den))
PY
}

# within <value> <bound>: true when value < bound
within()
{
    python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) < float(sys.argv[2]) else 1)' "$1" "$2"
}

# brae <row dir> <arm> [VAR=value ...]
brae()
{
    local d=$1 a=$2
    shift 2
    mkdir -p "$d/$a"
    cp -r "$d/of/constant" "$d/of/system" "$d/$a/"
    if [ -d "$d/of/0" ]
    then
        cp -r "$d/of/0" "$d/$a/"
    fi
    if [ -d "$d/of/$DEV" ]
    then
        cp -r "$d/of/$DEV" "$d/$a/"
    fi
    ( cd "$d/$a" && env -u BRAE_SIMPLE_CASE_SOLVER -u BRAE_CONTROL_U_FIXED_CAP -u BRAE_GS_HOST_SMOOTHER -u BRAE_GS_HOST_LOOP \
        "$@" "$BRAE" -case . > log 2> err )
    echo $? > "$d/$a/rc"
    counts "$d/$a/log" > "$d/$a.counts"
}

# endtime <case dir>: the last time directory written
endtime()
{
    ( cd "$1" && ls -d [0-9]* 2> /dev/null | sort -g | tail -1 )
}

ROWS=(
    "u-min simple tolerance:10,relTol:0,minIter:1"
    "u-cap simple tolerance:1e-15,relTol:0,maxIter:7"
    "u-minmax simple tolerance:10,relTol:0,minIter:5,maxIter:2"
    "p-min pimple tolerance:10,relTol:0,minIter:1 tolerance:10,relTol:0,minIter:2"
)

develop || { say "PREMISE: OpenFOAM ran $FX to its iteration $DEV" FAIL; echo FAIL; exit 1; }
for r in "${ROWS[@]}"
do
    read -r row run uspec ufspec <<< "$r"
    d="$W/$row"
    stage "$d/of" "$run" "$uspec" "${ufspec:-}"
    app=simpleFoam
    [ "$run" = pimple ] && app=pimpleFoam
    ( cd "$d/of" && $app > log 2>&1 )
    [ $? = 0 ] || { say "PREMISE $row: OpenFOAM ($app) ran" FAIL; continue; }
    counts "$d/of/log" > "$d/of.counts"
    [ "$(wc -l < "$d/of.counts")" -ge "$N" ] || { say "PREMISE $row: OpenFOAM's U solves" FAIL; continue; }
    if [ "$run" = pimple ]
    then
        # what makes the row see UFinal: OpenFOAM sweeps 1 (U's minIter) and 2 (UFinal's) and nothing else
        [ "$(awk '{print $4}' "$d/of.counts" | sort -u | tr '\n' ' ')" = "1 2 " ] \
            && say "PREMISE $row: OpenFOAM sweeps U 1 and UFinal 2 times (their minIter)" ok \
            || say "PREMISE $row: OpenFOAM sweeps U 1 and UFinal 2 times (their minIter)" FAIL
    fi
    minrow=0
    [[ "$row" == *-min ]] && minrow=1
    arms="case graph loop colour"
    [ "$run" = pimple ] && arms="case graph loop"
    brae "$d" case BRAE_SIMPLE_CASE_SOLVER=1
    brae "$d" graph BRAE_SIMPLE_CASE_SOLVER=1 BRAE_GS_HOST_SMOOTHER=0
    brae "$d" loop BRAE_SIMPLE_CASE_SOLVER=1 BRAE_GS_HOST_SMOOTHER=0 BRAE_GS_HOST_LOOP=1
    [ "$run" = pimple ] || brae "$d" colour
    brae "$d" ctl BRAE_SIMPLE_CASE_SOLVER=1 BRAE_CONTROL_U_FIXED_CAP=1
    for a in $arms ctl
    do
        [ "$(cat "$d/$a/rc")" = 0 ] && say "PREMISE $row: brae arm $a ran" ok || say "PREMISE $row: brae arm $a ran" FAIL
    done
    grep -q "swept in colour order on the GPU" "$d/case/log" \
        && say "PREMISE $row: CASE runs OpenFOAM's order" FAIL || say "PREMISE $row: CASE runs OpenFOAM's order" ok
    if [ "$run" != pimple ]
    then
        grep -q "U.*swept in colour order on the GPU\|swept in colour order on the GPU.*U" "$d/colour/log" \
            && say "PREMISE $row: COLOUR announces U in colour order" ok \
            || say "PREMISE $row: COLOUR announces U in colour order" FAIL
        for a in $arms
        do
            m=$(same "$d/of.counts" "$d/$a.counts")
            [ $? = 0 ] && say "COUNT $row $a: $m" ok || say "COUNT $row $a: $m" FAIL
        done
        m=$(same "$d/of.counts" "$d/ctl.counts")
        [ $? != 0 ] && say "CONTROL $row: the fixed cap fails the COUNT check: $m" ok \
            || say "CONTROL $row: the fixed cap fails the COUNT check: $m" FAIL
    fi

    if [ "$minrow" = 1 ]
    then
        bound=$UBOUND
        fa="case"
        if [ "$run" = pimple ]
        then
            bound=$UBOUNDP
            fa="case graph loop"
        fi
        t=$(endtime "$d/of")
        uk=$(udiff "$d/of/$t/U" "$d/ctl/$t/U")
        for a in $fa
        do
            uc=$(udiff "$d/of/$t/U" "$d/$a/$t/U")
            if [ -z "$bound" ]
            then
                say "FIELD $row $a: U at $t, $uc, control $uk (no bound: measuring)" FAIL
            else
                within "$uc" "$bound" && say "FIELD $row $a: U at $t within $bound of OpenFOAM's: $uc" ok \
                    || say "FIELD $row $a: U at $t within $bound of OpenFOAM's: $uc" FAIL
            fi
        done
        if [ -n "$bound" ]
        then
            within "$uk" "$bound" && say "CONTROL $row: the fixed cap fails the FIELD check: $uk" FAIL \
                || say "CONTROL $row: the fixed cap fails the FIELD check: $uk" ok
        fi
    fi
done

[ $fail -eq 0 ] && echo PASS || echo FAIL
exit $fail
