#!/usr/bin/env bash
# EACH TRANSPORTED TURBULENCE FIELD STOPS ON ITS OWN fvSolution ENTRY, in the legacy simpleFoam driver.
#
# OpenFOAM's lduMatrix::solver reads tolerance, relTol, maxIter and minIter from the dictionary of the field it
# solves (lduMatrixSolver.C:193-207; defaults 1e-6, 0, 1000, 0), and the models solve each equation on its own
# (kOmegaSSTBase.C:593/:618, kEpsilon.C:268/:288). brae solved k and epsilon/omega under ONE tolerance and relTol,
# the tighter of the two entries, and said nothing; its sweeps in OpenFOAM's order stopped at a literal 3000 with
# no minIter whatever the entry said, and its colour order took one maxIter and minIter for the pair. Staged at k
# 1e-13 beside omega 1e-15 on motorBike (14.1 M cells), k ran to that cap at every solve (2026-10-22).
#
# THE ORACLE is real serial OpenFOAM (simpleFoam, v2412) on the same staged case for five SIMPLE iterations: its
# log's `No Iterations` for every k, epsilon, omega and nuTilda solve. Every row stages p (GAMG) and U at 1e-12,
# relTol 0, and the transported fields as smoothSolver / GaussSeidel / nSweeps 1 -- the legacy driver's own sweep
# count -- with the entries below; each field's entry is its own dictionary.
#
# FROM OPENFOAM'S ITERATION DEV, NOT FROM REST. Each fixture runs once as shipped to OpenFOAM's iteration DEV (100),
# written at 17 digits, and every row starts there. From rest, k and omega start uniform under limitedLinear, where
# rounding in the gradient of a uniform field picks the limiter's branch: there OpenFOAM's own counts move with its
# own last bit. MEASURED (JUPITER job 2671225): from rest, OpenFOAM with k's or omega's start one ulp up took 1 to 4
# sweeps more or fewer than itself on solves of every SST tolerance and relTol row; from DEV its twin kept every count
# of all five SST rows, and so did brae. From rest the CASE check had passed on the GB10 (0 sweeps off, 2026-10-22)
# and failed on JUPITER and Arrhenius at the same commit (k 35 against 34, omega 19 against 20 and 15 against 16;
# job 2671173: 7a82d815, e95ed639, the 18-patch candidate, its device loop alike): after 5 iterations from rest the
# GB10's brae was k 3.6e-04 from OpenFOAM and JUPITER's 2.5e-02, OpenFOAM byte-identical on both. A PREMISE now
# asks it of every row: OpenFOAM with every internal value of the row's first field one ulp up at DEV keeps its own
# counts, or the row's counts are not a fact this oracle holds.
#
#   row         fixture        k (or nuTilda)                       epsilon / omega
#   sst-tol     pitzDailySST   tolerance 1e-6                       tolerance 1e-11             (relTol 0)
#   sst-tolrev  pitzDailySST   tolerance 1e-11                      tolerance 1e-6
#   sst-rel     pitzDailySST   relTol 0.1                           relTol 1e-4                 (tolerance 1e-15)
#   sst-cap     pitzDailySST   maxIter 7                            maxIter 13                  (tolerance 1e-15)
#   sst-min     pitzDailySST   minIter 3                            minIter 5                   (tolerance 10)
#   ke-tol      pitzDailyTurb  tolerance 1e-6                       tolerance 1e-11
#   ke-cap      pitzDailyTurb  maxIter 7                            maxIter 13
#   sa-cap      airFoil2D      nuTilda maxIter 7 (tolerance 1e-15)
#   sa-min      airFoil2D      nuTilda minIter 3 (tolerance 10)
#
# CHECKS
#   PREMISE  OpenFOAM's one-ulp twin at DEV gives OpenFOAM's counts, solve for solve.
#   CASE     brae in OpenFOAM's order (BRAE_SIMPLE_CASE_SOLVER=1): every solve's sweep count is OpenFOAM's, solve
#            for solve (MEASURED from DEV on JUPITER: 0 sweeps off on every SST row) -- but on ke-tol, which holds the
#            BAND below: brae's epsilon residual is not OpenFOAM's (iteration 1 starts at 0.165439 against
#            0.163500; brae writes epsilon's wall rows by setValues), so a solve to 1e-11 takes 28-31 sweeps
#            where OpenFOAM's takes 31-35, under 21d124d6 as under this change.
#   COLOUR   brae under the colour rule (its notice asserted): on the cap and minIter rows every count is
#            OpenFOAM's -- the cap and minIter set them, not the order. On the tolerance and relTol rows the order
#            moves the count, so each solve must stop on its OWN field's entry, the BAND: its final residual
#            (relTol row: final over initial) under the entry's figure and over FLOOR times it. One sweep
#            reduces the residual by far less than FLOOR; a solve run to the other field's figure ends three to
#            five decades under its own.
#   CONTROL  BRAE_CONTROL_TURB_PAIR_ONE_RULE=1, both fields under the merged figure as before: the CASE check and
#            the COLOUR check must each fail on every two-field row. BRAE_CONTROL_SCALAR_FIXED_CAP=1, OpenFOAM's
#            order at 3000 sweeps and no minIter as before: the CASE check must fail on every cap and minIter
#            row.
#
# What this does NOT claim: the fields' values (the rule decides where a solve stops; the solves themselves are
# gated elsewhere), the Final entries (a steady run never takes them; tests/test_linear_solver_setup.cu reads
# them), and the fields whose maxIter and minIter no driver hands down yet (kUnreadEntryMaxIter).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
V="$ROOT/validation"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
N=5
DEV=100                 # OpenFOAM's iteration every row starts from (FROM OPENFOAM'S ITERATION DEV)
SLACK=${SLACK:-0}       # sweeps a CASE count may differ from OpenFOAM's
FLOOR=${FLOOR:-1e-3}    # a BAND solve stops above FLOOR x its own entry's figure
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
for f in pitzDailySST pitzDailyTurb airFoil2D
do
    [ -d "$V/$f" ] || { echo "SKIP: fixture $f missing"; exit 77; }
done
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

# develop <fixture>: the fixture as shipped to OpenFOAM's iteration DEV, ascii at 17 digits, in $W/dev-<fixture>
develop()
{
    local fx=$1 d=$W/dev-$1
    [ -d "$d/$DEV" ] && return 0
    mkdir -p "$d"
    cp -r "$V/$fx/constant" "$V/$fx/system" "$V/$fx/0" "$d/"
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

# stage <fixture> <dir> <field>=<key>:<value>,... ...: from the fixture's iteration DEV, N iterations
stage()
{
    local fx=$1 d=$2
    shift 2
    mkdir -p "$d"
    cp -r "$W/dev-$fx/constant" "$W/dev-$fx/system" "$W/dev-$fx/$DEV" "$d/"
    python3 - "$d" "$N" "$DEV" "$@" <<'PY'
import re
import sys

d, n, dev = sys.argv[1], sys.argv[2], sys.argv[3]
sys.argv[3:] = sys.argv[4:]
c = d + '/system/controlDict'
s = open(c).read()
s = re.sub(r'functions\s*\{.*\n\}', 'functions\n{\n}', s, flags=re.S)
for k, v in (('startFrom', 'latestTime'), ('endTime', str(int(dev) + int(n))), ('writeInterval', n),
             ('writeControl', 'timeStep'), ('writeFormat', 'ascii'), ('writePrecision', '15'), ('writeCompression', 'off')):
    s, m = re.subn(r'(?m)^%s\s+\S+;' % k, '%s %s;' % (k.ljust(15), v), s)
    if m == 0:
        s = s.rstrip() + '\n%s %s;\n' % (k.ljust(15), v)
open(c, 'w').write(s)


def block(keys):
    body = ['        solver          smoothSolver;', '        smoother        GaussSeidel;', '        nSweeps         1;']
    body += ['        %s %s;' % (k.ljust(15), v) for k, v in keys]
    return '    {\n' + '\n'.join(body) + '\n    }'


p = '    p\n    {\n        solver          GAMG;\n        smoother        GaussSeidel;\n' \
    '        tolerance       1e-12;\n        relTol          0;\n    }'
out = [p, '    U\n' + block([('tolerance', '1e-12'), ('relTol', '0')])]
for spec in sys.argv[3:]:
    f, rest = spec.split('=', 1)
    out.append('    %s\n%s' % (f, block([tuple(kv.split(':', 1)) for kv in rest.split(',')])))
f = d + '/system/fvSolution'
s = open(f).read()
i = s.index('\nsolvers')
j = s.index('\n}', s.index('{', i))
s = s[:i] + '\nsolvers\n{\n' + '\n\n'.join(out) + s[j:]
s, m = re.subn(r'residualControl\s*\{[^}]*\}', 'residualControl\n    {\n    }', s)
open(f, 'w').write(s)
PY
}

# counts <log> <fields regex>: one line a solve, "<field> <initial> <final residual> <sweeps>", in the log's order
counts()
{
    grep -E "Solving for ($2)," "$1" \
        | sed -E 's/.*Solving for ([A-Za-z]+), Initial residual = ([^,]+), Final residual = ([^,]+), No Iterations ([0-9]+).*/\1 \2 \3 \4/'
}

# same <oracle counts> <brae counts> <slack>: "ok" when the solves are the same fields in the same order and every
# sweep count is within <slack> of the oracle's; else the first that is not
same()
{
    python3 - "$1" "$2" "$3" <<'PY'
import sys

a = [l.split() for l in open(sys.argv[1]) if l.strip()]
b = [l.split() for l in open(sys.argv[2]) if l.strip()]
slack = int(sys.argv[3])
if not a or len(a) != len(b):
    print('%d solves against the oracle\'s %d' % (len(b), len(a)))
    sys.exit(1)
worst = 0
for i, (x, y) in enumerate(zip(a, b)):
    if x[0] != y[0]:
        print('solve %d is %s, the oracle\'s %s' % (i + 1, y[0], x[0]))
        sys.exit(1)
    worst = max(worst, abs(int(x[3]) - int(y[3])))
    if abs(int(x[3]) - int(y[3])) > slack:
        print('solve %d (%s): %s sweeps, the oracle %s' % (i + 1, y[0], y[3], x[3]))
        sys.exit(1)
print('OpenFOAM\'s counts, %d solves, worst %d sweep(s) off' % (len(a), worst))
PY
}

# band <brae counts> <tol|rel> <field>=<figure> ...: "ok" when every solve of each field ends under its own entry's
# figure and over FLOOR times it -- its final residual (tol), or its final over its initial residual (rel)
band()
{
    python3 - "$FLOOR" "$@" <<'PY'
import sys

floor = float(sys.argv[1])
rows = [l.split() for l in open(sys.argv[2]) if l.strip()]
mode = sys.argv[3] if len(sys.argv) > 3 else ''
fig = dict((kv.split('=')[0], float(kv.split('=')[1])) for kv in sys.argv[4:])
if not rows or not fig or mode not in ('tol', 'rel'):
    print('no solves, or no figures to hold them to')
    sys.exit(1)
for f, t in fig.items():
    r = [float(x[2]) / (float(x[1]) if mode == 'rel' else 1.0) for x in rows if x[0] == f]
    if not r:
        print('no %s solve' % f)
        sys.exit(1)
    if max(r) >= t or min(r) <= floor * t:
        print('%s ends in [%.3e, %.3e], its %s %g' % (f, min(r), max(r), 'relTol' if mode == 'rel' else 'tolerance', t))
        sys.exit(1)
print('each on its own entry: ' + ', '.join('%s in (%g, %g)' % (f, floor * t, t) for f, t in fig.items()))
PY
}

ROWS=(
    "sst-tol pitzDailySST k|omega k=tolerance:1e-6,relTol:0 omega=tolerance:1e-11,relTol:0"
    "sst-tolrev pitzDailySST k|omega k=tolerance:1e-11,relTol:0 omega=tolerance:1e-6,relTol:0"
    "sst-rel pitzDailySST k|omega k=tolerance:1e-15,relTol:0.1 omega=tolerance:1e-15,relTol:1e-4"
    "sst-cap pitzDailySST k|omega k=tolerance:1e-15,relTol:0,maxIter:7 omega=tolerance:1e-15,relTol:0,maxIter:13"
    "sst-min pitzDailySST k|omega k=tolerance:10,relTol:0,minIter:3 omega=tolerance:10,relTol:0,minIter:5"
    "ke-tol pitzDailyTurb k|epsilon k=tolerance:1e-6,relTol:0 epsilon=tolerance:1e-11,relTol:0"
    "ke-cap pitzDailyTurb k|epsilon k=tolerance:1e-15,relTol:0,maxIter:7 epsilon=tolerance:1e-15,relTol:0,maxIter:13"
    "sa-cap airFoil2D nuTilda nuTilda=tolerance:1e-15,relTol:0,maxIter:7"
    "sa-min airFoil2D nuTilda nuTilda=tolerance:10,relTol:0,minIter:3"
)
# the BAND's figures, by row; CASEBAND: the rows whose CASE check is the band, not OpenFOAM's counts
declare -A BAND=(
    [sst-tol]="tol k=1e-6 omega=1e-11"
    [sst-tolrev]="tol k=1e-11 omega=1e-6"
    [sst-rel]="rel k=0.1 omega=1e-4"
    [ke-tol]="tol k=1e-6 epsilon=1e-11"
)
declare -A CASEBAND=(
    [ke-tol]=1
)

# casecheck <row dir> <arm>: the CASE check of one arm of the current row
casecheck()
{
    if [ -n "${CASEBAND[$row]:-}" ]
    then
        # shellcheck disable=SC2086
        band "$1/$2.counts" ${BAND[$row]}
    else
        same "$1/of.counts" "$1/$2.counts" "$SLACK"
    fi
}
# colourcheck <row dir> <arm>: the COLOUR check of one arm of the current row
colourcheck()
{
    if [ "$capmin" = 1 ]
    then
        same "$1/of.counts" "$1/$2.counts" 0
    else
        # shellcheck disable=SC2086
        band "$1/$2.counts" ${BAND[$row]:-}
    fi
}

# brae <row dir> <arm> [VAR=value ...]
brae()
{
    local d=$1 a=$2
    shift 2
    mkdir -p "$d/$a"
    cp -r "$d/of/constant" "$d/of/system" "$d/of/$DEV" "$d/$a/"
    ( cd "$d/$a" && env -u BRAE_SIMPLE_CASE_SOLVER -u BRAE_CONTROL_TURB_PAIR_ONE_RULE -u BRAE_CONTROL_SCALAR_FIXED_CAP \
        "$@" "$BRAE" -case . > log 2> err )
    echo $? > "$d/$a/rc"
    counts "$d/$a/log" "$fields" > "$d/$a.counts"
}

for r in "${ROWS[@]}"
do
    read -r row fx fields specs <<< "$r"
    d="$W/$row"
    develop "$fx" || { say "PREMISE $row: OpenFOAM ran $fx to its iteration $DEV" FAIL; continue; }
    # shellcheck disable=SC2086
    stage "$fx" "$d/of" $specs
    ( cd "$d/of" && simpleFoam > log 2>&1 )
    [ $? = 0 ] || { say "PREMISE $row: OpenFOAM ran" FAIL; continue; }
    counts "$d/of/log" "$fields" > "$d/of.counts"
    [ "$(wc -l < "$d/of.counts")" -ge "$N" ] || { say "PREMISE $row: OpenFOAM's solves of $fields" FAIL; continue; }
    # OpenFOAM against itself: every internal value of the row's first field one ulp up at DEV
    # shellcheck disable=SC2086
    stage "$fx" "$d/oft" $specs
    python3 - "$d/oft/$DEV/${fields%%|*}" <<'PY'
import math
import re
import sys

f = sys.argv[1]
s = open(f).read()
m = re.search(r'internalField\s+nonuniform\s+List<scalar>\s*\n?(\d+)\s*\n?\(', s)
if not m:
    sys.exit('twin: %s holds no ascii nonuniform internalField' % f)
e = s.index(')', m.end())
v = [float(x) for x in s[m.end():e].split()]
if len(v) != int(m.group(1)):
    sys.exit('twin: %s lists %d of %s values' % (f, len(v), m.group(1)))
open(f, 'w').write(s[:m.end()] + '\n' + '\n'.join(repr(math.nextafter(x, math.inf)) for x in v) + '\n' + s[e:])
PY
    ( cd "$d/oft" && simpleFoam > log 2>&1 ) && counts "$d/oft/log" "$fields" > "$d/oft.counts"
    m=$(same "$d/of.counts" "$d/oft.counts" 0)
    [ $? = 0 ] && say "PREMISE $row: OpenFOAM's one-ulp twin at $DEV: $m" ok \
        || say "PREMISE $row: OpenFOAM's one-ulp twin at $DEV: $m" FAIL
    two=0
    [[ "$fields" == *"|"* ]] && two=1
    capmin=0
    [[ "$row" == *-cap || "$row" == *-min ]] && capmin=1

    brae "$d" case BRAE_SIMPLE_CASE_SOLVER=1
    brae "$d" colour
    [ "$(cat "$d/case/rc")" = 0 ] && [ "$(cat "$d/colour/rc")" = 0 ] \
        && say "PREMISE $row: both brae arms ran" ok || say "PREMISE $row: both brae arms ran" FAIL
    grep -q "colour rule: .* swept in colour order on the GPU" "$d/colour/log" \
        && say "PREMISE $row: COLOUR announces the colour order" ok \
        || say "PREMISE $row: COLOUR announces the colour order" FAIL
    grep -q "swept in colour order on the GPU" "$d/case/log" \
        && say "PREMISE $row: CASE runs OpenFOAM's order" FAIL || say "PREMISE $row: CASE runs OpenFOAM's order" ok

    m=$(casecheck "$d" case)
    [ $? = 0 ] && say "CASE $row: $m" ok || say "CASE $row: $m" FAIL
    m=$(colourcheck "$d" colour)
    [ $? = 0 ] && say "COLOUR $row: $m" ok || say "COLOUR $row: $m" FAIL

    if [ "$two" = 1 ]
    then
        brae "$d" ctl-one-case BRAE_SIMPLE_CASE_SOLVER=1 BRAE_CONTROL_TURB_PAIR_ONE_RULE=1
        m=$(casecheck "$d" ctl-one-case)
        [ $? != 0 ] && say "CONTROL $row: one rule for the pair fails the CASE check: $m" ok \
            || say "CONTROL $row: one rule for the pair fails the CASE check: $m" FAIL
        brae "$d" ctl-one-colour BRAE_CONTROL_TURB_PAIR_ONE_RULE=1
        m=$(colourcheck "$d" ctl-one-colour)
        [ $? != 0 ] && say "CONTROL $row: ...and the COLOUR check: $m" ok \
            || say "CONTROL $row: ...and the COLOUR check: $m" FAIL
    fi
    if [ "$capmin" = 1 ]
    then
        brae "$d" ctl-cap BRAE_SIMPLE_CASE_SOLVER=1 BRAE_CONTROL_SCALAR_FIXED_CAP=1
        m=$(casecheck "$d" ctl-cap)
        [ $? != 0 ] && say "CONTROL $row: the fixed cap fails the CASE check: $m" ok \
            || say "CONTROL $row: the fixed cap fails the CASE check: $m" FAIL
    fi
done

[ $fail -eq 0 ] && echo PASS || echo FAIL
exit $fail
