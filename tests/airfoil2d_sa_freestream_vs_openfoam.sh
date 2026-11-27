#!/usr/bin/env bash
# SpalartAllmaras with freestream boundaries on incompressible/simpleFoam/airFoil2D, simpleFoam and pimpleFoam, iteration by
# iteration against OpenFOAM v2412.
#
# WHAT IT HOLDS. SpalartAllmarasBase::correct (v2412): the nuTilda equation, solve, bound(nuTilda_, 0) (:487) and then
# nuTilda_.correctBoundaryConditions() (:488), so a freestream outflow face takes its bounded cell back; correctNut, whose
# nut_.correctBoundaryConditions() evaluates the freestream nut patches (valueFraction = neg(phi)). The SpalartAllmaras
# constructor bounds nothing (kEpsilon's and kOmegaSST's do: kEpsilon.C:182-183, kOmegaSSTBase.C:438-439), so a start with
# nuTilda at or below 0 is solved from as read.
#
# THE CASE. The tutorial as shipped (SpalartAllmaras; U freestreamVelocity, p freestreamPressure, nuTilda and nut freestream on
# inlet and outlet; walls nutUSpaldingWallFunction), its mesh polyMesh.orig. OpenFOAM simpleFoam runs it to iteration 50 with
# the tutorial's own settings; that write is the start of every arm. Its nuTilda holds cells at exactly 0 (114), which is what
# the bound arms need. From there every linear solve is tight (tolerance 1e-12, relTol 0, maxIter 5000) and every step is
# written at writePrecision 17.
#   AS  simpleFoam 50 -> 53, phi read from OpenFOAM's write, p's relaxation factor 1 (U and nuTilda keep 0.7). Under the
#       tutorial's 0.3 OpenFOAM writes p's freestreamPressure patches as p.relax() left them (operator==, a relaxed value)
#       and brae evaluates them from the relaxed cells: 1.2e-01 on p's outlet entry, every other entry the same as with 1
#       (Arrhenius 5846805). That is p's boundary under relaxation, recorded as its own defect, not this gate's subject
#   AP  pimpleFoam from the same fields (moved to time 0, phi dropped so that both codes start from flux(U)), Euler,
#       deltaT 1e-5, 3 steps, nOuterCorrectors 2, nCorrectors 2, no relaxation
#   AN  AP with the start's nuTilda cells at exactly 0 set to -1e-8: solved from as read
# CHECKS, each entry (the internal field and every patch's written value) on its own scale (tests/pimple_entry_compare.py),
# at the three written times; phi on its own line:
#   AS      p U nuTilda nut and phi within BOUND_SIMPLE
#   AP, AN  U nuTilda nut and phi within BOUND. p is printed, not held: it is 1.9e-06 to 2.3e-06 from OpenFOAM, where
#           OpenFOAM's own PCG against its GAMG is 9.6e-11, an absolute ~0.2 that stays the same with 1 or 3 pressure
#           correctors, 1 or 2 outer, the phi file, laminar in place of SpalartAllmaras and p at 1e-15, and is ten times
#           larger relative to a ten times larger deltaT (Arrhenius 5847146). It is not SpalartAllmaras' and not the
#           pressure solver's: the freestream U and p conditions under pimpleFoam, an open defect with its own unit
# Controls, each red on the entry it guards:
#   BRAE_CONTROL_NUT_PATCH_TYPES_IGNORED=1   AS: nut's freestream outflow faces at the start file's value
#   BRAE_CONTROL_SA_CTOR_BOUND=1             AP and AN: bound(nuTilda, 0) at construction, which lifts the start's zero
#                                            cells (and AN's negative ones) to their neighbours' average
#   BRAE_CONTROL_SA_BOUND_CAP_HELD=1         AP: an outflow face held at the bound's floor until the next evaluate
# MEASURED (Arrhenius, one GH200, OpenFOAM v2412 serial, 2026-11-27):
#   before (develop's tree with the nut change, 0089de39; jobs 5845485, 5846077): AP nuTilda 5.8e-04 in a cell OpenFOAM keeps at
#     2.3e-09 (the start's zero cells lifted at construction), nuTilda on the outlet 8.8e-05 (a face held at the bound's floor);
#     AS nut's freestream outlet 1.2e-01 before the nut change
#   after (job 5846974, on 0ada6b26): AS 4.5e-10 (nuTilda), phi 1.2e-12; AP and AN nuTilda 8.5e-13, nut 3.1e-10, U 3.0e-09,
#     phi 4.1e-09 (p 2.3e-06, above). So BOUND_SIMPLE 5e-9 and BOUND 5e-8, a decade above. Controls: 1.2e-01, 5.8e-04,
#     1.9e-04, 8.8e-05. OpenFOAM's PCG against its GAMG on AP: 9.6e-11. About 20 s.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
PIMPLE="${BRAE_PIMPLE_BIN:-$ROOT/build/brae_pimpleFoam}"
OFBASHRC=${OF_BASHRC:-${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}}
CMP="${BRAE_CMP:-$ROOT/tests/pimple_entry_compare.py}"
BOUND=${BRAE_AF_BOUND:-5e-8}
BOUND_SIMPLE=${BRAE_AF_BOUND_SIMPLE:-5e-9}
[ "$BOUND" = report ] && BOUND_SIMPLE=report
[ -x "$BRAE" ] || { echo "SKIP: $BRAE not built"; exit 77; }
[ -x "$PIMPLE" ] || { echo "SKIP: $PIMPLE not built"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: no OpenFOAM v2412 at $OFBASHRC"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
set --
set +u   # the bashrc reads unset variables
# shellcheck disable=SC1090
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
TUT="${BRAE_OF_TUTORIALS:-${FOAM_TUTORIALS:-}}/incompressible/simpleFoam/airFoil2D"
[ -d "$TUT/system" ] || { echo "SKIP: no simpleFoam/airFoil2D tutorial at $TUT"; exit 77; }
command -v simpleFoam > /dev/null 2>&1 || { echo "SKIP: simpleFoam not on PATH"; exit 77; }
W=${BRAE_AF_KEEP:-$(mktemp -d)}
[ -n "${BRAE_AF_KEEP:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"
fail=0
say() { printf '  %-86s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
FIELDS="p U nuTilda nut"

# ctl <case> <key> <value> ...: controlDict entries set (added when absent), functions emptied
ctl()
{
    python3 - "$@" <<'EOF'
import re, sys
c = sys.argv[1] + '/system/controlDict'
kv = sys.argv[2:]
s = open(c).read()
for k, v in zip(kv[0::2], kv[1::2]):
    s, n = re.subn(r'(?m)^%s\s+[^;]*;' % re.escape(k), '%-16s%s;' % (k + ' ', v), s)
    if n == 0:
        s = s.rstrip() + '\n%-16s%s;\n' % (k + ' ', v)
i = s.find('\nfunctions')
if i >= 0:
    j = s.index('{', i)
    d = 0
    for e in range(j, len(s)):
        d += {'{': 1, '}': -1}.get(s[e], 0)
        if d == 0:
            break
    s = s[:i] + '\nfunctions {}' + s[e + 1:]
open(c, 'w').write(s)
EOF
}
tight()
{
    python3 - "$1/system/fvSolution" <<'EOF'
import re, sys
p = sys.argv[1]
s = open(p).read()
i = s.index('solvers')
j = re.search(r'\n(SIMPLE|PIMPLE)', s).start()
blk = re.sub(r'(\btolerance\s+)[^;]+;', r'\g<1>1e-12;', s[i:j])
blk = re.sub(r'(\brelTol\s+)[^;]+;', r'\g<1>0;', blk)
blk = re.sub(r'(\bsolver\s+[^;]+;)', r'\g<1>\n        maxIter         5000;', blk)
s = s[:i] + blk + s[j:]
s = re.sub(r'residualControl\s*\{[^}]*\}', '', s)
open(p, 'w').write(s)
EOF
}
mkdir -p "$W/seed"
cp -r "$TUT/constant" "$TUT/system" "$W/seed/"
cp -r "$TUT/0.orig" "$W/seed/0"
cp -r "$W/seed/constant/polyMesh.orig" "$W/seed/constant/polyMesh"
grep -q 'type            freestream;' "$W/seed/0/nut" && grep -q nutUSpaldingWallFunction "$W/seed/0/nut" \
    && grep -q 'SpalartAllmaras' "$W/seed/constant/turbulenceProperties" \
    || { echo "FAIL: the tutorial's nut patches or model moved -- the gate's premise moved"; exit 1; }
ctl "$W/seed" endTime 50 writeInterval 50 writePrecision 17 writeFormat ascii writeCompression off
( cd "$W/seed" && simpleFoam > log 2>&1 ) || { echo "FAIL: OpenFOAM's seed run"; tail -5 "$W/seed/log"; exit 1; }
[ -d "$W/seed/50" ] || { echo "FAIL: the seed run wrote no time 50"; exit 1; }
rm -rf "$W/seed/50/uniform"

ofrun() { ( cd "$1" && nice -n 10 "$2" > log 2>&1 ); }
judge()   # <arm> <times> [<bound> [<held fields>]]
{
    local a=$1 t=$2 m mp
    local BOUND=${3:-$BOUND}
    local FIELDS=${4:-$FIELDS}
    python3 "$CMP" "$W/$a-of" "$W/$a-br" "$t" $FIELDS > "$W/$a.cmp"
    sed 's/^/      /' "$W/$a.cmp"
    m=$(awk '/^MAX/{print $2, $3, $4, $5}' "$W/$a.cmp")
    python3 "$CMP" "$W/$a-of" "$W/$a-br" "$t" phi > "$W/$a.phi"
    mp=$(awk '/^MAX/{print $2}' "$W/$a.phi")
    if [ "$BOUND" = report ]; then
        say "$a: worst entry of $FIELDS at $t: $m; phi $mp (report only)" INFO
        return
    fi
    python3 -c "import sys; sys.exit(0 if float('${m%% *}') <= float('$BOUND') else 1)" \
        && say "$a: every entry of $FIELDS at $t within $BOUND (worst $m)" PASS \
        || say "$a: every entry of $FIELDS at $t within $BOUND (worst $m)" FAIL
    python3 -c "import sys; sys.exit(0 if float('$mp') <= float('$BOUND') else 1)" \
        && say "$a: phi, internal faces and every patch, at $t within $BOUND (worst $mp)" PASS \
        || say "$a: phi, internal faces and every patch, at $t within $BOUND (worst $mp)" FAIL
}

# AS: simpleFoam 50 -> 53
mkdir -p "$W/AS"
cp -r "$W/seed/constant" "$W/seed/system" "$W/AS/"
cp -r "$W/seed/50" "$W/AS/50"
ctl "$W/AS" startFrom latestTime endTime 53 writeInterval 1
tight "$W/AS"
python3 - "$W/AS/system/fvSolution" <<'EOF'
import re, sys
p = sys.argv[1]
s = open(p).read()
s, n = re.subn(r'(fields\s*\{\s*p\s+)0\.3;', r'\g<1>1;', s)
assert n == 1, 'the tutorial no longer relaxes p by 0.3'
open(p, 'w').write(s)
EOF
cp -r "$W/AS" "$W/AS-of"
cp -r "$W/AS" "$W/AS-br"
ofrun "$W/AS-of" simpleFoam || { echo "FAIL: OpenFOAM AS"; tail -5 "$W/AS-of/log"; exit 1; }
if ( cd "$W/AS-br" && BRAE_SIMPLE_CASE_SOLVER=1 "$BRAE" -case . > log 2>&1 ); then
    judge AS 51,52,53 "$BOUND_SIMPLE"
else
    say "AS: brae ran ($(grep -a -m1 -i -E 'refus|error|unsupported' "$W/AS-br/log" | cut -c1-120))" FAIL
fi

# AP: pimpleFoam from the same fields at time 0
mkdir -p "$W/AP"
cp -r "$W/seed/constant" "$W/seed/system" "$W/AP/"
cp -r "$W/seed/50" "$W/AP/0"
rm -f "$W/AP/0/phi"
ctl "$W/AP" application pimpleFoam startFrom latestTime startTime 0 deltaT 1e-05 endTime 3e-05 writeControl timeStep \
    writeInterval 1 adjustTimeStep no
python3 - "$W/AP" <<'EOF'
import re, sys
c = sys.argv[1]
fs = c + '/system/fvSchemes'
s = open(fs).read()
s, n = re.subn(r'(ddtSchemes\s*\{\s*default\s+)steadyState;', r'\1Euler;', s)
assert n == 1
open(fs, 'w').write(s)
fv = c + '/system/fvSolution'
s = open(fv).read()
s, n = re.subn(r'\nSIMPLE\s*\{.*?\n\}', '\nPIMPLE\n{\n    nOuterCorrectors 2;\n    nCorrectors     2;\n    nNonOrthogonalCorrectors 0;\n}',
               s, count=1, flags=re.S)
assert n == 1
s, n = re.subn(r'\nrelaxationFactors\s*\{.*?\n\}', '', s, count=1, flags=re.S)
assert n == 1
i = s.index('solvers')
j = s.index('{', i)
d = 0
for e in range(j, len(s)):
    d += {'{': 1, '}': -1}.get(s[e], 0)
    if d == 0:
        break
s = s[:e] + '\n    "(p|U|nuTilda)Final"\n    {\n        $p;\n    }\n' + s[e:]
open(fv, 'w').write(s)
EOF
# The Final entries: each takes its own field's entry
python3 - "$W/AP/system/fvSolution" <<'EOF'
import re, sys
p = sys.argv[1]
s = open(p).read()
s = s.replace('    "(p|U|nuTilda)Final"\n    {\n        $p;\n    }\n',
              '    pFinal\n    {\n        $p;\n    }\n    UFinal\n    {\n        $U;\n    }\n    nuTildaFinal\n    {\n        $nuTilda;\n    }\n')
open(p, 'w').write(s)
EOF
tight "$W/AP"
cp -r "$W/AP" "$W/AP-of"
cp -r "$W/AP" "$W/AP-br"
ofrun "$W/AP-of" pimpleFoam || { echo "FAIL: OpenFOAM AP"; tail -5 "$W/AP-of/log"; exit 1; }
# OpenFOAM against itself with another pressure solver at the same tolerances: the floor to which two converged solves of
# this stretched mesh's p agree (brae's pressure solve is its AMG-PCG). Reported, not held
cp -r "$W/AP" "$W/AP-ofpcg"
python3 - "$W/AP-ofpcg/system/fvSolution" <<'EOF'
import re, sys
p = sys.argv[1]
s = open(p).read()
s, n = re.subn(r'(\n    p\n    \{[^}]*?solver\s+)GAMG;', r'\g<1>PCG;\n        preconditioner  DIC;', s, count=1)
assert n == 1, n
open(p, 'w').write(s)
EOF
ofrun "$W/AP-ofpcg" pimpleFoam || { echo "FAIL: OpenFOAM AP with PCG"; tail -5 "$W/AP-ofpcg/log"; exit 1; }
say "AP: OpenFOAM PCG against OpenFOAM GAMG, worst entry: $(python3 "$CMP" "$W/AP-of" "$W/AP-ofpcg" 1e-05,2e-05,3e-05 $FIELDS | awk '/^MAX/{print $2, $3, $4, $5}')" INFO
if ( cd "$W/AP-br" && "$PIMPLE" -case . > log 2>&1 ); then
    judge AP 1e-05,2e-05,3e-05 "$BOUND" "U nuTilda nut"
    say "AP: p, not held (see the header): worst entry $(python3 "$CMP" "$W/AP-of" "$W/AP-br" 1e-05,2e-05,3e-05 p | awk '/^MAX/{print $2, $3, $5}')" INFO
else
    say "AP: brae ran ($(grep -a -m1 -i -E 'refus|error|unsupported' "$W/AP-br/log" | cut -c1-120))" FAIL
fi
# AN: AP from a start whose zero nuTilda cells are slightly negative
mkdir -p "$W/AN"
cp -r "$W/AP/constant" "$W/AP/system" "$W/AP/0" "$W/AN/"
NZ=$(python3 - "$W/AN/0/nuTilda" <<'EOF2'
import re, sys
p = sys.argv[1]
t = open(p).read()
m = re.search(r'internalField\s+nonuniform\s+List<scalar>\s*\n?(\d+)\s*\n?\(', t)
e = t.index(')', m.end())
vals = t[m.end():e].split()
z = sum(1 for x in vals if float(x) == 0.0)
t = t[:m.end()] + '\n' + '\n'.join('-1e-08' if float(x) == 0.0 else x for x in vals) + '\n' + t[e:]
open(p, 'w').write(t)
print(z)
EOF2
)
[ "${NZ:-0}" -gt 0 ] && say "premise: the start holds $NZ nuTilda cells at exactly 0 (AP's subject; AN sets them to -1e-8)" PASS \
    || say "premise: the start holds nuTilda cells at exactly 0 (found ${NZ:-none})" FAIL
cp -r "$W/AN" "$W/AN-of"
cp -r "$W/AN" "$W/AN-br"
ofrun "$W/AN-of" pimpleFoam || { echo "FAIL: OpenFOAM AN"; tail -5 "$W/AN-of/log"; exit 1; }
if ( cd "$W/AN-br" && "$PIMPLE" -case . > log 2>&1 ); then
    judge AN 1e-05,2e-05,3e-05 "$BOUND" "U nuTilda nut"
    say "AN: p, not held (see the header): worst entry $(python3 "$CMP" "$W/AN-of" "$W/AN-br" 1e-05,2e-05,3e-05 p | awk '/^MAX/{print $2, $3, $5}')" INFO
else
    say "AN: brae ran ($(grep -a -m1 -i -E 'refus|error|unsupported' "$W/AN-br/log" | cut -c1-120))" FAIL
fi

# The control must make ITS entry exceed the bound (`report` uses 1e-6 here). AP starts from OpenFOAM's write, whose
# freestream faces hold iteration 50's cells, which three steps of 1e-5 barely move: the control is held on AS
CB=$BOUND
[ "$CB" = report ] && CB=1e-6
if [ -d "$W/AS-of" ]; then
    rm -rf "$W/AS-ctl"
    cp -r "$W/AS" "$W/AS-ctl"
    ( cd "$W/AS-ctl" && BRAE_SIMPLE_CASE_SOLVER=1 BRAE_CONTROL_NUT_PATCH_TYPES_IGNORED=1 "$BRAE" -case . > log 2>&1 ) \
        || say "control BRAE_CONTROL_NUT_PATCH_TYPES_IGNORED=1: brae ran on AS" FAIL
    r=$(python3 "$CMP" "$W/AS-of" "$W/AS-ctl" 51 nut | awk '$3 == "outlet" {print $10}')
    python3 -c "import sys; sys.exit(0 if '$r' and float('$r') > float('$CB') else 1)" \
        && say "control BRAE_CONTROL_NUT_PATCH_TYPES_IGNORED=1: AS nut outlet at 51 goes red (rel $r)" PASS \
        || say "control BRAE_CONTROL_NUT_PATCH_TYPES_IGNORED=1: AS nut outlet at 51 goes red (rel ${r:-none})" FAIL
fi
ctlp()   # <env> <arm> <time> <field> <entry>: a pimpleFoam arm rerun by brae under a control
{
    local env=$1 arm=$2 t=$3 f=$4 entry=$5 name=${1%%=*}
    [ -d "$W/$arm-of" ] || return 0
    rm -rf "$W/$arm-$name"
    cp -r "$W/$arm" "$W/$arm-$name"
    ( cd "$W/$arm-$name" && env "$env" "$PIMPLE" -case . > log 2>&1 ) || { say "control $env: brae ran on $arm" FAIL; return; }
    local r
    r=$(python3 "$CMP" "$W/$arm-of" "$W/$arm-$name" "$t" "$f" | awk -v e="$entry" '$3 == e {print $10}')
    python3 -c "import sys; sys.exit(0 if '$r' and float('$r') > float('$CB') else 1)" \
        && say "control $env: $arm $f $entry at $t goes red (rel $r)" PASS \
        || say "control $env: $arm $f $entry at $t goes red (rel ${r:-none})" FAIL
}
ctlp BRAE_CONTROL_SA_CTOR_BOUND=1 AP 1e-05 nuTilda internal
ctlp BRAE_CONTROL_SA_CTOR_BOUND=1 AN 1e-05 nuTilda internal
ctlp BRAE_CONTROL_SA_BOUND_CAP_HELD=1 AP 1e-05 nuTilda outlet
[ $fail = 0 ] && echo "PASS" || echo "FAIL"
exit $fail
