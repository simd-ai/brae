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
#       (Arrhenius 5846805): p's boundary under relaxation, held in AS3
#   AS3 AS at the tutorial's own p relaxation 0.3: p's freestreamPressure patch values written as p.relax() left them
#   AP  pimpleFoam from the same fields (moved to time 0, phi dropped so that both codes start from flux(U)), Euler,
#       deltaT 1e-5, 3 steps, nOuterCorrectors 2, nCorrectors 2, no relaxation
#   AN  AP with the start's nuTilda cells at exactly 0 set to -1e-8: solved from as read
#   AW  the continuations from brae's writes (AS's 52, AP's 2e-05): the freestream patches' key, OpenFOAM and brae continued
#       from them against OpenFOAM continued from its own (see the AW block for why that is the oracle)
#   AR  AS with p's inlet freestreamPressure given `inletValue` in place of `freestreamValue`: both codes refuse it
#   AO  AP with momentumPredictor no: HbyA_b on the freestream faces is U's stored patch values (constrainHbyA)
#   AM  AS with U's inlet and outlet a plain `mixed` at the start's freestream valueFraction, fixed, and p plain (inlet
#       zeroGradient, outlet fixedValue); AMP the same under pimpleFoam (as AP). See the AO/AM block. AMP also holds what
#       AP's staging does not name: no relaxation factor, so OpenFOAM's UEqn.relax() does not call relax(alpha)
#       (fvMatrix.C:1250-1263) and brae must not relax with 1 either -- U's inflow faces at valueFraction < 1 carry a
#       negative boundary coefficient, which relax(1) moves (fvMatrix.C:1150, 1211, 1238, 1245)
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
#   BRAE_CONTROL_P_WRITE_EVALUATED=1         AS3: p's freestream patch values written as the blend of the relaxed cells, as before
#   BRAE_CONTROL_FREESTREAM_HBYA_REBLEND=1   AO: HbyA_b re-blended from the cells with the assembly's valueFraction, as before
#   BRAE_CONTROL_MIXED_VF_FREESTREAM=1       AM, AMP: a plain mixed's valueFraction replaced by the freestream formula, as before
#   BRAE_CONTROL_RELAX_UNNAMED_AS_ONE=1      AMP: U relaxed with factor 1 though the case names none, as before
#   BRAE_CONTROL_INLET_SLOT_KEY_INLETVALUE=1 AW: the freestream patches written under `inletValue`, as before; OpenFOAM must
#                                            refuse to continue from it
# MEASURED (Arrhenius, one GH200, OpenFOAM v2412 serial, 2026-11-27):
#   before (develop's tree with the nut change, 0089de39; jobs 5845485, 5846077): AP nuTilda 5.8e-04 in a cell OpenFOAM keeps at
#     2.3e-09 (the start's zero cells lifted at construction), nuTilda on the outlet 8.8e-05 (a face held at the bound's floor);
#     AS nut's freestream outlet 1.2e-01 before the nut change
#   after (job 5846974, on 0ada6b26): AS 4.5e-10 (nuTilda), phi 1.2e-12; AP and AN nuTilda 8.5e-13, nut 3.1e-10, U 3.0e-09,
#     phi 4.1e-09 (p 2.3e-06, above). So BOUND_SIMPLE 5e-9 and BOUND 5e-8, a decade above. Controls: 1.2e-01, 5.8e-04,
#     1.9e-04, 8.8e-05. OpenFOAM's PCG against its GAMG on AP: 9.6e-11. About 20 s.
#   AW/AR on develop 8f73f26e (job 5853257): every freestream patch of p U nuTilda nut written under `inletValue`, OpenFOAM
#     refused to continue ("Required entry 'freestreamValue' missing"), brae ran the misnamed case. With the change: the key
#     right, OpenFOAM continued from brae's write 3.0e-10 (simpleFoam, 53) and 4.0e-09 (pimpleFoam) from OpenFOAM continued
#     from its own, brae continued from its own 4.5e-10 and 3.0e-09 (5853995); brae refuses AR by name; the key control red.
#   AS3 (job 5856213): before, p's outlet entry 1.24e-01 at 51 (the evaluated blend written); with the change 4.6e-10, every
#     entry; the control 1.24e-01.
#   AO/AM/AMP on develop b266cca9 (job 5856015): AO U 3.5e-07 (outlet) and p 2.7e-04, AMP U 5.6e-05 and p 5.5e-02, AM 2.3e-03 (p).
#     With the change: AO 3.7e-09 (phi 8.0e-09), AMP 4.2e-09, AM 4.0e-10; p printed: AO 2.6e-06 as AP's, AMP 2.1e-05, which
#     simpleFoam does not have (AM's p 7e-11). Controls 3.5e-07, 4.9e-05, 2.6e-05.
#   AMP's p (job 5860186): 2.1e-05 with U relaxed with 1, 4.2e-10 without, so it is held; the control 2.1e-05. The same
#     change on 5860090's staging at valueFraction 0: 2.9e-05 -> 3.5e-11. AP's and AO's p stay printed: with freestreamVelocity
#     and freestreamPressure 2.3e-06 remains, at the inlet at step 1.
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

# AS3: AS at the tutorial's own p relaxation 0.3. p.relax() is operator==(prevIter() + 0.3*(p - prevIter())), which relaxes p's
# patch values with its cells, and nothing evaluates p before the write: a freestreamPressure face writes that relaxed value,
# not a blend of the relaxed cells under the valueFraction
rm -rf "$W/AS3" "$W/AS3-of" "$W/AS3-br"
cp -r "$W/AS" "$W/AS3"
python3 - "$W/AS3/system/fvSolution" <<'EOF'
import re, sys
p = sys.argv[1]
s = open(p).read()
s, n = re.subn(r'(fields\s*\{\s*p\s+)1;', r'\g<1>0.3;', s)
assert n == 1, 'AS no longer sets p relaxation 1'
open(p, 'w').write(s)
EOF
cp -r "$W/AS3" "$W/AS3-of"
cp -r "$W/AS3" "$W/AS3-br"
ofrun "$W/AS3-of" simpleFoam || { echo "FAIL: OpenFOAM AS3"; tail -5 "$W/AS3-of/log"; exit 1; }
if ( cd "$W/AS3-br" && BRAE_SIMPLE_CASE_SOLVER=1 "$BRAE" -case . > log 2>&1 ); then
    judge AS3 51,52,53 "$BOUND_SIMPLE"
else
    say "AS3: brae ran ($(grep -a -m1 -i -E 'refus|error|unsupported' "$W/AS3-br/log" | cut -c1-120))" FAIL
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

# AO: AP with momentumPredictor no. constrainHbyA.C:58-69 sets HbyA_b = U.boundaryField() on a patch that is not assignable (mixed,
# so freestreamVelocity): the STORED patch values. With no predictor nothing evaluates U before the first corrector, so they are
# the previous step's evaluate (at step 1 the file's value), not the cells re-blended with the valueFraction the assembly rebuilt.
# AM, AMP: U's inlet and outlet a plain `mixed` (refValue the freestream value, refGradient 0, valueFraction FIXED at the start's
# freestream value, from the start's patch values and the face normals) and p plain (inlet zeroGradient, outlet fixedValue at its
# written value), under simpleFoam (AM, 50 -> 53) and pimpleFoam (AMP, as AP). mixedFvPatchField's updateCoeffs changes nothing: the
# valueFraction stays the dictionary's
retype()   # <file> <patch> <type>: the patch's type replaced, its freestreamValue/inletValue dropped, its value kept
{
    python3 - "$@" <<'EOF'
import re, sys
f, patch, t = sys.argv[1:4]
s = open(f).read()
i = s.index('boundaryField')
m = re.search(r'\n(\s*)%s\s*\n\s*\{' % re.escape(patch), s[i:])
a = i + m.end()
d, e = 1, a
while d:
    d += {'{': 1, '}': -1}.get(s[e], 0)
    e += 1
body = s[a:e - 1]
body = re.sub(r'\btype\s+\w+\s*;', 'type            %s;' % t, body, count=1)
body = re.sub(r'\n\s*(freestreamValue|inletValue)\s+[^;]*;', '', body)
open(f, 'w').write(s[:a] + body + s[e - 1:])
EOF
}
tomixed()   # <case> <time>: U's freestreamVelocity patches -> plain mixed, valueFraction fixed at the start's freestream value
{
    python3 - "$1" "$2" <<'EOF'
import gzip, os, re, sys
c, t = sys.argv[1:3]
def rd(path):
    return open(path).read() if os.path.exists(path) else gzip.open(path + '.gz', 'rt').read()
def flist(path, conv):
    s = rd(path)
    m = re.search(r'\n(\d+)\s*\n\(', s)
    n = int(m.group(1))
    body = s[m.end():]
    if conv == 'vec':
        return [tuple(float(x) for x in v.split()) for v in re.findall(r'\(([^()]*)\)', body)[:n]]
    return [[int(x) for x in v.split()] for v in re.findall(r'\d+\(([^()]*)\)', body)[:n]]
pts = flist(c + '/constant/polyMesh/points', 'vec')
faces = flist(c + '/constant/polyMesh/faces', 'face')
bnd = rd(c + '/constant/polyMesh/boundary')
def area(f):
    x0 = pts[f[0]]
    a = [0.0, 0.0, 0.0]
    for p, q in zip(f[1:-1], f[2:]):
        u = [pts[p][k] - x0[k] for k in range(3)]
        v = [pts[q][k] - x0[k] for k in range(3)]
        a[0] += 0.5 * (u[1] * v[2] - u[2] * v[1])
        a[1] += 0.5 * (u[2] * v[0] - u[0] * v[2])
        a[2] += 0.5 * (u[0] * v[1] - u[1] * v[0])
    return a
U = open(c + '/' + t + '/U').read()
for patch in ('inlet', 'outlet'):
    m = re.search(r'\n\s*%s\s*\n\s*\{[^}]*?nFaces\s+(\d+);\s*startFace\s+(\d+);' % patch, bnd)
    nf, sf = int(m.group(1)), int(m.group(2))
    i0 = U.index('boundaryField')
    pm = re.search(r'\n(\s*)%s\s*\n\s*\{' % patch, U[i0:])
    a = i0 + pm.end()
    d, e = 1, a
    while d:
        d += {'{': 1, '}': -1}.get(U[e], 0)
        e += 1
    body = U[a:e - 1]
    fv = re.search(r'freestreamValue\s+uniform\s+\(([^)]*)\)\s*;', body).group(1)
    vm = re.search(r'value\s+nonuniform\s+List<vector>\s*\n?(\d+)\s*\n?\(', body)
    vals = [tuple(float(x) for x in v.split()) for v in re.findall(r'\(([^()]*)\)', body[vm.end():])[:nf]]
    vfs = []
    for k in range(nf):
        A = area(faces[sf + k])
        mA = sum(x * x for x in A) ** 0.5
        up = vals[k]
        mu = sum(x * x for x in up) ** 0.5
        vfs.append(0.5 - 0.5 * sum(up[q] * A[q] / mA for q in range(3)) / mu)
    new = ('\n        type            mixed;\n        refValue        uniform (%s);\n        refGradient     uniform (0 0 0);\n'
           '        valueFraction   nonuniform List<scalar>\n%d\n(\n%s\n)\n;\n        value           nonuniform List<vector>\n%d\n(\n%s\n)\n;\n    '
           % (fv, nf, '\n'.join('%.17g' % x for x in vfs), nf, '\n'.join('(%s)' % ' '.join('%.17g' % y for y in v) for v in vals)))
    U = U[:a] + new + U[e - 1:]
open(c + '/' + t + '/U', 'w').write(U)
EOF
}
pimple_arm()   # <arm> <note> [<held fields>]: OpenFOAM and brae on a staged pimpleFoam arm, U nuTilda nut held unless named, p
{              # printed when not held
    local held=${3:-U nuTilda nut}
    cp -r "$W/$1" "$W/$1-of"
    cp -r "$W/$1" "$W/$1-br"
    ofrun "$W/$1-of" pimpleFoam || { echo "FAIL: OpenFOAM $1"; tail -5 "$W/$1-of/log"; exit 1; }
    if ( cd "$W/$1-br" && "$PIMPLE" -case . > log 2>&1 ); then
        judge "$1" 1e-05,2e-05,3e-05 "$BOUND" "$held"
        case " $held " in
            *" p "*) ;;
            *) say "$1: p, not held ($2): worst entry $(python3 "$CMP" "$W/$1-of" "$W/$1-br" 1e-05,2e-05,3e-05 p | awk '/^MAX/{print $2, $3, $5}')" INFO ;;
        esac
    else
        say "$1: brae ran ($(grep -a -m1 -i -E 'refus|error|unsupported' "$W/$1-br/log" | cut -c1-120))" FAIL
    fi
}
rm -rf "$W/AO" "$W/AO-of" "$W/AO-br"
cp -r "$W/AP" "$W/AO"
python3 - "$W/AO/system/fvSolution" <<'EOF'
import re, sys
p = sys.argv[1]
s = open(p).read()
s, n = re.subn(r'(PIMPLE\s*\{)', r'\1\n    momentumPredictor no;', s, count=1)
assert n == 1
open(p, 'w').write(s)
EOF
pimple_arm AO "the open pimpleFoam p gap, as AP's"
rm -rf "$W/AMP" "$W/AMP-of" "$W/AMP-br"
cp -r "$W/AP" "$W/AMP"
tomixed "$W/AMP" 0
retype "$W/AMP/0/p" inlet zeroGradient
retype "$W/AMP/0/p" outlet fixedValue
[ "$(grep -c -E 'type +mixed;' "$W/AMP/0/U")" = 2 ] && say "AMP: staged (U inlet and outlet plain mixed, p plain)" PASS \
    || say "AMP: staged (U inlet and outlet plain mixed, p plain)" FAIL
pimple_arm AMP "" "p U nuTilda nut"
rm -rf "$W/AM" "$W/AM-of" "$W/AM-br"
cp -r "$W/AS" "$W/AM"
tomixed "$W/AM" 50
retype "$W/AM/50/p" inlet zeroGradient
retype "$W/AM/50/p" outlet fixedValue
cp -r "$W/AM" "$W/AM-of"
cp -r "$W/AM" "$W/AM-br"
ofrun "$W/AM-of" simpleFoam || { echo "FAIL: OpenFOAM AM"; tail -5 "$W/AM-of/log"; exit 1; }
if ( cd "$W/AM-br" && BRAE_SIMPLE_CASE_SOLVER=1 "$BRAE" -case . > log 2>&1 ); then
    judge AM 51,52,53 "$BOUND_SIMPLE"
else
    say "AM: brae ran ($(grep -a -m1 -i -E 'refus|error|unsupported' "$W/AM-br/log" | cut -c1-120))" FAIL
fi

# AW: OpenFOAM starts from brae's own writes. freestream, freestreamPressure and freestreamVelocity write their reference value
# under `freestreamValue` (freestreamFvPatchField.C:202, freestreamPressureFvPatchScalarField.C:129,
# freestreamVelocityFvPatchVectorField.C:115) and their dictionary constructors require it. The oracle for a continuation is
# OpenFOAM continued from ITS OWN write, not its continuous run: its start-up turbulence->validate() -> correctNut() evaluates
# nutUSpaldingWallFunction once more from the file's wall nut (tolerance 0.01, maxIter 10), which the continuous run does not,
# so OpenFOAM restarted is 3.5e-03 from OpenFOAM continuous on nut's walls at 53 (pimpleFoam 4.5e-06; Arrhenius 5853995).
# Held: every freestream patch of brae's writes carries that key alone; OpenFOAM continued from brae's write against OpenFOAM
# continued from its own (the write's content), and brae continued from its own write against the same (brae's start-up):
# simpleFoam from 52 to 53 at BOUND_SIMPLE, pimpleFoam from 2e-05 to 3e-05 on U nuTilda nut and phi at BOUND (p printed, as
# AP's). AR: a freestreamPressure given `inletValue`, the key of another type -- OpenFOAM refuses it ("Required entry
# 'freestreamValue' missing"), and so must brae, by name, rather than read it as freestreamValue
fskeys()   # <time dir>: "ok" when each freestream-family patch of p U nuTilda nut (8 of them) carries freestreamValue alone
{
    python3 - "$1" <<'EOF'
import os, re, sys
d = sys.argv[1]
out, bad = 0, []
for f in ('p', 'U', 'nuTilda', 'nut'):
    s = open(os.path.join(d, f)).read()
    s = s[s.index('boundaryField'):]
    for m in re.finditer(r'\n\s*(\w+)\s*\n\s*\{([^{}]*)\}', s):
        b = m.group(2)
        if re.search(r'type\s+freestream\w*\s*;', b):
            out += 1
            k = [x for x in ('freestreamValue', 'inletValue', 'outletValue') if re.search(r'\b%s\b' % x, b)]
            if k != ['freestreamValue']:
                bad.append('%s:%s=%s' % (f, m.group(1), '+'.join(k) or 'none'))
print('ok' if out == 8 and not bad else 'patches %d, wrong: %s' % (out, ' '.join(bad)))
EOF
}
awcont()   # <application> <run> <time> <tag>: continued from that run's write; "ok" or "refused <why>"
{
    local d="$W/AW-$4"
    rm -rf "$d"
    mkdir -p "$d"
    cp -r "$2/constant" "$2/system" "$2/$3" "$d/"
    ctl "$d" startFrom latestTime
    case $1 in
        brae) ( cd "$d" && BRAE_SIMPLE_CASE_SOLVER=1 "$BRAE" -case . > log 2>&1 ) ;;
        braePimple) ( cd "$d" && "$PIMPLE" -case . > log 2>&1 ) ;;
        *) ofrun "$d" "$1" ;;
    esac
    if [ $? = 0 ]; then
        echo ok
    else
        echo "refused $(grep -a -m1 -E 'Required entry|brae:' "$d/log" | tr -s ' ' | cut -c1-100)"
    fi
}
awheld()   # <label> <oracle dir> <dir> <time> <bound> <fields>
{
    local m mp
    m=$(python3 "$CMP" "$2" "$3" "$4" $6 | awk '/^MAX/{print $2, $3, $4, $5}')
    mp=$(python3 "$CMP" "$2" "$3" "$4" phi | awk '/^MAX/{print $2}')
    python3 -c "import sys; sys.exit(0 if max(float('${m%% *}'), float('$mp')) <= float('$5') else 1)" 2> /dev/null \
        && say "AW: $1 within $5 (worst $m; phi $mp)" PASS \
        || say "AW: $1 within $5 (worst $m; phi $mp)" FAIL
}
if [ -d "$W/AS-br/52" ]; then
    k=$(fskeys "$W/AS-br/51")
    [ "$k" = ok ] && say "AW: brae's simpleFoam write (51) puts every freestream patch's value under freestreamValue" PASS \
        || say "AW: brae's simpleFoam write (51) puts every freestream patch's value under freestreamValue ($k)" FAIL
    o=$(awcont simpleFoam "$W/AS-of" 52 SO)
    r=$(awcont simpleFoam "$W/AS-br" 52 S)
    b=$(awcont brae "$W/AS-br" 52 SB)
    if [ "$o" != ok ]; then
        say "AW: OpenFOAM simpleFoam continued from its own 52 ($o)" FAIL
    else
        [ "$r" = ok ] && awheld "OpenFOAM simpleFoam continued from brae's 52 against from its own, at 53" "$W/AW-SO" "$W/AW-S" 53 "$BOUND_SIMPLE" "$FIELDS" \
            || say "AW: OpenFOAM simpleFoam continued from brae's 52 ($r)" FAIL
        [ "$b" = ok ] && awheld "brae continued from its own 52 against OpenFOAM from its own, at 53" "$W/AW-SO" "$W/AW-SB" 53 "$BOUND_SIMPLE" "$FIELDS" \
            || say "AW: brae continued from its own 52 ($b)" FAIL
    fi
fi
if [ -d "$W/AP-br/2e-05" ]; then
    k=$(fskeys "$W/AP-br/1e-05")
    [ "$k" = ok ] && say "AW: brae's pimpleFoam write (1e-05) puts every freestream patch's value under freestreamValue" PASS \
        || say "AW: brae's pimpleFoam write (1e-05) puts every freestream patch's value under freestreamValue ($k)" FAIL
    o=$(awcont pimpleFoam "$W/AP-of" 2e-05 PO)
    r=$(awcont pimpleFoam "$W/AP-br" 2e-05 P)
    b=$(awcont braePimple "$W/AP-br" 2e-05 PB)
    if [ "$o" != ok ]; then
        say "AW: OpenFOAM pimpleFoam continued from its own 2e-05 ($o)" FAIL
    else
        [ "$r" = ok ] && awheld "OpenFOAM pimpleFoam continued from brae's 2e-05 against from its own, U nuTilda nut at 3e-05" "$W/AW-PO" "$W/AW-P" 3e-05 "$BOUND" "U nuTilda nut" \
            || say "AW: OpenFOAM pimpleFoam continued from brae's 2e-05 ($r)" FAIL
        [ "$b" = ok ] && awheld "brae pimpleFoam continued from its own 2e-05 against OpenFOAM from its own, U nuTilda nut at 3e-05" "$W/AW-PO" "$W/AW-PB" 3e-05 "$BOUND" "U nuTilda nut" \
            || say "AW: brae pimpleFoam continued from its own 2e-05 ($b)" FAIL
        [ "$b" = ok ] && say "AW: p of brae's continuation there, not held (as AP's): $(python3 "$CMP" "$W/AW-PO" "$W/AW-PB" 3e-05 p | awk '/^MAX/{print $2, $3, $5}')" INFO
    fi
fi
rm -rf "$W/AR" "$W/AR-of" "$W/AR-br"
cp -r "$W/AS" "$W/AR"
python3 - "$W/AR/50/p" <<'EOF'
import re, sys
p = sys.argv[1]
s = open(p).read()
i = s.index('boundaryField')
s2, n = re.subn(r'freestreamValue(\s)', r'inletValue     \1', s[i:], count=1)
assert n == 1
open(p, 'w').write(s[:i] + s2)
EOF
cp -r "$W/AR" "$W/AR-of"
cp -r "$W/AR" "$W/AR-br"
if ofrun "$W/AR-of" simpleFoam; then
    say "AR premise: OpenFOAM refuses a freestreamPressure without freestreamValue (it ran)" FAIL
else
    grep -a -q "Required entry 'freestreamValue'" "$W/AR-of/log" \
        && say "AR premise: OpenFOAM refuses a freestreamPressure without freestreamValue" PASS \
        || say "AR premise: OpenFOAM refuses a freestreamPressure without freestreamValue ($(grep -a -m1 FATAL "$W/AR-of/log"))" FAIL
fi
if ( cd "$W/AR-br" && BRAE_SIMPLE_CASE_SOLVER=1 "$BRAE" -case . > log 2>&1 ); then
    say "AR: brae refuses a freestreamPressure given inletValue (it ran)" FAIL
else
    grep -a -q "reads from \`freestreamValue\`, but it gives \`inletValue\`" "$W/AR-br/log" \
        && say "AR: brae refuses a freestreamPressure given inletValue, by name" PASS \
        || say "AR: brae refuses a freestreamPressure given inletValue ($(grep -a -m1 -i 'brae:' "$W/AR-br/log" | cut -c1-110))" FAIL
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
if [ -d "$W/AS3-of" ]; then
    rm -rf "$W/AS3-ctl"
    cp -r "$W/AS3" "$W/AS3-ctl"
    ( cd "$W/AS3-ctl" && BRAE_SIMPLE_CASE_SOLVER=1 BRAE_CONTROL_P_WRITE_EVALUATED=1 "$BRAE" -case . > log 2>&1 ) \
        || say "control BRAE_CONTROL_P_WRITE_EVALUATED=1: brae ran on AS3" FAIL
    r=$(python3 "$CMP" "$W/AS3-of" "$W/AS3-ctl" 51 p | awk '$3 == "outlet" {print $10}')
    python3 -c "import sys; sys.exit(0 if '$r' and float('$r') > float('$CB') else 1)" \
        && say "control BRAE_CONTROL_P_WRITE_EVALUATED=1: AS3 p outlet at 51 goes red (rel $r)" PASS \
        || say "control BRAE_CONTROL_P_WRITE_EVALUATED=1: AS3 p outlet at 51 goes red (rel ${r:-none})" FAIL
fi
ctlp BRAE_CONTROL_FREESTREAM_HBYA_REBLEND=1 AO 3e-05 U outlet
ctlp BRAE_CONTROL_MIXED_VF_FREESTREAM=1 AMP 1e-05 phi outlet
ctlp BRAE_CONTROL_RELAX_UNNAMED_AS_ONE=1 AMP 3e-05 p internal
if [ -d "$W/AM-of" ]; then
    rm -rf "$W/AM-ctl"
    cp -r "$W/AM" "$W/AM-ctl"
    ( cd "$W/AM-ctl" && BRAE_SIMPLE_CASE_SOLVER=1 BRAE_CONTROL_MIXED_VF_FREESTREAM=1 "$BRAE" -case . > log 2>&1 ) \
        || say "control BRAE_CONTROL_MIXED_VF_FREESTREAM=1: brae ran on AM" FAIL
    r=$(python3 "$CMP" "$W/AM-of" "$W/AM-ctl" 51 U | awk '$3 == "inlet" {print $10}')
    python3 -c "import sys; sys.exit(0 if '$r' and float('$r') > float('$CB') else 1)" \
        && say "control BRAE_CONTROL_MIXED_VF_FREESTREAM=1: AM U inlet at 51 goes red (rel $r)" PASS \
        || say "control BRAE_CONTROL_MIXED_VF_FREESTREAM=1: AM U inlet at 51 goes red (rel ${r:-none})" FAIL
fi
# The key control: brae writes the freestream patches' value as `inletValue`, and OpenFOAM must refuse to start from it
if [ -d "$W/AS-of" ]; then
    rm -rf "$W/AS-keyctl"
    cp -r "$W/AS" "$W/AS-keyctl"
    ( cd "$W/AS-keyctl" && BRAE_SIMPLE_CASE_SOLVER=1 BRAE_CONTROL_INLET_SLOT_KEY_INLETVALUE=1 "$BRAE" -case . > log 2>&1 ) \
        || say "control BRAE_CONTROL_INLET_SLOT_KEY_INLETVALUE=1: brae ran on AS" FAIL
    r=$(awcont simpleFoam "$W/AS-keyctl" 52 keyctl)
    case "$r" in
        refused\ *) say "control BRAE_CONTROL_INLET_SLOT_KEY_INLETVALUE=1: OpenFOAM refuses brae's write (${r#refused })" PASS ;;
        *) say "control BRAE_CONTROL_INLET_SLOT_KEY_INLETVALUE=1: OpenFOAM refuses brae's write ($r)" FAIL ;;
    esac
fi
[ $fail = 0 ] && echo "PASS" || echo "FAIL"
exit $fail
