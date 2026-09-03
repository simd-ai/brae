#!/usr/bin/env bash
# DISPATCH GATE for the rebuilt simpleFoam (BRAE_SIMPLEFOAM_V2).
#
# The rebuilt path covers a strict subset of what the existing solver runs, so the thing that has to be
# true is not "it produces the right answer on the cases it supports" -- the component tests cover that --
# but "it never runs a case it does not support". A path that quietly degrades is indistinguishable from a
# correct one in the output, and brae has already shipped a solver that ignored MRFProperties, converged,
# and said nothing.
#
# So this asserts, on real cases through the real binary:
#   1. off  -> nothing changes, the existing solver runs;
#   2. on + unsupported -> REFUSES, names the reason, exits non-zero;
#   3. on + supported   -> runs and writes;
#   4. a substitution brae makes on purpose (GAMG -> AMG-PCG) is ANNOUNCED, not hidden.
#
# usage: simplefoam_v2_dispatch.sh <pitzDailyCase>
set -u
SRC="${1:?case dir}"
BRAE="${BRAE_BIN:-$(cd "$(dirname "$0")/.." && pwd)/build/brae}"
[ -x "$BRAE" ] || { echo "SKIP: no brae binary at $BRAE"; exit 77; }
SRC="$(cd "$SRC" && pwd)"

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fails=0
ok()   { echo "  ok:   $1"; }
bad()  { echo "  FAIL: $1"; fails=$((fails+1)); }

mkcase() {   # mkcase <dir> ; copies SRC and strips time dirs
    cp -r "$SRC" "$1"
    find "$1" -mindepth 1 -maxdepth 1 -type d ! -name 0 ! -name constant ! -name system -exec rm -rf {} + 2>/dev/null
    rm -rf "$1/postProcessing"
    python3 - "$1/system/controlDict" <<'PY'
import re, sys
p = sys.argv[1]; s = open(p).read()
s = re.sub(r'^endTime\s+\S+;', 'endTime         3;', s, flags=re.M)
open(p, 'w').write(s)
PY
}

# Make a case the rebuilt path SUPPORTS: laminar, upwind, plain SIMPLE.
supported() {
    mkcase "$1"
    python3 - "$1" <<'PY'
import re, sys, os
d = sys.argv[1]
s = open(d + '/system/fvSchemes').read()
s = re.sub(r'div\(phi,U\)[^;]*;', 'div(phi,U)      Gauss upwind;', s)
# laplacianSchemes is deliberately LEFT ALONE at `Gauss linear corrected`, which is what pitzDaily ships
# and what OpenFOAM defaults to. It used to be rewritten to `orthogonal` here because the rebuilt
# fvm::laplacian was orthogonal only; both halves of the correction are now ported on both paths, so the
# supported case exercises the flag end to end instead of dodging it.
open(d + '/system/fvSchemes', 'w').write(s)
s = open(d + '/system/fvSolution').read()
s = re.sub(r'consistent\s+\S+;', 'consistent      no;', s)
open(d + '/system/fvSolution', 'w').write(s)
s = open(d + '/constant/turbulenceProperties').read()
s = re.sub(r'simulationType\s+\S+;', 'simulationType  laminar;', s)
open(d + '/constant/turbulenceProperties', 'w').write(s)
PY
}

echo "== 1. NOT selected: the existing solver runs =="
mkcase "$W/off"
( cd "$W/off" && "$BRAE" > log 2>&1 )
if [ -d "$W/off/3" ]; then ok "off: the case ran and wrote a time directory"; else bad "off: no output"; fi
if grep -q "simpleFoam v2" "$W/off/log"; then bad "off: the v2 path announced itself"; else ok "off: v2 stayed silent"; fi

echo "== 2. selected + supported: the rebuilt path runs =="
supported "$W/on"
( cd "$W/on" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" > log 2>&1 )
rc=$?
[ $rc -eq 0 ] && ok "supported: exit 0" || bad "supported: exit $rc"
[ -d "$W/on/3" ] && ok "supported: wrote a time directory" || bad "supported: no output"
grep -q "Time = 3" "$W/on/log" && ok "supported: ran the requested iterations" \
                                || bad "supported: did not reach the end time"

echo "== 3. the GAMG substitution is ANNOUNCED, not hidden =="
if grep -q "NOTICE (simpleFoam v2).*GAMG" "$W/on/log"; then
    ok "GAMG -> AMG-PCG substitution is stated"
else
    bad "GAMG substitution was silent"
fi

echo "== 4. selected + unsupported: REFUSES, with the reason =="
# Each of these is a component the rebuilt path does not implement, and each would otherwise produce a
# converged, plausible, wrong answer.
# The mirror of try_refusal: the mutation must be ACCEPTED, and brae must still SAY what it is doing.
# Accepting silently is its own defect -- the point of the envelope is that nothing is substituted
# without being named.
try_notice() {    # try_notice <label> <mutator-python> <expected-substring>
    local d="$W/note_$1"
    supported "$d"
    python3 - "$d" <<PY
import re, sys, os
d = sys.argv[1]
$2
PY
    ( cd "$d" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" > log 2>&1 )
    local rc=$?
    if [ $rc -ne 0 ]; then bad "$1: refused (exit $rc), but OpenFOAM accepts this"; sed -n '1,4p' "$d/log"; return; fi
    if grep -q "$3" "$d/log"; then ok "$1: accepted, and said so"; else
        bad "$1: accepted SILENTLY, without naming it ($3)"; sed -n '1,4p' "$d/log"; fi
}

try_refusal() {   # try_refusal <label> <mutator-python> <expected-substring>
    local d="$W/ref_$1"
    supported "$d"
    python3 - "$d" <<PY
import re, sys, os
d = sys.argv[1]
$2
PY
    ( cd "$d" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" > log 2>&1 )
    local rc=$?
    if [ $rc -eq 0 ]; then bad "$1: ran anyway (exit 0)"; return; fi
    if grep -q "$3" "$d/log"; then ok "$1: refused, and named it"; else
        bad "$1: refused but did not name the reason ($3)"; sed -n '1,4p' "$d/log"; fi
}

# MRF is IMPLEMENTED now (correctBoundaryVelocity, DDt, makeRelative), so declaring it is no longer a
# refusal. What IS still one -- and is the real hazard -- is a zone naming a cellZone the mesh does
# not carry: brae would find no cells, rotate nothing, and converge to a confidently wrong answer,
# which is exactly the failure mode that shipped once before on the compressible path.
try_refusal mrf_missing_zone \
  "open(d+'/constant/MRFProperties','w').write('MRF1\n{\n    cellZone nosuchzone;\n    active yes;\n    origin (0 0 0);\n    axis (0 0 1);\n    omega 10;\n}\n')" \
  "nosuchzone"

# ...while an MRFProperties that declares no zones at all is ACCEPTED, because OpenFOAM applies
# nothing for it either. Refusing here would block a case the reference solver runs.
try_notice mrf_empty \
  "open(d+'/constant/MRFProperties','w').write('// no zones\n')" \
  "Time ="

# An EMPTY fvOptions file is now correctly ACCEPTED -- the framework parses it, finds no options, and
# there is nothing to apply. What is refused is an option whose TYPE is not ported: the framework being
# present must not read as "fvOptions is supported", since ofscan counts 46 implementations and this port
# has one. `actuationDiskSource` is what turbineSiting asks for.
try_refusal fvoptions \
  "open(d+'/constant/fvOptions','w').write('disk1\\n{\\n    type actuationDiskSource;\\n    active true;\\n}\\n')" \
  "actuationDiskSource"

# ...and the porosity IS accepted, with an empty file accepted too.
try_supported() {   # try_supported <label> <python-mutator>
    local d="$W/ok_$1"; supported "$d"
    python3 -c "
import re,sys
d = sys.argv[1]
$2
" "$d"
    ( cd "$d" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" > log 2>&1 )
    if [ $? -eq 0 ]; then ok "$1: accepted"; else bad "$1: refused"; sed -n '1,5p' "$d/log"; fi
}
try_supported emptyfvoptions "open(d+'/constant/fvOptions','w').write('// no options\\n')"

# SIMPLEC is no longer a refusal -- it is implemented, and section 4d below asserts it RUNS. Neither is
# fixedFluxPressure: commit c2ff3e4 (2026-08-31) lifted the envelope's substring refusal of it once both
# halves were ported -- the factory builds the real FixedFluxPressurePatchField and pEqn.cu runs
# deviceConstrainPressure at pEqn.H:21 -- and gated the result against real OpenFOAM in
# validation/ffpi_vs_openfoam.sh. This block went on asserting the REFUSAL, so from that commit it
# demanded that brae reject a case it supports and validates, and it has been red ever since. Asserting
# the support instead is what the shipped behaviour actually is; the refusal that is STILL live on this
# path is the MRF combination, and it is asserted below rather than left untested.
try_supported fixedfluxp \
  "import re,glob;f=[x for x in glob.glob(d+'/0/p')][0];s=open(f).read();s=re.sub(r'(upperWall\s*\{[^}]*?type\s+)\w+;', r'\\1fixedFluxPressure;', s, count=1, flags=re.S);open(f,'w').write(s)"

# The refusal that IS still live on this path -- fixedFluxPressure together with MRF, because MRF.relative
# belongs inside constrainPressure (constrainPressure.C:70) and is not in the device kernel -- is NOT
# asserted here, and deliberately so. It was written and then removed: pitzDaily carries no cellZones, so
# any MRFProperties added to it refuses for the MISSING ZONE instead (the mrf_missing_zone case above),
# and the block passed on the wrong reason. It needs a fixture with a real rotating zone AND a
# fixedFluxPressure wall. Queue item 29.

# `bounded` is SUPPORTED: -fvm::Sp(fvc::div(phi),U) is implemented on both paths and matched to 2.9e-16.
# It used to be a refusal; assert it RUNS, since the term vanishes at convergence and a converged field
# comparison could not tell whether it was applied.
echo "== 4c. bounded div(phi,U) is supported =="
supported "$W/bnd"
python3 - "$W/bnd" <<'PYEOF'
import re, sys
d = sys.argv[1]
s = open(d + '/system/fvSchemes').read()
s = re.sub(r'div\(phi,U\)[^;]*;', 'div(phi,U)      bounded Gauss upwind;', s)
open(d + '/system/fvSchemes', 'w').write(s)
PYEOF
( cd "$W/bnd" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" > log 2>&1 )
brc=$?
if [ $brc -eq 0 ]; then ok "bounded: exit 0"; else bad "bounded: exit $brc"; sed -n '1,4p' "$W/bnd/log"; fi
grep -q "is bounded" "$W/bnd/log" && ok "bounded: the solver states it is applying the term" \
                                  || bad "bounded: applied silently or not at all"

# `limited <coeff>` is limitedSnGrad -- IMPLEMENTED now on both paths (tests/limitedsngrad_vs_openfoam.sh),
# so it must be accepted AND announced with its coefficient: running the uncapped correction under a
# capped name applies a larger correction than the case asked for, and it does not vanish at
# convergence. OpenFOAM writes the coefficient both ways -- `limited <k>` and `limited <scheme> <k>` --
# and both must resolve to the same k, so both forms are exercised.
try_notice limitedlaplacian \
    "import re,sys;p=sys.argv[1]+'/system/fvSchemes';s=open(p).read();open(p,'w').write(re.sub(r'laplacianSchemes\s*\{[^}]*\}','laplacianSchemes\n{\n    default         Gauss linear limited 0.33;\n}',s))" \
    "limited 0.33"

try_notice limitedlaplacian2 \
    "import re,sys;p=sys.argv[1]+'/system/fvSchemes';s=open(p).read();open(p,'w').write(re.sub(r'laplacianSchemes\s*\{[^}]*\}','laplacianSchemes\n{\n    default         Gauss linear limited corrected 0.33;\n}',s))" \
    "limited 0.33"

# ...while a coefficient OUTSIDE [0, 1] is not a limiter at all and is still refused by name.
try_refusal limitedlaplacianbad \
    "import re,sys;p=sys.argv[1]+'/system/fvSchemes';s=open(p).read();open(p,'w').write(re.sub(r'laplacianSchemes\s*\{[^}]*\}','laplacianSchemes\n{\n    default         Gauss linear limited 7;\n}',s))" \
    "limited"

# Five div(phi,U) schemes are implemented now (upwind, linearUpwind, limitedLinear, limitedLinearV,
# LUST) -- section 4e below asserts the three new ones RUN. `vanLeer` is one of the 73 OpenFOAM registers
# and this port does not, so it stands in as the refusal case.
try_refusal divscheme \
  "s=open(d+'/system/fvSchemes').read(); s=re.sub(r'div\(phi,U\)[^;]*;','div(phi,U)      bounded Gauss vanLeer;',s); open(d+'/system/fvSchemes','w').write(s)" \
  "vanLeer"

# linearUpwind's NAMED gradient. `cellLimited Gauss linear 1` is IMPLEMENTED now (both paths -- see
# tests/celllimited_vs_openfoam.sh), so it must be accepted AND announced: the correction is built
# from that gradient and does not vanish at convergence, so silently running the plain Gauss one
# under a limited name would be a different equation. Announcing it is what makes that checkable.
try_notice lugradcelllimited \
  "s=open(d+'/system/fvSchemes').read(); s=re.sub(r'div\(phi,U\)[^;]*;','div(phi,U)      bounded Gauss linearUpwind grad(U);',s); s=re.sub(r'gradSchemes\s*\{[^}]*\}','gradSchemes\n{\n    default         Gauss linear;\n    grad(U)         cellLimited Gauss linear 1;\n}',s); open(d+'/system/fvSchemes','w').write(s)" \
  "cellLimited Gauss linear 1"

# ...while a named gradient brae genuinely does NOT compute is still refused BY NAME. leastSquares is
# a different stencil altogether, not a limiter on the Gauss one.
try_refusal lugradscheme \
  "s=open(d+'/system/fvSchemes').read(); s=re.sub(r'div\(phi,U\)[^;]*;','div(phi,U)      bounded Gauss linearUpwind grad(U);',s); s=re.sub(r'gradSchemes\s*\{[^}]*\}','gradSchemes\n{\n    default         Gauss linear;\n    grad(U)         leastSquares;\n}',s); open(d+'/system/fvSchemes','w').write(s)" \
  "leastSquares"

# ddtSchemes is NOT a refusal, because it is not one for OpenFOAM either: simpleFoam's UEqn carries no
# fvm::ddt term, so the entry is never consulted. OpenFOAM's OWN squareBend tutorial ships
# `application simpleFoam` with `ddtSchemes default Euler` and runs it -- this was a refusal here, and it
# blocked a case the reference solver accepts. It must be ACCEPTED and ANNOUNCED as inert instead.
try_notice transient \
  "s=open(d+'/system/fvSchemes').read(); s=re.sub(r'default\s+steadyState;','default         Euler;',s); open(d+'/system/fvSchemes','w').write(s)" \
  "inert"

# RAS/kEpsilon is SUPPORTED: the driver's turbulence hook is wired to the device k-epsilon. This used to
# be a refusal, and flipping it is the point of that work -- so assert it RUNS and writes the turbulence
# fields, not merely that it is accepted. A case that ran without writing k/epsilon/nut would mean the
# hook was never called.
echo "== 4e. the limited/blended div schemes are supported =="
# Each must RUN and each must SAY which scheme it applied -- a silent fallback to upwind would still
# converge to a plausible answer, which is exactly the failure mode this guard exists to prevent.
for sc in "limitedLinear 1" "limitedLinearV 1" "LUST grad(U)" "linearUpwindV grad(U)"; do
    tag=$(echo "$sc" | awk '{print $1}')
    supported "$W/sch_$tag"
    python3 - "$W/sch_$tag" "$sc" <<'PY'
import re, sys
p = sys.argv[1] + '/system/fvSchemes'; s = open(p).read()
open(p, 'w').write(re.sub(r'div\(phi,U\)[^;]*;', 'div(phi,U)      bounded Gauss ' + sys.argv[2] + ';', s))
PY
    ( cd "$W/sch_$tag" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" > log 2>&1 )
    if [ $? -eq 0 ]; then ok "$tag: exit 0"; else bad "$tag: refused or crashed"; sed -n '1,5p' "$W/sch_$tag/log"; fi
    grep -q "div(phi,U) scheme: $tag" "$W/sch_$tag/log" && ok "$tag: the solver states which scheme it applied" \
                                                        || bad "$tag: ran without saying which scheme"
done

echo "== 4d. SIMPLEC is supported: the rebuilt path runs it =="
# It used to be a refusal. Assert it RUNS and SAYS so -- SIMPLEC changes the iteration and not the
# converged answer, so a silent skip would still converge to a plausible result.
supported "$W/simplec"
python3 - "$W/simplec" <<'PY'
import re, sys
p = sys.argv[1] + '/system/fvSolution'; s = open(p).read()
assert s.count('consistent') == 1
open(p, 'w').write(re.sub(r'consistent\s+\S+;', 'consistent      yes;', s))
PY
( cd "$W/simplec" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" > log 2>&1 )
if [ $? -eq 0 ]; then ok "SIMPLEC: exit 0"; else bad "SIMPLEC: refused or crashed"; sed -n '1,5p' "$W/simplec/log"; fi
grep -q "SIMPLE/consistent" "$W/simplec/log" && ok "SIMPLEC: the solver states it is applying it" \
                                             || bad "SIMPLEC: ran without saying so"

echo "== 4b. RAS/kEpsilon is supported: the turbulence hook runs =="
supported "$W/ras"
python3 - "$W/ras" <<'PYEOF'
import re, sys
d = sys.argv[1]
s = open(d + '/constant/turbulenceProperties').read()
s = re.sub(r'simulationType\s+\S+;', 'simulationType  RAS;', s)
open(d + '/constant/turbulenceProperties', 'w').write(s)
PYEOF
( cd "$W/ras" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" > log 2>&1 )
rasrc=$?
if [ $rasrc -eq 0 ]; then ok "RAS/kEpsilon: exit 0"; else bad "RAS/kEpsilon: exit $rasrc"; sed -n '1,4p' "$W/ras/log"; fi
for fld in k epsilon nut; do
    if [ -f "$W/ras/3/$fld" ]; then ok "RAS/kEpsilon: wrote $fld (the hook ran)"
    else bad "RAS/kEpsilon: no $fld written"; fi
done
# The turbulence must have CHANGED nut -- a hook that ran but did nothing would still write the file.
if [ -f "$W/ras/3/nut" ] && [ -f "$W/ras/0/nut" ]; then
    if cmp -s "$W/ras/3/nut" "$W/ras/0/nut"; then bad "RAS/kEpsilon: nut is unchanged (control)"
    else ok "RAS/kEpsilon: nut changed (control)"; fi
fi

echo "== 5. NEGATIVE CONTROL: the guard is not refusing everything =="
# The supported case above ran. If it had not, every "refused" line would be meaningless -- a guard that
# blocks unconditionally passes every refusal test.
[ -d "$W/on/3" ] && ok "the guard admits a supported case (control)" \
                 || bad "the guard refuses everything (control)"

echo
[ $fails -eq 0 ] && { echo "PASS"; exit 0; } || { echo "FAIL ($fails)"; exit 1; }
