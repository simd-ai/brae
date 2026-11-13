#!/usr/bin/env bash
# A CASE THAT NAMES ITS FILES THROUGH ${FOAM_CASE} AND COMPUTES ITS VALUES WITH #eval READS AS OPENFOAM READS IT.
#
# The OpenFOAM HPC committee's occDrivAerStaticMesh puts every setting in system/include/caseDefinition and reaches it
# from turbulenceProperties, transportProperties and each 0/ field with `#include
# "${FOAM_CASE}/system/include/caseDefinition"`; k, omega and nut are `#eval{ ... }` of those variables; its controlDict
# carries a coded function object whose `#{ ... #}` C++ has `#include "clockValue.H"`. OpenFOAM's argList sets FOAM_CASE
# (argList.C:891-893); brae never did, so the include resolved to /system/include/caseDefinition, $RASturbModel and every
# #eval were undefined, and `brae check` refused a case OpenFOAM runs. brae also took the #include lines inside the
# verbatim block for dictionary includes.
#
# Built here on pitzDailySST (kOmegaSST, omegaWallFunction): the structured case (b), the same case with every changed
# file replaced by OpenFOAM's own expansion of it (a, `foamDictionary -expand -precision 17`), 3 SIMPLE iterations each.
#   PREMISE  OpenFOAM(b) == OpenFOAM(a), byte for byte: the expansion is the only difference between the two.
#   ARM 1    brae(b) == brae(a), byte for byte from internalField on (the header is the input's, echoed): brae's
#            expansion of (b) is OpenFOAM's.
#   ARM 2    brae runs (b) from another working directory with no "#include ... not found" warning.
#   ARM 3    `brae check` on (b): not refused, names kOmegaSST, no include warning.
#   CONTROL  (c) = (b) with Tu one part in 1e9 larger: brae(c) != brae(a), so ARM 1 sees an expansion that moves a value.
#   Run with BRAE_BIN set to a binary from before the fix (lc6): ARMS 1-3 fail there.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="${1:-$ROOT/validation/pitzDailySST}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}

[ -x "$BRAE" ]                         || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -f "$SRC/constant/polyMesh/points" ]  || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ]                     || { echo "SKIP: real OpenFOAM not available"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v simpleFoam > /dev/null 2>&1 || { echo "SKIP: simpleFoam not on PATH"; exit 77; }

W=$(mktemp -d)
# BRAE_GATE_KEEP=1 keeps the work directory, to look at a red arm's fields
if [ -n "${BRAE_GATE_KEEP:-}" ]
then
    echo "work directory kept: $W"
else
    trap 'rm -rf "$W"' EXIT
fi
fail=0
say()
{
    printf '  %-96s %s\n' "$1" "$2"
    [ "$2" = FAIL ] && fail=1 || true
}

# structured <dir> <Tu>
structured()
{
    local d=$1
    rm -rf "$d"
    mkdir -p "$d"
    cp -r "$SRC/constant" "$SRC/system" "$SRC/0" "$d/"
    rm -f "$d"/constant/polyMesh/.brae_*
    python3 - "$d" "$2" << 'PY'
import re, sys
d, tu = sys.argv[1], sys.argv[2]
inc = '#include "${FOAM_CASE}/system/include/caseDefinition"\n'
import os
os.makedirs(d + '/system/include', exist_ok=True)
open(d + '/system/include/caseDefinition', 'w').write(
    'nuVal        1e-05;\nUin          10;\nTu           %s;\nviscRatio    10;\nRASturbModel kOmegaSST;\n' % tu)
def after_header(s, add):
    i = s.index('}', s.index('FoamFile')) + 1
    return s[:i] + '\n' + add + s[i:]
p = d + '/constant/transportProperties'
s = open(p).read()
s = re.sub(r'\n\s*nu\s+[^;]*;', '\nnu              $nuVal;', s)
open(p, 'w').write(after_header(s, inc))
p = d + '/constant/turbulenceProperties'
s = open(p).read()
s = re.sub(r'RASModel\s+\w+;', 'RASModel        $RASturbModel;', s)
open(p, 'w').write(after_header(s, inc))
evals = {
    'k':     'kref            #eval{ 1.5*sqr($Uin*$Tu) };\n',
    'omega': 'kref            #eval{ 1.5*sqr($Uin*$Tu) };\nnutref          #eval{ $viscRatio * $nuVal };\n'
             'omegaref        #eval{ $kref/$nutref };\n',
    'nut':   'nutref          #eval{ $viscRatio * $nuVal };\n',
}
ref = {'k': '$kref', 'omega': '$omegaref', 'nut': '$nutref'}
for f in evals:
    p = d + '/0/' + f
    s = open(p).read()
    s = re.sub(r'internalField\s+uniform\s+[^;]*;', 'internalField   uniform %s;' % ref[f], s)
    s = re.sub(r'boundaryField\s*\{', 'boundaryField\n{\n    #includeEtc "caseDicts/setConstraintTypes"\n', s, count=1)
    open(p, 'w').write(after_header(s, inc + evals[f]))
p = d + '/system/controlDict'
s = open(p).read()
s = re.sub(r'functions\s*\{.*\n\}', 'functions\n{\n}', s, flags=re.S)
for k, v in (('endTime', '3'), ('writeInterval', '3'), ('writePrecision', '17'), ('writeControl', 'timeStep')):
    s = re.sub(r'(?<![A-Za-z])' + k + r'\s+[^;]*;', '%s %s;' % (k, v), s)
coded = '''functions
{
    wallClock
    {
        type        coded;
        libs        (utilityFunctionObjects);
        name        wallClock;
        enabled     false;
        codeInclude
        #{
            #include "clockValue.H"
            #include "OFstream.H"
        #};
        codeExecute
        #{
        #};
    }
}'''
s = s.replace('functions\n{\n}', coded)
open(p, 'w').write(s)
f = d + '/system/fvSolution'
s = open(f).read()
open(f, 'w').write(re.sub(r'residualControl\s*\{[^}]*\}', 'residualControl { }', s, flags=re.S))
PY
}

structured "$W/b" 0.05
structured "$W/c" 0.0500000000500
# (a): every file (b) changed, replaced by OpenFOAM's own expansion of it
rm -rf "$W/a"
cp -r "$W/b" "$W/a"
for f in constant/transportProperties constant/turbulenceProperties 0/k 0/omega 0/nut
do
    ( cd "$W/b" && foamDictionary -expand -precision 17 "$f" > "$W/a/$f" 2> "$W/a.$$.err" ) \
        || { echo "FAIL: foamDictionary -expand $f"; cat "$W/a.$$.err"; exit 1; }
done
grep -q 'FOAM_CASE\|#eval\|\$' "$W/a/0/omega" "$W/a/0/k" "$W/a/0/nut" && { echo "FAIL: the oracle files still hold a directive"; exit 1; }
for v in a b
do
    ( cd "$W/$v" && simpleFoam > log.of 2>&1 ) || { echo "FAIL: OpenFOAM on ($v)"; tail -n 20 "$W/$v/log.of"; exit 1; }
done
eqdirs()   # eqdirs <case1> <case2>: time 3's fields byte for byte, from internalField to the end
{
    # From internalField on: every value and every boundary entry as written. NOT the header: brae's writer carries the
    # INPUT file's banner, FoamFile block and dimensions spacing into what it writes (OpenFOAM writes a header of its own),
    # and (a)'s input is foamDictionary's output, (b)'s the fixture's -- job 5115746: those lines were the whole difference,
    # every one of the 36,675 + 4 x 12,225 values and 24,170 fluxes identical.
    local f
    for f in U p k omega nut
    do
        [ -f "$1/3/$f" ] && [ -f "$2/3/$f" ] || return 1
        cmp -s <(sed -n '/^internalField/,$p' "$1/3/$f") <(sed -n '/^internalField/,$p' "$2/3/$f") || return 1
        [ -n "$(sed -n '/^internalField/p' "$1/3/$f")" ] || return 1
    done
    return 0
}
say "PREMISE OpenFOAM(b) == OpenFOAM(a), every field at 3 byte for byte from internalField" \
    "$(eqdirs "$W/a" "$W/b" && echo ok || echo FAIL)"
mkdir -p "$W/br"
for v in a b c
do
    rm -rf "$W/br/$v"
    cp -r "$W/$v" "$W/br/$v"
    rm -rf "$W/br/$v/3" "$W/br/$v"/log.*
    # from another working directory: FOAM_CASE must come from -case, not from the shell's directory
    ( cd "$W" && "$BRAE" -case "$W/br/$v" > "$W/br/$v.log" 2>&1 )
    echo "$?" > "$W/br/$v.rc"
done
say "ARM 1  brae(b) == brae(a), every field at 3 byte for byte from internalField (rc $(cat "$W/br/b.rc")/$(cat "$W/br/a.rc"))" \
    "$([ "$(cat "$W/br/b.rc")" = 0 ] && eqdirs "$W/br/a" "$W/br/b" && echo ok || echo FAIL)"
nw=$(grep -c '#include file .* not found\|cannot evaluate #eval' "$W/br/b.log")
say "ARM 2  brae run on (b) from another directory: $nw include/#eval warnings" \
    "$([ "$(cat "$W/br/b.rc")" = 0 ] && [ "$nw" = 0 ] && echo ok || echo FAIL)"
( cd "$W" && "$BRAE" check -case "$W/b" > "$W/check.log" 2>&1 )
ck=$?
nwc=$(grep -c '#include file .* not found\|cannot evaluate' "$W/check.log")
say "ARM 3  brae check (b): exit $ck, $(grep -c 'kOmegaSST' "$W/check.log") kOmegaSST line, $nwc include warnings" \
    "$([ $ck -lt 2 ] && grep -q 'turbulence .*RAS kOmegaSST' "$W/check.log" && [ "$nwc" = 0 ] && echo ok || echo FAIL)"
say "CONTROL brae(c), Tu one part in 1e9 larger, differs from brae(a) (rc $(cat "$W/br/c.rc"))" \
    "$([ "$(cat "$W/br/c.rc")" = 0 ] && ! eqdirs "$W/br/a" "$W/br/c" && echo ok || echo FAIL)"
[ $fail = 0 ] || { echo "--- check (b)"; cat "$W/check.log"; echo "--- brae (b)"; grep -iE "warn|error|refus" "$W/br/b.log" | head -n 20; }
[ $fail = 0 ] && echo "CASE INCLUDE EVAL PASS" || echo "CASE INCLUDE EVAL FAIL"
exit $fail
