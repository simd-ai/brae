#!/usr/bin/env bash
# THE TURBULENCE MODEL IS THE ONE simulationType NAMES.
#
# turbulenceModel::New selects RASModel or LESModel by `simulationType` and never reads the other sub-dictionary. brae's
# readTurbulenceModel took LES{} whenever the file held one: occDrivAerRotMesh's RANS stage keeps both blocks under
# `simulationType RAS`, and brae ran its kOmegaSSTDDES where OpenFOAM runs kOmegaSST.
#
# Transient pitzDailySST (pimpleFoam, which takes RAS and LES), 3 steps:
#   (a) as shipped (RAS kOmegaSST); (b) the same with an LES{} block (kOmegaSSTDDES, maxDeltaxyz) added and simulationType
#   still RAS; (c) (b) with simulationType LES.
#   PREMISE  OpenFOAM(b) == OpenFOAM(a), every field byte for byte: OpenFOAM does not read the LES block under RAS.
#   ARM      brae(b) == brae(a), byte for byte.
#   CONTROL  brae(c) != brae(a): the LES model, when it is the one named, moves the fields, so the ARM would see it taken.
#   Run with BRAE_BIN set to a binary from before the fix (lc7): the ARM fails there.
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
command -v pimpleFoam > /dev/null 2>&1 || { echo "SKIP: pimpleFoam not on PATH"; exit 77; }

W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
fail=0
say()
{
    printf '  %-96s %s\n' "$1" "$2"
    [ "$2" = FAIL ] && fail=1 || true
}

# variant <dir> <a|b|c>
variant()
{
    rm -rf "$1"
    mkdir -p "$1"
    cp -r "$SRC/constant" "$SRC/system" "$SRC/0" "$1/"
    rm -f "$1"/constant/polyMesh/.brae_*
    python3 - "$1" "$2" << 'PY'
import re, sys
d, v = sys.argv[1], sys.argv[2]
def edit(p, f):
    s = open(d + '/' + p).read()
    t = f(s)
    assert t != s, (p, v)
    open(d + '/' + p, 'w').write(t)
c = lambda s: re.sub(r'functions\s*\{.*\n\}', 'functions\n{\n}', s, flags=re.S)
edit('system/controlDict', lambda s: re.sub(r'deltaT\s+[^;]+;', 'deltaT 1e-5;', re.sub(r'writeInterval\s+[^;]+;', 'writeInterval 3;',
     re.sub(r'endTime\s+[^;]+;', 'endTime 3e-5;', re.sub(r'application\s+\w+;', 'application pimpleFoam;',
     re.sub(r'writePrecision\s+[^;]+;', 'writePrecision 17;', c(s)))))))
edit('system/fvSchemes', lambda s: re.sub(r'ddtSchemes\s*\{[^}]*\}', 'ddtSchemes { default Euler; }', s))
def pimple(s):
    a = re.search(r'\bSIMPLE\b\s*\{', s)
    k, depth = a.end() - 1, 0
    while True:
        depth += {'{': 1, '}': -1}.get(s[k], 0)
        k += 1
        if depth == 0:
            break
    return s[:a.start()] + 'PIMPLE\n{\n    nOuterCorrectors 1;\n    nCorrectors 2;\n    nNonOrthogonalCorrectors 0;\n}' + s[k:]
edit('system/fvSolution', pimple)
# OpenFOAM's pimpleFoam asks for the Final solvers: the same entries at relTol 0, at the END of solvers{} (after p, which $p names)
def finals(s):
    a = re.search(r'\bsolvers\b\s*\{', s)
    k, depth = a.end() - 1, 0
    while True:
        depth += {'{': 1, '}': -1}.get(s[k], 0)
        k += 1
        if depth == 0:
            break
    add = ('    "(U|k|omega|epsilon)Final" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-05; relTol 0; }\n'
           '    pFinal { $p; relTol 0; }\n')
    return s[:k - 1] + add + s[k - 1:]
edit('system/fvSolution', finals)
if v in ('b', 'c'):
    edit('constant/turbulenceProperties', lambda s: s + '\nLES\n{\n    LESModel kOmegaSSTDDES;\n    delta maxDeltaxyz;\n    turbulence on;\n'
         '    printCoeffs on;\n}\n')
if v == 'c':
    edit('constant/turbulenceProperties', lambda s: re.sub(r'simulationType\s+\w+;', 'simulationType LES;', s))
PY
}

same()
{
    local f
    for f in U p k omega nut
    do
        [ -f "$1/3e-05/$f" ] && cmp -s "$1/3e-05/$f" "$2/3e-05/$f" || return 1
    done
    return 0
}

for v in a b c
do
    variant "$W/of-$v" $v
    variant "$W/br-$v" $v
done
for v in a b
do
    ( cd "$W/of-$v" && pimpleFoam > log 2>&1 ) || { echo "FAIL: OpenFOAM on ($v)"; tail -n 15 "$W/of-$v/log"; exit 1; }
done
for v in a b c
do
    ( cd "$W" && "$BRAE" -case "$W/br-$v" > "$W/br-$v.log" 2>&1 )
    echo $? > "$W/br-$v.rc"
done
say "PREMISE OpenFOAM(b), an LES block under simulationType RAS, == OpenFOAM(a), every field byte for byte" \
    "$(same "$W/of-a" "$W/of-b" && echo ok || echo FAIL)"
say "ARM     brae(b) == brae(a), byte for byte (rc $(cat "$W/br-b.rc")/$(cat "$W/br-a.rc"); brae(b) names $(grep -m1 -oE 'kOmegaSST[A-Z]*' "$W/br-b.log"))" \
    "$([ "$(cat "$W/br-a.rc")" = 0 ] && [ "$(cat "$W/br-b.rc")" = 0 ] && same "$W/br-a" "$W/br-b" && echo ok || echo FAIL)"
say "CONTROL brae(c), simulationType LES, differs from brae(a) (rc $(cat "$W/br-c.rc"))" \
    "$([ "$(cat "$W/br-c.rc")" = 0 ] && ! same "$W/br-a" "$W/br-c" && echo ok || echo FAIL)"
[ $fail = 0 ] && echo "SIMULATION TYPE PASS" || echo "SIMULATION TYPE FAIL"
exit $fail
