#!/bin/bash
# The default simpleFoam driver's smoother differences are ANNOUNCED. It runs one sweep between residual
# evaluations on U and the turbulence solves (the calls pass nSweeps 1), and symGaussSeidel on nuTilda
# whatever nuTilda's entry names -- kept, not ported (user decision 2026-10-15: brae keeps its own smoother
# path where it has one), so each difference must say so rather than run silently. The rebuilt driver
# (BRAE_SIMPLEFOAM_V2=1) honours both entries and must stay quiet.
#
#   ARM      validation/airFoil2D as shipped (GaussSeidel, nSweeps 2 on U and nuTilda): the U nSweeps, the
#            nuTilda nSweeps and the nuTilda smoother notices are all printed.
#   ARM      validation/pitzDaily with `nSweeps 2` added to its (U|k|epsilon) entry: the U and k nSweeps
#            notices are printed.
#   CONTROL  airFoil2D asking exactly what the driver runs (symGaussSeidel, nSweeps 1 on nuTilda; nSweeps 1
#            on U), and pitzDaily as shipped: none of them is printed, so a notice keys on the case's
#            entry and is not printed unconditionally.
#   CONTROL  airFoil2D as shipped on the rebuilt driver, which honours both entries: none is printed.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
[ -x "$BRAE" ] || { echo "SKIP: no brae binary"; exit 77; }
for f in airFoil2D pitzDaily
do
    [ -d "$ROOT/validation/$f/constant/polyMesh" ] || { echo "SKIP: fixture $f missing"; exit 77; }
done
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }

W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
fail=0
say()
{
    printf '  %-84s %s\n' "$1" "$2"
    [ "$2" = FAIL ] && fail=1 || true
}

stage()   # stage <fixture> <dir>
{
    mkdir -p "$2"
    cp -r "$ROOT/validation/$1/constant" "$ROOT/validation/$1/system" "$2/"
    if [ -d "$ROOT/validation/$1/0.orig" ]
    then
        cp -r "$ROOT/validation/$1/0.orig" "$2/0"
    else
        cp -r "$ROOT/validation/$1/0" "$2/0"
        rm -rf "$2/0/0.orig"
    fi
    sed -i 's/^endTime .*/endTime         1;/;s/^writeInterval .*/writeInterval   1;/' "$2/system/controlDict"
}

run()   # run <dir> [env...]: the log, notices included
{
    local d=$1
    shift
    ( cd "$d" && env "$@" "$BRAE" -case . > log 2>&1; echo $? > exit.txt )
}

has()   # has <dir> <subject>
{
    grep -q "NOTICE \[approximated\] solvers/$2:" "$1/log"
}

N_U="U nSweeps"
N_NT="nuTilda nSweeps"
N_NS="nuTilda smoother"
N_K="k nSweeps"

stage airFoil2D "$W/af"
run "$W/af"
has "$W/af" "$N_U" && has "$W/af" "$N_NT" && has "$W/af" "$N_NS" && [ "$(cat "$W/af/exit.txt")" = 0 ] \
    && say "ARM      airFoil2D as shipped: U nSweeps, nuTilda nSweeps and nuTilda smoother announced" ok \
    || say "ARM      airFoil2D as shipped: U nSweeps, nuTilda nSweeps and nuTilda smoother announced" FAIL

stage airFoil2D "$W/af_match"
python3 - "$W/af_match/system/fvSolution" <<'PY'
import re
import sys
p = sys.argv[1]
t = open(p).read()
t2 = re.sub(r'(nuTilda\s*\{[^}]*?)smoother\s+GaussSeidel;', r'\1smoother        symGaussSeidel;', t, count=1)
t2 = re.sub(r'nSweeps\s+2;', 'nSweeps         1;', t2)
if t2 == t or 'symGaussSeidel' not in t2 or 'nSweeps         2' in t2:
    raise SystemExit('fvSolution not rewritten')
open(p, 'w').write(t2)
PY
run "$W/af_match"
! has "$W/af_match" "$N_U" && ! has "$W/af_match" "$N_NT" && ! has "$W/af_match" "$N_NS" \
    && [ "$(cat "$W/af_match/exit.txt")" = 0 ] \
    && say "CONTROL  airFoil2D asking what the driver runs: none announced" ok \
    || say "CONTROL  airFoil2D asking what the driver runs: none announced" FAIL

stage pitzDaily "$W/pd"
run "$W/pd"
! has "$W/pd" "$N_U" && ! has "$W/pd" "$N_K" && [ "$(cat "$W/pd/exit.txt")" = 0 ] \
    && say "CONTROL  pitzDaily as shipped (symGaussSeidel, nSweeps 1): none announced" ok \
    || say "CONTROL  pitzDaily as shipped (symGaussSeidel, nSweeps 1): none announced" FAIL

stage pitzDaily "$W/pd2"
sed -i -E 's/^( *)smoother( +)symGaussSeidel;/\1smoother\2symGaussSeidel;\n\1nSweeps         2;/' "$W/pd2/system/fvSolution"
grep -q 'nSweeps         2;' "$W/pd2/system/fvSolution" || { say "pitzDaily nSweeps 2 not staged" FAIL; }
run "$W/pd2"
has "$W/pd2" "$N_U" && has "$W/pd2" "$N_K" && [ "$(cat "$W/pd2/exit.txt")" = 0 ] \
    && say "ARM      pitzDaily with nSweeps 2: U and k nSweeps announced" ok \
    || say "ARM      pitzDaily with nSweeps 2: U and k nSweeps announced" FAIL

stage airFoil2D "$W/af_v2"
run "$W/af_v2" BRAE_SIMPLEFOAM_V2=1
! has "$W/af_v2" "$N_U" && ! has "$W/af_v2" "$N_NT" && ! has "$W/af_v2" "$N_NS" \
    && [ "$(cat "$W/af_v2/exit.txt")" = 0 ] \
    && say "CONTROL  airFoil2D on the rebuilt driver, which honours both: none announced" ok \
    || say "CONTROL  airFoil2D on the rebuilt driver, which honours both: none announced" FAIL

[ $fail = 0 ] && echo PASS || { echo FAIL; exit 1; }
