#!/usr/bin/env bash
# A case whose div(phi,epsilon) and div(phi,omega) name different schemes: the RAS model's own field takes its own entry. One flag held
# both and either entry set it, so pitzDailySST with a spare `div(phi,epsilon) $turbulence` ran omega on limitedLinear under
# `div(phi,omega) upwind`, about 1 % from OpenFOAM's fixed point against 2e-5 on its own entry (JUPITER jobs 2611227, 2611465).
#
# THE ORACLE: the same case with the other entry removed, the reading nothing can confuse: the fields byte for byte.
#   SST     pitzDailySST, div(phi,omega) upwind, div(phi,epsilon) limitedLinear, against the epsilon entry removed
#   KE      pitzDailyTurb (kEpsilon), div(phi,epsilon) upwind, div(phi,omega) limitedLinear, against the omega entry removed
#   SST-U   and the other way about: div(phi,omega) limitedLinear, div(phi,epsilon) upwind, must be limited (the log's flag)
# CONTROL (asserted red): BRAE_CONTROL_SCHEME_SECOND_BOTH=1 on SST, both entries setting the flags: its fields must differ from the oracle's.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
W=$(mktemp -d)
[ -n "${KEEP_W:-}" ] && { W="$KEEP_W"; mkdir -p "$W"; } || trap 'rm -rf "$W"' EXIT
export BRAE_SIMPLE_CASE_SOLVER=1
fail=0
say()
{
    printf '  %-74s %s\n' "$1" "$2"
    [ "$2" = FAIL ] && fail=1
    return 0
}
# run <tag> <case> <epsilon entry or -> <omega entry or -> [VAR=value]
run()
{
    local tag=$1 src=$ROOT/validation/$2 e=$3 o=$4 d=$W/$1
    shift 4
    mkdir -p "$d"
    cp -r "$src/0" "$src/constant" "$src/system" "$d/"
    rm -f "$d"/constant/polyMesh/.brae_*
    python3 - "$d" "$e" "$o" << 'PY'
import re, sys
d, e, o = sys.argv[1:]
c = d + '/system/controlDict'
s = open(c).read()
s = re.sub(r'functions\s*\{.*?\n\}', '', s, flags=re.S)
s = re.sub(r'startFrom\s+\S+;', 'startFrom startTime;', s)
s = re.sub(r'startTime\s+\S+;', 'startTime 0;', s)
s = re.sub(r'endTime\s+\S+;', 'endTime 5;', s)
s = re.sub(r'writeInterval\s+\S+;', 'writeInterval 5;', s)
s = re.sub(r'writeFormat\s+\S+;', 'writeFormat ascii;', s)
s = re.sub(r'writePrecision\s+\S+;', 'writePrecision 17;', s)
open(c, 'w').write(s)
f = d + '/system/fvSchemes'
t = open(f).read()
t = re.sub(r'\n[^\n]*div\(phi,epsilon\)[^\n]*', '', t)
t = re.sub(r'\n[^\n]*div\(phi,omega\)[^\n]*', '', t)
extra = ''
if e != '-':
    extra += '    div(phi,epsilon) %s;\n' % e
if o != '-':
    extra += '    div(phi,omega) %s;\n' % o
t2 = re.sub(r'(divSchemes\s*\{)', lambda m: m.group(1) + '\n' + extra, t, count=1)
assert t2 != t or not extra
open(f, 'w').write(t2)
PY
    (cd "$d" && env "$@" timeout 300 "$BRAE" -case . > log 2>&1 < /dev/null; echo $? > rc)
    [ "$(cat "$d/rc")" = 0 ] || say "$tag: exit $(cat "$d/rc"): $(grep -a -m1 -E 'brae ERROR|refused' "$d/log" | cut -c1-120)" FAIL
}
same()
{
    local a=$1 b=$2 f
    shift 2
    for f in "$@"
    do
        [ -s "$W/$a/5/$f" ] || return 1
        cmp -s "$W/$a/5/$f" "$W/$b/5/$f" || return 1
    done
    return 0
}
LIM='bounded Gauss limitedLinear 1'
UP='bounded Gauss upwind'
run sst pitzDailySST "$LIM" "$UP"
run sst-oracle pitzDailySST - "$UP"
run sst-other pitzDailySST "$UP" "$LIM"
run sst-other-oracle pitzDailySST - "$LIM"
run ke pitzDailyTurb "$UP" "$LIM"
run ke-oracle pitzDailyTurb "$UP" -
run sst-x pitzDailySST "$LIM" "$UP" BRAE_CONTROL_SCHEME_SECOND_BOTH=1
same sst sst-oracle U p k omega nut && say "SST: omega takes div(phi,omega), the epsilon entry changes nothing" ok || say "SST: omega takes div(phi,omega), the epsilon entry changes nothing" FAIL
same sst-other sst-other-oracle U p k omega nut && say "SST the other way about: byte for byte" ok || say "SST the other way about: byte for byte" FAIL
same ke ke-oracle U p k epsilon nut && say "kEpsilon: epsilon takes div(phi,epsilon), the omega entry changes nothing" ok || say "kEpsilon: epsilon takes div(phi,epsilon), the omega entry changes nothing" FAIL
same sst sst-other U p k omega nut && say "the omega entry reaches omega: the two schemes differ" FAIL || say "the omega entry reaches omega: the two schemes differ" ok
if same sst-x sst-oracle U p k omega nut
then
    say "CONTROL BRAE_CONTROL_SCHEME_SECOND_BOTH: stayed green" FAIL
else
    say "CONTROL BRAE_CONTROL_SCHEME_SECOND_BOTH: red" ok
fi
[ "$fail" = 0 ] && echo "simple_second_scalar_scheme: PASS" || { echo "simple_second_scalar_scheme: FAIL"; exit 1; }
