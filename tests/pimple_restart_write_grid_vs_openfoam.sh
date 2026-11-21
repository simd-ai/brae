#!/usr/bin/env bash
# THE WRITE GRID AFTER A RESTART, against REAL OpenFOAM pimpleFoam.
#
# WHY: under `writeControl runTime` (or adjustableRunTime) OpenFOAM's Time::operator++ writes when
#     label((value() - startTime_ + 0.5*deltaT)/writeInterval) > writeTimeIndex_        (Time.C:1116-1128)
# with startTime_ the RESOLVED start -- the latestTime directory a restart begins from (Time.C:149-193) -- and
# writeTimeIndex_ 0. The write grid therefore continues from the start: a restart at 0.37 with writeInterval 0.1 writes
# 0.47, 0.57. brae_pimpleFoam's WriteControl measured from controlDict's startTime instead (Time(controlDict, ...) and
# setRunStart, which leaves the write cadence alone, gpuPimpleFoam.cu), so the first step after the restart was a write and
# the later ones sat on the absolute grid.
#
# THE CASE: the committed pitzDaily fixture made transient as tests/pimple_run.sh makes it (kEpsilon, Euler, PIMPLE 2 x 2),
# deltaT 1e-5. Both codes run 12 steps from 0 and write 0.00012 (writeControl timeStep 12). Each then restarts with
# `startFrom latestTime; writeControl runTime; writeInterval 5e-05; endTime 0.00022`, a start off the write grid:
#     OpenFOAM from its own 0.00012          the reference set of written times
#     brae from its own 0.00012              must write the same set
#     brae from OpenFOAM's 0.00012           must write the same set
#     OpenFOAM from brae's 0.00012           must write the same set, and run
# The reference is 0.00017, 0.00022. endTime is on the restarted grid, so brae's own final write (it always writes the last
# step) is one of OpenFOAM's and the sets can be equal.
#
# TOLERANCE: none. The written time names are compared as sets of directory names, exactly.
#
# CONTROL, asserted red: BRAE_CONTROL_WRITE_GRID_ABSOLUTE=1 measures the grid from controlDict's startTime, as before the
# fix; brae must then write another set (0.00013 0.00015 0.0002 0.00022).
#
# NOT CLAIMED: the field values. This change moves which steps are written, not what is computed; the fields are not compared.
set -u
BIN="${1:?brae_pimpleFoam binary}"
SRC="${2:?committed pitzDaily case dir}"
W="${3:?work dir}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}

[ -x "$BIN" ] || { echo "SKIP: $BIN not built"; exit 77; }
[ -f "$SRC/constant/polyMesh/points" ] || { echo "SKIP: fixture '$SRC' not present"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v pimpleFoam > /dev/null 2>&1 || { echo "SKIP: pimpleFoam not on PATH"; exit 77; }

rm -rf "$W"
mkdir -p "$W"
fail=0

stage()   # stage <dir>: the transient fixture, 12 steps from 0 written at the end
{
    rm -rf "$1"
    mkdir -p "$1"
    cp -r "$SRC/constant" "$SRC/system" "$SRC/0" "$1/"
    rm -f "$1"/constant/polyMesh/.brae_*
    cat > "$1/system/controlDict" <<EOF
FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application     pimpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         0.00012;
deltaT          1e-05;
writeControl    timeStep;
writeInterval   12;
purgeWrite      0;
writeFormat     ascii;
writePrecision  17;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable false;
EOF
    python3 - "$1" <<'PYEOF' || return 1
import re, sys
d = sys.argv[1]
s = open(d + '/system/fvSchemes').read()
s = re.sub(r'ddtSchemes\s*\{[^}]*\}', 'ddtSchemes\n{\n    default         Euler;\n}', s, flags=re.S)
open(d + '/system/fvSchemes', 'w').write(s)
s = open(d + '/system/fvSolution').read()
def cut(text, key):   # the block of `key` (a top-level dictionary), by matching braces
    m = re.search(r'\n%s\s*\{' % key, text)
    if not m:
        return text
    i, depth = m.end() - 1, 0
    while i < len(text):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                break
        i += 1
    return text[:m.start()] + text[i + 1:]
s = cut(cut(s, 'SIMPLE'), 'PIMPLE')
s += '\nPIMPLE\n{\n    nOuterCorrectors 2;\n    nCorrectors 2;\n    nNonOrthogonalCorrectors 0;\n}\n'
# the last outer corrector selects <field>Final, which OpenFOAM stops without: each solved field's block copied under it
def block(text, key):
    m = re.search(r'\n(\s*)%s\s*\n?\s*\{' % re.escape(key), text)
    if not m:
        return None
    i, depth = text.index('{', m.start()), 0
    while i < len(text):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[text.index('{', m.start()):i + 1]
        i += 1
    return None
finals = ''
for key in ('p', '"(U|k|epsilon|omega|f|v2)"'):
    b = block(s, key)
    if b:
        finals += '\n    %s\n    %s\n' % (key[:-1] + 'Final"' if key.startswith('"') else key + 'Final', b)
assert finals.count('Final') >= 2, 'the fixture has no p and momentum solver blocks to copy'
m = re.search(r'\nsolvers\s*\n?\s*\{', s)
i, depth = s.index('{', m.start()), 0
while i < len(s):
    if s[i] == '{':
        depth += 1
    elif s[i] == '}':
        depth -= 1
        if depth == 0:
            break
    i += 1
s = s[:i] + finals + s[i:]
open(d + '/system/fvSolution', 'w').write(s)
PYEOF
}

restart_dict()   # restart_dict <dir>: the restart's controlDict, a start off the runTime write grid
{
    python3 - "$1/system/controlDict" <<'PYEOF'
import re, sys
p = sys.argv[1]
s = open(p).read()
for k, v in (('startFrom', 'latestTime'), ('endTime', '0.00022'), ('writeControl', 'runTime'), ('writeInterval', '5e-05')):
    s = re.sub(r'(?m)^%s\s+[^;]*;' % k, '%s %s;' % (k, v), s)
open(p, 'w').write(s)
PYEOF
}

written()   # written <dir>: the time directories after 0.00012, sorted
{
    (cd "$1" && for d in [0-9]*; do
        [ -f "$d/U" ] || continue
        python3 -c "import sys; t=float(sys.argv[1]); print(sys.argv[1]) if t > 0.00012 + 1e-12 else None" "$d"
    done) | sort -g | tr '\n' ' ' | sed 's/ $//'
}

run_of()   # run_of <dir> <log>
{
    (cd "$1" && nice -n 10 pimpleFoam > "$2" 2>&1)
}

run_brae()   # run_brae <dir> <log> [env...]
{
    local d=$1 log=$2
    shift 2
    env "$@" "$BIN" -case "$d" > "$log" 2>&1
}

# the two starting runs
stage "$W/of0" || { echo "FAIL: staging"; exit 1; }
cp -r "$W/of0" "$W/brae0"
run_of "$W/of0" "$W/of0.log" || { echo "FAIL: OpenFOAM's first run"; tail -5 "$W/of0.log"; exit 1; }
run_brae "$W/brae0" "$W/brae0.log" || { echo "FAIL: brae's first run"; tail -5 "$W/brae0.log"; exit 1; }
for d in of0 brae0; do
    [ -f "$W/$d/0.00012/U" ] || { echo "FAIL: $d did not write 0.00012"; exit 1; }
done

# the four restarts and the control
restart_from()   # restart_from <name> <source run>
{
    rm -rf "$W/$1"
    mkdir -p "$W/$1"
    cp -r "$W/$2/constant" "$W/$2/system" "$W/$2/0" "$W/$2/0.00012" "$W/$1/"
    restart_dict "$W/$1"
}
restart_from of_of of0
restart_from brae_brae brae0
restart_from brae_of of0
restart_from of_brae brae0
restart_from control brae0
run_of "$W/of_of" "$W/of_of.log" || { echo "FAIL: OpenFOAM's restart from its own write"; tail -5 "$W/of_of.log"; exit 1; }
REF=$(written "$W/of_of")
echo "OpenFOAM, from its own 0.00012:    $REF"
[ "$REF" = "0.00017 0.00022" ] || { echo "FAIL: OpenFOAM's own set is not the one this gate was built on"; fail=1; }
check()   # check <name> <set>
{
    if [ "$2" = "$REF" ]; then
        printf '  %-40s %-30s ok\n' "$1" "$2"
    else
        printf '  %-40s %-30s FAIL (OpenFOAM %s)\n' "$1" "$2" "$REF"
        fail=1
    fi
}
run_brae "$W/brae_brae" "$W/brae_brae.log" || { echo "FAIL: brae's restart from its own write"; tail -5 "$W/brae_brae.log"; fail=1; }
check "brae, from its own 0.00012" "$(written "$W/brae_brae")"
run_brae "$W/brae_of" "$W/brae_of.log" || { echo "FAIL: brae's restart from OpenFOAM's write"; tail -5 "$W/brae_of.log"; fail=1; }
check "brae, from OpenFOAM's 0.00012" "$(written "$W/brae_of")"
run_of "$W/of_brae" "$W/of_brae.log" || { echo "FAIL: OpenFOAM's restart from brae's write"; tail -5 "$W/of_brae.log"; fail=1; }
check "OpenFOAM, from brae's 0.00012" "$(written "$W/of_brae")"
run_brae "$W/control" "$W/control.log" BRAE_CONTROL_WRITE_GRID_ABSOLUTE=1 \
    || { echo "FAIL: the control run"; tail -5 "$W/control.log"; fail=1; }
CTL=$(written "$W/control")
if [ "$CTL" != "$REF" ] && grep -qF "CONTROL MODE" "$W/control.log"; then
    printf '  %-40s %-30s red, as it must be\n' "CONTROL: grid from controlDict's start" "$CTL"
else
    printf '  %-40s %-30s FAIL: the control does not differ\n' "CONTROL: grid from controlDict's start" "$CTL"
    fail=1
fi

[ "$fail" -eq 0 ] && echo "PASS: after a restart brae writes on OpenFOAM's write grid"
exit "$fail"
