#!/usr/bin/env bash
# velocityDampingConstraint, occDrivAerStaticMesh's one fvOption, against real OpenFOAM: diag += C*V^(2/3)*(|U| - UMax) on each
# selected cell where |U| > UMax, after UEqn.relax() (velocityDampingConstraint.C, addDamping; fvOptions.constrain(UEqn)), so in
# the predictor and in rAU and SIMPLEC's rAtU. brae ran it on one GPU from the start, ungated; its reader took a missing UMax as
# "no option" and a missing selectionMode, an unknown one or a cellZone not in the mesh as the whole mesh.
#
# pitzDaily (kEpsilon, SIMPLEC), restarted from OpenFOAM's own iteration 50, 5 iterations, every solve tight (p 1e-12, the rest
# 1e-13, relTol 0), OpenFOAM's sweep order; UMax 7 where the core flow runs at about 10.
#   ARM all   selectionMode all
#   ARM zone  selectionMode cellZone, a box zone made by OpenFOAM's topoSet (the zone's own cells, not the mesh)
#   BOUND     every field within 1e-10 relative L2 of OpenFOAM's at 55 (pitzDaily tight holds 6.5e-12 over 10 iterations,
#             the package's pass mark)
#   PREMISE   OpenFOAM's log damped cells in every iteration of both arms, and its all and zone answers each differ from its
#             undamped one by at least 1e-6 (so an arm that ignored the option, or took the mesh for the zone, would miss)
#   CONTROLS  brae with no fvOptions against OpenFOAM's all; brae's all against OpenFOAM's zone: both must miss
#   REFUSALS  no UMax; selectionMode points; a cellZone the mesh does not have: brae refuses each at its set-up and `brae check`
#             refuses each (exit 2), as OpenFOAM stops on each
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="${1:-$ROOT/validation/pitzDaily}"
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
trap 'rm -rf "$W"' EXIT
fail=0
say()
{
    printf '  %-100s %s\n' "$1" "$2"
    [ "$2" = FAIL ] && fail=1 || true
}

cat > "$W/gate.py" << 'PY'
import math, os, re, sys
def field(p):
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform\s+List<(\w+)>\s*\n?(\d+)\s*\(', s)
    n, w = int(m.group(2)), (3 if m.group(1) == 'vector' else 1)
    body = s[m.end():s.index('\n)', m.end())]
    v = [float(t) for t in re.findall(r'[-+0-9.eE]+', body)]
    assert len(v) == n * w, p
    return v
def rel(a, b):
    num = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    den = math.sqrt(sum(x * x for x in a)) or 1.0
    return num / den
what = sys.argv[1]
if what == 'stage':
    d, mode = sys.argv[2], sys.argv[3]
    c = d + '/system/controlDict'
    s = open(c).read()
    s = re.sub(r'functions\s*\{.*\n\}', 'functions\n{\n}', s, flags=re.S)
    for k, v in (('startFrom', 'startTime'), ('startTime', '50' if mode != 'seed' else '0'),
                 ('endTime', '55' if mode != 'seed' else '50'), ('writeInterval', '5' if mode != 'seed' else '50'),
                 ('writePrecision', '17'), ('writeControl', 'timeStep')):
        s = re.sub(r'(?<![A-Za-z])' + k + r'\s+[^;]*;', '%s %s;' % (k, v), s)
    open(c, 'w').write(s)
    f = d + '/system/fvSolution'
    s = open(f).read()
    s = re.sub(r'residualControl\s*\{[^}]*\}', 'residualControl { }', s, flags=re.S)
    if mode != 'seed':
        s = re.sub(r'(\n    p\s*\{[^}]*?)tolerance\s+[^;]*;', r'\1tolerance 1e-12;', s, count=1)
        s = re.sub(r'tolerance\s+1e-0[0-9];', 'tolerance 1e-13;', s)
        s = re.sub(r'relTol\s+[^;]*;', 'relTol 0;', s)
        s = re.sub(r'(\n    p\s*\{)', r'\1\n        maxIter 2000;', s, count=1)
    open(f, 'w').write(s)
elif what == 'fields':
    # fields <ref dir> <dir> <bound> [miss]: every field's relative L2
    ref, got, bound = sys.argv[2], sys.argv[3], float(sys.argv[4])
    miss = len(sys.argv) > 5
    g = {f: rel(field(os.path.join(ref, f)), field(os.path.join(got, f))) for f in ('U', 'p', 'k', 'epsilon', 'nut')}
    ok = all(v <= bound for v in g.values())
    print(('within' if ok else 'MISSES') + ' ' + ' '.join('%s %.2e' % kv for kv in g.items()))
    sys.exit(0 if ok != miss else 1)
PY

fvo()   # fvo <case> <all|zone|nozone|noumax|points>
{
    local body
    case $2 in
        all) body="selectionMode all; UMax 7;" ;;
        zone) body="selectionMode cellZone; cellZone dampZone; UMax 7;" ;;
        nozone) body="selectionMode cellZone; cellZone noSuchZone; UMax 7;" ;;
        noumax) body="selectionMode all;" ;;
        points) body="selectionMode points; points ((0.05 0 0)); UMax 7;" ;;
    esac
    printf 'FoamFile\n{\n    version 2.0;\n    format ascii;\n    class dictionary;\n    object fvOptions;\n}\nvelocityDamping\n{\n    type velocityDampingConstraint;\n    %s\n}\n' "$body" > "$1/system/fvOptions"
}

# the seed: OpenFOAM to iteration 50, as the case ships, and a box cellZone of OpenFOAM's topoSet
seed=$W/seed
mkdir -p "$seed"
cp -r "$SRC/constant" "$SRC/system" "$SRC/0" "$seed/"
rm -f "$seed"/constant/polyMesh/.brae_*
python3 -B "$W/gate.py" stage "$seed" seed
printf 'FoamFile\n{\n    version 2.0;\n    format ascii;\n    class dictionary;\n    object topoSetDict;\n}\nactions\n(\n    { name dampSet; type cellSet; action new; source boxToCell; box (-0.02 -0.03 -1) (0.15 0.03 1); }\n    { name dampZone; type cellZoneSet; action new; source setToCellZone; set dampSet; }\n);\n' > "$seed/system/topoSetDict"
( cd "$seed" && topoSet > log.topoSet 2>&1 && simpleFoam > log.seed 2>&1 ) < /dev/null || { echo "FAIL: OpenFOAM seed"; tail -n 5 "$seed/log.seed" "$seed/log.topoSet"; exit 1; }

case_at50()   # case_at50 <dir> <fvo mode or none>
{
    rm -rf "$1"
    mkdir -p "$1"
    cp -r "$seed/constant" "$seed/system" "$seed/50" "$1/"
    rm -rf "$1/50/uniform" "$1"/constant/polyMesh/.brae_* "$1/system/topoSetDict"
    python3 -B "$W/gate.py" stage "$1" run
    [ "$2" != none ] && fvo "$1" "$2"
    return 0
}

for v in none all zone
do
    case_at50 "$W/of-$v" "$v"
    ( cd "$W/of-$v" && simpleFoam > log 2>&1 ) < /dev/null || { echo "FAIL: OpenFOAM $v"; tail -n 5 "$W/of-$v/log"; exit 1; }
    case_at50 "$W/br-$v" "$v"
    ( cd "$W" && BRAE_SIMPLE_CASE_SOLVER=1 "$BRAE" -case "$W/br-$v" > "$W/br-$v.log" 2>&1 ) < /dev/null
    echo $? > "$W/br-$v.rc"
done
for v in all zone
do
    d=$(grep -o 'damped [0-9]* ([0-9.e+-]*%)' "$W/of-$v/log" | tr '\n' ' ')
    say "PREMISE $v: OpenFOAM damped cells in every iteration: $d" \
        "$([ "$(echo "$d" | wc -w)" -ge 10 ] && ! echo "$d" | grep -q 'damped 0 ' && echo ok || echo FAIL)"
    r=$(python3 -B "$W/gate.py" fields "$W/of-none/55" "$W/of-$v/55" 1e-6 miss) && s=ok || s=FAIL
    say "PREMISE $v: OpenFOAM's damped answer leaves its undamped one: $r" "$s"
    r=$(python3 -B "$W/gate.py" fields "$W/of-$v/55" "$W/br-$v/55" 1e-10) && s=ok || s=FAIL
    say "ARM $v: brae within 1e-10 of OpenFOAM at 55 (rc $(cat "$W/br-$v.rc")): $r" "$s"
done
r=$(python3 -B "$W/gate.py" fields "$W/of-all/55" "$W/br-none/55" 1e-10 miss) && s=ok || s=FAIL
say "CONTROL brae with no fvOptions against OpenFOAM's all misses: $r" "$s"
r=$(python3 -B "$W/gate.py" fields "$W/of-zone/55" "$W/br-all/55" 1e-10 miss) && s=ok || s=FAIL
say "CONTROL brae's all against OpenFOAM's zone misses: $r" "$s"
for v in noumax points nozone
do
    case_at50 "$W/ref-$v" "$v"
    ( cd "$W" && "$BRAE" check -case "$W/ref-$v" > "$W/ref-$v.check" 2>&1 )
    ck=$?
    ( cd "$W" && "$BRAE" -case "$W/ref-$v" > "$W/ref-$v.log" 2>&1 ) < /dev/null
    rc=$?
    say "REFUSAL $v: brae check exit $ck, brae run exit $rc ($(grep -m1 -o 'velocityDampingConstraint[^;]*' "$W/ref-$v.check" | cut -c1-70))" \
        "$([ $ck = 2 ] && [ $rc != 0 ] && grep -q "velocityDampingConstraint" "$W/ref-$v.check" && echo ok || echo FAIL)"
done
[ $fail = 0 ] && echo "VELOCITY DAMPING PASS" || echo "VELOCITY DAMPING FAIL"
exit $fail
