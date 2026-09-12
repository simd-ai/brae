#!/usr/bin/env bash
# fvOptions: explicitPorositySource/DarcyForchheimer, end to end against REAL OPENFOAM.
#
# NOT simpleCar, which is the tutorial that motivated this. brae -- BOTH the shipped driver and the
# rebuilt one, identically -- disagrees with OpenFOAM on simpleCar by ~50% on U with the fvOptions file
# REMOVED from both, so that case cannot say anything about the porosity. That is a real, pre-existing
# disagreement and it is recorded in the manifest; it is not what this gate is for.
#
# Instead the case is built here: pitzDaily, which brae reproduces to ~5e-03 on U, with a porous cellZone
# cut out of the middle by topoSet. That makes the porosity the ONLY variable, which is what the control
# then exploits -- brae WITHOUT the fvOptions file, against the same OpenFOAM reference, must be clearly
# worse. Without that control the gate would pass with the resistance deleted, since pitzDaily lands at
# ~5e-03 on U either way in the max norm.
set -u
SRC="${1:?case dir}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
OFBIN=/usr/lib/openfoam/openfoam2412/platforms/linuxARM64GccDPInt32Opt
[ -x "$BRAE" ]                 || { echo "SKIP: no brae binary at $BRAE"; exit 77; }
[ -x "$OFBIN/bin/simpleFoam" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
SRC="$(cd "$SRC" && pwd)"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
export WM_PROJECT_DIR=/usr/lib/openfoam/openfoam2412
export FOAM_ETC="$WM_PROJECT_DIR/etc"
export PATH="$OFBIN/bin:$PATH"
export LD_LIBRARY_PATH="$OFBIN/lib/sys-openmpi:$OFBIN/lib:$OFBIN/lib/dummy"

cp -r "$SRC" "$W/base"
find "$W/base" -mindepth 1 -maxdepth 1 -type d ! -name 0 ! -name constant ! -name system -exec rm -rf {} + 2>/dev/null
rm -f "$W/base"/log.* ; rm -rf "$W/base/postProcessing"
( cd "$W/base/0" && rm -f C Cx Cy Cz V y nuTilda omega ) 2>/dev/null

cat > "$W/base/system/topoSetDict" <<'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object topoSetDict; }
actions
(
    { name porousCells; type cellSet;     action new; source boxToCell;
      box (0.05 -0.0254 -1) (0.15 0.0254 1); }
    { name porousZone;  type cellZoneSet; action new; source setToCellZone; set porousCells; }
);
EOF
python3 - "$W/base" <<'PY'
import re, sys
d = sys.argv[1]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'endTime\s+\S+;', 'endTime         2000;', s)
s = re.sub(r'writeInterval\s+\S+;', 'writeInterval   2000;', s)
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
open(c, 'w').write(s)
v = d + '/system/fvSolution'; s = open(v).read()
s = re.sub(r'residualControl\s*\{.*?\n    \}', 'residualControl\n    {\n    }', s, flags=re.S)
s = re.sub(r'consistent\s+\S+;', 'consistent      no;', s)
s = re.sub(r'relaxationFactors\s*\{.*?\n\}',
           'relaxationFactors\n{\n    fields\n    {\n        p               0.3;\n    }\n'
           '    equations\n    {\n        U               0.7;\n        ".*"            0.7;\n    }\n}',
           s, flags=re.S)
open(v, 'w').write(s)
f = d + '/system/fvSchemes'; s = open(f).read()
open(f, 'w').write(re.sub(r'div\(phi,U\)[^;]*;', 'div(phi,U)      bounded Gauss upwind;', s))
PY
( cd "$W/base" && topoSet > log.ts 2>&1 ) || { echo "FAIL: topoSet did not run"; exit 1; }
NZ=$(grep -oP 'porousZone now size \K\d+' "$W/base/log.ts" | tail -1)
[ -n "$NZ" ] && [ "$NZ" -gt 0 ] || { echo "FAIL: no porous cellZone was created"; exit 1; }
echo "  ok:   porous cellZone created -- $NZ cells"

cat > "$W/base/system/fvOptions" <<'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object fvOptions; }
porosity1
{
    type            explicitPorositySource;
    active          true;
    explicitPorositySourceCoeffs
    {
        type            DarcyForchheimer;
        selectionMode   cellZone;
        cellZone        porousZone;
        DarcyForchheimerCoeffs
        {
            d   d [0 -2 0 0 0 0 0] (2e5 2e5 2e5);
            f   f [0 -1 0 0 0 0 0] (0 0 0);
            coordinateSystem { origin (0 0 0); e1 (1 0 0); e2 (0 1 0); }
        }
    }
}
EOF

for tag in of v2 noP ofneg v2neg; do rm -rf "$W/$tag"; cp -r "$W/base" "$W/$tag"; done

# THE NEGATIVE-RESISTANCE VARIANT. A negative component in `d` is NOT a literal negative coefficient:
# porosityModel::adjustNegativeResistance replaces it with val*(-maxCmpt), so it becomes POSITIVE and
# scaled by the largest component -- here d (2e5 -1e3 -1e3) is really (2e5, 2e8, 2e8). Reading it
# verbatim gives a resistance that ACCELERATES the flow across the zone, which is a different physics
# problem, not a small error.
#
# NOTHING covered this before. brae had the bug AND a unit test asserting it ("d.y = -1000 survives with
# its sign"), and this gate could not see it because its own fixture is all-positive, where the
# adjustment is a no-op. That combination -- a blind gate and a test pinning the defect in place -- is
# why the case below is run against real OpenFOAM rather than against a hand-computed tensor.
for tag in ofneg v2neg; do
    sed -i 's/d   d \[0 -2 0 0 0 0 0\] (2e5 2e5 2e5);/d   d [0 -2 0 0 0 0 0] (2e5 -1e3 -1e3);/' \
        "$W/$tag/system/fvOptions"
    grep -q -- "-1e3" "$W/$tag/system/fvOptions" \
        || { echo "FAIL: could not build the negative-resistance fixture"; exit 1; }
done
rm -f "$W/noP/system/fvOptions"          # the control: same mesh, same oracle, no resistance

( cd "$W/of"  && timeout 1800 simpleFoam > log.of 2>&1 ) || { echo "FAIL: OpenFOAM did not run"; exit 1; }
( cd "$W/v2"  && BRAE_SIMPLEFOAM_V2=1 timeout 1800 "$BRAE" > log 2>&1 ) || {
    echo "FAIL: brae refused or crashed with fvOptions"; sed -n '1,8p' "$W/v2/log"; exit 1; }
grep -q "explicitPorositySource/DarcyForchheimer on $NZ cells" "$W/v2/log" || {
    echo "FAIL: brae did not report the porosity on $NZ cells"; grep -i fvoption "$W/v2/log"; exit 1; }
( cd "$W/noP" && BRAE_SIMPLEFOAM_V2=1 timeout 1800 "$BRAE" > log 2>&1 ) || { echo "FAIL: control run failed"; exit 1; }
( cd "$W/ofneg" && timeout 1800 simpleFoam > log.of 2>&1 ) \
    || { echo "FAIL: OpenFOAM did not run the negative-resistance case"; tail -20 "$W/ofneg/log.of"; exit 1; }
( cd "$W/v2neg" && BRAE_SIMPLEFOAM_V2=1 timeout 1800 "$BRAE" > log 2>&1 ) \
    || { echo "FAIL: brae did not run the negative-resistance case"; tail -20 "$W/v2neg/log"; exit 1; }
echo "  ok:   OpenFOAM, brae and the no-porosity control all ran"

latest() { ls -d "$1"/[0-9]* | xargs -n1 basename | sort -n | tail -1; }
python3 - "$W/v2/$(latest "$W/v2")" "$W/noP/$(latest "$W/noP")" "$W/of/$(latest "$W/of")" \
         "$W/v2neg/$(latest "$W/v2neg")" "$W/ofneg/$(latest "$W/ofneg")" <<'PY'
import re, sys, os, gzip
def read(f):
    for c in (f, f + '.gz'):
        if os.path.exists(c): f = c; break
    else: return None
    s = (gzip.open(f,'rt',errors='ignore') if f.endswith('.gz') else open(f,errors='ignore')).read()
    m = re.search(r'internalField\s+nonuniform\s+List<(\w+)>\s*(\d+)\s*\(', s)
    if not m: return None
    typ, n, body = m.group(1), int(m.group(2)), s[m.end():]
    if typ == 'scalar':
        return [float(x) for x in body.replace('\n',' ').split(')')[0].split()][:n]
    return [tuple(float(x) for x in t.split()) for t in re.findall(r'\(([^()]*)\)', body)[:n]]
def l2(a, b):
    if a is None or b is None or len(a) != len(b): return None
    if isinstance(b[0], tuple):
        d = [max(abs(x[i]-y[i]) for i in range(3)) for x, y in zip(a,b)]
        g = [max(abs(y[i]) for i in range(3)) for y in b]
    else:
        d = [abs(x-y) for x, y in zip(a,b)]; g = [abs(y) for y in b]
    return (sum(x*x for x in d)/max(sum(x*x for x in g),1e-300))**0.5

v2, noP, of = sys.argv[1], sys.argv[2], sys.argv[3]
fails = 0
print("  %-9s %13s %13s" % ("field", "with porosity", "without(ctl)"))
for f in ['U', 'p', 'k']:
    a, b = l2(read(v2 + '/' + f), read(of + '/' + f)), l2(read(noP + '/' + f), read(of + '/' + f))
    if a is None or b is None: print("  %-9s (missing)" % f); fails += 1; continue
    print("  %-9s %13.4e %13.4e" % (f, a, b))
    if f in ('U', 'p'):
        # Measured: U 5.29e-03 with against 1.62e-02 without (3.1x), p 2.72e-02 against 1.66e-01 (6.1x).
        if b > 2.0*a: print("  ok:   %s -- the porosity is %.1fx closer to OpenFOAM than omitting it (control)" % (f, b/a))
        else:         print("  FAIL: %s -- omitting the porosity is not clearly worse (%.3e vs %.3e)" % (f, b, a)); fails += 1
    if f == 'U':
        if a <= 1e-2: print("  ok:   U L2 %.3e <= 1e-02 vs OpenFOAM" % a)
        else:         print("  FAIL: U L2 %.3e > 1e-02 vs OpenFOAM" % a); fails += 1
# ---- THE NEGATIVE-RESISTANCE VARIANT, against real OpenFOAM ---------------------------------------
# d (2e5 -1e3 -1e3) is really (2e5, 2e8, 2e8) after porosityModel::adjustNegativeResistance. Taking it
# verbatim gives a NEGATIVE cross-stream resistance, which accelerates the flow rather than blocking it,
# so a port that skips the adjustment does not merely differ -- it solves a different problem. This is
# the only place a negative component is exercised against OpenFOAM; the fixture above is all-positive,
# where the adjustment is a no-op, and that blindness is how the defect survived with a unit test
# asserting it.
if len(sys.argv) > 5:
    v2n, ofn = sys.argv[4], sys.argv[5]
    print("  negative-resistance variant  d (2e5 -1e3 -1e3) -> (2e5, 2e8, 2e8)")
    for f in ['U', 'p']:
        a = l2(read(v2n + '/' + f), read(ofn + '/' + f))
        if a is None:
            print("  %-9s (missing)" % f); fails += 1; continue
        print("  %-9s %13.4e" % (f, a))
        if f == 'U':
            if a <= 1e-2: print("  ok:   U L2 %.3e <= 1e-02 with a negative d component" % a)
            else:         print("  FAIL: U L2 %.3e > 1e-02 -- adjustNegativeResistance" % a); fails += 1
    # CONTROL: the negative case must actually differ from the all-positive one, or this variant is
    # measuring the same problem twice and proves nothing about the adjustment.
    du = l2(read(v2n + '/U'), read(v2 + '/U'))
    if du is not None and du > 1e-3:
        print("  ok:   the negative-d case is a different problem from the positive one (%.3e, control)" % du)
    else:
        print("  FAIL: negative-d and positive-d give the same field -- the variant tests nothing")
        fails += 1

print("PASS" if not fails else "FAIL")
sys.exit(1 if fails else 0)
PY
