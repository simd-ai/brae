#!/usr/bin/env bash
# The limited/blended div(phi,U) schemes, end to end against REAL OPENFOAM.
#
# For each scheme the oracle is OpenFOAM running THAT scheme on the same case, and the CONTROL is brae
# running plain `upwind` against the same oracle. Without the control this gate proves nothing: every one
# of these schemes lands within ~5.5e-03 of OpenFOAM on U, and so does upwind, because pitzDaily's answer
# is not very scheme-sensitive in the max norm. What separates them is the turbulence fields, where the
# convection scheme bites hardest -- measured for limitedLinear: k 1.88e-02 with the scheme against
# 5.39e-02 with upwind, a factor of 2.9.
#
# The three schemes differ in KIND, which is what makes this worth gating rather than assuming:
#   limitedLinear   weights only, limiter on the SCALAR magSqr(U)  (LimitedScheme.H: NVDTVD + magSqr)
#   limitedLinearV  weights only, ONE vector limiter per face      (NVDVTVDV)
#   LUST            weights 0.75*linear+0.25*upwind AND 0.25 of linearUpwind's deferred correction
#   linearUpwindV   weights UNCHANGED; a DIFFERENT correction -- linearUpwind's, limited so it cannot
#                   overshoot the owner-to-neighbour jump along its own direction (linearUpwindV.C)
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

mkcase() {   # mkcase <dir> <div-entry>
    rm -rf "$1"; cp -r "$SRC" "$1"
    find "$1" -mindepth 1 -maxdepth 1 -type d ! -name 0 ! -name constant ! -name system -exec rm -rf {} + 2>/dev/null
    rm -f "$1"/log.* 2>/dev/null; rm -rf "$1/postProcessing"
    ( cd "$1/0" && rm -f C Cx Cy Cz V y nuTilda omega ) 2>/dev/null
    python3 - "$1" "$2" <<'PY'
import re, sys
d, sc = sys.argv[1], sys.argv[2]
p = d + '/system/fvSchemes'; s = open(p).read()
n = len(re.findall(r'div\(phi,U\)[^;]*;', s)); assert n == 1, "expected one div(phi,U), found %d" % n
open(p, 'w').write(re.sub(r'div\(phi,U\)[^;]*;', 'div(phi,U)      bounded Gauss ' + sc + ';', s))
v = d + '/system/fvSolution'; s = open(v).read()
open(v, 'w').write(re.sub(r'residualControl\s*\{.*?\n    \}', 'residualControl\n    {\n    }', s, flags=re.S))
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'endTime\s+\S+;', 'endTime         2000;', s)
s = re.sub(r'writeInterval\s+\S+;', 'writeInterval   2000;', s)
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
open(c, 'w').write(s)
PY
}
latest() { ls -d "$1"/[0-9]* | xargs -n1 basename | sort -n | tail -1; }

cmp2() {   # cmp2 <braeDir> <ofDir> <field>  -> prints the relative L2
python3 - "$1" "$2" "$3" <<'PY'
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
a, b = read(sys.argv[1] + '/' + sys.argv[3]), read(sys.argv[2] + '/' + sys.argv[3])
if a is None or b is None or len(a) != len(b): print("nan"); raise SystemExit
if isinstance(b[0], tuple):
    d = [max(abs(x[i]-y[i]) for i in range(3)) for x, y in zip(a,b)]
    g = [max(abs(y[i]) for i in range(3)) for y in b]
else:
    d = [abs(x-y) for x, y in zip(a,b)]; g = [abs(y) for y in b]
print("%.6e" % ((sum(x*x for x in d)/max(sum(x*x for x in g),1e-300))**0.5))
PY
}

fails=0
for sc in "limitedLinear 1" "limitedLinearV 1" "LUST grad(U)" "linearUpwindV grad(U)"; do
    tag=$(echo "$sc" | awk '{print $1}')
    mkcase "$W/of" "$sc"
    ( cd "$W/of" && timeout 1800 simpleFoam > log.of 2>&1 ) || { echo "FAIL: OpenFOAM did not run ($tag)"; fails=1; continue; }
    OT=$(latest "$W/of"); [ "$OT" = "0" ] && { echo "FAIL: OpenFOAM wrote nothing ($tag)"; fails=1; continue; }

    mkcase "$W/b" "$sc"
    ( cd "$W/b" && BRAE_SIMPLEFOAM_V2=1 timeout 1800 "$BRAE" > log 2>&1 ) || {
        echo "FAIL: brae refused or crashed on $tag"; sed -n '1,6p' "$W/b/log"; fails=1; continue; }
    grep -q "div(phi,U) scheme: $tag" "$W/b/log" || { echo "FAIL: brae did not report applying $tag"; fails=1; continue; }
    BT=$(latest "$W/b")

    mkcase "$W/c" "upwind"                       # the control: same oracle, plain upwind
    ( cd "$W/c" && BRAE_SIMPLEFOAM_V2=1 timeout 1800 "$BRAE" > log 2>&1 ) || { echo "FAIL: control run failed"; fails=1; continue; }
    CT=$(latest "$W/c")

    echo "  -- $sc"
    for f in U k nut; do
        s=$(cmp2 "$W/b/$BT" "$W/of/$OT" "$f")
        c=$(cmp2 "$W/c/$CT" "$W/of/$OT" "$f")
        printf "     %-4s scheme %s   upwind %s\n" "$f" "$s" "$c"
        if [ "$f" = "k" ]; then
            python3 -c "
import sys
s,c=float('$s'),float('$c')
# k is where the convection scheme actually bites on this case: measured 1.88e-02 with the scheme against
# 5.39e-02 with upwind. U alone cannot discriminate (1.22x), so the gate asserts on k.
if s < c: print('  ok:   %s is %.2fx closer to OpenFOAM than upwind on k (control)' % ('$tag', c/s))
else:     print('  FAIL: %s is NOT closer than upwind on k (%.3e vs %.3e)' % ('$tag', s, c)); sys.exit(1)
if s <= 3e-2: print('  ok:   %s k L2 %.3e <= 3e-02 vs OpenFOAM' % ('$tag', s))
else:         print('  FAIL: %s k L2 %.3e > 3e-02' % ('$tag', s)); sys.exit(1)
" || fails=1
        fi
    done
done
echo $([ $fails -eq 0 ] && echo PASS || echo FAIL)
exit $fails
