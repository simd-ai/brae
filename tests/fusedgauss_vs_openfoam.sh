#!/usr/bin/env bash
# OpenFOAM's `fusedGauss` scheme family (pitzDaily_fused), against real OpenFOAM.
#
# fusedGauss is NOT a different discretisation. src/fused/finiteVolume/ holds fusedGaussConvectionScheme,
# fusedGaussDivScheme, fusedGaussGrad and fusedGaussLaplacianScheme, and every one is the plain Gauss
# scheme with the field-expression temporaries replaced by fused loops. OpenFOAM leaves the original
# lines in as comments directly above the fused calls:
#     //fvm.lower() = -weights.primitiveField()*faceFlux.primitiveField();
#     multiplySubtract(fvm.lower(), weights.primitiveField(), faceFlux.primitiveField());
# and fusedGaussLaplacianScheme::fvmLaplacian is line-for-line identical to gaussLaplacianScheme's.
# The tutorial is stock pitzDaily with `libs (fusedFiniteVolume)` and the scheme words renamed -- diff
# the two system/ directories and that is all there is.
#
# brae therefore reads fusedGauss AS Gauss (scheme_parse.cuh: readFvSchemesText) and says so with a
# NOTICE. This gate is what keeps that from being a convenient assumption:
#
#   1. EQUIVALENCE, from OpenFOAM ITSELF. The same case is run by real OpenFOAM twice, once with the
#      fused scheme words and once with the plain ones. If OpenFOAM's own two answers did not agree,
#      brae's alias would be wrong no matter what brae did. Fused loops sum in a different ORDER, so
#      this is a floating-point agreement, not a bitwise one -- the measured gap is what the brae bound
#      below is calibrated against.
#   2. brae must be BIT-IDENTICAL on the two spellings. This is the assertion about the code actually
#      added: the rewrite is a rename and nothing else. A partial rewrite -- catching `fusedGauss`
#      inside a key, missing it after a tab -- shows up here as a nonzero difference.
#   3. brae must match OpenFOAM on the fused case end to end.
#   4. THE CONTROL: the same case run with div(phi,U) forced to plain `upwind` must be far worse. Without
#      it, assertion 3's bound says nothing about whether the scheme words were read at all.
set -u
SRC="${1:?stock pitzDaily case dir}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
OFBIN=/usr/lib/openfoam/openfoam2412/platforms/linuxARM64GccDPInt32Opt
TUT=/usr/lib/openfoam/openfoam2412/tutorials/incompressible/simpleFoam
[ -x "$BRAE" ]                     || { echo "SKIP: no brae binary at $BRAE"; exit 77; }
[ -x "$OFBIN/bin/simpleFoam" ]     || { echo "SKIP: real OpenFOAM not available"; exit 77; }
[ -f "$TUT/pitzDaily_fused/system/fvSchemes" ] || { echo "SKIP: no pitzDaily_fused tutorial"; exit 77; }
[ -f "$OFBIN/lib/libfusedFiniteVolume.so" ] || { echo "SKIP: OpenFOAM has no fusedFiniteVolume library"; exit 77; }
SRC="$(cd "$SRC" && pwd)"

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
export WM_PROJECT_DIR=/usr/lib/openfoam/openfoam2412
export FOAM_ETC="$WM_PROJECT_DIR/etc"
export PATH="$OFBIN/bin:$PATH"
export LD_LIBRARY_PATH="$OFBIN/lib:$OFBIN/lib/dummy:${LD_LIBRARY_PATH:-}"

# <dir> <fused|plain|upwind>. The scheme block comes from OpenFOAM's OWN tutorial files, not from a
# rename applied here -- generating the fused spelling with the same substitution brae uses would be
# testing that substitution against itself.
mkcase() {
    rm -rf "$1"; cp -r "$SRC" "$1"
    find "$1" -mindepth 1 -maxdepth 1 -type d ! -name 0 ! -name constant ! -name system -exec rm -rf {} + 2>/dev/null
    rm -f "$1"/log.* 2>/dev/null
    case "$2" in
      fused)  cp "$TUT/pitzDaily_fused/system/fvSchemes" "$1/system/fvSchemes" ;;
      plain)  cp "$TUT/pitzDaily/system/fvSchemes"       "$1/system/fvSchemes" ;;
      upwind) cp "$TUT/pitzDaily_fused/system/fvSchemes" "$1/system/fvSchemes"
              sed -i 's/div(phi,U) *bounded fusedGauss linearUpwind grad(U);/div(phi,U)      bounded fusedGauss upwind;/' "$1/system/fvSchemes" ;;
    esac
    python3 - "$1" "$2" <<'PY'
import re, sys
d, kind = sys.argv[1], sys.argv[2]
p = d + '/system/fvSolution'; s = open(p).read()
s = re.sub(r'residualControl\s*\{.*?\n    \}', 'residualControl\n    {\n    }', s, flags=re.S)
open(p, 'w').write(s)
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'endTime\s+\S+;',       'endTime         1000;', s)
s = re.sub(r'writeInterval\s+\S+;', 'writeInterval   1000;', s)
s = re.sub(r'writeFormat\s+\S+;',   'writeFormat     ascii;', s)
s = re.sub(r'writePrecision\s+\S+;','writePrecision  15;', s)
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
if kind != 'plain' and 'fusedFiniteVolume' not in s:
    s = s.replace('FoamFile', 'FoamFile', 1)
    s = re.sub(r'(\n\s*application\s+\S+;)', r'\1\nlibs            (fusedFiniteVolume);', s, count=1)
open(c, 'w').write(s)
PY
}

for k in fused plain; do
    mkcase "$W/of_$k" "$k"
    ( cd "$W/of_$k" && timeout 1800 simpleFoam > log.of 2>&1 ) \
        || { echo "FAIL: OpenFOAM did not run the $k case"; tail -5 "$W/of_$k/log.of"; exit 1; }
done
for k in fused plain upwind; do
    mkcase "$W/brae_$k" "$k"
    ( cd "$W/brae_$k" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" > run.log 2>&1 ) \
        || { echo "FAIL: brae refused or crashed on the $k case"; sed -n '1,10p' "$W/brae_$k/run.log"; exit 1; }
done
grep -q 'equivalent.*fusedGauss' "$W/brae_fused/run.log" \
    || { echo "FAIL: brae ran the fused case without announcing the fusedGauss -> Gauss equivalence"; exit 1; }
grep -q 'fusedGauss' "$W/brae_plain/run.log" \
    && { echo "FAIL: brae announced fusedGauss on a case that does not use it"; exit 1; }

python3 - "$W" <<'PY'
import os, re, sys
import numpy as np
W = sys.argv[1]

def lastTime(d):
    ts = [x for x in os.listdir(d) if re.fullmatch(r'[0-9]+(\.[0-9]+)?', x) and x != '0']
    return None if not ts else max(ts, key=float)

def read(d, f):
    fn = os.path.join(d, f)
    b = open(fn, 'rb').read()
    m = re.search(rb'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n(\d+)\s*\n\(', b)
    if not m:
        m2 = re.search(rb'internalField\s+uniform\s+([^;]+);', b)
        return None if not m2 else np.array([[float(x) for x in re.findall(rb'[-+0-9.eE]+', m2.group(1))]])
    typ = m.group(1).decode(); n = int(m.group(2)); start = m.end()
    nc = 3 if typ == 'vector' else 1
    fm = re.search(r'format\s+(\w+)', b[:1024].decode('latin-1'))
    if fm and fm.group(1) == 'binary':
        return np.frombuffer(b[start:start+n*nc*8], dtype='<f8').reshape(n, nc)
    txt = b[start:].decode('latin-1')
    vals = re.findall(r'[-+0-9.eE]+', txt.split(')\n;')[0] if ')\n;' in txt else txt)
    return np.array([float(x) for x in vals[:n*nc]]).reshape(n, nc)

FIELDS = ('U', 'p', 'k', 'epsilon', 'nut')
dirs = {}
for k in ('of_fused', 'of_plain', 'brae_fused', 'brae_plain', 'brae_upwind'):
    t = lastTime(os.path.join(W, k))
    if t is None:
        print("  FAIL: %s wrote no result" % k); sys.exit(1)
    dirs[k] = os.path.join(W, k, t)

def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))

rc = 0

# 1. OpenFOAM's own equivalence. This is the claim brae's alias rests on, measured rather than assumed.
print("  1. OpenFOAM fusedGauss vs OpenFOAM Gauss (same case, same iterations)")
ofGap = {}
for f in FIELDS:
    e = rel(read(dirs['of_fused'], f), read(dirs['of_plain'], f))
    ofGap[f] = e
    ok = e < 2e-03
    print("     %-8s L2 rel %.3e   %s" % (f, e, "ok" if ok else "FAIL (>2e-03: not the same scheme)"))
    if not ok: rc = 1

# 2. brae's rewrite is a RENAME. Same solver, same everything, two spellings -- so unlike OpenFOAM's own
#    two runs there is no reordered arithmetic to excuse a difference. Bitwise or it is not a rename.
print("  2. brae fusedGauss vs brae Gauss (must be bit-identical)")
for f in FIELDS:
    a, b = read(dirs['brae_fused'], f), read(dirs['brae_plain'], f)
    d = float(np.abs(a - b).max())
    ok = (d == 0.0)
    print("     %-8s Linf abs %.3e   %s" % (f, d, "ok" if ok else "FAIL (not a pure rename)"))
    if not ok: rc = 1

# 3. end to end against OpenFOAM. Bounded no tighter than OpenFOAM's own two spellings differ, plus the
#    brae-vs-OpenFOAM gap that stock pitzDaily already carries.
print("  3. brae fusedGauss vs OpenFOAM fusedGauss")
# Set just above where brae lands (U 1.0e-04, p 3.6e-04, k 7.8e-04, epsilon 9.6e-04,
# nut 1.3e-03) -- which is TIGHTER than OpenFOAM's own two spellings differ from each other,
# because brae runs identical code for both and has no reordered summation of its own.
# TIGHTENED 2026-09-04, and the reason is the whole of queue item 42. brae's residual turbulence error on
# this fixture WAS the k/epsilon laplacian's non-orthogonal correction, which the V2 driver hardcoded off:
# the k error read 7.759e-04 against an OpenFOAM-vs-OpenFOAM measurement of what that correction is worth
# on pitzDaily, 7.759e-04 -- the same number to four significant figures. With the case's own scheme
# forwarded the five errors fell to U 4.224e-05, p 8.145e-05, k 1.261e-04, epsilon 6.893e-04,
# nut 6.182e-04, so the bounds come down with them (about 4x the measured value, epsilon already there).
BOUND = {'U': 2e-04, 'p': 4e-04, 'k': 6e-04, 'epsilon': 3e-03, 'nut': 3e-03}
errOn = {}
for f in FIELDS:
    e = rel(read(dirs['brae_fused'], f), read(dirs['of_fused'], f))
    errOn[f] = e
    ok = e < BOUND[f]
    print("     %-8s L2 rel %.3e   bound %.1e   %s" % (f, e, BOUND[f], "ok" if ok else "FAIL"))
    if not ok: rc = 1

# 4. THE CONTROL. `upwind` is a real different discretisation of the same term; if reading the scheme
#    words made no difference to the answer, bound 3 would be measuring nothing.
e = rel(read(dirs['brae_upwind'], 'U'), read(dirs['of_fused'], 'U'))
ratio = e / max(errOn['U'], 1e-30)
ok = ratio >= 3.0
print("  4. control  div(phi,U) forced to upwind: U L2 rel %.3e   %.1fx the gated error   %s"
      % (e, ratio, "ok" if ok else "FAIL (want >=3x)"))
if not ok: rc = 1

if rc == 0:
    print("  ok:   pitzDaily_fused runs end to end, and fusedGauss is Gauss on both sides")
sys.exit(rc)
PY
