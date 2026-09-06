#!/usr/bin/env bash
# STOCK pitzDaily, end to end, against real OpenFOAM.
#
# This is the case the rebuilt simpleFoam was aimed at, with its own settings: SIMPLEC
# (`consistent yes`), `bounded Gauss linearUpwind grad(U)`, `Gauss linear corrected`, RAS/kEpsilon with
# wall functions, on a non-orthogonal mesh. Every one of those was a refusal at some point in this port,
# so this gate is the statement that the envelope now covers a real tutorial rather than a reduced one.
#
# WHAT IS CHANGED, and why none of it is physics:
#   * residualControl is REMOVED, so both solvers simply run to endTime. At the stock `p 1e-2` both stop
#     early at visibly different states and the comparison measures where each happened to stop; and
#     neither solver can be asked to stop at a tight residual, because both plateau -- OpenFOAM near
#     p 3e-03 and brae near p 2e-04, flat from iteration ~700 onward -- once normFactor collapses and the
#     inner solves are the case's own loose `relTol 0.1`. Settledness is therefore ASSERTED by comparing
#     the halfway write against the final one, not assumed from a residual.
#   * endTime is capped and function objects are dropped. Neither touches the solution.
# The mesh, the turbulence model, the algorithm and every discretisation scheme are untouched.
#
# TWO ASSERTIONS, and the second is the one that earns its keep:
#   1. brae agrees with OpenFOAM.
#   2. brae's SIMPLEC and brae's plain SIMPLE converge to the SAME fixed point. They must: SIMPLEC changes
#      the ITERATION, not the discrete system, so at convergence phi and U are identical -- the rAtU terms
#      cancel exactly against the pressure equation's own flux. That check is what exposed a missing
#      fvMatrix::faceFluxCorrection: the non-orthogonal correction was in the pressure equation's SOURCE
#      but not in pEqn.flux(), so `phi = phiHbyA - pEqn.flux()` silently dropped it. It measured 5.3e-03
#      then and 2.8e-04 now, while every per-stage test passed at 1e-16 throughout.
set -u
SRC="${1:?case dir}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
OFBIN=/usr/lib/openfoam/openfoam2412/platforms/linuxARM64GccDPInt32Opt
[ -x "$BRAE" ]                 || { echo "SKIP: no brae binary at $BRAE"; exit 77; }
[ -x "$OFBIN/bin/simpleFoam" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
SRC="$(cd "$SRC" && pwd)"
grep -q "consistent *yes" "$SRC/system/fvSolution"          || { echo "SKIP: case is not SIMPLEC"; exit 77; }
grep -q "linearUpwind"    "$SRC/system/fvSchemes"           || { echo "SKIP: case is not linearUpwind"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT

mkcase() {
    rm -rf "$1"; cp -r "$SRC" "$1"
    find "$1" -mindepth 1 -maxdepth 1 -type d ! -name 0 ! -name constant ! -name system -exec rm -rf {} + 2>/dev/null
    rm -f "$1"/log.* 2>/dev/null
    python3 - "$1" <<'PY'
import re, sys
d = sys.argv[1]
p = d + '/system/fvSolution'; s = open(p).read()
s = re.sub(r'residualControl\s*\{.*?\n    \}', 'residualControl\n    {\n    }', s, flags=re.S)
open(p, 'w').write(s)
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'endTime\s+\S+;',       'endTime         2000;', s)
s = re.sub(r'writeInterval\s+\S+;', 'writeInterval   1000;', s)
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
open(c, 'w').write(s)
PY
}

mkcase "$W/of"
export WM_PROJECT_DIR=/usr/lib/openfoam/openfoam2412
export FOAM_ETC="$WM_PROJECT_DIR/etc"
export PATH="$OFBIN/bin:$PATH"
export LD_LIBRARY_PATH="$OFBIN/lib:$OFBIN/lib/dummy:${LD_LIBRARY_PATH:-}"
( cd "$W/of" && timeout 1800 simpleFoam > log.of 2>&1 ) || { echo "FAIL: OpenFOAM did not run"; tail -3 "$W/of/log.of"; exit 1; }
OFT=$(ls -d "$W/of"/[0-9]* | xargs -n1 basename | sort -n | tail -1)
[ "$OFT" = "0" ] && { echo "FAIL: OpenFOAM wrote no result"; exit 1; }
echo "  ok:   OpenFOAM reference generated at t=$OFT"

# 1. the case EXACTLY as it ships.
mkcase "$W/simplec"
( cd "$W/simplec" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" > run.log 2>&1 ) || {
    echo "FAIL: brae refused or crashed on stock pitzDaily"; sed -n '1,8p' "$W/simplec/run.log"; exit 1; }
# Every feature the case asks for must be reported as APPLIED, not silently skipped.
for want in "SIMPLE/consistent" "linearUpwind" "bounded" "non-orthogonal correction ON"; do
    grep -q "$want" "$W/simplec/run.log" || { echo "FAIL: brae did not report applying '$want'"; exit 1; }
done
BT=$(ls -d "$W/simplec"/[0-9]* | xargs -n1 basename | sort -n | tail -1)
echo "  ok:   brae ran stock pitzDaily to t=$BT"

# The settledness probe is a SEPARATE SHORTER RUN, not an intermediate write: the rebuilt driver writes
# only its final time (it does not honour writeInterval yet), so half the run is the only way to get a
# second point on the trajectory.
mkcase "$W/half"
python3 - "$W/half" <<'PY'
import re, sys
c = sys.argv[1] + '/system/controlDict'; s = open(c).read()
open(c, 'w').write(re.sub(r'endTime\s+\S+;', 'endTime         1000;', s))
PY
( cd "$W/half" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" > run.log 2>&1 ) || { echo "FAIL: brae half run failed"; exit 1; }
BH=$(ls -d "$W/half"/[0-9]* | xargs -n1 basename | sort -n | tail -1)
echo "  ok:   settledness probe ran to t=$BH"

# 2. the same case under plain SIMPLE, for the fixed-point control. Plain SIMPLE needs pressure
#    under-relaxation, which SIMPLEC does not, so the factors change -- at convergence they cannot affect
#    where the iteration lands, only how fast it gets there, which is exactly what is being asserted.
mkcase "$W/simple"
python3 - "$W/simple" <<'PY'
import re, sys
p = sys.argv[1] + '/system/fvSolution'; s = open(p).read()
assert s.count('consistent') == 1
s = re.sub(r'consistent\s+\S+;', 'consistent      no;', s)
s = re.sub(r'relaxationFactors\s*\{.*?\n\}',
           'relaxationFactors\n{\n    fields\n    {\n        p               0.3;\n    }\n'
           '    equations\n    {\n        U               0.7;\n        ".*"            0.7;\n    }\n}',
           s, flags=re.S)
open(p, 'w').write(s)
PY
( cd "$W/simple" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" > run.log 2>&1 ) || { echo "FAIL: brae SIMPLE run failed"; exit 1; }
ST=$(ls -d "$W/simple"/[0-9]* | xargs -n1 basename | sort -n | tail -1)
echo "  ok:   brae ran the same case under plain SIMPLE (t=$ST)"

python3 - "$W/simplec/$BT" "$W/simple/$ST" "$W/of/$OFT" "$W/half/$BH" <<'PY'
import re, sys, os, gzip
def read(f):
    for c in (f, f + '.gz'):
        if os.path.exists(c): f = c; break
    else: return None
    s = (gzip.open(f, 'rt', errors='ignore') if f.endswith('.gz') else open(f, errors='ignore')).read()
    m = re.search(r'internalField\s+nonuniform\s+List<(\w+)>\s*(\d+)\s*\(', s)
    if not m: return None
    typ, n, body = m.group(1), int(m.group(2)), s[m.end():]
    if typ == 'scalar':
        return [float(x) for x in body.replace('\n', ' ').split(')')[0].split()][:n]
    return [tuple(float(x) for x in t.split()) for t in re.findall(r'\(([^()]*)\)', body)[:n]]
def l2(a, b):
    if a is None or b is None or len(a) != len(b): return None
    if isinstance(b[0], tuple):
        d = [max(abs(x[i] - y[i]) for i in range(3)) for x, y in zip(a, b)]
        g = [max(abs(y[i]) for i in range(3)) for y in b]
    else:
        d = [abs(x - y) for x, y in zip(a, b)]; g = [abs(y) for y in b]
    return (sum(x * x for x in d) / max(sum(x * x for x in g), 1e-300)) ** 0.5

simplec, simple, of, half = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
fails = 0

# SETTLEDNESS, asserted rather than assumed. Neither solver's residual reaches a tight bound here -- both
# plateau once normFactor collapses -- so "has it converged" is answered by whether the field still moves.
u_half = l2(read(half + '/U'), read(simplec + '/U'))
if u_half is not None and u_half <= 1e-4:
    print("  ok:   brae is settled -- U moves %.3e (L2) between t=1000 and t=2000" % u_half)
else:
    print("  FAIL: brae still moving, U %s between t=1000 and t=2000" % u_half); fails += 1

print("  %-9s %13s %13s" % ("field", "vs OpenFOAM", "SIMPLEC-SIMPLE"))
for f in ['U', 'p', 'k', 'epsilon', 'nut']:
    a = l2(read(simplec + '/' + f), read(of + '/' + f))
    b = l2(read(simplec + '/' + f), read(simple + '/' + f))
    if a is None or b is None: print("  %-9s (missing)" % f); fails += 1; continue
    print("  %-9s %13.4e %13.4e" % (f, a, b))
    if f == 'U':
        # Measured 5.18e-03 on U. The bar is 8e-03: above it, but below the 8.5e-03 the same case
        # measured before faceFluxCorrection was ported, so that defect could not pass this gate either.
        if a <= 8e-3: print("  ok:   U L2 %.3e <= 8e-03 vs OpenFOAM" % a)
        else:         print("  FAIL: U L2 %.3e > 8e-03 vs OpenFOAM" % a); fails += 1
        # THE CONTROL. SIMPLEC and SIMPLE must reach the same fixed point. Measured 2.8e-04; it was
        # 5.3e-03 with the flux correction missing, so 1e-3 separates the two states.
        if b <= 1e-3: print("  ok:   SIMPLEC and SIMPLE agree to %.3e -- same fixed point (control)" % b)
        else:         print("  FAIL: SIMPLEC and SIMPLE differ by %.3e -- not the same fixed point" % b); fails += 1
print("PASS" if not fails else "FAIL")
sys.exit(1 if fails else 0)
PY
