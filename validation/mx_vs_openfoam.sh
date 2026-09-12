#!/bin/bash
# mixedEnergy: a `mixed` (Robin) T wall vs OF v2412 rhoSimpleFoam.
#
# basicThermo::heBoundaryTypes() maps a `mixed` T patch to mixedEnergy, which sets THREE things
# (mixedEnergyFvPatchScalarField.C:120-123):
#     valueFraction() = Tw.valueFraction()
#     refValue()      = thermo.he(pw, Tw.refValue())
#     refGrad()       = thermo.Cpv(pw, Tw, patchi) * Tw.refGrad()
# brae's MixedPatchField hardcoded refGrad = 0, so this whole BC was refused. It is how an external-
# convection wall is written, and it is the last of the three energy BC families OF dispatches to.
#
# The subtlety the fixedGradient path never exercises: OF weights the refGrad term by (1-vf), not 1
# (mixedFvPatchField.C:279-310, both valueBoundaryCoeffs and gradientBoundaryCoeffs are `lerp`s). With
# vf = 0 that reduces to 1, which is why B5 was right without it. Measured cost of getting it wrong:
# boundaryCoeffs 9.81e-02 on `bc_vs_openfoam`'s mixed patch.
#
# Measured here, converged: T 8.75e-08, p 5.20e-09, U 2.60e-06, rho 8.63e-08, and brae stops at OF's own
# iteration count (104). The wall is genuinely active -- OF's T spans 300..330.6 K.
set -e
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
SRC=${SRC:-$(cd "$(dirname "$0")/rhoBoxM" && pwd)}
BUILD=${BUILD:-$(cd "$(dirname "$0")/../build" && pwd)}
WORK=${WORK:-/tmp/brae_mx_vs_of}
TOL=${TOL:-1e-5}

if [ ! -f "$OFBASHRC" ]; then echo "OpenFOAM not found at $OFBASHRC -- skipping"; exit 77; fi
source "$OFBASHRC" 2>/dev/null || true

rm -rf "$WORK"; cp -r "$SRC" "$WORK"
mkdir -p "$WORK/0" && cp "$WORK"/0.orig/* "$WORK/0/"
( cd "$WORK" && blockMesh > log.blockMesh 2>&1 && rhoSimpleFoam > log.rhoSimpleFoam 2>&1 )
OFLAST=$(ls -d "$WORK"/[0-9]* | grep -v '\.orig$' | grep -v '/0$' | sort -g | tail -1)

rm -rf "$WORK.brae"; cp -r "$SRC" "$WORK.brae"
cp -r "$WORK/constant/polyMesh" "$WORK.brae/constant/"
mkdir -p "$WORK.brae/0" && cp "$WORK.brae"/0.orig/* "$WORK.brae/0/"
"$BUILD/brae_rhoSimpleFoam" -case "$WORK.brae" > "$WORK.brae/log.brae" 2>&1
BRLAST=$(ls -d "$WORK.brae"/[0-9]* | grep -v '\.orig$' | grep -v '/0$' | sort -g | tail -1)

python3 - "$OFLAST" "$BRLAST" "$TOL" <<'PY'
import re, sys, math
ofd, brd, tol = sys.argv[1], sys.argv[2], float(sys.argv[3])
def rd(p):
    try: t = open(p).read()
    except OSError: return None
    m = re.search(r'internalField\s+nonuniform[^(]*\((.*?)\)\s*;\s*boundaryField', t, re.S)
    return [float(x) for x in m.group(1).replace('(',' ').replace(')',' ').split()] if m else None

bad, checked = 0, 0
for f in ("T", "p", "U", "rho"):
    of, br = rd(f"{ofd}/{f}"), rd(f"{brd}/{f}")
    if of is None or br is None:
        print(f"  {f}: FAIL not written on both sides"); bad += 1; continue
    n = min(len(of), len(br))
    l2 = math.sqrt(sum((a-b)**2 for a, b in zip(of[:n], br[:n])) / (sum(a*a for a in of[:n]) or 1))
    ok = l2 <= tol
    print(f"  {f:4s}: L2rel {l2:.4e}  tol {tol:.0e}  {'OK' if ok else 'FAIL'}")
    checked += 1
    if not ok: bad += 1

# The EVALUATED boundary value, asserted separately. OF's mixed evaluate() is
#     lerp(patchInternal + refGrad/deltaCoeffs, refValue, vf)
# so the refGrad term rides inside the (1-vf) branch there too. The internal-field comparison above does
# NOT cover it: with `upwind` convection and an `orthogonal` laplacian nothing reads the boundary value, so
# dropping refGrad from the value kernel leaves every internal field unchanged (verified by mutation). It
# would surface the moment the case used linearUpwind or a corrected laplacian -- i.e. silently, later.
def bval(path, patch):
    t = open(path).read(); t = t[t.index('boundaryField'):]
    m = re.search(patch + r'\s*\{(.*?)\n\s{4}\}', t, re.S)
    if not m: return None
    v = re.search(r'value\s+nonuniform[^(]*\((.*?)\)', m.group(1), re.S)
    if v: return [float(x) for x in v.group(1).split()]
    u = re.search(r'value\s+uniform\s+([-\d.eE+]+)', m.group(1))
    return [float(u.group(1))] if u else None

ob, bb = bval(f"{ofd}/T", "hotWall"), bval(f"{brd}/T", "hotWall")
if ob is None or bb is None:
    print(f"  T/hotWall: FAIL no boundary value written (OF={ob is not None} brae={bb is not None})")
    bad += 1
else:
    n = min(len(ob), len(bb))
    if len(ob) == 1: ob = ob * len(bb); n = len(bb)
    e = math.sqrt(sum((a-b)**2 for a, b in zip(ob[:n], bb[:n])) / (sum(a*a for a in ob[:n]) or 1))
    print(f"  T/hotWall (evaluated mixed value): L2rel {e:.4e}  OF {ob[0]:.4f}  brae {bb[0]:.4f}  "
          f"{'OK' if e <= 1e-5 else 'FAIL'}")
    if e > 1e-5:
        print("       OF: lerp(internal + refGrad/dc, refValue, vf). A value near the pure-Robin blend")
        print("       (1-vf)*internal + vf*refValue means refGrad is missing from the VALUE kernel.")
        bad += 1

# The mixed wall must actually DO something, else "they agree" is vacuous -- a refGrad that never reached
# the discretisation would leave T nearly uniform and still pass a norm comparison.
T = rd(f"{ofd}/T")
if T and (max(T) - min(T)) < 10.0:
    print(f"  FAIL OF's T spread is {max(T)-min(T):.3g} K -- the mixed wall stopped being active")
    bad += 1
if checked < 4:
    print(f"  FAIL only {checked} fields compared"); bad += 1
print(f"mx_vs_openfoam: {bad} failures over {checked} fields")
sys.exit(1 if bad else 0)
PY

# THE MIRROR ARM. Everything above gates the LEGACY binary; the OF-mirror path refused T `mixed`
# outright until the he mapping learned it (rhoCreateFields_cpp.cu: gradient slots SCALE by Cpv, they
# do not go through the affine heOf -- gradientEnergy/mixedEnergyFvPatchScalarField.C, second term
# identically zero for a pureMixture). The engagement check keeps a fixture regression from silently
# turning this arm into a second copy of the plain gate.
grep -q "mixed" "$WORK/0/T" || { echo "FAIL: fixture no longer carries mixed T"; exit 1; }
OFLAST=$(cd "$WORK" && ls -d [0-9]* | grep -vx 0 | sort -g | tail -1)
mout=$("$BUILD/test_rho_simple_step_cpp" "$WORK" 0 "$OFLAST" 2>&1) \
    || { echo "$mout" | tail -15; echo "FAIL(mirror)"; exit 1; }
echo "$mout" | grep -E "^     T |^     U " | head -2
echo "$mout" | grep -q "^PASS" || { echo "$mout" | tail -5; echo "FAIL(mirror)"; exit 1; }
echo "PASS(mirror)"
