#!/bin/bash
# Gate 4: brae_rhoSimpleFoam vs OpenFOAM rhoSimpleFoam on a TURBULENT (kOmegaSST) heated duct.
#
# Gate 3 runs the same geometry laminar, so it never touches the compressible turbulence path. Everything
# phase 4 added is only exercised here:
#
#   - every k/omega term rho-weighted (production, destruction, (2/3)divU dilatation, CDkOmega),
#     matching OF's alpha*rho* prefactors in kOmegaSSTBase.C
#   - DkEff/DomegaEff = rho*D + mu           (OF laplacian(alpha*rho*DkEff, k))
#   - F1/F2 blending on the per-cell nu = mu/rho   (OF this->mu()/this->rho_)
#   - muEff = mu + rho*nut in the momentum equation, cells AND wall faces
#     (OF rho*nuEff() in linearViscousStress.C, mut(patchi) = rho.boundaryField()*nut(patchi))
#   - the wall functions reading a per-FACE nu = mu_b/rho_b (OF turbulenceModel::nu(patchi))
#   - alphat = rho*nut/Prt feeding the energy equation
#
# Any one of those being wrong still CONVERGES -- it just converges to the wrong wall shear or the wrong
# wall heat flux. That is why this compares seven fields against OF rather than checking residuals.
set -e
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
SRC=${SRC:-$(cd "$(dirname "$0")/rhoSST" && pwd)}
BUILD=${BUILD:-$(cd "$(dirname "$0")/../build" && pwd)}
WORK=${WORK:-/tmp/brae_sst_vs_of}
# 2e-5. Tightened a third time, again because the "floor" was a bug rather than discretisation:
#   - nut on a 'calculated' patch must be EVALUATED, a1*k_b/max(a1*om_b, b1*F2_b*sqrt(S2_b)), not
#     extrapolated from the cell. That took p to machine precision.
#   - the k/omega laplacian must use the PATCH diffusivity alphaK(F1_b)*nut_b + nu_b.
# Order mattered: with an extrapolated nut_b the second change made omega 100x WORSE, so it was measured
# and disabled until the first landed. One iteration from an identical developed state now agrees to
# ~2e-8; the residual is F1_b extrapolated from the cell rather than evaluated. Fields land at ~3e-6
# here, so 2e-5 keeps ~7x margin and still catches every bug this gate has caught (5e-2, 3e-2, 3e-4).
TOL=${TOL:-2e-5}

if [ ! -f "$OFBASHRC" ]; then echo "OpenFOAM not found at $OFBASHRC -- skipping"; exit 77; fi
source "$OFBASHRC" 2>/dev/null || true

rm -rf "$WORK"; mkdir -p "$WORK"
cp -r "$SRC"/* "$WORK/"
mkdir -p "$WORK/0" && cp "$WORK"/0.orig/* "$WORK/0/"
cd "$WORK"
blockMesh > log.blockMesh 2>&1
rhoSimpleFoam > log.rhoSimpleFoam 2>&1
OFLAST=$(ls -d [0-9]* | grep -vx 0 | sort -g | tail -1)

# brae runs the same case in a clean copy, sharing only the mesh OF generated
rm -rf "$WORK.brae"; cp -r "$SRC" "$WORK.brae"
cp -r "$WORK/constant/polyMesh" "$WORK.brae/constant/"
mkdir -p "$WORK.brae/0" && cp "$WORK.brae"/0.orig/* "$WORK.brae/0/"
"$BUILD/brae_rhoSimpleFoam" -case "$WORK.brae" > "$WORK.brae/log.brae" 2>&1
BRLAST=$(ls -d "$WORK.brae"/[0-9]* | grep -v '/0$' | sort -g | tail -1)

# THE STEP HARNESS ON AN SST FIXTURE (queue 13e). tests/test_rho_simple_step_cpp.cu carries a control
# that perturbs the case's closure coefficient and requires nut to move; it mutated kEpsilon's Cmu whatever
# the model, read nut 0.0000e+00 on this fixture and FAILED -- unseen because no registered gate pointed
# the harness at an SST case until this arm. It now perturbs betaStar on kOmegaSST (nut 2.2347e-02 on
# rhoSST). Fail-proof: with the closure handed default KOmegaSSTCoeffs instead of the case's, this arm FAILS.
hout=$("$BUILD/test_rho_simple_step_cpp" "$WORK" 0 "$OFLAST" 2>&1) \
    || { echo "$hout" | grep -E "FAIL|control" | head -12; echo "FAIL(harness on rhoSST)"; exit 1; }
echo "$hout" | grep -E "betaStar from the CASE|Prt from the CASE" | head -2
echo "$hout" | grep -q "^PASS" || { echo "$hout" | tail -5; echo "FAIL(harness on rhoSST)"; exit 1; }
echo "PASS(harness on rhoSST)"

python3 - "$WORK/$OFLAST" "$BRLAST" "$TOL" <<'PY'
import re, sys, math
ofd, brd, tol = sys.argv[1], sys.argv[2], float(sys.argv[3])

def rd(path):
    # NOTE: non-greedy and anchored on `boundaryField`. The greedy form matched to the LAST ')' in
    # the file, which was harmless only while boundary values were written as `uniform <x>`. Once the
    # writer began emitting the SOLVED boundary as `nonuniform List<...>( ... )`, the greedy match
    # swallowed the boundary values into the internal field and three gates failed on a correct solve.
    """Internal field as a flat list. Handles scalars and (x y z) vectors alike."""
    try:
        t = open(path).read()
    except OSError:
        return None
    m = re.search(r'internalField\s+nonuniform[^(]*\((.*?)\)\s*;\s*boundaryField', t, re.S)
    if not m:
        return None
    return [float(x) for x in re.findall(r'-?\d+\.?\d*[eE]?[-+]?\d*', m.group(1).replace('(', ' ').replace(')', ' '))]

# rho is derived by OF, not written by default -- rhoSimpleFoam writes it only with writeObjects. It is
# compared when present on BOTH sides and skipped (not silently passed) otherwise.
FIELDS = ("T", "p", "U", "k", "omega", "nut", "rho")
bad = 0
checked = 0
for f in FIELDS:
    of, br = rd(f"{ofd}/{f}"), rd(f"{brd}/{f}")
    if of is None or br is None:
        print(f"  {f}: SKIP (not written on both sides: OF={of is not None} brae={br is not None})")
        continue
    n = min(len(of), len(br))
    denom = sum(a*a for a in of[:n])
    if denom <= 0.0:
        print(f"  {f}: FAIL OF field is all zeros -- nothing to compare, fix the case not the tolerance")
        bad += 1
        continue
    l2 = math.sqrt(sum((a-b)**2 for a, b in zip(of[:n], br[:n])) / denom)
    spread = (max(of[:n]) - min(of[:n])) / (abs(sum(of[:n])/n) + 1e-300)
    # spread guard: a uniform field agrees to machine precision while testing nothing. Gate 3 was
    # rebuilt once for exactly this reason; the guard is here so it cannot happen again silently.
    ok = l2 <= tol and spread > 1e-6
    print(f"  {f}: L2rel {l2:.4e}  tol {tol:.0e}  spread {spread:.3e}  {'OK' if ok else 'FAIL'}")
    checked += 1
    if not ok:
        bad += 1

if checked < 5:
    print(f"  FAIL only {checked} fields compared -- the gate is not testing what it claims")
    bad += 1
print(f"sst_vs_openfoam: {bad} failures over {checked} fields")
sys.exit(1 if bad else 0)
PY

# kOmegaSSTLM must be REFUSED compressibly, not run as the plain SST it sets ctl.sst for -- the
# gamma-ReThetat transition equations are wired on the incompressible drivers only. The mutation only
# touches the dict, so the refusal fires before any field is read.
LMW=$(mktemp -d); trap 'rm -rf "$LMW"' EXIT
cp -r "$WORK"/* "$LMW/"   # the main arm's workdir, mesh already built
sed -i 's/kOmegaSST;/kOmegaSSTLM;/' "$LMW/constant/turbulenceProperties"
grep -q "kOmegaSSTLM" "$LMW/constant/turbulenceProperties" || { echo "FAIL: LM mutation did not apply"; exit 1; }
lmout=$("$BUILD/brae_rhoSimpleFoam" -case "$LMW" 2>&1) && { echo "FAIL: kOmegaSSTLM ran compressibly as plain SST"; exit 1; }
echo "$lmout" | grep -q "kOmegaSSTLM" \
    && echo "PASS(lm-refused)" \
    || { echo "$lmout" | tail -4; echo "FAIL: the refusal does not name kOmegaSSTLM"; exit 1; }

