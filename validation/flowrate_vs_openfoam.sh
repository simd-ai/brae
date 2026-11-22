#!/bin/bash
# Gate: flowRateInletVelocity must be fed the SOLVER's rho, not thermo.rho(), when rho is relaxed.
#
# WHY THIS GATE EXISTS. OF's flowRateInletVelocityFvPatchVectorField::updateCoeffs sets
#     avgU = -mdot / gSum(rho*magSf)      with rho = patch().lookupPatchField<volScalarField>("rho")
# i.e. the REGISTERED solver rho -- the relaxed field createFields.H builds. That is the same field
# `phiHbyA = fvc::interpolate(rho)*fvc::flux(HbyA)` weights the inlet flux with, so the two cancel:
#     sum_i rho_i|Sf|_i * (-mdot / sum_j rho_j|Sf|_j)  ==  -mdot     exactly, every iteration.
# brae fed avgU thermo.rho() at the boundary while weighting the flux with the relaxed rho. With
# relaxRho = 1 those are the same field and nothing happens. With relaxRho < 1 the cancellation breaks,
# the inlet stops delivering the prescribed mass flow, and the resulting net mass imbalance drives the
# pressure LEVEL. Measured on this case before the fix:
#     inlet flux   -0.100024  -0.099690  -0.104411  -0.068195   (OF: -0.100000 at every iteration)
#     pre-limit p   102 kPa    100 kPa     163 kPa   -339 kPa    (OF: 99999.7 .. 101911.0, static)
# By iteration 5 p reached +2.34 MPa with 19600 of 28000 cells pinned on the pressureControl limits, and
# the case never converged (p L2rel 5.10e-01). After the fix brae converges in 422 iterations against
# OF's 638, at p 1.53e-03  T 7.91e-04  U 4.87e-03  rho 4.10e-03.
#
# WHY THIS CASE AND NOTHING ELSE. The defect needs BOTH a relaxed rho and a flowRateInletVelocity inlet.
# Every other compressible case in validation/ sets `relaxationFactors { fields { rho 1.0; } }`, where
# the two densities coincide; naca0012 -- the only case with rho 0.01 -- is external aero with
# freestream boundaries and no flow-rate inlet. angledDuct is the sole intersection, which is why this
# survived every existing gate.
#
# WHAT IT ASSERTS: that brae CONVERGES on this case, and to OF's answer. That is the thing the defect
# took away -- the pre-fix run reached endTime in a period-2 limit cycle at p L2rel 5.10e-01.
#
# WHY NOT A SHORT COLD-START RUN, which would be much cheaper. Measured: OF's own cold start saturates
# the pressureControl ceiling for the first dozens of iterations (`pressureControl: p max 1.47598e+06`
# at iteration 1, decaying 1.15e6 -> 835k -> 616k -> ...), and at iteration 8 OF has 17600 of 28000
# cells pinned at exactly 150000 against brae's 17200. Comparing there compares two clamped transients,
# where most of the field is the same saturated constant on both sides and the rest is path noise --
# it neither reflects the defect nor holds a tolerance. "p stays inside the limiter window" is likewise
# NOT an invariant on this case: OF violates it too. Convergence is the honest discriminator.
#
# IT ALSO ASSERTS THE INVARIANT DIRECTLY, on every time directory brae writes:
#     |sum_{f in inlet} phi_f| == massFlowRate
# This is the exact statement of the fix and it is exact arithmetic, not a tuned tolerance -- the two
# densities cancel or they do not. It goes red the moment someone feeds thermo.rho() back into the
# velocity BC, instead of waiting for the pressure to diverge hundreds of iterations later. (Measured on
# brae's own output after the fix: sum(phi_inlet) = -1.000000000000e-01 against mdot 0.1, |sum|-mdot = 0.)
set -e
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
BUILD=${BUILD:-$(cd "$(dirname "$0")/../build" && pwd)}
WORK=${WORK:-/tmp/brae_flowrate_vs_of}
# Measured at convergence (brae 422 iterations, OF 638):
#     p 1.53e-03   T 7.91e-04   U 4.87e-03   rho 4.10e-03
# 2e-2 keeps ~4x margin on the worst field (U) on a case with isoCd ~1e6, where the implicit diagonal
# and the explicit deviatoric source are both ~1e6 and nearly cancel. The pre-fix run does not reach
# this comparison at all -- it fails the INVARIANT first (checked below, before convergence, precisely
# so the diagnostic fires on a run that never converges) -- and its p was 5.10e-01.
TOL=${TOL:-2e-2}

if [ ! -f "$OFBASHRC" ]; then echo "OpenFOAM not found at $OFBASHRC -- skipping"; exit 77; fi
source "$OFBASHRC" 2>/dev/null || true
TUT="$FOAM_TUTORIALS/compressible/rhoSimpleFoam/angledDuctExplicitFixedCoeff"
[ -d "$TUT" ] || { echo "flowrate_vs_openfoam: tutorial not found at $TUT -- skipping"; exit 77; }

rm -rf "$WORK" "$WORK.br"; cp -r "$TUT" "$WORK"
rm -rf "$WORK/0"; cp -r "$WORK/0.orig" "$WORK/0"
cd "$WORK"
blockMesh > log.blockMesh 2>&1
# The gate depends on the case having both ingredients. If the tutorial changes, fail loudly rather
# than silently asserting nothing.
grep -qE '^\s*rho\s+0*\.[0-9]' system/fvSolution || {
    echo "flowrate_vs_openfoam: case no longer relaxes rho -- the defect is unreachable, gate is void"; exit 1; }
grep -q 'flowRateInletVelocity' 0/U || {
    echo "flowrate_vs_openfoam: case no longer has a flowRateInletVelocity inlet -- gate is void"; exit 1; }
sed -i 's/relTol *[0-9.eE-]*;/relTol 0;/g' system/fvSolution
# The invariant below reads brae's WRITTEN phi at 1e-9. The tutorial writes 6 digits, and brae now honours
# writePrecision as OpenFOAM does (it used to write 12 whatever the case said), so both codes write 17 here: the
# check keeps its full strength and neither code's arithmetic changes (writePrecision only formats the output).
sed -i 's/^writePrecision .*/writePrecision  17;/' system/controlDict
grep -q '^writePrecision  17;' system/controlDict || { echo "flowrate_vs_openfoam: could not set writePrecision"; exit 1; }
cp -r "$WORK" "$WORK.br"

rhoSimpleFoam > log.of 2>&1 || true
"$BUILD/brae_rhoSimpleFoam" -case "$WORK.br" > "$WORK.br/log.brae" 2>&1 || true

# ---------------------------------------------------------------- the invariant, on brae's own phi
python3 - "$WORK.br" <<'PY'
import re, sys, os, math
work = sys.argv[1]

def inlet_flux(path):
    s = open(path).read()
    s = re.sub(r'/\*.*?\*/', '', s, flags=re.S)
    b = s[s.index('boundaryField'):]
    m = re.search(r'\binlet\b\s*\{.*?value\s+nonuniform\s+List<scalar>\s*\n?\s*(\d+)\s*\((.*?)\)', b, re.S)
    if not m:
        return None, None
    v = [float(x) for x in m.group(2).split()]
    return sum(v), len(v)

# massFlowRate as the case states it, so the gate cannot drift from the input it is checking
u = open(os.path.join(work, '0', 'U')).read()
m = re.search(r'massFlowRate\s+(?:constant\s+)?([-\d.eE+]+)\s*;', u)
if not m:
    print("  invariant: case has no massFlowRate to check against"); sys.exit(1)
mdot = float(m.group(1))

times = sorted((d for d in os.listdir(work) if re.fullmatch(r'[0-9]+', d) and d != '0'), key=int)
phis = [t for t in times if os.path.exists(os.path.join(work, t, 'phi'))]
if not phis:
    print("  invariant: brae wrote no phi -- the flux is not observable, gate is weaker than it claims")
    sys.exit(1)

bad = 0
for t in phis:
    flux, n = inlet_flux(os.path.join(work, t, 'phi'))
    if flux is None:
        print(f"  t={t}: no inlet entry in phi"); bad += 1; continue
    rel = abs(abs(flux) - mdot) / mdot
    ok = rel <= 1e-9      # exact arithmetic; this is round-off headroom on a 400-face sum, not a fit
    print(f"  t={t:<5} sum(phi_inlet) {flux:+.12e}  mdot {mdot:g}  rel err {rel:.2e}  ({n} faces)  {'OK' if ok else 'FAIL'}")
    if not ok:
        bad += 1
print(f"flowrate_vs_openfoam(invariant): {bad} failures")
sys.exit(1 if bad else 0)
PY

grep -q "SIMPLE solution converged" log.of || {
    echo "flowrate_vs_openfoam: OF did NOT converge -- the oracle is not a converged field"; exit 1; }
if ! grep -q "SIMPLE solution converged" "$WORK.br/log.brae"; then
    echo "flowrate_vs_openfoam: brae did NOT converge -- this is the defect this gate exists for"
    tail -3 "$WORK.br/log.brae"
    exit 1
fi
echo "  OF   converged: $(grep -o 'converged in [0-9]* iterations' log.of | tail -1)"
echo "  brae converged: $(grep -o 'converged in [0-9]* iterations' "$WORK.br/log.brae" | tail -1)"

OFL=$(ls -d "$WORK"/[0-9]* | grep -v '/0$' | sed 's:.*/::' | sort -g | tail -1)
BRL=$(ls -d "$WORK.br"/[0-9]* | grep -v '/0$' | sed 's:.*/::' | sort -g | tail -1)

python3 - "$WORK/$OFL" "$WORK.br/$BRL" "$TOL" <<'PY'
import re, sys, math, os
ofd, brd, tol = sys.argv[1], sys.argv[2], float(sys.argv[3])

def rd(path):
    if not os.path.exists(path):
        return None
    t = open(path).read()
    m = re.search(r'internalField\s+nonuniform[^(]*\(([^)]*)\)', t, re.S)
    if not m:
        return None
    return [float(x) for x in m.group(1).replace('(', ' ').replace(')', ' ').split()]

bad = 0
for f in ("p", "T", "U", "rho"):
    of, br = rd(f"{ofd}/{f}"), rd(f"{brd}/{f}")
    if of is None or br is None:
        print(f"  {f}: FAIL missing/uniform field (OF {ofd}, brae {brd})")
        bad += 1
        continue
    n = min(len(of), len(br))
    den = math.sqrt(sum(x * x for x in of[:n]))
    l2 = math.sqrt(sum((a - b) ** 2 for a, b in zip(of[:n], br[:n]))) / den
    spread = (max(of[:n]) - min(of[:n])) / (abs(sum(of[:n]) / n) + 1e-300)
    ok = l2 <= tol and spread > 1e-6           # spread guard: refuse to pass on a flat field
    print(f"  {f:<2} L2rel {l2:.3e}  tol {tol:.0e}  spread {spread:.3e}  {'OK' if ok else 'FAIL'}")
    if not ok:
        bad += 1

# At CONVERGENCE the limiter must be off the solution -- unlike the cold transient, where OF saturates
# it too. A converged field sitting on 1.5e5 would mean the run stalled against the ceiling rather than
# solving, which is exactly what the pre-fix period-2 limit cycle did.
p = rd(f"{brd}/p")
if p:
    pmin, pmax = min(p), max(p)
    clamped = sum(1 for x in p if x >= 1.49999e5 or x <= 4.00001e4)
    print(f"  p  converged range [{pmin:.1f}, {pmax:.1f}] Pa, {clamped} cells on the limits"
          f"  {'OK' if clamped == 0 else 'FAIL -- converged onto the pressureControl ceiling'}")
    if clamped:
        bad += 1

print(f"flowrate_vs_openfoam: {bad} failures")
sys.exit(1 if bad else 0)
PY
