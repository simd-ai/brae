#!/bin/bash
# B5: a HEAT-FLUX wall -- fixedGradient on T -- against OF v2412 rhoSimpleFoam.
#
# Every compressible gate before this one drove the energy equation with fixedValue walls only. That is
# half of what OF's basicThermo::heBoundaryTypes() dispatches on: fixedValue T -> fixedEnergy, but
# zeroGradient/fixedGradient T -> gradientEnergy. brae had no refGrad on DeviceBoundary at all, so a
# `fixedGradient` T patch was accepted by the reader and then discretised as zeroGradient -- an
# ADIABATIC wall. The case still converged and still looked physical; it was simply a different problem.
#
# Two independent things have to be right for this gate to pass, and they fail differently:
#
#   1. The laplacian's boundary source. OF's fixedGradient contributes gradientBoundaryCoeffs = g, i.e.
#      zeroGradient PLUS a source proportional to g. Missing it -> no wall heat at all. Measured here:
#      T L2rel 7.97e-03 (brae's first cell row flat at 300 K while OF ran 300.3 -> 318.5 K).
#      The subtlety that cost the first attempt: the energy equation is the ONLY caller of the
#      FACE-diffusivity variant deviceBCLaplacianCoeffsFace (it passes alphaEff per face), so patching
#      the cell-diffusivity kernel alone left the one case this gate exists for completely unheated.
#
#   2. The T -> he gradient conversion. OF's gradientEnergy sets gradient() = Cpv*snGrad(T)
#      (gradientEnergyFvPatchScalarField.C:111), NOT snGrad(T). Copying the gradient across unscaled is
#      wrong by Cpv ~ 718 J/kg/K here, which is the difference between a heat-flux wall and a nearly
#      adiabatic one -- the same failure signature as (1) but a different cause.
#
# Measured, converged, vs OF v2412 (104 iters, residualControl):
#     fixedGradient dropped : T 7.97e-03
#     implemented           : T 4.94e-08   p 4.63e-09
# 1e-6 is ~20x above the achieved value and ~5 orders below the broken one.
set -e
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
SRC=${SRC:-$(cd "$(dirname "$0")/rhoBoxQ" && pwd)}
BUILD=${BUILD:-$(cd "$(dirname "$0")/../build" && pwd)}
WORK=${WORK:-/tmp/brae_hf_vs_of}
TOL=${TOL:-1e-6}

if [ ! -f "$OFBASHRC" ]; then echo "OpenFOAM not found at $OFBASHRC -- skipping"; exit 77; fi
source "$OFBASHRC" 2>/dev/null || true

rm -rf "$WORK"; mkdir -p "$WORK"
cp -r "$SRC"/* "$WORK/"
mkdir -p "$WORK/0" && cp "$WORK"/0.orig/* "$WORK/0/"
cd "$WORK"
blockMesh > log.blockMesh 2>&1
rhoSimpleFoam > log.rhoSimpleFoam 2>&1
OFLAST=$(ls -d [0-9]* | grep -vx 0 | sort -g | tail -1)

rm -rf "$WORK.brae"; cp -r "$SRC" "$WORK.brae"
cp -r "$WORK/constant/polyMesh" "$WORK.brae/constant/"
mkdir -p "$WORK.brae/0" && cp "$WORK.brae"/0.orig/* "$WORK.brae/0/"
"$BUILD/brae_rhoSimpleFoam" -case "$WORK.brae" > "$WORK.brae/log.brae" 2>&1
BRLAST=$(ls -d "$WORK.brae"/[0-9]* | grep -v '/0$' | sort -g | tail -1)

# brae's own output must be readable BY OpenFOAM. Asserted here rather than assumed: the writer emitted
# "nonuniform List<List<scalar>>" (foamListType already carries the List<>, and it was wrapped again) on
# every scalar and vector patch, so nothing brae wrote could be restarted from or post-processed by OF.
# No gate caught it for the whole project, because gates regex the internalField and never re-read the file.
#
# The check has to read the LOG, not the exit code. OF reports "FOAM FATAL IO ERROR: incorrect first token"
# for the malformed list and then DEMOTES it to a warning -- the boundary entry is optional, so it falls
# back to a default-constructed patch field, drops the values, and postProcess still exits 0. A silent
# wrong read is exactly the failure mode this project keeps hitting, so grep for it explicitly.
( cd "$WORK.brae" && postProcess -func "mag(T)" -time "$(basename "$BRLAST")" > log.readback 2>&1 || true )
if grep -q "FATAL IO ERROR\|incorrect first token" "$WORK.brae/log.readback"; then
    echo "  FAIL OpenFOAM cannot parse the fields brae wrote to $BRLAST"
    grep -m 3 -A 2 "FATAL IO ERROR\|incorrect first token" "$WORK.brae/log.readback"
    exit 1
fi

python3 - "$WORK/$OFLAST" "$BRLAST" "$TOL" <<'PY'
import re, sys, math
ofd, brd, tol = sys.argv[1], sys.argv[2], float(sys.argv[3])

def rd(p):
    t = open(p).read()
    m = re.search(r'internalField\s+nonuniform[^(]*\((.*?)\)\s*;\s*boundaryField', t, re.S)
    return [float(x) for x in m.group(1).split()] if m else None

def rdb(path, patch):
    t = open(path).read()
    t = t[t.index('boundaryField'):]
    m = re.search(patch + r'\s*\{(.*?)\n\s{4}\}', t, re.S)
    if not m:
        return None
    v = re.search(r'value\s+nonuniform[^(]*\((.*?)\)', m.group(1), re.S)
    return [float(x) for x in v.group(1).split()] if v else None

bad, checked = 0, 0
scores = {}
for f in ("T", "p"):
    of, br = rd(f"{ofd}/{f}"), rd(f"{brd}/{f}")
    if of is None or br is None:
        print(f"  {f}: FAIL not written on both sides (OF={of is not None} brae={br is not None})")
        bad += 1
        continue
    n = min(len(of), len(br))
    denom = sum(a*a for a in of[:n])
    l2 = math.sqrt(sum((a-b)**2 for a, b in zip(of[:n], br[:n])) / denom)
    scores[f] = l2
    ok = l2 <= tol
    print(f"  {f}: L2rel {l2:.4e}  tol {tol:.0e}  OF range [{min(of):.6g}, {max(of):.6g}]  {'OK' if ok else 'FAIL'}")
    checked += 1
    if not ok:
        bad += 1

# A11: rho is a DERIVED field -- brae computes it but no 0/rho exists, so it is written from 0/T as a
# template. It used to inherit T's identity wholesale: `object T`, dimensions of temperature, and T's
# boundary types AND values, so the inlet density read back as 300 and (once B5 landed) the hot wall
# became a fixedGradient density of 20000 kg/m^4. OF writes a derived field as `calculated` + the
# computed values, which is now what brae does. The boundary check below is the sharp one: an inlet
# density of 300 against OF's 1.161 is a factor of 258.
RHO_TOL = 1e-6      # fields land at 4.8e-08 internal / 3.5e-07 at the outlet
of_rho, br_rho = rd(f"{ofd}/rho"), rd(f"{brd}/rho")
if of_rho is None or br_rho is None:
    print(f"  rho: FAIL not written on both sides (OF={of_rho is not None} brae={br_rho is not None})")
    bad += 1
else:
    n = min(len(of_rho), len(br_rho))
    l2 = math.sqrt(sum((a-b)**2 for a, b in zip(of_rho[:n], br_rho[:n])) / sum(a*a for a in of_rho[:n]))
    print(f"  rho: L2rel {l2:.4e}  tol {RHO_TOL:.0e}  OF range [{min(of_rho):.6g}, {max(of_rho):.6g}]  "
          f"{'OK' if l2 <= RHO_TOL else 'FAIL'}")
    checked += 1
    if l2 > RHO_TOL:
        bad += 1
    for patch in ("inlet", "hotWall", "outlet"):
        ob, bb = rdb(f"{ofd}/rho", patch), rdb(f"{brd}/rho", patch)
        if ob is None or bb is None:
            print(f"  rho/{patch}: FAIL no computed boundary values written "
                  f"(OF={ob is not None} brae={bb is not None}) -- a derived field must be `calculated` "
                  f"with values on every non-constraint patch, as OF writes it")
            bad += 1
            continue
        n = min(len(ob), len(bb))
        lb = math.sqrt(sum((a-b)**2 for a, b in zip(ob[:n], bb[:n])) / sum(a*a for a in ob[:n]))
        print(f"  rho/{patch}: L2rel {lb:.4e}  OF {ob[0]:.6f}  brae {bb[0]:.6f}  "
              f"{'OK' if lb <= RHO_TOL else 'FAIL'}")
        if lb > RHO_TOL:
            print(f"       a value near {rd(f'{brd}/T')[0]:.0f} here means rho is echoing the 0/T TEMPLATE "
                  f"again instead of the solved density.")
            bad += 1

# brae's rho must equal the EOS of brae's own written p and T, using OPENFOAM'S gas constant.
#
# R is written out here on purpose. An unexplained 9.06e-07 floor on rho survived even after brae was made
# to stop at OF's own iteration count, and OF's written rho missed OF's own p/(R T) by exactly 8.958e-07
# while brae's matched its own to 1e-11 -- which pointed at the CONSTANT, not at convergence. OF v2412
# computes RR = NA*k from its own rounded constants (NA = 6.022141793e23, k = 1.38065e-23) giving
# 8314.47006650545, the pre-2019 value; brae had the CODATA-2018 8314.46261815324. brae's was the more
# physically current number and that is precisely why it was wrong: the contract is to reproduce OF.
# Measured from OF itself, not from a data sheet. Fixing it took rho from 9.06e-07 to 4.82e-08 and the
# inlet boundary from 9.00e-07 to 4.61e-09.
br_p, br_T = rd(f"{brd}/p"), rd(f"{brd}/T")
if br_rho and br_p and br_T:
    R = 8314.47006650545 / 28.96
    eos = max(abs(br_rho[i] - br_p[i]/(R*br_T[i]))/br_rho[i] for i in range(len(br_rho)))
    print(f"  rho self-consistency vs p/(R T): {eos:.3e}  {'OK' if eos <= 1e-9 else 'FAIL'}")
    if eos > 1e-9:
        print("       ~9e-07 here means brae's thermoRR has drifted from OpenFOAM's 8314.47006650545")
        print("       (thermophysicalModels/thermo_parse.cuh). Nothing else produces a CONSTANT relative")
        print("       offset on every cell and every patch at once.")
        bad += 1

# The whole point is that the wall actually HEATS. If T came out uniform the L2 above would be tiny
# against a uniform OF field too, so assert the spread independently of the comparison.
of_T = rd(f"{ofd}/T")
if of_T and (max(of_T) - min(of_T)) < 5.0:
    print(f"  FAIL OF's own T spread is {max(of_T)-min(of_T):.3g} K -- the case stopped testing a heat-flux wall")
    bad += 1
if checked < 2:
    print("  FAIL fewer than 2 fields compared")
    bad += 1
if scores.get("T", 1.0) > 1e-4:
    print("  NOTE T >> 1e-4 with p fine means the ENERGY boundary specifically: either the fixedGradient")
    print("       source is missing from deviceBCLaplacianCoeffsFace (the energy equation's variant), or")
    print("       the T->he gradient is not being scaled by Cpv (gradientEnergyFvPatchScalarField.C:111).")

print(f"hf_vs_openfoam: {bad} failures over {checked} fields")
sys.exit(1 if bad else 0)
PY

# THE MIRROR ARM. Everything above gates the LEGACY binary; the OF-mirror path refused T `fixedGradient`
# outright until the he mapping learned it (rhoCreateFields_cpp.cu: gradient slots SCALE by Cpv, they
# do not go through the affine heOf -- gradientEnergy/mixedEnergyFvPatchScalarField.C, second term
# identically zero for a pureMixture). The engagement check keeps a fixture regression from silently
# turning this arm into a second copy of the plain gate.
grep -q "fixedGradient" "$WORK/0/T" || { echo "FAIL: fixture no longer carries fixedGradient T"; exit 1; }
OFLAST=$(cd "$WORK" && ls -d [0-9]* | grep -vx 0 | sort -g | tail -1)
mout=$("$BUILD/test_rho_simple_step_cpp" "$WORK" 0 "$OFLAST" 2>&1) \
    || { echo "$mout" | tail -15; echo "FAIL(mirror)"; exit 1; }
echo "$mout" | grep -E "^     T |^     U " | head -2
echo "$mout" | grep -q "^PASS" || { echo "$mout" | tail -5; echo "FAIL(mirror)"; exit 1; }
echo "PASS(mirror)"
