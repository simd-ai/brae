#!/usr/bin/env bash
# brae's interFoam under kOmegaSST against REAL OpenFOAM's, on RAS/waterChannel AS SHIPPED, field by
# field and solve by solve.
#
# THE METHOD is tests/interfoam_ras_dambreak_vs_openfoam.sh's: exactly N identical FIXED steps, both
# codes read at one instant, because an adaptive step turns any difference into a clock. The tutorial's
# own deltaT (0.1) is kept; only adjustTimeStep is turned off and the write made ASCII at 15 digits.
#
# WHAT THE CASE EXERCISES that no other gated interFoam case does:
#   kOmegaSST          the ordinary incompressible model (no `density` line, so the uniform lineage):
#                      the mixture's nu as a FIELD in F1, F2 and both diffusivities, the volumetric phi,
#                      validate() at construction, omega solved before k
#   the wall distance  wallDist's cell y for F1 and F2, beside the near-wall distance the wall
#                      functions take -- two fields OpenFOAM both calls y
#   the walls          omegaWallFunction (binomial n = 2) and nutkWallFunction on 6600 wall faces, k a
#                      kqRWallFunction
#   a pattern key      `"div\(phi,(k|omega)\)" Gauss upwind;` -- fvSchemes is a dictionary, and brae's
#                      scheme parser searched for the literal key and refused the case
#   the rest           flowRateInletVelocity (volumetric), MULESCorr, GAMG on p_rgh with a GaussSeidel
#                      smoother, a 3-D extruded mesh non-orthogonal to 13.7 degrees under
#                      `Gauss linear corrected` with no non-orthogonal corrector
# The mesh is the tutorial's own: blockMesh and two extrudeMesh passes, 28000 cells.
#
# THE CONTROL is OpenFOAM's own answer with `simulationType laminar` at the same instant: 51% of U.
#
# MEASURED, ten steps of 0.1: all 20 p_rgh, 10 alpha, 10 omega and 10 k iteration counts OpenFOAM's,
# initial residuals within 9.3e-13; alpha 7.9e-13, p_rgh 7.6e-13, U 4.3e-12, k 2.4e-12, omega 3.9e-12,
# nut 3.5e-12, the wall's nut 9.2e-13 of its largest value.
#
# WHAT THE GATE FOUND. pressureInletOutletVelocity's snGrad() was (stored value - cell)*deltaCoeffs,
# where OpenFOAM's directionMixed builds it from the valueFraction and the cell and never reads the
# stored value. At construction the atmosphere holds the file's (0 0 0) over cells at (1 0 0), so
# kOmegaSST's correctNut read a shear of 1/d there and wrote nut 3.7e-05 where OpenFOAM writes
# k/omega = 3.33e-02: that patch 100% out, the FIRST p_rgh residual of the run 7.1e-03, and at t = 1
# U 3.5e-03, omega 4.0e-02, nut 2.3e-01. The laminar run of the same case was already exact (6e-13),
# which is what put the defect in the closure's construction and nowhere else.
#
# BROKEN ONCE EACH, at t = 1 (U, omega):
#   validate() skipped, the first UEqn on the file's nut          5.1e-01, 1.7e-01
#   fvm::ddt dropped from both equations                          3.8e-01, 2.2e+01, 0 of 10 omega counts
#   the wall distance doubled                                     4.1e-04, 5.1e-01
#   PBiCGStab for the case's symGaussSeidel                       1.7e-04, 3.7e-02
#   water's nu as a scalar for the mixture's field                3.5e-05, 8.5e-03
#   the laplacian's non-orthogonal correction dropped             3.7e-07, 7.2e-04
# NOT DISCRIMINATED, measured: the inline (value - cell)*deltaCoeffs on k's, omega's and the wall U's
# boundary gradient in place of the patch's snGrad() (identical to the last digit: those patches hold
# refreshed values whenever the closure reads them), and relax() not called at the case's factor of 1.
#
# PROFILE nutPatches: nut PINNED at the inlet (fixedValue 0.01) and FLUX-CONDITIONAL at the outlet
# (inletOutlet, inletValue 0.002) -- the two patch kinds kOmegaSST's field assignment does not write:
# a fixedValue's operator= is empty, and correctBoundaryConditions evaluates an inletOutlet against the
# flux. MEASURED: U 3.4e-12, nut 1.7e-12. CONTROL: the shipped patches, OpenFOAM's nut 3.2e-02 away.
# BROKEN ONCE EACH: the outlet never evaluated (as brae had it) U 9.2e-07, nut 8.5e-04; the inlet
# written by the assignment U 1.2e-01, nut 3.2e-02.
#
# PROFILE nutInletZeroGrad: the inlet's nut zeroGradient -- a patch kind kOmegaSST's closure used to skip
# entirely, so it kept the value it was built with (0) where correctBoundaryConditions gives OpenFOAM the
# cell's nut; the inlet's U fixes a value, so the viscous term reads it. MEASURED: U 3.8e-12, nut 1.0e-12.
# CONTROL: OpenFOAM with the patch pinned at 0 (brae's old answer), nut 1.4e-05 away. BROKEN (as brae
# had it): U 8.7e-05.
#
# THE DEVICE RUNS ALL THREE PROFILES, held to the same bounds -- worst of the three, ten steps: alpha
# 2.8e-12, p_rgh 2.6e-12, U 5.9e-12, k 3.3e-12, omega 4.7e-12, nut 3.5e-12, and all 20 p_rgh iteration
# counts OpenFOAM's (the case names GAMG with a GaussSeidel smoother, so that is the device's GAMG taking
# OpenFOAM's counts on a real case). THIS CASE IS WHERE FOUR DEVICE MODULES MEET: kOmegaSST, nut's
# boundary, U's inletOutlet outlet and the GAMG smoother. WHAT IT FOUND, each with the HOST closure run
# inside the device loop to say whether the loop or the closure owned it:
#   U's inletOutlet outlet assembled from the switch dbU was BUILT with -- buildDeviceVectorBoundary
#   seeds category 3 as fixedValue at its inletValue on every face, so the outlet was a wall at (0 0 0)
#   for the whole run, and OpenFOAM's updateCoeffs sets valueFraction = neg(phi) at every assembly:
#   U 8.9e-02, nut 8.7e-01 (host loop 4.3e-12 on the same case, alpha still exact to 5.3e-15, every solve
#   tightened on both codes, and 1.1e-02 of U with the schemes forced orthogonal -- so neither the
#   stopping point nor the non-orthogonal correction).
#   nut's boundary evaluated on its flux-conditional faces ONLY, where the host closure evaluates every
#   patch that is not a wall, not `empty` and not `calculated`: the zeroGradient inlet kept its built
#   value, U 1.9e-05, nut 5.5e-06, with the host closure in the same loop at 2.3e-12.
#
# PROFILE oneCorrector (and its laminar twin oneCorrectorLaminar, the control): the atmosphere's U an
# inletOutlet in place of pressureInletOutletVelocity, `nCorrectors 1`, twenty steps of 0.01 (Courant
# 0.1; at the tutorial's 0.1 one corrector runs at Courant 54 and both codes follow the same blow-up).
# This is the one shape where OpenFOAM's `updated_` lag is the step's LAST evaluate: with no momentum
# predictor the first corrector's U.correctBoundaryConditions() keeps the valueFraction the assembly
# set (mixedFvPatchField.C:234-237), and with one corrector nothing re-evaluates U before the closure and
# the next assembly read it -- on 700 of the 1400 atmosphere faces the switch has moved by then (713
# inflow faces at t = 1, none at round-off). No tutorial pairs an inletOutlet U with a corrector count
# of one; the shipped profiles cannot see it. MEASURED, both arms: alpha 1.1e-12, p_rgh 3.5e-12,
# U 5.3e-12, k 9.4e-14, omega 1.8e-12, nut 6.1e-13, all 20 p_rgh counts OpenFOAM's. WHAT IT FOUND,
# each at step 2 exactly (step 1 to 1e-13 in every field):
#   HOST: the flux pushed to U's patches after the corrector loop moved inletOutlet's valueFraction
#   where OpenFOAM's stays until the next assembly's updateCoeffs, and kOmegaSST's grad(U) read the
#   patch snGrad() with the new switch: UEqn.A 3.2e-02 in the atmosphere cells (5e-10 elsewhere),
#   U 2.7e-05, nut 2.4e-03. The laminar twin, whose closure reads nothing, stayed at 8.9e-12.
#   DEVICE: the io switch at the corrector's end taken from the new flux (the host lags it): U 1.3e-07;
#   and the boundary gradient's snGrad on an inletOutlet face formed as (stored value - cell)*dc where
#   the class is MIXED and OpenFOAM's is vf*(inletValue - cell)*dc with the CURRENT switch -- the two
#   agree only while the stored value was blended with that switch: laminar U 6.2e-09, HbyA.z 6.7e-07
#   in the atmosphere cells.
# CONTROL: OpenFOAM laminar against OpenFOAM kOmegaSST on the same staging, U 4.3e-03.
#
# NOT CLAIMED: `density variable` with kOmegaSST, F3, decayControl, a wall-function blending other than
# binomial n = 2, and a moving mesh -- all refused by name; the scalarTransport function object `s`,
# which brae does not run (it does not feed back into the flow).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_inter_waterchannel_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/RAS/waterChannel"
STEPS=${STEPS:-10}
DT=${DT:-0.1}
# the oneCorrector pair's step -- see the header
STEPS1=${STEPS1:-20}
DT1=${DT1:-0.01}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: RAS/waterChannel tutorial not found at $SRC"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

W=${KEEP_W:-$(mktemp -d)}
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v blockMesh > /dev/null 2>&1   || { echo "SKIP: blockMesh not on PATH"; exit 77; }
command -v extrudeMesh > /dev/null 2>&1 || { echo "SKIP: extrudeMesh not on PATH"; exit 77; }
command -v interFoam > /dev/null 2>&1   || { echo "SKIP: interFoam not on PATH"; exit 77; }

END=$(python3 -c "print('%.10g' % ($STEPS*float('$DT')))")
END1=$(python3 -c "print('%.10g' % ($STEPS1*float('$DT1')))")

# stage <profile>: copy the tutorial, apply the profile, fix the step, mesh it as Allrun.pre does, run
stage()
{
    local profile="$1"
    local C="$W/$profile"
    # the oneCorrector pair runs at its own step: one corrector at the tutorial's 0.1 is Courant 54
    local steps="$STEPS" dt="$DT"
    case "$profile" in oneCorrector|oneCorrectorLaminar) steps="$STEPS1" dt="$DT1" ;; esac
    rm -rf "$C"
    cp -r "$SRC" "$C" || return 1
    rm -rf "$C"/[1-9]* "$C"/0 "$C"/processor* "$C"/log.*
    cp -r "$C/0.orig" "$C/0"
    grep -q "RASModel  *kOmegaSST;" "$C/constant/turbulenceProperties" \
        || { echo "FAIL: the tutorial no longer names kOmegaSST"; return 1; }
    grep -q "^density " "$C/constant/turbulenceProperties" \
        && { echo "FAIL: the tutorial now carries a \`density\` line, so the lineage is another"; return 1; }
    grep -q '"div\\(phi,(k|omega)\\)"' "$C/system/fvSchemes" \
        || { echo "FAIL: the tutorial no longer names div(phi,k) through a pattern key"; return 1; }
    if [ "$profile" = laminar ] || [ "$profile" = oneCorrectorLaminar ]; then
        sed -i 's/^simulationType .*/simulationType laminar;/' "$C/constant/turbulenceProperties"
    fi
    # oneCorrector: the atmosphere's U an inletOutlet, one pressure corrector -- see the header
    if [ "$profile" = oneCorrector ] || [ "$profile" = oneCorrectorLaminar ]; then
        python3 - "$C" <<'PYEOF' || { echo "FAIL: the $profile profile was not staged"; return 1; }
import sys
d = sys.argv[1]
p = d + '/0/U'
s = open(p).read()
a = ('    atmosphere\n    {\n        type            pressureInletOutletVelocity;\n'
     '        value           uniform (0 0 0);\n    }')
b = ('    atmosphere\n    {\n        type            inletOutlet;\n        inletValue      uniform (0 0 0);\n'
     '        value           uniform (0 0 0);\n    }')
assert s.count(a) == 1, 'the tutorial no longer has the pressureInletOutletVelocity atmosphere this edit expects'
open(p, 'w').write(s.replace(a, b))
p = d + '/system/fvSolution'
s = open(p).read()
assert s.count('    nCorrectors     2;') == 1, 'the tutorial no longer asks for nCorrectors 2'
open(p, 'w').write(s.replace('    nCorrectors     2;', '    nCorrectors     1;'))
PYEOF
    fi
    # nutPatches: nut pinned at the inlet (fixedValue) and flux-conditional at the outlet (inletOutlet),
    # the two patch kinds kOmegaSST's field assignment does not write
    # nutInletZeroGrad: the inlet's nut zeroGradient, which correctBoundaryConditions sets to the cell's
    # nut; nutInletZero, its control: the same patch pinned at 0, which is what brae's closure left there
    if [ "$profile" = nutInletZeroGrad ] || [ "$profile" = nutInletZero ]; then
        PROFILE="$profile" python3 - "$C" <<'PYEOF' || { echo "FAIL: the $profile profile was not staged"; return 1; }
import os, sys
p = sys.argv[1] + '/0/nut'
s = open(p).read()
i = s.index('    ".*"')
body = ('        type            zeroGradient;\n' if os.environ['PROFILE'] == 'nutInletZeroGrad'
        else '        type            fixedValue;\n        value           uniform 0;\n')
s = s[:i] + '    inlet\n    {\n' + body + '    }\n\n' + s[i:]
open(p, 'w').write(s)
PYEOF
    fi
    # limitedLinear: `Gauss limitedLinear 1` for k and omega. The closures have carried the scheme
    # since they were ported and no interFoam tutorial names it, so nothing gated it. THIS PROFILE
    # GATES FIELDS, not matrix coefficients: where a face's two cells hold the same omega, gradf is
    # EXACTLY zero, NVDTVD takes its 1000x branch and r turns on sign(gradcf) -- which is 1e-18
    # there. brae's omega and grad(omega) are OpenFOAM's to 7.3e-15 and 5.7e-15 and the limiter still
    # lands the other way on 2 of 79,800 faces, each worth an O(1) coefficient. The fields are not
    # affected: host omega 7.2e-12, k 5.9e-11, U 4.2e-12. The DEVICE closure refuses the scheme by
    # name (omega 1.7822e-04) and its arm is skipped.
    if [ "$profile" = limitedLinear ]; then
        python3 - "$C" <<'PYEOF' || { echo "FAIL: the limitedLinear profile was not staged"; return 1; }
import sys
p = sys.argv[1] + '/system/fvSchemes'
s = open(p).read()
old = '"div\\(phi,(k|omega)\\)"      Gauss upwind;'
assert s.count(old) == 1, 'the tutorial no longer names div(phi,(k|omega)) Gauss upwind'
open(p, 'w').write(s.replace(old, '"div\\(phi,(k|omega)\\)"      Gauss limitedLinear 1;'))
PYEOF
        grep -q "Gauss limitedLinear 1" "$C/system/fvSchemes" \
            || { echo "FAIL: the limitedLinear profile did not reach div(phi,k)"; return 1; }
    fi
    # PROFILE pbicg: `solver PBiCG; preconditioner DILU;` for k and omega, where the tutorial names a
    # symGaussSeidel smoothSolver. NEITHER ARM READ IT. The host SST driver set
    # `which.smoothSolver = true` whatever fvSolution gave and the device branch set gsK and gsOmega
    # the same way, so a case naming PBiCG ran symGaussSeidel sweeps under PBiCG's tolerance and said
    # nothing -- the substitution the kEpsilon twin was fixed for on waves/mangroveInteraction, in the
    # twin nobody had looked at. The reader refused PBiCG for k and omega under SST on top of that,
    # which is what kept it invisible: the case could not run at all, so no gate could hold it.
    # MEASURED with the substitution put back, both arms: k 1.2575e-01, omega 1.3608e-01, nut
    # 3.0634e-02, U 4.4383e-04, alpha 1.9940e-05, 4 of 10 omega counts, 22 failures. With the entry
    # read: host k 2.4249e-12 / omega 2.9758e-12, device 2.1456e-12 / 4.0655e-12, every count
    # OpenFOAM's on both.
    # `correctWallsOff`: fvSchemes' `wallDist { correctWalls no; }`. meshWavePatchDistMethod.C:59 reads it
    # (default true) and hands it to patchWave(mesh, patchIDs, correctWalls_); patchWave::correct()
    # (patchWave.C:178-236) always runs the wave and takes the cell value from it, and only THEN,
    # `if (correctWalls_)`, overwrites the wall-adjacent cells with the exact distance to the face
    # POLYGON. So `no` leaves those cells on the wave's face-CENTRE distance. brae hardcoded the
    # correction and REFUSED the entry. Its control is the `sst` run, which is the same case with the
    # correction on -- so the only difference is y on the cells that touch a wall.
    # `wallCoeffs`: the WALL PATCH names its OWN Cmu, kappa, E and beta1 on nut and omega. OpenFOAM builds
    # `wallFunctionCoefficients` from the PATCH dictionary in every wall function
    # (wallFunctionCoefficients.C:68-80), deriving yPlusLam from that patch's kappa and E, and
    # omegaWallFunction reads its own `beta1` there too (omegaWallFunctionFvPatchScalarField.C:408). None
    # of them comes from the model dictionary. brae's kOmegaSST closure used model-wide values and the
    # reader REFUSED any patch that named its own -- refusing a case OpenFOAM runs. Its control is the
    # `sst` run, which is the same case at the defaults.
    # `splitGrad`: grad(k) and grad(omega) name DIFFERENT schemes. Under kOmegaSST this is the only arm
    # that exercises CDkOmega, which is `(2*alphaOmega2)*(fvc::grad(k) & fvc::grad(omega))/omega`
    # (kOmegaSSTBase.C:555-558) -- TWO independent lookups in ONE expression, which is why a single flag
    # chosen per call cannot serve. This case also ships a `corrected` laplacian on a 13.7 deg
    # non-orthogonal mesh, so the deferred non-orthogonal correction reads the gradients too.
    #
    # grad(k) is written explicitly as the case`s own `Gauss linear`, so one thing changes. MEASURED,
    # OpenFOAM against OpenFOAM at this gate`s own end: omega 9.414e-02 over all 28,000 cells, nut
    # 8.695e-02, k 6.269e-03; and the mirror direction omega 9.741e-02. The control is the `sst` run.
    if [ "$profile" = splitGrad ]; then
        python3 - "$C" <<'WGEOF' || { echo "FAIL: the splitGrad profile was not staged"; return 1; }
import re, sys
q = sys.argv[1] + "/system/fvSchemes"
t = open(q).read()
m = re.search(r"gradSchemes\s*\{([^}]*)\}", t)
assert m, "no gradSchemes block"
assert "grad(k)" not in m.group(1), "the tutorial names grad(k) already"
t = t[:m.end(1)] + "    grad(k)         Gauss linear;\n    grad(omega)     leastSquares;\n" + t[m.end(1):]
open(q, "w").write(t)
WGEOF
        grep -q "grad(omega)     leastSquares;" "$C/system/fvSchemes" \
            && grep -q "grad(k)         Gauss linear;" "$C/system/fvSchemes" \
            && grep -q "default         Gauss linear;" "$C/system/fvSchemes" \
            || { echo "FAIL: the gradient split was not staged, or it changed the default too"; return 1; }
    fi
    if [ "$profile" = wallCoeffs ]; then
        python3 - "$C" <<'WCEOF' || { echo "FAIL: the wallCoeffs profile was not staged"; return 1; }
import re, sys
d = sys.argv[1]
for fld, extra in (("nut", ""), ("omega", "        beta1           0.08;\n")):
    q = d + "/0/" + fld
    t = open(q).read()
    # every wall-function entry in the file gets the patch's own coefficients
    pat = r"(type\s+(?:nutk|omega)WallFunction;\n)"
    t, n = re.subn(pat, r"\1        Cmu             0.085;\n        kappa           0.40;\n"
                        r"        E               9.0;\n" + extra, t)
    assert n >= 1, "no wall function entry in 0/%s" % fld
    open(q, "w").write(t)
    print("  %s: %d wall-function entries given their own coefficients" % (fld, n))
WCEOF
        grep -q "Cmu             0.085;" "$C/0/nut" && grep -q "beta1           0.08;" "$C/0/omega" \
            || { echo "FAIL: the patch coefficients were not written"; return 1; }
    fi
    if [ "$profile" = correctWallsOff ]; then
        python3 - "$C" <<'CWEOF' || { echo "FAIL: the correctWallsOff profile was not staged"; return 1; }
import re, sys
q = sys.argv[1] + "/system/fvSchemes"
t = open(q).read()
m = re.search(r"wallDist\s*\{([^}]*)\}", t)
assert m, "no wallDist block to add correctWalls to"
assert "correctWalls" not in m.group(1), "the tutorial names correctWalls already"
t = t[:m.end(1)] + "    correctWalls    no;\n" + t[m.end(1):]
open(q, "w").write(t)
CWEOF
        grep -q "correctWalls    no;" "$C/system/fvSchemes" \
            || { echo "FAIL: correctWalls was not written"; return 1; }
    fi
    if [ "$profile" = pbicg ]; then
        python3 - "$C" <<'PYEOF' || { echo "FAIL: the pbicg profile was not staged"; return 1; }
import re, sys
p = sys.argv[1] + '/system/fvSolution'
s = open(p).read()
old = '''    "(U|k|omega|s).*"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        nSweeps         1;
        tolerance       1e-6;
        relTol          0.1;
    };'''
assert s.count(old) == 1, 'the tutorial no longer names one smoothSolver for U, k and omega'
new = '''    "U.*"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        nSweeps         1;
        tolerance       1e-6;
        relTol          0.1;
    };

    "(k|omega|s).*"
    {
        solver          PBiCG;
        preconditioner  DILU;
        tolerance       1e-6;
        relTol          0.1;
    };'''
open(p, 'w').write(s.replace(old, new))
PYEOF
        grep -q "solver          PBiCG" "$C/system/fvSolution" \
            || { echo "FAIL: the pbicg profile did not name PBiCG for the closure"; return 1; }
        grep -q "preconditioner  DILU" "$C/system/fvSolution" \
            || { echo "FAIL: the pbicg profile did not name DILU"; return 1; }
    fi
    if [ "$profile" = nutPatches ]; then
        python3 - "$C" <<'PYEOF' || { echo "FAIL: the nutPatches profile was not staged"; return 1; }
import sys
p = sys.argv[1] + '/0/nut'
s = open(p).read()
i = s.index('    ".*"')
s = s[:i] + ('    inlet\n    {\n        type            fixedValue;\n        value           uniform 0.01;\n    }\n\n'
             '    outlet\n    {\n        type            inletOutlet;\n        inletValue      uniform 0.002;\n'
             '        value           uniform 0;\n    }\n\n') + s[i:]
open(p, 'w').write(s)
PYEOF
    fi

    STEPS="$steps" DT="$dt" python3 - "$C" <<'PYEOF' || { echo "FAIL: staging $profile"; return 1; }
import os, re, sys
d = sys.argv[1]
n = int(os.environ['STEPS'])
dt = os.environ['DT']
c = os.path.join(d, 'system/controlDict')
s = open(c).read()
# the function objects write nothing the gate reads, and `s` is a scalarTransport brae does not run
s = re.sub(r'\nfunctions\s*\{.*\n\}\s*\n', '\n', s, flags=re.S)
for key, val in [('adjustTimeStep', 'no'), ('deltaT', dt), ('endTime', '%.10g' % (n*float(dt))),
                 ('writeControl', 'timeStep'), ('writeInterval', str(n)), ('writeFormat', 'ascii'),
                 ('writePrecision', '15')]:
    s, k = re.subn(r'^%s\s.*' % key, '%s %s;' % (key.ljust(15), val), s, flags=re.M)
    assert k == 1, key
s = re.sub(r'^writeCompression\s.*', 'writeCompression off;', s, flags=re.M)
open(c, 'w').write(s)
PYEOF
    ( cd "$C" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh [$profile]"; tail -20 "$C/log.blockMesh"; return 1; }
    local i
    for i in 1 2; do
        cp "$C/system/extrudeMeshDict.$i" "$C/system/extrudeMeshDict"
        ( cd "$C" && extrudeMesh > "log.extrudeMesh.$i" 2>&1 ) \
            || { echo "FAIL: extrudeMesh $i [$profile]"; tail -20 "$C/log.extrudeMesh.$i"; return 1; }
    done
    ( cd "$C" && setFields > log.setFields 2>&1 ) || { echo "FAIL: setFields [$profile]"; tail -20 "$C/log.setFields"; return 1; }
    ( cd "$C" && interFoam > log.interFoam 2>&1 ) || { echo "FAIL: interFoam [$profile]"; tail -30 "$C/log.interFoam"; return 1; }
    local end
    end=$(python3 -c "print('%.10g' % ($steps*float('$dt')))")
    [ -d "$C/$end" ] || { echo "FAIL: OpenFOAM wrote no $end directory [$profile]"; ls "$C"; return 1; }
    echo "OpenFOAM ran $steps steps of deltaT $dt to t = $end   [$profile]"
}

rc=0
for p in laminar sst nutPatches nutInletZeroGrad nutInletZero oneCorrector oneCorrectorLaminar \
         limitedLinear pbicg correctWallsOff wallCoeffs splitGrad; do
    stage "$p" || { rc=1; break; }
done
[ $rc = 0 ] || { echo "interfoam_waterchannel_vs_openfoam: staging failed"; exit 1; }

# the oracle took the path
grep -q "Selecting RAS turbulence model kOmegaSST" "$W/sst/log.interFoam" \
    || { echo "FAIL: OpenFOAM's log does not select kOmegaSST"; exit 1; }
grep -q "Solving for omega" "$W/sst/log.interFoam" \
    || { echo "FAIL: OpenFOAM's log never solves omega"; exit 1; }
grep -q "Solving for omega" "$W/laminar/log.interFoam" \
    && { echo "FAIL: OpenFOAM's laminar control solved omega"; exit 1; }

"$BIN" "$W/sst" "$W/sst/0" "$W/sst/$END" "$STEPS" "$W/sst/log.interFoam" "$W/laminar/$END" || rc=1
# ...and the closure convected with `Gauss limitedLinear 1`. Its control is the LAMINAR run, as the
# shipped profile's is: what the model is worth on this case, against how far brae is from OpenFOAM.
# MEASURED, both arms where both run: host alpha 1.5e-12, p_rgh 1.4e-12, U 4.2e-12, k 5.9e-11,
# omega 7.2e-12, nut 2.4e-12. BROKEN ONCE: the scheme READ but not applied (the closure handed
# `false`) -- omega 1.2992e-01, k 1.0489e-01, nut 3.7393e-02, U 2.6309e-04.
# ...with TWO controls: the laminar run (what the model is worth) and the `sst` profile's own
# OpenFOAM output, which is THIS case under `Gauss upwind` at the same instant -- what the SCHEME is
# worth. MEASURED at one step: the scheme moves OpenFOAM's own omega by 2.56e-02 and k by 4.01e-03,
# against brae's 7.2e-12 from OpenFOAM, so the profile is not measuring nothing.
"$BIN" "$W/limitedLinear" "$W/limitedLinear/0" "$W/limitedLinear/$END" "$STEPS" \
       "$W/limitedLinear/log.interFoam" "$W/laminar/$END" "$W/sst/$END" 1e-6 || rc=1
# ...and the SOLVER THE CASE NAMES for the closure: PBiCG with DILU on k and omega, where the
# tutorial names a symGaussSeidel smoothSolver. Both arms ran the smoothSolver whatever fvSolution
# said; the reader refused the combination on top of that, so nothing could hold it. Its controls are
# the profile's own: the laminar run (the closure is live at all) and the `sst` run at the tutorial's
# own solver -- which is a DIFFERENT answer, because a substituted solver at the same tolerance stops
# somewhere else, and that is the whole point of reading the entry.
"$BIN" "$W/pbicg" "$W/pbicg/0" "$W/pbicg/$END" "$STEPS" "$W/pbicg/log.interFoam" \
       "$W/laminar/$END" "$W/sst/$END" || rc=1
# ...and `wallDist { correctWalls no; }`, whose control is the `sst` run: the same case with the
# correction on, so the only thing between them is y on the wall-adjacent cells.
"$BIN" "$W/correctWallsOff" "$W/correctWallsOff/0" "$W/correctWallsOff/$END" "$STEPS" \
       "$W/correctWallsOff/log.interFoam" "$W/laminar/$END" "$W/sst/$END" || rc=1
# ...and the wall patch's OWN Cmu/kappa/E/beta1, whose control is the `sst` run at the defaults.
"$BIN" "$W/wallCoeffs" "$W/wallCoeffs/0" "$W/wallCoeffs/$END" "$STEPS" \
       "$W/wallCoeffs/log.interFoam" "$W/laminar/$END" "$W/sst/$END" || rc=1
# ...and the GRADIENT SPLIT, the only arm that exercises CDkOmega. Control: the `sst` run.
"$BIN" "$W/splitGrad" "$W/splitGrad/0" "$W/splitGrad/$END" "$STEPS" \
       "$W/splitGrad/log.interFoam" "$W/laminar/$END" "$W/sst/$END" || rc=1
"$BIN" "$W/nutPatches" "$W/nutPatches/0" "$W/nutPatches/$END" "$STEPS" "$W/nutPatches/log.interFoam" \
       "$W/laminar/$END" "$W/sst/$END" || rc=1
# the control is OpenFOAM's answer with the patch pinned at 0: nut 1.4e-05 away, a floor of 1e-6 (six
# orders above the round-off both codes reach)
"$BIN" "$W/nutInletZeroGrad" "$W/nutInletZeroGrad/0" "$W/nutInletZeroGrad/$END" "$STEPS" \
       "$W/nutInletZeroGrad/log.interFoam" "$W/laminar/$END" "$W/nutInletZero/$END" 1e-6 || rc=1
# the switch has to have MOVED on the atmosphere for this profile to test anything: the fail-proof for
# a staging that lost the inletOutlet atmosphere or the corrector count
python3 - "$W/oneCorrector/$END1/phi" <<'PYEOF' || rc=1
import re, sys
s = open(sys.argv[1]).read()
m = re.search(r"atmosphere\s*\{[^}]*?value\s+nonuniform List<scalar>\s*(\d+)\s*\(([^)]*)\)", s, re.S)
v = [float(x) for x in m.group(2).split()] if m else []
n = sum(1 for x in v if x < 0)
ok = len(v) > 0 and n > 100 and len(v) - n > 100
print("  oneCorrector: %d of %d atmosphere faces inflow at the end -- %s" % (n, len(v), "ok" if ok else "FAIL: the switch never moved"))
sys.exit(0 if ok else 1)
PYEOF
grep -q "nCorrectors     1;" "$W/oneCorrector/system/fvSolution" || { echo "FAIL: oneCorrector was not staged with one corrector"; rc=1; }
"$BIN" "$W/oneCorrector" "$W/oneCorrector/0" "$W/oneCorrector/$END1" "$STEPS1" \
       "$W/oneCorrector/log.interFoam" "$W/oneCorrectorLaminar/$END1" || rc=1

echo "interfoam_waterchannel_vs_openfoam: rc $rc"
exit $rc
