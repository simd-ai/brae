#!/usr/bin/env bash
# THE INCOMPRESSIBLE pressureInletOutletVelocity PATH, converged, against the OpenFOAM output the fixture
# ships -- validation/piov (simpleFoam, laminar, a pressure-driven inlet with backflow allowed) against
# validation/piov_of/393, real simpleFoam's converged answer for the same case.
#
# WHY THIS EXISTS. The piov device kernel (device_boundary_flow.cu, deviceUpdatePressureInletOutletVelocity)
# and the host class (fv_patch_field.cuh, PressureInletOutletVelocityPatchField) are SHARED between the
# rhoSimpleFoam mirror and the shipped incompressible driver (device_simple_foam.cu), and until this file
# no registered test ran an incompressible piov fixture: queue item 22. Item 19 rewrote both for the
# compressible mirror (OpenFOAM's directionMixed coefficients, gated at 1e-12 on rhoTP) and the shipped
# `brae` binary on THIS case moved from the fixture's recorded U 1.15e-04 / p 1.09e-03 (validation/piov_cf,
# 378 iterations) to U 1.49e-03 / p 1.29e-02 at 394 iterations -- a change the mirror's gates could not
# see. Bisected 2026-09-03 with the host class held new: the KERNEL typing alone moves it (old kernel
# 1.1459e-04 / 1.0915e-03, exactly the fixture's record). The kernel now carries a `directionMixed` mode
# the mirror asks for and the frozen driver does not, and this gate holds the driver at its record:
# bounds ~3.5x it. Fail-proof: the directionMixed form forced on the legacy call site reads
# U 1.4911e-03 / p 1.2878e-02 and FAILS both rows -- and re-measured 2026-09-22 it is worse than that:
# the run does not reach residualControl at all within the case's cap ("brae did not report
# convergence").
#
# AND THE TWO FORMS DISAGREE IN OPPOSITE DIRECTIONS, which is what says a COMPENSATING defect is in
# here somewhere. On incompressible/pimpleFoam/RAS/TJunction -- a TRANSIENT case whose inlet is a piov,
# run laminar for ten steps of 0.002 against real pimpleFoam, every solver pinned at 1e-14/relTol 0 so
# the stopping point is out of it -- the legacy typing reads U 3.863e-03 / p 2.119e-03 and the
# directionMixed form reads U 1.165e-03 / p 1.072e-03: three times CLOSER, where on this fixture it
# does not converge. The same experiment says the TJunction gap IS the inlet: with a plain fixedValue
# inlet in both codes it falls to U 5.631e-04 / p 3.259e-05 (65x on p), and it is NOT the convection
# scheme (`Gauss upwind` for the case's limitedLinearV: 3.810e-03, unchanged) nor the closure (the
# turbulent run is 7.078e-03 against this laminar 3.863e-03).
# So the directionMixed typing is OpenFOAM's and this driver cannot take it yet: something around it --
# its flux and matrix machinery, which grew up on the legacy typing -- carries an error the legacy form
# partly cancels. Finding THAT is the unit; until then this gate holds the driver where it is.
#
# WHERE IT IS, narrowed 2026-09-22 (the next person starts here):
#   1. ON AN AXIS-ALIGNED PATCH THE TWO TYPINGS DIFFER IN EXACTLY ONE THING. The tangential components
#      are identical -- d_k = sqrt(1 - n_k^2) = 1 there, so the mixed face is a fixedValue face, and
#      bcDivKernel and bcLaplacianFaceKernel were read side by side to confirm they compute the same
#      coefficients for vf = 1 as for type 1. The NORMAL component is the difference: fixedValue at
#      n(n.U_cell) under the legacy typing, d_x = sqrt(1 - 1) = 0 -> pure zeroGradient under
#      directionMixed, which is what OpenFOAM's `neg(phip)*(I - sqr(nf()))` gives.
#   2. THE RUN DOES NOT SETTLE, it LIMIT-CYCLES. With directionMixed, Ux converges (initial residual
#      4.3e-12) and p converges (5.8e-09), while Uy's initial residual freezes at 2.74669e-04 -- the
#      same digits every iteration to 2000 -- so residualControl is never met. The field it cycles
#      around is 1.3495e-02 from OpenFOAM's converged answer in U (the legacy record is 1.1459e-04).
#   3. AND YET THE PATCH IS CLOSER. On that same run the outlet's own values are nearer OpenFOAM's than
#      the legacy typing's: max|Uy| 1.447e-01 against OpenFOAM's 1.435e-01, Ux from -1.148e-01 to
#      1.005e+00 against -1.196e-01 to 1.006e+00. The boundary is right and the interior does not
#      settle, which says the defect is in what the FREED NORMAL COMPONENT exposes -- the outlet's flux
#      and the pressure equation that now has to set it -- not in the typing.
#   4. AND IT IS THE BACKFLOW FACES, in Uy, in the cells that touch the outlet. At the fixed point the
#      whole field error is the TANGENTIAL component (|dU| == |dUy| in every one of the worst cells),
#      it sits in the outlet's own cells and decays one cell inward (1.512e-02, then 5.180e-03), and it
#      is four times larger on the faces that take flow IN than on the ones that let it out: mean
#      |dUy| 6.105e-03 over the 9 inflow faces against 1.447e-03 over the 39 outflow ones, the worst
#      four cells being the four strongest backflow faces in order. That is the vf = 1 branch of the
#      directionMixed typing -- where OpenFOAM pins the tangential velocity to zero and frees the
#      normal one -- and it is NOT the coefficients: bcDivKernel and bcLaplacianFaceKernel give a
#      vf = 1 mixed face exactly what they give a fixedValue face, read line by line to check it.
#      The residual agrees with that picture: Uy DECAYS smoothly to 2.74669e-04 over some 600
#      iterations and then holds it, each solve cutting it 13x and the next assembly restoring it --
#      a steady state of the OUTER iteration whose Uy equation carries a constant imbalance, not an
#      oscillation.
#   5. THE IMBALANCE IS NINE CELLS. Restarting that fixed point for one iteration with BRAE_DUMP_STAGE
#      writes the per-cell momentum residual the solver itself normalises (stage_mResid1 and its
#      normFactor; sum|r|/normFactor = 2.74669e-04, the reported number to every digit). Of that sum,
#      84.5% is in the 48 outlet cells and 79% is in the NINE that take flow in -- 7.5797e-05 against
#      5.0498e-06 over the 39 outflow ones -- and every one of the nine has the SAME SIGN
#      (signed sum +7.5754e-05 on inflow against -5.0498e-06 on outflow). The freed NORMAL component
#      over those same cells is exact: Ux's residual there is 7.3089e-12.
#      So it is the vf = 1 branch, the TANGENTIAL component, on INFLOW faces, and nothing else.
#      Where to look next, given the coefficients are identical to a fixedValue face at vf = 1: what
#      differs is only the two numbers the mixed slot is fed -- the per-component `d` and the `ref`
#      built from it (piovComponent, device_boundary_flow.cu). brae takes d_k = sqrt(1 - n_k^2), which
#      is OpenFOAM's snGradTransformDiag (directionMixedFvPatchField.C:180-200) -- the GRADIENT half --
#      while OF's VALUE half uses (I - valueFraction) itself, whose diagonal is 1 - n_k^2 without the
#      root. The two coincide on an axis-aligned patch, which this outlet is, so that is not yet the
#      answer -- but it is the one place the two typings are fed different arithmetic, and the next
#      step is OpenFOAM's own numbers for those nine cells (the of-instrument route).
#   6. AND OPENFOAM'S OWN NUMBERS SAY THE MOMENTUM SYSTEM IS RIGHT. Both codes were run ONE iteration
#      from OpenFOAM's converged 393 -- identical inputs -- with tools/dumpSimpleFoam (extended here to
#      dump the y and z solve systems: internalCoeffs and boundaryCoeffs are VECTORS, so the three
#      components solve three different systems and an x-only dump cannot see a defect that lives in
#      y) and brae's BRAE_DUMP_STAGE. At the nine inflow faces, EVERY piece of the y-momentum equation
#      agrees: internalCoeffs to six digits, boundaryCoeffs 0 = 0, the patch value Uy_b = 0 = 0, the
#      diagonal to 1e-10, and the assembled residual itself -- OF 1.7473e-10 against brae 1.7472e-10
#      over those cells, ratio 1.00 on every one, and 2.2860e-07 against 2.2860e-07 over the whole
#      field. So the 2.74669e-04 is not an assembly defect: it is the residual of brae's OWN fixed
#      point, which is a different one.
#   7. WHAT IS ACTUALLY WRONG IS HbyA AT THE PATCH. At that same instant, HbyA's boundary value on the
#      nine inflow faces is off by a NEARLY CONSTANT absolute amount -- 7.1e-04, 9.9e-04, ... 5.5e-04
#      where the values themselves span 2.6e-02 to 1.2e-01, so 2.78e-02 relative on the smallest face
#      and 4.2e-03 on the largest -- and phiHbyA inherits EXACTLY those percentages (2.78e-02,
#      1.47e-02, 9.09e-03, ... face for face). The pressure equation is therefore built on a flux that
#      is ~1% wrong at the backflow faces while the momentum matrix is exact, which is what walks the
#      SIMPLE iteration to a different fixed point.
#   8. AND HERE IT IS: rAU FOLDS THE BOUNDARY DIAGONAL WITH ONE COMPONENT. HbyA was compared over the
#      WHOLE field at OpenFOAM's own state: the interior agrees to round-off (median 6.6e-11, max
#      5.8e-06) and so do most outlet cells (median 5.5e-11) -- only the BACKFLOW ones are out, and
#      they are the five worst cells in the field (2.78e-02, 2.03e-02, 1.47e-02, 9.09e-03, 7.08e-03).
#      In both codes HbyA's patch value EQUALS its cell value, so it is not a patch-value defect; and
#      the momentum matrix at those cells is OpenFOAM's to round-off. What is left is rAU, and
#      device_simple_foam.cu folds the boundary diagonal into A() as
#          hasSym_ ? cmptAvIC : iC[0]
#      -- the per-component average ONLY when the mesh has a symmetry/slip U patch (hasSym_ is set by
#      isSymmetry() alone). A directionMixed pressureInletOutletVelocity has exactly the same
#      per-component asymmetry -- x zeroGradient, y and z fixed: at these faces iC.x = -5.63e-05
#      against iC.y = +8.33e-04, a factor of fifteen -- and it does not set hasSym_, so this fixture
#      folds the X component into a diagonal all three share. rAU is wrong there, HbyA inherits it,
#      phiHbyA inherits HbyA face for face, and SIMPLE settles somewhere else.
#      THAT IS THE COMPENSATION: under the LEGACY typing all three components are fixedValue with the
#      same iC, so iC[0] IS the average and none of this shows. The legacy typing is not "closer" --
#      it is hiding a diagonal that is only ever right when every component agrees.
#      THE FIX belongs with the directionMixed port (the two must land together): the fold has to key
#      on "any U patch whose boundary coefficients differ per component" -- symmetry, wedge AND
#      directionMixed -- not on isSymmetry(). A wedge mesh with no symmetry patch is the same hazard
#      and no fixture on this driver has one, so it is unmeasured rather than safe.
#
# The comparison is CONVERGED (both runs stop on the case's own residualControl), so it cannot see an
# ordering defect -- only a boundary-condition or matrix-coefficient one, which is what it is here for.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILDDIR="${BUILD:-$ROOT/build}"
BIN="$BUILDDIR/brae"
SRC="$ROOT/validation/piov"
REF="$ROOT/validation/piov_of/393"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
U_BOUND=${U_BOUND:-4e-04}
P_BOUND=${P_BOUND:-4e-03}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -d "$REF" ]      || { echo "SKIP: reference $REF missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: OpenFOAM (blockMesh) not available"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v blockMesh > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
cp -r "$SRC" "$W/c"
grep -q "pressureInletOutletVelocity" "$W/c/0/U" || { echo "FAIL: the fixture lost its pressureInletOutletVelocity patch"; exit 1; }
( cd "$W/c" && blockMesh > log.blockMesh 2>&1 ) || { tail -5 "$W/c/log.blockMesh"; echo "FAIL: blockMesh"; exit 1; }
( cd "$W/c" && "$BIN" -case "$W/c" > run.log 2>&1 ) || { tail -8 "$W/c/run.log"; echo "FAIL: brae did not run"; exit 1; }
grep -q "converged" "$W/c/run.log" || { tail -5 "$W/c/run.log"; echo "FAIL: brae did not report convergence"; exit 1; }
last=$(ls -d "$W"/c/[0-9]* | xargs -n1 basename | grep -vx 0 | sort -g | tail -1)
[ -n "$last" ] || { echo "FAIL: brae wrote no time directory"; exit 1; }
echo "  brae converged at iteration $last (OpenFOAM: 393)"

BRAE_DIR="$W/c/$last" REF_DIR="$REF" START_DIR="$W/c/0" U_BOUND="$U_BOUND" P_BOUND="$P_BOUND" python3 - <<'PIOVCMP'
import os, re, sys
import numpy as np
def read(p):
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n?(\d+)\s*\n\(\n(.*?)\n\)\s*;', s, re.S)
    if m:
        if m.group(1) == 'scalar':
            return np.array([float(x) for x in m.group(3).split()])
        return np.array([[float(c) for c in v.split()] for v in re.findall(r'\(([^)]*)\)', m.group(3))])
    u = re.search(r'internalField\s+uniform\s+\(?([^);]+)\)?;', s)
    return None if not u else np.array([float(x) for x in u.group(1).split()])
ok = True
for f, key in (('U', 'U_BOUND'), ('p', 'P_BOUND')):
    a = read(os.path.join(os.environ['BRAE_DIR'], f)); b = read(os.path.join(os.environ['REF_DIR'], f))
    bound = float(os.environ[key])
    r = float(np.linalg.norm(a - b) / np.linalg.norm(b))
    good = r < bound
    print('     %-3s relL2(brae vs OpenFOAM 393) %.4e   (bound %.1e)   %s' % (f, r, bound, 'ok' if good else 'FAIL'))
    ok = ok and good
    # NON-VACUITY: the start state must miss the bound by >= 10x, or a solver that did nothing would pass.
    s0 = read(os.path.join(os.environ['START_DIR'], f))
    if s0 is not None:
        s0 = np.broadcast_to(s0, b.shape) if s0.ndim < b.ndim or s0.shape != b.shape else s0
        r0 = float(np.linalg.norm(s0 - b) / np.linalg.norm(b))
        print('     %-3s start state %.3e (needs >= 10x the bound)   %s' % (f, r0, 'ok' if r0 >= 10 * bound else 'FAIL (vacuous)'))
        ok = ok and r0 >= 10 * bound
sys.exit(0 if ok else 1)
PIOVCMP
rc=$?
[ $rc = 0 ] && echo PASS || { echo FAIL; exit 1; }
