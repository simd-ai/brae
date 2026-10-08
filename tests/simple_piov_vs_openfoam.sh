#!/usr/bin/env bash
# THE INCOMPRESSIBLE pressureInletOutletVelocity PATH, converged, against the OpenFOAM output the fixture
# ships -- validation/piov (simpleFoam, laminar, a pressure-driven inlet with backflow allowed) against
# validation/piov_of/393, real simpleFoam's converged answer for the same case.
#
# WHY THIS EXISTS, and what it finally found (2026-09-22). The piov device kernel
# (device_boundary_flow.cu, deviceUpdatePressureInletOutletVelocity) and the host class
# (fv_patch_field.cuh, PressureInletOutletVelocityPatchField) are SHARED between the rhoSimpleFoam
# mirror and the shipped incompressible driver (device_simple_foam.cu), and until this file no
# registered test ran an incompressible piov fixture: queue item 22. Item 19 rewrote both for the
# compressible mirror -- OpenFOAM's directionMixed coefficients, gated at 1e-12 on rhoTP -- and the
# shipped binary on THIS case got WORSE with them, from U 1.15e-04 to 1.49e-03 and later to no
# convergence at all. That looked like the legacy typing being "closer to OpenFOAM". It was not.
#
# THE TWO WERE A COMPENSATING PAIR, and both halves are fixed now:
#   1. THE TYPING. Every inflow component was typed fixedValue at n(n.U_cell). OpenFOAM's piov is a
#      directionMixed whose valueFraction is neg(phi)*(I - sqr(nf())): the NORMAL component is free
#      (zeroGradient) and only the tangential ones are fixed
#      (pressureInletOutletVelocityFvPatchVectorField.C, directionMixedFvPatchField.C:139-155).
#   2. THE SHARED DIAGONAL. device_simple_foam.cu folded the boundary diagonal into A() as
#      `hasSym_ ? cmptAvIC : iC[0]` -- the per-component average ONLY when a symmetry/slip patch
#      exists. A directionMixed piov has the same per-component asymmetry (at these backflow faces
#      iC.x = -5.63e-05 against iC.y = +8.33e-04, a factor of fifteen) and set no such flag, so the X
#      component was folded into a diagonal all three share. With the LEGACY typing all three agree,
#      so iC[0] IS the average and the defect could not be seen. The fold now keys on hasCmptBC_ --
#      symmetry, wedge OR piov -- and H()'s matching bdDiag term with it.
#
# HOW IT WAS FOUND, because the route matters more than the fix: the gap first showed on
# incompressible/pimpleFoam/RAS/TJunction (U 7.1e-03 against real pimpleFoam). A laminar twin, every
# solver pinned at 1e-14/relTol 0, and `Gauss upwind` for the case's limitedLinearV each moved it
# nowhere; a plain fixedValue inlet moved it to 5.6e-04, which named the piov patch. Then both codes
# were run ONE iteration from OpenFOAM's converged 393 with tools/dumpSimpleFoam (extended to dump the
# y and z SOLVE systems: internalCoeffs and boundaryCoeffs are VECTORS, so an x-only dump cannot see a
# defect that lives in y) against brae's BRAE_DUMP_STAGE. That said the momentum matrix was OpenFOAM's
# to round-off -- residual 1.7473e-10 against 1.7472e-10 over the nine backflow cells -- while HbyA was
# exact everywhere in the field (median 6.6e-11) EXCEPT those same cells (2.78e-02), with phiHbyA
# inheriting its percentages face for face. Matrix right, HbyA wrong, patch value equal to the cell
# value in both codes: the only ingredient left was rAU, and rAU is where the fold lives.
#
# MEASURED, both halves in: brae converges at iteration 393 -- OpenFOAM's own count -- and reads
# U 4.0249e-09 and p 2.1312e-08 against OpenFOAM's converged answer, from 1.1459e-04 and 1.0915e-03.
# The bounds below are that measurement.
# BROKEN ONCE EACH: the fold back to `hasSym_ ? ... : iC[0]` with the typing kept -- the run DIVERGES
# at iteration 417 (non-finite p and Ux); the legacy typing back with the fold fixed -- U 1.1477e-04
# and p 1.0931e-03, five orders outside the bounds, which is the old answer to the digit.
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
# THE BOUNDS ARE THE MEASUREMENT, and they moved by five orders when the compensation was resolved
# (see the header): brae now converges at iteration 393, OpenFOAM's own count, and reads
# U 4.0249e-09 / p 2.1312e-08 against its converged answer. Pinned at ~5x that.
U_BOUND=${U_BOUND:-2e-08}
P_BOUND=${P_BOUND:-1e-07}

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
