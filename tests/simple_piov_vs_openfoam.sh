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
