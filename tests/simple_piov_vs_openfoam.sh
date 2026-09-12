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
# U 1.4911e-03 / p 1.2878e-02 and FAILS both rows.
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
