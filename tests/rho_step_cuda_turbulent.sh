#!/usr/bin/env bash
# THE DEVICE kEpsilon CLOSURE INSIDE THE DRIVER, per iteration, against the host reference.
#
# The --turbulent arm of tests/test_rho_simple_step_cuda.cu was registered nowhere: the device closure
# hook (rhoTurbulenceHook.cu -> kEpsilon.cu -> device_kepsilon.cu) ran in no gate at all. Its standalone
# gate (rho_kepsilon_cuda) assembles ONE system from adopted fields and cannot see the closure's
# position in the iteration or the nut/alphat -> muEff/alphaEff feedback into the next one; this can.
#
# validation/rhoKE: 3200 cells, kEpsilon, hePsiThermo, kqRWallFunction/epsilonWallFunction on two walls,
# nutkWallFunction + compressible::alphatWallFunction, calculated alphat on inlet and outlet. It ships no
# mesh (sbMatched does, at 112000 cells), so blockMesh runs here and the gate SKIPs without OpenFOAM --
# the same arrangement as rho_angledduct_vs_openfoam.sh. Three iterations from 0/, never a converged
# state: the closure runs last, so iteration 1 sees no feedback at all and iteration 2 is the first
# whose momentum and energy equations carry the device closure's own nut and alphat.
#
# Measured 2026-09-02 with the harness's tight solvers (U/he/p 1e-14, k/eps 1e-12, relTol 0):
#   before the alphat-boundary fix   k 4.2e-09  eps 4.9e-09  nut 2.2e-09 (iter 3), he 1.3e-06, T 1.2e-08
#   the closure's inlet/outlet alphat faces were left at the seed by the device (wall-masked kernel)
#   while the host and OpenFOAM (EddyDiffusivity.C:38, a whole-field assignment) recompute them.
# The bounds live in the harness (its turbulent arm), pinned at ~30x the measured floors there.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_rho_simple_step_cuda"
SRC="$ROOT/validation/rhoKE"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
ITERS=${ITERS:-3}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available (blockMesh)"; exit 77; }

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v blockMesh > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
cp -r "$SRC" "$W/c" || exit 1
( cd "$W/c" && blockMesh > log.blockMesh 2>&1 ) || { tail -5 "$W/c/log.blockMesh"; echo "FAIL: blockMesh"; exit 1; }
[ -f "$W/c/constant/polyMesh/owner" ] || { echo "FAIL: blockMesh wrote no mesh"; exit 1; }

# The fixture must still be what the header says: a wall-function kEpsilon case with calculated
# alphat off the walls, or the closure comparison below covers less than it claims.
grep -q 'kEpsilon' "$W/c/constant/turbulenceProperties" || { echo "FAIL: rhoKE is no longer kEpsilon"; exit 1; }
grep -q 'epsilonWallFunction' "$W/c/0.orig/epsilon"      || { echo "FAIL: rhoKE lost its epsilon wall function"; exit 1; }
grep -q 'calculated' "$W/c/0.orig/alphat"                 || { echo "FAIL: rhoKE lost its calculated alphat patches"; exit 1; }

rc=0
"$BIN" "$W/c" 0.orig "$ITERS" --turbulent || rc=1

# ---- nut's FLUX-CONDITIONAL patches: refused by name on this path -------------------------------
# The closure decides an inletOutlet nut face from the flux its `phi` entry names, and the compressible
# caller hands it none (Compressible::nutPhi is null here -- interFoam sets it at every site, this path
# never does). kEpsilon_cpp.cu:940-945 therefore THROWS rather than leave such a patch at the value it
# was built with, which is the defect the waterChannel gate found on the interFoam device arm: a
# zeroGradient nut inlet that kept its seed read nut 5.5e-06 and U 1.9e-05 from OpenFOAM. The refusal
# had no arm until now, and tools/default_audit.py's ledger CITES it for `Compressible comp nutPhi` --
# a cited refusal that nothing exercises is a claim, not a guarantee.
python3 - "$W/c/0.orig/nut" <<'PYNUT' || { echo "FAIL: the nut arm was not staged"; exit 1; }
import sys
p = sys.argv[1]
s = open(p).read()
a = "    outlet    { type calculated; value uniform 0; }"
assert s.count(a) == 1, "rhoKE's outlet nut is no longer the `calculated` this arm rewrites"
s = s.replace(a, "    outlet    { type inletOutlet; inletValue uniform 0; value uniform 0; }")
open(p, "w").write(s)
PYNUT
out=$("$BIN" "$W/c" 0.orig 2 --turbulent 2>&1) && { echo "FAIL: an inletOutlet nut RAN on the device path"; rc=1; }
if echo "$out" | grep -qE "flux-conditional .inletOutlet.*no flux to decide inflow by"; then
    echo "  nut inletOutlet refused by name                  ok"
else
    echo "$out" | tail -3; echo "FAIL: no nut REFUSAL fired (a log line is not a refusal)"; rc=1
fi

exit $rc
