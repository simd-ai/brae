#!/usr/bin/env bash
# THE DEVICE kEpsilon CLOSURE'S fvm::ddt UNDER `ddtSchemes default Euler`, inside the driver, per
# iteration, against the host reference -- the driver-level twin of test_rho_kepsilon_cuda's Euler arm.
#
# kEpsilon.C:254,275 carry fvm::ddt(alpha, rho, epsilon|k) because the model is shared with the transient
# solvers: an empty matrix under steadyState (every fixture in the tree), rho*V/deltaT on the diagonal
# and rho.oldTime()*psi.oldTime()*V/deltaT in the source under Euler -- what gasMixing/injectorPipe
# ships. What the standalone arm cannot see is rho.oldTime(): the closure is handed it by the STEP, and
# the rule is the process's first iteration takes the closure-time rho and every later one the rho the
# iteration started with (RhoStepInput::firstIteration, measured on the host arm against OpenFOAM's own
# restart: rhoSimpleFoam_cpp.cuh). Iterations 2 and 3 here are where the second half of that rule runs.
#
# validation/rhoKE, as rho_step_cuda_turbulent.sh stages it, with ONE change: ddtSchemes default Euler
# (deltaT 1 -> rDeltaT 1). The steady arm next door is the control that the change is what is measured.
# validation/rhoSST (kOmegaSST, kOmegaSSTBase.C:572,602 -- the same term) under the same change; its
# steady control is measured here too, since no registered arm runs the device SST inside the driver.
#
# MEASURED, 3 iterations, device against host: k 4.4e-13, epsilon 3.1e-12, nut 1.8e-11 (the steady arm
# reads 4.5e-13 / 3.1e-12 / 1.8e-11), Ux 2.6e-13, p 2.4e-13. FAIL-PROOF, run below as part of the gate:
# the device closure handed rDeltaT 0 under the same case reads k 7.87e-05, epsilon 1.81e-04, nut
# 9.65e-05 -- eight orders above the bounds. The standalone twin (test_rho_kepsilon_cuda, Euler arm)
# holds the closure to 1e-15 and pins rho.oldTime() as the caller's field (rhoOld := rho reads 1.79e-05).
# kOmegaSST on rhoSST: k 3.5e-13, omega 3.1e-12, nut 1.2e-11 under Euler against 3.5e-13 / 3.1e-12 /
# 1.2e-11 under the fixture's own steadyState (the control run at the end); with the device term
# withheld k 8.64e-05, omega 1.02e-04, nut 9.89e-05.
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
fail=0

stage() {   # stage <fixture> <dest> <ddt: Euler|steadyState>
    rm -rf "$2"; cp -r "$ROOT/validation/$1" "$2" || exit 1
    if [ ! -f "$2/constant/polyMesh/owner" ]; then
        ( cd "$2" && blockMesh > log.blockMesh 2>&1 ) || { tail -5 "$2/log.blockMesh"; echo "FAIL: blockMesh ($1)"; exit 1; }
    fi
    [ -f "$2/constant/polyMesh/owner" ] || { echo "FAIL: no mesh ($1)"; exit 1; }
    sed -i "s/default *steadyState;/default $3;/" "$2/system/fvSchemes"
    grep -q "default *$3;" "$2/system/fvSchemes" || { echo "FAIL: could not stage ddtSchemes $3 on $1"; exit 1; }
}

arm() {   # arm <fixture> <model-grep>
    local fx="$1" model="$2" d="$W/$1"
    stage "$fx" "$d" Euler
    grep -q "$model" "$d/constant/turbulenceProperties" || { echo "FAIL: $fx is no longer $model"; exit 1; }
    echo "== $fx ($model), ddtSchemes default Euler (deltaT $(grep -E '^deltaT' "$d/system/controlDict" | awk '{print $2}' | tr -d ';')), $ITERS iterations =="
    "$BIN" "$d" 0.orig "$ITERS" --turbulent || fail=1

    # THE FAIL-PROOF, inside the gate: the same run with the device closure's fvm::ddt withheld
    # (--device-ddt-off, harness control) must MISS the closure bounds, or a device that dropped the
    # term would pass this file.
    echo "== $fx control: device closure WITHOUT fvm::ddt (must FAIL on k) =="
    local out
    out=$("$BIN" "$d" 0.orig "$ITERS" --turbulent --device-ddt-off 2>&1)
    echo "$out" | grep -E "CONTROL|closure output\)|^PASS|^FAIL" | tail -5
    if echo "$out" | grep -qE "^\s*k \(closure output\)\s+[0-9.e+-]+\s+FAIL"; then
        echo "     the device WITHOUT the term misses the k bound                                ok"
    else
        echo "     the device WITHOUT the term PASSED the k bound -- the gate is not measuring it   FAIL"; fail=1
    fi
}

arm rhoKE  kEpsilon
arm rhoSST kOmegaSST

# The SST steady CONTROL: the same fixture under its shipped steadyState, so the Euler numbers above
# are read against a floor measured on the same closure and driver (no registered arm runs it).
stage rhoSST "$W/sstSteady" steadyState
echo "== rhoSST (kOmegaSST), ddtSchemes default steadyState (control floor), $ITERS iterations =="
"$BIN" "$W/sstSteady" 0.orig "$ITERS" --turbulent || fail=1

[ $fail = 0 ] && echo "== PASSED ==" || echo "== FAILED =="
exit $fail
