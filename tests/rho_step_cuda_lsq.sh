#!/usr/bin/env bash
# grad(p) RESOLVING TO leastSquares on the CUDA driver, per iteration, against the host reference -- the
# driver-level twin for every device consumer of fvc::grad(p): the momentum source -grad(p)*V, U = HbyA -
# rAtU*grad(p) (pEqn.H:86 / pcEqn.H:99), SIMPLEC's HbyA correction (pcEqn.H:30,65) and the pressure
# equation's non-orthogonal correction (correctedSnGrad.C:52-55 takes grad(p)'s own scheme). The gradient
# itself is gated against OpenFOAM's own by tests/leastsquares_grad_vs_openfoam.sh (device scalar
# 2.5e-13); the host consumers by tests/rho_leastsquares_closure_vs_openfoam.sh (grad(p) 1.7e-05 ->
# 2.9e-12 restarted); this gates the DEVICE consumers against that host, and the OpenFOAM oracle for the
# device arm is that script's lsqp arm.
#
# TWO FIXTURES, because the sites split by mesh: rhoKE (orthogonal, `laplacian orthogonal`, kEpsilon)
# exercises the momentum source and the velocity correction; sbMatched (`Gauss linear corrected`,
# transonic, kEpsilon) adds the pressure equation's non-orthogonal correction. Each with ONE change:
# `grad(p) leastSquares` named explicitly, so grad(U) (refused by this arm) is untouched.
#
# FAIL-PROOF, inside the gate: the same run with the device handed the Gauss grad(p)
# (--device-gradp-gauss, harness control) must MISS the momentum bounds.
#
# MEASURED, device against host: rhoKE 3 iterations Ux <= 1e-12-class, k 4e-13 (its own arm's floor);
# sbMatched 2 iterations Ux 3.7e-12, p 1.4e-12, k 8.0e-12, epsilon 3.4e-11, nut 2.4e-11, against
# 3.9e-12 / 1.7e-12 / 1.2e-11 shipped. The Gauss-on-the-device control reads Ux 1.4e-05 (rhoKE) and
# 4.7e-03 (sbMatched). WHAT THIS FOUND: on sbMatched at iteration 2 the two arms first read Ux 1.9e-09
# apart -- and OpenFOAM sided with the DEVICE (tests/rho_gradp_lsq_simplec_vs_openfoam.sh): the host
# fvc::snGrad's non-orthogonal correction took a hardcoded Gauss gradient where correctedSnGrad.C:52-55
# resolves grad(p)'s own entry. Fixed in fvc.cu; both arms now sit at 2e-11 of OpenFOAM there.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_rho_simple_step_cuda"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
ITERS=${ITERS:-3}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available (blockMesh)"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v blockMesh > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0

stage() {   # stage <fixture> <dest>
    rm -rf "$2"; cp -r "$ROOT/validation/$1" "$2" || exit 1
    if [ ! -f "$2/constant/polyMesh/owner" ]; then
        ( cd "$2" && blockMesh > log.blockMesh 2>&1 ) || { tail -5 "$2/log.blockMesh"; echo "FAIL: blockMesh ($1)"; exit 1; }
    fi
    [ -f "$2/constant/polyMesh/owner" ] || { echo "FAIL: no mesh ($1)"; exit 1; }
    python3 - "$2/system/fvSchemes" <<'PYEOF'
import re, sys
p = sys.argv[1]; s = open(p).read()
s2, n = re.subn(r'(gradSchemes\s*\{\s*default\s+Gauss linear;)', r'\1\n    grad(p)         leastSquares;', s)
assert n == 1, 'gradSchemes default Gauss linear not found once'
open(p, 'w').write(s2)
PYEOF
    grep -q "grad(p) *leastSquares;" "$2/system/fvSchemes" || { echo "FAIL: could not stage grad(p) leastSquares on $1"; exit 1; }
}

arm() {   # arm <fixture> <iterations> <extra harness args>
    local fx="$1" its="$2" d="$W/$1"; shift 2
    stage "$fx" "$d"
    echo "== $fx, grad(p) leastSquares ($(grep -h laplacianSchemes -A1 "$d/system/fvSchemes" | grep -o 'Gauss linear [a-z]*' | head -1)), $its iterations =="
    "$BIN" "$d" 0.orig "$its" "$@" || fail=1
    echo "== $fx control: device with the Gauss grad(p) (must FAIL on Ux) =="
    local out
    out=$("$BIN" "$d" 0.orig "$its" "$@" --device-gradp-gauss 2>&1)
    echo "$out" | grep -E "CONTROL|^\s+Ux |^\s+p  |^PASS|^FAIL" | tail -4
    if echo "$out" | grep -qE "^\s*Ux\s+[0-9.e+-]+\s+FAIL"; then
        echo "     the device with the Gauss grad(p) misses the Ux bound                        ok"
    else
        echo "     the device with the Gauss grad(p) PASSED the Ux bound -- the gate is not measuring it   FAIL"; fail=1
    fi
}

arm rhoKE     "$ITERS" --turbulent
arm sbMatched 2        --turbulent

[ $fail = 0 ] && echo "== PASSED ==" || echo "== FAILED =="
exit $fail
