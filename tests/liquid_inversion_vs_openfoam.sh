#!/usr/bin/env bash
# brae's H2O he->T inversion against OpenFOAM's OWN answer -- the PATH, not just the root.
#
# OpenFOAM's species::thermo<>::T (thermoI.H:43-88) is a do-while that stops when the TEMPERATURE STEP
# falls below T0*tol_, with tol_ = 1e-4 fixed from the INITIAL guess (thermo.C:33). It does not iterate
# to convergence, so the temperature it returns depends on where it started. Measured by tools/liqref at
# p = 1e5 with the true answer 400 K:
#
#     T0 = 400    -> 400                    T0 = 450 -> 400.00000000125675
#     T0 = 400.1  -> 400.00000000002609     T0 = 250 -> 400.00000002381233
#     T0 = 350    -> 400.00000000028433     T0 = 550 -> 400.00000000001796
#
# Six starting guesses, six answers. brae used to drive Newton to a 1e-12 energy residual, which returns
# ONE answer for all six -- arguably the better numeric, and a different function from OpenFOAM's.
#
# THE ORACLE is tools/liqref, an OpenFOAM utility in this repo that constructs the same mixture
# liquidThermo.H builds for `properties liquid` + sensibleInternalEnergy and calls OpenFOAM's own
# TEs/THs over a grid of (p, Ttrue, T0). Nothing in the table is brae's arithmetic.
#
#   ARM 1  every row reproduced, brae vs OpenFOAM, at 1e-12 relative
#   ARM 2  the rows where OPENFOAM ITSELF diverges (it returns nan at p = 1e6 from a hot start) -- brae
#          must refuse there too rather than quietly return a number
#   ARM 3  CONTROL, computed live inside the test: a residual-converged inversion must MISS OpenFOAM on
#          a substantial number of rows. Without it, a table OpenFOAM happened to converge on would make
#          this gate unfalsifiable.
#
# Measured when this landed: 432 rows, worst 0.000e+00 (bit-exact); 28 divergent rows mirrored; the
# control misses on 136 of 404 rows, worst 3.515e-08.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_liqref_inversion"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
BOUND=${BOUND:-1e-12}

[ -x "$BIN" ] || { echo "SKIP: $BIN not built"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u

# The oracle is built from source in this repo rather than committed as a table: a committed table can
# drift from the OpenFOAM actually installed, and this gate exists to catch exactly that class of drift.
if ! command -v liqref > /dev/null 2>&1; then
    command -v wmake > /dev/null 2>&1 || { echo "SKIP: wmake not available to build tools/liqref"; exit 77; }
    ( cd "$ROOT/tools/liqref" && wmake > /tmp/liqref_build.$$ 2>&1 ) \
        || { tail -5 /tmp/liqref_build.$$; rm -f /tmp/liqref_build.$$; echo "SKIP: tools/liqref did not build"; exit 77; }
    rm -f /tmp/liqref_build.$$
fi
command -v liqref > /dev/null 2>&1 || { echo "SKIP: liqref not on PATH after build"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
( cd "$W" && liqref > oracle.txt 2>&1 ) || { tail -5 "$W/oracle.txt"; echo "FAIL: liqref did not run"; exit 1; }
grep -q "^INV " "$W/oracle.txt" || { echo "FAIL: liqref produced no INV rows"; exit 1; }

"$BIN" "$W/oracle.txt" "$BOUND"
rc=$?
printf '  %-72s %s\n' "brae's he->T inversion reproduces OpenFOAM's own answer, path and all" \
       "$([ $rc = 0 ] && echo ok || echo FAIL)"
exit $rc
