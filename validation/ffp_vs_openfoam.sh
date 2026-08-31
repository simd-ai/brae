#!/bin/bash
# fixedFluxPressure vs real OpenFOAM on rhoBoxP -- see validation/rhoBoxP/README.md.
#
# The defect this exists to catch: fixedFluxPressure silently built as zeroGradient. The harness's ffp
# arm compares the solver-set boundary snGrad against the gradient OpenFOAM WRITES (-0.73 Pa/m here);
# the substitution reads 0 there, four orders past the bound. Runs twice: SIMPLE, then SIMPLEC
# (consistent yes), which reaches rhoPcEqn's constrainPressure.
set -e
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
SRC=${SRC:-$(cd "$(dirname "$0")/rhoBoxP" && pwd)}
BUILD=${BUILD:-$(cd "$(dirname "$0")/../build" && pwd)}
WORK=${WORK:-/tmp/brae_ffp_vs_of}
if [ ! -f "$OFBASHRC" ]; then echo "OpenFOAM not found at $OFBASHRC -- skipping"; exit 77; fi
source "$OFBASHRC" 2>/dev/null || true

runArm() { # $1 = arm name, $2 = fvSolution mutation (sed expr or empty)
    local W="$WORK.$1"
    rm -rf "$W"; mkdir -p "$W"
    cp -r "$SRC"/* "$W/"
    mkdir -p "$W/0" && cp "$W"/0.orig/* "$W/0/"
    [ -n "$2" ] && sed -i "$2" "$W/system/fvSolution"
    ( cd "$W" && rhoSimpleFoam > log.rhoSimpleFoam 2>&1 )
    local OFLAST=$(cd "$W" && ls -d [0-9]* | grep -vx 0 | sort -g | tail -1)
    [ -n "$OFLAST" ] || { echo "FAIL($1): OpenFOAM produced no output"; exit 1; }
    local out
    out=$("$BUILD/test_rho_simple_step_cpp" "$W" 0 "$OFLAST" 2>&1) || { echo "$out" | tail -20; echo "FAIL($1)"; exit 1; }
    echo "$out" | grep -E "fixedFluxPressure arm|boundary snGrad|matches OpenFOAM|NOT ~0" | head -4
    # the arm must ENGAGE: a factory regression back to zeroGradient turns updateableSnGrad() off and
    # the arm silently never runs -- which is exactly the substitution this gate exists to catch
    echo "$out" | grep -q "fixedFluxPressure arm" || { echo "FAIL($1): the ffp arm never engaged"; exit 1; }
    echo "$out" | grep -q "^PASS" || { echo "FAIL($1)"; exit 1; }
    echo "PASS($1)"
}

runArm simple ""
runArm simplec "s/SIMPLE { nNonOrthogonalCorrectors 0;/SIMPLE { nNonOrthogonalCorrectors 0; consistent yes;/"

# THE CUDA ARM: the device constrainPressure kernel against the host one, on the SIMPLE workdir
# (device-vs-host at ~1e-13/iteration; the host was matched to OpenFOAM above, which closes the
# triangle). --boundary on purpose: the plain arm's forced-limiter block is calibrated for rhoBox and
# clamps most of rhoBoxP's field, drowning the comparison in a state the case never asked for. This is
# also the arm that caught the device's hardcoded closedVolume shifting p +43 Pa per iteration.
out=$("$BUILD/test_rho_simple_step_cuda" "$WORK.simple" 0 60 --boundary 2>&1) \
    || { echo "$out" | tail -20; echo "FAIL(cuda)"; exit 1; }
echo "$out" | grep -q "^PASS" || { echo "$out" | tail -10; echo "FAIL(cuda)"; exit 1; }
echo "PASS(cuda)"
echo PASS
