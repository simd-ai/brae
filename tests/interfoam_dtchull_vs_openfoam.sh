#!/usr/bin/env bash
# brae's interFoam under localEuler (LTS) against REAL OpenFOAM's, on RAS/DTCHull: the local time step
# setRDeltaT.H forms, step by step, and the fields it advances.
#
# THE MESH is the tutorial's own, made by real OpenFOAM SERIALLY -- the Allrun's surfaceFeatureExtract,
# blockMesh, six topoSet/refineMesh passes and snappyHexMesh, then setFields and renumberMesh, without
# its parallel redistribution (the oracle is a serial run). 845,536 cells; about two minutes.
#
# THE METHOD is the other interFoam gates': exactly N steps of both codes, read at one instant. Under
# localEuler there is no global step to match -- deltaT is 1 and every cell advances by its own
# 1/rDeltaT -- and OpenFOAM WRITES that field (createRDeltaT.H: `rDeltaT`, AUTO_WRITE), so the local time
# step is compared at EVERY step, cell by cell, and so are setRDeltaT.H's three Info lines.
#
# TWO PROFILES, one mesh. `laminar` isolates the four localEuler consumers of the flow (the alpha
# pre-solve, CMULES, fvm::ddt(rho, U), ddtCorr); `ras` is the tutorial's kOmegaSST, whose fvm::ddt(omega)
# and fvm::ddt(k) take the local step as well, under the tutorial's own `div(phi,k|omega) Gauss
# linearUpwind limitedGrad` over `limitedGrad cellLimited Gauss linear 1`.
#
# STAGED, on both codes, and why:
#   functions removed        the forces function object writes, and changes no field
#   write every step, ascii  the oracle
#   laminar                  (`laminar` only) the flow's consumers without the closure, and without
#                            fvSolution's `cache { grad(U); }`: with no closure to form grad(U), the UEqn's
#                            first request would form and store it, and brae refuses that (the order of
#                            UEqn.H's operands is not modelled)
#
# MEASURED, ten steps, OpenFOAM's one-ulp floor beside each (the same run with ONE interface cell's alpha
# moved by one ulp, OpenFOAM against OpenFOAM):
#   laminar  rDeltaT 1.6e-16 at step 1, small after; alpha 3.6e-10, p_rgh 6.5e-10, U 5.1e-13, the outlet's
#            U face by face 3.2e-12; all p_rgh and alpha counts OpenFOAM's
#   ras      alpha 6.0e-10 (2.4e-10), p_rgh 1.1e-09 (5.0e-10), U 5.1e-13 (1.7e-13), k 3.3e-12 (5.7e-13),
#            omega 9.4e-12 (1.3e-12), nut 1.0e-11 (3.4e-12), the HULL's wall nut face by face 6.6e-11, the
#            OUTLET's U face by face 2.1e-12; p_rgh initial residuals 9.4e-11 (2.1e-10); every p_rgh,
#            alpha, omega and k count and final residual OpenFOAM's. The tutorial AS SHIPPED but for the
#            staging above: linearUpwind limitedGrad (5e-02 of OpenFOAM's own k, omega and nut over these
#            steps), nutkRoughWallFunction (rough against smooth: OpenFOAM's own nut 2.3e-01 at the first
#            step; 5,605 of 27,438 hull faces in fnRough's 2.25-90 regime by the last),
#            outletPhaseMeanVelocity (against an inletOutlet: OpenFOAM's own k 2.7e-02 at the first step,
#            alpha 9.9e-04 at the last) and the cached grad(U) (against the uncached run, OpenFOAM against
#            itself: U 2.0e-06 at the first step, k 8.1e-03 and nut 2.6e-02 at the second -- the atmosphere's
#            pressureInletOutletVelocity evaluates inside updateCoeffs, which fvMatrix does not let move U's
#            eventNo, so the reused gradient never sees it).
# At the FIRST step, before anything amplifies, U is 6e-13 on BOTH profiles -- with the outlet or the cache
# or neither -- so that is not the closure's, the wall's, the outlet's or the cache's; by the tenth step
# brae sits 2x to 7x above OpenFOAM's one-ulp run. From step 2 all but 165 of the 845,536 cells sit above
# the floor 1/maxDeltaT = 1, so the local step is live almost everywhere.
#   laminar DEVICE  setRDeltaT on the HOST from the loop's fields; the alpha pre-solve, CMULES,
#            fvm::ddt(rho, U) and ddtCorr on the GPU; outletPhaseMeanVelocity updated in the loop's U hook
#            at the host loop's instants. ONE step, at the host arm's bounds: alpha 1.6e-14, p_rgh 4.8e-15,
#            U 7.7e-13, the outlet face by face 4.6e-15. TEN steps: alpha 1.1e-09, p_rgh 2.3e-08, U 6.0e-10,
#            the outlet 1.4e-08, rDeltaT 5.2e-10, every count OpenFOAM's -- the device loop's arithmetic
#            amplified by CMULES' limiter (at the second step the pre-solve agrees with the host arm to
#            2.8e-14 and the alpha step leaves 8.0e-12, whichever consumer is switched off on both loops);
#            OpenFOAM against itself with every water cell one ulp off reads alpha 1.1e-10, p_rgh 2.8e-10 by
#            the tenth. Bounds in the gate, DEV_BOUND_*.
#   ras DEVICE  THE TUTORIAL AS SHIPPED on the GPU: kOmegaSST with the local step, the closure's linearUpwind
#            limitedGrad, the rough hull (nutkRoughWallValue, history = the wall nut as correctNut is entered),
#            outletPhaseMeanVelocity and the cached grad(U) (DeviceGradUCache: validate()'s uploaded, then each
#            correct's re-formed, consumed by every assembly -- counted, 10 of 10). ONE step, the host arm's
#            bounds and digits against OpenFOAM's CACHED run: U 6.1e-13, k 6.7e-14, omega 4.0e-13, nut 4.7e-13,
#            the hull's wall nut 1.7e-12 (OpenFOAM's own cached-vs-uncached gap is U 2.0e-06 there). TEN: alpha
#            6.9e-10, p_rgh 2.3e-08, U 4.5e-10, k 4.0e-10, omega 2.8e-10, nut 2.9e-10, the wall nut 3.7e-09, the
#            outlet 1.0e-08, every count OpenFOAM's. The first k solve's initial residual is 1.0e-08 from
#            OpenFOAM's on the device arm only: k starts uniform, so normFactor carries a pure round-off term
#            (see the gate). The rough formula is not bit-identical to the host's: libdevice's pow/sin/log are
#            not glibc's (nut_wall_function.cuh has the measurement).
#
# THE CONTROLS, three steps each against the same oracle, each asserted to FAIL on a number:
#   laminar  BRAE_CONTROL_LTS_NOSMOOTH        fvc::smooth skipped         rDeltaT at step 1 9.5e-01
#            BRAE_CONTROL_LTS_NODAMP          damping skipped             rDeltaT at step 3 5.1e-02
#            BRAE_CONTROL_LTS_SCALAR=alpha    alpha pre-solve and MULES on 1/deltaT  alpha 2.1e+02
#            BRAE_CONTROL_LTS_SCALAR=ueqn     fvm::ddt(rho, U) on 1/deltaT           U 5.3e+01
#            BRAE_CONTROL_LTS_SCALAR=ddtcorr  ddtCorr on 1/deltaT         rDeltaT at step 3 3.0e-01
#            BRAE_CONTROL_RHOPHI_ALPHAFLUX    the initial rhoPhi as the alpha flux's mass flux (what brae
#                                             built)                      rDeltaT at step 1 5.4e-01
#   ras      BRAE_CONTROL_LTS_SCALAR=turbulence  kOmegaSST's two ddts on 1/deltaT  k 1.1e+01, omega 1.1e+00
#            BRAE_CONTROL_SST_LU_OFF          linearUpwind's correction dropped (upwind)
#                                             k 9.6e-03, omega 1.7e-02 -- OpenFOAM's own upwind-vs-linearUpwind
#                                             difference to the digit
#            BRAE_CONTROL_SST_LU_UNLIMITED    the named gradient unlimited (grad(k)'s `default Gauss linear`)
#                                             k 2.3e-02, omega 2.6e-01
#            BRAE_CONTROL_NUTK_SMOOTH         the hull's nutkRoughWallFunction run as nutkWallFunction
#                                             wall nut 1.0e+00, U 4.0e-03 -- OpenFOAM's own rough-vs-smooth
#            BRAE_CONTROL_NUTKROUGH_NOHISTORY the rough limiter taken against nu_w, not the previous wall nut
#                                             wall nut 4.6e-01, U 3.2e-04
#            BRAE_CONTROL_OPMV_FROZEN         the outlet never updated (held at its file value)
#                                             alpha 7.4e-02, p_rgh 2.7e-01
#            BRAE_CONTROL_OPMV_NOLAG          the outlet updated in the first corrector too, which
#                                             OpenFOAM's updated() flag skips   U 9.6e-05, k 9.8e-03
#            BRAE_CONTROL_GRADU_UNCACHED      every grad(U) formed afresh, OpenFOAM's uncached answer
#                                             U 1.8e-05, k 2.6e-03, nut 2.3e-02
#   laminar DEVICE  the five localEuler controls and the two outlet controls on the GPU loop, each asserted
#                   to fail (a consumer switched off on one loop only reads alpha 1.3e+01, U 4.5e+00 and p_rgh
#                   1.8e-01 after ONE step against the host arm; the outlet frozen, its outlet 2.7e-02, and
#                   updated in the lagged corrector, U 1.5e-05, after three steps against a 6.3e-11 run)
#   ras DEVICE  three steps against a run reading the wall nut 9.4e-11, U 1.1e-10:
#            BRAE_CONTROL_LTS_SCALAR=turbulence  k 1.1e+01, omega 1.1e+00
#            BRAE_CONTROL_SST_LU_OFF / _UNLIMITED  k 9.6e-03 / 2.3e-02 (the host arm's digits)
#            BRAE_CONTROL_NUTK_SMOOTH_DEVICE      the hull smooth on THIS closure only -- not the validate()
#                                                 it is built from, which BRAE_CONTROL_NUTK_SMOOTH also reaches
#                                                 -- wall nut 1.0e+00, U 3.8e-03
#            BRAE_CONTROL_NUTKROUGH_NOHISTORY     wall nut 4.6e-01, U 3.2e-04 -- a DEVICE witness on DTCHull:
#                                                 the file's hull nut 5e-07 is below nu_w, so validate()'s
#                                                 limiter reads nu_w with or without the history
#            BRAE_CONTROL_GRADU_UNCACHED          U 1.8e-05, k 2.6e-03, nut 2.3e-02 -- the host arm's digits --
#                                                 and 0 of 3 assemblies took the cache
#            BRAE_CONTROL_GRADU_STALE             the cache never re-formed after the closure, so every step
#                                                 reuses validate()'s: U 5.6e-01, nut 6.1e-01, 3 of 3 took it --
#                                                 the DEVICE-formed half, which step one never reads
#            NOT WITNESSED, measured: BRAE_CONTROL_GRADU_BND_LIVE (the cached cells, the dev2 term's boundary
#            rebuilt at the assembly) reads the plain run's numbers to 5e-04 of themselves (wall nut 9.3988e-11
#            against 9.3941e-11) -- DTCHull cannot tell the boundary's instant, so it is not in the list below.
#
# NOT CLAIMED: the cached grad(U) under a laminar, kEpsilon or LES closure, MRF, or with U changed since
# the closure last formed it (refused: the UEqn would form it itself), on a refining mesh (refused: OpenFOAM
# bypasses the registry on the steps the mesh changes; a motion-solver mesh changes at every step, so there
# the cache is inert and runs, interfoam_moving_vs_openfoam `esd`), and any other cached name (refused);
# linearUpwind for k and omega naming
# different gradients or on a coupled mesh (refused), nutkRoughWallFunction's KsPlus >= 90 regime (no hull
# face reaches it in ten steps; 5,605 of 27,438 are in the 2.25-90 regime by the last), a rough wall on a
# refining mesh or under kEpsilon (refused), and both on the device (refused), fvc::spread and fvc::sweep (refused: DTCHull sets both iteration counts to 0),
# a restart (setRDeltaT damps from the third step of EVERY run, and a restart's rhoPhi is rebuilt as
# interpolate(rho)*phi -- not a continuation of the continuous run), kEpsilon and LES under localEuler
# (refused), and anything after ten steps. On the DEVICE loop: the cached grad(U) beside a limited or
# least-squares grad(U) or a coupled pair on a static mesh (refused), a device-side setRDeltaT (the host forms
# it -- its smoothing is a FaceCellWave -- and the loop uploads it and interpolate(rDeltaT) each step).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_inter_dtchull_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/RAS/DTCHull"
STL="$TUT/resources/geometry/DTC-scaled.stl.gz"
STEPS=${STEPS:-10}
CSTEPS=3
MODE=${MEASURE:+measure}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: RAS/DTCHull tutorial not found at $SRC"; exit 77; }
[ -f "$STL" ]      || { echo "SKIP: the DTC hull surface is not at $STL"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

W=${KEEP_W:-$(mktemp -d)}

# real OpenFOAM's runs (and meshes) are cached by a hash of the staged case: tests/of_oracle_cache.sh
. "$(dirname "$0")/of_oracle_cache.sh"
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v snappyHexMesh > /dev/null 2>&1 || { echo "SKIP: snappyHexMesh not on PATH"; exit 77; }
command -v interFoam > /dev/null 2>&1 || { echo "SKIP: interFoam not on PATH"; exit 77; }

M="$W/mesh"
rm -rf "$M"
cp -r "$SRC" "$M" || exit 1
rm -rf "$M"/[1-9]* "$M"/0 "$M"/processor* "$M"/log.*
grep -q "default  *localEuler;" "$M/system/fvSchemes" \
    || { echo "FAIL: the tutorial's ddtSchemes no longer names localEuler"; exit 1; }

# the Allrun's meshing, serially -- cached (oracleMesh): the key is the tutorial as copied, this function's
# text and the hull's STL
meshDTCHull()
{
(
    cd "$M" || exit 1
    mkdir -p constant/triSurface
    cp -f "$STL" constant/triSurface/
    surfaceFeatureExtract > log.surfaceFeatureExtract 2>&1 || exit 1
    blockMesh > log.blockMesh 2>&1 || exit 1
    for i in 1 2 3 4 5 6
    do
        topoSet -dict system/topoSetDict.$i > log.topoSet.$i 2>&1 || exit 1
        refineMesh -dict system/refineMeshDict -overwrite > log.refineMesh.$i 2>&1 || exit 1
    done
    snappyHexMesh -overwrite > log.snappyHexMesh 2>&1 || exit 1
    # refineMesh leaves its cellMap in 0/polyMesh; the mesh itself is in constant
    rm -rf 0
    cp -r 0.orig 0
    setFields > log.setFields 2>&1 || exit 1
    renumberMesh -overwrite > log.renumberMesh 2>&1 || exit 1
)
}
oracleMesh "$M" interfoam_dtchull meshDTCHull "$(sha256sum < "$STL" | cut -c1-16)" \
    || { echo "FAIL: meshing DTCHull"; ls "$M"; exit 1; }

# stage <profile>: the meshed case, the profile's staging, OpenFOAM run for STEPS steps
stage()
{
    local profile="$1"
    local C="$W/$profile"
    rm -rf "$C"
    cp -r "$M" "$C" || return 1
    PROFILE="$profile" STEPS="$STEPS" python3 - "$C" <<'PYEOF2' || { echo "FAIL: staging $profile"; return 1; }
import os, re, sys
d = sys.argv[1]
n = int(os.environ['STEPS'])
profile = os.environ['PROFILE']
c = os.path.join(d, 'system/controlDict')
s = open(c).read()
s, k = re.subn(r'functions\s*\{.*\}\s*(?=//)', 'functions {}\n\n', s, flags=re.S)
assert k == 1, 'functions'
for key, val in [('endTime', str(n)), ('writeInterval', '1'), ('writeFormat', 'ascii'),
                 ('writePrecision', '18')]:
    s, k = re.subn(r'^%s\s.*' % key, '%s %s;' % (key.ljust(15), val), s, flags=re.M)
    assert k == 1, key
open(c, 'w').write(s)
p = os.path.join(d, 'system/fvSolution')
t = open(p).read()
assert re.search(r'\ncache\s*\{\s*grad\(U\);\s*\}', t), 'the tutorial caches grad(U)'
if profile == 'laminar':
    t, k = re.subn(r'\ncache\s*\{[^}]*\}\s*', '\n', t)
    assert k == 1, 'the cache block'
    open(p, 'w').write(t)
assert re.search(r'type\s+outletPhaseMeanVelocity;', open(os.path.join(d, '0/U')).read()), \
    'the tutorial names outletPhaseMeanVelocity on the U outlet'
if profile == 'laminar':
    tp = os.path.join(d, 'constant/turbulenceProperties')
    t = open(tp).read()
    t, k = re.subn(r'simulationType\s+RAS;', 'simulationType  laminar;', t)
    assert k == 1, 'simulationType'
    open(tp, 'w').write(t)
if profile == 'ras':
    fs = os.path.join(d, 'system/fvSchemes')
    t = open(fs).read()
    assert re.search(r'div\(phi,k\)\s+Gauss\s+linearUpwind\s+limitedGrad;', t), 'the tutorial names linearUpwind'
    assert re.search(r'limitedGrad\s+cellLimited\s+Gauss\s+linear\s+1;', t), 'the tutorial names cellLimited'
    assert re.search(r'type\s+nutkRoughWallFunction;', open(os.path.join(d, '0/nut')).read()), \
        'the tutorial names nutkRoughWallFunction on the hull'
PYEOF2
    oracleRun "$C" interfoam_dtchull "$profile" || { echo "FAIL: interFoam [$profile]"; tail -30 "$C/log.interFoam"; return 1; }
    [ -d "$C/$STEPS" ] || { echo "FAIL: OpenFOAM wrote no $STEPS directory [$profile]"; ls "$C"; return 1; }
    echo "OpenFOAM ran $STEPS localEuler steps of DTCHull [$profile]"
}

# control <profile> <env>: brae on CSTEPS steps against OpenFOAM's log cut before its step CSTEPS+1 --
# setRDeltaT's lines for a step print ABOVE that step's `Time =`, so the cut is at the next step's first
# Flow line -- asserted to fail on a number, not on a structural check
control()
{
    local C="$W/$1"
    local ctl="$2"
    local arm="${3:-}"
    awk -v n="$CSTEPS" '/^Flow time scale min\/max/ { if (++k > n) exit } { print }' \
        "$C/log.interFoam" > "$W/log.cut"
    # shellcheck disable=SC2086
    env "$ctl" "$BIN" "$C" "$C" "$CSTEPS" "$W/log.cut" $arm > "$W/control.log" 2>&1
    local crc=$?
    local pat='FAIL: (rDeltaT at step|time scale|alpha,|p_rgh,|U,|k,|omega,|nut,)'
    grep -q "CONTROL MODE" "$W/control.log" || { echo "  FAIL: [$1] control $ctl did not run as a control"; return 1; }
    if [ $crc -ne 0 ] && grep -qE "$pat" "$W/control.log"; then
        echo "  ok:   [$1${arm:+ $arm}] control $ctl fails on a number: $(grep -m1 -E "$pat" "$W/control.log" | sed 's/^ *FAIL: //' | tr -s ' ')"
        return 0
    fi
    echo "  FAIL: [$1] control $ctl passed the gate -- the gate cannot see what it breaks"
    return 1
}

# WHAT RUNS. The files under tests/interfoam_dtchull/ each name a few of these and are one ctest test each,
# so no test is the whole gate (745 s on 848k cells as one script, 2026-10-01):
#   DTCHULL_RUN="laminar:host ras:device ..."            a profile on an arm, all STEPS steps -- and on the
#                                                         device arm one step too, at the host arm's bounds
#   DTCHULL_CONTROL="laminar:host:CTL=1 ras:device:..."  one control each, asserted to fail on a number
# Both unset, everything runs, in the order the gate always had. A profile is staged when first needed.
ALL_RUN="laminar:host ras:host laminar:device ras:device"
ALL_CONTROL=""
for ctl in BRAE_CONTROL_LTS_NOSMOOTH=1 BRAE_CONTROL_LTS_NODAMP=1 BRAE_CONTROL_LTS_SCALAR=alpha \
           BRAE_CONTROL_LTS_SCALAR=ueqn BRAE_CONTROL_LTS_SCALAR=ddtcorr BRAE_CONTROL_RHOPHI_ALPHAFLUX=1
do
    ALL_CONTROL="$ALL_CONTROL laminar:host:$ctl"
done
for ctl in BRAE_CONTROL_LTS_SCALAR=turbulence BRAE_CONTROL_SST_LU_OFF=1 BRAE_CONTROL_SST_LU_UNLIMITED=1 \
           BRAE_CONTROL_NUTK_SMOOTH=1 BRAE_CONTROL_NUTKROUGH_NOHISTORY=1 BRAE_CONTROL_OPMV_FROZEN=1 \
           BRAE_CONTROL_OPMV_NOLAG=1 BRAE_CONTROL_GRADU_UNCACHED=1
do
    ALL_CONTROL="$ALL_CONTROL ras:host:$ctl"
done
for ctl in BRAE_CONTROL_LTS_SCALAR=turbulence BRAE_CONTROL_SST_LU_OFF=1 BRAE_CONTROL_SST_LU_UNLIMITED=1 \
           BRAE_CONTROL_NUTK_SMOOTH_DEVICE=1 BRAE_CONTROL_NUTKROUGH_NOHISTORY=1 BRAE_CONTROL_GRADU_UNCACHED=1 \
           BRAE_CONTROL_GRADU_STALE=1
do
    ALL_CONTROL="$ALL_CONTROL ras:device:$ctl"
done
# ...and the device loop's own consumers, each switched off in turn: the same controls, the GPU loop
for ctl in BRAE_CONTROL_LTS_NOSMOOTH=1 BRAE_CONTROL_LTS_NODAMP=1 BRAE_CONTROL_LTS_SCALAR=alpha \
           BRAE_CONTROL_LTS_SCALAR=ueqn BRAE_CONTROL_LTS_SCALAR=ddtcorr BRAE_CONTROL_OPMV_FROZEN=1 \
           BRAE_CONTROL_OPMV_NOLAG=1
do
    ALL_CONTROL="$ALL_CONTROL laminar:device:$ctl"
done
if [ -z "${DTCHULL_RUN:-}" ] && [ -z "${DTCHULL_CONTROL:-}" ]; then
    DTCHULL_RUN="$ALL_RUN"
    DTCHULL_CONTROL="$ALL_CONTROL"
fi

declare -A STAGED
need()   # need <profile> -- staged, with OpenFOAM's run, once
{
    [ -n "${STAGED[$1]:-}" ] && return 0
    stage "$1" || { echo "interfoam_dtchull_vs_openfoam: staging failed"; exit 1; }
    STAGED[$1]=1
}

rc=0
for spec in ${DTCHULL_RUN:-}
do
    p=${spec%%:*}
    arm=${spec#*:}
    need "$p"
    if [ "$arm" = host ]; then
        echo "== [$p]"
        "$BIN" "$W/$p" "$W/$p" "$STEPS" "$W/$p/log.interFoam" $MODE || rc=1
    else
        # THE DEVICE ARM: all STEPS steps at its own bounds, and ONE step at the host arm's -- every
        # localEuler consumer and the outlet have run by then and nothing has amplified (DEV_BOUND_* in the
        # gate; on `ras`, the tutorial AS SHIPPED, the closure's residuals keep DEV_BOUND_TURB_RESIDUAL too)
        echo "== [$p device]"
        "$BIN" "$W/$p" "$W/$p" "$STEPS" "$W/$p/log.interFoam" $MODE device || rc=1
        awk '/^Flow time scale min\/max/ { if (++k > 1) exit } { print }' "$W/$p/log.interFoam" > "$W/log.one"
        echo "== [$p device, one step]"
        "$BIN" "$W/$p" "$W/$p" 1 "$W/log.one" $MODE device || rc=1
    fi
done
for spec in ${DTCHULL_CONTROL:-}
do
    p=${spec%%:*}
    rest=${spec#*:}
    arm=${rest%%:*}
    ctl=${rest#*:}
    need "$p"
    if [ "$arm" = device ]; then
        control "$p" "$ctl" device || rc=1
    else
        control "$p" "$ctl" || rc=1
    fi
done

echo "interfoam_dtchull_vs_openfoam: rc $rc"
exit $rc
