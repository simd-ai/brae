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
#   rasDevice                (`rasDevice` only) `ras` with the three things the DEVICE loop does not carry
#                            yet staged to what it does: div(phi,k|omega) `Gauss upwind`, the hull's nut
#                            `nutkWallFunction`, the cache stripped. Each goes as its device unit lands.
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
#   rasDevice DEVICE  kOmegaSST's two fvm::ddts on the local step, on the GPU. ONE step, the host arm's
#            bounds: k 6.7e-14, omega 4.0e-13, nut 4.4e-13 (the host arm's own digits). TEN: alpha 9.5e-10,
#            p_rgh 6.3e-09, U 4.1e-11, k 3.8e-10, omega 3.5e-10, nut 1.5e-10, the wall nut 4.2e-09, every
#            count OpenFOAM's. The first k solve's initial residual is 1.0e-08 from OpenFOAM's on the device
#            arm only: k starts uniform, so normFactor carries a pure round-off term (see the gate).
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
#   rasDevice DEVICE  BRAE_CONTROL_LTS_SCALAR=turbulence  k 1.1e+01, omega 1.1e+00 after three steps
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
# (refused), and anything after ten steps. On the DEVICE loop:
# the closure's linearUpwind, nutkRoughWallFunction and the cached grad(U) on a static mesh (all refused,
# hence `rasDevice`), a device-side setRDeltaT (the host forms it -- its smoothing is a
# FaceCellWave -- and the loop uploads it and interpolate(rDeltaT) each step).
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

# the Allrun's meshing, serially
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
) || { echo "FAIL: meshing DTCHull"; ls "$M"; exit 1; }

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
if profile == 'rasDevice':
    # the three things the device loop does not carry yet, each staged to what it does: the closure's
    # convection upwind, the hull's wall function smooth, fvSolution's cache stripped
    fs = os.path.join(d, 'system/fvSchemes')
    t = open(fs).read()
    t, k1 = re.subn(r'div\(phi,k\)\s+Gauss\s+linearUpwind\s+limitedGrad;', 'div(phi,k)      Gauss upwind;', t)
    t, k2 = re.subn(r'div\(phi,omega\)\s+Gauss\s+linearUpwind\s+limitedGrad;', 'div(phi,omega)  Gauss upwind;', t)
    assert k1 == 1 and k2 == 1, 'the closure convection'
    open(fs, 'w').write(t)
    fv = os.path.join(d, 'system/fvSolution')
    t = open(fv).read()
    t, k = re.subn(r'\ncache\s*\{[^}]*\}\s*', '\n', t)
    assert k == 1, 'the cache block'
    open(fv, 'w').write(t)
    nf = os.path.join(d, '0/nut')
    t = open(nf, encoding='latin-1').read()
    t, k = re.subn(r'type\s+nutkRoughWallFunction;[^}]*?(value)', r'type            nutkWallFunction;\n        \1', t,
                   count=1, flags=re.S)
    assert k == 1, 'the hull wall function'
    open(nf, 'w', encoding='latin-1').write(t)
if profile == 'ras':
    fs = os.path.join(d, 'system/fvSchemes')
    t = open(fs).read()
    assert re.search(r'div\(phi,k\)\s+Gauss\s+linearUpwind\s+limitedGrad;', t), 'the tutorial names linearUpwind'
    assert re.search(r'limitedGrad\s+cellLimited\s+Gauss\s+linear\s+1;', t), 'the tutorial names cellLimited'
    assert re.search(r'type\s+nutkRoughWallFunction;', open(os.path.join(d, '0/nut')).read()), \
        'the tutorial names nutkRoughWallFunction on the hull'
PYEOF2
    ( cd "$C" && interFoam > log.interFoam 2>&1 ) || { echo "FAIL: interFoam [$profile]"; tail -30 "$C/log.interFoam"; return 1; }
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

rc=0
for p in laminar ras
do
    stage "$p" || { echo "interfoam_dtchull_vs_openfoam: staging failed"; exit 1; }
    echo "== [$p]"
    "$BIN" "$W/$p" "$W/$p" "$STEPS" "$W/$p/log.interFoam" $MODE || rc=1
done
# `rasDevice` is the device arm's alone: the host arm is gated on `ras`, as the tutorial ships it
stage rasDevice || { echo "interfoam_dtchull_vs_openfoam: staging failed"; exit 1; }
# THE DEVICE ARM, on `laminar`: all STEPS steps at its own bounds, and ONE step at the host arm's --
# every localEuler consumer and the outlet have run by then and nothing has amplified (DEV_BOUND_* in the gate)
echo "== [laminar device]"
"$BIN" "$W/laminar" "$W/laminar" "$STEPS" "$W/laminar/log.interFoam" $MODE device || rc=1
awk '/^Flow time scale min\/max/ { if (++k > 1) exit } { print }' "$W/laminar/log.interFoam" > "$W/log.one"
echo "== [laminar device, one step]"
"$BIN" "$W/laminar" "$W/laminar" 1 "$W/log.one" $MODE device || rc=1

for ctl in BRAE_CONTROL_LTS_NOSMOOTH=1 BRAE_CONTROL_LTS_NODAMP=1 BRAE_CONTROL_LTS_SCALAR=alpha \
           BRAE_CONTROL_LTS_SCALAR=ueqn BRAE_CONTROL_LTS_SCALAR=ddtcorr BRAE_CONTROL_RHOPHI_ALPHAFLUX=1
do
    control laminar "$ctl" || rc=1
done
for ctl in BRAE_CONTROL_LTS_SCALAR=turbulence BRAE_CONTROL_SST_LU_OFF=1 BRAE_CONTROL_SST_LU_UNLIMITED=1 \
           BRAE_CONTROL_NUTK_SMOOTH=1 BRAE_CONTROL_NUTKROUGH_NOHISTORY=1 BRAE_CONTROL_OPMV_FROZEN=1 \
           BRAE_CONTROL_OPMV_NOLAG=1 BRAE_CONTROL_GRADU_UNCACHED=1
do
    control ras "$ctl" || rc=1
done

# ...and kOmegaSST under the local step on the GPU, on `rasDevice`: ten steps at the device bounds, one at
# the host's (the closure's residuals keep DEV_BOUND_TURB_RESIDUAL there too -- see the gate)
echo "== [rasDevice device]"
"$BIN" "$W/rasDevice" "$W/rasDevice" "$STEPS" "$W/rasDevice/log.interFoam" $MODE device || rc=1
awk '/^Flow time scale min\/max/ { if (++k > 1) exit } { print }' "$W/rasDevice/log.interFoam" > "$W/log.one"
echo "== [rasDevice device, one step]"
"$BIN" "$W/rasDevice" "$W/rasDevice" 1 "$W/log.one" $MODE device || rc=1
control rasDevice BRAE_CONTROL_LTS_SCALAR=turbulence device || rc=1

# ...and the device loop's own consumers, each switched off in turn: the same controls, the GPU loop
for ctl in BRAE_CONTROL_LTS_NOSMOOTH=1 BRAE_CONTROL_LTS_NODAMP=1 BRAE_CONTROL_LTS_SCALAR=alpha \
           BRAE_CONTROL_LTS_SCALAR=ueqn BRAE_CONTROL_LTS_SCALAR=ddtcorr BRAE_CONTROL_OPMV_FROZEN=1 \
           BRAE_CONTROL_OPMV_NOLAG=1
do
    control laminar "$ctl" device || rc=1
done

echo "interfoam_dtchull_vs_openfoam: rc $rc"
exit $rc
