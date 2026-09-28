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
# STAGED, on both codes, and why:
#   laminar                  the closure's fvm::ddt under localEuler is not ported yet; the case reader
#                            refuses a turbulent localEuler case by name
#   U outlet inletOutlet     outletPhaseMeanVelocity is not ported (refused by name); inletOutlet with
#     (0 0 0)                (0 0 0) is the same mixed shape -- fixed on inflow, zero-gradient on outflow --
#                            and keeps the file's value, so the first step's fluxes are the tutorial's
#   `cache` removed          brae does not read fvSolution's cache block
#   functions removed        the forces function object writes, and changes no field
#   write every step, ascii  the oracle
#
# MEASURED, laminar, ten steps: rDeltaT at step 1 1.6e-16 relative (every cell), at every later step
# 3.4e-12 or less; the time-scale lines 3.4e-12; alpha 2.6e-10, p_rgh 6.1e-10, U 6.4e-13; all 20 p_rgh
# and 10 alpha iteration counts OpenFOAM's. From step 2 all but 165 of the 845,536 cells sit above the
# floor 1/maxDeltaT = 1, so the local step is live almost everywhere.
# THE FLOOR IS OPENFOAM'S OWN: the same run with ONE interface cell's alpha moved by one ulp reads, OpenFOAM
# against OpenFOAM after ten steps, alpha 8.8e-11, p_rgh 1.2e-10, U 3.7e-13 -- brae is 2x to 5x that.
#
# THE CONTROLS, three steps each against the same oracle, each asserted to FAIL on a number:
#   BRAE_CONTROL_LTS_NOSMOOTH        fvc::smooth skipped              rDeltaT at step 1 9.5e-01
#   BRAE_CONTROL_LTS_NODAMP          damping skipped                  rDeltaT at step 3 5.1e-02
#   BRAE_CONTROL_LTS_SCALAR=alpha    alpha pre-solve and MULES on 1/deltaT      alpha 2.1e+02
#   BRAE_CONTROL_LTS_SCALAR=ueqn     fvm::ddt(rho, U) on 1/deltaT               U 5.3e+01
#   BRAE_CONTROL_LTS_SCALAR=ddtcorr  ddtCorr on 1/deltaT              rDeltaT at step 3 3.0e-01
#   BRAE_CONTROL_RHOPHI_ALPHAFLUX    the initial rhoPhi as the alpha  rDeltaT at step 1 5.4e-01
#                                    flux's mass flux (what brae built)
#
# NOT CLAIMED: the turbulent case, outletPhaseMeanVelocity, nutkRoughWallFunction and the closure's
# linearUpwind limitedGrad (refused; later units), fvc::spread and fvc::sweep (refused: DTCHull sets both
# iteration counts to 0), a restart (setRDeltaT damps from the third step of EVERY run, and a restart's
# rhoPhi is rebuilt as interpolate(rho)*phi -- not a continuation of the continuous run), the device loop
# (refused by name), and anything after ten steps.
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

C="$W/laminar"
rm -rf "$C"
cp -r "$SRC" "$C" || exit 1
rm -rf "$C"/[1-9]* "$C"/0 "$C"/processor* "$C"/log.*
grep -q "default  *localEuler;" "$C/system/fvSchemes" \
    || { echo "FAIL: the tutorial's ddtSchemes no longer names localEuler"; exit 1; }

# the Allrun's meshing, serially
(
    cd "$C" || exit 1
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
) || { echo "FAIL: meshing DTCHull"; ls "$C"; exit 1; }

STEPS="$STEPS" python3 - "$C" <<'PYEOF' || { echo "FAIL: staging"; exit 1; }
import os, re, sys
d = sys.argv[1]
n = int(os.environ['STEPS'])
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
t, k = re.subn(r'\ncache\s*\{[^}]*\}\s*', '\n', t)
assert k == 1, 'the cache block'
open(p, 'w').write(t)
q = os.path.join(d, '0/U')
t = open(q).read()
t, k = re.subn(r'outlet\s*\{[^}]*outletPhaseMeanVelocity[^}]*\}',
               'outlet\n    {\n        type            inletOutlet;\n        inletValue      uniform (0 0 0);\n'
               '        value           $internalField;\n    }', t, count=1)
assert k == 1, 'the U outlet'
open(q, 'w').write(t)
tp = os.path.join(d, 'constant/turbulenceProperties')
t = open(tp).read()
t, k = re.subn(r'simulationType\s+RAS;', 'simulationType  laminar;', t)
assert k == 1, 'simulationType'
open(tp, 'w').write(t)
PYEOF

( cd "$C" && interFoam > log.interFoam 2>&1 ) || { echo "FAIL: interFoam"; tail -30 "$C/log.interFoam"; exit 1; }
[ -d "$C/$STEPS" ] || { echo "FAIL: OpenFOAM wrote no $STEPS directory"; ls "$C"; exit 1; }
echo "OpenFOAM ran $STEPS localEuler steps of DTCHull [laminar]"

rc=0
"$BIN" "$C" "$C" "$STEPS" "$C/log.interFoam" $MODE || rc=1

# the controls run CSTEPS steps, against OpenFOAM's log cut before its step CSTEPS+1 -- setRDeltaT's
# lines for a step print ABOVE that step's `Time =`, so the cut is at the next step's first Flow line
awk -v n="$CSTEPS" '/^Flow time scale min\/max/ { if (++k > n) exit } { print }' \
    "$C/log.interFoam" > "$W/log.cut"
for ctl in BRAE_CONTROL_LTS_NOSMOOTH=1 BRAE_CONTROL_LTS_NODAMP=1 BRAE_CONTROL_LTS_SCALAR=alpha \
           BRAE_CONTROL_LTS_SCALAR=ueqn BRAE_CONTROL_LTS_SCALAR=ddtcorr BRAE_CONTROL_RHOPHI_ALPHAFLUX=1
do
    env "$ctl" "$BIN" "$C" "$C" "$CSTEPS" "$W/log.cut" > "$W/control.log" 2>&1
    crc=$?
    grep -q "CONTROL MODE" "$W/control.log" || { echo "  FAIL: control $ctl did not run as a control"; rc=1; continue; }
    if [ $crc -ne 0 ] && grep -qE "FAIL: (rDeltaT at step|time scale|alpha,|p_rgh,|U,)" "$W/control.log"; then
        echo "  ok:   control $ctl fails on a number: $(grep -m1 -E 'FAIL: (rDeltaT at step|time scale|alpha,|p_rgh,|U,)' "$W/control.log" | sed 's/^ *FAIL: //' | tr -s ' ')"
    else
        echo "  FAIL: control $ctl passed the gate -- the gate cannot see what it breaks"
        rc=1
    fi
done

echo "interfoam_dtchull_vs_openfoam: rc $rc"
exit $rc
