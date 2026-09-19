#!/usr/bin/env bash
# brae's displacementLaplacian MESH MOTION against REAL OpenFOAM's, with no flow solver in the way: the
# moved points, the motion solver's cellDisplacement and pointDisplacement, meshPhi, every GAMG solve of
# the displacement equation, and C of the moved mesh, per time step.
#
# THE ORACLE is OpenFOAM's moveDynamicMesh, run SERIALLY with a fixed deltaT and written every step at
# writePrecision 18; its log carries one GAMG line per solved component. moveDynamicMesh does not link
# the waveModels library the waveMaker condition lives in -- interFoam does -- so the staged controlDict
# loads it with `libs (waveModels)`, which changes nothing else.
#
# PROFILES, each the tutorial's own mesh and dictionaries, run with a fixed deltaT from rest:
#   piston       waveMakerPiston, 20 steps of 0.05 to t = 1: a paddle moving as one along x, 56000 cells
#   flap         waveMakerFlap, the same: a paddle hinged at the bed, so the wall TILTS and the cells
#                beside it shear
#   solitary     waveMakerSolitary, the same: a solitary-wave stroke, 0.26 m of a 6 m tank by t = 1 --
#                the largest deformation -- on a mesh of two blocks
#   multiFlap    waveMakerMultiPaddleFlap, 4 steps of 0.1: four flaps side by side at 45 degrees, in 3-D
#   multiPiston  waveMakerMultiPaddlePiston, the same with pistons; 448000 cells each
#   pistonSecondOrder, flapSecondOrder
#                the piston and the flap with `secondOrder yes` staged into the paddle's entry, which no
#                tutorial turns on
#
# MEASURED, worst of the seven: all 204 GAMG solves take OpenFOAM's iteration count; the points are
# OpenFOAM's to 2.9e-16 of the extent, cellDisplacement and pointDisplacement to 1.3e-14 of the largest
# displacement, meshPhi to 2.4e-13 of the largest |meshPhi| -- the run's, not the step's, which goes to
# zero where a paddle turns -- V to 1.1e-13 relative and C to 5.9e-16 of the extent.
# THOSE ARE WHAT TWO SHARED OPERATORS COST, and the test binary's bounds say so: fvm::laplacian's face
# coefficient as (deltaCoeffs*gamma)*magSf, not gaussLaplacianScheme's deltaCoeffs*(gamma*magSf), and
# linear interpolation as w*P + (1 - w)*N, not dotInterpolate's lambda*(P - N) + N. Switched to
# OpenFOAM's order, every profile is EXACT for its first four to seven steps and a few ulps apart after.
#
# THE GAMG SOLVE IS WHERE AN EXACT TRANSCRIPTION IS NOT OPTIONAL. The paddle's faces have wall distance
# SMALL and so diffusivity 1e15, which dominates GAMG's residual normalisation: the equation is declared
# converged after ONE V-cycle, and the answer is that V-cycle's -- a hierarchy that is merely valid gives
# another mesh.
#
# BROKEN ONCE EACH, worst of piston, flap and solitary, as a fraction of the largest displacement:
#   the paddle's values not written back over the interpolated ones            3.7e-02
#   y on the named patch the next cell's, not the wave's SMALL                 4.0e-01, 20 of 40 counts
#   the wall distance without correctWalls                                     2.3e-05 (the flap only:
#                                                                              its wall tilts)
#   the 2-D correction skipped                                                 7.4e-02 (the solitary only)
#   volPointInterpolation's weights made once, not remade on the moved mesh   6.2e-04
#   the diffusivity computed once                                              9.1e-02
#   the GAMG hierarchy kept from step one                                      1.9e-01
#   the non-orthogonal correction dropped                                      7.9e-02
#
# NOT CLAIMED, because these meshes cannot tell: pointCells' ORDER (the walk over pointFaces that
# OpenFOAM takes here, and the ascending order, give the same lists on a block mesh), and face::average
# against a vertex mean for cellMotion (equal on the paddle's flat, parallel faces). Nor the Final solver
# entry, which moveDynamicMesh never selects; nor the device.
#
# THE CONTROL: brae reading `inverseDistance (rightwall)` against OpenFOAM's leftwall must FAIL, and the
# script asserts that it does. THE REFUSALS: ten arms, each an input brae must name rather than run.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_displacement_laplacian_vs_openfoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
WAVES="$TUT/multiphase/interFoam/laminar/waves"

[ -x "$BIN" ]                     || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$WAVES/waveMakerPiston" ]   || { echo "SKIP: waveMakerPiston tutorial not found under $WAVES"; exit 77; }
[ -f "$OFBASHRC" ]                || { echo "SKIP: real OpenFOAM not available"; exit 77; }

W=${BRAE_KEEP_STAGING:-$(mktemp -d)}
[ -n "${BRAE_KEEP_STAGING:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v blockMesh > /dev/null 2>&1       || { echo "SKIP: blockMesh not on PATH"; exit 77; }
command -v moveDynamicMesh > /dev/null 2>&1 || { echo "SKIP: moveDynamicMesh not on PATH"; exit 77; }

# stage <name> <tutorial> <deltaT> <nSteps> [a python edit of the staged case, run in its directory]
stage()
{
    local name="$1" tutorial="$2" dt="$3" n="$4" edit="${5:-}"
    local C="$W/$name"
    rm -rf "$C"
    cp -r "$WAVES/$tutorial" "$C" || return 1
    rm -rf "$C"/[1-9]* "$C"/0 "$C"/processor* "$C"/log.*
    cp -r "$C/0.orig" "$C/0"
    DT="$dt" N="$n" python3 - "$C" <<'PYEOF' || { echo "FAIL: staging $name"; return 1; }
import os, re, sys
d = sys.argv[1]
dt, n = os.environ['DT'], int(os.environ['N'])
c = os.path.join(d, 'system/controlDict')
s = open(c).read()
s = re.sub(r'functions\s*\{.*\n\}\s*\n', '\n', s, flags=re.S)
for key, val in [('startFrom', 'startTime'), ('startTime', '0'), ('adjustTimeStep', 'no'), ('deltaT', dt),
                 ('endTime', '%.10g' % (n*float(dt))), ('writeControl', 'timeStep'),
                 ('writeInterval', '1'), ('writeFormat', 'ascii'), ('writePrecision', '18'),
                 ('timePrecision', '12')]:
    s, k = re.subn(r'^%s\s.*' % key, '%s %s;' % (key.ljust(15), val), s, flags=re.M)
    assert k == 1, key
s = re.sub(r'^writeCompression\s.*', 'writeCompression off;', s, flags=re.M)
s += '\nlibs (waveModels);\n'
open(c, 'w').write(s)
PYEOF
    if [ -n "$edit" ]; then
        ( cd "$C" && python3 -c "$edit" ) || { echo "FAIL: staging edit [$name]"; return 1; }
    fi
    ( cd "$C" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh [$name]"; tail -20 "$C/log.blockMesh"; return 1; }
    ( cd "$C" && moveDynamicMesh > log.moveDynamicMesh 2>&1 ) || { echo "FAIL: moveDynamicMesh [$name]"; tail -30 "$C/log.moveDynamicMesh"; return 1; }
    local end
    end=$(python3 -c "print('%.10g' % ($n*float('$dt')))")
    ( cd "$C" && postProcess -func writeCellVolumes -time "$end" > log.V 2>&1 \
              && postProcess -func writeCellCentres -time "$end" > log.C 2>&1 ) \
        || { echo "FAIL: postProcess at $end [$name]"; tail -20 "$C/log.V"; return 1; }
    [ -f "$C/$end/meshPhi" ] || { echo "FAIL: OpenFOAM wrote no $end/meshPhi [$name]"; ls "$C"; return 1; }
    echo "OpenFOAM moved the mesh $n steps of deltaT $dt to t = $end   [$name]"
}

# timeDirs <case> <deltaT> <nSteps>: the time directories, in order, as OpenFOAM named them
timeDirs()
{
    python3 - "$1" "$2" "$3" <<'PYEOF'
import sys
c, dt, n = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
print(' '.join('%s/%.10g' % (c, k*dt) for k in range(1, n + 1)))
PYEOF
}

# gate <name> <deltaT> <nSteps> [the case brae reads its dictionaries from, if not the oracle's]
gate()
{
    local name="$1" dt="$2" n="$3" braeCase="${4:-$W/$1}"
    if [ "$braeCase" != "$W/$name" ]; then
        # brae reads the ORACLE's log, and its own case's dictionaries and mesh
        cp "$W/$name/log.moveDynamicMesh" "$braeCase/"
    fi
    # shellcheck disable=SC2046
    "$BIN" "$braeCase" "$name" "$dt" $(timeDirs "$W/$name" "$dt" "$n")
}

# the waveMaker's second-order correction, which no tutorial turns on: staged into the paddle's entry
SECOND="import re; p='0/pointDisplacement'; s=open(p).read(); s2=re.sub(r'(motionType\s+\w+;)', r'\1 secondOrder yes;', s, count=1); assert s2!=s; open(p,'w').write(s2)"

PROFILES=${BRAE_DL_PROFILES:-"piston flap solitary multiFlap multiPiston pistonSecondOrder flapSecondOrder"}
rc=0
for p in $PROFILES; do
    case "$p" in
        piston)   stage piston   waveMakerPiston   0.05 20 || rc=1 ;;
        flap)     stage flap     waveMakerFlap     0.05 20 || rc=1 ;;
        solitary) stage solitary waveMakerSolitary 0.05 20 || rc=1 ;;
        multiFlap)   stage multiFlap   waveMakerMultiPaddleFlap   0.1 4 || rc=1 ;;
        multiPiston) stage multiPiston waveMakerMultiPaddlePiston 0.1 4 || rc=1 ;;
        pistonSecondOrder) stage pistonSecondOrder waveMakerPiston 0.05 20 "$SECOND" || rc=1 ;;
        flapSecondOrder)   stage flapSecondOrder   waveMakerFlap   0.05 20 "$SECOND" || rc=1 ;;
    esac
done
[ $rc = 0 ] || { echo "displacement_laplacian_vs_openfoam: staging failed"; exit 1; }
for p in $PROFILES; do
    case "$p" in
        multiFlap|multiPiston) gate "$p" 0.1 4 || rc=1 ;;
        *) gate "$p" 0.05 20 || rc=1 ;;
    esac
done
# THE CONTROL: brae's piston measuring its wall distance from the OTHER wall must not be OpenFOAM's
if echo " $PROFILES " | grep -q " piston "; then
    X="$W/control_rightwall"
    rm -rf "$X"
    mkdir -p "$X"
    cp -r "$W/piston/constant" "$W/piston/system" "$W/piston/0" "$X/"
    python3 - "$X/constant/dynamicMeshDict" <<'PYEOF' || { echo "FAIL: staging the control"; exit 1; }
import re, sys
p = sys.argv[1]
s = open(p).read()
s2 = re.sub(r'inverseDistance\s*\(\s*leftwall\s*\)', 'inverseDistance (rightwall)', s)
assert s2 != s, 'the diffusivity entry was not found'
open(p, 'w').write(s2)
PYEOF
    out=$(gate piston 0.05 20 "$X")
    if echo "$out" | grep -q "FAIL: the points are OpenFOAM's" && echo "$out" | grep -q "FAIL: cellDisplacement is OpenFOAM's"; then
        echo "ok:   CONTROL: with the wall distance from rightwall, brae's points and cellDisplacement are NOT OpenFOAM's"
        echo "$out" | grep "worst over the run" | sed 's/^/      /'
    else
        echo "FAIL: CONTROL: the diffusivity's patch made no difference the gate could see"
        rc=1
    fi
fi

# THE REFUSALS: brae's side of the piston staged with one input it does not have, which it must name
# rather than run as something else. OpenFOAM is not run; the oracle is the refusal's own text.
# refuse <name> <expected text> <python edit, run in the staged case>
refuse()
{
    local name="$1" expect="$2" edit="$3"
    local R="$W/refuse_$name"
    rm -rf "$R"
    mkdir -p "$R"
    cp -r "$W/piston/constant" "$W/piston/system" "$W/piston/0" "$R/"
    ( cd "$R" && python3 -c "$edit" ) || { echo "FAIL: staging refusal arm $name"; return 1; }
    local out
    out=$(gate piston 0.05 1 "$R" 2>&1)
    if echo "$out" | grep -q "REFUSED:.*$expect"; then
        echo "ok:   REFUSED [$name]: $expect"
    else
        echo "FAIL: [$name] was not refused with \"$expect\""
        echo "$out" | tail -5 | sed 's/^/      /'
        return 1
    fi
}

if echo " $PROFILES " | grep -q " piston "; then
    SUB="import re,sys; p='%s'; s=open(p).read(); s2=re.sub(r'%s', r'%s', s, count=1, flags=re.S); assert s2!=s; open(p,'w').write(s2)"
    refuse uniformDiffusivity "Only inverseDistance is ported" \
        "$(printf "$SUB" constant/dynamicMeshDict 'diffusivity\s+inverseDistance\s*\([^)]*\);' 'diffusivity uniform;')" || rc=1
    refuse pcgSolver "Only GAMG is ported" \
        "$(printf "$SUB" system/fvSolution 'solver\s+GAMG;' 'solver PCG; preconditioner DIC;')" || rc=1
    refuse uncorrected "Gauss linear corrected" \
        "$(printf "$SUB" system/fvSchemes '(laplacianSchemes\s*\{\s*default\s+)Gauss linear corrected;' '\1Gauss linear uncorrected;')" || rc=1
    refuse slipPatch "is \`slip\`" \
        "$(printf "$SUB" 0/pointDisplacement '(top\s*\{\s*type\s+)zeroGradient;' '\1slip;')" || rc=1
    refuse motionType "knows piston, flap and solitary" \
        "$(printf "$SUB" 0/pointDisplacement 'motionType\s+piston;' 'motionType wiggle;')" || rc=1
    refuse noValue "has no \`value\`" \
        "$(printf "$SUB" 0/pointDisplacement '(leftwall\s*\{\s*type\s+waveMaker;)\s*value\s+uniform\s*\([^)]*\);' '\1')" || rc=1
    refuse frozenPoints "frozenPointsZone" \
        "$(printf "$SUB" constant/dynamicMeshDict '(diffusivity\s+inverseDistance\s*\([^)]*\);)' '\1 frozenPointsZone none;')" || rc=1
    # the diffusivity's wall distance reads fvSchemes' `patchDist` (wallDist's patch type name is "patch")
    refuse patchDistPoisson "patchDist { method Poisson; }" \
        "open('system/fvSchemes', 'a').write(chr(10) + 'patchDist { method Poisson; }' + chr(10))" || rc=1
    refuse patchDistNoCorrect "patchDist { correctWalls false; }" \
        "open('system/fvSchemes', 'a').write(chr(10) + 'patchDist { correctWalls false; }' + chr(10))" || rc=1
    refuse patchDistInterval "patchDist { updateInterval 2; }" \
        "open('system/fvSchemes', 'a').write(chr(10) + 'patchDist { updateInterval 2; }' + chr(10))" || rc=1
    # ...and its opposite: the dictionary naming exactly the defaults must RUN
    R="$W/accept_patchDistDefaults"
    rm -rf "$R"
    mkdir -p "$R"
    cp -r "$W/piston/constant" "$W/piston/system" "$W/piston/0" "$R/"
    printf '\npatchDist { method meshWave; correctWalls true; updateInterval 1; }\n' >> "$R/system/fvSchemes"
    out=$(gate piston 0.05 20 "$R" 2>&1)
    st=$?
    if [ $st -ne 0 ] || echo "$out" | grep -q "REFUSED:"; then
        echo "FAIL: [patchDistDefaults] the defaults, written out, did not run and pass (status $st)"
        echo "$out" | tail -5 | sed 's/^/      /'
        rc=1
    else
        echo "ok:   RUNS [patchDistDefaults]: patchDist naming meshWave, correctWalls true, updateInterval 1"
    fi
fi

echo "displacement_laplacian_vs_openfoam: rc $rc"
exit $rc
