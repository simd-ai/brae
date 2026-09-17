#!/usr/bin/env bash
# What brae's interFoam REFUSES, and what it must NOT refuse -- on damBreak, one input changed at a time.
#
# WHY THIS GATE EXISTS. brae was run over all 44 shipped interFoam tutorials and the ones that reached
# `End:` were counted. Two did that should not have: laminar/damBreakWithObstacle and
# laminar/oscillatingBox both ask for `dynamicRefineFvMesh`, and brae ran them on the mesh as written
# without a word. brae's interFoam never opened constant/dynamicMeshDict. Seventeen more tutorials carry a
# moving mesh and were only stopped because they hit some OTHER refusal first. The same sweep found MRF
# and fvOptions running silently -- both listed as refused in braeInterFoam.cu's own header -- a
# dictionary-form `sigma` read as ZERO surface tension, a setTimeStep function object ignored, and the
# device loop taking nOuterCorrectors and nNonOrthogonalCorrectors as 1 and 0 whatever the case said.
#
# EVERY ARM HAS ITS OPPOSITE. A refusal that fires on a case OpenFOAM would run unchanged is a defect
# too, so beside each refused input sits the form of it that OpenFOAM treats as nothing: `staticFvMesh`,
# an MRF zone and an fvOption with `active no`, a scalar sigma. Those must RUN.
#
# THEN TURBULENCE ARRIVED, and with it the question of what ELSE had only been kept out by the
# turbulence refusal. Two things, both read and then never used: `ddtSchemes default` went into
# f.ddtU and no further, so CrankNicolson and localEuler would have run as Euler; and laplacianSchemes
# and snGradSchemes were never opened, so `corrected` -- 28 of 44 tutorials -- ran orthogonal. On a mesh
# of rectangles the second is not a substitution (the correction vector is zero, which is why damBreak
# agrees with OpenFOAM to 1e-12 under `Gauss linear corrected`); on any other mesh it is. So the mesh
# arms below shear damBreak's upper blocks by six degrees: refused under the case's own `corrected`,
# and it must RUN once the case itself says `orthogonal`.
#
# No OpenFOAM solver is run -- only its mesh generator, because damBreak ships no mesh.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/brae_interFoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/laminar/damBreak/damBreak"

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: damBreak tutorial not found at $SRC"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available (blockMesh is needed)"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v blockMesh > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }

B="$W/base"
cp -r "$SRC" "$B" || exit 1
cp -r "$B/0.orig" "$B/0"
( cd "$B" && blockMesh > log.blockMesh 2>&1 && setFields > log.setFields 2>&1 ) \
    || { echo "SKIP: blockMesh/setFields failed"; exit 77; }
# two fixed steps: long enough to reach the time loop, short enough to cost nothing
sed -i 's/^endTime .*/endTime         0.0002;/; s/^deltaT .*/deltaT          1e-4;/; s/^adjustTimeStep .*/adjustTimeStep  no;/' \
    "$B/system/controlDict"

# ...and RAS/damBreak, the turbulent twin, for the arms that need a case that IS turbulent
SRCR="$TUT/multiphase/interFoam/RAS/damBreak/damBreak"
[ -d "$SRCR" ] || { echo "SKIP: RAS/damBreak tutorial not found at $SRCR"; exit 77; }
BR="$W/baseRAS"
cp -r "$SRCR" "$BR" || exit 1
cp -r "$BR/0.orig" "$BR/0"
( cd "$BR" && blockMesh > log.blockMesh 2>&1 && setFields > log.setFields 2>&1 ) \
    || { echo "SKIP: blockMesh/setFields failed on RAS/damBreak"; exit 77; }
sed -i 's/^endTime .*/endTime         0.0002;/; s/^deltaT .*/deltaT          1e-4;/; s/^adjustTimeStep .*/adjustTimeStep  no;/' \
    "$BR/system/controlDict"

# ...and laminar/waves/stokesI, for the wave boundary conditions, on a coarser mesh than it ships
SRCW="$TUT/multiphase/interFoam/laminar/waves/stokesI"
[ -d "$SRCW" ] || { echo "SKIP: waves/stokesI tutorial not found at $SRCW"; exit 77; }
BW="$W/baseWaves"
cp -r "$SRCW" "$BW" || exit 1
cp -r "$BW/0.orig" "$BW/0"
sed -i 's/(500 1 75) simpleGrading/(50 1 75) simpleGrading/' "$BW/system/blockMeshDict"
( cd "$BW" && blockMesh > log.blockMesh 2>&1 && setFields > log.setFields 2>&1 ) \
    || { echo "SKIP: blockMesh/setFields failed on waves/stokesI"; exit 77; }
sed -i 's/^endTime .*/endTime         0.02;/; s/^deltaT .*/deltaT          0.01;/; s/^adjustTimeStep .*/adjustTimeStep  no;/' \
    "$BW/system/controlDict"

HAVE_GPU=0
if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi > /dev/null 2>&1; then HAVE_GPU=1; fi

fails=0
HDR='FoamFile { version 2.0; format ascii; class dictionary; object X; }'

# arm <name> <expect: refused|runs> <substring the message must carry, or -> <flags> <edit snippet>
arm()
{
    local name="$1" expect="$2" needle="$3" flags="$4"
    shift 4
    local C="$W/$name"
    cp -r "${BASE:-$B}" "$C"
    ( cd "$C" && eval "$@" ) || { echo "  FAIL: $name -- the staging edit itself failed"; fails=$((fails+1)); return; }
    local out
    # shellcheck disable=SC2086
    out=$("$BIN" -case "$C" $flags 2>&1)
    local got=refused
    echo "$out" | grep -q "^End: t" && got=runs
    local ok=1
    [ "$got" = "$expect" ] || ok=0
    if [ "$expect" = refused ] && [ "$needle" != "-" ]; then
        echo "$out" | grep -qF -- "$needle" || ok=0
    fi
    if [ $ok = 1 ]; then
        printf "  ok:   %-34s %s\n" "$name" "$got"
    else
        printf "  FAIL: %-34s expected %s%s, got %s: %s\n" "$name" "$expect" \
               "$([ "$needle" != "-" ] && echo " naming \`$needle\`")" "$got" \
               "$(echo "$out" | grep -E '^brae interFoam:|^End:' | head -1 | cut -c1-140)"
        fails=$((fails+1))
    fi
}

echo "== brae interFoam: what it refuses, and what it must not =="

arm baseline                runs    -                        "" true

# the mesh
arm mesh_dynamicRefine      refused "dynamicRefineFvMesh"     "" "printf '%s\ndynamicFvMesh dynamicRefineFvMesh;\n' '$HDR' > constant/dynamicMeshDict"
arm mesh_motionSolver       refused "dynamicMotionSolverFvMesh" "" "printf '%s\ndynamicFvMesh dynamicMotionSolverFvMesh;\n' '$HDR' > constant/dynamicMeshDict"
arm mesh_noType             refused "no \`dynamicFvMesh\` entry" "" "printf '%s\n' '$HDR' > constant/dynamicMeshDict"
arm mesh_static             runs    -                        "" "printf '%s\ndynamicFvMesh staticFvMesh;\n' '$HDR' > constant/dynamicMeshDict"

# MRF
arm mrf_active              refused "MRFProperties"           "" "printf '%s\nMRF1 { cellZone all; origin (0 0 0); axis (0 0 1); omega 10; }\n' '$HDR' > constant/MRFProperties"
arm mrf_inactive            runs    -                        "" "printf '%s\nMRF1 { cellZone all; active no; origin (0 0 0); axis (0 0 1); omega 10; }\n' '$HDR' > constant/MRFProperties"
arm mrf_empty               runs    -                        "" "printf '%s\n' '$HDR' > constant/MRFProperties"

# fvOptions, in both places OpenFOAM looks
arm fvoptions_system        refused "system/fvOptions"        "" "printf '%s\nsrc { type scalarSemiImplicitSource; }\n' '$HDR' > system/fvOptions"
arm fvoptions_constant      refused "constant/fvOptions"      "" "printf '%s\nsrc { type scalarSemiImplicitSource; }\n' '$HDR' > constant/fvOptions"
arm fvoptions_inactive      runs    -                        "" "printf '%s\nsrc { type scalarSemiImplicitSource; active no; }\n' '$HDR' > system/fvOptions"
# constant/ is looked up FIRST and OpenFOAM stops there: an inactive one in constant/ hides an active
# one in system/
arm fvoptions_constant_wins runs    -                        "" "printf '%s\nsrc { type x; active no; }\n' '$HDR' > constant/fvOptions; printf '%s\nsrc { type x; }\n' '$HDR' > system/fvOptions"

# surface tension
arm sigma_model             refused "temperatureDependent"    "" "sed -i 's/^sigma .*/sigma { type temperatureDependent; sigma table ((0 0.07)); }/' constant/transportProperties"
arm sigma_absent            refused "has no \`sigma\`"         "" "sed -i '/^sigma /d' constant/transportProperties"
arm sigma_zero              runs    -                        "" "sed -i 's/^sigma .*/sigma 0;/' constant/transportProperties"

# the one function object that changes the solution
arm fo_setTimeStep          refused "setTimeStep"             "" "sed -i 's|^// \*\*\*.*||' system/controlDict; printf 'functions { dt { type setTimeStep; libs (utilityFunctionObjects); deltaT 1e-5; } }\n' >> system/controlDict"
arm fo_harmless             runs    -                        "" "sed -i 's|^// \*\*\*.*||' system/controlDict; printf 'functions { p { type probes; libs (sampling); fields (p); probeLocations ((0.1 0.1 0)); } }\n' >> system/controlDict"

# already refused before this gate; here so they stay refused
arm nonNewtonian            refused "CrossPowerLaw"           "" "sed -i '0,/transportModel  *Newtonian;/s//transportModel  CrossPowerLaw;/' constant/transportProperties"

# the time scheme: read into f.ddtU and then handed to nobody, so these ran as Euler
arm ddt_CrankNicolson       refused "CrankNicolson"           "" "sed -i '/^ddtSchemes/,/^}/ s/default .*/default         CrankNicolson 0.5;/' system/fvSchemes"
arm ddt_localEuler          refused "localEuler"              "" "sed -i '/^ddtSchemes/,/^}/ s/default .*/default         localEuler;/' system/fvSchemes"

# a solver-entry floor neither the alpha pre-solve nor the momentum predictor honours yet
arm alpha_minIter           refused "minIter 1"               "" "sed -i 's/^\\( *\\)MULESCorr  *yes;/\\1MULESCorr       yes;\\n\\1minIter 1;/' system/fvSolution"

# the non-orthogonal correction: damBreak says `corrected`, brae assembles orthogonal. SHEAR holds the
# edit that makes that matter -- the upper blocks' top edge moved 0.4 in x, six degrees.
SHEAR="sed -i 's/(0 4 /(0.4 4 /; s/(2 4 /(2.4 4 /; s/(2.16438 4 /(2.56438 4 /; s/(4 4 /(4.4 4 /' system/blockMeshDict && blockMesh > log.blockMesh 2>&1 && rm -rf 0 && cp -r 0.orig 0 && setFields > log.setFields 2>&1"
ORTHO="sed -i '/^laplacianSchemes/,/^}/ s/default .*/default         Gauss linear orthogonal;/; /^snGradSchemes/,/^}/ s/default .*/default         orthogonal;/' system/fvSchemes"
arm mesh_sheared_corrected  refused "non-orthogonal"          "" "$SHEAR"
arm mesh_sheared_orthogonal runs    -                        "" "$SHEAR && $ORTHO"
arm mesh_square_corrected   runs    -                        "" true

# TURBULENCE. laminar damBreak made RAS carries no k, epsilon or nut, and OpenFOAM stops on it too.
arm ras_noFields            refused "does not exist"          "" "sed -i 's/simulationType .*/simulationType RAS;\\nRAS { RASModel kEpsilon; turbulence on; }/' constant/turbulenceProperties; sed -i 's/div(rhoPhi,U) .*/&\\n    div(phi,k) Gauss upwind;\\n    div(phi,epsilon) Gauss upwind;/' system/fvSchemes"
BASE="$BR"
arm ras_baseline            runs    -                        "" true
arm ras_kOmegaSST           refused "kOmegaSST"               "" "sed -i 's/RASModel .*/RASModel        kOmegaSST;/' constant/turbulenceProperties"
arm ras_LES                 refused "LES"                     "" "sed -i 's/^simulationType .*/simulationType LES;/' constant/turbulenceProperties"
arm ras_turbulenceOff       refused "turbulence off"          "" "sed -i 's/turbulence  *on;/turbulence      off;/' constant/turbulenceProperties"
arm ras_densityBad          refused "density mixture"         "" "sed -i 's/^density .*/density mixture;/' constant/turbulenceProperties"
# `density uniform` looks up div(phi,k), which this tutorial does not carry: OpenFOAM stops there too
arm ras_uniform_noDivPhiK   refused "div(phi,k)"              "" "sed -i 's/^density .*/density uniform;/' constant/turbulenceProperties"
arm ras_uniform             runs    -                        "" "sed -i 's/^density .*/density uniform;/' constant/turbulenceProperties; sed -i 's/div(rhoPhi,k) /div(phi,k) /; s/div(rhoPhi,epsilon) /div(phi,epsilon) /' system/fvSchemes"
arm ras_limitedLinear       refused "Gauss upwind"            "" "sed -i 's/div(rhoPhi,k) .*/div(rhoPhi,k) Gauss limitedLinear 1;/' system/fvSchemes"
arm ras_nutSpalding         refused "nutUSpaldingWallFunction" "" "sed -i 's/nutkWallFunction/nutUSpaldingWallFunction/' 0/nut"
arm ras_nutCalculatedWall   refused "no nut wall function"    "" "sed -i '/leftWall/,/}/ s/nutkWallFunction/calculated/' 0/nut"
# a wall function on a patch that is not a `wall`: OpenFOAM's nutWallFunction::checkType stops on it
arm ras_wallFnOnPatch       refused "must be a \`wall\`"       "" "sed -i '/leftWall/,/}/ s/type  *wall;/type            patch;/' constant/polyMesh/boundary"
arm ras_noKFinal            refused "kFinal"                  "" "sed -i 's/\"(U|k|epsilon)\.\*\"/\"(U|k|epsilon)\"/' system/fvSolution"
arm ras_PBiCGStab           refused "smoothSolver"            "" "sed -i '/(U|k|epsilon)/,/}/ s/solver  *smoothSolver;/solver          PBiCGStab;/' system/fvSolution"
arm ras_everyOuter          refused "turbOnFinalIterOnly"     "" "sed -i 's/nOuterCorrectors  *1;/nOuterCorrectors 2;\\n    turbOnFinalIterOnly no;/' system/fvSolution"
# ...and the form of it that changes nothing: one outer corrector IS the final one
arm ras_everyOuter_single   runs    -                        "" "sed -i 's/nOuterCorrectors  *1;/nOuterCorrectors 1;\\n    turbOnFinalIterOnly no;/' system/fvSolution"
BASE="$B"

# the momentum predictor's solver entry: damBreak names `U` only, and with one outer corrector
# fvMatrix::solve() selects `UFinal`, so real OpenFOAM stops on it. brae ran it, reading neither.
arm mompred_noUFinal        refused "UFinal"                  "" "sed -i 's/momentumPredictor  *no;/momentumPredictor yes;/' system/fvSolution"
arm mompred_withUFinal      runs    -                        "" "sed -i 's/momentumPredictor  *no;/momentumPredictor yes;/; s/^\( *\)U\$/\1\"U.*\"/' system/fvSolution"

# WAVES. The host runs waveAlpha and waveVelocity over StokesI and shallowWaterAbsorption, and
# nothing else under those names.
BASE="$BW"
arm waves_baseline          runs    -                        "" true
arm waves_StokesII          runs    -                        "" "sed -i 's/waveModel  *StokesI;/waveModel       StokesII;/' constant/waveProperties"
arm waves_unknownModel      refused "StokesIII"               "" "sed -i 's/waveModel  *StokesI;/waveModel       StokesIII;/' constant/waveProperties"
arm waves_streamFn_noBjs    refused "Bjs"                     "" "sed -i 's/waveModel  *StokesI;/waveModel       streamFunction;\n    uMean 1;\n    waveLength 6;\n    Ejs (0.05 0.01);/' constant/waveProperties"
# THE FLUX A CONDITION NAMES. totalPressure's `phi rhoPhi;` is three tutorials' own; brae read no `phi`
# entry at all and told every condition phi.
arm flux_rhoPhi             runs    -                        "" "sed -i '/totalPressure/a\        phi             rhoPhi;' 0/p_rgh"
arm flux_unknown            refused "phiAbsolute"             "" "sed -i '/totalPressure/a\        phi             phiAbsolute;' 0/p_rgh"
arm waves_noPatchEntry      refused "no entry for patch"      "" "sed -i 's/^outlet\$/outletElsewhere/' constant/waveProperties"
arm waves_noProperties      refused "no constant/waveProperties" "" "rm constant/waveProperties"
arm waves_otherAlpha        refused "alpha.oil"               "" "sed -i '0,/alpha  *alpha.water;/s//alpha           alpha.oil;/' constant/waveProperties"
arm waves_noRampTime        refused "rampTime"                "" "sed -i '/rampTime/d' constant/waveProperties"
arm waves_noActiveAbsorption refused "activeAbsorption"       "" "sed -i '/activeAbsorption/d' constant/waveProperties"
arm waves_restart           refused "this is a restart"       "" "mkdir -p 0/uniform; printf '%s\nwaterDepthRef 0.6;\n' '$HDR' > 0/uniform/waveProperties.inlet"
arm waves_otherWaveDict     refused "waveDict"                "" "sed -i '0,/type  *waveVelocity;/s//type            waveVelocity;\n        waveDict        otherWaves;/' 0/U"
BASE="$B"

# PIMPLE controls the HOST honours...
arm host_nOuter2            runs    -                        "" "sed -i 's/nOuterCorrectors  *1;/nOuterCorrectors 2;/' system/fvSolution"
arm host_nNonOrth1          runs    -                        "" "sed -i 's/nNonOrthogonalCorrectors  *0;/nNonOrthogonalCorrectors 1;/' system/fvSolution"

# ...and the DEVICE loop does not, so there they are refused rather than run as 1 and 0
if [ $HAVE_GPU = 1 ]; then
    arm device_baseline     runs    -                        "-device" true
    arm device_nOuter2      refused "nOuterCorrectors 2"      "-device" "sed -i 's/nOuterCorrectors  *1;/nOuterCorrectors 2;/' system/fvSolution"
    arm device_nNonOrth1    refused "nNonOrthogonalCorrectors 1" "-device" "sed -i 's/nNonOrthogonalCorrectors  *0;/nNonOrthogonalCorrectors 1;/' system/fvSolution"
    arm device_mesh_dynamic refused "dynamicRefineFvMesh"     "-device" "printf '%s\ndynamicFvMesh dynamicRefineFvMesh;\n' '$HDR' > constant/dynamicMeshDict"
    # the device loop carries the kEpsilon closure now, in both lineages
    # the wave conditions are the host's only: the device loop would freeze them at the file's value
    BASE="$BW"
    arm device_waves        refused "waveAlpha/waveVelocity"  "-device" true
    # ...and so is a condition that names a flux other than phi
    BASE="$B"
    arm device_flux_rhoPhi  refused "names the flux"          "-device" "sed -i '/totalPressure/a\        phi             rhoPhi;' 0/p_rgh"
    BASE="$BR"
    arm device_ras          runs    -                        "-device" true
    arm device_ras_uniform  runs    -                        "-device" "sed -i 's/^density .*/density uniform;/' constant/turbulenceProperties; sed -i 's/div(rhoPhi,k) /div(phi,k) /; s/div(rhoPhi,epsilon) /div(phi,epsilon) /' system/fvSchemes"
    arm device_ras_kOmegaSST refused "kOmegaSST"             "-device" "sed -i 's/RASModel .*/RASModel        kOmegaSST;/' constant/turbulenceProperties"
    BASE="$B"
else
    echo "  (no GPU: the -device arms are skipped)"
fi

echo "interfoam_refusals: $fails failures"
[ $fails = 0 ]
