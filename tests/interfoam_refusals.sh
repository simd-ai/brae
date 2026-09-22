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

# ...and laminar/testTubeMixer, for the moving mesh, AS SHIPPED
SRCM="$TUT/multiphase/interFoam/laminar/testTubeMixer"
[ -d "$SRCM" ] || { echo "SKIP: testTubeMixer tutorial not found at $SRCM"; exit 77; }
# a brace-aware rewrite of one fvSolution entry, for the arms below that swap a pressure SOLVER.
# A regex cannot do it: `p_rghFinal` on the mixer holds a nested `preconditioner { ... }`.
cat > "$W/setSolver.py" <<'PYEOF'
import sys
key, path = sys.argv[1], 'system/fvSolution'
body = sys.argv[2].replace('\\n', '\n')   # the caller writes the entry on one shell line
t = open(path).read()
i = t.index('\n    ' + key + '\n    {')
j = t.index('{', i)
depth, k = 0, j
while True:
    if t[k] == '{': depth += 1
    elif t[k] == '}': depth -= 1
    if depth == 0: break
    k += 1
open(path, 'w').write(t[:j] + '{\n' + body + '    }' + t[k + 1:])
PYEOF

BM="$W/baseMoving"
cp -r "$SRCM" "$BM" || exit 1
cp -r "$BM/0.orig" "$BM/0"
( cd "$BM" && blockMesh > log.blockMesh 2>&1 && setFields > log.setFields 2>&1 ) \
    || { echo "SKIP: blockMesh/setFields failed on testTubeMixer"; exit 77; }
sed -i 's/^endTime .*/endTime         0.0002;/; s/^deltaT .*/deltaT          1e-4;/; s/^adjustTimeStep .*/adjustTimeStep  no;/' \
    "$BM/system/controlDict"

# ...and RAS/damBreakPorousBaffle, for the cyclic baffle, meshed as its Allrun does
SRCB="$TUT/multiphase/interFoam/RAS/damBreakPorousBaffle"
[ -d "$SRCB" ] || { echo "SKIP: damBreakPorousBaffle tutorial not found at $SRCB"; exit 77; }
command -v createBaffles > /dev/null 2>&1 || { echo "SKIP: createBaffles not on PATH"; exit 77; }
BB="$W/baseBaffle"
cp -r "$SRCB" "$BB" || exit 1
cp -r "$BB/0.orig" "$BB/0"
( cd "$BB" && blockMesh > log.blockMesh 2>&1 && setFields > log.setFields 2>&1 \
      && createBaffles -overwrite > log.createBaffles 2>&1 ) \
    || { echo "SKIP: blockMesh/setFields/createBaffles failed on damBreakPorousBaffle"; exit 77; }
sed -i 's/^endTime .*/endTime         0.0002;/; s/^deltaT .*/deltaT          1e-4;/; s/^adjustTimeStep .*/adjustTimeStep  no;/' \
    "$BB/system/controlDict"

# ...and RAS/damBreakLeakage, for the coded cyclicACMI baffle, meshed as its Allrun does
SRCK="$TUT/multiphase/interFoam/RAS/damBreakLeakage"
[ -d "$SRCK" ] || { echo "SKIP: damBreakLeakage tutorial not found at $SRCK"; exit 77; }
BK="$W/baseLeak"
cp -r "$SRCK" "$BK" || exit 1
cp -r "$BK/0.orig" "$BK/0"
( cd "$BK" && blockMesh > log.blockMesh 2>&1 && setFields > log.setFields 2>&1 \
      && createBaffles -overwrite > log.createBaffles 2>&1 ) \
    || { echo "SKIP: blockMesh/setFields/createBaffles failed on damBreakLeakage"; exit 77; }
sed -i 's/^endTime .*/endTime         0.0002;/; s/^deltaT .*/deltaT          1e-4;/; s/^adjustTimeStep .*/adjustTimeStep  no;/' \
    "$BK/system/controlDict"

# ...and laminar/waves/mangroveInteraction, for the mangrove fvOptions, meshed as Allrun does on a block
# a fifth of the tutorial's in each direction
SRCG="$TUT/multiphase/interFoam/laminar/waves/mangroveInteraction"
[ -d "$SRCG" ] || { echo "SKIP: mangroveInteraction tutorial not found at $SRCG"; exit 77; }
BG="$W/baseMangrove"
cp -r "$SRCG" "$BG" || exit 1
cp -r "$BG/0.orig" "$BG/0"
sed -i 's/(350 28 42)/(70 6 8)/' "$BG/system/blockMeshDict"
( cd "$BG" && blockMesh > log.blockMesh 2>&1 && setFields > log.setFields 2>&1 && topoSet > log.topoSet 2>&1 ) \
    || { echo "SKIP: blockMesh/setFields/topoSet failed on mangroveInteraction"; exit 77; }
sed -i 's/^startFrom .*/startFrom       startTime;/; s/^endTime .*/endTime         0.02;/; s/^deltaT .*/deltaT          0.01;/; s/^adjustTimeStep .*/adjustTimeStep  no;/' \
    "$BG/system/controlDict"
python3 -c "import re; p='$BG/system/controlDict'; t=open(p).read(); t=re.sub(r'\nfunctions\s*\{.*\n\}\s*\n', '\n', t, flags=re.S); open(p,'w').write(t)"

# ...and LES/nozzleFlow2D, for LES kEqn on a wedge, meshed as its Allrun does, two steps at a fixed 1e-9
SRCL="$TUT/multiphase/interFoam/LES/nozzleFlow2D"
[ -d "$SRCL" ] || { echo "SKIP: nozzleFlow2D tutorial not found at $SRCL"; exit 77; }
BL="$W/baseLES"
cp -r "$SRCL" "$BL" || exit 1
cp -r "$BL/0.orig" "$BL/0"
( cd "$BL" && blockMesh > log.blockMesh 2>&1 \
      && topoSet -dict system/topoSetDict.1 > log.topoSet.1 2>&1 \
      && refineMesh -dict system/refineMeshDict -overwrite > log.refineMesh.1 2>&1 \
      && topoSet -dict system/topoSetDict.2 > log.topoSet.2 2>&1 \
      && refineMesh -dict system/refineMeshDict -overwrite > log.refineMesh.2 2>&1 ) \
    || { echo "SKIP: meshing failed on nozzleFlow2D"; exit 77; }
sed -i 's/^endTime .*/endTime         2e-09;/; s/^deltaT .*/deltaT          1e-9;/; s/^adjustTimeStep .*/adjustTimeStep  no;/; s/^startFrom .*/startFrom       startTime;/' \
    "$BL/system/controlDict"

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
    # a needle on a `runs` arm is a NOTICE the run must carry: a declared substitution, not a silent one
    if [ "$needle" != "-" ]; then
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
# dynamicMotionSolverFvMesh IS ported for a solidBody motion of the whole mesh; without a motionSolver
# it is refused by that name, and the moving arms below hold the rest
arm mesh_motionSolver       refused "motionSolver"             "" "printf '%s\ndynamicFvMesh dynamicMotionSolverFvMesh;\n' '$HDR' > constant/dynamicMeshDict"
arm mesh_noType             refused "no \`dynamicFvMesh\` entry" "" "printf '%s\n' '$HDR' > constant/dynamicMeshDict"
arm mesh_static             runs    -                        "" "printf '%s\ndynamicFvMesh staticFvMesh;\n' '$HDR' > constant/dynamicMeshDict"

# MRF
# MRF IS PORTED on the host (tests/interfoam_mrf_vs_openfoam.sh holds laminar/mixerVessel2D to OpenFOAM).
# What is refused is what no gate holds, each by name. ZONE writes a 100-cell `rotor` cellZone into
# damBreak, so the arms below reach the refusal they name and not "no such zone".
ZONE="python3 -c \"open('constant/polyMesh/cellZones','w').write('FoamFile { version 2.0; format ascii; class regIOobject; location \\\"constant/polyMesh\\\"; object cellZones; }\\n1\\n(\\nrotor\\n{\\n    type cellZone;\\n    cellLabels List<label> 100(' + ' '.join(str(i) for i in range(100)) + ');\\n}\\n)\\n')\""
MRFD="printf '%s\nMRF1 { cellZone rotor; origin (0 0 0); axis (0 0 1); OMEGA }\n' '$HDR' > constant/MRFProperties"
arm mrf_noSuchZone          refused "is not in constant/polyMesh/cellZones" "" "printf '%s\nMRF1 { cellZone all; origin (0 0 0); axis (0 0 1); omega 10; }\n' '$HDR' > constant/MRFProperties"
# damBreak's walls are fixedFluxPressure, where constrainPressure takes MRF.relative(Sf & U_b)
arm mrf_fixedFluxPressure   refused "is a fixedFluxPressure"  "" "$ZONE; ${MRFD/OMEGA/omega 10;}"
arm mrf_omegaConstant       refused "is a fixedFluxPressure"  "" "$ZONE; ${MRFD/OMEGA/omega constant 10;}"
arm mrf_omegaDict           refused "is a fixedFluxPressure"  "" "$ZONE; ${MRFD/OMEGA/omega \{ type constant; value 10; \}}"
arm mrf_omegaTable          refused "Function1 of type \`table\`" "" "$ZONE; ${MRFD/OMEGA/omega table ((0 0) (1 10));}"
arm mrf_noOmega             refused "has no \`omega\` entry"  "" "$ZONE; ${MRFD/OMEGA/}"
arm mrf_inactive            runs    -                        "" "printf '%s\nMRF1 { cellZone all; active no; origin (0 0 0); axis (0 0 1); omega 10; }\n' '$HDR' > constant/MRFProperties"
arm mrf_empty               runs    -                        "" "printf '%s\n' '$HDR' > constant/MRFProperties"

# THE PERMEABLE WALL is ported (tests/interfoam_permeable_vs_openfoam.sh). What it refuses, by name:
PERMU="python3 -c \"import re; p='0/U'; t=open(p).read(); t=re.sub(r'rightWall\\s*\\{[^}]*\\}', 'rightWall { type permeableAlphaPressureInletOutletVelocity; alpha alpha.water; alphaMin 0.01; PHI value uniform (0 0 0); }', t, count=1); open(p,'w').write(t)\""
PERMP="python3 -c \"import re; p='0/p_rgh'; t=open(p).read(); t=re.sub(r'rightWall\\s*\\{[^}]*\\}', 'rightWall { type prghPermeableAlphaTotalPressure; alpha alpha.water; alphaMin 0.01; PENTRY value uniform 0; }', t, count=1); open(p,'w').write(t)\""
arm permeable_runs          runs    -                        "" "${PERMU/PHI /}; ${PERMP/PENTRY/p uniform 0;}"
arm permeable_massFlux      refused "MASS flux"               "" "${PERMU/PHI /phi rhoPhi; }; ${PERMP/PENTRY/p uniform 0;}"
arm permeable_pTable        refused "PatchFunction1"          "" "${PERMU/PHI /}; ${PERMP/PENTRY/p table ((0 0) (1 10));}"
arm permeable_noP           refused "has no \`p\` entry"      "" "${PERMU/PHI /}; ${PERMP/PENTRY/}"
PERMUOIL="${PERMU/PHI /}"
arm permeable_otherAlpha    refused "names \`alpha alpha.oil\`" "" "${PERMUOIL/alpha.water/alpha.oil}; ${PERMP/PENTRY/p uniform 0;}"
# fvOptions, in both places OpenFOAM looks
arm fvoptions_system        refused "scalarSemiImplicitSource" "" "printf '%s\nsrc { type scalarSemiImplicitSource; }\n' '$HDR' > system/fvOptions"
arm fvoptions_constant      refused "scalarSemiImplicitSource" "" "printf '%s\nsrc { type scalarSemiImplicitSource; }\n' '$HDR' > constant/fvOptions"
# explicitPorositySource/DarcyForchheimer IS ported (tests/interfoam_angledduct_vs_openfoam.sh); its
# fixedCoeff model is not, and neither is any other type
arm fvoptions_fixedCoeff    refused "fixedCoeff"              "" "$ZONE; printf '%s\nsrc { type explicitPorositySource; explicitPorositySourceCoeffs { selectionMode cellZone; cellZone rotor; type fixedCoeff; alpha (1 1 1); beta (0 0 0); rhoRef 1; coordinateSystem { origin (0 0 0); e1 (1 0 0); e2 (0 1 0); } } }\n' '$HDR' > constant/fvOptions"
arm fvoptions_darcy         runs    -                        "" "$ZONE; printf '%s\nsrc { type explicitPorositySource; explicitPorositySourceCoeffs { selectionMode cellZone; cellZone rotor; type DarcyForchheimer; d (1e5 1e5 1e5); f (0 0 0); coordinateSystem { origin (0 0 0); e1 (1 0 0); e2 (0 1 0); } } }\n' '$HDR' > constant/fvOptions"
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

# the time scheme: read into f.ddtU and then handed to nobody, so these ran as Euler. CRANKNICOLSON RUNS
# now, on both loops (tests/interfoam_cn_vs_openfoam.sh holds RAS/damBreak under it); what it refuses,
# by name: a Function1 ocCoeff (the ramp form), a coefficient outside [0, 1], the scheme on one of the two
# operand sets and Euler on the other, the scheme beside a moving mesh, beside a mangrove source (whose
# added mass takes fvm::ddt(U) under the case's scheme), beside a closure other than kEpsilon, with alpha
# sub-cycling (OpenFOAM's own FatalError), and a restart directory that holds the ddt0 fields or alphaPhi0
CNSET="sed -i '/^ddtSchemes/,/^}/ s/default .*/default         CrankNicolson 0.5;/' system/fvSchemes"
arm ddt_CrankNicolson       runs    -                        "" "$CNSET"
arm ddt_cnFull              runs    -                        "" "sed -i '/^ddtSchemes/,/^}/ s/default .*/default         CrankNicolson 1;/' system/fvSchemes"
arm ddt_cnBare              runs    -                        "" "sed -i '/^ddtSchemes/,/^}/ s/default .*/default         CrankNicolson;/' system/fvSchemes"
arm ddt_cnRamp              refused "Function1 of time"      "" "sed -i '/^ddtSchemes/,/^}/ s/default .*/default         CrankNicolson ocCoeff { type scale; scale linearRamp; duration 0.01; value 0.9; };/' system/fvSchemes"
arm ddt_cnOutOfRange        refused "should be >= 0 and <= 1" "" "sed -i '/^ddtSchemes/,/^}/ s/default .*/default         CrankNicolson 1.5;/' system/fvSchemes"
arm ddt_cnAlphaOnly         refused "mixed case"              "" "sed -i '/^ddtSchemes/,/^}/ s/default .*/default         Euler;\n    ddt(alpha)      CrankNicolson 0.5;/' system/fvSchemes"
arm ddt_cnSubCycles         refused "nAlphaSubCycles > 1"    "" "$CNSET; sed -i 's/nAlphaSubCycles  *1;/nAlphaSubCycles 2;/' system/fvSolution"
arm ddt_cnDdt0Present       refused "ddt0(rho,U)"             "" "$CNSET; printf 'FoamFile { version 2.0; format ascii; class volVectorField; object ddt0(rho,U); }\ndimensions [1 -2 -2 0 0 0 0];\ninternalField uniform (0 0 0);\nboundaryField { \".*\" { type calculated; value uniform (0 0 0); } }\n' > '0/ddt0(rho,U)'"
arm ddt_cnAlphaPhi0Present  refused "alphaPhi0"               "" "$CNSET; printf 'FoamFile { version 2.0; format ascii; class surfaceScalarField; object alphaPhi0.water; }\ndimensions [0 3 -1 0 0 0 0];\ninternalField uniform 0;\nboundaryField { \".*\" { type calculated; value uniform 0; } }\n' > 0/alphaPhi0.water"
arm ddt_localEuler          refused "localEuler"              "" "sed -i '/^ddtSchemes/,/^}/ s/default .*/default         localEuler;/' system/fvSchemes"
arm ddt_backward            refused "backward"                "" "sed -i '/^ddtSchemes/,/^}/ s/default .*/default         backward;/' system/fvSchemes"
BASE="$BM"
arm ddt_cnMoving            refused "the mesh moves"          "" "$CNSET"
BASE="$BG"
arm ddt_cnMangroves         refused "multiphaseMangrovesSource" "" "$CNSET"
BASE="$B"

# a solver-entry floor neither the alpha pre-solve nor the momentum predictor honours yet
# the host's alpha pre-solve honours minIter (tests/interfoam_dambreak_vs_openfoam.sh `alphaminiter`); the
# device's does not, and refuses it
arm alpha_minIter           runs    -                        "" "sed -i 's/^\\( *\\)MULESCorr  *yes;/\\1MULESCorr       yes;\\n\\1minIter 1;/' system/fvSolution"

# the non-orthogonal correction: damBreak says `corrected`, brae assembles orthogonal. SHEAR holds the
# edit that makes that matter -- the upper blocks' top edge moved 0.4 in x, six degrees.
SHEAR="sed -i 's/(0 4 /(0.4 4 /; s/(2 4 /(2.4 4 /; s/(2.16438 4 /(2.56438 4 /; s/(4 4 /(4.4 4 /' system/blockMeshDict && blockMesh > log.blockMesh 2>&1 && rm -rf 0 && cp -r 0.orig 0 && setFields > log.setFields 2>&1"
ORTHO="sed -i '/^laplacianSchemes/,/^}/ s/default .*/default         Gauss linear orthogonal;/; /^snGradSchemes/,/^}/ s/default .*/default         orthogonal;/' system/fvSchemes"
# the corrected schemes RUN on a non-orthogonal mesh now (interfoam_moving_vs_openfoam.sh's tanks);
# `uncorrected` there is refused, because OpenFOAM's takes the non-orthogonal delta coefficient
UNCORR="sed -i '/^laplacianSchemes/,/^}/ s/default .*/default         Gauss linear uncorrected;/; /^snGradSchemes/,/^}/ s/default .*/default         uncorrected;/' system/fvSchemes"
arm mesh_sheared_corrected  runs    -                        "" "$SHEAR"
arm mesh_sheared_orthogonal runs    -                        "" "$SHEAR && $ORTHO"
arm mesh_sheared_uncorrected refused "uncorrected"           "" "$SHEAR && $UNCORR"
arm mesh_square_uncorrected runs    -                        "" "$UNCORR"
arm mesh_square_corrected   runs    -                        "" true
# the host takes every gradient by its own entry -- Gauss linear, leastSquares, cellLimited over either
# (tests/interfoam_dambreak_vs_openfoam.sh `gradLsqLimited` and `nHatLimited`); another scheme or limiter
# is refused by name
arm grad_cellLimited        runs    -                        "" "sed -i '/^gradSchemes/,/^}/ s/default .*/default         cellLimited Gauss linear 1;/' system/fvSchemes"
arm grad_leastSquares       runs    -                        "" "sed -i '/^gradSchemes/,/^}/ s/default .*/default         leastSquares;/' system/fvSchemes"
arm grad_namedU             runs    -                        "" "sed -i '/^gradSchemes/,/^}/ s/default .*/default         Gauss linear;\n    grad(U)         cellLimited Gauss linear 1;/' system/fvSchemes"
arm grad_namedPrgh          runs    -                        "" "sed -i '/^gradSchemes/,/^}/ s/default .*/default         Gauss linear;\n    grad(p_rgh)     cellLimited Gauss linear 1;/' system/fvSchemes"
arm grad_faceLimited        refused "faceLimited"            "" "sed -i '/^gradSchemes/,/^}/ s/default .*/default         faceLimited Gauss linear 1;/' system/fvSchemes"
arm grad_cellMDLimited      refused "cellMDLimited"          "" "sed -i '/^gradSchemes/,/^}/ s/default .*/default         Gauss linear;\n    nHat            cellMDLimited Gauss linear 1;/' system/fvSchemes"
arm grad_pointCells         refused "pointCellsLeastSquares" "" "sed -i '/^gradSchemes/,/^}/ s/default .*/default         pointCellsLeastSquares;/' system/fvSchemes"

# TURBULENCE. laminar damBreak made RAS carries no k, epsilon or nut, and OpenFOAM stops on it too.
arm ras_noFields            refused "does not exist"          "" "sed -i 's/simulationType .*/simulationType RAS;\\nRAS { RASModel kEpsilon; turbulence on; }/' constant/turbulenceProperties; sed -i 's/div(rhoPhi,U) .*/&\\n    div(phi,k) Gauss upwind;\\n    div(phi,epsilon) Gauss upwind;/' system/fvSchemes"
BASE="$BR"
arm ras_baseline            runs    -                        "" true
arm mrf_RAS                 refused "MRF zone AND is turbulent" "" "$ZONE; ${MRFD/OMEGA/omega 10;}"
arm ras_otherModel          refused "realizableKE"            "" "sed -i 's/RASModel .*/RASModel        realizableKE;/' constant/turbulenceProperties"
# kOmegaSST IS ported, in the uniform lineage (tests/interfoam_waterchannel_vs_openfoam.sh holds it to
# OpenFOAM). RAS/damBreak made kOmegaSST: `density variable` with it is refused, and so is each thing
# the closure does not carry -- on a base that RUNS, so a refusal is the one edit's.
arm sst_variableDensity     refused "density variable"        "" "sed -i 's/RASModel .*/RASModel        kOmegaSST;/' constant/turbulenceProperties"
SSTBASE="sed -i 's/RASModel .*/RASModel        kOmegaSST;/; /^density /d' constant/turbulenceProperties; sed -i 's/div(rhoPhi,k) .*/div(phi,k) Gauss upwind;/; s/div(rhoPhi,epsilon) .*/div(phi,omega) Gauss upwind;/' system/fvSchemes; sed -i 's/(U|k|epsilon)/(U|k|omega)/' system/fvSolution; sed 's/epsilonWallFunction/omegaWallFunction/; s/object  *epsilon;/object      omega;/; s/\\[0 2 -3 0 0 0 0\\]/[0 0 -1 0 0 0 0]/' 0/epsilon > 0/omega; printf '\\nwallDist { method meshWave; }\\n' >> system/fvSchemes"
arm sst_baseline            runs    -                        "" "$SSTBASE"
arm sst_noOmega             refused "does not exist"          "" "$SSTBASE; rm 0/omega"
# kOmegaSST's own wallDist reads `method` with no default (patchDistMethod.C): OpenFOAM stops without it
arm sst_noWallDist          refused "wallDist { method ...; }" "" "$SSTBASE; sed -i '/^wallDist/d' system/fvSchemes"
arm sst_wallDistPoisson     refused "wallDist { method Poisson; }" "" "$SSTBASE; sed -i 's/^wallDist .*/wallDist { method Poisson; }/' system/fvSchemes"
arm sst_wallDistNoCorrect   refused "correctWalls false"     "" "$SSTBASE; sed -i 's/^wallDist .*/wallDist { method meshWave; correctWalls false; }/' system/fvSchemes"
arm sst_decayControl        refused "decayControl"            "" "$SSTBASE; sed -i 's/RASModel .*/&\\n    kOmegaSSTCoeffs { decayControl yes; kInf 1e-5; omegaInf 1; }/' constant/turbulenceProperties"
arm sst_F3                  refused "F3"                      "" "$SSTBASE; sed -i 's/RASModel .*/&\\n    kOmegaSSTCoeffs { F3 yes; }/' constant/turbulenceProperties"
arm sst_blending            refused "blending stepwise"       "" "$SSTBASE; sed -i '0,/omegaWallFunction;/ s/omegaWallFunction;/omegaWallFunction;\\n        blending        stepwise;/' 0/omega"
arm sst_nutU                refused "nutUWallFunction"        "" "$SSTBASE; sed -i '0,/nutkWallFunction/ s/nutkWallFunction/nutUWallFunction/' 0/nut"
arm sst_wallWithoutOmegaWF  refused "omegaWallFunction"       "" "$SSTBASE; sed -i '0,/omegaWallFunction;/ s/omegaWallFunction;/zeroGradient;/' 0/omega"
arm sst_linearUpwindOmega   refused "div(phi,omega)"          "" "$SSTBASE; sed -i 's/div(phi,omega) .*/div(phi,omega) Gauss linearUpwind grad(omega);/' system/fvSchemes"
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

# p_rghFinal NAMES GAMG in this tutorial as shipped, and waves_baseline above ran it. What else the
# entry may say: every control whose branch of GAMGSolver or GAMGAgglomeration is not ported is
# refused by name, because each one moves where the solve stops and none moves a converged field.
GE="sed -i '/p_rghFinal/,/}/ s/smoother  *DIC;/smoother        DIC;\\n        "
arm gamg_GaussSeidel        runs    -                        "" "sed -i '/p_rghFinal/,/}/ s/smoother  *DIC;/smoother        GaussSeidel;/' system/fvSolution"
arm gamg_DICGaussSeidel     runs    -                        "" "sed -i '/p_rghFinal/,/}/ s/smoother  *DIC;/smoother        DICGaussSeidel;/' system/fvSolution"
arm gamg_smootherDILU       refused "smoother DILU"           "" "sed -i '/p_rghFinal/,/}/ s/smoother  *DIC;/smoother        DILU;/' system/fvSolution"
arm gamg_noSmoother         refused "names no \`smoother\`"   "" "sed -i '/p_rghFinal/,/}/ {/smoother/d}' system/fvSolution"
arm gamg_sweeps             runs    -                        "" "${GE}nPreSweeps 2; nFinestSweeps 3;/' system/fvSolution"
arm gamg_mergeLevels1       runs    -                        "" "${GE}mergeLevels 1; agglomerator faceAreaPair; cacheAgglomeration on;/' system/fvSolution"
arm gamg_mergeLevels2       refused "mergeLevels 2"           "" "${GE}mergeLevels 2;/' system/fvSolution"
arm gamg_agglomerator       refused "agglomerator algebraicPair" "" "${GE}agglomerator algebraicPair;/' system/fvSolution"
arm gamg_updateInterval     refused "updateInterval"          "" "${GE}updateInterval 5;/' system/fvSolution"
arm gamg_noCache            refused "cacheAgglomeration no"   "" "${GE}cacheAgglomeration no;/' system/fvSolution"
arm gamg_interpolate        refused "interpolateCorrection yes" "" "${GE}interpolateCorrection yes;/' system/fvSolution"
arm gamg_directCoarsest     refused "directSolveCoarsest yes" "" "${GE}directSolveCoarsest yes;/' system/fvSolution"
arm gamg_coarsestLevelCorr  refused "coarsestLevelCorr"       "" "${GE}coarsestLevelCorr { solver PCG; preconditioner DIC; tolerance 1e-3; relTol 0; }/' system/fvSolution"
arm gamg_procAgglomerator   refused "processorAgglomerator"   "" "${GE}processorAgglomerator masterCoarsest;/' system/fvSolution"
arm gamg_notASwitch         refused "is not a Switch"         "" "${GE}scaleCorrection maybe;/' system/fvSolution"
BASE="$B"

# THE MOVING MESH: a solid-body motion of the whole mesh runs; everything else about a moving mesh is
# refused by name -- and so is what the SOLVER does not do on one yet
BASE="$BM"
arm moving_baseline         runs    -                        "" true
# a cellZone's solid-body motion is ported (tests/interfoam_ami_vs_openfoam.sh); a zone the mesh does not
# have, a cellSet and a regular expression are refused by name
arm moving_cellZoneMissing  refused "No matching cellZones: rotor" "" "sed -i 's/^motionSolver .*/motionSolver    solidBody;\ncellZone        rotor;/' constant/dynamicMeshDict"
arm moving_cellZone         runs    -                        "" "$ZONE; sed -i 's/^motionSolver .*/motionSolver    solidBody;\ncellZone        rotor;/' constant/dynamicMeshDict"
arm moving_cellZoneRegex    refused "regular expression"     "" "$ZONE; sed -i 's/^motionSolver .*/motionSolver    solidBody;\ncellZone        \\\"rot.*\\\";/' constant/dynamicMeshDict"
arm moving_cellSet          refused "cellSet rotor"          "" "sed -i 's/^motionSolver .*/motionSolver    solidBody;\ncellSet         rotor;/' constant/dynamicMeshDict"
arm moving_cellSetNone      runs    -                        "" "sed -i 's/^motionSolver .*/motionSolver    solidBody;\ncellSet         none;/' constant/dynamicMeshDict"
# displacementLaplacian is ported (tests/displacement_laplacian_vs_openfoam.sh); a motion solver that is
# not is still refused by name, and displacementLaplacian without its mandatory diffusivity by that
arm moving_velocityLap      refused "motionSolver velocityLaplacian" "" "sed -i 's/^motionSolver .*/motionSolver    velocityLaplacian;/' constant/dynamicMeshDict"
arm moving_displacementLap  refused "has no \`diffusivity\`"   "" "sed -i 's/^motionSolver .*/motionSolver    displacementLaplacian;/' constant/dynamicMeshDict"
arm moving_unknownFunction  refused "solidBodyMotionFunction \`wobble\`" "" "sed -i 's/^solidBodyMotionFunction .*/solidBodyMotionFunction wobble;/' constant/dynamicMeshDict"
arm moving_drivenLinear     refused "drivenLinearMotion"      "" "sed -i 's/^solidBodyMotionFunction .*/solidBodyMotionFunction drivenLinearMotion;/' constant/dynamicMeshDict"
arm moving_omegaTable       refused "Function1 \`table\`"    "" "sed -i 's/omega  *6.2832;.*/omega           table ((0 6.2832) (1 6.2832));/' constant/dynamicMeshDict"
arm moving_points0          refused "points0 exists"          "" "cp constant/polyMesh/points constant/polyMesh/points0"
# correctPhi is ported (tests/interfoam_moving_vs_openfoam.sh's *CorrectPhi profiles); CorrectPhi's
# pcorrFinal entry, which every case now reads at its start, is refused by name when it is missing
arm moving_correctPhi       runs    -                        "" "sed -i 's/correctPhi  *no;/correctPhi      yes;/' system/fvSolution"
arm moving_noPcorr          refused "solvers/pcorrFinal"     "" "sed -i 's/\"pcorr\.\*\"/pcorrNot/' system/fvSolution"
arm moving_noRefValue       refused "no pRefValue"            "" "sed -i '/pRefValue/d' system/fvSolution"
arm moving_noRefPoint       refused "neither pRefCell nor pRefPoint" "" "sed -i '/pRefPoint/d' system/fvSolution"
arm moving_refPointOutside  refused "lies in no cell"         "" "sed -i 's/^\( *\)pRefPoint .*/\1pRefPoint (1 1 1);/' system/fvSolution"
arm moving_refCell          runs    -                        "" "sed -i 's/^\( *\)pRefPoint .*/\1pRefCell 3;/' system/fvSolution"
KEFIELDS='for n, dim, t, v in [("k", "[0 2 -2 0 0 0 0]", "kqRWallFunction", "0.1"), ("epsilon", "[0 2 -3 0 0 0 0]", "epsilonWallFunction", "0.1"), ("nut", "[0 2 -1 0 0 0 0]", "nutkWallFunction", "0")]: open("0/" + n, "w").write("FoamFile { version 2.0; format ascii; class volScalarField; object %s; }\ndimensions %s;\ninternalField uniform %s;\nboundaryField { walls { type %s; value uniform %s; } }\n" % (n, dim, v, t, v))'
# kEpsilon (tests/interfoam_ami_vs_openfoam.sh) and kOmegaSST (tests/interfoam_moving_vs_openfoam.sh
# `pistonSST`) run on a moving mesh; a wallDist updateInterval other than 1 there is refused, and so is LES
MOVSST="sed -i 's/^simulationType .*/simulationType RAS;\nRAS { RASModel kOmegaSST; turbulence on; }/' constant/turbulenceProperties; sed -i 's/div(rhoPhi,U) .*/&\n    div(phi,k) Gauss upwind;\n    div(phi,omega) Gauss upwind;/' system/fvSchemes; printf '\\nwallDist { method meshWave; }\\n' >> system/fvSchemes; sed -i 's/(U|k|epsilon)/XX/; s/^    U$/    \"(U|k|omega).*\"/' system/fvSolution; python3 -c '${KEFIELDS//epsilon/omega}'"
arm moving_SST              runs    -                        "" "$MOVSST"
arm moving_SSTInterval      refused "updateInterval 2"       "" "$MOVSST; sed -i 's/^wallDist .*/wallDist { method meshWave; updateInterval 2; }/' system/fvSchemes"
arm moving_RAS              runs    -                        "" "sed -i 's/^simulationType .*/simulationType RAS;\nRAS { RASModel kEpsilon; turbulence on; }/' constant/turbulenceProperties; sed -i 's/div(rhoPhi,U) .*/&\n    div(phi,k) Gauss upwind;\n    div(phi,epsilon) Gauss upwind;/' system/fvSchemes; sed -i 's/(U|k|epsilon)/XX/; s/^    U$/    \"(U|k|epsilon).*\"/' system/fvSolution; python3 -c '$KEFIELDS'"
# a dictionary-form preconditioner other than GAMG or DIC is substituted under a notice, not run silently
arm moving_precondDILU      runs    "preconditioner { DILU ... }" "" "sed -i '/p_rghFinal/,/^    }/ s/preconditioner  *GAMG;/preconditioner DILU;/' system/fvSolution"
arm moving_precondNoSmoother refused "names no \`smoother\`" "" "sed -i '/p_rghFinal/,/^    }/ {/smoother/d}' system/fvSolution"
# the tutorial's own vanLeerV runs (moving_baseline), and so does limitedLinear, gated on
# eulerianInjection; the vector scheme brae still lacks does not
arm moving_limitedLinear    runs    -                        "" "sed -i 's/div(rhoPhi,U) .*/div(rhoPhi,U)  Gauss limitedLinear 1;/' system/fvSchemes"
arm moving_unknownScheme    refused "Gauss QUICKV"             "" "sed -i 's/div(rhoPhi,U) .*/div(rhoPhi,U)  Gauss QUICKV;/' system/fvSchemes"
BASE="$B"

# limitedLinear's coefficient has no default and must lie in [0, 1] (limitedLinear.H:67-76)
arm ll_runs                 runs    -                        "" "sed -i 's/div(rhoPhi,U) .*/div(rhoPhi,U)  Gauss limitedLinear 0.2;/' system/fvSchemes"
arm ll_noCoeff              refused "with no coefficient"      "" "sed -i 's/div(rhoPhi,U) .*/div(rhoPhi,U)  Gauss limitedLinear;/' system/fvSchemes"
arm ll_coeffAboveOne        refused "outside [0, 1]"           "" "sed -i 's/div(rhoPhi,U) .*/div(rhoPhi,U)  Gauss limitedLinear 1.5;/' system/fvSchemes"
arm ll_coeffNegative        refused "outside [0, 1]"           "" "sed -i 's/div(rhoPhi,U) .*/div(rhoPhi,U)  Gauss limitedLinear -0.1;/' system/fvSchemes"
# its limiter's gradient is the case's grad(magSqr(U)) entry, and a least-squares one is not ported
# ...nor a limited one reached through the default
arm ll_gradMagSqrDefault    refused "grad(magSqr(U)) default" "" "sed -i 's/div(rhoPhi,U) .*/div(rhoPhi,U)  Gauss limitedLinear 0.2;/' system/fvSchemes; sed -i '/^gradSchemes/,/^}/ s/default .*/default         cellLimited Gauss linear 1;/' system/fvSchemes"
arm ll_gradMagSqrLSQ        refused "grad(magSqr(U)) leastSquares" "" "sed -i 's/div(rhoPhi,U) .*/div(rhoPhi,U)  Gauss limitedLinear 0.2;/' system/fvSchemes; sed -i '/^gradSchemes/,/^}/ s/default .*/default         Gauss linear;\n    grad(magSqr(U)) leastSquares;/' system/fvSchemes"

# PIMPLE controls the HOST honours...
arm host_nOuter2            runs    -                        "" "sed -i 's/nOuterCorrectors  *1;/nOuterCorrectors 2;/' system/fvSolution"
arm host_nNonOrth1          runs    -                        "" "sed -i 's/nNonOrthogonalCorrectors  *0;/nNonOrthogonalCorrectors 1;/' system/fvSolution"

# ...and the DEVICE loop does not, so there they are refused rather than run as 1 and 0
# THE CYCLIC BAFFLE is ported in the host loop (tests/interfoam_baffle_vs_openfoam.sh): every operator,
# matrix and linear solver on RAS/damBreakPorousBaffle's path couples the pair, and p_rgh's
# porousBafflePressure carries its jump. What is NOT carried across a cyclic is refused, each by name --
# a solver without interface coefficients would run the pair as two walls and converge.
BASE="$BB"
PRGH="python3 -c \"import re; p='0/p_rgh'; t=open(p).read(); t=re.sub(r'(porous_half[01]\\s*\\{[^}]*?)length', r'\\1EXTRA length', t); open(p,'w').write(t)\""
arm baffle_runs             runs    -                                  "" true
arm baffle_plainCyclic      runs    -                                  "" "python3 -c \"import re; p='0/p_rgh'; t=open(p).read(); t=re.sub(r'(porous_half[01]\\s*\\{)[^}]*\\}', r'\\1 type cyclic; }', t); open(p,'w').write(t)\""
arm baffle_relax            refused "sets \`relax\` or \`minJump\`"      "" "${PRGH/EXTRA/relax 0.5;}"
arm baffle_minJump          refused "sets \`relax\` or \`minJump\`"      "" "${PRGH/EXTRA/minJump 0;}"
arm baffle_massFlux         refused "MASS flux"                        "" "${PRGH/EXTRA/phi rhoPhi;}"
arm baffle_DTable           refused "a Function1 other than"           "" "sed -i 's/^\\( *D  *\\)1000;/\\1table ((0 1000) (1 2000));/' 0/p_rgh"
arm baffle_noLength         refused "needs \`D\`, \`I\` and \`length\`"   "" "sed -i '/^ *length  *0.15;/d' 0/p_rgh"
arm baffle_noJump           refused "has no \`jump\` entry"             "" "sed -i '/^ *jump  *uniform 0;/d' 0/p_rgh"
arm baffle_GAMG             refused "GAMG does not carry the interface" "" "python3 -c \"import re; p='system/fvSolution'; t=open(p).read(); t=re.sub(r'(\\n    p_rgh\\s*\\{\\s*solver\\s+)PCG;\\s*preconditioner\\s+DIC;', r'\\1GAMG; smoother DIC;', t); open(p,'w').write(t)\""
arm baffle_momentumPredictor refused "a momentum predictor across the coupled patch" "" "sed -i 's/momentumPredictor  *no;/momentumPredictor   yes;/; /^ *minIter  *1;/d' system/fvSolution"
arm baffle_cellLimitedGradU refused "grad(U) across the coupled patch" "" "sed -i '/^gradSchemes/,/^}/ s/default .*/default         Gauss linear;\n    grad(U)         cellLimited Gauss linear 1;/' system/fvSchemes"
arm baffle_vanLeerV         refused "does not carry them onto the coupled patch" "" "sed -i 's/div(rhoPhi,U)  *Gauss linearUpwind grad(U);/div(rhoPhi,U)   Gauss vanLeerV;/' system/fvSchemes"
arm baffle_compression      refused "\`interfaceCompression\` across the coupled patch" "" "sed -i 's/div(phirb,alpha)  *Gauss linear;/div(phirb,alpha) Gauss interfaceCompression;/' system/fvSchemes"
# every other gradient entry is gated on damBreak, which has no coupled patch
arm baffle_cellLimitedPrgh  refused "grad(p_rgh) cellLimited across the coupled patch" "" "sed -i '/^gradSchemes/,/^}/ s/default .*/default         Gauss linear;\n    grad(p_rgh)     cellLimited Gauss linear 1;/' system/fvSchemes"
arm baffle_leastSquaresNHat refused "nHat leastSquares across the coupled patch" "" "sed -i '/^gradSchemes/,/^}/ s/default .*/default         Gauss linear;\n    nHat            leastSquares;/' system/fvSchemes"
BASE="$B"
# ...and what writing that arm found: `Gauss interfaceCompression vanLeer 1` is ANOTHER scheme, which a
# substring match read as plain vanLeer and ran
arm compressionNew_refused  refused "limited scheme with a compression coefficient" "" "sed -i 's/div(phi,alpha)  *Gauss vanLeer;/div(phi,alpha)  Gauss interfaceCompression vanLeer 1;/' system/fvSchemes"

# LES kEqn is ported in the host loop (tests/interfoam_les_vs_openfoam.sh). What it refuses, by name:
BASE="$BL"
TP=constant/turbulenceProperties
arm les_runs                runs    -                                  "" true
arm les_smagorinsky         refused "LESModel \`Smagorinsky\`"         "" "sed -i 's/LESModel  *kEqn;/LESModel Smagorinsky;/' $TP"
arm les_deltaVanDriest      refused "LES delta \`vanDriest\`"          "" "sed -i 's/^\\( *\\)delta  *smooth;/\\1delta vanDriest;/' $TP"
arm les_smoothPrandtl       refused "smooths the LES delta \`Prandtl\`" "" "python3 -c \"import re; p='$TP'; t=open(p).read(); t=re.sub(r'(\\nsmoothCoeffs|\\n    smoothCoeffs)(\\s*\\{\\s*)delta\\s+cubeRootVol;', r'\\1\\2delta Prandtl;', t); open(p,'w').write(t)\""
arm les_noMaxDeltaRatio     refused "no \`maxDeltaRatio\`"            "" "python3 -c \"import re; p='$TP'; t=open(p).read(); i=t.index('\\n    smoothCoeffs'); j=t.index('maxDeltaRatio', i); t=t[:j]+'// '+t[j:]; open(p,'w').write(t)\""
arm les_densityVariable     refused "pairs \`density variable\` with LES" "" "sed -i 's/^simulationType .*/simulationType LES;\\ndensity variable;/' $TP"
arm les_linearUpwindK       refused "neither \`Gauss upwind\` nor"     "" "sed -i 's/div(phi,k)  *Gauss limitedLinear 1;/div(phi,k) Gauss linearUpwind grad(k);/' system/fvSchemes"
arm les_cellLimitedGradU    refused "the LES closure computes plain \`Gauss linear\` only" "" "sed -i 's/^\\( *default  *\\)Gauss linear;/\\1cellLimited Gauss linear 1;/' system/fvSchemes"
BASE="$B"

# the mangrove fvOptions: they run under kEpsilon with PBiCG; each coefficient OpenFOAM reads with
# readEntry is required, a missing zone is refused, a field-name override and another closure are not taken
BASE="$BG"
MG_SRC="python3 -c \"import re; p='system/fvOptions'; t=open(p).read(); i=t.index('TurbulenciaMangroves'); "
arm mg_runs                 runs    -                        "" true
arm mg_noZone               refused "names cellZone \`c9\`"   "" "sed -i '0,/cellZone        c0;/s//cellZone        c9;/' system/fvOptions"
arm mg_noCd                 refused "has no \`Cd\`"           "" "sed -i '0,/Cd              1.52;/s///' system/fvOptions"
arm mg_UNames               refused "UNames"                 "" "sed -i '0,/regions/s//UNames (U);\n        regions/' system/fvOptions"
arm mg_epsilonNames         refused "epsilonNames"           "" "${MG_SRC}t=t[:i]+t[i:].replace('regions', 'epsilonNames (epsilon);\\n        regions', 1); open(p,'w').write(t)\""
arm mg_laminar              refused "multiphaseMangrovesTurbulenceModel" "" "sed -i 's/^simulationType .*/simulationType laminar;/' constant/turbulenceProperties"
arm mg_kPBiCGStab           refused "PBiCGStab"              "" "sed -i 's/solver  *PBiCG;/solver          PBiCGStab;/' system/fvSolution"
BASE="$B"

# the coded cyclicACMI baffle: it runs; the rescale point is ported for MULESCorr and one sub-cycle only;
# the coded scale refuses what OpenFOAM would compile against its own headers, and a name its shim lacks
BASE="$BK"
arm leak_runs               runs    -                        "" true
arm leak_explicitMULES      refused "MULESCorr no"           "" "sed -i 's/MULESCorr  *yes;/MULESCorr       no;/' system/fvSolution"
arm leak_subCycles          refused "nAlphaSubCycles 2"      "" "sed -i 's/nAlphaSubCycles  *1;/nAlphaSubCycles 2;/' system/fvSolution"
arm leak_codeInclude        refused "codeInclude"            "" "sed -i '0,/type            coded;/s//type            coded;\n            codeInclude #{ #};/' constant/polyMesh/boundary"
arm leak_timeIndex          refused "did not compile"        "" "sed -i 's/this->time().value()/scalar(this->time().timeIndex())/' constant/polyMesh/boundary"
arm leak_noNonOverlap       refused "nonOverlapPatch"        "" "sed -i '0,/nonOverlapPatch wall_block;/s///' constant/polyMesh/boundary"
BASE="$B"

if [ $HAVE_GPU = 1 ]; then
    BASE="$BL"
    # LES kEqn RUNS on the device now, on a WEDGE mesh (tests/interfoam_les_vs_openfoam.sh holds the
    # numbers). It was refused twice -- for the closure, then for a momentum gap that turned out to be
    # three wedge defects -- so this arm is a `runs`, and a blanket refusal coming back fails it
    arm device_les          runs    -                      "-device" true
    # the MANGROVE PAIR RUNS on the device now, with k and epsilon under the PBiCG/DILU the case names
    # (tests/interfoam_mangrove_vs_openfoam.sh holds both arms to OpenFOAM, solve by solve). It was a
    # refusal by the option's name, and behind that refusal the closure would have run a Gauss-Seidel
    # sweep under PBiCG's entry -- so this arm is a `runs`, and a blanket refusal coming back fails it
    BASE="$BG"
    arm device_mangrove     runs    -                       "-device" true
    # ...and the turbulence option under the `density variable` k-epsilon, where OpenFOAM's addSup is
    # -Sp(rho*coeff): the device names it before the first step (the host closure refuses the same
    # lineage where it meets it)
    arm device_mangrove_rhoKE refused "density variable"    "-device" "sed -i 's/^simulationType .*/density variable;\nsimulationType RAS;/' constant/turbulenceProperties; sed -i 's/div(phi,k) /div(rhoPhi,k) /; s/div(phi,epsilon) /div(rhoPhi,epsilon) /' system/fvSchemes"
    # the device's alpha pre-solve does not honour minIter (the host's does)
    BASE="$B"
    arm device_alphaMinIter    refused "minIter 1"               "-device" "sed -i 's/^\\( *\\)MULESCorr  *yes;/\\1MULESCorr       yes;\\n\\1minIter 1;/' system/fvSolution"
    # the device loop RUNS the coded cyclicACMI baffle now: the binary couples the pair for it as for
    # the host loop, and tests/interfoam_leakage_vs_openfoam.sh holds the numbers (its harness still
    # asserts that the pair handed over UNCOUPLED is refused, which this binary can no longer do)
    BASE="$BK"
    arm device_leak         runs    -                       "-device" true
    # ...but only at the rescale point both loops gate -- `MULESCorr yes`, one alpha sub-cycle, no
    # icAlpha or scAlpha -- so the explicit path is refused on the device as it is on the host
    arm device_leak_explicit refused "MULESCorr"            "-device" "sed -i 's/^\\( *\\)MULESCorr  *yes;/\\1MULESCorr       no;/' system/fvSolution"
    # the device momentum runs limitedLinear now -- one magSqr limiter per face, as OpenFOAM's, gated on
    # eulerianInjection in tests/interfoam_limitedlinear_vs_openfoam.sh. It was refused here while its
    # branch accumulated magSqr(U) into a buffer resize() had not zeroed
    BASE="$B"
    arm device_limitedLinear runs    -                        "-device" "sed -i 's/div(rhoPhi,U) .*/div(rhoPhi,U)  Gauss limitedLinear 0.2;/' system/fvSchemes"
    BASE="$B"
    # THE BAFFLE TUTORIAL on the device. Its pair and its JUMP both run now
    # (tests/interfoam_cyclic_vs_openfoam.sh's `jump` profile measures them), so what it refuses is
    # the one thing left: the case sets `nOuterCorrectors 3` and the device loop runs one.
    BASE="$BB"
    # the baffle tutorial RUNS on the device now -- pair, jump, nOuterCorrectors 3 and a RAS closure
    # whose k and epsilon cross the pair; tests/interfoam_baffle_vs_openfoam.sh measures it against
    # OpenFOAM. The shipped binary attaches the coupling itself, so this arm runs; the mesh handed over
    # UNCOUPLED is still refused by name, and test_inter_baffle_vs_openfoam.cu holds that refusal.
    arm device_baffle       runs    -                        "-device" true
    BASE="$B"
    # the device's gradient operators are Gauss linear; a limited or least-squares one is refused
    arm device_gradLsq      refused "leastSquares or cellLimited" "-device" "sed -i '/^gradSchemes/,/^}/ s/default .*/default         leastSquares;/' system/fvSchemes"
    arm device_gradNHat     refused "leastSquares or cellLimited" "-device" "sed -i '/^gradSchemes/,/^}/ s/default .*/default         Gauss linear;\n    nHat            cellLimited Gauss linear 1;/' system/fvSchemes"
    arm device_baseline     runs    -                        "-device" true
    # CrankNicolson RUNS on the device loop (tests/interfoam_cn_vs_openfoam.sh holds RAS/damBreak under it
    # on both arms); a coupled pair under it is refused by name there, where the host loop carries it
    arm device_cn           runs    -                        "-device" "$CNSET"
    BASE="$BB"
    arm device_cn_baffle    refused "coupled pair"            "-device" "$CNSET"
    BASE="$B"
    # the permeable-wall pair RUNS on the device loop now (tests/interfoam_permeable_vs_openfoam.sh holds
    # it to OpenFOAM on both profiles); this arm is here because it was a blanket refusal
    arm device_permeable    runs    -                        "-device" "${PERMU/PHI /}; ${PERMP/PENTRY/p uniform 0;}"
    # nOuterCorrectors IS the device loop now (tests/interfoam_cyclic_vs_openfoam.sh's `outer`
    # profile measures it against OpenFOAM, with the one-corrector answer as its control); what is
    # still refused is frozenFlow, which skips the momentum, the pressure AND the turbulence corrector
    arm device_nOuter2      runs    -                        "-device" "sed -i 's/nOuterCorrectors  *1;/nOuterCorrectors 2;/' system/fvSolution"
    arm device_frozenFlow   refused "solveFlow no"           "-device" "sed -i 's/nOuterCorrectors  *1;/nOuterCorrectors 1;\n    solveFlow       no;/' system/fvSolution"
    # the device pressure step runs the non-orthogonal loop (laminar/damBreak `nonorth` holds it)
    arm device_nNonOrth1    runs    -                        "-device" "sed -i 's/nNonOrthogonalCorrectors  *0;/nNonOrthogonalCorrectors 1;/' system/fvSolution"
    arm device_mesh_dynamic refused "dynamicRefineFvMesh"     "-device" "printf '%s\ndynamicFvMesh dynamicRefineFvMesh;\n' '$HDR' > constant/dynamicMeshDict"
    # ...nor a non-orthogonal correction where it is not zero
    # the non-orthogonal correction runs on the device now (tests/interfoam_dambreak_vs_openfoam.sh
    # `sheared` holds it to OpenFOAM); `uncorrected` on a mesh that is not orthogonal is still refused
    arm device_sheared_corrected runs    -                        "-device" "$SHEAR"
    arm device_sheared_uncorrected refused "uncorrected"          "-device" "$SHEAR && $UNCORR"
    # the device loop moves no mesh and pins no pressure reference; both are refused by name there
    BASE="$BM"
    # the reason is now the SPECIFIC one -- the mesh-update stage is on the host loop only -- because
    # the device loop carries the pieces around it (the ddt's V0, refreshDeviceMeshGeometry) and a
    # caller passing a MutableMesh must not get a silent run on the mesh as it started
    # a moving mesh RUNS on the device now, AS SHIPPED: testTubeMixer's p_rgh is `solver GAMG` and its
    # p_rghFinal is `solver PCG; preconditioner { preconditioner GAMG; ... }`, and the loop runs both
    # (tests/interfoam_moving_vs_openfoam.sh profiles `mixer`, `cylinder` and `solitaryGamg`). This
    # arm is here because each of the three was a refusal in turn, and a blanket one would pass every
    # other arm on this page
    arm device_moving       runs    -                      "-device" true
    # ...and the same mesh with p_rghFinal as a plain GAMG SOLVER rather than the preconditioner form,
    # so that BOTH device GAMG entry points are held on a moving mesh (the hierarchy is the mesh's and
    # is rebuilt on every move for either)
    arm device_moving_gamg  runs    -                      "-device" "python3 '$W/setSolver.py' p_rghFinal '        solver          GAMG;\n        smoother        DIC;\n        tolerance       2e-09;\n        relTol          0;\n'"
    # ...and the PERMEABLE-WALL pair on it is refused, by name: the pressure half reads the flux as it
    # stands at constrainPressure, which on a moving mesh this loop makes relative at another point
    # than the host loop. On a static mesh the pair runs -- device_permeable, below, on the base case.
    PERMUW="python3 -c \"import re; p='0/U'; t=open(p).read(); t=re.sub(r'walls\\s*\\{[^}]*\\}', 'walls { type permeableAlphaPressureInletOutletVelocity; alpha alpha.water; alphaMin 0.01; value uniform (0 0 0); }', t, count=1); open(p,'w').write(t)\""
    PERMPW="python3 -c \"import re; p='0/p_rgh'; t=open(p).read(); t=re.sub(r'walls\\s*\\{[^}]*\\}', 'walls { type prghPermeableAlphaTotalPressure; alpha alpha.water; alphaMin 0.01; p uniform 0; value uniform 0; }', t, count=1); open(p,'w').write(t)\""
    arm device_permeable_moving refused "and the mesh moves" "-device" "$PERMUW; $PERMPW"
    BASE="$B"
    # a case that needs a pressure reference RUNS on the device now (gated on laminar/mixerVessel2D,
    # where every patch is a wall); this one keeps a pressure-driven atmosphere, which is what adjustPhi
    # would have to weigh, so it stays refused -- by that name now, not by the reference's
    arm device_closed       refused "adjustPhi"               "-device" "sed -i '/atmosphere/,/}/ s/type  *totalPressure;/type            fixedFluxPressure;/' 0/p_rgh; sed -i '/nNonOrthogonalCorrectors/a\    pRefPoint (0.292 0.292 0.0073);\n    pRefValue 0;' system/fvSolution"
    BASE="$B"
    # the device loop carries the kEpsilon closure now, in both lineages
    # the device loop drives the wave conditions through its alpha and velocity hooks
    BASE="$BW"
    # ...WITH THE TUTORIAL'S OWN GAMG for p_rghFinal, and so under no notice about the p_rgh solve: a
    # `runs` arm cannot say "and did not substitute", so that is checked on the output directly
    arm device_waves        runs    -                        "-device" true
    if "$BIN" -case "$W/device_waves" -device 2>&1 | grep -q "approximated.*p_rgh"; then
        echo "  FAIL: device_waves                       ran GAMG's entry under a p_rgh substitution notice"
        fails=$((fails+1))
    else
        echo "  ok:   device_waves                       no p_rgh substitution notice"
    fi
    # the device's GAMG runs the host's four smoothers now (tests/interfoam_gamg_vs_openfoam.sh's three
    # Gauss-Seidel profiles have device arms); a smoother NONE of them names is still refused there
    arm device_gamg_GaussSeidel runs    -                        "-device" "sed -i '/p_rghFinal/,/}/ s/smoother  *DIC;/smoother        GaussSeidel;/' system/fvSolution"
    arm device_gamg_symGaussSeidel runs -                        "-device" "sed -i '/p_rghFinal/,/}/ s/smoother  *DIC;/smoother        symGaussSeidel;/' system/fvSolution"
    arm device_gamg_smootherDILU refused "smoother DILU"         "-device" "sed -i '/p_rghFinal/,/}/ s/smoother  *DIC;/smoother        DILU;/' system/fvSolution"
    arm device_gamg_sweeps  runs    -                        "-device" "${GE}nPreSweeps 2; nFinestSweeps 3;/' system/fvSolution"
    # a p_rgh or alpha condition that names rhoPhi is evaluated on the host and handed rhoPhi; U's
    # pressureInletOutletVelocity switch runs ON the device and reads phi, so there the name is refused
    BASE="$B"
    arm device_flux_rhoPhi  runs    -                        "-device" "sed -i '/totalPressure/a\        phi             rhoPhi;' 0/p_rgh"
    arm device_Uflux_rhoPhi refused "names the flux"          "-device" "sed -i '/pressureInletOutletVelocity/a\        phi             rhoPhi;' 0/U"
    arm host_Uflux_rhoPhi   runs    -                        "" "sed -i '/pressureInletOutletVelocity/a\        phi             rhoPhi;' 0/U"
    BASE="$BR"
    arm device_ras          runs    -                        "-device" true
    arm device_ras_uniform  runs    -                        "-device" "sed -i 's/^density .*/density uniform;/' constant/turbulenceProperties; sed -i 's/div(rhoPhi,k) /div(phi,k) /; s/div(rhoPhi,epsilon) /div(phi,epsilon) /' system/fvSchemes"
    arm device_ras_otherModel refused "realizableKE"         "-device" "sed -i 's/RASModel .*/RASModel        realizableKE;/' constant/turbulenceProperties"
    # the device loop carries kOmegaSST too now (tests/interfoam_ras_dambreak_vs_openfoam.sh `sst`), and
    # an inletOutlet nut with it -- evaluated against the flux after the closure, as the host closure does
    # (tests/interfoam_ras_dambreak_vs_openfoam.sh `nutAtmosphere`, on both closures)
    arm device_sst          runs    -                        "-device" "$SSTBASE"
    arm device_sstNutIO     runs    -                        "-device" "$SSTBASE; python3 -c \"import re; p='0/nut'; t=open(p).read(); t=re.sub(r'atmosphere\s*\{[^}]*\}', 'atmosphere { type inletOutlet; inletValue uniform 0.001; value uniform 0; }', t, count=1); open(p,'w').write(t)\""
    BASE="$B"
else
    echo "  (no GPU: the -device arms are skipped)"
fi

echo "interfoam_refusals: $fails failures"
[ $fails = 0 ]
