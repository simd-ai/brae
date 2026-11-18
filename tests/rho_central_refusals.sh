#!/usr/bin/env bash
# brae_rhoCentralFoam refuses, at start-up and by name, every input it does not run -- on OpenFOAM
# v2412's own rhoCentralFoam tutorials and on single-edit variants of shockTube. Each row: a case, the edit, the
# words the refusal must contain. A refusal must exit non-zero and write no time directory.
#
# The shipped tutorials run as shipped and stop on their first unported input. wedge15Ma5 and obliqueShock run since
# milestone C (tests/rho_central_wedge_vs_openfoam.sh), forwardStep since milestone D, which ported inletOutlet and
# purgeWrite (tests/rho_central_forwardstep_vs_openfoam.sh); one row here checks OpenFOAM's pairing of constraint patch
# and field types on wedge15Ma5, three check what inletOutlet's port leaves out, and three (milestone E) a start
# directory without the fields OpenFOAM must read: brae names it where OpenFOAM stops on it, never starting from an older
# directory instead. What this does NOT show: that the
# refused inputs are rare (they are not -- one of the seven shipped tutorials, movingCone, is refused today, after
# milestones H, I and J; ARCHITECTURE.md lists the order of the work).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BRAE_RCF_BIN:-$ROOT/build/brae_rhoCentralFoam}"
OFBASHRC=${OF_BASHRC:-${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}}
TUT="${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}/compressible/rhoCentralFoam"
[ -x "$BIN" ] || { echo "SKIP: no brae_rhoCentralFoam at $BIN"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: no OpenFOAM v2412 at $OFBASHRC"; exit 77; }
[ -d "$TUT/shockTube" ] || { echo "SKIP: no rhoCentralFoam tutorials at $TUT"; exit 77; }
W=${BRAE_GATE_OUT:-$(mktemp -d /tmp/brae-rcf-refusals.XXXXXX)}
mkdir -p "$W"
fails=0
passes=0
of()
{
    bash -c "set +u; source '$OFBASHRC' > /dev/null 2>&1; set -e; $*"
}
# row <name> <tutorial> <mesh: yes|no> <expected words> [sed-edit file expression]...
row()
{
    local name=$1 tut=$2 mesh=$3 want=$4
    shift 4
    local d="$W/$name"
    rm -rf "$d"
    cp -r "$TUT/$tut" "$d"
    [ -d "$d/0" ] || cp -r "$d/0.orig" "$d/0"
    while [ $# -ge 2 ]
    do
        sed -i -E "$2" "$d/$1"
        shift 2
    done
    if [ "$mesh" = yes ]; then of "cd '$d' && blockMesh > log.blockMesh 2>&1" < /dev/null || { echo "FAIL $name: blockMesh"; fails=$((fails + 1)); return; }; fi
    (cd "$d" && timeout 120 "$BIN" -case . > log 2>&1 < /dev/null; echo $? > rc)
    local written
    written=$(find "$d" -maxdepth 1 -type d -name '[0-9]*' ! -name 0 ! -name '0.orig' | head -n 1)
    if [ "$(cat "$d/rc")" != 0 ] && grep -qF -- "$want" "$d/log" && [ -z "$written" ]
    then
        passes=$((passes + 1))
        echo "ok   $name: $(grep -o 'brae rhoCentralFoam: ERROR: .*' "$d/log" | cut -c29-200)"
    else
        fails=$((fails + 1))
        echo "FAIL $name: exit $(cat "$d/rc"), wanted \"$want\"${written:+, wrote $written}"
        tail -n 3 "$d/log" | sed 's/^/     /'
    fi
}
# milestone I: biconic25-55Run35 runs as shipped (tests/rho_central_lts_vs_openfoam.sh); what its port leaves out
row sod_janafConst shockTube no "transport with janaf: OpenFOAM v2412 has no such psiThermo" \
    constant/thermophysicalProperties 's/hConst/janaf/'
row sod_ddtMixed shockTube no "a mix of localEuler and Euler terms is not ported" \
    system/fvSchemes 's/^(\s*default\s+)Euler;/\1localEuler; ddt(rho) Euler;/'
row sod_waveTransmissiveLTS shockTube yes "waveTransmissive with ddtSchemes localEuler" \
    system/fvSchemes 's/^(\s*default\s+)Euler;/\1localEuler;/' \
    0/p '/sides/,/\}/ s/zeroGradient;/waveTransmissive; psi thermo:psi; gamma 1.4; value uniform 1;/'
row sod_maxwellSlipUInviscid shockTube yes "on a viscous case only" \
    0/U '/sides/,/\}/ s/zeroGradient;/maxwellSlipU; accommodationCoeff 0.85; Uwall uniform (0 0 0); value uniform (0 0 0);/'
row sod_smoluchowskiNonuniform shockTube yes "must be \`uniform <value>\` here" \
    constant/thermophysicalProperties 's/^(\s*mu\s+)\S+;/\11.8e-05;/' system/fvSolution 's/^(\s*)h$/\1"(e|h)"/' \
    0/T '/sides/,/\}/ s/zeroGradient;/smoluchowskiJumpT; accommodationCoeff 0.85; Twall nonuniform List<scalar> 2(300 300); value uniform 300;/'
# the nozzle boundaries (milestone H): their forms and fields beyond the ported ones refused by name (LadenburgJet60psi,
# sutherland and all, now runs: tests/rho_central_nozzle_vs_openfoam.sh)
row sod_totalPressurePsiNone shockTube yes "is \`totalPressure\` with psi \`none\`" \
    0/p '/sides/,/\}/ s/zeroGradient;/totalPressure; p0 uniform 1; value uniform 1;/'
row sod_totalPressurePhiv shockTube yes "is \`totalPressure\` on \`phi phiv\`" \
    0/p '/sides/,/\}/ s/zeroGradient;/totalPressure; psi thermo:psi; gamma 1.4; phi phiv; p0 uniform 1; value uniform 1;/'
row sod_totalTemperatureNoT0 shockTube yes "is \`totalTemperature\` without its \`T0\`" \
    0/T '/sides/,/\}/ s/zeroGradient;/totalTemperature; psi thermo:psi; gamma 1.4; value uniform 300;/'
row sod_waveTransmissiveOnT shockTube yes "T patch 'sides' is \`waveTransmissive\`" \
    0/T '/sides/,/\}/ s/zeroGradient;/waveTransmissive; field T; psi thermo:psi; gamma 1.4; value uniform 300;/'
row sod_waveTransmissiveLInfNoFieldInf shockTube yes "with lInf and no fieldInf" \
    0/p '/sides/,/\}/ s/zeroGradient;/waveTransmissive; psi thermo:psi; gamma 1.4; lInf 1; value uniform 1;/'
row movingCone_shipped movingCone no "dynamicMeshDict selects"
row sod_fluxScheme shockTube no "fluxScheme \`Bogus\` is not a valid choice" \
    system/fvSchemes 's/^(fluxScheme\s+)\S+;/\1Bogus;/'
row sod_reconstruct shockTube no "reconstruct(rho) to \`QUICK\`" \
    system/fvSchemes 's/^(\s*reconstruct\(rho\)\s+)\S+;/\1QUICK;/'
row sod_reconstructUScalarForm shockTube no "on a vector, OpenFOAM's scalar-form limiter is built from magSqr(rhoU)" \
    system/fvSchemes 's/^(\s*reconstruct\(U\)\s+)\S+;/\1Minmod;/'
row sod_reconstructVOnScalar shockTube no "a V scheme is defined for vectors only" \
    system/fvSchemes 's/^(\s*reconstruct\(rho\)\s+)\S+;/\1MinmodV;/'
row sod_limitedLinearK shockTube no "should be >= 0 and <= 1" \
    system/fvSchemes 's/^(\s*reconstruct\(T\)\s+)\S+;/\1limitedLinear 1.5;/'
row sod_grad shockTube no "grad(rho) to \`leastSquares\`" \
    system/fvSchemes 's/^(\s*default\s+)Gauss linear;/\1leastSquares;/'
# mu > 0 runs the viscous correctors (milestone G): the tutorials' fvSolution names `h`, so OpenFOAM stops at its first
# e solve; brae refuses before reading a field. Then what the correctors do not port, one edit past the e entry.
row sod_viscous shockTube no "has no entry for e" \
    constant/thermophysicalProperties 's/^(\s*mu\s+)\S+;/\11.8e-05;/'
row sod_viscousUncorrected shockTube no "resolves laplacian(muEff,U) to \`Gauss linear uncorrected\`" \
    constant/thermophysicalProperties 's/^(\s*mu\s+)\S+;/\11.8e-05;/' system/fvSolution 's/^(\s*)h$/\1"(e|h)"/' \
    system/fvSchemes 's/^(\s*default\s+)Gauss linear corrected;/\1Gauss linear uncorrected;/'
row sod_viscousSmoother shockTube no "smoother \`DIC\` for U" \
    constant/thermophysicalProperties 's/^(\s*mu\s+)\S+;/\11.8e-05;/' system/fvSolution 's/^(\s*)h$/\1"(e|h)"/' \
    system/fvSolution 's/^(\s*smoother\s+)GaussSeidel;/\1DIC;/'
row sod_RAS shockTube no "simulationType RAS with a laminar mu of 0" \
    constant/turbulenceProperties 's/^(simulationType\s+)\S+;/\1RAS;/'
row sod_partialSlip shockTube yes "is \`partialSlip\`" 0/U '/sides/,/\}/ s/zeroGradient/partialSlip/'
row sod_symmetryPlaneOnPatch shockTube yes "is \`symmetryPlane\` on a mesh patch of type \`patch\`" \
    0/U '/sides/,/\}/ s/zeroGradient/symmetryPlane/'
row wedge15Ma5_zeroGradientOnSymmetryPlane wedge15Ma5 yes \
    "is \`zeroGradient\` on a mesh patch of type \`symmetryPlane\`" 0/p '/bottom/,/\}/ s/symmetryPlane/zeroGradient/'
row sod_enthalpy shockTube no "energy is \`sensibleEnthalpy\`" \
    constant/thermophysicalProperties 's/sensibleInternalEnergy/sensibleEnthalpy/'
row sod_clockTime shockTube no "writeControl \`clockTime\` is not ported" \
    system/controlDict 's/^(writeControl\s+)\S+;/\1clockTime;/'
row sod_outletInlet shockTube yes "is \`outletInlet\`" 0/U '/sides/,/\}/ s/zeroGradient;/outletInlet; outletValue uniform (0 0 0);/'
row sod_inletOutletPhi shockTube yes "is \`inletOutlet\` on \`phi phiEp\`" \
    0/T '/sides/,/\}/ s/zeroGradient;/inletOutlet; inletValue uniform 300; phi phiEp;/'
row sod_inletOutletNoInletValue shockTube yes "is \`inletOutlet\` without \`inletValue\`" \
    0/p '/sides/,/\}/ s/zeroGradient;/inletOutlet;/'
# milestone J: kOmegaSST runs (tests/rho_central_turb_vs_openfoam.sh); what its port leaves out, on its cases.
# turbrow <name> <kind> <expected words> [file sed-expression | rm file]...: the case staged as that gate stages it
# (LadenburgJet60psi's system/ and constant/, its 0.orig for jet_sst, tests/rho_central_turb.py setup <kind>), then edited
turbrow()
{
    local name=$1 kind=$2 want=$3
    shift 3
    local d="$W/$name"
    rm -rf "$d"
    mkdir -p "$d"
    cp -r "$TUT/LadenburgJet60psi/system" "$TUT/LadenburgJet60psi/constant" "$d/"
    if [ "$kind" = jet_sst ]; then cp -r "$TUT/LadenburgJet60psi/0.orig" "$d/0"; gunzip -f "$d/0/"*.gz; fi
    python3 "$ROOT/tests/rho_central_turb.py" setup "$kind" "$d" || { echo "FAIL $name: setup"; fails=$((fails + 1)); return; }
    while [ $# -ge 2 ]
    do
        if [ "$1" = rm ]; then rm -f "$d/$2"; else sed -i -E "$2" "$d/$1"; fi
        shift 2
    done
    of "cd '$d' && blockMesh > log.blockMesh 2>&1" < /dev/null || { echo "FAIL $name: blockMesh"; fails=$((fails + 1)); return; }
    (cd "$d" && timeout 120 "$BIN" -case . > log 2>&1 < /dev/null; echo $? > rc)
    local written
    written=$(find "$d" -maxdepth 1 -type d -name '[0-9]*' ! -name 0 | head -n 1)
    if [ "$(cat "$d/rc")" != 0 ] && grep -qF -- "$want" "$d/log" && [ -z "$written" ]
    then
        passes=$((passes + 1))
        echo "ok   $name: $(grep -o 'brae rhoCentralFoam: ERROR: .*' "$d/log" | cut -c29-220)"
    else
        fails=$((fails + 1))
        echo "FAIL $name: exit $(cat "$d/rc"), wanted \"$want\"${written:+, wrote $written}"
        tail -n 3 "$d/log" | sed 's/^/     /'
    fi
}
turbrow jet_kEpsilon jet_sst "RAS model \`kEpsilon\`; ported: kOmegaSST" constant/turbulenceProperties 's/kOmegaSST/kEpsilon/'
turbrow jet_LES jet_sst "simulationType is \`LES\`; laminar and RAS (kOmegaSST, milestone J) are ported" constant/turbulenceProperties 's/^simulationType(\s+)RAS;/simulationType\1LES;/'
turbrow jet_turbulenceOff jet_sst "RAS turbulence off" constant/turbulenceProperties 's/^(\s*turbulence\s+)on;/\1off;/'
turbrow jet_kIntensityInlet jet_sst "k patch 'inlet' is \`turbulentIntensityKineticEnergyInlet\`" \
    0/k '0,/fixedValue;/ s/fixedValue;/turbulentIntensityKineticEnergyInlet; intensity 0.01;/'
turbrow jet_wallDistPoisson jet_sst "wallDist method \`Poisson\`; ported: meshWave" system/fvSchemes 's/method meshWave;/method Poisson;/'
turbrow jet_divMixed jet_sst "name different schemes" system/fvSchemes 's/^(\s*div\(phi,omega\)\s+)Gauss limitedLinear 1;/\1Gauss upwind;/'
turbrow jet_noK jet_sst "the start time directory has no k" rm 0/k
turbrow plate_ltsBounded plate_lts "a bounded div(phi,k|omega) under localEuler" \
    system/fvSchemes 's/^(\s*div\(phi,(k|omega)\)\s+)Gauss/\1bounded Gauss/'
# rowx <name> <tutorial> <expected words> <shell run in the case after blockMesh> -- and OpenFOAM must stop on the same case
rowx()
{
    local name=$1 tut=$2 want=$3 setup=$4
    local d="$W/$name"
    rm -rf "$d"
    cp -r "$TUT/$tut" "$d"
    [ -d "$d/0" ] || cp -r "$d/0.orig" "$d/0"
    of "cd '$d' && blockMesh > log.blockMesh 2>&1" < /dev/null || { echo "FAIL $name: blockMesh"; fails=$((fails + 1)); return; }
    (cd "$d" && eval "$setup")
    rm -rf "$d.of"
    cp -r "$d" "$d.of"
    local before
    before=$(cd "$d" && ls -d [0-9]* 2> /dev/null | sort | tr '\n' ' ')
    (cd "$d" && timeout 120 "$BIN" -case . > log 2>&1 < /dev/null; echo $? > rc)
    of "cd '$d.of' && timeout 120 rhoCentralFoam > log 2>&1" < /dev/null
    local ofrc=$?
    local after
    after=$(cd "$d" && ls -d [0-9]* 2> /dev/null | sort | tr '\n' ' ')
    if [ "$(cat "$d/rc")" != 0 ] && grep -qF -- "$want" "$d/log" && [ "$before" = "$after" ] && [ "$ofrc" != 0 ]
    then
        passes=$((passes + 1))
        echo "ok   $name: $(grep -o 'brae rhoCentralFoam: ERROR: .*' "$d/log" | cut -c29-220) (OpenFOAM stops too: $(grep -m 1 -o 'cannot find file[^\"]*\|cannot find.*' "$d.of/log" | cut -c1-60))"
    else
        fails=$((fails + 1))
        echo "FAIL $name: exit $(cat "$d/rc"), wanted \"$want\"; OpenFOAM exit $ofrc"
        tail -n 3 "$d/log" | sed 's/^/     /'
    fi
}
LATEST='sed -i -E "s/^(startFrom\s+)\S+;/\1latestTime;/" system/controlDict'
rowx sod_latestTimeEmpty shockTube "selects time directory 0.004" "$LATEST; mkdir 0.004"
rowx sod_latestTimeNoT shockTube "has no T" "$LATEST; mkdir 0.004; cp 0/U 0/p 0.004/"
rowx sod_startTimeNoT shockTube "the start time directory 0 has no T" "rm 0/T"
echo "rho_central_refusals: $passes passed, $fails failed (work in $W)"
[ "$fails" = 0 ] && [ "$passes" -ge 28 ]
