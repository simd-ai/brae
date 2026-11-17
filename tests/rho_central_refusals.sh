#!/usr/bin/env bash
# brae_rhoCentralFoam refuses, at start-up and by name, every input its first stage does not run -- on OpenFOAM
# v2412's own rhoCentralFoam tutorials and on single-edit variants of shockTube. Each row: a case, the edit, the
# words the refusal must contain. A refusal must exit non-zero and write no time directory.
#
# The shipped tutorials stop on their first unported input; a row with `adjustTimeStep no` steps past that to show
# the next one is also named. What this does NOT show: that the refused inputs are rare (they are not -- every
# shipped tutorial but shockTube with a fixed deltaT is refused today; PORT.md lists the order of the work).
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
FIXED='s/^(adjustTimeStep\s+)\S+;/\1no;/'
row shockTube_shipped shockTube no "adjustTimeStep yes"
row wedge15Ma5_shipped wedge15Ma5 yes "is of type \`symmetryPlane\`"
row forwardStep_fixedDt forwardStep yes "is of type \`symmetryPlane\`" system/controlDict "$FIXED"
row obliqueShock_fixedDt obliqueShock yes "is of type \`symmetryPlane\`" system/controlDict "$FIXED"
row biconic_fixedDt biconic25-55Run35 no "ddtSchemes resolves to \`localEuler\`" system/controlDict "$FIXED"
row LadenburgJet60psi_fixedDt LadenburgJet60psi no "thermoType transport is \`sutherland\`" system/controlDict "$FIXED"
row movingCone_fixedDt movingCone no "dynamicMeshDict selects" system/controlDict "$FIXED"
row sod_fluxScheme shockTube no "fluxScheme \`Bogus\` is not a valid choice" system/controlDict "$FIXED" \
    system/fvSchemes 's/^(fluxScheme\s+)\S+;/\1Bogus;/'
row sod_reconstruct shockTube no "reconstruct(rho) to \`Minmod\`" system/controlDict "$FIXED" \
    system/fvSchemes 's/^(\s*reconstruct\(rho\)\s+)\S+;/\1Minmod;/'
row sod_grad shockTube no "grad(rho) to \`leastSquares\`" system/controlDict "$FIXED" \
    system/fvSchemes 's/^(\s*default\s+)Gauss linear;/\1leastSquares;/'
row sod_viscous shockTube no "transport mu is" system/controlDict "$FIXED" \
    constant/thermophysicalProperties 's/^(\s*mu\s+)\S+;/\11.8e-05;/'
row sod_RAS shockTube no "simulationType is \`RAS\`" system/controlDict "$FIXED" \
    constant/turbulenceProperties 's/^(simulationType\s+)\S+;/\1RAS;/'
row sod_slip shockTube yes "is \`slip\`" system/controlDict "$FIXED" 0/U '/sides/,/\}/ s/zeroGradient/slip/'
row sod_enthalpy shockTube no "energy is \`sensibleEnthalpy\`" system/controlDict "$FIXED" \
    constant/thermophysicalProperties 's/sensibleInternalEnergy/sensibleEnthalpy/'
echo "rho_central_refusals: $passes passed, $fails failed (work in $W)"
[ "$fails" = 0 ] && [ "$passes" -ge 14 ]
