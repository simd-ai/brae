#!/bin/bash
# The RAS model BY ITS KEYWORD on the simpleFoam drivers. RASModel::New reads getCompat<word>("model",
# {{"RASModel", -2006}}) (RASModel.C:140-144): `model`, and the older `RASModel` only where that is absent.
# brae's simpleFoam readers took the older name alone, so a case written `RAS { model SpalartAllmaras; }`
# -- the boxy5 UAV case, staged from OpenFOAM-13's momentumTransport -- was refused as an empty RASModel.
# interFoam's reader had the same fix earlier (tests/interfoam_write/core/turbulence_model_keyword.sh).
#
# validation/airFoil2D (SpalartAllmaras), 20 SIMPLE iterations:
#   PREMISE  real simpleFoam writes the same files, byte for byte, with `model` as with `RASModel`, and
#            with BOTH present it takes `model`: `model SpalartAllmaras; RASModel kEpsilon;` writes the
#            SpalartAllmaras files.
#   REPRO    each brae driver run twice on the `RASModel` case writes identical files, so byte identity
#            below can witness anything at all.
#   ARM      each brae driver (the default one, and BRAE_SIMPLEFOAM_V2=1) writes with `model` the files it
#            writes with `RASModel`, byte for byte -- and with both present writes those files too.
#   CONTROL  BRAE_CONTROL_TURB_MODEL_COMPAT_ONLY=1 (the older name alone again): the `model` case is
#            refused as an empty RASModel, on both drivers.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="$ROOT/validation/airFoil2D"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
[ -x "$BRAE" ]         || { echo "SKIP: no brae binary"; exit 77; }
[ -d "$SRC/constant" ] || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ]     || { echo "SKIP: real OpenFOAM not available"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v simpleFoam > /dev/null 2>&1 || { echo "SKIP: simpleFoam not on PATH"; exit 77; }

W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
fail=0
say()
{
    printf '  %-84s %s\n' "$1" "$2"
    [ "$2" = FAIL ] && fail=1 || true
}

stage()   # stage <dir> <RASModel|model|both>
{
    rm -rf "$1"
    mkdir -p "$1"
    cp -r "$SRC/0" "$SRC/constant" "$SRC/system" "$1/"
    rm -rf "$1/0/0.orig"
    sed -i 's/^endTime .*/endTime         20;/;s/^writeInterval .*/writeInterval   20;/' "$1/system/controlDict"
    foamDictionary "$1/system/controlDict" -entry functions -remove > /dev/null 2>&1
    local tp="$1/constant/turbulenceProperties"
    grep -qE '^ *RASModel +SpalartAllmaras;' "$tp" || return 1
    case $2 in
        RASModel)
            ;;
        model)
            sed -i -E 's/^( *)RASModel( +)SpalartAllmaras;/\1model   \2SpalartAllmaras;/' "$tp"
            grep -qE '^ *model +SpalartAllmaras;' "$tp" && ! grep -q 'RASModel' "$tp" || return 1
            ;;
        both)
            sed -i -E 's/^( *)RASModel( +)SpalartAllmaras;/\1model           SpalartAllmaras;\n\1RASModel        kEpsilon;/' "$tp"
            grep -qE '^ *model +SpalartAllmaras;' "$tp" && grep -qE '^ *RASModel +kEpsilon;' "$tp" || return 1
            ;;
    esac
}

same20()   # same20 <dirA> <dirB>: every file of time 20 identical; prints the differing count
{
    local n=0
    local f
    [ -d "$1/20" ] && [ -d "$2/20" ] || { echo missing; return; }
    for f in $(cd "$1/20" && find . -type f | sort)
    do
        cmp -s "$1/20/$f" "$2/20/$f" || n=$((n + 1))
    done
    echo $n
}

# ---- PREMISE, on OpenFOAM ----------------------------------------------------------------------------
for k in RASModel model both
do
    stage "$W/of_$k" $k || { say "staging the $k case" FAIL; echo FAIL; exit 1; }
    ( cd "$W/of_$k" && simpleFoam > log 2>&1 )
done
n=$(same20 "$W/of_RASModel" "$W/of_model")
what="PREMISE  OpenFOAM writes time 20 alike with \`model\` and \`RASModel\`: $n files differ"
[ "$n" = 0 ] && grep -q "Selecting RAS turbulence model SpalartAllmaras" "$W/of_model/log" && say "$what" ok || say "$what" FAIL
n=$(same20 "$W/of_RASModel" "$W/of_both")
what="PREMISE  with both keywords OpenFOAM takes \`model\` (SpalartAllmaras): $n files differ"
[ "$n" = 0 ] && grep -q "Selecting RAS turbulence model SpalartAllmaras" "$W/of_both/log" && say "$what" ok || say "$what" FAIL

# ---- REPRO, ARM and CONTROL on each brae driver ------------------------------------------------------
for drv in default v2
do
    envs=""
    [ $drv = v2 ] && envs="BRAE_SIMPLEFOAM_V2=1"
    for k in RASModel RASModel2 model both
    do
        stage "$W/${drv}_$k" "${k%2}"
        ( cd "$W/${drv}_$k" && env $envs "$BRAE" -case . > log 2>&1 )
    done
    n=$(same20 "$W/${drv}_RASModel" "$W/${drv}_RASModel2")
    [ "$n" = 0 ] && say "REPRO    [$drv] two runs of the \`RASModel\` case write time 20 alike" ok \
                 || say "REPRO    [$drv] two runs of the \`RASModel\` case write time 20 alike ($n differ)" FAIL
    n=$(same20 "$W/${drv}_RASModel" "$W/${drv}_model")
    what="ARM      [$drv] \`model\` writes the files \`RASModel\` writes: $n differ"
    [ "$n" = 0 ] && say "$what" ok || say "$what" FAIL
    n=$(same20 "$W/${drv}_RASModel" "$W/${drv}_both")
    what="ARM      [$drv] with both keywords brae takes \`model\`: $n differ"
    [ "$n" = 0 ] && say "$what" ok || say "$what" FAIL
    stage "$W/${drv}_control" model
    ( cd "$W/${drv}_control" && env $envs BRAE_CONTROL_TURB_MODEL_COMPAT_ONLY=1 "$BRAE" -case . > log 2>&1; echo $? > exit.txt )
    e=$(cat "$W/${drv}_control/exit.txt")
    what="CONTROL  [$drv] the older name alone: the \`model\` case is refused (exit $e) as an empty model"
    [ "$e" != 0 ] && grep -aqE "RASModel (is )?['\`]['\`]" "$W/${drv}_control/log" && say "$what" ok \
        || { tail -3 "$W/${drv}_control/log"; say "$what" FAIL; }
done

[ $fail = 0 ] && echo PASS || { echo FAIL; exit 1; }
