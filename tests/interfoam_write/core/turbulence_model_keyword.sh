#!/usr/bin/env bash
# The write gate: the RAS model BY ITS KEYWORD. RASModel::New reads getCompat<word>("model", {{"RASModel",
# -2006}}) (RASModel.C:140-144): `model`, and the older `RASModel` only where that is absent. brae read the
# older name alone, so a turbulenceProperties written `RAS { model kEpsilon; }` was refused as an empty model.
# Three checks on RAS/weirOverflow's pinned row with its `RASModel kEpsilon;` rewritten `model kEpsilon;`.
# (1) THE PREMISE, on OpenFOAM: real interFoam with the new keyword writes the files it writes with the older
# one, byte for byte. (2) Each loop within the row's own bound of that run. (3) CONTROL:
# BRAE_CONTROL_TURB_MODEL_COMPAT_ONLY=1 reads the older name alone again, and the run is refused.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase weirOverflow of > "$W/tk_stage.txt" 2>&1
o="$W/w_of_weirOverflow"
[ -d "$o" ] || { say "weirOverflow did not stage" FAIL; finish "the RAS model's keyword"; }
restage()   # restage <dir>: the staged row with `model` for `RASModel`
{
    rm -rf "${1:?}"
    mkdir -p "$1"
    cp -r "$o/0" "$o/constant" "$o/system" "$1/"
    sed -i -E 's/^( *)RASModel( +)kEpsilon;/\1model   \2kEpsilon;/' "$1/constant/turbulenceProperties"
    grep -qE '^ *model +kEpsilon;' "$1/constant/turbulenceProperties" \
        && ! grep -q "RASModel" "$1/constant/turbulenceProperties"
}
restage "$W/tk_of" || { say "the keyword was not rewritten" FAIL; finish "the RAS model's keyword"; }
runof "$W/tk_of"
ot=$(timedirs "$W/tk_of")
n=0
for t in $ot; do
    for f in $(cd "$o/$t" && find . -type f | sort); do
        cmp -s "$o/$t/$f" "$W/tk_of/$t/$f" || n=$((n + 1))
    done
done
what="PREMISE  OpenFOAM with \`model kEpsilon\` writes [$ot] as with \`RASModel kEpsilon\`: $n files differ"
[ "$(echo $ot | wc -w)" = 2 ] && [ "$ot" = "$(timedirs "$o")" ] && [ "$n" = 0 ] \
    && grep -q "Selecting RAS turbulence model kEpsilon" "$W/tk_of/log.interFoam" && say "$what" ok || say "$what" FAIL
for v in host device; do
    restage "$W/tk_$v"
    runbrae "$W/tk_$v" $v
    python3 "$CMP" "$W/tk_of" "$W/tk_$v" $ot > "$W/cmp_tk_$v.txt" 2>&1
done
BH=5e-12
BD=3e-11
judge "the model keyword, host" "$W/cmp_tk_host.txt" "$BH" "$W/tk_of/log.interFoam" \
    && judge "the model keyword, device" "$W/cmp_tk_device.txt" "$BD" "$W/tk_of/log.interFoam" \
    && say "[host, device] every written file within $BH and $BD of OpenFOAM" ok \
    || say "[host, device] every written file within $BH and $BD of OpenFOAM" FAIL
restage "$W/tk_control"
(
    cd "$W/tk_control" || exit 1
    env BRAE_CONTROL_TURB_MODEL_COMPAT_ONLY=1 "$BIN" -case . > log.brae 2>&1
    echo $? > exit.txt
)
e=$(cat "$W/tk_control/exit.txt")
what="CONTROL  the older name alone: the run is refused (exit $e) as an empty RASModel"
[ "$e" != 0 ] && grep -aq "asks for RASModel \`\`" "$W/tk_control/log.brae" && say "$what" ok || say "$what" FAIL
finish "the RAS model is read by \`model\`, and by \`RASModel\` only where that is absent"
