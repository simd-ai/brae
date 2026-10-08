#!/usr/bin/env bash
# The write gate: `alphaApplyPrevCorr yes` WITHOUT `MULESCorr`, which is a switch that does nothing --
# alphaEqn.H applies the previous correction inside `if (MULESCorr)` (:99, :133) and stores it under
# `alphaApplyPrevCorr && MULESCorr` (:228); otherwise the cache is cleared every step. The GPU loop stored it
# under the switch alone: it subtracted from the empty cache and the first step ended on an illegal memory
# access (MEASURED 2026-10-07, laminar/capillaryRise with the switch added) where the host loop ran.
# Three checks on laminar/capillaryRise's pinned row with the switch added (the tutorial has no MULESCorr).
# (1) THE PREMISE, on OpenFOAM: real interFoam with the switch writes the files it writes without it, byte for
# byte. (2) Both loops are within the row's own bound of that run. (3) CONTROL:
# BRAE_CONTROL_PREVCORR_WITHOUT_MULES=1 stores under the switch alone again, and the GPU loop does not end.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase capillaryRise of > "$W/pc_stage.txt" 2>&1
o="$W/w_of_capillaryRise"
[ -d "$o" ] || { say "capillaryRise did not stage" FAIL; finish "alphaApplyPrevCorr without MULESCorr"; }
restage()   # restage <dir>: the staged row with `alphaApplyPrevCorr yes` beside nAlphaCorr
{
    mkdir -p "$1"
    cp -r "$o/0" "$o/constant" "$o/system" "$1/"
    python3 - "$1/system/fvSolution" <<'PY'
import re, sys
p = sys.argv[1]
t = open(p).read()
assert 'MULESCorr' not in t and 'alphaApplyPrevCorr' not in t
t, n = re.subn(r'(\n\s*)(nAlphaCorr\s)', r'\1alphaApplyPrevCorr yes;\1\2', t, count=1)
assert n == 1
open(p, 'w').write(t)
PY
}
restage "$W/pc_of"
runof "$W/pc_of"
ot=$(timedirs "$W/pc_of")
n=0
for t in $ot; do
    for f in $(cd "$o/$t" && find . -type f | sort); do
        cmp -s "$o/$t/$f" "$W/pc_of/$t/$f" || n=$((n + 1))
    done
done
what="PREMISE  OpenFOAM with the switch writes [$ot] as without it: $n files differ"
[ "$(echo $ot | wc -w)" = 2 ] && [ "$ot" = "$(timedirs "$o")" ] && [ "$n" = 0 ] && say "$what" ok || say "$what" FAIL
for v in host device; do
    restage "$W/pc_$v"
    runbrae "$W/pc_$v" $v
    python3 "$CMP" "$W/pc_of" "$W/pc_$v" $ot > "$W/cmp_pc_$v.txt" 2>&1
done
B=4e-13
judge "the switch without MULESCorr, host" "$W/cmp_pc_host.txt" "$B" "$W/pc_of/log.interFoam" \
    && judge "the switch without MULESCorr, device" "$W/cmp_pc_device.txt" "$B" "$W/pc_of/log.interFoam" \
    && say "[host, device] every written file within $B of OpenFOAM" ok \
    || say "[host, device] every written file within $B of OpenFOAM" FAIL
restage "$W/pc_control"
(
    cd "$W/pc_control" || exit 1
    env BRAE_CONTROL_PREVCORR_WITHOUT_MULES=1 "$BIN" -case . -device > log.brae 2>&1
    echo $? > exit.txt
)
e=$(cat "$W/pc_control/exit.txt")
what="CONTROL  stored under the switch alone: the GPU loop does not end (exit $e,"
what="$what $(grep -ac '^ *t = ' "$W/pc_control/log.brae") steps)"
[ "$e" != 0 ] && ! grep -aq "^End: t" "$W/pc_control/log.brae" && say "$what" ok || say "$what" FAIL
finish "alphaApplyPrevCorr without MULESCorr does nothing on either loop, as in OpenFOAM"
