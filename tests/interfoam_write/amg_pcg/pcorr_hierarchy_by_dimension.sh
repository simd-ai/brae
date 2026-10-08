#!/usr/bin/env bash
# WHICH hierarchy pcorr's AMG-PCG takes on a mesh that moves and keeps its topology: a smoothed-aggregation one of
# its own on a two-dimensional mesh, the plain one p_rgh uses on a three-dimensional mesh. Until 2026-10-04 it was
# the smoothed one on both, a rule measured on the 2-D wave makers alone. MEASURED that day, pcorr's iterations,
# smoothed / plain: waveMakerPiston at 896,000 cells 1,427 / 5,530; RAS/DTCHullMoving at 845,536 cells 626 / 354,
# CorrectPhi 166.8 / 97.8 ms a step (and the smoothed hierarchy is 16 s to build there, or a 4.3 GB cache file).
# Here on DTCHullMovingCoarse (108,833 cells, three-dimensional) and waveMakerFlap (56,000, two-dimensional), at
# the write gate's pinned tolerances. The second check IS the measurement the rule stands on, re-made at every
# run: the smoothed hierarchy forced onto the 3-D mesh (BRAE_PCORR_AMG=sa) against the default -- 811 iterations
# for 216. Why a proxy of the face areas shapes a good prolongator on a structured 2-D mesh and not on a snappy
# 3-D one is not established; the rule is what was measured, on six meshes.
. "$(dirname "$0")/../lib.sh"
unset BRAE_PRESSURE_CASE_SOLVER
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
for k in DTCHullMovingCoarse waveMakerFlap; do
    wcase $k of > "$W/pd_stage_$k.txt" 2>&1
    [ -d "$W/w_of_$k" ] || { say "$k did not stage" FAIL; finish "pcorr hierarchy by dimension"; }
    rm -f "$W/w_of_$k/constant/polyMesh/.brae_amgcache" "$W/w_of_$k/constant/polyMesh/.brae_amgcache_sa"
done
# arm <case> <name> [env...]: brae on a copy of the staged case; never ends the gate
arm()
{
    local o="$W/w_of_$1" e="$W/pd_$1_$2"
    shift 2
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    ( cd "$e" && env BRAE_PRINT_TURB_SOLVES=1 "$@" "$BIN" -case . -device > log.brae 2>&1; echo $? > exit.txt )
}
# its <case> <arm>: pcorr's iteration counts, a solve each;  sum: their total
its() { grep -a "Solving for pcorr" "$W/pd_$1_$2/log.brae" | sed -E 's/.*No Iterations ([0-9]+).*/\1/' | tr '\n' ' '; }
sum() { its $1 $2 | tr ' ' '\n' | awk '{s += $1} END {print s + 0}'; }
mark="pcorr: its AMG hierarchy is a smoothed-aggregation one of its own"
h=DTCHullMovingCoarse
arm $h default BRAE_X=1
arm $h plain BRAE_PCORR_AMG=plain
arm $h sa BRAE_PCORR_AMG=sa
arm waveMakerFlap default BRAE_X=1
what="[device] three-dimensional moving mesh: pcorr takes the plain hierarchy, $(its $h default)(plain $(its $h plain))"
[ "$(cat "$W/pd_${h}_default/exit.txt")" = 0 ] && [ "$(cat "$W/pd_${h}_plain/exit.txt")" = 0 ] \
    && ! grep -aq "$mark" "$W/pd_${h}_default/log.brae" \
    && [ ! -e "$W/pd_${h}_default/constant/polyMesh/.brae_amgcache_sa" ] \
    && [ "$(sum $h default)" -gt 0 ] && [ "$(its $h default)" = "$(its $h plain)" ] \
    && say "$what" ok || say "$what" FAIL
what="[device] ...where the smoothed one, forced, takes $(sum $h sa) iterations for the default's $(sum $h default)"
[ "$(cat "$W/pd_${h}_sa/exit.txt")" = 0 ] && grep -aq "$mark" "$W/pd_${h}_sa/log.brae" \
    && [ "$(sum $h sa)" -ge $(( 2 * $(sum $h default) )) ] && say "$what" ok || say "$what" FAIL
what="[device] two-dimensional moving mesh: pcorr takes its smoothed hierarchy, says why, and caches it"
[ "$(cat "$W/pd_waveMakerFlap_default/exit.txt")" = 0 ] \
    && grep -aq "$mark, built once (the mesh keeps its topology and is two-dimensional)" \
        "$W/pd_waveMakerFlap_default/log.brae" \
    && [ -s "$W/pd_waveMakerFlap_default/constant/polyMesh/.brae_amgcache_sa" ] \
    && say "$what" ok || say "$what" FAIL
finish "pcorr's hierarchy is the smoothed one on a 2-D moving mesh and the plain one on a 3-D one"
