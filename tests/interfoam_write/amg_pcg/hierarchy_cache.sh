#!/usr/bin/env bash
# The AMG hierarchies' disk cache -- the plain one p_rgh takes and pcorr's smoothed-aggregation one, a file each
# in constant/polyMesh -- on laminar/sloshingTank2D, a mesh that moves and keeps its topology (so pcorr takes the
# smoothed hierarchy) and whose run reproduces byte for byte.
# pcorr's smoothed hierarchy was built on the host at every start: MEASURED 2026-10-04, 2.7 s at 896,000 cells
# (2-D) and 15.9 s on the 845,536-cell hull, against 0.27 and 0.9 s read from the cache. The file is large: a
# smoothed hierarchy's RAP recipe is ~1 kB a cell in 2-D and ~5 kB in 3-D (953 MB and 4.3 GB on those two).
# The cache is KEYED on what a hierarchy is a function of (amgHierarchySignature: the internal faces' owner,
# neighbour and |Sf| by content, the build's parameters and version); before, a file was read on its sizes alone.
# Three arms on brae's DEFAULT pressure path (under BRAE_PRESSURE_CASE_SOLVER, what lib.sh exports, no hierarchy
# exists): a cold run that writes the files and a warm one that reads them with
# BRAE_CONTROL_AMG_CACHE_CHECK=1 (each hierarchy read is built as well and compared buffer by buffer); the same
# files carried to ANOTHER MESH of the same counts (the points scaled 5% in y), which must not be read; and the
# CONTROL, that mesh with the key not compared (BRAE_CONTROL_AMG_CACHE_STALE=1), which the check must stop.
. "$(dirname "$0")/../lib.sh"
unset BRAE_PRESSURE_CASE_SOLVER
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase sloshingTank2D of > "$W/hc_stage.txt" 2>&1
o="$W/w_of_sloshingTank2D"
[ -d "$o" ] || { say "sloshingTank2D did not stage" FAIL; finish "AMG hierarchy cache"; }
# arm <name> <fromDir> [env...]: brae on a copy of <fromDir>'s case, cache files and all; never ends the gate
arm()
{
    local e="$W/hc_$1" from="$2"
    shift 2
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$from/0" "$from/constant" "$from/system" "$e/"
    [ -n "${SCALE:-}" ] && ( cd "$e" && transformPoints -scale "$SCALE" > log.transformPoints 2>&1 )
    ( cd "$e" && env "$@" "$BIN" -case . -device > log.brae 2>&1; echo $? > exit.txt )
}
rm -f "$o/constant/polyMesh/.brae_amgcache" "$o/constant/polyMesh/.brae_amgcache_sa"
arm cold "$o" BRAE_X=1
arm warm "$W/hc_cold" BRAE_CONTROL_AMG_CACHE_CHECK=1
SCALE="(1 1.05 1)" arm other "$W/hc_cold" BRAE_X=1
SCALE="(1 1.05 1)" arm stale "$W/hc_cold" BRAE_CONTROL_AMG_CACHE_STALE=1 BRAE_CONTROL_AMG_CACHE_CHECK=1
read="one is read from"
n=0
t=0
for d in $(timedirs "$o"); do
    for f in $(cd "$W/hc_cold/$d" && find . -type f | sort); do
        t=$((t + 1))
        cmp -s "$W/hc_cold/$d/$f" "$W/hc_warm/$d/$f" || n=$((n + 1))
    done
done
what="[device] the warm run reads both hierarchies, each the one a build gives: $t written files, $n differ"
[ -s "$W/hc_cold/constant/polyMesh/.brae_amgcache" ] && [ -s "$W/hc_cold/constant/polyMesh/.brae_amgcache_sa" ] \
    && [ "$(grep -ac "smoothed-aggregation $read" "$W/hc_cold/log.brae")" = 0 ] \
    && [ "$(grep -ac "smoothed-aggregation $read" "$W/hc_warm/log.brae")" = 1 ] \
    && [ "$(grep -ac "$read" "$W/hc_warm/log.brae")" = 2 ] \
    && [ "$(cat "$W/hc_warm/exit.txt")" = 0 ] && [ "$t" -gt 0 ] && [ "$n" = 0 ] \
    && say "$what" ok || say "$what" FAIL
other="another mesh's or another build's"
what="[device] carried to another mesh of the same counts, neither file is read: both are built and rewritten"
[ "$(cat "$W/hc_other/exit.txt")" = 0 ] && [ "$(grep -ac "$other" "$W/hc_other/log.brae")" = 2 ] \
    && [ "$(grep -ac "$read" "$W/hc_other/log.brae")" = 0 ] \
    && [ "$(timedirs "$W/hc_other")" = "$(timedirs "$o")" ] && say "$what" ok || say "$what" FAIL
what="CONTROL  with the key not compared the other mesh's file is read, and the check stops the run"
[ "$(cat "$W/hc_stale/exit.txt")" != 0 ] \
    && grep -q "AMG_CACHE_CHECK: of the hierarchy read from" "$W/hc_stale/log.brae" \
    && say "$what" ok || say "$what" FAIL
finish "an AMG hierarchy read from the case's cache is the one a build gives, and only this mesh's is read"
