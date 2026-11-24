#!/usr/bin/env bash
# The captured BiCGStab loop across a coupled pair whose buffers move. deviceJacobiBiCGStabGraph captures the
# whole iteration, the pair's products with it, and a capture bakes every pointer it saw: a moving cyclicAMI's
# stencil is rebuilt into fresh buffers at every move, and the graph's key held the matrix and the topology
# but not the pair -- a replay read the previous step's buffers, freed. MEASURED 2026-10-07 on RAS/
# mixerVesselAMI with `PBiCGStab` on U (no shipped tutorial pairs the two): an illegal memory access in the
# fourth step. The key now holds the pair's ten pointers and three counts.
# Two checks on RAS/mixerVesselAMI's staged mesh, four steps on the GPU loop with U on PBiCGStab and the
# tutorial's own tolerances (the staging pins every one at 1e-13 for an OpenFOAM comparison this gate does
# not make; the pinned six steps took seven minutes). (1) THE
# CAPTURED LOOP IS THE PLAIN LOOP'S RUN: against BRAE_BICG_HOST_LOOP=1 -- the loop the capture was taken of,
# which rebuilds nothing and holds no pointer -- every written file the same bytes, and both name PBiCGStab on
# U. (2) CONTROL: BRAE_CONTROL_BICG_GRAPH_PAIR_UNKEYED=1 leaves the pair out of the key again, and the defect
# shows: the run does not end, OR what it wrote differs from the plain loop's -- the comparison (1) makes. Which of
# the two depends on what the stale addresses hold when a replay reads them, and that follows the device's
# allocation sequence, not the arithmetic. MEASURED 2026-11-21 on main 26d0a176, with the pool on and with
# BRAE_NO_DEVICE_POOL=1 alike:
#   - GB10 (sm_121): an illegal memory access, after 3 steps (2 without the pool);
#   - GH200 (sm_90): the run reaches its end on stale pair data, and 13 of the 17 written files differ from the
#     plain loop's. The graph and plain arms are byte-identical on both.
# A run the driver stops as diverged (exit 1, nothing written after) does not end either.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
s="$W/bg_stage"
rm -rf "${s:?}"
stage_allrun "$TUT/multiphase/interFoam/RAS/mixerVesselAMI" "$s" "" > "$W/bg_stage.txt" 2>&1 \
    || { say "mixerVesselAMI did not stage" FAIL; finish "the BiCGStab graph across a moving pair"; }
cp "$TUT/multiphase/interFoam/RAS/mixerVesselAMI/system/fvSolution" "$s/system/fvSolution"
python3 - "$s" <<'PY' || { say "the PBiCGStab staging did not apply" FAIL; finish "the BiCGStab graph, a moving pair"; }
import re, sys
d = sys.argv[1]
p = d + '/system/fvSolution'
t = open(p).read()
a = '"(U|T|k|epsilon).*"'
assert a in t
u = '"U.*"\n    {\n        solver          PBiCGStab;\n        preconditioner  DILU;\n        tolerance       1e-06;\n'
u += '        relTol          0;\n    }\n\n    "(T|k|epsilon).*"'
open(p, 'w').write(t.replace(a, u, 1))
c = d + '/system/controlDict'
t = open(c).read()
dt = float(re.search(r'^deltaT\s+([^;]+);', t, re.M).group(1))
t = re.sub(r'^endTime\s.*', 'endTime         %.12g;' % (4*dt), t, flags=re.M)
t = re.sub(r'^writeInterval\s.*', 'writeInterval   4;', t, flags=re.M)
open(c, 'w').write(t)
PY
# arm <name> [env...]: brae's GPU loop on a copy of the staged case; never ends the gate
arm()
{
    local e="$W/bg_$1"
    shift
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$s/0" "$s/constant" "$s/system" "$e/"
    ( cd "$e" && env BRAE_PRINT_TURB_SOLVES=1 "$@" "$BIN" -case . -device > log.brae 2>&1; echo $? > exit.txt )
}
arm graph BRAE_X=1
arm plain BRAE_BICG_HOST_LOOP=1
arm unkeyed BRAE_CONTROL_BICG_GRAPH_PAIR_UNKEYED=1
tg=$(timedirs "$W/bg_graph")
n=0
k=0
for t in $tg; do
    for f in $(cd "$W/bg_graph/$t" && find . -type f | sort); do
        k=$((k + 1))
        cmp -s "$W/bg_graph/$t/$f" "$W/bg_plain/$t/$f" || n=$((n + 1))
    done
done
steps=$(grep -ac '^ *t = ' "$W/bg_graph/log.brae")
what="[device] the captured loop across a moving pair: $steps steps, $n of $k written files differ from the"
what="$what plain loop's"
[ "$(cat "$W/bg_graph/exit.txt")" = 0 ] && [ "$(cat "$W/bg_plain/exit.txt")" = 0 ] && [ "$steps" = 4 ] \
    && [ -n "$tg" ] && [ "$tg" = "$(timedirs "$W/bg_plain")" ] && [ "$k" -ge 8 ] && [ "$n" = 0 ] \
    && grep -aq "BiCGStab: host loop" "$W/bg_plain/log.brae" \
    && ! grep -aq "BiCGStab: host loop" "$W/bg_graph/log.brae" \
    && grep -aq "PBiCGStab" "$W/bg_graph/log.brae" && say "$what" ok || say "$what" FAIL
e=$(cat "$W/bg_unkeyed/exit.txt")
su=$(grep -ac '^ *t = ' "$W/bg_unkeyed/log.brae")
if [ "$e" != 0 ] && ! grep -aq "^End: t" "$W/bg_unkeyed/log.brae"; then
    say "CONTROL  the pair left out of the graph's key: the run does not end (exit $e, $su steps)" ok
else
    # it ended: then what it wrote must differ from the plain loop's, file by file (a file missing counts)
    nu=0
    ku=0
    for t in $(timedirs "$W/bg_plain"); do
        for f in $(cd "$W/bg_plain/$t" && find . -type f | sort); do
            ku=$((ku + 1))
            cmp -s "$W/bg_unkeyed/$t/$f" "$W/bg_plain/$t/$f" || nu=$((nu + 1))
        done
    done
    what="CONTROL  the pair left out of the graph's key: the run ends (exit $e, $su steps), and $nu of $ku"
    what="$what written files differ from the plain loop's"
    [ "$ku" -ge 8 ] && [ "$nu" -gt 0 ] && say "$what" ok || say "$what" FAIL
fi
finish "the BiCGStab graph is recaptured when a coupled pair's buffers move"
