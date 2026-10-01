#!/usr/bin/env bash
# One arm of the write gate (tests/interfoam_write/lib.sh has the gate's header and helpers).
. "$(dirname "$0")/../lib.sh"

# ---------------------------------------------------------------------------------------------------
# S: a SUB-CYCLED alpha (laminar/mixerVessel2D, nAlphaSubCycles 2, MRF) writes alpha.water_0 -- its old
# time, which GeometricField::storeOldTime makes AUTO_WRITE once the sub-cycle has given it an old-old
# level. OpenFOAM writes it at EVERY write time, the first included, and it holds alpha as the step found
# it (bit-identical to the previous step's alpha.water, measured). Three fixed steps of 1e-3.
if [ -d "$MV" ]; then
    stage "$MV" "$W/of_s" 0.003 timeStep 1 0 adjustTimeStep=no || exit 1
    runof "$W/of_s"
    [ "$(timedirs "$W/of_s")" = "0.001 0.002 0.003 " ] && [ -f "$W/of_s/0.001/alpha.water_0" ] \
        && say "premise: OpenFOAM writes alpha.water_0 at every step of a sub-cycled alpha, the first included" ok \
        || say "premise: OpenFOAM writes alpha.water_0 at every step of a sub-cycled alpha, the first included" FAIL
    for arm in $ARMS; do
        stage "$MV" "$W/br_s_$arm" 0.003 timeStep 1 0 adjustTimeStep=no || exit 1
        runbrae "$W/br_s_$arm" "$arm"
        ok=1
        for t in 0.001 0.002 0.003; do
            [ "$(filesets "$W/of_s" $t)" = "$(filesets "$W/br_s_$arm" $t)" ] || ok=0
        done
        [ $ok -eq 1 ] && say "ARM S  [$arm] OpenFOAM's file set at every step, alpha.water_0 included" ok \
                      || say "ARM S  [$arm] OpenFOAM's file set at every step, alpha.water_0 included" FAIL
        python3 "$CMP" "$W/of_s" "$W/br_s_$arm" 0.001 0.002 0.003 > "$W/cmp_s_$arm.txt" 2>&1
        judge "mixerVessel2D $arm" "$W/cmp_s_$arm.txt" "$BOUND_FIELDS" "$W/of_s/log.interFoam" \
            && say "ARM S  [$arm] every file's structure is OpenFOAM's, every value within $BOUND_FIELDS" ok \
            || { say "ARM S  [$arm] every file's structure is OpenFOAM's, every value within $BOUND_FIELDS" FAIL; grep -v RESULT "$W/cmp_s_$arm.txt" | grep -B1 "^      " | head -20; }
        # brae's own alpha_0 IS its previous alpha -- and, the control, not its current one
        python3 - "$W/br_s_$arm" <<'PY' && say "ARM S  [$arm] brae's alpha.water_0 at 0.002 is its 0.001 alpha bit for bit, not its 0.002 one" ok \
                                       || say "ARM S  [$arm] brae's alpha.water_0 at 0.002 is its 0.001 alpha bit for bit, not its 0.002 one" FAIL
import re, sys
d = sys.argv[1]
def cells(p):
    t = open(p).read()
    return t[t.find('internalField'):t.find('boundaryField')].replace('alpha.water_0', '')
old, prev, cur = cells(d + '/0.002/alpha.water_0'), cells(d + '/0.001/alpha.water'), cells(d + '/0.002/alpha.water')
print('      alpha_0(0.002) == alpha(0.001): %s; == alpha(0.002): %s' % (old == prev, old == cur))
sys.exit(0 if old == prev and old != cur else 1)
PY
    done
else
    say "ARM S  mixerVessel2D tutorial missing" FAIL
fi

# ---------------------------------------------------------------------------------------------------

finish "arm S: a sub-cycled alpha writes its old level"
