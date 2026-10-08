#!/usr/bin/env bash
# The write gate: the pressure hierarchy's SMALL-MESH RULE -- two coarsening passes a level on a fine grid of
# 10,000 cells or fewer (device_amg.cu, amgMergeFor; brae_interFoam asks for it). On a mesh that small a level
# costs its kernel launches, not its cells: damBreak's 2,268 cells had seven grids. MEASURED 2026-10-06 over the
# 42 tutorials with two passes everywhere: every mesh of 10,000 cells or fewer 3-15% faster a step, its step
# count unchanged; flat from 400,000 cells; and four wave cases between 14,000 and 160,000 cells took one more
# time step in thirty, a step further from serial OpenFOAM's count -- so the rule stops at 10,000.
# HELD HERE: the rule's scope and the run it leaves. damBreak as shipped (adjustTimeStep, its own tolerances)
# to t = 0.2 takes serial OpenFOAM's time steps on the merged hierarchy, which has half the levels; stokesI
# (37,500 cells) is left on one pass. The ANSWER on the merged hierarchy is the other amg_pcg gates', which
# run it on every row of 10,000 cells or fewer against OpenFOAM with their own controls.
# The CONTROL takes one pass on damBreak (BRAE_AMG_MERGE=1): the notice has to go and the levels to double,
# or the first check's path proof proves nothing.
. "$(dirname "$0")/../lib.sh"
unset BRAE_PRESSURE_CASE_SOLVER
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
arm()   # arm <name> <tutorial> <endTime>: the tutorial as shipped, no write, to <endTime>
{
    local d="$W/sm_$1" src="$TUT/multiphase/interFoam/$2"
    [ -d "$W/sm_stage_$(basename "$2")" ] || stage_allrun "$src" "$W/sm_stage_$(basename "$2")" "" \
        > "$W/sm_stage_$1.txt" 2>&1 || { say "$2 did not stage" FAIL; finish "the hierarchy's small-mesh rule"; }
    rm -rf "${d:?}"
    mkdir -p "$d"
    cp -r "$W/sm_stage_$(basename "$2")/0" "$W/sm_stage_$(basename "$2")/constant" \
        "$W/sm_stage_$(basename "$2")/system" "$d/"
    cp "$src/system/fvSolution" "$d/system/fvSolution"
    python3 - "$src/system/controlDict" "$d/system/controlDict" "$3" <<'PY'
import re, sys
s = open(sys.argv[1]).read()
s = re.sub(r'\nfunctions\s*\{.*?\n\}', '\nfunctions\n{\n}', s, flags=re.S)
for k, v in [('writeControl', 'timeStep'), ('writeInterval', '1000000'), ('startFrom', 'startTime'),
             ('stopAt', 'endTime'), ('endTime', sys.argv[3])]:
    s = re.sub(r'^%s\s.*' % k, '%s %s;' % (k.ljust(15), v), s, flags=re.M)
open(sys.argv[2], 'w').write(s)
PY
}
levels()   # the plain hierarchy's coarse levels, as its build prints them
{
    grep -a "^\[AMG\] hierarchy:" "$1/log.brae" | head -1 | sed -E 's/.*hierarchy: ([0-9]+) levels.*/\1/'
}
arm of laminar/damBreak/damBreak 0.2
runof "$W/sm_of"
arm rule laminar/damBreak/damBreak 0.2
runbrae "$W/sm_rule" device BRAE_AMG_DEBUG=1 BRAE_AMG_CACHE=0
arm one laminar/damBreak/damBreak 0.2
runbrae "$W/sm_one" device BRAE_AMG_DEBUG=1 BRAE_AMG_CACHE=0 BRAE_AMG_MERGE=1
arm big laminar/waves/stokesI 0.05
runbrae "$W/sm_big" device BRAE_AMG_DEBUG=1 BRAE_AMG_CACHE=0
arm bigone laminar/waves/stokesI 0.05
runbrae "$W/sm_bigone" device BRAE_AMG_DEBUG=1 BRAE_AMG_CACHE=0 BRAE_AMG_MERGE=1
notice="a small mesh (2268 cells) takes two coarsening passes a level"
no=$(grep -c "^Time = " "$W/sm_of/log.interFoam")
nb=$(grep -ac "^ *t = .* dt = " "$W/sm_rule/log.brae")
lr=$(levels "$W/sm_rule")
lo=$(levels "$W/sm_one")
what="[device] damBreak as shipped to t = 0.2 on $lr coarse levels: $nb time steps, serial OpenFOAM $no (within one)"
[ "$no" -ge 50 ] && [ "$nb" -le $((no + 1)) ] && [ "$nb" -ge $((no - 1)) ] && grep -q "$notice" "$W/sm_rule/log.brae" \
    && [ -n "$lr" ] && say "$what" ok || say "$what" FAIL
lb=$(levels "$W/sm_big")
lbo=$(levels "$W/sm_bigone")
what="[device] stokesI's 37,500 cells are left on one pass a level: $lb coarse levels, $lbo with BRAE_AMG_MERGE=1"
[ -n "$lb" ] && [ "$lb" = "$lbo" ] && ! grep -q "takes two coarsening passes" "$W/sm_big/log.brae" \
    && say "$what" ok || say "$what" FAIL
what="CONTROL  one pass on damBreak: $lo coarse levels for the rule's $lr, and no notice"
[ -n "$lo" ] && [ "$lo" -ge $((2*lr - 1)) ] && ! grep -q "takes two coarsening passes" "$W/sm_one/log.brae" \
    && say "$what" ok || say "$what" FAIL
finish "a mesh of 10,000 cells or fewer takes two coarsening passes a level and OpenFOAM's time steps"
