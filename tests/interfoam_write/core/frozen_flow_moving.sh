#!/usr/bin/env bash
# The write gate: PIMPLE's `frozenFlow yes` on a mesh that moves. interFoam.C:156-159 `continue`s past the
# momentum and the pressure equations, so alpha is carried on the flux the mesh update left and Uf is not
# touched (pEqn.H:66-72 is never reached). The GPU loop still ran its end-of-step fvc::correctUf, on the
# absolute flux the pressure corrector never formed: empty buffers, a segmentation fault in the first step
# (MEASURED 2026-10-07, laminar/sloshingTank2D with the switch added) where the host loop ran.
# Three checks on laminar/sloshingTank2D's pinned row with the switch added. ORACLE: real interFoam on the same
# staged case. (1) THE PREMISE: OpenFOAM solves no p_rgh there and its alpha is not the row's own -- the
# switch is taken and the mesh carries the interface. (2) Both loops within the row's bound of OpenFOAM.
# (3) CONTROLS: BRAE_CONTROL_FROZEN_FLOW_CORRECT_UF=1 runs the block under the switch again, and the GPU loop
# does not end; BRAE_CONTROL_FROZEN_FLOW_P_REBUILT=1 rebuilds p's patch values at the write, and p goes over.
# WHAT (2) FOUND: p. pEqn.H's `p == p_rgh + rho*gh` is all that assigns p after createFields.H, so under the
# switch OpenFOAM writes it as the start left it; both loops rebuilt its patch values at each write from the
# live density and gh. MEASURED 2026-10-07: p 7.0e-03 after one step and 1.4e-02 after two, every other
# file at round-off (U 1.9e-14).
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase sloshingTank2D of > "$W/ff_stage.txt" 2>&1
o="$W/w_of_sloshingTank2D"
[ -d "$o" ] || { say "sloshingTank2D did not stage" FAIL; finish "frozenFlow on a moving mesh"; }
restage()   # restage <dir>: the staged row with `frozenFlow yes` in PIMPLE
{
    mkdir -p "$1"
    cp -r "$o/0" "$o/constant" "$o/system" "$1/"
    python3 - "$1/system/fvSolution" <<'PY'
import re, sys
p = sys.argv[1]
t = open(p).read()
t, n = re.subn(r'(\nPIMPLE\s*\{)', r'\1\n    frozenFlow      yes;', t, count=1)
assert n == 1
open(p, 'w').write(t)
PY
}
restage "$W/ff_of"
runof "$W/ff_of"
ot=$(timedirs "$W/ff_of")
last=$(echo $ot | awk '{print $NF}')
np=$(grep -c "Solving for p_rgh" "$W/ff_of/log.interFoam")
python3 "$CMP" "$o" "$W/ff_of" $ot > "$W/cmp_ff_premise.txt" 2>&1
moved=$(python3 - "$W/cmp_ff_premise.txt" "$last" <<'PY'
import json, sys
try:
    r = json.loads([l for l in open(sys.argv[1]) if l.startswith('RESULT ')][-1][7:])
    print('%.1e' % r['files'][sys.argv[2] + '/alpha.water']['rel'])
except Exception:
    print('-')
PY
)
what="PREMISE  OpenFOAM under frozenFlow solves p_rgh $np times in [$ot] and its alpha at $last is $moved from"
what="$what the row's own"
[ "$(echo $ot | wc -w)" = 2 ] && [ "$np" = 0 ] && above "$moved" 1e-9 && say "$what" ok || say "$what" FAIL
for v in host device; do
    restage "$W/ff_$v"
    runbrae "$W/ff_$v" $v
    python3 "$CMP" "$W/ff_of" "$W/ff_$v" $ot > "$W/cmp_ff_$v.txt" 2>&1
done
B=1e-11
judge "frozenFlow on a moving mesh, host" "$W/cmp_ff_host.txt" "$B" "$W/ff_of/log.interFoam" \
    && judge "frozenFlow on a moving mesh, device" "$W/cmp_ff_device.txt" "$B" "$W/ff_of/log.interFoam" \
    && say "[host, device] every written file within $B of OpenFOAM" ok \
    || say "[host, device] every written file within $B of OpenFOAM" FAIL
restage "$W/ff_control"
(
    cd "$W/ff_control" || exit 1
    env BRAE_CONTROL_FROZEN_FLOW_CORRECT_UF=1 "$BIN" -case . -device > log.brae 2>&1
    echo $? > exit.txt
)
e=$(cat "$W/ff_control/exit.txt")
restage "$W/ff_rebuilt"
runbrae "$W/ff_rebuilt" device BRAE_CONTROL_FROZEN_FLOW_P_REBUILT=1
python3 "$CMP" "$W/ff_of" "$W/ff_rebuilt" $ot > "$W/cmp_ff_rebuilt.txt" 2>&1
judge "control p rebuilt" "$W/cmp_ff_rebuilt.txt" "$B" "$W/ff_of/log.interFoam" > "$W/ff_rebuilt.txt"; rb=$?
what="CONTROLS correctUf under frozenFlow: the GPU loop does not end (exit $e,"
what="$what $(grep -ac '^ *t = ' "$W/ff_control/log.brae") steps); p's patches rebuilt at the write:"
what="$what $(grep -c 'over the bound: [0-9.e+-]*/p ' "$W/ff_rebuilt.txt") writes of p over the bound"
[ "$e" != 0 ] && ! grep -aq "^End: t" "$W/ff_control/log.brae" && [ $rb != 0 ] \
    && [ "$(grep -c 'over the bound: [0-9.e+-]*/p ' "$W/ff_rebuilt.txt")" = 2 ] && say "$what" ok || say "$what" FAIL
finish "frozenFlow on a moving mesh leaves Uf alone on the GPU loop, as in OpenFOAM"
