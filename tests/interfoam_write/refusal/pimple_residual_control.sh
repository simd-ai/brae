#!/usr/bin/env bash
# The write gate: PIMPLE's `residualControl` is read as OpenFOAM reads it, and refused where OpenFOAM would
# leave the outer correctors early -- a convergence control neither interFoam loop ports.
. "$(dirname "$0")/../lib.sh"
# pimple.loop() asks criteriaSatisfied() from the second outer corrector on and, once every named residual is
# under its tolerance, runs one more corrector as the final one and leaves (pimpleControl.C:62-128, 219-240).
# The block was named nowhere in brae's interFoam until 2026-10-06: such a case ran every outer corrector.
# ORACLE: real interFoam on laminar/damBreakPermeable's pinned row restaged with five outer correctors and a
# residualControl for p_rgh: it has to say `PIMPLE: converged in` and run fewer than five -- the block changes
# the run, so ignoring it is a substitution. And on a block that gives a number where PIMPLE takes a
# dictionary it has to stop, at ONE outer corrector too (solutionControl.C:86-91).
# MEASURED 2026-10-06: OpenFOAM runs 2 outer correctors a step where the case allows 5 (`PIMPLE: converged in
# 2 iterations`, each step), and stops with `Residual data for p_rgh must be specified as a dictionary`.
# CONTROL: the well-formed block at one outer corrector, where OpenFOAM reads it and it has no effect
# (pimpleControl.C:65): brae runs it on both loops and agrees with OpenFOAM's run of the same case -- the
# refusal is of the early exit and of nothing else in the block. Worst file 1.6e-13 on the host loop and
# 7.8e-14 on the device loop (phi), so the bound is 2e-12.
wcase damBreakPermeable of > "$W/rc_stage.txt" 2>&1
o="$W/w_of_damBreakPermeable"
[ -d "$o" ] || { say "damBreakPermeable did not stage" FAIL; finish "PIMPLE residualControl"; }
restage()   # restage <dir> <nOuterCorrectors> <residualControl body>
{
    mkdir -p "$1"
    cp -r "$o/0" "$o/constant" "$o/system" "$1/"
    python3 - "$1/system/fvSolution" "$2" "$3" <<'PY'
import re, sys
p, n, body = sys.argv[1:4]
t = open(p).read()
t, c = re.subn(r'nOuterCorrectors\s+\d+;',
               'nOuterCorrectors    %s;\n    residualControl\n    {\n        %s\n    }' % (n, body), t)
assert c == 1, c
open(p, 'w').write(t)
PY
}
stops()     # stops <dir> <of|host|device>: run a case that is expected to stop, keep its exit code
{
    case $2 in
        of)     ( cd "$1" && interFoam > log.run 2>&1; echo $? > exit.txt ) ;;
        host)   ( cd "$1" && "$BIN" -case . > log.run 2>&1; echo $? > exit.txt ) ;;
        device) ( cd "$1" && "$BIN" -case . -device > log.run 2>&1; echo $? > exit.txt ) ;;
    esac
}
DICT='p_rgh { tolerance 1; relTol 0; }'
restage "$W/rc_of_early" 5 "$DICT"
( cd "$W/rc_of_early" && interFoam > log.interFoam 2>&1 ) || say "real interFoam did not run the fixture" FAIL
restage "$W/rc_of_value" 1 'p_rgh 1e-3;'
stops "$W/rc_of_value" of
steps=$(grep -ac "^Time = " "$W/rc_of_early/log.interFoam")
outer=$(grep -ac "^PIMPLE: iteration " "$W/rc_of_early/log.interFoam")
conv=$(grep -ac "^PIMPLE: converged in " "$W/rc_of_early/log.interFoam")
what="ORACLE  real interFoam leaves the outer loop early ($outer outer correctors in $steps steps of 5, "
what="$what'converged' $conv times) and stops on a value entry at one outer corrector"
[ "$steps" -gt 0 ] && [ "$conv" = "$steps" ] && [ "$outer" -lt $((5 * steps)) ] \
    && [ "$(cat "$W/rc_of_value/exit.txt")" != 0 ] \
    && grep -aq "Residual data for p_rgh must be specified as a dictionary" "$W/rc_of_value/log.run" \
    && say "$what" ok || say "$what" FAIL
ok=1
for v in host device; do
    restage "$W/rc_early_$v" 5 "$DICT"
    stops "$W/rc_early_$v" $v
    restage "$W/rc_value_$v" 1 'p_rgh 1e-3;'
    stops "$W/rc_value_$v" $v
    [ "$(cat "$W/rc_early_$v/exit.txt")" != 0 ] && [ -z "$(timedirs "$W/rc_early_$v")" ] \
        && grep -aq "names \`residualControl\` (for \`p_rgh\`) with nOuterCorrectors 5" "$W/rc_early_$v/log.run" \
        && [ "$(cat "$W/rc_value_$v/exit.txt")" != 0 ] && [ -z "$(timedirs "$W/rc_value_$v")" ] \
        && grep -aq "residualControl gives \`p_rgh\` as a value" "$W/rc_value_$v/log.run" || ok=0
done
what="[host, device] brae refuses both by name, before its first step, having written nothing"
[ $ok = 1 ] && say "$what" ok || say "$what" FAIL
restage "$W/rc_of_one" 1 "$DICT"
( cd "$W/rc_of_one" && interFoam > log.interFoam 2>&1 ) || say "real interFoam did not run the control" FAIL
B=2e-12
ok=1
for v in host device; do
    restage "$W/rc_one_$v" 1 "$DICT"
    runbrae "$W/rc_one_$v" $v
    python3 "$CMP" "$W/rc_of_one" "$W/rc_one_$v" $(timedirs "$W/rc_of_one") > "$W/cmp_rc_$v.txt" 2>&1
    judge "residualControl at one outer corrector, $v" "$W/cmp_rc_$v.txt" "$B" "$W/rc_of_one/log.interFoam" || ok=0
done
what="CONTROL  the same block at one outer corrector runs on both loops, every file within $B of OpenFOAM"
[ $ok = 1 ] && [ -n "$(timedirs "$W/rc_of_one")" ] && say "$what" ok || say "$what" FAIL
finish "PIMPLE residualControl is refused where OpenFOAM would leave the outer correctors early"
