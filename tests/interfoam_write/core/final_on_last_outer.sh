#!/usr/bin/env bash
# The write gate: PIMPLE's `finalOnLastPimpleIterOnly` -- which pressure solves take the Final entry -- against
# real OpenFOAM, on laminar/damBreakPermeable restaged with three outer correctors.
. "$(dirname "$0")/../lib.sh"
# pEqn.H:50 solves with p_rgh.select(pimple.finalInnerIter()). That is the last corrector of EVERY outer
# corrector by default and of the LAST outer corrector alone when the switch is on (pimpleControlI.H:98-112).
# Neither interFoam loop read it until 2026-10-06: a case that set it took p_rghFinal three times a step where
# OpenFOAM takes it once. THE FIXTURE makes the two entries tell themselves apart -- p_rgh stops at 1e-4,
# p_rghFinal at 1e-13 -- so each solve's final residual says which entry it took, in OpenFOAM's log and brae's.
# ORACLE: real interFoam on the same staged case: its sequence of entries, solve for solve, and its fields.
# MEASURED 2026-10-06: OpenFOAM's sequence is ppppppppF a step (and its iteration counts 20 0 0 2 0 0 2 0 77,
# which both loops repeat); the worst written file 2.6e-13 on the host loop and 2.1e-13 on the device loop
# (phi), so the bound is 3e-12.
# CONTROL: BRAE_CONTROL_FINAL_ON_EVERY_OUTER=1 ignores the switch, as both loops did -- the sequence then has
# the Final entry at every outer corrector, ppFppFppF, and the fields are 6.7e-02 from OpenFOAM's.
wcase damBreakPermeable of > "$W/fo_stage.txt" 2>&1
o="$W/w_of_damBreakPermeable"
[ -d "$o" ] || { say "damBreakPermeable did not stage" FAIL; finish "finalOnLastPimpleIterOnly"; }
restage()   # restage <dir>: three outer correctors, the switch on, p_rgh loose and p_rghFinal tight
{
    mkdir -p "$1"
    cp -r "$o/0" "$o/constant" "$o/system" "$1/"
    python3 - "$1/system/fvSolution" <<'PY'
import re, sys
p = sys.argv[1]
t = open(p).read()
t, a = re.subn(r'(\n    p_rgh\n    \{[^}]*?tolerance\s+)[^;]+;', r'\g<1>1e-4;', t)
t, b = re.subn(r'(\n    p_rghFinal\n    \{\n        \$p_rgh;\n)', r'\g<1>        tolerance       1e-13;\n', t)
t, c = re.subn(r'nOuterCorrectors\s+\d+;', 'nOuterCorrectors    3;\n    finalOnLastPimpleIterOnly yes;', t)
assert (a, b, c) == (1, 1, 1), (a, b, c)
open(p, 'w').write(t)
PY
}
# entries <log>: one letter a p_rgh solve, F where its final residual met the Final entry's 1e-13
entries()
{
    grep -a "Solving for p_rgh," "$1" | sed -E 's/.*Final residual = ([^,]+),.*/\1/' \
        | awk '{printf "%s", ($1 + 0 < 1e-9) ? "F" : "p"} END {print ""}'
}
restage "$W/fo_of"
( cd "$W/fo_of" && interFoam > log.interFoam 2>&1 ) || { say "real interFoam did not run the fixture" FAIL; }
for v in host device control; do
    restage "$W/fo_$v"
    case $v in
        host)    runbrae "$W/fo_$v" host BRAE_PRINT_TURB_SOLVES=1 ;;
        device)  runbrae "$W/fo_$v" device BRAE_PRINT_TURB_SOLVES=1 ;;
        control) runbrae "$W/fo_$v" device BRAE_PRINT_TURB_SOLVES=1 BRAE_CONTROL_FINAL_ON_EVERY_OUTER=1 ;;
    esac
    python3 "$CMP" "$W/fo_of" "$W/fo_$v" $(timedirs "$W/fo_of") > "$W/cmp_fo_$v.txt" 2>&1
done
eo=$(entries "$W/fo_of/log.interFoam")
eh=$(entries "$W/fo_host/log.brae")
ed=$(entries "$W/fo_device/log.brae")
ec=$(entries "$W/fo_control/log.brae")
nF=$(printf '%s' "$eo" | tr -cd F | wc -c)
what="OpenFOAM takes the Final entry $nF times in ${#eo} solves ($eo); brae's host and device loops the same"
[ "${#eo}" = 18 ] && [ "$nF" = 2 ] && [ "$eh" = "$eo" ] && [ "$ed" = "$eo" ] \
    && say "$what" ok || say "$what (host $eh, device $ed)" FAIL
B=3e-12
judge "final on the last outer, host" "$W/cmp_fo_host.txt" "$B" "$W/fo_of/log.interFoam" \
    && judge "final on the last outer, device" "$W/cmp_fo_device.txt" "$B" "$W/fo_of/log.interFoam" \
    && say "[host, device] every written file within $B of OpenFOAM" ok \
    || say "[host, device] every written file within $B of OpenFOAM" FAIL
what="CONTROL  the switch ignored: the Final entry at every outer corrector ($ec), not OpenFOAM's sequence"
[ "$ec" != "$eo" ] && [ "$(printf '%s' "$ec" | tr -cd F | wc -c)" = 6 ] && say "$what" ok || say "$what" FAIL
finish "finalOnLastPimpleIterOnly moves the Final pressure entry to the last outer corrector, as OpenFOAM does"
