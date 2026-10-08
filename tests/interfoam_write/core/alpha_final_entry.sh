#!/usr/bin/env bash
# The write gate: WHICH fvSolution entry the MULESCorr pre-solve takes. alphaEqn.H:122's alpha1Eqn.solve()
# looks the solver up by psi.select(final iteration) (fvMatrix.C:1536-1542): `<alpha>Final` on the final outer
# corrector, `<alpha>` on the others. Both loops read `<alpha>` alone and used it on every corrector; the
# tutorials write `"alpha.water.*"`, one block for both names.
# THE FIXTURE makes the two entries tell themselves apart, on laminar/damBreakPermeable's pinned row with two
# outer correctors: `alpha.water` stops at 1e-3 and `alpha.waterFinal` at 1e-13, so each pre-solve's final
# residual says which entry it took, in OpenFOAM's log and in brae's. ORACLE: real interFoam on the same case.
# Three checks. (1) The sequence of entries, solve for solve: OpenFOAM's, and both loops the same. (2) Every
# written file within the bound of OpenFOAM, both loops. (3) CONTROL: BRAE_CONTROL_ALPHA_ENTRY_NON_FINAL=1,
# `<alpha>` throughout -- no solve takes the Final entry and the fields go over.
# MEASURED 2026-10-07: OpenFOAM's sequence over the two steps is 0FaF, which both loops repeat, and the control's
# is 0aaa; the worst written file is 6.4e-13 on the host loop and 2.9e-13 on the device loop (alphaPhi0), so
# the bound is 7e-12; the control's alphaPhi0 is 8.8e+00 from OpenFOAM's.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase damBreakPermeable of > "$W/af_stage.txt" 2>&1
o="$W/w_of_damBreakPermeable"
[ -d "$o" ] || { say "damBreakPermeable did not stage" FAIL; finish "the alpha Final entry"; }
restage()   # restage <dir>: two outer correctors, alpha.water loose and alpha.waterFinal tight
{
    rm -rf "${1:?}"
    mkdir -p "$1"
    cp -r "$o/0" "$o/constant" "$o/system" "$1/"
    python3 - "$1/system/fvSolution" <<'PY'
import re, sys
p = sys.argv[1]
t = open(p).read()
m = re.search(r'\n    "alpha\.water\.\*"\n    \{(.*?)\n    \}', t, re.S)
assert m and 'MULESCorr       yes' in m.group(1)
loose = re.sub(r'(\n\s*tolerance\s+)[^;]+;', r'\g<1>1e-3;', m.group(1), count=1)
tight = re.sub(r'(\n\s*tolerance\s+)[^;]+;', r'\g<1>1e-13;', m.group(1), count=1)
assert loose != m.group(1) and tight != loose
two = '\n    alpha.water\n    {' + loose + '\n    }\n\n    alpha.waterFinal\n    {' + tight + '\n    }'
t = t.replace(m.group(0), two, 1)
t, c = re.subn(r'nOuterCorrectors\s+\d+;', 'nOuterCorrectors    2;', t)
assert c == 1
open(p, 'w').write(t)
PY
}
# entries <log>: one letter an alpha pre-solve -- F where its final residual met the Final entry's 1e-13, a
# where it stopped at alpha.water's 1e-3, and 0 for a solve that began at a zero residual and so says nothing
# (the first one: the column has not moved yet)
entries()
{
    grep -a "Solving for alpha.water," "$1" \
        | sed -E 's/.*Initial residual = ([^,]+), Final residual = ([^,]+),.*/\1 \2/' \
        | awk '{printf "%s", ($1 + 0 == 0) ? "0" : (($2 + 0 < 1e-9) ? "F" : "a")} END {print ""}'
}
restage "$W/af_of"
runof "$W/af_of"
ot=$(timedirs "$W/af_of")
for v in host device control; do
    restage "$W/af_$v"
    case $v in
        host)    runbrae "$W/af_$v" host BRAE_PRINT_TURB_SOLVES=1 ;;
        device)  runbrae "$W/af_$v" device BRAE_PRINT_TURB_SOLVES=1 ;;
        control) runbrae "$W/af_$v" host BRAE_PRINT_TURB_SOLVES=1 BRAE_CONTROL_ALPHA_ENTRY_NON_FINAL=1 ;;
    esac
    python3 "$CMP" "$W/af_of" "$W/af_$v" $ot > "$W/cmp_af_$v.txt" 2>&1
done
eo=$(entries "$W/af_of/log.interFoam")
eh=$(entries "$W/af_host/log.brae")
ed=$(entries "$W/af_device/log.brae")
ec=$(entries "$W/af_control/log.brae")
nF=$(printf '%s' "$eo" | tr -cd F | wc -c)
what="OpenFOAM's pre-solves take the Final entry $nF times in ${#eo} ($eo); brae's host and device loops the same"
[ "${#eo}" = 4 ] && [ "$nF" = 2 ] && [ "$eh" = "$eo" ] && [ "$ed" = "$eo" ] \
    && say "$what" ok || say "$what (host $eh, device $ed)" FAIL
B=7e-12
judge "the alpha Final entry, host" "$W/cmp_af_host.txt" "$B" "$W/af_of/log.interFoam" \
    && judge "the alpha Final entry, device" "$W/cmp_af_device.txt" "$B" "$W/af_of/log.interFoam" \
    && say "[host, device] every written file within $B of OpenFOAM" ok \
    || say "[host, device] every written file within $B of OpenFOAM" FAIL
judge "control: alpha's entry throughout" "$W/cmp_af_control.txt" "$B" "$W/af_of/log.interFoam" > "$W/af_c.txt"
rc=$?
what="CONTROL  \`alpha.water\` on every corrector: its sequence $ec, and the fields over the bound"
[ "$ec" != "$eo" ] && [ "$(printf '%s' "$ec" | tr -cd F | wc -c)" = 0 ] && [ $rc != 0 ] \
    && say "$what" ok || say "$what" FAIL
finish "the MULESCorr pre-solve takes the Final entry on the final outer corrector, as OpenFOAM does"
