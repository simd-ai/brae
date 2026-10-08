#!/usr/bin/env bash
# The write gate: the momentum's relaxation factor BY THE NAME fvMatrix::relax() asks for -- `UFinal` on the
# final outer corrector, which is every step at nOuterCorrectors 1, and `U` on the others (fvMatrix.C:1249-1263;
# solution.C:379-416: the name, else `default`, else no relaxation at all). Both loops looked up `U` once and
# used it on every corrector.
# TWO FIXTURES on laminar/damBreakPermeable's pinned row, each against real interFoam on the same staged case.
# `alone`: `equations { U 0.5; }` at the tutorial's one outer corrector -- nothing answers for UFinal, so
# OpenFOAM does not relax at all. `both`: `equations { U 0.7; UFinal 1; }` with three outer correctors.
# Three checks. (1) THE PREMISE of `alone`, on OpenFOAM: its run writes the files of its run with an empty
# equations block, byte for byte -- and brae's loops are within the bound of it. (2) `both`: each loop within
# the bound. (3) CONTROL: BRAE_CONTROL_RELAX_U_NAME_ONLY=1, `U` for every corrector, on both fixtures.
# MEASURED 2026-10-07: `alone` host 1.6e-13, device 7.8e-14 (phi); `both` host 5.7e-13, device 8.3e-13
# (alphaPhi0), so the bound is 9e-12; the control's U is 1.0e+00 from OpenFOAM's on `alone` (two steps from
# rest, every bit of it the relaxation OpenFOAM does not make) and 2.4e-02 on `both`.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase damBreakPermeable of > "$W/rf_stage.txt" 2>&1
o="$W/w_of_damBreakPermeable"
[ -d "$o" ] || { say "damBreakPermeable did not stage" FAIL; finish "the Final relaxation name"; }
restage()   # restage <dir> <alone|none|both>
{
    rm -rf "${1:?}"
    mkdir -p "$1"
    cp -r "$o/0" "$o/constant" "$o/system" "$1/"
    python3 - "$1/system/fvSolution" "$2" <<'PY'
import re, sys
p, kind = sys.argv[1:3]
t = open(p).read()
eq = {'alone': '        U               0.5;',
      'none': '',
      'both': '        U               0.7;\n        UFinal          1;'}[kind]
block = 'relaxationFactors\n{\n    equations\n    {\n' + eq + '\n    }\n}'
t, a = re.subn(r'relaxationFactors\s*\{.*?\n\}', block, t, flags=re.S)
b = 1
if kind == 'both':
    t, b = re.subn(r'nOuterCorrectors\s+\d+;', 'nOuterCorrectors    3;', t)
assert (a, b) == (1, 1), (a, b)
open(p, 'w').write(t)
PY
}
for k in alone none both; do
    restage "$W/rf_of_$k" $k
    runof "$W/rf_of_$k"
done
ot=$(timedirs "$W/rf_of_alone")
n=0
for t in $ot; do
    for f in $(cd "$W/rf_of_alone/$t" && find . -type f | sort); do
        cmp -s "$W/rf_of_alone/$t/$f" "$W/rf_of_none/$t/$f" || n=$((n + 1))
    done
done
for k in alone both; do
    for v in host device control; do
        restage "$W/rf_${k}_$v" $k
        case $v in
            host)    runbrae "$W/rf_${k}_$v" host BRAE_X=1 ;;
            device)  runbrae "$W/rf_${k}_$v" device BRAE_X=1 ;;
            control) runbrae "$W/rf_${k}_$v" host BRAE_CONTROL_RELAX_U_NAME_ONLY=1 ;;
        esac
        python3 "$CMP" "$W/rf_of_$k" "$W/rf_${k}_$v" $ot > "$W/cmp_rf_${k}_$v.txt" 2>&1
    done
done
B=9e-12
what="\`U 0.5\` alone at one outer corrector: OpenFOAM's run differs from its unrelaxed one in $n files;"
what="$what [host, device] within $B of it"
[ "$(echo $ot | wc -w)" = 2 ] && [ "$n" = 0 ] \
    && judge "U alone, host" "$W/cmp_rf_alone_host.txt" "$B" "$W/rf_of_alone/log.interFoam" \
    && judge "U alone, device" "$W/cmp_rf_alone_device.txt" "$B" "$W/rf_of_alone/log.interFoam" \
    && say "$what" ok || say "$what" FAIL
what="\`U 0.7; UFinal 1\` with three outer correctors: [host, device] within $B of OpenFOAM"
judge "U and UFinal, host" "$W/cmp_rf_both_host.txt" "$B" "$W/rf_of_both/log.interFoam" \
    && judge "U and UFinal, device" "$W/cmp_rf_both_device.txt" "$B" "$W/rf_of_both/log.interFoam" \
    && say "$what" ok || say "$what" FAIL
judge "control, alone" "$W/cmp_rf_alone_control.txt" "$B" "$W/rf_of_alone/log.interFoam" > "$W/rf_c1.txt"; c1=$?
judge "control, both" "$W/cmp_rf_both_control.txt" "$B" "$W/rf_of_both/log.interFoam" > "$W/rf_c2.txt"; c2=$?
u1=$(grep -oE "over the bound: [0-9.e+-]+/U [0-9.e+-]+" "$W/rf_c1.txt" | tail -1 | awk '{print $NF}')
u2=$(grep -oE "over the bound: [0-9.e+-]+/U [0-9.e+-]+" "$W/rf_c2.txt" | tail -1 | awk '{print $NF}')
what="CONTROL  \`U\` for every corrector: U ${u1:--} from OpenFOAM on the first fixture, ${u2:--} on the second"
[ $c1 != 0 ] && [ $c2 != 0 ] && above "$u1" 1e-6 && above "$u2" 1e-6 && say "$what" ok || say "$what" FAIL
finish "the momentum is relaxed by the factor of the name fvMatrix::relax() asks for"
