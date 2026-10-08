#!/usr/bin/env bash
# The turbulence rule's SCOPE: where it does not apply, the case's own cell order runs -- to the byte.
. "$(dirname "$0")/../lib.sh"
# The rule takes a Gauss-Seidel smoothSolver on k, epsilon or omega asked to converge (relTol 0), on a mesh of
# 4,000 cells or more (device_inter_turbulence.cu). Two cases it must leave alone, each held byte for byte
# against the run with the rule switched off (BRAE_TURBULENCE_CASE_SOLVER=1):
#   an entry that stops at a relative tolerance -- weirOverflow (5,080 cells) restaged with `relTol 0.1`;
#   a mesh under the size -- RAS/damBreakPorousBaffle's 2,268 cells, where the log has to say so. MEASURED at
#   that size, ms a step, colour order / own: damBreakLeakage 13.6 / 12.6, damBreakPorousBaffle 15.1 / 15.1.
# CONTROL: weirOverflow as the row pins it (relTol 0). There the rule applies, the log says so, and the files
# are NOT the switched-off run's: if they were, "byte for byte" above would hold for a switch that does nothing.
# (A one-to-one coupled pair is carried since 2026-10-06 -- porous_baffle.sh, dam_break_leakage.sh; a cyclicAMI's
# stencil is refused by the sweep itself, which tests/test_colour_gs_fused.cu holds at arm j2.)
unset BRAE_TURBULENCE_CASE_SOLVER
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
unset BRAE_TURBULENCE_COLOUR_MIN_CELLS
wcase weirOverflow of > "$W/sc_stage_weir.txt" 2>&1
wcase damBreakPorousBaffle of > "$W/sc_stage_baffle.txt" 2>&1
ow="$W/w_of_weirOverflow"
ob="$W/w_of_damBreakPorousBaffle"
[ -d "$ow" ] && [ -d "$ob" ] \
    || { say "weirOverflow or damBreakPorousBaffle did not stage" FAIL; finish "the rule's scope"; }
for v in loose_rule loose_own small_rule small_own tight_rule tight_own; do
    e="$W/sc_$v"
    mkdir -p "$e"
    case $v in
        small_*) cp -r "$ob/0" "$ob/constant" "$ob/system" "$e/" ;;
        *)       cp -r "$ow/0" "$ow/constant" "$ow/system" "$e/" ;;
    esac
    case $v in
        loose_*) python3 - "$e/system/fvSolution" <<'PY'
import re, sys
p = sys.argv[1]
t = open(p).read()
def loose(m):
    return re.sub(r'relTol\s+[^;]+;', 'relTol          0.1;', m.group(0))
t, n = re.subn(r'"\(U\|k\|epsilon\)[^"]*"\s*\{[^}]*\}', loose, t)
assert n >= 1
open(p, 'w').write(t)
PY
        ;;
    esac
    case $v in
        *_rule) runbrae "$e" device ;;
        *_own)  runbrae "$e" device BRAE_TURBULENCE_CASE_SOLVER=1 ;;
    esac
done
differs()
{
    local n=0 t f
    for t in $(timedirs "$1"); do
        for f in $(cd "$1/$t" && find . -type f | sort); do
            cmp -s "$1/$t/$f" "$2/$t/$f" || n=$((n + 1))
        done
    done
    echo $n
}
mark="brae sweeps it in colour order"
nl=$(differs "$W/sc_loose_rule" "$W/sc_loose_own")
what="an entry at relTol 0.1 keeps the case's own order: no notice, $nl written files differ from the rule off"
! grep -aq "$mark" "$W/sc_loose_rule/log.brae" && [ "$nl" = 0 ] && [ -n "$(timedirs "$W/sc_loose_rule")" ] \
    && say "$what" ok || say "$what" FAIL
ns=$(differs "$W/sc_small_rule" "$W/sc_small_own")
what="a mesh of 2,268 cells keeps the case's own order and says so: $ns written files differ from the rule off"
! grep -aq "$mark" "$W/sc_small_rule/log.brae" && [ "$ns" = 0 ] && [ -n "$(timedirs "$W/sc_small_rule")" ] \
    && grep -aq "keeps the case's own smoothSolver order for k and its partner" "$W/sc_small_rule/log.brae" \
    && say "$what" ok || say "$what" FAIL
nt=$(differs "$W/sc_tight_rule" "$W/sc_tight_own")
what="CONTROL  at relTol 0 the rule applies and is announced: $nt written files differ from the rule off"
grep -aq "$mark" "$W/sc_tight_rule/log.brae" && ! grep -aq "$mark" "$W/sc_tight_own/log.brae" && [ "$nt" -gt 0 ] \
    && say "$what" ok || say "$what" FAIL
finish "the turbulence rule leaves a relative-tolerance entry and a small mesh to the case's own order"
