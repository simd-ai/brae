#!/usr/bin/env bash
# The turbulence rule's SCOPE: where it does not apply, the case's own cell order runs -- to the byte.
. "$(dirname "$0")/../lib.sh"
# The rule takes a Gauss-Seidel smoothSolver on k, epsilon or omega asked to converge (relTol 0), and not across
# a coupled pair (device_inter_turbulence.cu). Two cases it must leave alone, each held byte for byte against
# the run with the rule switched off (BRAE_TURBULENCE_CASE_SOLVER=1):
#   an entry that stops at a relative tolerance -- weirOverflow restaged with `relTol 0.1` on k and epsilon;
#   a mesh with a coupled pair -- RAS/damBreakLeakage, where the log has to say the pair keeps the own order.
# CONTROL: weirOverflow as the row pins it (relTol 0). There the rule applies, the log says so, and the files
# are NOT the switched-off run's: if they were, "byte for byte" above would hold for a switch that does nothing.
unset BRAE_TURBULENCE_CASE_SOLVER
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase weirOverflow of > "$W/sc_stage_weir.txt" 2>&1
wcase damBreakLeakage of > "$W/sc_stage_leak.txt" 2>&1
ow="$W/w_of_weirOverflow"
ol="$W/w_of_damBreakLeakage"
[ -d "$ow" ] && [ -d "$ol" ] || { say "weirOverflow or damBreakLeakage did not stage" FAIL; finish "the rule's scope"; }
for v in loose_rule loose_own pair_rule pair_own tight_rule tight_own; do
    e="$W/sc_$v"
    mkdir -p "$e"
    case $v in
        pair_*) cp -r "$ol/0" "$ol/constant" "$ol/system" "$e/" ;;
        *)      cp -r "$ow/0" "$ow/constant" "$ow/system" "$e/" ;;
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
np=$(differs "$W/sc_pair_rule" "$W/sc_pair_own")
what="a mesh with a coupled pair keeps the case's own order and says so: $np written files differ from the rule off"
! grep -aq "$mark" "$W/sc_pair_rule/log.brae" && [ "$np" = 0 ] && [ -n "$(timedirs "$W/sc_pair_rule")" ] \
    && grep -aq "this mesh has a coupled pair, which the colour-order sweep does not carry" "$W/sc_pair_rule/log.brae" \
    && say "$what" ok || say "$what" FAIL
nt=$(differs "$W/sc_tight_rule" "$W/sc_tight_own")
what="CONTROL  at relTol 0 the rule applies and is announced: $nt written files differ from the rule off"
grep -aq "$mark" "$W/sc_tight_rule/log.brae" && ! grep -aq "$mark" "$W/sc_tight_own/log.brae" && [ "$nt" -gt 0 ] \
    && say "$what" ok || say "$what" FAIL
finish "the turbulence rule leaves a relative-tolerance entry and a coupled mesh to the case's own order"
