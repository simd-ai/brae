#!/usr/bin/env bash
# The smoothed-aggregation hierarchy's two sums that were atomicAdd scatters -- the restriction P^T r of every
# cycle and the coarse matrices P^T A P of every solve -- are gathered in a fixed order (amgSaFixedOrder). A
# scatter's order is whichever thread arrives first, and pcorr on a 2-D moving mesh takes this hierarchy by
# default: MEASURED 2026-10-06, as shipped, two runs of one binary wrote 44 of 51 files differently on
# waveMakerFlap and 79 of 102 on waveMakerPiston, and none after (runs/bench/shipped_repro.py).
# What the answer is held to is not here: amg_pcg/pcorr_wave_maker_piston.sh holds this default to OpenFOAM.
# Three checks on waveMakerFlap's pinned row. (1) THE SAME BYTES TWICE, on the single-precision cycle and on
# the double-precision one (BRAE_CONTROL_AMG_SA_DOUBLE=1): the two instantiations of the restriction.
# (2) IT IS THE SCATTER'S SOLVE: against BRAE_CONTROL_AMG_SA_SCATTER=1, pcorr's iterations within one, solve for
# solve, and every written file within a bound -- the scatter arm does not reproduce, so no closer oracle
# exists: five runs of it landed 1.4e-10 to 1.9e-10 from the fixed order's (U), the same iterations each
# time, so the bound is 2e-09. (3) CONTROL: BRAE_CONTROL_AMG_SA_GATHER_REVERSED=1 takes each sum from its last
# term to its first -- another fixed order -- and the written files have to change: the comparison of (1)
# sees a sum's order.
. "$(dirname "$0")/../lib.sh"
unset BRAE_PRESSURE_CASE_SOLVER
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase waveMakerFlap of > "$W/fo_stage.txt" 2>&1
o="$W/w_of_waveMakerFlap"
[ -d "$o" ] || { say "waveMakerFlap did not stage" FAIL; finish "smoothed hierarchy in a fixed order"; }
# arm <name> [env...]: brae on a copy of the staged case; never ends the gate
arm()
{
    local e="$W/fo_$1"
    shift
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    ( cd "$e" && env BRAE_PRINT_TURB_SOLVES=1 "$@" "$BIN" -case . -device > log.brae 2>&1; echo $? > exit.txt )
}
# differs <a> <b>: the number of written files that are not byte-identical
differs()
{
    local n=0 t f
    for t in $(timedirs "$o"); do
        for f in $(cd "$W/fo_$1/$t" && find . -type f | sort); do
            cmp -s "$W/fo_$1/$t/$f" "$W/fo_$2/$t/$f" || n=$((n + 1))
        done
    done
    echo $n
}
lines() { grep -a "Solving for pcorr" "$W/fo_$1/log.brae"; }
its() { lines "$1" | sed -E 's/.*No Iterations ([0-9]+).*/\1/' | tr '\n' ' '; }
# ran <arm>: it ended well, wrote OpenFOAM's times and solved pcorr
ran()
{
    [ "$(cat "$W/fo_$1/exit.txt")" = 0 ] && [ "$(timedirs "$W/fo_$1")" = "$(timedirs "$o")" ] && [ -n "$(its "$1")" ]
}
smoothed() { grep -aq "pcorr: its AMG hierarchy is a smoothed-aggregation one of its own" "$W/fo_$1/log.brae"; }
# near <a> <b>: the two arms' pcorr solves, one for one, each within one iteration of the other
near()
{
    local a=($(its $1)) b=($(its $2)) i d
    [ ${#a[@]} -gt 0 ] && [ ${#a[@]} = ${#b[@]} ] || return 1
    for i in "${!a[@]}"; do
        d=$(( a[i] - b[i] ))
        [ ${d#-} -le 1 ] || return 1
    done
}
arm first BRAE_X=1
arm second BRAE_X=1
arm double_first BRAE_CONTROL_AMG_SA_DOUBLE=1
arm double_second BRAE_CONTROL_AMG_SA_DOUBLE=1
arm scatter BRAE_CONTROL_AMG_SA_SCATTER=1
arm reversed BRAE_CONTROL_AMG_SA_GATHER_REVERSED=1
n1=$(differs first second)
n2=$(differs double_first double_second)
what="[device] two runs write the same bytes: $n1 files differ on the single-precision cycle, $n2 on the"
what="$what double-precision one; pcorr's iterations $(its first)and $(its double_first)"
ran first && ran second && ran double_first && ran double_second && smoothed first && smoothed double_first \
    && [ "$n1" = 0 ] && [ "$n2" = 0 ] && [ "$(lines first)" = "$(lines second)" ] \
    && [ "$(lines double_first)" = "$(lines double_second)" ] && say "$what" ok || say "$what" FAIL
B=2e-09
python3 "$CMP" "$W/fo_scatter" "$W/fo_first" $(timedirs "$o") > "$W/cmp_fo_scatter.txt" 2>&1
worst=$(python3 - "$W/cmp_fo_scatter.txt" <<'PY'
import json, sys
try:
    r = json.loads([l for l in open(sys.argv[1]) if l.startswith('RESULT ')][-1][7:])
    f = max(((k, v['rel']) for k, v in r['files'].items() if 'cumulativeContErr' not in k), key=lambda kv: kv[1])
    print('%.1e %s' % (f[1], f[0]))
except Exception:
    print('nan -')
PY
)
what="[device] it is the scatter's solve: pcorr's iterations $(its first)(scatter $(its scatter)), the worst"
what="$what file ${worst#* } ${worst%% *} from the scatter arm's, bound $B"
ran scatter && smoothed scatter && near first scatter \
    && python3 -c "import sys; sys.exit(0 if float('${worst%% *}') <= float('$B') else 1)" \
    && say "$what" ok || say "$what" FAIL
n=$(differs first reversed)
what="CONTROL  each sum taken from its last term to its first: $n written files change"
ran reversed && smoothed reversed && [ "$n" != 0 ] && say "$what" ok || say "$what" FAIL
finish "the smoothed hierarchy's restriction and coarse matrices are summed in a fixed order"
