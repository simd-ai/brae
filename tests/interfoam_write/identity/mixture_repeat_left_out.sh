#!/usr/bin/env bash
# The write gate: the mixture.correct() after the alpha sub-cycle LEFT OUT where it repeats the last
# corrector's, against the pass made as before (BRAE_CONTROL_MIXTURE_REPEAT_KEPT=1), on four W rows:
# laminar/waves/stokesI (explicit MULES, three sub-cycles, a wave inlet: the pass that evaluates alpha's
# patches); laminar/waves/solitaryGrimshaw (the same, and its p_rgh names `phi rhoPhi`, whose boundary the
# sub-cycle has just summed -- what is left of the hook pushes the flux whole there); laminar/damBreakPermeable
# (MULESCorr with a relaxed second corrector: the pass that does not evaluate, and the two wall conditions that
# read alpha's patch values -- what is left of the hook tells them); and laminar/sloshingTank2D, a mesh that
# moves, rigidly, under its sub-cycles (the dynamic-mesh hooks run in the step).
# interFoam.C:154 calls mixture.correct() after alphaEqnSubCycle.H, and alphaEqn.H:225 has just called it at
# the bottom of the last corrector: the same curvature, normal and mixture from the same alpha. The device loop
# made both, a host hook and a dozen kernels each. MEASURED 2026-10-05 over three pairs:
# laminar/waves/streamFunction 37.2-37.9 -> 36.3-36.4 ms a step, RAS/electrostaticDeposition 80.4-82.9 ->
# 76.6-79.6.
# HELD, on each row, to the made arm: (a) file for file, the DEFAULT run -- no switch at all -- and the run with
# BRAE_CONTROL_MIXTURE_REPEAT_CHECK=1, which runs what is left of the hook as the default does, then makes the
# pass and compares the eight device buffers it writes and the host's state before and after, bitwise, and
# must say so at every step with both counts above zero; (b) the hook's CALLS a step, from a second pair run
# with the phase table (BRAE_INTER_PHASE_TIME=1, which no byte comparison here leans on): the default makes
# exactly one call fewer of the hook the last pass takes (updateBoundary, or mixtureCorrect where the last
# corrector relaxed) and one of what is left -- a default run that said `left out` and made the pass would
# write the same files.
# CONTROLS, one line. laminar/capillaryRise, whose wall is a contact angle, where the pass is NOT a repeat:
# the default run says it makes it and writes the made arm's files; left out all the same
# (BRAE_CONTROL_MIXTURE_REPEAT_LEFT_OUT_ANYWAY=1) the run finishes and its files differ; under the check it
# stops at a device buffer, and with `=host` at the host's state. RAS/damBreakPorousBaffle (a coupled pair) says
# it makes the pass, and so does stokesI restaged with `phi rhoPhi;` on ALPHA's `top` (an alpha patch that
# names the flux the sub-cycle has just summed). And what is left of the hook dropped
# (BRAE_CONTROL_MIXTURE_REPEAT_NOTHING_LEFT=1) under the check on solitaryGrimshaw: the host's rhoPhi boundary
# is then the last sub-step's and the check stops.
# The check over every tutorial of the table, 2026-10-05: 38 of 42 leave the pass out, and at every one of
# their 1,137 steps the pass left the eight buffers and the host's state as they stood; four make it
# (capillaryRise's contact angle, the three with a coupled pair). Three skeptics reading the code as it stood
# before the change for a configuration that breaks the claim found none.
# DOES NOT CLAIM: file identity on a mesh whose cells CHANGE VOLUME under a sub-cycle (the wave makers: they
# pass the in-code check at every step, and their device arm does not reproduce run to run, so no files are
# compared; sloshingTank2D's motion is rigid). The answer of the caller nothing here gives: an alpha patch of
# a class not known to repeat (no shipped tutorial has one). The host loop, which this does not touch.
# NOT DISCRIMINATED without the check: what is left of the hook where no patch names rhoPhi. MEASURED
# 2026-10-05 with it dropped, its CONTROL MODE line and the missing phase row counted in that arm: 0 of 24
# files differ from the default's on damBreakPermeable and on stokesI -- the U hook pushes the same again
# before anything reads it.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
out="mixture.correct() after the alpha sub-cycle: left out, it repeats the last corrector's"
made="mixture.correct() after the alpha sub-cycle: made"
counted="mixture repeat check: [1-9][0-9]* passes made where they would be left out"
# same <dirA> <dirB> <oracle>: how many files of A are not B's, byte for byte; "-" unless both hold the
# oracle's times and the same files at each
same()
{
    local a="$1" b="$2" o="$3" n=0 d f
    [ "$(timedirs "$a")" = "$(timedirs "$o")" ] && [ "$(timedirs "$b")" = "$(timedirs "$o")" ] || { echo "-"; return; }
    for d in $(timedirs "$o"); do
        [ "$(filesets "$a" "$d")" = "$(filesets "$b" "$d")" ] || { echo "-"; return; }
        for f in $(cd "$a/$d" && find . -type f | sort); do
            cmp -s "$a/$d/$f" "$b/$d/$f" || n=$((n + 1))
        done
    done
    echo $n
}
# stage <dir> <key>: a fresh copy of a W row
stage()
{
    rm -rf "${1:?}"
    mkdir -p "$1"
    cp -r "$W/w_of_$2/0" "$W/w_of_$2/constant" "$W/w_of_$2/system" "$1/"
}
# stopped <dir> [env...]: an arm that is expected to stop, or to finish wrong -- run directly, runbrae would
# end the gate; its exit status is left in exit.txt
stopped()
{
    local d="$1"
    shift
    ( cd "$d" && env "$@" "$BIN" -case . -device > log.brae 2>&1; echo $? > exit.txt )
}
# arms <key> <tag>: the made arm, the default arm (no switch) and the checked arm of one W row, whose files are
# compared; and the made and the default arm once more WITH the phase table, whose call counts are read. NOT
# inside a substitution: an arm that stops ends the gate with runbrae's own message and the log's tail
arms()
{
    local tag="$2" v
    for v in made default checked madeTimed defaultTimed; do
        stage "$W/${tag}_$v" "$1"
        case $v in
            made) runbrae "$W/${tag}_$v" device BRAE_CONTROL_MIXTURE_REPEAT_KEPT=1 ;;
            default) runbrae "$W/${tag}_$v" device ;;
            checked) runbrae "$W/${tag}_$v" device BRAE_CONTROL_MIXTURE_REPEAT_CHECK=1 ;;
            madeTimed) runbrae "$W/${tag}_$v" device BRAE_INTER_PHASE_TIME=1 BRAE_CONTROL_MIXTURE_REPEAT_KEPT=1 ;;
            defaultTimed) runbrae "$W/${tag}_$v" device BRAE_INTER_PHASE_TIME=1 ;;
        esac
    done
}
# calls <log> <hook row>: that row's calls a step in the phase table, 0 where the row is absent
calls()
{
    local c
    c=$(grep -a "hook alpha\.$2 " "$1" | head -1 | sed -n -E 's/.*\(([0-9.]+) calls\/step\).*/\1/p')
    echo "${c:-0}"
}
# verdict <key> <tag> <update|mixture>: the three arms read, as "<ok|no> <files an arm> <default differs>
# <checked differs> <passes checked> <the last pass's hook: calls a step made> <default>". ok: each arm says
# which path it took; no written file of the default or the checked arm differs from the made arm's; the check
# says, at every step, that the buffers and the host's state stood; and the default arm makes one call a step
# fewer of the hook named -- none fewer of the other -- and one of what is left
verdict()
{
    local o="$W/w_of_$1" tag="$2" hook other t n1 n2 last passes steps said=ok
    local m="$W/${tag}_made/log.brae" d="$W/${tag}_default/log.brae" c="$W/${tag}_checked/log.brae"
    local mt="$W/${tag}_madeTimed/log.brae" dt="$W/${tag}_defaultTimed/log.brae"
    hook=updateBoundary
    other=mixtureCorrect
    [ "$3" = mixture ] && { hook=mixtureCorrect; other=updateBoundary; }
    t=$(for td in $(timedirs "$W/${tag}_made"); do find "$W/${tag}_made/$td" -type f; done | wc -l)
    n1=$(same "$W/${tag}_made" "$W/${tag}_default" "$o")
    n2=$(same "$W/${tag}_made" "$W/${tag}_checked" "$o")
    steps=$(timedirs "$o" | wc -w)
    last=$(grep -a "$counted, each leaving 8 buffers ([1-9][0-9]* values) and the host's state ([1-9][0-9]* values)" \
           "$c" | tail -1)
    passes=$(echo "$last" | sed -n -E 's/.*check: ([0-9]+) passes made.*/\1/p')
    grep -q "$out" "$d" && grep -q "$out" "$c" || said=no
    grep -q "$made (BRAE_CONTROL_MIXTURE_REPEAT_KEPT is set)" "$m" || said=no
    grep -q "$out" "$m" && said=no
    grep -q "$counted" "$d" && said=no
    [ "${passes:-0}" = "$steps" ] && [ "$t" -gt 0 ] && [ "$n1" = 0 ] && [ "$n2" = 0 ] || said=no
    grep -q "$made (BRAE_CONTROL_MIXTURE_REPEAT_KEPT is set)" "$mt" && grep -q "$out" "$dt" || said=no
    python3 -c "import sys; sys.exit(0 if abs($(calls "$mt" $hook) - $(calls "$dt" $hook) - 1.0) < 1e-9 \
        and $(calls "$mt" $other) == $(calls "$dt" $other) and $(calls "$dt" mixtureRepeatLeftOut) == 1.0 \
        and $(calls "$mt" mixtureRepeatLeftOut) == 0 else 1)" || said=no
    echo "$said $t $n1 $n2 ${passes:-0} $(calls "$mt" $hook) $(calls "$dt" $hook)"
}
for k in stokesI solitaryGrimshaw damBreakPermeable sloshingTank2D capillaryRise damBreakPorousBaffle; do
    wcase $k of >> "$W/mr_stage.txt" 2>&1
    [ -d "$W/w_of_$k" ] || { say "$k did not stage" FAIL; finish "mixture repeat identity"; }
done
grep -q "rhoPhi" "$W/w_of_solitaryGrimshaw/0/p_rgh" \
    || { say "PREMISE  solitaryGrimshaw's p_rgh names rhoPhi" FAIL; finish "mixture repeat identity"; }
arms stokesI mrs
arms solitaryGrimshaw mrg
arms damBreakPermeable mrp
arms sloshingTank2D mrt
# PREMISE: the mesh moved between the row's two steps. The points the made arm wrote at its last time against
# those it wrote at its first, as NUMBERS (the files' headers differ whether or not a point moved)
firstTime=$(timedirs "$W/mrt_made" | awk '{print $1}')
lastTime=$(timedirs "$W/mrt_made" | awk '{print $NF}')
movedBy=$(python3 - "$W/mrt_made/$lastTime/polyMesh/points" "$W/mrt_made/$firstTime/polyMesh/points" <<'PY'
import gzip, os, re, sys
def points(path):
    if not os.path.exists(path) and os.path.exists(path + '.gz'):
        text = gzip.open(path + '.gz', 'rt').read()
    elif os.path.exists(path):
        text = open(path).read()
    else:
        return []
    body = text[text.index('// * *'):] if '// * *' in text else text
    num = r'([-+0-9.eE]+)'
    triple = r'\(\s*' + num + r'\s+' + num + r'\s+' + num + r'\s*\)'
    return [tuple(float(v) for v in m) for m in re.findall(triple, body)]
a, b = points(sys.argv[1]), points(sys.argv[2])
if not a or len(a) != len(b):
    print('-')
else:
    print('%.3e' % max(abs(p[k] - q[k]) for p, q in zip(a, b) for k in range(3)))
PY
)
above "$movedBy" 1e-4 \
    || { say "PREMISE  sloshingTank2D's mesh moved in the row's steps [$movedBy m]" FAIL; finish "mixture repeat"; }
read -r v1 t1 a1 b1 p1 m1 d1 <<< "$(verdict stokesI mrs update)"
read -r v2 t2 a2 b2 p2 m2 d2 <<< "$(verdict solitaryGrimshaw mrg update)"
what="[device] explicit: stokesI $t1 files, $a1 and $b1 differ, hook $m1 -> $d1 calls a step; solitaryGrimshaw"
what="$what (rhoPhi) $t2, $a2 and $b2, $m2 -> $d2"
[ "$v1" = ok ] && [ "$v2" = ok ] && say "$what" ok || say "$what" FAIL
read -r v3 t3 a3 b3 p3 m3 d3 <<< "$(verdict damBreakPermeable mrp mixture)"
read -r v4 t4 a4 b4 p4 m4 d4 <<< "$(verdict sloshingTank2D mrt update)"
what="[device] relaxed: damBreakPermeable $t3 files, $a3 and $b3 differ, hook $m3 -> $d3; moving: sloshingTank2D"
what="$what $t4, $a4 and $b4, $m4 -> $d4"
[ "$v3" = ok ] && [ "$v4" = ok ] && say "$what" ok || say "$what" FAIL
# the controls
stage "$W/mrc_made" capillaryRise
runbrae "$W/mrc_made" device BRAE_CONTROL_MIXTURE_REPEAT_KEPT=1
stage "$W/mrc_default" capillaryRise
runbrae "$W/mrc_default" device
stage "$W/mrc_anyway" capillaryRise
stopped "$W/mrc_anyway" BRAE_CONTROL_MIXTURE_REPEAT_LEFT_OUT_ANYWAY=1
stage "$W/mrc_anyc" capillaryRise
stopped "$W/mrc_anyc" BRAE_CONTROL_MIXTURE_REPEAT_LEFT_OUT_ANYWAY=1 BRAE_CONTROL_MIXTURE_REPEAT_CHECK=1
stage "$W/mrc_anyh" capillaryRise
stopped "$W/mrc_anyh" BRAE_CONTROL_MIXTURE_REPEAT_LEFT_OUT_ANYWAY=1 BRAE_CONTROL_MIXTURE_REPEAT_CHECK=host
stage "$W/mrb_default" damBreakPorousBaffle
runbrae "$W/mrb_default" device
stage "$W/mrg_nothing" solitaryGrimshaw
stopped "$W/mrg_nothing" BRAE_CONTROL_MIXTURE_REPEAT_CHECK=1 BRAE_CONTROL_MIXTURE_REPEAT_NOTHING_LEFT=1
stage "$W/mrs_rhophi" stokesI
restaged=yes
python3 - "$W/mrs_rhophi/0/alpha.water" <<'PY' || restaged=no
import re, sys
p = sys.argv[1]
s = open(p).read()
m = re.search(r'\n    top\n    \{\n        type            inletOutlet;\n', s)
if not m:
    sys.exit(1)
open(p, 'w').write(s[:m.end()] + '        phi             rhoPhi;\n' + s[m.end():])
PY
[ $restaged = yes ] || { say "stokesI's alpha top was not restaged" FAIL; finish "mixture repeat identity"; }
runbrae "$W/mrs_rhophi" device
# named <dir>: what the check's message says did not repeat
named()
{
    grep -a "MIXTURE_REPEAT_CHECK: .* does not repeat the last corrector's:" "$1/log.brae" | head -1 \
        | sed -E "s/.*does not repeat the last corrector's: ([^,]+), entry.*/\1/"
}
o="$W/w_of_capillaryRise"
keptSame=$(same "$W/mrc_made" "$W/mrc_default" "$o")
anywayDiffers=$(same "$W/mrc_default" "$W/mrc_anyway" "$o")
buf=$(named "$W/mrc_anyc")
host=$(named "$W/mrc_anyh")
left=$(named "$W/mrg_nothing")
ok=yes
grep -q "$made (alpha's patch .* is a contact angle" "$W/mrc_default/log.brae" && [ "$keptSame" = 0 ] || ok=no
anyLine="CONTROL MODE: the mixture.correct() after the alpha sub-cycle is left out though"
[ "$(cat "$W/mrc_anyway/exit.txt")" = 0 ] && grep -q "$anyLine" "$W/mrc_anyway/log.brae" \
    && [ "$anywayDiffers" != - ] && [ "$anywayDiffers" -gt 0 ] || ok=no
[ "$(cat "$W/mrc_anyc/exit.txt")" != 0 ] && [ -n "$buf" ] && [ "$buf" != "the host's state" ] || ok=no
[ "$(cat "$W/mrc_anyh/exit.txt")" != 0 ] && [ "$host" = "the host's state" ] || ok=no
grep -q "$made (a coupled pair's interface sums are atomic)" "$W/mrb_default/log.brae" || ok=no
grep -q "$made (alpha's patch \`top\` names rhoPhi" "$W/mrs_rhophi/log.brae" || ok=no
[ "$(cat "$W/mrg_nothing/exit.txt")" != 0 ] && [ "$left" = "the host's state" ] \
    && grep -q "CONTROL MODE: what is left of the alpha hook where that pass is left out is dropped" \
            "$W/mrg_nothing/log.brae" || ok=no
what="CONTROL  contact angle: made by default ($keptSame differ), left out anyway $anywayDiffers differ, the"
what="$what check stops at ${buf:-?} / ${host:-?}; a pair and an alpha rhoPhi patch keep it; nothing left: stops"
[ $ok = yes ] && say "$what" ok || say "$what" FAIL
finish "the mixture.correct() after the sub-cycle is left out only where it repeats the last corrector's"
