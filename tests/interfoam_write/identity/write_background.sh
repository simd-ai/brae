#!/usr/bin/env bash
# The write gate: a time directory formatted and written IN THE BACKGROUND, against the same directory
# formatted and written inside the step (BRAE_CONTROL_WRITE_IN_STEP=1, the path every write gate held to
# OpenFOAM until 2026-10-06 -- and the default path is what those gates hold now).
# A write cost the step 386 ms on laminar/waves/waveMakerPiston (56,000 cells, 16 MB of ASCII in 12 files):
# seven time steps' worth, all but 10 ms of it formatting numbers on the solver's thread. Now the step builds
# every file's structure and COPIES each long list; the lists' text, the files and purgeWrite's removals are
# a job on another thread. MEASURED on that case with a write every 30 steps: the write's share of a step
# 11.7 -> 0.1 ms, the step 65.9 -> 54.1.
# THE FIXTURES: six pinned steps with a write every second one, so two writes are still in flight while the
# next steps run and the last is waited for at the end. waveMakerFlap on the device loop (a moving mesh: the
# points, Uf, meshPhi and the motion's fields are written too) and capillaryRise on the host loop.
# TWO CONTROLS. The copy is what makes a background write safe, so one leaves alpha's list UNCOPIED and makes
# the write when the next one waits for it, two steps on (host loop, which hands the writer alpha's own
# cells): the first two directories' alpha files must then differ, and only they (the last is waited for at
# the end of the run, with alpha where its write left it). The other fails the job behind the solver: the
# run must stop, say so, and leave no time directory.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase waveMakerFlap of > "$W/wb_stage_d.txt" 2>&1
wcase capillaryRise of > "$W/wb_stage_h.txt" 2>&1
od="$W/w_of_waveMakerFlap"
oh="$W/w_of_capillaryRise"
[ -d "$od" ] && [ -d "$oh" ] || { say "the two rows did not stage" FAIL; finish "background write identity"; }
arm()   # arm <name> <staged case>: its 0, constant and system, six steps, a write every second one
{
    local e="$W/wb_$1"
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$2/0" "$2/constant" "$2/system" "$e/"
    python3 - "$e/system/controlDict" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
dt = float(re.search(r'^deltaT\s+(\S+);', s, flags=re.M).group(1))
for k, v in [('endTime', '%.12g' % (6*dt)), ('writeInterval', '2')]:
    s, n = re.subn(r'^%s\s.*' % k, '%s %s;' % (k.ljust(15), v), s, flags=re.M)
    assert n == 1, k
open(p, 'w').write(s)
PY
}
differs()   # differs <a> <b>: the files of a's time directories that are not b's bytes, and how many there are
{
    local n=0 t=0 d f
    for d in $(timedirs "$1"); do
        for f in $(cd "$1/$d" && find . -type f | sort); do
            t=$((t + 1))
            cmp -s "$1/$d/$f" "$2/$d/$f" || n=$((n + 1))
        done
    done
    echo "$n $t"
}
arm dstep "$od"
runbrae "$W/wb_dstep" device BRAE_CONTROL_WRITE_IN_STEP=1
arm dbg "$od"
runbrae "$W/wb_dbg" device
arm hstep "$oh"
runbrae "$W/wb_hstep" host BRAE_CONTROL_WRITE_IN_STEP=1
arm hbg "$oh"
runbrae "$W/wb_hbg" host
arm halias "$oh"
runbrae "$W/wb_halias" host BRAE_CONTROL_WRITE_ALPHA_ALIASED=1
arm dfail "$od"
# this control is expected to stop: run it directly, runbrae would end the gate
( cd "$W/wb_dfail" && BRAE_CONTROL_WRITE_FAIL_IN_BACKGROUND=1 "$BIN" -case . -device > log.brae 2>&1
  echo $? > exit.txt )

bg="long lists are formatted and its files written in the background"
read -r nd td <<< "$(differs "$W/wb_dstep" "$W/wb_dbg")"
read -r nh th <<< "$(differs "$W/wb_hstep" "$W/wb_hbg")"
what="the background write's bytes are the in-step write's: device $nd of $td files differ, host $nh of $th"
grep -q "$bg" "$W/wb_dbg/log.brae" && grep -q "$bg" "$W/wb_hbg/log.brae" \
    && grep -q "inside the step (BRAE_CONTROL_WRITE_IN_STEP)" "$W/wb_dstep/log.brae" \
    && grep -q "inside the step (BRAE_CONTROL_WRITE_IN_STEP)" "$W/wb_hstep/log.brae" \
    && [ "$nd" = 0 ] && [ "$nh" = 0 ] && [ "$td" -ge 30 ] && [ "$th" -ge 20 ] \
    && [ "$(timedirs "$W/wb_dbg" | wc -w)" = 3 ] && [ "$(timedirs "$W/wb_hbg" | wc -w)" = 3 ] \
    && [ "$(timedirs "$W/wb_dbg")" = "$(timedirs "$W/wb_dstep")" ] \
    && [ "$(timedirs "$W/wb_hbg")" = "$(timedirs "$W/wb_hstep")" ] \
    && say "$what" ok || say "$what" FAIL
na=0
other=0
for d in $(timedirs "$W/wb_hstep"); do
    for f in $(cd "$W/wb_hstep/$d" && find . -type f | sort); do
        cmp -s "$W/wb_hstep/$d/$f" "$W/wb_halias/$d/$f" && continue
        case "$f" in ./alpha.*) na=$((na + 1)) ;; *) other=$((other + 1)) ;; esac
    done
done
what="CONTROL  alpha's list not copied, the write made at the next: $na alpha files differ, $other others"
grep -q "CONTROL MODE: alpha's list is not copied" "$W/wb_halias/log.brae" \
    && [ "$na" = 2 ] && [ "$other" = 0 ] && say "$what" ok || say "$what" FAIL
left=$(timedirs "$W/wb_dfail" | wc -w)
what="CONTROL  the job fails behind the solver: the run stops and says so, $left time directories written"
[ "$(cat "$W/wb_dfail/exit.txt")" != 0 ] && [ "$left" = 0 ] \
    && grep -q "BRAE_CONTROL_WRITE_FAIL_IN_BACKGROUND fails" "$W/wb_dfail/log.brae" \
    && say "$what" ok || say "$what" FAIL
finish "a time directory written in the background is the one written inside the step"
