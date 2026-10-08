#!/usr/bin/env bash
# One arm of the write gate (tests/interfoam_write/lib.sh has the gate's header and helpers).
. "$(dirname "$0")/../lib.sh"

# R: a file brae cannot write yet -- a wave model's entry holding a sub-dictionary, which OpenFOAM would
# write back as a nested block and no shipped wave tutorial holds (staged here on laminar/waves/stokesI's
# inlet) -- is named at startup, before the first step, and the run stops at its first write time having
# written nothing. The arm moves as the writer grows: capillaryRise's contact angle, stokesI's wave state,
# then irregularMultiDirection's lists are written now. CONTROL: the same case whose only write time lies
# past endTime runs to its end: the refusal is of the OUTPUT, and a run that never reaches a write is not
# refused.
STK="$TUT/multiphase/interFoam/laminar/waves/stokesI"
subdict()   # subdict <case> -- an `extra { note 1; }` entry in waveProperties' inlet
{
    python3 - "$1/constant/waveProperties" <<'EOF_SD'
import re, sys
p = sys.argv[1]
s = open(p).read()
s, n = re.subn(r'(\ninlet\s*\n\{\n)', r'\1    extra\n    {\n        note            1;\n    }\n\n', s)
open(p, 'w').write(s)
sys.exit(0 if n == 1 else 1)
EOF_SD
}
if [ -d "$STK" ]; then
    stage "$STK" "$W/stk" 0.05 timeStep 3 0 adjustTimeStep=no deltaT=0.01 || exit 1
    subdict "$W/stk" || say "ARM R  the sub-dictionary was not staged in stokesI's waveProperties" FAIL
    # line-buffered: the refusal goes to stderr, the steps to stdout, and only then is file order time order
    ( cd "$W/stk" && stdbuf -oL -eL "$BIN" -case . > log.brae 2>&1 ); rc=$?
    python3 - "$W/stk/log.brae" "$rc" "$(timedirs "$W/stk")" <<'PY' && say "ARM R  an unwritten file is named before step 1; the run stops at its first write, nothing written" ok \
                                                                    || say "ARM R  an unwritten file is named before step 1; the run stops at its first write, nothing written" FAIL
import re, sys
log, rc, dirs = open(sys.argv[1]).read(), int(sys.argv[2]), sys.argv[3].strip()
named = log.find('uniform/waveProperties.<patch> will not be written')
first = re.search(r'^\s*t = ', log, flags=re.M)
# the third step's line prints after runTime.write(), which is where the run stops
steps = len(re.findall(r'^\s*t = ', log, flags=re.M))
stop = log.find('this is a write time, and OpenFOAM would write files brae does not write')
print('      exit %d, named at offset %d, first step at %s, %d step lines, stop at %d, time directories [%s]'
      % (rc, named, first.start() if first else -1, steps, stop, dirs))
sys.exit(0 if rc != 0 and 0 <= named < (first.start() if first else -1) and steps == 2
              and stop > first.start() and not dirs else 1)
PY
    stage "$STK" "$W/stk_ctl" 0.02 timeStep 1000 0 adjustTimeStep=no deltaT=0.01 || exit 1
    subdict "$W/stk_ctl" || say "ARM R  the sub-dictionary was not staged in the control's waveProperties" FAIL
    ( cd "$W/stk_ctl" && "$BIN" -case . > log.brae 2>&1 ) \
        && [ -z "$(timedirs "$W/stk_ctl")" ] && grep -q "will not be written" "$W/stk_ctl/log.brae" \
        && say "CONTROL  the same case with no write time before endTime runs to its end (exit 0)" ok \
        || { say "CONTROL  the same case with no write time before endTime runs to its end (exit 0)" FAIL; tail -3 "$W/stk_ctl/log.brae" | sed 's/^/      /'; }
else
    say "ARM R  stokesI tutorial missing" FAIL
fi


finish "arm R: a file brae cannot write is named at start-up"
