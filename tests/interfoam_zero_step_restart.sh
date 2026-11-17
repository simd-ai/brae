#!/bin/bash
# A `startFrom latestTime` restart whose latest time is already >= endTime takes ZERO steps and must end cleanly:
# the End line, exit 0, and no time directory written or touched. Cluster scripts restart a finished stage just to
# confirm it is at the end. (Regression: the end-of-run download of the interface forces' boundary rho read an
# empty device buffer when the hook had never run, and the run died with a segfault.)
# Also checks the neighbours: a restart that takes one step, and one that takes the rest of the run.
#   tests/interfoam_zero_step_restart.sh <case dir with startFrom latestTime, adjustTimeStep yes> [brae_interFoam]
# a short interFoam case run to a mid-run write time first
set -uo pipefail
CASE=$(cd "${1:?case dir}" && pwd)
BRAE=${2:-$(cd "$(dirname "$0")/.." && pwd)/build/brae_interFoam}
[ -x "$BRAE" ] || { echo "no binary at $BRAE" >&2; exit 2; }
WORK=$(mktemp -d /tmp/zero_step_XXXXXX)
trap 'rm -rf "$WORK"' EXIT
export PATH=/usr/local/cuda/bin:/usr/lib64/openmpi/bin:$PATH LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:${LD_LIBRARY_PATH:-}
fail=0
setEnd() { sed -i "s/^endTime .*/endTime         $2;/; s/^writeInterval .*/writeInterval   $3;/" "$1/system/controlDict"; }
tree() { (cd "$1" && find [0-9]* -type f 2>/dev/null | sort | xargs md5sum | md5sum); }
run() {  # dir label expectEnd
    ( cd "$1" && "$BRAE" -case . -device > "log.$2" 2>&1 ); local rc=$?
    if [ $rc -ne 0 ] || ! grep -aq '^End' "$1/log.$2"; then
        echo "FAIL $2: exit $rc, End line: $(grep -ac '^End' "$1/log.$2")"; tail -n 3 "$1/log.$2"; fail=1; return 1
    fi
    echo "ok   $2: $(grep -a '^End' "$1/log.$2" | cut -c1-100)"
}
cp -r "$CASE/." "$WORK/"; setEnd "$WORK" 0.008 0.004
run "$WORK" first || exit 1
before=$(tree "$WORK")
run "$WORK" zero-step-same-endTime
[ "$(tree "$WORK")" = "$before" ] || { echo "FAIL zero-step rewrote a time directory"; fail=1; }
setEnd "$WORK" 0.004 0.004
run "$WORK" zero-step-endTime-below-start
[ "$(tree "$WORK")" = "$before" ] || { echo "FAIL endTime-below-start rewrote a time directory"; fail=1; }
rm -rf "$WORK/0.008"; setEnd "$WORK" 0.0041 0.004
run "$WORK" one-step-from-0.004
rm -rf "$WORK"/0.004[1-9]* "$WORK/0.008"; setEnd "$WORK" 0.008 0.004
run "$WORK" restart-0.004-to-0.008
[ $fail -eq 0 ] && echo "PASS" || { echo "FAILED"; exit 1; }
