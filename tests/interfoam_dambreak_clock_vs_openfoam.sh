#!/usr/bin/env bash
# brae's ADAPTIVE CLOCK against real OpenFOAM's, on damBreak exactly as it ships.
#
# WHY THIS GATE EXISTS SEPARATELY. Every other interFoam gate here rewrites the case to
# `adjustTimeStep no` and a fixed deltaT, and for a good reason: two runs at different physical times
# disagree in every field by an amount that looks like a discretisation error and is actually a clock.
# But that rewrite also means NOTHING measured the clock -- and the clock is four pieces of OpenFOAM
# (CourantNo.H, alphaCourantNo.H, setDeltaT.H and Time::adjustDeltaT) that brae has to reproduce for a
# tutorial to run to the time its author asked for.
#
# WHAT IT FOUND. brae read maxCo and maxAlphaCo and ignored writeControl entirely. Time::setDeltaT
# takes `adjust = true` by default and calls Time::adjustDeltaT, which under damBreak's
# `writeControl adjustable; writeInterval 0.05` trims the step to land on the next write time:
# OpenFOAM's first step is 0.05/417 = 0.000119904 where brae's was setDeltaT.H's raw 1.2x = 0.00012.
# Eleven steps of `endTime 0.004` later, OpenFOAM ended at 0.00385757 and brae at 0.00385805.
#
# THE ORACLE IS OpenFOAM'S OWN LOG, and its precision is the gate's precision. At OpenFOAM's default
# six significant figures the comparison bottomed out at 3.2e-06 -- which is half an ULP of the
# PRINTING (0.00119047619 logged as 0.00119048), not a difference between the two runs, and a bound
# set there would have been measuring a printf. controlDict gets `writePrecision 14` below, which
# TimeIO.C:375-389 feeds to IOstream::defaultPrecision and hence to Info, so the log carries the
# number rather than a rounding of it. It changes no arithmetic.
#
# `timePrecision 14` goes with it, and is a SECOND setting: the "Time = " line is timeName(), whose
# precision is timeName's own (TimeIO.C:338), not writePrecision's. With only writePrecision raised,
# deltaT agreed to 2.4e-09 and t was still stuck at the 3.2e-06 of its own printing.
#
# THE CASE IS NOT OTHERWISE REWRITTEN. Its own p_rgh tolerance, its own maxCo, its own write cadence:
# the point is the shipped tutorial.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/brae_interFoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/laminar/damBreak/damBreak"
# ONE FULL WRITE INTERVAL. damBreak writes every 0.05 and Time::adjustDeltaT's whole job is to land on
# that instant exactly, so a run that stops short of it never tests the thing. At 0.05 both solvers
# take thirteen steps and both end ON 0.05; at 0.004 they stop mid-interval, where the last step is
# whatever the loop bound left and the sequences agree only to 3.2e-06.
ENDTIME=${ENDTIME:-0.05}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: damBreak tutorial not found at $SRC"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
cp -r "$SRC" "$W/case" || exit 1
rm -rf "$W"/case/[1-9]* "$W"/case/0 "$W"/case/processor* "$W"/case/log.*

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v blockMesh > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }
command -v interFoam > /dev/null 2>&1 || { echo "SKIP: interFoam not on PATH"; exit 77; }

cp -r "$W/case/0.orig" "$W/case/0"
( cd "$W/case" && blockMesh > log.blockMesh 2>&1 ) || { echo "SKIP: blockMesh failed"; exit 77; }
( cd "$W/case" && setFields > log.setFields 2>&1 ) || { echo "SKIP: setFields failed"; exit 77; }
sed -i "s/^endTime .*/endTime         $ENDTIME;/" "$W/case/system/controlDict"
for kv in "writePrecision 14" "timePrecision 14"; do
    k=${kv% *}; v=${kv#* }
    sed -i "s/^$k .*/$k  $v;/" "$W/case/system/controlDict"
    grep -q "^$k" "$W/case/system/controlDict" || echo "$k  $v;" >> "$W/case/system/controlDict"
done

( cd "$W/case" && interFoam > log.interFoam 2>&1 ) || { echo "SKIP: interFoam failed"; exit 77; }
"$BIN" -case "$W/case" > "$W/log.brae" 2>&1 || { sed -n '$p' "$W/log.brae"; echo "FAIL: brae_interFoam did not run"; exit 1; }

# THE DEVICE BINARY TOO, because `-device` refused adjustTimeStep until this clock was wired into it,
# and a gate that runs only the host would leave the path this port exists for unmeasured. Skipped by
# name, not silently, where there is no GPU.
DEVLOG=""
if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi > /dev/null 2>&1; then
    "$BIN" -case "$W/case" -device > "$W/log.device" 2>&1 \
        || { sed -n '$p' "$W/log.device"; echo "FAIL: brae_interFoam -device did not run"; exit 1; }
    DEVLOG="$W/log.device"
else
    echo "  (no GPU: the -device clock arm is skipped)"
fi

python3 - "$W/case/log.interFoam" "$W/log.brae" "$ENDTIME" "$DEVLOG" <<'PYEOF'
import re, sys

of  = open(sys.argv[1]).read()
bra = open(sys.argv[2]).read()
dev = open(sys.argv[4]).read() if len(sys.argv) > 4 and sys.argv[4] else None

ofDt = [float(x) for x in re.findall(r'^deltaT = (\S+)', of,  flags=re.M)]
ofT  = [float(x) for x in re.findall(r'^Time = (\S+)',   of,  flags=re.M)]
brDt = [float(x) for x in re.findall(r'^\s*t = \S+\s+dt = (\S+)', bra, flags=re.M)]
brT  = [float(x) for x in re.findall(r'^\s*t = (\S+)\s+dt =',     bra, flags=re.M)]

fails = 0
def check(msg, ok):
    global fails
    print(("  ok:   " if ok else "  FAIL: ") + msg)
    if not ok: fails += 1

# The FAIL-PROOF: nothing below can pass on an empty parse.
check("both logs gave a time step sequence to compare (%d OpenFOAM, %d brae)"
      % (len(ofDt), len(brDt)), len(ofDt) > 5 and len(brDt) > 5)
if not ofDt or not brDt:
    print("interfoam_dambreak_clock_vs_openfoam: %d failures" % max(fails, 1)); sys.exit(1)

check("brae took the same NUMBER of steps to reach endTime (%d vs %d)"
      % (len(ofT), len(brT)), len(ofT) == len(brT))

n = min(len(ofDt), len(brDt))
wDt = max(abs(brDt[i] - ofDt[i])/ofDt[i] for i in range(n))
wT  = max(abs(brT[i]  - ofT[i]) /ofT[i]  for i in range(min(len(ofT), len(brT))))
print("  deltaT worst %.3e relative over %d steps;  t worst %.3e;  ends OF %.9g / brae %.9g"
      % (wDt, n, wT, ofT[-1], brT[-1]))

# THE CONTROL. Without Time::adjustDeltaT the first step is setDeltaT.H's raw 1.2 x deltaT0 = 1.2e-04,
# and every check above would still pass to three digits for the first few steps. This one would not:
# OpenFOAM's first step is 0.05/417 and differs from 1.2e-04 in the fourth.
check("the first step is the write cadence's and not setDeltaT.H's raw 1.2x (%.9g, not 1.2e-04)"
      % brDt[0], abs(brDt[0] - 1.2e-4) > 1e-8)

# 1e-08, and what sets it is NOT the clock. deltaT is a function of the Courant number, the Courant
# number is a function of the flux, and brae's flux is not bit-identical to OpenFOAM's -- the
# companion gate interfoam_dambreak_vs_openfoam.sh measures the fields at alpha 1.24e-08 and U
# 9.76e-06 relative. A 2.4e-09 spread in deltaT is that difference arriving here, so this bound moves
# when the FIELDS get closer and not when setDeltaT does. The clock arithmetic itself is held exactly
# by the last check in this file, which is bit-level: both runs land on 0.05.
check("brae's time step tracks OpenFOAM's, step for step", wDt < 1e-8)
check("...so the two clocks stay together", wT < 1e-8)
check("...and land on the same end time", abs(brT[-1] - ofT[-1])/ofT[-1] < 1e-12)

# Landing ON the write time is the whole point of adjustDeltaT, and it is a sharper statement than
# agreeing with OpenFOAM: a brae that reproduced OpenFOAM's arithmetic but drifted would fail here.
end = float(sys.argv[3])
check("...which is the write time itself, hit exactly (%.12g)" % brT[-1],
      abs(brT[-1] - end) < 1e-12*end)

if dev is not None:
    dvDt = [float(x) for x in re.findall(r'^\s*t = \S+\s+dt = (\S+)', dev, flags=re.M)]
    dvT  = [float(x) for x in re.findall(r'^\s*t = (\S+)\s+dt =',     dev, flags=re.M)]
    check("the device binary gave a time step sequence too (%d)" % len(dvDt), len(dvDt) > 5)
    if dvDt:
        m = min(len(ofDt), len(dvDt))
        vDt = max(abs(dvDt[i] - ofDt[i])/ofDt[i] for i in range(m))
        vT  = max(abs(dvT[i]  - ofT[i]) /ofT[i]  for i in range(min(len(ofT), len(dvT))))
        print("  DEVICE: deltaT worst %.3e relative over %d steps;  t worst %.3e;  ends %.12g"
              % (vDt, m, vT, dvT[-1]))
        check("-device took the same number of steps (%d vs %d)" % (len(dvT), len(ofT)),
              len(dvT) == len(ofT))
        check("-device's time step tracks OpenFOAM's, step for step", vDt < 1e-8)
        check("...and lands on the write time exactly", abs(dvT[-1] - end) < 1e-12*end)

print("interfoam_dambreak_clock_vs_openfoam: %d failures" % fails)
sys.exit(1 if fails else 0)
PYEOF
