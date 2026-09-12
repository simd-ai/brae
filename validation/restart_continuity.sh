#!/bin/bash
# A restart must reproduce a continuous run. OF's createPhi.H:45 builds phi with READ_IF_PRESENT paired
# with AUTO_WRITE, so a restart RESUMES the corrected flux. brae's simpleFoam did neither: it rebuilt
# phi from fvc::flux(U) every start, which discards the Rhie-Chow pressure correction a converged SIMPLE
# phi carries.
#
# MEASURED before the fix, on validation/channel, restart-at-30 vs continuous-to-60:
#     U 7.85e-06   p 2.08e-05      (relative)
# against a restart-to-restart reproducibility floor of 9.5e-12 -- six orders above noise. After reading
# and writing phi, both collapsed to that floor.
#
# WHY A GATE. Nothing in the suite covered restart fidelity for simpleFoam, which is exactly why the
# defect shipped: the same fix had already been made for the compressible driver, and this one simply
# never received it. A structural duplicate is only a latent bug until something proves it diverged.
#
# The threshold is the FLOOR, not a tolerance: this comparison is deterministic (same binary, same
# arithmetic), so anything above write precision means state was lost across the restart.
set -u
BRAE=${1:?usage: restart_continuity.sh <brae> <caseDir> <workDir>}
SRC=${2:?}
WORK=${3:?}

[ -x "$BRAE" ] || { echo "SKIP: $BRAE not built"; exit 77; }
[ -d "$SRC" ]  || { echo "SKIP: $SRC absent"; exit 77; }

HALF=30
FULL=60

setup() {   # $1 = dir, $2 = startFrom
    rm -rf "$1"; mkdir -p "$1"; cp -r "$SRC"/* "$1/"
    [ -d "$1/0" ] || { mkdir -p "$1/0"; cp "$1"/0.orig/* "$1/0/" 2>/dev/null; }
    sed -i "s/endTime [0-9.]*;/endTime $FULL;/; s/writeInterval [0-9.]*;/writeInterval $HALF;/; s/startFrom [a-zA-Z]*;/startFrom $2;/" \
        "$1/system/controlDict"
}

# 1. continuous 0 -> FULL, writing at HALF
setup "$WORK.cont" startTime
( cd "$WORK.cont" && "$BRAE" -case . > log 2>&1 ) || { echo "FAIL: continuous run failed"; tail -3 "$WORK.cont/log"; exit 1; }
[ -d "$WORK.cont/$FULL" ] || { echo "FAIL: continuous run wrote no $FULL/"; exit 1; }
[ -f "$WORK.cont/$HALF/phi" ] || {
    echo "FAIL: $HALF/phi was not written. OF pairs READ_IF_PRESENT with AUTO_WRITE (createPhi.H:45);"
    echo "      reading phi without writing it leaves every restart falling back to fvc::flux(U)."
    exit 1; }

# 2. restart from HALF -> FULL, in a copy carrying the intermediate state
rm -rf "$WORK.rest"; cp -r "$WORK.cont" "$WORK.rest"; rm -rf "$WORK.rest/$FULL"
sed -i "s/startFrom [a-zA-Z]*;/startFrom latestTime;/" "$WORK.rest/system/controlDict"
( cd "$WORK.rest" && "$BRAE" -case . > log 2>&1 ) || { echo "FAIL: restart run failed"; tail -3 "$WORK.rest/log"; exit 1; }

python3 - "$WORK.cont/$FULL" "$WORK.rest/$FULL" <<'PY' || exit 1
import re, sys, os
a_dir, b_dir = sys.argv[1], sys.argv[2]
def rd(p):
    s = open(p, errors='replace').read()
    m = re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n(\d+)\s*\n\(', s)
    if not m: return None
    st = m.end(); en = s.index('\n)', st)
    return [float(x) for x in s[st:en].replace('(',' ').replace(')',' ').split()]

FLOOR = 1e-9          # write precision is 12 significant digits; the measured floor was ~1e-11
bad = 0
for f in ('U', 'p'):
    pa, pb = os.path.join(a_dir, f), os.path.join(b_dir, f)
    if not (os.path.isfile(pa) and os.path.isfile(pb)):
        print(f"  FAIL {f}: missing from one of the runs"); bad += 1; continue
    a, b = rd(pa), rd(pb)
    if a is None or b is None or len(a) != len(b):
        print(f"  FAIL {f}: unreadable or size mismatch"); bad += 1; continue
    d = max(abs(x - y) for x, y in zip(a, b))
    s = max(max(map(abs, a)), 1e-30)
    rel = d / s
    print(f"  {f}: restart-vs-continuous rel={rel:.3e}")
    if rel > FLOOR:
        print(f"  FAIL {f}: {rel:.3e} exceeds the write-precision floor {FLOOR:.0e}. The restart did not\n"
              f"       resume the solver state -- most likely phi is being rebuilt from fvc::flux(U)\n"
              f"       instead of read (OF createPhi.H:45 READ_IF_PRESENT + AUTO_WRITE).")
        bad += 1
raise SystemExit(1 if bad else 0)
PY

echo "PASS restart_continuity"
