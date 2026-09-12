#!/usr/bin/env bash
# `brae -partition` and the warm reload it promises.
#
# -partition builds the mesh and the AMG hierarchy once and caches both into constant/polyMesh
# (.brae_meshcache + .brae_amgcache), printing "Run the solve normally; it will reload them warm."
# THAT RELOAD ALWAYS CRASHED. The cache serialises the agglomeration STRUCTURE and re-Galerkins the
# matrix values, but loadAMGCache did not rebuild the per-level Galerkin GATHER LISTS
# (galCellStart/galCellList/galDFaceStart/galDFaceList/galFaceStart/galFaceList/galFaceFlipList) --
# they are not in the file, and buildGalerkinGather's own comment had said for as long as it existed
# that "the AMG cache stores the agglomeration this is derived from, so a cached hierarchy rebuilds
# these for free". Nothing did. The first Galerkin re-fill after a load then read index 0 of a
# zero-length buffer: "Invalid __global__ read of size 4 bytes / Access to 0x180 is out of bounds" in
# galDiagGatherK, surfacing at the next error check as "amul: an illegal memory access was
# encountered". Nothing was gated on a warm reload, so nothing caught it.
#
# THE ASSERTION IS EQUALITY, not a tolerance: the values are re-Galerkined every step from the same
# matrix, so a warm run must reproduce a cold one exactly. A tolerance would hide a hierarchy that
# loaded but is not the one that was built.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BRAE_BIN:-${BUILD:-$ROOT/build}/brae}"
SRC="${CASE:-$ROOT/validation/backwardFacingStep2D}"
ITERS=${ITERS:-20}

[ -x "$BIN" ] || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC/constant" ] || { echo "SKIP: fixture $SRC missing"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-62s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

cp -r "$SRC/0" "$SRC/constant" "$SRC/system" "$W/"
sed -i "s/^endTime.*/endTime $ITERS;/;s/writeInterval.*/writeInterval $ITERS;/" "$W/system/controlDict"

# ---- 1. COLD: no cache present, run and keep the answer -------------------------------------------
rm -f "$W/constant/polyMesh/.brae_amgcache" "$W/constant/polyMesh/.brae_meshcache"
( cd "$W" && "$BIN" -case "$W" > cold.log 2>&1 ) \
    && say "the cold run completes" ok \
    || { tail -4 "$W/cold.log"; say "the cold run completes" FAIL; }
[ -d "$W/$ITERS" ] || { echo "FAIL: the cold run wrote no $ITERS/"; exit 1; }
cp -r "$W/$ITERS" "$W/cold"; rm -rf "$W/$ITERS"

# ---- 2. -partition writes the caches --------------------------------------------------------------
( cd "$W" && "$BIN" -partition -case "$W" > part.log 2>&1 ) || true
[ -s "$W/constant/polyMesh/.brae_amgcache" ] \
    && say "-partition writes an AMG hierarchy cache" ok \
    || say "-partition writes an AMG hierarchy cache" FAIL

# ---- 3. WARM: the reload -partition promises ------------------------------------------------------
if ( cd "$W" && "$BIN" -case "$W" > warm.log 2>&1 ); then
    say "the warm run reloads the cached hierarchy and completes" ok
else
    grep -q "illegal memory access" "$W/warm.log" \
        && { tail -2 "$W/warm.log"; say "the warm run reloads the cached hierarchy and completes (ILLEGAL ACCESS)" FAIL; } \
        || { tail -4 "$W/warm.log"; say "the warm run reloads the cached hierarchy and completes" FAIL; }
fi

# ---- 4. and it is the SAME answer ------------------------------------------------------------------
COLD="$W/cold" WARM="$W/$ITERS" python3 - <<'PYEOF' || fail=1
import os, re, sys
import numpy as np
def read(p):
    try: s = open(p).read()
    except OSError: return None
    m = re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n?(\d+)\s*\n\(\n(.*?)\n\)\s*;', s, re.S)
    if not m: return None
    if m.group(1) == 'scalar':
        return np.array([float(x) for x in m.group(3).split()])
    return np.array([[float(c) for c in v.split()] for v in re.findall(r'\(([^)]*)\)', m.group(3))])
ok = True
seen = 0
for fld in ('U', 'p', 'k', 'omega', 'epsilon', 'nut'):
    a = read(os.path.join(os.environ['COLD'], fld))
    b = read(os.path.join(os.environ['WARM'], fld))
    if a is None or b is None or a.shape != b.shape:
        continue
    seen += 1
    d = float(np.max(np.abs(a - b)))
    print('     %-8s cold vs warm max|diff| %.3e   %s' % (fld, d, 'ok' if d == 0.0 else 'FAIL'))
    ok = ok and d == 0.0
if seen == 0:
    print('     no field could be compared -- the arm proves nothing   FAIL'); ok = False
sys.exit(0 if ok else 1)
PYEOF
say "the warm run reproduces the cold one bit for bit" "$([ $fail = 0 ] && echo ok || echo FAIL)"

[ $fail = 0 ] && echo PASS || { echo FAIL; exit 1; }
