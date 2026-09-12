#!/usr/bin/env bash
# fv::actuationDiskSource (Froude) on the MIRRORED _cpp path, against real OpenFOAM.
#
# OpenFOAM's actuationDiskSource WRITES ITS OWN ANSWER every iteration it is applied, to
# postProcessing/<name>/<t>/actuationDiskSource.dat:
#     time   Uref(x y z)   Cp   Ct   a   T   diskDir
# Uref and T between them cover the whole model -- the monitor-cell selection (findCell on the case's
# upstream point), the mean over those cells, the projection onto diskDir, the induction factor a and
# the disk area all feed them. That makes this an unusually direct oracle: no reimplementation of
# OpenFOAM's post-processing, just its own numbers.
#
# THE ONE-ITERATION OFFSET, and why this gate is EXACT rather than approximate. calcFroudeMethod runs
# while UEqn is being assembled at iteration N, so the row written at time N is computed from the field
# that was WRITTEN at time N-1. Comparing brae's answer on the t=N-1 field against OpenFOAM's row at
# time N is therefore a like-for-like comparison of the same arithmetic on the same input, and it agrees
# to 13 digits -- which is why the tolerance below is 1e-10 and not a physics-sized number. Comparing
# against the SAME time's row instead leaves a 0.4-0.9% gap that is one SIMPLE iteration, not an error.
#
# THE ORACLE IS REAL OPENFOAM: the turbineSiting tutorial on its real 120246-cell snappyHexMesh
# (blockMesh + snappyHexMesh + topoSet, as its Allrun does), run to its own convergence.
#
# THE CONTROL: brae must NOT match a row it was not computed from. The t=N row is compared as well, and
# the gate fails if brae is as close to it as to the t=N+1 row -- otherwise a probe that returned a
# constant, or one insensitive to its input field, would pass the tolerance check above by accident.
set -u
SRC="${1:?case dir}"
T0="${2:-100}"                       # the time whose field is read
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${ACTUATIONDISK_PROBE:-$ROOT/build/actuationdisk_probe}"
[ -x "$BIN" ] || { echo "SKIP: no actuationdisk_probe at $BIN"; exit 77; }
SRC="$(cd "$SRC" && pwd)"
[ -d "$SRC/$T0" ]                    || { echo "SKIP: no OpenFOAM state at $SRC/$T0"; exit 77; }
[ -d "$SRC/postProcessing/disk1" ]   || { echo "SKIP: no OpenFOAM actuationDiskSource.dat"; exit 77; }

OUT=$("$BIN" "$SRC" "$T0" 2>&1) || { echo "FAIL: the probe crashed"; echo "$OUT" | tail -5; exit 1; }
echo "$OUT" | sed -n '1,3p'

python3 - "$SRC" "$T0" <<PY
import glob, os, sys
brae = """$OUT"""
src, t0 = sys.argv[1], float(sys.argv[2])

def rows(disk):
    hits = glob.glob(os.path.join(src, "postProcessing", disk, "*", "actuationDiskSource.dat"))
    if not hits: return {}
    out = {}
    for line in open(sorted(hits)[0]):
        if line.lstrip().startswith("#"): continue
        f = line.replace("(", " ").replace(")", " ").split()
        if len(f) < 11: continue
        # time  Uref(x y z)  Cp  Ct  a  T  diskDir(x y z)
        out[float(f[0])] = dict(Uref=[float(f[1]), float(f[2]), float(f[3])],
                                a=float(f[6]), T=float(f[7]))
    return out

def braeVals(i):
    U = T = a = None
    for line in brae.splitlines():
        s = line.split()
        if not s: continue
        if s[0] == "disk%d" % i and s[1] == "Uref":
            U = [float(x) for x in line.split("(")[1].split(")")[0].split()]
        elif s[0] == "disk%d" % i and s[1] == "T":
            T = float(s[2])
        elif s[0] == "disk%d" % i and "a" in s:
            a = float(s[s.index("a") + 1])
    return U, T, a

rc = 0
for i in (1, 2):
    dat = rows("disk%d" % i)
    if not dat:
        print("  FAIL: no OpenFOAM .dat for disk%d" % i); rc = 1; continue
    later = sorted(x for x in dat if x > t0)
    if not later:
        print("  FAIL: the .dat has no row after t=%g for disk%d" % (t0, i)); rc = 1; continue
    tN = later[0]                                  # the row COMPUTED FROM the t0 field
    ref, same = dat[tN], dat.get(t0)
    bU, bT, ba = braeVals(i)
    if bU is None or bT is None:
        print("  FAIL: the probe printed no disk%d values" % i); rc = 1; continue

    for name, b, o, tol in (("Uref.x", bU[0], ref["Uref"][0], 1e-10),
                            ("Uref.y", bU[1], ref["Uref"][1], 1e-10),
                            ("Uref.z", bU[2], ref["Uref"][2], 1e-10),
                            ("T",      bT,    ref["T"],       1e-10),
                            ("a",      ba,    ref["a"],       1e-12)):
        rel = abs(b - o) / max(abs(o), 1e-30)
        ok = rel < tol
        print("  disk%d %-6s brae %18.10f   OpenFOAM %18.10f   rel %.2e   %s"
              % (i, name, b, o, rel, "ok" if ok else "FAIL (>%.0e)" % tol))
        if not ok: rc = 1

    # THE CONTROL. brae read the t0 field, so it must reproduce the row OpenFOAM computed FROM that
    # field (t=tN) and NOT the row written AT t0, which came from the iteration before. A probe that
    # ignored its input -- returning a constant, or the dictionary's own numbers -- would sit equally
    # close to both. Demanding a clear separation is what makes the 1e-10 above mean something.
    if same is not None:
        dN   = abs(bT - ref["T"])  / max(abs(ref["T"]), 1e-30)
        dSame= abs(bT - same["T"]) / max(abs(same["T"]), 1e-30)
        if not (dSame > 1000.0 * max(dN, 1e-16)):
            print("  disk%d control FAIL: brae is no closer to the t=%g row (rel %.2e) than to the "
                  "t=%g row (rel %.2e); the probe may not depend on the field it read"
                  % (i, tN, dN, t0, dSame)); rc = 1
        else:
            print("  disk%d control  the t=%g row is %.0fx further away  ok" % (i, t0, dSame/max(dN,1e-16)))

if rc == 0:
    print("  ok:   Uref, the induction factor and the thrust match OpenFOAM's own written values")
sys.exit(rc)
PY
