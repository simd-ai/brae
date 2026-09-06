#!/usr/bin/env bash
# turbineSiting end to end through the CUDA V2 driver, against real OpenFOAM.
#
# This is the gate that tests what tests/actuationdisk_vs_openfoam.sh CANNOT: not the thrust, but the
# SIGN IT IS APPLIED WITH. The .dat oracle compares numbers the model produces; only running the case
# shows which way the turbines push. The rotorDiskSource gate had exactly that blind spot -- it matched
# OpenFOAM's reported drag and lift to 2e-05 while the driver applied the force with the wrong sign,
# because both are computed from OpenFOAM's own converged velocity and neither depends on the sign.
#
# THE SIGN, from OpenFOAM's source and not from the physics one expects:
#   calcFroudeMethod:  Usource[celli] += ((V[celli]/V())*T)*diskDir     -> the OPTION matrix's source
#   UEqn.H:            fvm::div(phi,U) + ... == fvOptions(U)
#   fvMatrix.C:        operator==(A, B) is  A - B
# so the MOMENTUM source LOSES the thrust. With the plus instead, this case still converges -- to a wind
# field about 15% too fast at the turbines, which is the failure mode this gate exists to catch.
#
# THE ORACLE IS REAL OPENFOAM: validation/turbineSiting is the tutorial on its real 120246-cell
# snappyHexMesh (blockMesh + snappyHexMesh + topoSet, as its Allrun does), run by OpenFOAM v2412 to
# DEEP convergence (residualControl 1e-9, 600 iterations).
#
# WHY DEEP CONVERGENCE. The tutorial's own residualControl (p 1e-3) stops OpenFOAM at 105 iterations,
# and that state is nowhere near steady: the monitored velocity at disk 2 is 12.9 there and 16.8 at the
# real fixed point, a 30% drift AFTER the case declared itself converged. Comparing two solvers at their
# own early stopping points compares two arbitrary points on two transients. Both are run to 1e-9 here
# so the comparison is between fixed points.
#
# THE CONTROL: the same run with the turbines switched off must be far worse. Without it, a bound of
# 3e-03 on U says nothing -- the turbines might be contributing almost nothing to the answer.
set -u
SRC="${1:?case dir}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
PROBE="${ACTUATIONDISK_PROBE:-$ROOT/build/actuationdisk_probe}"
[ -x "$BRAE" ]  || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -x "$PROBE" ] || { echo "SKIP: no actuationdisk_probe at $PROBE"; exit 77; }
SRC="$(cd "$SRC" && pwd)"
[ -d "$SRC/600" ] || { echo "SKIP: no OpenFOAM reference at $SRC/600"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }

W="$(mktemp -d)"
trap 'rm -rf "$W"' EXIT

prep() {   # prep <dir> <turbines on|off>
    mkdir -p "$1"
    cp -r "$SRC/constant" "$SRC/system" "$SRC/0.orig" "$1/"
    cp -r "$1/0.orig" "$1/0"
    python3 - "$1/system/fvSolution" <<'PY'
import io, re, sys
p = sys.argv[1]
s = io.open(p).read()
s = re.sub(r'residualControl\s*\{[^}]*\}',
           'residualControl\n    {\n        p               1e-9;\n'
           '        U               1e-9;\n        "(k|epsilon)"   1e-9;\n    }', s, flags=re.S)
io.open(p, 'w').write(s)
PY
    sed -i -e 's/^endTime.*/endTime         600;/' -e 's/^writeInterval.*/writeInterval   600;/' \
           "$1/system/controlDict"
    if [ "$2" = "off" ]; then
        # The turbines removed entirely, not deactivated: brae refuses a source it recognises but cannot
        # apply, and `active false` is a path of its own. Removing the file is the cleanest "no turbines".
        rm -f "$1/constant/fvOptions" "$1/system/fvOptions"
    fi
}

prep "$W/on"  on
prep "$W/off" off
BRAE_SIMPLEFOAM_V2=1 "$BRAE" "$W/on"  > "$W/on.log"  2>&1 || { echo "FAIL: brae crashed"; tail -15 "$W/on.log"; exit 1; }
BRAE_SIMPLEFOAM_V2=1 "$BRAE" "$W/off" > "$W/off.log" 2>&1 || { echo "FAIL: the control run crashed"; tail -15 "$W/off.log"; exit 1; }
grep -E 'fvOptions actuationDisk' "$W/on.log" | sed 's/^/  /'
[ -d "$W/on/600" ] || { echo "FAIL: brae wrote no 600 directory"; tail -5 "$W/on.log"; exit 1; }

PROBE_OUT=$("$PROBE" "$W/on" 600 2>&1) || { echo "FAIL: the probe crashed on brae's own field"; exit 1; }

python3 - "$W/on/600" "$W/off/600" "$SRC/600" "$SRC" <<PY
import glob, os, re, sys
import numpy as np
probe = """$PROBE_OUT"""
on, off, ref, src = sys.argv[1:5]

def read(fn):
    for cand in (fn, fn + ".gz"):
        if os.path.exists(cand):
            b = (__import__("gzip").open(cand, "rb") if cand.endswith(".gz") else open(cand, "rb")).read()
            break
    else:
        return None
    m = re.search(rb'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n(\d+)\s*\n\(', b)
    if not m: return None
    typ = m.group(1).decode(); n = int(m.group(2)); start = m.end()
    nc = 3 if typ == 'vector' else 1
    fm = re.search(r'format\s+(\w+)', b[:1024].decode('latin-1'))
    if fm and fm.group(1) == 'binary':
        return np.frombuffer(b[start:start+n*nc*8], dtype='<f8').reshape(n, nc)
    txt = b[start:].decode('latin-1')
    vals = re.findall(r'[-+0-9.eE]+', txt.split(')\n;')[0] if ')\n;' in txt else txt)
    return np.array([float(x) for x in vals[:n*nc]]).reshape(n, nc)

# Set just above where the CUDA path lands. U and p are the solved fields; k, epsilon and nut are the
# transported/derived turbulence quantities and sit an order of magnitude looser here as they do in
# every other brae end-to-end gate.
BOUND = {'U': 3e-03, 'p': 5e-03, 'k': 3e-02, 'epsilon': 6e-02, 'nut': 3e-02}
rc, errOn = 0, {}
for f in ('U', 'p', 'k', 'epsilon', 'nut'):
    a, b = read(os.path.join(on, f)), read(os.path.join(ref, f))
    if a is None or b is None:
        print("  FAIL: could not read %s" % f); rc = 1; continue
    e = float(np.linalg.norm(a - b) / np.linalg.norm(b))
    errOn[f] = e
    ok = e < BOUND[f]
    print("  %-8s L2 rel %.3e   bound %.1e   %s" % (f, e, BOUND[f], "ok" if ok else "FAIL"))
    if not ok: rc = 1

# The converged OPERATING POINT: the thrust brae settles on, against the row OpenFOAM computed from its
# own converged field. This is the number a turbine siting study is actually for.
def datRow(disk):
    hits = glob.glob(os.path.join(src, "postProcessing", disk, "*", "actuationDiskSource.dat"))
    if not hits: return None
    last = None
    for line in open(sorted(hits)[0]):
        if line.lstrip().startswith("#"): continue
        g = line.replace("(", " ").replace(")", " ").split()
        if len(g) >= 11: last = g
    return None if last is None else dict(Uref=float(last[1]), T=float(last[7]))

for i in (1, 2):
    r = datRow("disk%d" % i)
    if r is None:
        print("  FAIL: no OpenFOAM .dat for disk%d" % i); rc = 1; continue
    mT = re.search(r'disk%d\s+T\s+([-0-9.eE+]+)' % i, probe)
    mU = re.search(r'disk%d\s+Uref\s+\(\s*([-0-9.eE+]+)' % i, probe)
    if not mT or not mU:
        print("  FAIL: the probe printed no disk%d values" % i); rc = 1; continue
    for name, b, o in (("Uref.x", float(mU.group(1)), r["Uref"]), ("thrust", float(mT.group(1)), r["T"])):
        rel = abs(b - o) / max(abs(o), 1e-30)
        ok = rel < 2e-03
        print("  disk%d %-6s brae %14.6f   OpenFOAM %14.6f   rel %.2e   %s"
              % (i, name, b, o, rel, "ok" if ok else "FAIL (>2e-03)"))
        if not ok: rc = 1

# THE CONTROL. Same solver, same mesh, same 600 iterations, no turbines. If that is not far worse then
# the bounds above are not measuring the actuator disk at all.
a, b = read(os.path.join(off, 'U')), read(os.path.join(ref, 'U'))
if a is None or b is None:
    print("  FAIL: could not read the control U"); rc = 1
else:
    eOff = float(np.linalg.norm(a - b) / np.linalg.norm(b))
    ratio = eOff / max(errOn.get('U', 1e-30), 1e-30)
    ok = ratio >= 15.0
    print("  control  no turbines: U L2 rel %.3e   %.0fx the gated error   %s"
          % (eOff, ratio, "ok" if ok else "FAIL (want >=15x)"))
    if not ok: rc = 1

if rc == 0:
    print("  ok:   turbineSiting runs end to end on the CUDA V2 driver and matches real OpenFOAM")
sys.exit(rc)
PY
