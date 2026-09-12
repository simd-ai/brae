#!/usr/bin/env bash
# fv::rotorDiskSource on the MIRRORED _cpp path, against real OpenFOAM.
#
# OpenFOAM's rotorDiskSource PRINTS its own answer every iteration it is applied:
#     min/max(AOA), Effective drag = sum(rhoRef*localForce.y), Effective lift = sum(rhoRef*localForce.z)
# summed over the disk cells, with localForce in the cylindrical (e1, e2, e3) frame BEFORE the rotation
# back to Cartesian. Those three numbers are an unusually direct oracle: the coordinate frame, the blade
# and profile table interpolation, the tip factor, the blade-relative velocity and the dynamic pressure
# all feed them, so agreeing on all three is agreement on the whole blade-element calculation. Computing
# them from OpenFOAM's OWN converged U makes it a statement about the model and nothing else.
#
# THE ORACLE IS REAL OPENFOAM: validation/rotorDisk is the tutorial on its real 71734-cell snappyHexMesh
# (blockMesh + surfaceFeatureExtract + snappyHexMesh, as its Allrun does), run to its own convergence at
# t=224, gzipped and trimmed to what this gate reads.
#
# THE SIGN this pins down: OpenFOAM's addSup is `eqn -= force` with force carrying eqn's dimensions PER
# VOLUME, and fvMatrix::operator-= is `source() += V*su` -- so the extensive source GAINS the raw force.
# brae's pre-existing device rotorDisk carries a comment describing the opposite (`relaxSrc -= force`) and
# has no test at all, so nothing had ever checked which way the thrust points. A body force that pushes
# the wrong way still converges; it just converges to the wrong answer.
#
# It also checks the DEVICE force against the host one, which is what settled that question: they agree
# to 5.6e-16, so the device force was right all along and the comment describes the solver's application
# rather than the force. The V2 driver applies it as OpenFOAM does, source += the raw force.
#
# The residual difference below is the one-iteration offset between OpenFOAM's printed value (start of
# iteration 224, from the t=223 velocity) and the written t=224 field, not a discretisation difference.
set -u
SRC="${1:?case dir}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${ROTORDISK_PROBE:-$ROOT/build/rotordisk_probe}"
[ -x "$BIN" ] || { echo "SKIP: no rotordisk_probe at $BIN"; exit 77; }
SRC="$(cd "$SRC" && pwd)"
[ -d "$SRC/224" ] || { echo "SKIP: no OpenFOAM state at $SRC/224"; exit 77; }
[ -f "$SRC/log.simpleFoam" ] || { echo "SKIP: no OpenFOAM log at $SRC/log.simpleFoam"; exit 77; }

OUT=$("$BIN" "$SRC" 224 2>&1) || { echo "FAIL: the probe crashed"; echo "$OUT" | tail -5; exit 1; }
echo "$OUT" | sed -n '1,3p'

python3 - "$SRC/log.simpleFoam" <<PY
import re, sys
brae = """$OUT"""
# OpenFOAM's LAST rotorDisk block in the log -- the converged one.
txt = open(sys.argv[1]).read()
blocks = re.findall(
    r'rotorDisk output:\s*\n\s*min/max\(AOA\)\s*=\s*([-0-9.eE+]+),\s*([-0-9.eE+]+)\s*\n'
    r'\s*Effective drag\s*=\s*([-0-9.eE+]+)\s*\n\s*Effective lift\s*=\s*([-0-9.eE+]+)', txt)
if not blocks:
    print("  FAIL: no rotorDisk output block in the OpenFOAM log"); sys.exit(1)
ofa0, ofa1, ofd, ofl = (float(x) for x in blocks[-1])

def grab(pat):
    m = re.search(pat, brae)
    return float(m.group(1)) if m else float('nan')
ba0 = grab(r'min/max\(AOA\)\s*=\s*([-0-9.eE+]+),')
ba1 = grab(r'min/max\(AOA\)\s*=\s*[-0-9.eE+]+,\s*([-0-9.eE+]+)')
bd  = grab(r'Effective drag\s*=\s*([-0-9.eE+]+)')
bl  = grab(r'Effective lift\s*=\s*([-0-9.eE+]+)')

rc = 0
for name, b, o, tol in (("min AOA", ba0, ofa0, 1e-3), ("max AOA", ba1, ofa1, 1e-3),
                        ("drag",    bd,  ofd,  1e-3), ("lift",    bl,  ofl,  1e-3)):
    rel = abs(b - o) / max(abs(o), 1e-30)
    ok = rel < tol
    print("  %-8s brae %14.6f   OpenFOAM %14.6f   rel %.2e   %s"
          % (name, b, o, rel, "ok" if ok else "FAIL (>%.0e)" % tol))
    if not ok: rc = 1

# THE CONTROL: the thrust must point the way OpenFOAM says. Lift is the rotor's whole purpose, and a
# sign error there is the failure mode the ungated device implementation could have carried.
if bl * ofl <= 0:
    print("  FAIL: the lift has the opposite sign to OpenFOAM's"); rc = 1

# ...and the two brae paths must agree, or the CUDA driver is not running what was just validated.
md = re.search(r'device force vs _cpp: L_inf rel ([0-9.eE+-]+)', brae)
if not md:
    print("  FAIL: the probe did not report the device comparison"); rc = 1
else:
    d = float(md.group(1))
    ok = d < 1e-12
    print("  %-8s device vs _cpp force  L_inf rel %.2e   %s" % ("paths", d, "ok" if ok else "FAIL"))
    if not ok: rc = 1
if rc == 0:
    print("  ok:   the blade-element force matches OpenFOAM's own reported drag, lift and AOA range")
sys.exit(rc)
PY
