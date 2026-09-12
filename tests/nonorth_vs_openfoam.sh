#!/usr/bin/env bash
# NON-ORTHOGONAL CORRECTION vs REAL OPENFOAM.
#
# The correction is invisible on a near-orthogonal mesh -- on pitzDaily every brae path agrees to 4 digits
# whether or not it is applied, so a gate there would pass with the term deleted. shearedChannel is
# genuinely non-orthogonal AND uses only schemes the rebuilt path implements (`bounded Gauss upwind`,
# `Gauss linear corrected`, laminar, steady), so it isolates this one term.
#
# The oracle is generated HERE by running real simpleFoam, not checked in: the point is agreement with
# OpenFOAM, and a stored reference cannot be re-derived if the case changes.
#
# The control is the whole test. Without the correction the same solver is 8.5e-02 on U; with it, 3.1e-05.
# Asserting only the second number would pass on a mesh where the term does not matter.
#
# That 3.1e-05 was 6.9e-04 until fvMatrix's faceFluxCorrection was ported: the correction was in the
# pressure equation's SOURCE but not in pEqn.flux(), so `phi = phiHbyA - pEqn.flux()` dropped it and phi
# was not conservative. The pressure equation solved perfectly well either way, which is why only a
# comparison against OpenFOAM could see it.
set -u
SRC="${1:?case dir}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIAG="${DIAG_BIN:-$ROOT/build/diag_simple_loop}"
OFBIN=/usr/lib/openfoam/openfoam2412/platforms/linuxARM64GccDPInt32Opt
[ -x "$DIAG" ]            || { echo "SKIP: no diag_simple_loop at $DIAG"; exit 77; }
[ -x "$OFBIN/bin/simpleFoam" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
SRC="$(cd "$SRC" && pwd)"

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
cp -r "$SRC" "$W/of"
# OpenFOAM resolves etc/controlDict through WM_PROJECT_DIR; without it simpleFoam aborts with
# "Could not find mandatory etc entry 'controlDict'" before reading the case at all.
export WM_PROJECT_DIR=/usr/lib/openfoam/openfoam2412
export FOAM_ETC="$WM_PROJECT_DIR/etc"
export PATH="$OFBIN/bin:$PATH"
export LD_LIBRARY_PATH="$OFBIN/lib:$OFBIN/lib/dummy:${LD_LIBRARY_PATH:-}"
( cd "$W/of" && timeout 600 simpleFoam > log.of 2>&1 ) || { echo "FAIL: OpenFOAM did not run"; tail -3 "$W/of/log.of"; exit 1; }
grep -q converged "$W/of/log.of" || { echo "FAIL: OpenFOAM did not converge"; exit 1; }
OFT=$(ls -d "$W/of"/[0-9]* | grep -vE '/0$' | sort -t/ -k99 -n | tail -1)
echo "  ok:   OpenFOAM reference generated -- $(grep converged "$W/of/log.of" | head -1)"

OUT=$("$DIAG" "$SRC" 0 500 "$OFT" 2>/dev/null | tail -4)
echo "$OUT" | sed 's/^/  /'
# The U figure is the field after the literal "U" on each line -- indexed by the marker, not by a column
# number, so a change in the label's wording cannot silently shift which number is read.
CORR=$(echo "$OUT" | awk '/WITH non-orth/{for(i=1;i<=NF;i++) if($i=="U"){print $(i+1); exit}}')
UNCO=$(echo "$OUT" | awk '/WITHOUT the correction/{for(i=1;i<=NF;i++) if($i=="U"){print $(i+1); exit}}')
CUDA=$(echo "$OUT" | awk '/^ *CUDA/{for(i=1;i<=NF;i++) if($i=="U"){print $(i+1); exit}}')
[ -n "$CORR" ] && [ -n "$UNCO" ] && [ -n "$CUDA" ] || { echo "FAIL: could not parse the comparison"; exit 1; }

python3 - "$CORR" "$UNCO" "$CUDA" <<'PY'
import sys
corr, unco, cuda = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
fails = 0
# The correction must bring U close to OpenFOAM. 1e-4 is above the measured 3.1e-05 without being so
# loose that a partially-wrong correction would pass -- the pre-faceFluxCorrection 6.9e-04 would not.
if corr <= 1e-4: print("  ok:   corrected U error %.3e <= 1e-04" % corr)
else:            print("  FAIL: corrected U error %.3e > 1e-04" % corr); fails += 1
# CONTROL: and the mesh must be non-orthogonal enough that omitting it is clearly worse. Without this the
# test would pass on a mesh where the term does nothing -- i.e. it would not be testing the term.
if unco >= 20*corr: print("  ok:   uncorrected is %.0fx worse (%.3e) -- the case discriminates (control)" % (unco/corr, unco))
else:               print("  FAIL: uncorrected only %.1fx worse -- this case cannot test the correction" % (unco/max(corr,1e-30))); fails += 1
# The CUDA path carries the same correction and must land on the reference's side of that gap, not the
# uncorrected one. Bounding it against `unco` rather than against a fixed number keeps the assertion tied
# to the discriminating quantity: a device correction that were dropped or mis-signed lands near `unco`.
if cuda <= 1e-4: print("  ok:   CUDA U error %.3e <= 1e-04" % cuda)
else:            print("  FAIL: CUDA U error %.3e > 1e-04" % cuda); fails += 1
if cuda <= unco/20: print("  ok:   CUDA is %.0fx better than uncorrected -- the device term is live" % (unco/cuda))
else:               print("  FAIL: CUDA %.3e is not clearly better than uncorrected %.3e" % (cuda, unco)); fails += 1
print("PASS" if not fails else "FAIL"); sys.exit(1 if fails else 0)
PY
