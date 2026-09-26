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

# THE SECOND AND THIRD ORACLES: the same case under `uncorrected` and under `orthogonal`.
#
# THREE SCHEMES, TWO FACTS. uncorrectedSnGrad.H:113-119 returns mesh().nonOrthDeltaCoeffs() exactly as
# correctedSnGrad.H:108-114 does and differs only in corrected(); only orthogonalSnGrad.H:113-119 returns
# deltaCoeffs(). brae's SHARED parser (scheme_parse.cuh readSnGrad) tested `hasWord(ln, "corrected")`,
# which is word-boundaried and so does not match `uncorrected` -- so the flag stayed false and brae ran
# ORTHOGONAL under the case's own name, indistinguishable from the case having said `orthogonal`.
#
# THE ARM NEEDS NO NEW ASSERTIONS. In an `uncorrected` arm the harness's existing "OLD host (WITHOUT the
# correction)" column IS the old wrong behaviour -- diag_simple_loop leaves that column's SimpleControls
# at their defaults, so it runs the fully orthogonal form -- which makes the existing 20x control the
# defect's own signature. The `orthogonal` oracle is staged as an INDEPENDENT, brae-free control on top.
for w in uncorrected orthogonal; do
    cp -r "$SRC" "$W/$w"
    python3 - "$W/$w/system/fvSchemes" "$w" <<'SCHEOF' || { echo "FAIL: staging $w"; exit 1; }
import re, sys
q, w = sys.argv[1], sys.argv[2]
t = open(q).read()
t, n = re.subn(r"(laplacianSchemes\s*\{\s*default\s+)Gauss linear corrected", r"\g<1>Gauss linear " + w, t)
assert n == 1, "laplacianSchemes"
t, n = re.subn(r"(snGradSchemes\s*\{\s*default\s+)corrected", r"\g<1>" + w, t)
assert n == 1, "snGradSchemes"
open(q, "w").write(t)
SCHEOF
    grep -qE "default +Gauss linear $w;" "$W/$w/system/fvSchemes"         && grep -qE "default +$w;" "$W/$w/system/fvSchemes"         || { echo "FAIL: $w schemes were not staged"; exit 1; }
    ( cd "$W/$w" && timeout 600 simpleFoam > log.of 2>&1 )         || { echo "FAIL: OpenFOAM did not run [$w]"; tail -3 "$W/$w/log.of"; exit 1; }
    grep -q converged "$W/$w/log.of" || { echo "FAIL: OpenFOAM did not converge [$w]"; exit 1; }
    echo "  ok:   OpenFOAM $w reference generated -- $(grep converged "$W/$w/log.of" | head -1)"
done
UNCOT=$(ls -d "$W/uncorrected"/[0-9]* | grep -vE '/0$' | sort -t/ -k99 -n | tail -1)
ORTHT=$(ls -d "$W/orthogonal"/[0-9]* | grep -vE '/0$' | sort -t/ -k99 -n | tail -1)

# THE BRAE-FREE CONTROL: OpenFOAM's `orthogonal` against its own `uncorrected`. MEASURED on this fixture,
# U 1.1226e-02 and p 1.4040e-01 over all 5000 cells -- every one of the mesh's 9850 internal faces sits at
# exactly 30.9637565321 deg (checkMesh prints max == average, which forces it), so the coefficient ratio
# 1/max(cos a, 0.05) = 1.16619 applies uniformly with nowhere to cancel. One ulp of this case's own chaos
# reaches 3.06e-15, so the signal is 12.6 orders clear of it.
python3 - "$UNCOT" "$ORTHT" <<'CTLEOF' || { echo "FAIL: the scheme control does not discriminate"; exit 1; }
import re, sys
def cells(path, vector):
    t = open(path).read()
    kind = "vector" if vector else "scalar"
    m = re.search(r"internalField\s+nonuniform List<%s>\s*(\d+)\s*\((.*?)\n\)\s*;" % kind, t, re.S)
    if m is None:
        raise SystemExit("FAIL: %s is uniform, so it cannot witness" % path)
    if vector:
        return [tuple(float(x) for x in v.split()) for v in re.findall(r"\(([^()]*)\)", m.group(2))]
    return [(float(x),) for x in m.group(2).split()]
bad = 0
for fld, vec, floor in (("U", True, 1e-4), ("p", False, 1e-3)):
    a = cells(sys.argv[1] + "/" + fld, vec)
    b = cells(sys.argv[2] + "/" + fld, vec)
    assert len(a) == len(b) and a, ("empty or mismatched", fld)
    d = max(max(abs(x - y) for x, y in zip(u, v)) for u, v in zip(a, b))
    ref = max(max(abs(x) for x in u) for u in a) or 1e-300
    n = sum(1 for u, v in zip(a, b) if any(x != y for x, y in zip(u, v)))
    print("  ok:   SCHEME CONTROL: OpenFOAM `orthogonal` vs `uncorrected`, %s %.4e over %d of %d cells "
          "(floor %.0e)" % (fld, d / ref, n, len(a), floor))
    if not d / ref > floor:
        bad = 1
sys.exit(bad)
CTLEOF

# arm <label> <caseDir> <ofTimeDir>: run the four brae columns and hold all four existing assertions
arm()
{
    local label="$1" case="$2" oft="$3"
    echo "  -- arm: $label"
    local OUT
    OUT=$("$DIAG" "$case" 0 500 "$oft" 2>/dev/null | tail -4)
    echo "$OUT" | sed 's/^/  /'
# The U figure is the field after the literal "U" on each line -- indexed by the marker, not by a column
# number, so a change in the label's wording cannot silently shift which number is read.
    local CORR UNCO CUDA
    CORR=$(echo "$OUT" | awk '/WITH non-orth/{for(i=1;i<=NF;i++) if($i=="U"){print $(i+1); exit}}')
    UNCO=$(echo "$OUT" | awk '/WITHOUT the correction/{for(i=1;i<=NF;i++) if($i=="U"){print $(i+1); exit}}')
    CUDA=$(echo "$OUT" | awk '/^ *CUDA/{for(i=1;i<=NF;i++) if($i=="U"){print $(i+1); exit}}')
    [ -n "$CORR" ] && [ -n "$UNCO" ] && [ -n "$CUDA" ] || { echo "FAIL: could not parse the comparison [$label]"; return 1; }

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
}

rc=0
# the shipped case, `Gauss linear corrected`: the correction itself
arm corrected "$SRC" "$OFT" || rc=1
# ...and the case's own `uncorrected`, where the coefficients are the corrected ones and the flux is not.
# brae read that word as `orthogonal`, which is precisely what the OLD-host column computes, so the same
# four assertions hold: `corr` is brae honouring it, `unco` is brae's old answer, and they must differ 20x.
arm uncorrected "$W/uncorrected" "$UNCOT" || rc=1
exit $rc
