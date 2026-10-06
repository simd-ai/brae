#!/usr/bin/env bash
# tools/cache_key_audit.py, and the proof that it fails when the guard is pointer-only.
#
# It reports nothing over src/ -- which is the right answer and also what a broken audit looks like --
# so this cuts the surviving identity out of two real guards and checks each is named. Nothing in the
# tree is modified: the files are copied to a scratch directory and edited there.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AUDIT="$ROOT/tools/cache_key_audit.py"
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
fails=0

echo "== tools/cache_key_audit.py: the tree, and the defect it exists for =="

if python3 "$AUDIT" "$ROOT/src" > "$W/tree.log" 2>&1; then
    echo "  ok:   every cache guard over a device pointer also compares a surviving identity"
else
    echo "  FAIL: one does not -- read it, this gate is not the place to silence it"
    sed 's/^/        /' "$W/tree.log"
    fails=$((fails+1))
fi
n=$(sed -n 's/.*files, \([0-9]*\) cache guards.*/\1/p' "$W/tree.log")
if [ "${n:-0}" -ge 8 ]; then
    echo "  ok:   it found $n such guards to ask about"
else
    echo "  FAIL: only ${n:-0} guards found -- the parser is not reading the tree"
    fails=$((fails+1))
fi

# (a) the AMG PCG's V-cycle graph, with the addressing and the epoch cut out of its guard
mkdir -p "$W/pcg"
cp "$ROOT/src/matrices/lduMatrix/solvers/PCG/device_amg_pcg.cu" "$W/pcg/"
if python3 "$AUDIT" "$W/pcg" > "$W/pcg.clean.log" 2>&1; then
    echo "  ok:   ...and that file is clean on its own, before the edit"
else
    echo "  FAIL: the unmodified copy already reports"; fails=$((fails+1))
fi
sed -i 's/ || gcf.keyEpoch != deviceReductionScratchEpoch()//; s/ || gc.keyEpoch != deviceReductionScratchEpoch()//' \
    "$W/pcg/device_amg_pcg.cu"
# ...and the pair's epoch, which these guards have compared since 2026-10-04 (f4744e6). It survives pool
# recycling too, so with it left in the guard was NOT keyed on the pointer alone, the audit rightly passed
# the edited copy, and this arm failed from that commit on. Cut with the rest, and checked to have gone.
# ...and amgGraphViewMoved, which they ask since 2026-10-06 (the off-diagonal pointers and the addressing).
sed -i 's/^\s*|| gcf.keyPairEpoch != amg.pair.epoch || amgGraphViewMoved(gcf, A))/)/' "$W/pcg/device_amg_pcg.cu"
sed -i 's/^\s*|| gc.keyPairEpoch != amg.pair.epoch || amgGraphViewMoved(gc, A))/)/' "$W/pcg/device_amg_pcg.cu"
left="gcf\?.keyPairEpoch != amg.pair.epoch\|gcf\?.keyEpoch != deviceReductionScratchEpoch\|amgGraphViewMoved"
grep -q "$left" "$W/pcg/device_amg_pcg.cu" \
    && { echo "  FAIL: the PCG injection did not take"; fails=$((fails+1)); }
if python3 "$AUDIT" "$W/pcg" > "$W/pcg.log" 2>&1; then
    echo "  FAIL: a graph keyed on A.diag alone passed"
    fails=$((fails+1))
elif grep -q "keyed on .diag. alone" "$W/pcg.log"; then
    echo "  ok:   a V-cycle graph keyed on A.diag alone is named"
else
    echo "  FAIL: it failed, but not for the pointer -- read it"; sed 's/^/        /' "$W/pcg.log"; fails=$((fails+1))
fi

# (b) THE STATE THIS AUDIT FOUND: the AMG V-cycle's own graph caches, before the epoch and the
# addressing were added to their guards. That is the historical defect, restored by two substitutions
# on the file that holds it -- not an injection invented for the test.
mkdir -p "$W/vcycle"
cp "$ROOT/src/matrices/lduMatrix/preconditioners/GAMGPreconditioner/device_amg_vcycle.cu" "$W/vcycle/"
sed -i 's/ || gcf.keyEpoch != deviceReductionScratchEpoch()//; s/ || gc.keyEpoch != deviceReductionScratchEpoch()//' \
    "$W/vcycle/device_amg_vcycle.cu"
# (the pair's epoch shares the addressing's line since 2026-10-04 and goes with it: see (a))
sed -i 's/^\s*|| gcf.keyAddressingId != A.addressingId || gcf.keyPairEpoch != amg.pair.epoch$//' \
    "$W/vcycle/device_amg_vcycle.cu"
sed -i 's/^\s*|| gc.keyAddressingId != A.addressingId || gc.keyPairEpoch != amg.pair.epoch$//' \
    "$W/vcycle/device_amg_vcycle.cu"
# (and the line after each, which asks amgGraphViewMoved since 2026-10-06 and closes the condition)
sed -i 's/^\s*|| amgGraphViewMoved(gcf\?, A))/)/' "$W/vcycle/device_amg_vcycle.cu"
left="keyAddressingId != A.addressingId\|gcf\?.keyPairEpoch != amg.pair.epoch\|amgGraphViewMoved"
grep -q "$left" "$W/vcycle/device_amg_vcycle.cu" \
    && { echo "  FAIL: the V-cycle injection did not take"; fails=$((fails+1)); }
if python3 "$AUDIT" "$W/vcycle" > "$W/vcycle.log" 2>&1; then
    echo "  FAIL: the V-cycle graphs keyed on A.diag alone passed"
    fails=$((fails+1))
elif [ "$(grep -c 'keyed on .diag. alone' "$W/vcycle.log")" = 2 ]; then
    echo "  ok:   both V-cycle graphs keyed on A.diag alone are named"
else
    echo "  FAIL: it failed, but not for both -- read it"; sed 's/^/        /' "$W/vcycle.log"; fails=$((fails+1))
fi

# (c) what must NOT be reported: a symmetry check on one matrix, and a guard that REFUSES
mkdir -p "$W/notacache"
cp "$ROOT/src/matrices/lduMatrix/solvers/GAMG/gamg_solver_cpp.cu" \
   "$ROOT/src/matrices/lduMatrix/smoothers/GaussSeidel/device_colour_gauss_seidel.cu" "$W/notacache/"
if python3 "$AUDIT" "$W/notacache" > "$W/notacache.log" 2>&1; then
    echo '  ok:   a symmetry check (M.upper != M.lower) and a refusal are not read as caches'
else
    echo "  FAIL: a symmetry check or a refusal was reported as a stale-cache risk"
    sed 's/^/        /' "$W/notacache.log"
    fails=$((fails+1))
fi

echo "cache_key_audit_selftest: $fails failures"
[ "$fails" = 0 ]
