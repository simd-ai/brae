#!/usr/bin/env bash
# tools/resize_audit.py, and the proof that it fails when the defect is there.
#
# The audit reports nothing over src/ -- which is the right answer and also what a broken audit looks
# like, so this injects the defect it exists for and checks it is named. Both shapes:
#
#   SAME FUNCTION  resize a buffer, accumulate into it, never zero it. The historical one:
#                  interFoam's vector limitedLinear built its magSqr limiter in a recycled pool block
#                  and read U 5.1e-01 of OpenFOAM's (2026-09-22, #28).
#   CROSS-FUNCTION the resize is in the caller and the accumulation is in a shared helper, so neither
#                  file shows both. deviceAxpy over a residual the caller forgot to fill.
#
# Nothing in the tree is modified: each file is copied to a scratch directory, the defect is cut into
# the copy with `sed`, and the audit is run over the copy. The unmodified copies are run first, in the
# same directory, so a hit cannot be an artefact of the isolation.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AUDIT="$ROOT/tools/resize_audit.py"
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
fails=0

echo "== tools/resize_audit.py: the tree, and the defect it exists for =="

# (a) the tree itself
if python3 "$AUDIT" "$ROOT/src" > "$W/tree.log" 2>&1; then
    echo "  ok:   src/ has no accumulation into an unzeroed buffer"
else
    echo "  FAIL: src/ reports one -- read it, this gate is not the place to silence it"
    sed 's/^/        /' "$W/tree.log"
    fails=$((fails+1))
fi
grep -q "accumulating kernels" "$W/tree.log" || { echo "  FAIL: the audit printed no kernel count"; fails=$((fails+1)); }
n=$(sed -n 's/.*files, \([0-9]*\) accumulating kernels.*/\1/p' "$W/tree.log")
if [ "${n:-0}" -ge 50 ]; then
    echo "  ok:   it found $n accumulating kernels to ask about"
else
    echo "  FAIL: only ${n:-0} accumulating kernels found -- the parser is not reading the tree"
    fails=$((fails+1))
fi

# (b) the two files this test injects into, UNMODIFIED, in isolation
mkdir -p "$W/clean"
cp "$ROOT/src/applications/solvers/interFoam/device_inter_peqn.cu" "$W/clean/"
cp "$ROOT/src/matrices/lduMatrix/preconditioners/GAMGPreconditioner/device_amg_gauss_seidel.cu" "$W/clean/"
cp "$ROOT/src/matrices/lduMatrix/lduMatrix/blas1.cu" "$W/clean/"
if python3 "$AUDIT" "$W/clean" > "$W/clean.log" 2>&1; then
    echo "  ok:   ...and the three files this test edits are clean on their own"
else
    echo "  FAIL: the unmodified copies already report -- a later hit would prove nothing"
    sed 's/^/        /' "$W/clean.log"
    fails=$((fails+1))
fi

# (c) SAME FUNCTION: cut the memset that stands between a resize and an accumulation
mkdir -p "$W/same"
cp "$W/clean/device_inter_peqn.cu" "$W/same/"
sed -i '/cudaMemset(P.source.data(), 0, sizeof(scalar)\*nC);/d' "$W/same/device_inter_peqn.cu"
grep -q "cudaMemset(P.source.data(), 0, sizeof(scalar)\*nC);" "$W/same/device_inter_peqn.cu" \
    && { echo "  FAIL: the injection did not take -- the memset is still there"; fails=$((fails+1)); }
if python3 "$AUDIT" "$W/same" > "$W/same.log" 2>&1; then
    echo "  FAIL: the audit passed a resized-and-accumulated P.source"
    fails=$((fails+1))
else
    if grep -q "accumulates into .source. (P.source)" "$W/same.log"; then
        echo "  ok:   the same-function defect is named at $(grep -o 'device_inter_peqn.cu:[0-9]*' "$W/same.log" | head -1)"
    else
        echo "  FAIL: it failed, but not for P.source -- read it"
        sed 's/^/        /' "$W/same.log"
        fails=$((fails+1))
    fi
fi

# (d) CROSS-FUNCTION: cut the deviceCopy that fills the residual the helper accumulates into
mkdir -p "$W/cross"
cp "$W/clean/device_amg_gauss_seidel.cu" "$W/clean/blas1.cu" "$W/cross/"
sed -i '/deviceCopy(c.r\[k\], \*comps\[k\].b);/d' "$W/cross/device_amg_gauss_seidel.cu"
# the file has four deviceCopy(c.r[k], ...) lines; only the two this pattern names are cut
grep -q "deviceCopy(c.r\[k\], \*comps\[k\].b);" "$W/cross/device_amg_gauss_seidel.cu" \
    && { echo "  FAIL: the cross-function injection did not take"; fails=$((fails+1)); }
if python3 "$AUDIT" "$W/cross" > "$W/cross.log" 2>&1; then
    echo "  FAIL: the audit passed a caller that resized c.r[k] and never filled it"
    fails=$((fails+1))
else
    if grep -q "deviceAxpy(c.r\[k\])" "$W/cross.log"; then
        echo "  ok:   the cross-function defect is named: deviceAxpy(c.r[k]), accumulated in blas1.cu"
    else
        echo "  FAIL: it failed, but not for c.r[k] -- read it"
        sed 's/^/        /' "$W/cross.log"
        fails=$((fails+1))
    fi
fi

# (e) a comment must not count, in EITHER direction: the audit's first run reported a comment as a
# defect, and a zeroer named only in a comment would have hidden one
mkdir -p "$W/comment"
cp "$W/same/device_inter_peqn.cu" "$W/comment/"
sed -i 's|^    P.source.resize(static_cast<std::size_t>(nC));|    // cudaMemset(P.source.data(), 0, sizeof(scalar)*nC);\n    P.source.resize(static_cast<std::size_t>(nC));|' \
    "$W/comment/device_inter_peqn.cu"
grep -q "// cudaMemset(P.source.data()" "$W/comment/device_inter_peqn.cu" || {
    echo "  FAIL: the comment injection did not take"; fails=$((fails+1)); }
if python3 "$AUDIT" "$W/comment" > "$W/comment.log" 2>&1; then
    echo "  FAIL: a memset written in a COMMENT counted as zeroing the buffer"
    fails=$((fails+1))
else
    echo "  ok:   a memset in a comment does not count -- the defect is still named"
fi

echo "resize_audit_selftest: $fails failures"
[ "$fails" = 0 ]
