#!/usr/bin/env bash
# The cyclicAMI CUDA path against the _cpp REFERENCE, stage by stage.
#
# Twelve checks, each pairing one device entry point with its host twin at identical inputs, bounded at
# 1e-12 because these are the same arithmetic in two places rather than two discretisations. They pass
# at 0.000e+00 -- bit-identical.
#
# WHY THIS EXISTS AND WHAT IT IS FOR. brae's device AMI is a fused path: until now a disagreement
# anywhere inside it produced one number for the whole interface. pipeCyclic puts 97% of its momentum
# residual on interface cells with the interior exactly zero, and four passes of READING the device code
# failed to explain it. Every other defect in this port fell out quickly once a _cpp reference existed to
# compare against stage by stage -- the pitzDaily inlet diffusivity, the Spalding wall seed,
# turbineSiting's profile origin, the kOmegaSSTLM lambda loop. The AMI simply had no such reference.
#
# WHAT IT ALREADY CAUGHT: the reference's own assembleLaplacian sign, backwards on the first writing
# because it was copied across from assembleMomentum. Momentum carries -laplacian(nuEff,U) and the
# pressure equation +laplacian(rAUf,p), so the two interface assemblies have OPPOSITE signs. The gate
# failed on its first run and named the stage.
#
# WHAT IT RULES OUT, now that all twelve pass bit-identically: the interface interpolation (scalar and
# rotated vector), the face value, both matrix assemblies with their diagonals, the matrix action, the
# interface flux, UEqn.H()'s interface term, and the interface's share of div(phi). Whatever puts
# pipeCyclic's momentum residual on the interface cells, it is not one of these stages computing the
# wrong number from the right inputs.
set -u
SRC="${1:?case dir}"
REF="${2:-2000}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${AMI_CPP_VS_DEVICE:-$ROOT/build/ami_cpp_vs_device}"
[ -x "$BIN" ] || { echo "SKIP: no ami_cpp_vs_device at $BIN"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
SRC="$(cd "$SRC" && pwd)"
[ -d "$SRC/$REF" ] || { echo "SKIP: no state at $SRC/$REF"; exit 77; }

OUT=$("$BIN" "$SRC" "$REF" 2>&1); rc=$?
echo "$OUT" | sed 's/^/  /'
[ $rc -eq 77 ] && exit 77

# THE GEOMETRY CROSS-CHECK, when a conformal case is supplied as $3. Every stage above is a function of
# (weights, deltaCoeffs, Sf), and proving the device reproduces those says nothing about whether they are
# RIGHT -- both sides read the same buildAMIInterfaces. On a CONFORMAL interface an AMI *is* a cyclic, so
# the same mesh with the patches declared cyclic must give the same numbers, and the cyclic geometry is
# covered independently by cyclic_geometry/cyclic_rotational. The cyclic twin is made here by changing
# one word in constant/polyMesh/boundary: the mesh, the points and the faces are untouched.
#
# It runs as a SECOND invocation because the two checks want opposite meshes -- the stencil test above
# needs a non-conforming interface, and only a conforming one has a cyclic twin at all.
CONF="${3:-}"
if [ -n "$CONF" ] && [ -d "$CONF" ]; then
    CONF="$(cd "$CONF" && pwd)"
    W2=$(mktemp -d); trap 'rm -rf "$W2"' EXIT
    cp -r "$CONF" "$W2/cyc"
    sed -i 's/type            cyclicAMI;/type            cyclic;/' "$W2/cyc/constant/polyMesh/boundary"
    grep -q 'type            cyclic;' "$W2/cyc/constant/polyMesh/boundary" \
        || { echo "  FAIL: could not derive a cyclic twin from $CONF"; exit 1; }
    OUT2=$("$BIN" "$CONF" 0 "$W2/cyc" 2>&1); rc2=$?
    echo "$OUT2" | grep -E 'cross-check|AMI vs cyclic' | sed 's/^/  /'
    [ $rc2 -ne 0 ] && [ $rc2 -ne 77 ] && rc=$rc2
fi
exit $rc
