#!/usr/bin/env bash
# tools/header_odr_audit.py, and the proof that it names the definition it exists for.
#
# It reports nothing over src/ -- which is the right answer and also what a broken parser looks like --
# so this restores the TWO REAL defects it was written for, in scratch copies, and checks each is
# named: readSurfaceField before it was made inline, and device_scalar_transport.cuh's kernels before
# they were made static. It then checks what must NOT be reported, because a parser that reports a
# declaration or a default argument would be useless long before it was wrong.
# Nothing in the tree is modified.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AUDIT="$ROOT/tools/header_odr_audit.py"
ALLOW="$ROOT/tools/header_odr_audit_allow.txt"
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
fails=0

echo "== tools/header_odr_audit.py: the tree, and the defect it exists for =="

if python3 "$AUDIT" --allow "$ALLOW" "$ROOT/src" > "$W/tree.log" 2>&1; then
    echo "  ok:   no .cuh defines a function with external linkage"
else
    echo "  FAIL: one does -- read it, this gate is not the place to silence it"
    sed 's/^/        /' "$W/tree.log"
    fails=$((fails+1))
fi
n=$(sed -n 's/.*: \([0-9]*\) headers.*/\1/p' "$W/tree.log")
if [ "${n:-0}" -ge 200 ]; then
    echo "  ok:   it read $n headers to ask about"
else
    echo "  FAIL: only ${n:-0} headers read -- the walk is not reaching the tree"
    fails=$((fails+1))
fi

# (a) THE DEFECT: readSurfaceField without `inline`. Six TUs include that header and two reach the
# same binary, so each emitted a strong symbol and the link failed on the pair.
mkdir -p "$W/a"
cp "$ROOT/src/applications/solvers/common/read_surface_field.cuh" "$W/a/"
if python3 "$AUDIT" "$W/a" > "$W/a.clean.log" 2>&1; then
    echo "  ok:   ...and that header is clean on its own, before the edit"
else
    echo "  FAIL: the unmodified copy already reports"; fails=$((fails+1))
fi
sed -i 's/^inline SurfaceScalarField readSurfaceField(/SurfaceScalarField readSurfaceField(/' "$W/a/read_surface_field.cuh"
grep -q '^SurfaceScalarField readSurfaceField(' "$W/a/read_surface_field.cuh" \
    || { echo "  FAIL: the readSurfaceField injection did not take"; fails=$((fails+1)); }
if python3 "$AUDIT" "$W/a" > "$W/a.log" 2>&1; then
    echo "  FAIL: a non-inline readSurfaceField passed"
    fails=$((fails+1))
elif grep -q "\`readSurfaceField\` is defined here with external linkage" "$W/a.log"; then
    echo "  ok:   a non-inline readSurfaceField is named"
else
    echo "  FAIL: it failed, but not for that -- read it"; sed 's/^/        /' "$W/a.log"; fails=$((fails+1))
fi

# (b) THE SECOND DEFECT: the four __global__ kernels this header defines, before they were static.
# Twenty-four objects define each of them strongly.
mkdir -p "$W/b"
cp "$ROOT/src/cuda/device_scalar_transport.cuh" "$W/b/"
sed -i 's/^static __global__$/__global__/' "$W/b/device_scalar_transport.cuh"
if python3 "$AUDIT" "$W/b" > "$W/b.log" 2>&1; then
    echo "  FAIL: header __global__ kernels with external linkage passed"
    fails=$((fails+1))
else
    k=$(grep -c "is defined here with external linkage" "$W/b.log")
    if [ "$k" -ge 4 ]; then
        echo "  ok:   $k header kernels with external linkage are named"
    else
        echo "  FAIL: only $k named -- read it"; sed 's/^/        /' "$W/b.log"; fails=$((fails+1))
    fi
fi

# (c) WHAT MUST NOT BE REPORTED. A declaration, a default argument whose value is braced, a template,
# a class member and a `static`/`inline`/`constexpr` definition are all ordinary and all look a little
# like a definition to a careless parser. alpha_eqn_cpp.cuh alone carries four of the five.
mkdir -p "$W/c"
cp "$ROOT/src/applications/solvers/interFoam/alpha_eqn_cpp.cuh" \
   "$ROOT/src/finiteVolume/finiteVolume/fvc.cuh" \
   "$ROOT/src/finiteVolume/fields/fv_patch_field.cuh" \
   "$ROOT/src/applications/solvers/interFoam/device_inter_turbulence.cuh" "$W/c/"
if python3 "$AUDIT" "$W/c" > "$W/c.log" 2>&1; then
    echo "  ok:   declarations, braced default arguments, templates and class members are not reported"
else
    echo "  FAIL: one of them was read as an external definition"
    sed 's/^/        /' "$W/c.log"
    fails=$((fails+1))
fi

# (d) the ledger cannot rot: an entry naming nothing is reported
printf 'read_surface_field.cuh noSuchFunction\n' > "$W/allow.txt"
if python3 "$AUDIT" --allow "$W/allow.txt" "$ROOT/src" > "$W/d.log" 2>&1; then
    echo "  FAIL: a stale ledger entry passed"
    fails=$((fails+1))
elif grep -q "STALE" "$W/d.log"; then
    echo "  ok:   a ledger entry that names nothing is reported stale"
else
    echo "  FAIL: it failed, but not as stale -- read it"; sed 's/^/        /' "$W/d.log"; fails=$((fails+1))
fi

echo "header_odr_audit_selftest: $fails failures"
[ "$fails" = 0 ]
