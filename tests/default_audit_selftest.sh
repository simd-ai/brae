#!/usr/bin/env bash
# tools/default_audit.py's ledger keys, and the proof that a name shared by two types cannot be
# exempted with one line.
#
# `Compressible` is defined in kEpsilonRef AND in kOmegaSST, and the two carry the same field names.
# Keyed on the struct name alone, one ledger line exempted the field in both, and a line that had gone
# stale in one was held alive by the other. The key carries the NAMESPACE now, as C++ does and as
# every site in this tree writes. Nothing in the tree is modified: the ledger is copied and edited in
# a scratch directory.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AUDIT="$ROOT/tools/default_audit.py"
ALLOW="$ROOT/tools/default_audit_allow.txt"
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
fails=0
# SCOPED to the two subtrees that hold every Compressible definition and every site that builds one --
# the closures define them, the solvers build them. The WHOLE tree is scanned by default_audit_interfoam
# beside this gate; scanning it four more times here cost nine minutes and told no one anything new.
SCOPE="$ROOT/src/TurbulenceModels $ROOT/src/applications"

echo "== tools/default_audit.py: the ledger, and the two types that share a name =="

if python3 "$AUDIT" --allow "$ALLOW" $SCOPE > "$W/tree.log" 2>&1; then
    echo "  ok:   the closures and the solvers are clean under the ledger as it stands"
else
    echo "  FAIL: they are not -- read it, this gate is not the place to silence it"
    sed 's/^/        /' "$W/tree.log"; fails=$((fails+1))
fi
if grep -q "kEpsilonRef::Compressible" "$ALLOW" && grep -q "kOmegaSST::Compressible" "$ALLOW"; then
    echo "  ok:   both Compressible types carry ledger lines of their own"
else
    echo "  FAIL: the ledger no longer separates the two Compressible types"; fails=$((fails+1))
fi

# (a) the BARE form, which used to cover both types, is reported rather than matched
sed 's/^kEpsilonRef::Compressible comp rho /Compressible comp rho /' "$ALLOW" > "$W/a.txt"
if python3 "$AUDIT" --allow "$W/a.txt" $SCOPE > "$W/a.log" 2>&1; then
    echo "  FAIL: a bare \`Compressible comp rho\` line was matched"; fails=$((fails+1))
elif grep -q "AMBIGUOUS Compressible comp rho" "$W/a.log"; then
    echo "  ok:   a line naming the struct alone is reported ambiguous, not matched"
else
    echo "  FAIL: it failed, but not as ambiguous -- read it"; sed 's/^/        /' "$W/a.log"; fails=$((fails+1))
fi

# (b) removing ONE namespace's line reports that namespace's site and no other
grep -v "^kOmegaSST::Compressible comp rhoOld" "$ALLOW" > "$W/b.txt"
if python3 "$AUDIT" --allow "$W/b.txt" $SCOPE > "$W/b.log" 2>&1; then
    echo "  FAIL: the other type's line covered the missing one"; fails=$((fails+1))
elif [ "$(grep -c '^UNLISTED  kOmegaSST::Compressible' "$W/b.log")" = 1 ] \
     && ! grep -q '^UNLISTED  kEpsilonRef::Compressible' "$W/b.log"; then
    echo "  ok:   removing one type's line reports that type's site alone"
else
    echo "  FAIL: the wrong sites were reported -- read it"; sed 's/^/        /' "$W/b.log"; fails=$((fails+1))
fi

# (c) THE CONFLATION ITSELF: a line filed under the wrong type goes stale AND leaves the right one bare
sed 's/^kEpsilonRef::Compressible comp phiByRho /kEpsilonRef::Compressible sstComp phiByRho /' "$ALLOW" > "$W/c.txt"
if python3 "$AUDIT" --allow "$W/c.txt" $SCOPE > "$W/c.log" 2>&1; then
    echo "  FAIL: a line filed under the wrong variable of the wrong type passed"; fails=$((fails+1))
elif grep -q "^STALE     kEpsilonRef::Compressible sstComp phiByRho" "$W/c.log" \
     && grep -q "^UNLISTED  kEpsilonRef::Compressible" "$W/c.log"; then
    echo "  ok:   a misfiled line is stale where it sits and unlisted where it belongs"
else
    echo "  FAIL: it failed, but not both ways -- read it"; sed 's/^/        /' "$W/c.log"; fails=$((fails+1))
fi

echo "default_audit_selftest: $fails failures"
[ "$fails" = 0 ]
