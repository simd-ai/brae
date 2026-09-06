#!/bin/bash
# The device-twin refusal flags are DERIVED FROM THE CASE, not only from fail-proofs.
#
# rhoUEqn.cu/rhoEEqn.cu/rhoPEqn.cu/rhoPcEqn.cu have refused hasMRF/hasFvOptions since they were
# written, but no harness ever set the flags from a real dictionary -- a case declaring MRFProperties
# or an fvOption ran the CUDA path with the term silently dropped. Each arm mutates a copy of rhoBox
# and requires the refusal BY NAME; the pristine rhoBox run (ctest rho_simple_step_cuda) is the
# standing negative control.
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN="${BUILD:-$ROOT/build}/test_rho_simple_step_cuda"
SRC="$ROOT/validation/rhoBox"
[ -x "$BIN" ] || { echo "SKIP: no cuda harness"; exit 77; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0

mkarm() { rm -rf "$W/c"; cp -r "$SRC" "$W/c"; }

# --- MRFProperties present -> refused naming MRF -------------------------------------------------
mkarm
cat > "$W/c/constant/MRFProperties" <<'MEOF'
FoamFile { version 2.0; format ascii; class dictionary; object MRFProperties; }
zone1 { cellZone rotor; active yes; origin (0 0 0); axis (0 0 1); omega 10; }
MEOF
out=$("$BIN" "$W/c" 0.orig 2 2>&1) && { echo "FAIL: MRFProperties ran on the CUDA path"; fail=1; }
echo "$out" | grep -q "what():.*declares MRF" \
    && echo "  MRF refused by name                              ok" \
    || { echo "$out" | tail -3; echo "FAIL: no MRF REFUSAL fired (a log line is not a refusal)"; fail=1; }

# --- an fvOption present -> refused naming it ----------------------------------------------------
mkarm
cat > "$W/c/system/fvOptions" <<'FEOF'
FoamFile { version 2.0; format ascii; class dictionary; object fvOptions; }
heater { type semiImplicitSource; active yes; }
FEOF
out=$("$BIN" "$W/c" 0.orig 2 2>&1) && { echo "FAIL: an fvOption ran on the CUDA path"; fail=1; }
echo "$out" | grep -qE "what\(\):.*(fvOption|semiImplicitSource)" \
    && echo "  fvOption refused                                 ok" \
    || { echo "$out" | tail -3; echo "FAIL: no fvOption REFUSAL fired"; fail=1; }

[ $fail = 0 ] && echo PASS || { echo FAIL; exit 1; }
