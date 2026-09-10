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

# --- properties liquid on THIS arm ---------------------------------------------------------------
# Until stage H3.6 this arm asserted the CUDA arm REFUSED a liquid: the host step had H3.4's accessors
# and the device kernels still called the perfect-gas closed forms. The device asks the same accessors
# now, so the arm asserts the two halves of the new contract on the same staging -- rhoBox with H2O:
#   (a) in range (hot wall lowered to 350 K) the liquid RUNS through the CUDA harness, and
#   (b) at the fixture's own 700 K hot wall -- above H2O's 647.13 K critical point, where the density
#       correlation is a NaN -- it is refused by name, naming the substance and the range, on this
#       arm as on the host (createFields is shared).
# Whether the liquid runs CORRECTLY on this arm is the three liquid gates' question
# (liquid_thermo/liquid_turbulence/rho_squarebendliq_vs_openfoam, each with a CUDA arm).
mkliq() {
    mkarm
    cat > "$W/c/constant/thermophysicalProperties" <<'TEOF'
FoamFile { version 2.0; format ascii; class dictionary; object thermophysicalProperties; }
thermoType { type heRhoThermo; mixture pureMixture; properties liquid; energy sensibleInternalEnergy; }
mixture { H2O; }
TEOF
    # A liquid is sensibleInternalEnergy, so the energy variable is `e` and the kinetic term `Ekp`;
    # rhoBox ships `h` and `K`. Without these the arm is refused on the missing div(phi,e) scheme -- a
    # correct refusal, and not the one this arm exists to see.
    sed -i 's/    div(phi,h) bounded Gauss upwind;/    div(phi,e) bounded Gauss upwind;/; s/    div(phi,K) bounded Gauss upwind;/    div(phi,Ekp) bounded Gauss upwind;/' "$W/c/system/fvSchemes"
    grep -q "div(phi,e)" "$W/c/system/fvSchemes" && grep -q "div(phi,Ekp)" "$W/c/system/fvSchemes" \
        || { echo "FAIL: could not give the liquid arm its energy schemes"; fail=1; }
}
mkliq
sed -i 's/hotWall  { type fixedValue; value uniform 700; }/hotWall  { type fixedValue; value uniform 350; }/' "$W/c/0.orig/T"
grep -q "uniform 350" "$W/c/0.orig/T" || { echo "FAIL: could not lower the hot wall into H2O's range"; fail=1; }
# The harness's own controls are written for the GAS rhoBox (the pressure limiter binds, p moves) and
# report FAIL on a liquid without any of them being a refusal, so its exit status says nothing here --
# the question is only whether a refusal fired.
out=$("$BIN" "$W/c" 0.orig 2 2>&1) || true
if echo "$out" | grep -q "what():"; then
    echo "$out" | grep "what():" | head -2; echo "FAIL: a liquid in its range was refused on the CUDA path"; fail=1
else
    echo "  properties liquid in range runs on the CUDA path      ok"
fi

mkliq
out=$("$BIN" "$W/c" 0.orig 2 2>&1) && { echo "FAIL: H2O at 700 K ran on the CUDA path"; fail=1; }
echo "$out" | grep -q "what():.*H2O.*647.13" \
    && echo "  H2O above its critical point refused, naming the range ok" \
    || { echo "$out" | tail -3; echo "FAIL: no correlation-range REFUSAL fired on the CUDA path"; fail=1; }

[ $fail = 0 ] && echo PASS || { echo FAIL; exit 1; }
