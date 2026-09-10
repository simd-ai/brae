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

# --- properties liquid -> refused on THIS arm, naming the liquid and the arm ----------------------
# Stage H3.4 lifted the liquid refusal from createFields, which both arms share, because the HOST step
# now evaluates every property through liquid_thermo.cuh. The device kernels still call the perfect-gas
# closed forms, so createDeviceFields refuses instead -- and the harness, like the driver, passes through
# it. The hot wall is lowered to 350 K so the case stays inside H2O's [273.16, 647.13] K: at the
# fixture's own 700 K the HOST createFields range refusal would fire first and this arm would be
# measuring that one instead.
#
# Fail-proof, measured by disabling the refusal and re-running: the harness then gets as far as the host
# reference's thermo.correct(), where the new he -> T inversion refuses -- he = -84235.39 J/kg at cell 0,
# T = nan -- because the device projection had built he with the PERFECT-GAS formula (Cv*T-scale numbers,
# where H2O's Es is ~ -1.6e7). Not silent, thanks to the inversion's own check, but misattributed; with
# the refusal it stops at the right place under the right name.
mkarm
cat > "$W/c/constant/thermophysicalProperties" <<'TEOF'
FoamFile { version 2.0; format ascii; class dictionary; object thermophysicalProperties; }
thermoType { type heRhoThermo; mixture pureMixture; properties liquid; energy sensibleInternalEnergy; }
mixture { H2O; }
TEOF
sed -i 's/hotWall  { type fixedValue; value uniform 700; }/hotWall  { type fixedValue; value uniform 350; }/' "$W/c/0.orig/T"
grep -q "uniform 350" "$W/c/0.orig/T" || { echo "FAIL: could not lower the hot wall into H2O's range"; fail=1; }
# A liquid is sensibleInternalEnergy, so the energy variable is `e` and the kinetic term `Ekp`; rhoBox
# ships `h` and `K`. Without these the arm is refused on the missing div(phi,e) scheme -- a correct
# refusal, and not the one this arm exists to see.
sed -i 's/    div(phi,h) bounded Gauss upwind;/    div(phi,e) bounded Gauss upwind;/; s/    div(phi,K) bounded Gauss upwind;/    div(phi,Ekp) bounded Gauss upwind;/' "$W/c/system/fvSchemes"
grep -q "div(phi,e)" "$W/c/system/fvSchemes" && grep -q "div(phi,Ekp)" "$W/c/system/fvSchemes" \
    || { echo "FAIL: could not give the liquid arm its energy schemes"; fail=1; }
out=$("$BIN" "$W/c" 0.orig 2 2>&1) && { echo "FAIL: a liquid thermo ran on the CUDA path"; fail=1; }
echo "$out" | grep -q "what():.*CUDA arm implements perfectGas.*properties liquid" \
    && echo "  properties liquid refused on the CUDA arm        ok" \
    || { echo "$out" | tail -3; echo "FAIL: no CUDA-arm liquid REFUSAL fired"; fail=1; }

[ $fail = 0 ] && echo PASS || { echo FAIL; exit 1; }
