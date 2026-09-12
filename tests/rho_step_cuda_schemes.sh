#!/usr/bin/env bash
# THE CUDA STEP HARNESS ASSEMBLES THE CASE'S OWN DIV SCHEMES -- held against fvSchemes, not against the
# host arm.
#
# tests/test_rho_simple_step_cuda.cu built its StepInput by hand and never set schemeU/schemeHe/schemeKE,
# so both of its arms assembled upwind on U, he and K whatever the case named. Its four registered gates
# stayed green throughout, for two reasons that make this file necessary:
#   1. rhoBox and sbMatched are `bounded Gauss upwind` on every entry, so the omission changed nothing.
#   2. The device and host arms SHARE that input, so on a fixture that is not upwind both ran the same
#      wrong scheme and agreed with each other to 1e-11. A device-against-host comparison cannot see a
#      substitution made upstream of both.
# The harness now takes its input from cpu::rhoSimple::buildStepInput (the parse `brae -case` ships) and
# prints the schemes it resolved. This gate reads that line and checks each word against the fixture's
# own system/fvSchemes, read here with an independent regex -- the oracle is the case file itself.
#
#   ARM 1  validation/rhoKE2  (e-thermo)  div(phi,U) upwind, div(phi,e) linearUpwind, div(phi,Ekp) linearUpwind
#   ARM 2  validation/rhoBox  (h-thermo)  all three upwind
#   CONTROL  the two fixtures must DISAGREE on at least one entry, or a harness that printed one word
#            for everything would pass both arms.
#
# Fail-proof, 2026-09-03: with schemeHe/schemeKE forced back to upwind after the shared parse, arm 1
# reads `div(phi,e) harness upwind, fvSchemes linearUpwind` and `div(phi,Ekp)` the same, the control
# reads `upwind vs upwind`, and the script exits 1 -- while the harness's own device-vs-host bounds
# stay green (Ux 1.5e-11), which is point 2 above, measured. The first version of this file ran its arms
# inside a command substitution, so the fail flag never reached the verdict and it printed PASS over
# FAIL rows; the arms are now called directly.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_rho_simple_step_cuda"
[ -x "$BIN" ] || { echo "SKIP: $BIN not built"; exit 77; }

fail=0
say() { printf '  %-70s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

# The scheme word of div(phi,<field>) in a fixture's fvSchemes, independently of brae's parser.
expect()
{
    python3 - "$1/system/fvSchemes" "$2" <<'PYEOF_SCHEME'
import re, sys
s = open(sys.argv[1]).read()
s = re.sub(r'//.*', '', s)
m = re.search(r'div\(phi,%s\)\s+(?:bounded\s+)?Gauss\s+(\w+)' % re.escape(sys.argv[2]), s)
print(m.group(1) if m else 'MISSING')
PYEOF_SCHEME
}

# The energy variable a fixture solves (thermophysicalProperties energy), and its kinetic-energy entry.
heName()
{
    # No line anchor: brae's fixtures write the whole thermo dictionary on one line.
    if grep -qE 'energy\s+sensibleInternalEnergy' "$1/constant/thermophysicalProperties"; then echo e; else echo h; fi
}

# One arm: run the harness, take the resolved line, compare the three words. Called DIRECTLY, never
# inside $(...): the fail flag the rows set must reach the verdict, and a subshell would drop it (the
# first version of this file did exactly that and could not fail). The energy word goes to ARM_HE.
ARM_HE=""
arm()
{
    local fx="$1" name="$2"
    local he ke line got want fld
    ARM_HE=""
    he=$(heName "$fx")
    ke=$([ "$he" = e ] && echo Ekp || echo K)
    line=$("$BIN" "$fx" 0.orig 2 2>&1 | grep 'resolved schemes:' | head -1)
    [ -n "$line" ] || { say "$name: the harness printed its resolved schemes" FAIL; return; }
    for fld in U "$he" "$ke"; do
        got=$(echo "$line" | sed -n "s/.*div(phi,$fld)=\([A-Za-z]*\).*/\1/p")
        want=$(expect "$fx" "$fld")
        [ "$want" != MISSING ] || { say "$name: fvSchemes names div(phi,$fld)" FAIL; continue; }
        if [ "$got" = "$want" ]; then
            say "$name: div(phi,$fld) harness $got == fvSchemes $want" ok
        else
            say "$name: div(phi,$fld) harness ${got:-<none>}, fvSchemes $want" FAIL
        fi
        [ "$fld" = "$he" ] && ARM_HE="$got"
    done
}

arm "$ROOT/validation/rhoKE2" "arm 1 rhoKE2"; a1="$ARM_HE"
arm "$ROOT/validation/rhoBox" "arm 2 rhoBox";  a2="$ARM_HE"

# CONTROL: non-vacuity. The energy entries of the two fixtures must resolve to DIFFERENT words.
if [ -n "$a1" ] && [ -n "$a2" ] && [ "$a1" != "$a2" ]; then
    say "control: the two fixtures resolve different energy schemes ($a1 vs $a2)" ok
else
    say "control: the two fixtures resolve different energy schemes (${a1:-?} vs ${a2:-?})" FAIL
fi

[ $fail = 0 ] && echo PASS || { echo FAIL; exit 1; }
