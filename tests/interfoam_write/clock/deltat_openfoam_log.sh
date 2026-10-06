#!/usr/bin/env bash
# The gate: brae's time-step control against OPENFOAM'S OWN LOG, a step a row, bit for bit.
# interFoam prints at the top of every step the two Courant numbers it has just taken and then the deltaT
# setDeltaT.H chose from them; at writePrecision 17 each is the double it held. A row -- deltaT before, the two
# numbers, deltaT after -- is the formula's input and output with none of the flow in between, so
# setDeltaTVoF handed the first three must return the fourth to the bit
# (tests/test_time_controls_openfoam_log.cu). That is what lets a constant be held that moves deltaT by a few
# ulp: no gate that compares two RUNS can, a Courant number being already 1e-14 off after one step.
# THE DEFECT, found 2026-10-05 by reading: the three functions added 1e-37 to the Courant number -- the float
# build's VSMALL -- where setDeltaT.H adds SMALL, 1e-15 in the double build (doubleScalar.H:62), and maxDeltaT
# defaulted to 1e300 where readTimeControls.H says GREAT, 1e15. Every step the Courant number limits was a
# few ulp off OpenFOAM's expression.
# THE RUNS, three of laminar/damBreak (adjustTimeStep yes, maxCo 1, maxAlphaCo 1, maxDeltaT 1 as shipped) to
# t = STOP: as shipped, where the Courant number limits the step; with maxAlphaCo 0.5, where the interface's
# does -- the two branches of the min; and with maxDeltaT 0.001, where the last min does. All on `writeControl
# timeStep`, so that Time::adjustDeltaT, which lands a step on a write time, has nothing to land on.
# PREMISES, counted: each run has the steps its term decides, and the shipped one has damped and capped steps.
# MEASURED 2026-10-05: 186 steps as shipped (174 limited by the Courant number, 2 damped, 10 at the 1.2 cap)
# and 306 with maxAlphaCo 0.5 (280 limited by the interface's), every one OpenFOAM's bits; with 1e-37 back 175
# and 298 are not, up to 1.3e-15 and 2.3e-15 relative. 2026-10-06: 401 steps with maxDeltaT 0.001 (387 of them
# at maxDeltaT), every one OpenFOAM's bits.
# The CONTROL puts 1e-37 back (BRAE_CONTROL_DELTAT_SMALL_FLOAT=1): steps must then differ from the log in both
# functions, and the largest gap is printed.
# DOES NOT CLAIM: setInitialDeltaT.H by a run -- its cases are worked from the source, and NEITHER interFoam
# loop calls it yet (interFoam.C:81-85 does, before the loop: a start that has a flux, or a write cadence
# deltaT does not divide, can take another first step there). That brae's own run takes OpenFOAM's steps to
# the bit -- its Courant numbers are its own flux's. The rounding of `1 + 0.1*maxDeltaTFact` by measurement:
# it is a fused multiply-add in this machine's interFoam (read from the binary, time_controls.cuh) and is
# written as one here, but the logs hold two damped steps, too few to tell a fused sum from a rounded one.
. "$(dirname "$0")/../lib.sh"
REPLAY="${BUILD:-$ROOT/build}/test_time_controls_openfoam_log"
[ -x "$REPLAY" ] || { echo "SKIP: $REPLAY not built"; exit 77; }
STOP=0.4
# ctl <dir> <key>: a control as OpenFOAM read it
ctl() { sed -n -E "s/^$2 +([^;]+);.*/\1/p" "$1/system/controlDict" | head -1; }
# got <file> <key>: a field of the program's RESULT line
got() { grep -a "^RESULT " "$1" | tail -1 | tr ' ' '\n' | sed -n "s/^$2=//p"; }
# replayed <tag> [key=value ...]: OpenFOAM run on the staged case (cached), its log turned into rows -- deltaT
# before, Co, alphaCo, deltaT after -- and the rows handed to brae's functions, as they are and with the control
replayed()
{
    local tag="$1" o="$W/dt_$1"
    shift
    stage "$LAM" "$o" "$STOP" timeStep 100000 0 "$@" > "$W/dt_stage_$tag.txt" 2>&1 \
        || { say "laminar/damBreak did not stage [$tag]" FAIL; finish "the time step against OpenFOAM's log"; }
    runof "$o"
    [ "$(ctl "$o" adjustTimeStep)" = yes ] && [ -n "$(ctl "$o" maxCo)" ] && [ -n "$(ctl "$o" maxAlphaCo)" ] \
        && [ -n "$(ctl "$o" maxDeltaT)" ] \
        || { say "PREMISE  the staged case adjusts its step and names its three limits" FAIL; finish "time step"; }
    python3 - "$o/log.interFoam" "$(ctl "$o" deltaT)" > "$W/dt_rows_$tag.txt" <<'PY'
import re, sys
before = sys.argv[2]
co = alpha = None
for line in open(sys.argv[1], errors='replace'):
    m = re.match(r'^Courant Number mean: \S+ max: (\S+)', line)
    if m:
        co = m.group(1)
        continue
    m = re.match(r'^Interface Courant Number mean: \S+ max: (\S+)', line)
    if m:
        alpha = m.group(1)
        continue
    m = re.match(r'^deltaT = (\S+)', line)
    if m and co is not None and alpha is not None:
        print(before, co, alpha, m.group(1))
        before = m.group(1)
        co = alpha = None
PY
    "$REPLAY" "$W/dt_rows_$tag.txt" "$(ctl "$o" maxCo)" "$(ctl "$o" maxAlphaCo)" "$(ctl "$o" maxDeltaT)" \
        > "$W/dt_replay_$tag.txt" 2>&1
    BRAE_CONTROL_DELTAT_SMALL_FLOAT=1 "$REPLAY" "$W/dt_rows_$tag.txt" "$(ctl "$o" maxCo)" \
        "$(ctl "$o" maxAlphaCo)" "$(ctl "$o" maxDeltaT)" > "$W/dt_control_$tag.txt" 2>&1
}
replayed shipped
replayed interface maxAlphaCo=0.5
replayed capped maxDeltaT=0.001
a="$W/dt_replay_shipped.txt"
b="$W/dt_replay_interface.txt"
m="$W/dt_replay_capped.txt"
[ "$(ctl "$W/dt_interface" maxAlphaCo)" = 0.5 ] && [ "$(ctl "$W/dt_capped" maxDeltaT)" = 0.001 ] \
    || { say "PREMISE  the second and third runs' limits were restaged" FAIL; finish "the time step"; }
# every step of each log is a row
rowsHeld=yes
for tag in shipped interface capped; do
    [ "$(got "$W/dt_replay_$tag.txt" rows)" = "$(grep -c "^Time = " "$W/dt_$tag/log.interFoam")" ] || rowsHeld=no
    grep -q "CONTROL MODE" "$W/dt_replay_$tag.txt" && rowsHeld=no
done
what="[OpenFOAM's log] setDeltaTVoF: $(got "$a" rows) steps ($(got "$a" byCo) by Co, $(got "$a" damped) damped,"
what="$what $(got "$a" capped) capped), $(got "$b" rows) ($(got "$b" byAlphaCo) by the interface's),"
what="$what $(got "$m" rows) ($(got "$m" byMaxDeltaT) at maxDeltaT): $(got "$a" vofWrong), $(got "$b" vofWrong),"
what="$what $(got "$m" vofWrong) not its bits"
[ $rowsHeld = yes ] && [ "$(got "$a" byCo)" -ge 20 ] && [ "$(got "$a" damped)" -ge 1 ] \
    && [ "$(got "$a" capped)" -ge 1 ] && [ "$(got "$b" byAlphaCo)" -ge 20 ] && [ "$(got "$m" byMaxDeltaT)" -ge 20 ] \
    && [ "$(got "$a" vofWrong)" = 0 ] && [ "$(got "$b" vofWrong)" = 0 ] && [ "$(got "$m" vofWrong)" = 0 ] \
    && say "$what" ok || say "$what" FAIL
what="[OpenFOAM's log] setDeltaT where maxCo's branch decides: $(got "$a" generalWrong) of $(got "$a" generalRows)"
what="$what and $(got "$m" generalWrong) of $(got "$m" generalRows) not its bits; setInitialDeltaT, GREAT by source"
[ "$(got "$a" generalRows)" -ge 20 ] && [ "$(got "$a" generalWrong)" = 0 ] \
    && [ "$(got "$m" generalRows)" -ge 20 ] && [ "$(got "$m" generalWrong)" = 0 ] \
    && [ "$(got "$a" initialKept)" = ok ] && [ "$(got "$a" initialSet)" = ok ] && [ "$(got "$a" great)" = ok ] \
    && say "$what" ok || say "$what" FAIL
c="$W/dt_control_shipped.txt"
d="$W/dt_control_interface.txt"
what="CONTROL  1e-37 for SMALL: $(got "$c" vofWrong) and $(got "$d" vofWrong) steps not OpenFOAM's bits (worst"
what="$what $(got "$c" worst), $(got "$d" worst)), setDeltaT $(got "$c" generalWrong); setInitialDeltaT's first"
what="$what case $(got "$c" initialKept)"
grep -q "CONTROL MODE: the time-step control adds 1e-37" "$c" && grep -q "CONTROL MODE" "$d" \
    && [ "$(got "$c" vofWrongLimited)" -ge 1 ] && [ "$(got "$d" vofWrongLimited)" -ge 1 ] \
    && [ "$(got "$c" generalWrong)" -ge 1 ] && above "$(got "$c" worst)" 1e-16 && above "$(got "$d" worst)" 1e-16 \
    && [ "$(got "$c" initialKept)" = no ] && say "$what" ok || say "$what" FAIL
finish "brae's time-step control returns the deltaT of every step of OpenFOAM's log, to the bit"
