#!/usr/bin/env bash
# The gate: setInitialDeltaT.H before the time loop, brae's first steps against OPENFOAM'S OWN LOG.
# interFoam.C:81-85 runs CourantNo.H and setInitialDeltaT.H once, before the loop. At time index 0 with a
# Courant number above SMALL it calls runTime.setDeltaT(min(maxCo*deltaT/CoNum, min(deltaT, maxDeltaT))), and
# Time::setDeltaT lands the value on the write cadence on the way in (`adjust` defaults to true,
# Time.C:981-990) -- also when the value is the deltaT it already holds. With a Courant number of 0 (a start
# from rest) the statement is not reached and nothing is adjusted. Neither of brae's loops made it until
# 2026-10-06 (found by a review of the time-step constants): a start with a flux went into its first step
# with the controlDict's deltaT where OpenFOAM's is already trimmed to the write interval.
# THE FIXTURE is laminar/damBreak with `writeInterval 0.011` (adjustable) and `deltaT 0.0025`, which does
# not divide it (0.011/0.0025 = 4.4), run to STOP, twice:
#   with a flux at the start (`internalField uniform (0.01 0 0)` in 0/U; Courant number 2e-03): OpenFOAM
#     trims deltaT to 0.011/4 before the loop, the first step grows that by 1.2 and lands on 0.011/3;
#   from rest, as shipped: nothing before the loop, the first step grows 0.0025 by 1.2 and lands on 0.011/4.
# So the two rows' first steps are 0.011/3 and 0.011/4, a quarter apart, and a brae that made the adjustment
# always, or never, takes the wrong one on one row. Both are PREMISES read from OpenFOAM's log, as numbers.
# HELD, host arm and device arm: the Courant number each loop hands the statement, against the one OpenFOAM
# prints before `Starting time loop` (CourantNo.H, at writePrecision 17); the first step OpenFOAM's and every
# step after it over the run (the Courant number never limits here, so the sequence is the clock's
# arithmetic), within BOUND. Two later steps are ROUNDING TIES of Time::adjustDeltaT held bit for bit, as the
# damBreak clock gate holds bits: the flux row's fourth step rounds 2.4999999999999996 and the rest row's
# second 2.5000000000000004, so a one-ulp difference there would show as a step of another size, not as
# something BOUND absorbs.
# The CONTROL leaves the statement out (BRAE_CONTROL_INITIAL_DELTAT_SKIPPED=1) on the row with a flux, in each
# loop: the first step must be 0.011/4 there.
# NOT HERE: the branch where the Courant number at the start exceeds maxCo, and the pair's faces in the device's
# sum at that call -- clock/initial_deltat_pair.sh holds both, on a baffle. DOES NOT CLAIM a restart (time
# index above 0, where the statement is not reached) or LTS (interFoam.C skips it).
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
STOP=0.022
WI=0.011
DT=0.0025
# MEASURED 2026-10-06: 0 on every step of both rows, host and device (the clock is arithmetic here); the
# bound is headroom for another compiler's contraction, as the damBreak clock gate's is
BOUND=1e-13
# row <tag> <U internalField>: the staged case, OpenFOAM's run, brae's host and device runs, and on the row
# with a flux the control
row()
{
    local tag="$1" o="$W/id_$1" arm
    stage "$LAM" "$o" "$STOP" adjustable "$WI" 0 deltaT=$DT timePrecision=17 > "$W/id_stage_$tag.txt" 2>&1 \
        || { say "laminar/damBreak did not stage [$tag]" FAIL; finish "the initial deltaT"; }
    sed -i -E "s/^internalField +uniform +\(0 0 0\);/internalField   uniform $2;/" "$o/0/U"
    grep -q "^internalField   uniform $2;" "$o/0/U" \
        || { say "PREMISE  0/U's internalField was set to $2 [$tag]" FAIL; finish "the initial deltaT"; }
    runof "$o"
    for arm in host device control controlHost; do
        [ "${arm#control}" != "$arm" ] && [ "$tag" != flux ] && continue
        mkdir -p "$W/id_${tag}_$arm"
        cp -r "$o/0" "$o/constant" "$o/system" "$W/id_${tag}_$arm/"
        case $arm in
            host) runbrae "$W/id_${tag}_$arm" host ;;
            device) runbrae "$W/id_${tag}_$arm" device ;;
            control) runbrae "$W/id_${tag}_$arm" device BRAE_CONTROL_INITIAL_DELTAT_SKIPPED=1 ;;
            controlHost) runbrae "$W/id_${tag}_$arm" host BRAE_CONTROL_INITIAL_DELTAT_SKIPPED=1 ;;
        esac
    done
}
# steps <OpenFOAM log> <brae log> <expected first step as a python expression>: "<ok|no> <OpenFOAM's first
# step> <brae's first step> <steps OpenFOAM> <steps brae> <worst relative gap of a step>". ok: OpenFOAM's
# first step is the expected one to 1e-12 (the premise), both took the same number of steps, more than
# three, and no step of brae's is further than BOUND from OpenFOAM's
steps()
{
    python3 - "$1" "$2" "$3" "$BOUND" <<'PY'
import re, sys
of = [float(x) for x in re.findall(r'^deltaT = (\S+)', open(sys.argv[1]).read(), flags=re.M)]
br = [float(x) for x in re.findall(r'^\s*t = \S+\s+dt = (\S+)', open(sys.argv[2]).read(), flags=re.M)]
want = eval(sys.argv[3])
if not of or not br:
    print('no - - %d %d -' % (len(of), len(br)))
    sys.exit(0)
worst = max(abs(a - b)/a for a, b in zip(of, br))
ok = abs(of[0] - want) < 1e-12*want and len(of) == len(br) and len(of) > 3 and worst < float(sys.argv[4])
print('%s %.17g %.17g %d %d %.1e' % ('ok' if ok else 'no', of[0], br[0], len(of), len(br), worst))
PY
}
# co0 <OpenFOAM log> <brae log>: the start's Courant number in both, as "<OpenFOAM's> <brae's> <relative gap>";
# OpenFOAM's is the first `Courant Number mean: .. max: ..` line, printed before the time loop
co0()
{
    python3 - "$1" "$2" <<'PY'
import re, sys
of = re.search(r'^Courant Number mean: \S+ max: (\S+)', open(sys.argv[1]).read(), flags=re.M)
br = re.search(r'setInitialDeltaT: Courant number (\S+) at the start', open(sys.argv[2]).read())
if not of or not br:
    print('- - 1')
    sys.exit(0)
a, b = float(of.group(1)), float(br.group(1))
print('%.17g %.17g %.1e' % (a, b, abs(a - b)/a if a else abs(b)))
PY
}
row flux "(0.01 0 0)"
row rest "(0 0 0)"
read -r coOf coH gapH <<< "$(co0 "$W/id_flux/log.interFoam" "$W/id_flux_host/log.brae")"
read -r coOf coD gapD <<< "$(co0 "$W/id_flux/log.interFoam" "$W/id_flux_device/log.brae")"
read -r vh of1 h1 no nh wh <<< "$(steps "$W/id_flux/log.interFoam" "$W/id_flux_host/log.brae" "$WI/3")"
read -r vd of1 d1 no nd wd <<< "$(steps "$W/id_flux/log.interFoam" "$W/id_flux_device/log.brae" "$WI/3")"
what="[host] [device] a start with a flux (Courant number $coOf, brae's $gapH and $gapD off): OpenFOAM's first"
what="$what step $of1 (0.011/3); brae's $h1 and $d1; $no steps, worst $wh and $wd"
[ "$vh" = ok ] && [ "$vd" = ok ] && within "$gapH" 1e-12 && within "$gapD" 1e-12 \
    && python3 -c "import sys; sys.exit(0 if 1e-15 < $coOf < 1 else 1)" \
    && grep -q "setInitialDeltaT: Courant number" "$W/id_flux_device/log.brae" \
    && ! grep -q "CONTROL MODE" "$W/id_flux_device/log.brae" "$W/id_flux_host/log.brae" \
    && say "$what" ok || say "$what" FAIL
read -r vh of1 h1 no nh wh <<< "$(steps "$W/id_rest/log.interFoam" "$W/id_rest_host/log.brae" "$WI/4")"
read -r vd of1 d1 no nd wd <<< "$(steps "$W/id_rest/log.interFoam" "$W/id_rest_device/log.brae" "$WI/4")"
what="[host] [device] from rest, nothing is adjusted before the loop: OpenFOAM's first step $of1 (0.011/4);"
what="$what brae's $h1 and $d1; $no steps, worst $wh and $wd"
[ "$vh" = ok ] && [ "$vd" = ok ] && say "$what" ok || say "$what" FAIL
read -r vc of1 c1 no nc wc <<< "$(steps "$W/id_flux/log.interFoam" "$W/id_flux_control/log.brae" "$WI/3")"
read -r vk of1 k1 no nk wk <<< "$(steps "$W/id_flux/log.interFoam" "$W/id_flux_controlHost/log.brae" "$WI/3")"
what="CONTROL  the statement left out on the row with a flux: the first step is $c1 [device] and $k1 [host]"
what="$what where OpenFOAM's is $of1"
mode="CONTROL MODE: setInitialDeltaT.H is not made before the time loop"
[ "$vc" = no ] && [ "$vk" = no ] \
    && python3 -c "import sys; sys.exit(0 if abs($c1 - $WI/4) < 1e-12 and abs($k1 - $WI/4) < 1e-12 else 1)" \
    && grep -q "$mode" "$W/id_flux_control/log.brae" && grep -q "$mode" "$W/id_flux_controlHost/log.brae" \
    && say "$what" ok || say "$what" FAIL
finish "brae makes setInitialDeltaT.H before its time loop where OpenFOAM does, and lands the step as it does"
