#!/usr/bin/env bash
# The gate: a start ABOVE maxCo on a mesh with a COUPLED PAIR -- the two things clock/initial_deltat.sh does not
# claim -- brae's first steps against OPENFOAM'S OWN LOG.
# Above maxCo, setInitialDeltaT.H trims deltaT to maxCo*deltaT/CoNum before the loop, Time::setDeltaT lands it
# on the write cadence (round(timeToNextWrite/deltaT) steps, Time.C adjustDeltaT), and the loop's first
# setDeltaT.H computes the same step again from the same flux. So the first step's size is decided by the
# Courant number of the START flux -- and on a mesh with a pair that number has the pair's faces in each
# cell's sum (fvc::surfaceSum(mag(phi)), fvcSurfaceIntegrate.C:170-182). The device kept them apart and left
# them out until 2026-10-05; every gate that found that pinned the time step, and none let one move on such a
# mesh.
# WHAT THE STATEMENT ITSELF DOES NOT SHOW HERE, held so that it stays true: with it left out
# (BRAE_CONTROL_INITIAL_DELTAT_SKIPPED=1) the loop's own statement trims and lands the same step, so the first
# step is the same on this branch. The branch is held through the NUMBER it uses.
# THE FIXTURE: RAS/damBreakPorousBaffle as shipped (adjustTimeStep on, maxCo 0.1, deltaT 0.001, writeInterval
# 0.05), its baffle made, with U painted (4 0 0) in the one column of cells on each side of the baffle, run to
# 0.004. interFoam corrects that flux before its loop (initCorrectPhi.H; brae does too), which leaves a jet
# through the baffle: MEASURED 2026-10-07, the start's Courant number 0.1484 (maxCo 0.1), OpenFOAM's first
# step 6.757e-04 = 0.05/74; with the pair left out of the sum 0.1442 and 6.944e-04 = 0.05/72.
# HELD, host arm and device arm: the Courant number each loop hands the statement against the one OpenFOAM
# prints before its loop; the first step, OpenFOAM's; and the run's steps after it (the flow limits them:
# within a measured bound). PREMISES, as numbers from OpenFOAM's log: its start Courant number is above
# maxCo, and its first step is below the controlDict's deltaT.
# CONTROL: BRAE_CONTROL_COURANT_PAIR_LEFT_OUT=1 on the device arm: its start number and its first step are
# not OpenFOAM's.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
SRC="$TUT/multiphase/interFoam/RAS/damBreakPorousBaffle"
STOP=0.004
WI=0.05
o="$W/ip"
stage "$SRC" "$o" "$STOP" adjustable "$WI" 0 timePrecision=17 > "$W/ip_stage.txt" 2>&1 \
    || { say "RAS/damBreakPorousBaffle did not stage" FAIL; finish "the initial deltaT on a pair"; }
cat > "$o/system/setFieldsDictFlux" <<'EOF_D'
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      setFieldsDictFlux;
}
defaultFieldValues
(
    volVectorFieldValue U (0 0 0)
);
regions
(
    boxToCell
    {
        box (0.2982 0.0493 -1) (0.3102 0.2077 1);
        fieldValues
        (
            volVectorFieldValue U (4 0 0)
        );
    }
);
EOF_D
( cd "$o" && createBaffles -overwrite > log.createBaffles 2>&1 && setFields -dict system/setFieldsDictFlux \
      > log.setFieldsFlux 2>&1 ) \
    || { say "the baffle or the start flux was not staged" FAIL; finish "the initial deltaT on a pair"; }
painted=$(grep -ac "^(4 0 0)$" "$o/0/U")
runof "$o"
for arm in host device control skipped; do
    mkdir -p "$W/ip_$arm"
    cp -r "$o/0" "$o/constant" "$o/system" "$W/ip_$arm/"
    case $arm in
        host) runbrae "$W/ip_$arm" host ;;
        device) runbrae "$W/ip_$arm" device ;;
        control) runbrae "$W/ip_$arm" device BRAE_CONTROL_COURANT_PAIR_LEFT_OUT=1 ;;
        skipped) runbrae "$W/ip_$arm" device BRAE_CONTROL_INITIAL_DELTAT_SKIPPED=1 ;;
    esac
done
# co0 <OpenFOAM log> <brae log>: "<OpenFOAM's start Courant number> <brae's> <relative gap>"
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
# steps <OpenFOAM log> <brae log>: "<OpenFOAM's first step> <brae's> <its relative gap> <steps OpenFOAM>
# <steps brae> <worst relative gap of a later step>"
steps()
{
    python3 - "$1" "$2" <<'PY'
import re, sys
of = [float(x) for x in re.findall(r'^deltaT = (\S+)', open(sys.argv[1]).read(), flags=re.M)]
br = [float(x) for x in re.findall(r'^\s*t = \S+\s+dt = (\S+)', open(sys.argv[2]).read(), flags=re.M)]
if not of or not br:
    print('- - 1 %d %d 1' % (len(of), len(br)))
    sys.exit(0)
later = max([abs(a - b)/a for a, b in list(zip(of, br))[1:]] or [0.0])
print('%.17g %.17g %.1e %d %d %.1e' % (of[0], br[0], abs(of[0] - br[0])/of[0], len(of), len(br), later))
PY
}
maxCo=$(sed -n -E 's/^maxCo +([^;]+);/\1/p' "$o/system/controlDict")
dt=$(sed -n -E 's/^deltaT +([^;]+);/\1/p' "$o/system/controlDict")
read -r coOf coH gapH <<< "$(co0 "$o/log.interFoam" "$W/ip_host/log.brae")"
read -r coOf coD gapD <<< "$(co0 "$o/log.interFoam" "$W/ip_device/log.brae")"
read -r of1 h1 fh no nh lh <<< "$(steps "$o/log.interFoam" "$W/ip_host/log.brae")"
read -r of1 d1 fd no nd ld <<< "$(steps "$o/log.interFoam" "$W/ip_device/log.brae")"
what="[host] [device] a start above maxCo on a pair (Courant number $coOf for maxCo $maxCo, $painted cells"
what="$what painted; brae's $gapH and $gapD off): OpenFOAM's first step $of1, brae's $fh and $fd off"
[ "$painted" -gt 0 ] && above "$coOf" "$maxCo" && within "$gapH" 1e-12 && within "$gapD" 1e-12 \
    && python3 -c "import sys; sys.exit(0 if $of1 < 0.9*$dt else 1)" \
    && within "$fh" 1e-12 && within "$fd" 1e-12 \
    && grep -q "the coupled pair's .* faces are in each cell's flux sum" "$W/ip_device/log.brae" \
    && say "$what" ok || say "$what" FAIL
# MEASURED 2026-10-07: 0 on every one of the five steps, host and device -- past the first step the jet has
# decayed and the steps grow by the clock's own arithmetic; the bound is headroom for another compiler's
# contraction, as the damBreak clock gate's is
LATER=1e-13
read -r of1 s1 fs no ns ls <<< "$(steps "$o/log.interFoam" "$W/ip_skipped/log.brae")"
what="[host] [device] the steps after it: $no in OpenFOAM's run, $nh and $nd in brae's, the worst $lh and $ld"
what="$what off (bound $LATER); with the statement left out the first step is the same ($fs off)"
[ "$no" -gt 3 ] && [ "$nh" = "$no" ] && [ "$nd" = "$no" ] && within "$lh" "$LATER" && within "$ld" "$LATER" \
    && within "$fs" 1e-12 && grep -q "CONTROL MODE: setInitialDeltaT.H is not made" "$W/ip_skipped/log.brae" \
    && say "$what" ok || say "$what" FAIL
read -r coOf coC gapC <<< "$(co0 "$o/log.interFoam" "$W/ip_control/log.brae")"
read -r of1 c1 fc no nc lc <<< "$(steps "$o/log.interFoam" "$W/ip_control/log.brae")"
what="CONTROL  [device] the pair left out of the sum: the start number $coC for OpenFOAM's $coOf, the first"
what="$what step $c1 for $of1"
above "$gapC" 1e-3 && above "$fc" 1e-3 && say "$what" ok || say "$what" FAIL
finish "a start above maxCo on a mesh with a coupled pair takes OpenFOAM's first steps"
