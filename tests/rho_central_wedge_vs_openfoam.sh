#!/usr/bin/env bash
# brae_rhoCentralFoam against OpenFOAM v2412 rhoCentralFoam on the transform patches: symmetryPlane, symmetry, slip
# and wedge (milestone C). wedge15Ma5 and obliqueShock run AS SHIPPED; three variants of wedge15Ma5, each made by
# tests/rho_central_cases.py, exercise what the shipped two do not: `symmetry` (per-face normals, on the inclined
# ramp too), `slip` on p and T (a transform energy BC on a plain patch), and `wedge` (a 15-degree cone, axisymmetric).
#
# CASES (blockMesh on the tutorial's files; nothing else changed unless named)
#   wedge, oblique      the tutorials as shipped (writePrecision 6): wedge15Ma5 fixed deltaT 1e-4 to 0.2, runTime
#                       writes; obliqueShock adjustTimeStep (maxCo 0.2), adjustable writes to 10
#   wedge17, oblique17  the same at writePrecision 17 (its one change: the files' digits and Info's)
#   sym17, slipPT17, cone17   the variants above, at writePrecision 17
#   <case>e             the 17-digit case written EVERY STEP for its first steps (writeControl timeStep, interval 1;
#                       endTime 0.002 = 20 steps, obliqueShock 0.05 = 16): before the shocks amplify round-off
#   rs7                 obliqueShock RESTARTED from OpenFOAM's own t = 7 (the reflected shock standing on the
#                       symmetryPlane), written every step (adjustable, 0.0025) to 7.04: the developed flow, step by step
# REFERENCE  OpenFOAM serial on every case, and OpenFOAM's own spread on every 17-digit one: OpenFOAM decomposed in two
#   (decomposePar simple (2 1 1)) and in four (scotch, the partition frozen: tests/rho_central_gate_lib.sh), mpirun, reconstructPar; the spread at a write time is the larger
#   of the two distances from serial. THESE RUNS AMPLIFY ROUND-OFF. Mach 5 and Mach 2.9 shocks switch limiters on the
#   last bits, and the e -> T inversion of these non-dimensional cases (T = 1 against Tstd = 298.15: e is -744) carries
#   a unit in the last place of e into T as 6e-14. MEASURED, GB10 2026-11-17, max |diff|/max|ref| against OpenFOAM
#   serial: OpenFOAM in two 1.2e-13 (T) after ONE step, 7e-06 on wedge15Ma5 by 0.14; brae CUDA 1.9e-13 and 7.0e-06.
#   WHY TWO DECOMPOSITIONS: on obliqueShock the two-way split's one processor column lies in smooth flow and its distance
#   DECAYS (2.3e-11 at t = 7) while OpenFOAM in four, scotch, stays at 3.2e-10 (in eight 2.9e-10). brae CUDA is 5.9e-10
#   there and the host arm 2.3e-10; brae's two arms, the same logic with and without fused multiply-adds, are 3.6e-10
#   apart, and restarted from OpenFOAM's own t = 7 every arm is 2.0e-13 from OpenFOAM after one step (rs7; OpenFOAM
#   in four 1.6e-13). Against the two-way split alone the CUDA arm stood at up to 2.9x its bound at t = 5 to 8 (first
#   run, 2026-11-17).
# BOUND  so every field bound is ten times OpenFOAM's own spread for that field at that write time, never below the
#   shockTube gates' 8e-14 (tests/rho_central_shocktube_vs_openfoam.sh, arm F's rule); the max Courant number likewise
#   on OpenFOAM's spread over the steps. The as-shipped arms: max(1e-6, that), their files carrying 6 digits. The
#   envelope was approved with milestone C on the condition that both measurements stay visible: every arm reports
#   itself against each decomposition alone as well (tests/rho_central_gate_lib.sh), the two-way one's failure
#   on obliqueShock included.
# ARMS  CUDA on every case; the host reference on wedge17, oblique17, cone17, rs7 and every early case. Per arm: OpenFOAM's
#   step count and time names; deltaT digit for digit where Time::adjustDeltaT lands it (obliqueShock, adjustable),
#   within BOUND where nothing does (the early obliqueShock, timeStep writes); the max Courant number; the same time
#   directories; U p T rho at every write time; the absolute errors at the last one.
# SHOCKS (tests/rho_central_fields.py shock): the angle and position fitted to the pressure's mid-level crossings, on
#   brae's and OpenFOAM's fields alike, against each other (within ten times OpenFOAM's own spread of the same
#   measure, never below 1e-6 degrees) and against inviscid theory: wedge 24.32 (theta-beta-M), cone 20.03
#   (Taylor-Maccoll), obliqueShock incident 29.00 and reflected 23.28 meeting the wall at x 1.804. Theory is a premise
#   (OpenFOAM's own angle within 1 degree of it, these meshes' resolution), not a bound on brae.
# CPU/GPU  the two arms against each other at every early step (reported) and both against OpenFOAM (asserted).
# CONSERVATION  brae's `diag balance` (totals' change plus the integrated boundary fluxes) within its `diag roundoff`:
#   four units in the last place of every cell's total (|rhoU|'s for each momentum component), every step. The start
#   totals are not the scale here: e is an offset quantity (-744), so a rounding of a cell's rhoE is 2e-13 of a value
#   whose domain total is 0.6.
# STABILITY  rho, p, T > 0 and finite at every write time; how steady the last write is (max change of p since the
#   one before) printed for brae and OpenFOAM.
# CONTROLS (rhoCentralState.cuh), each red against OpenFOAM by more than 100 x its bound:
#   noTransform   U on every transform face the cell value: wedge17 at 0.2, cone17e at its last step
#   laggedRhoB    rhoU_b, rhoE_b on the constraint patches by last step's rho_b: oblique17 at some write time, and
#                 cone17e. NOT red on wedge15Ma5 (measured: inside OpenFOAM's spread): its symmetry planes see only the
#                 uniform stream, where the new and old rho_b agree.
# NOT CLAIMED  viscous flow, any other patch type, a `symmetryPlane` not planar (refused, as OpenFOAM does), 3-D
#   cases, cyclic or processor patches, restart.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BRAE_RCF_BIN:-$ROOT/build/brae_rhoCentralFoam}"
STEPS="$ROOT/tests/rho_central_steps.py"
FIELDS="$ROOT/tests/rho_central_fields.py"
CASES="$ROOT/tests/rho_central_cases.py"
OFBASHRC=${OF_BASHRC:-${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}}
TUT="${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}/compressible/rhoCentralFoam"
[ -x "$BIN" ] || { echo "SKIP: no brae_rhoCentralFoam at $BIN"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: no OpenFOAM v2412 at $OFBASHRC"; exit 77; }
[ -d "$TUT/wedge15Ma5/0" ] && [ -d "$TUT/obliqueShock/0" ] || { echo "SKIP: no wedge15Ma5/obliqueShock at $TUT"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
python3 -c 'import numpy' 2> /dev/null || { echo "SKIP: no numpy"; exit 77; }
W=${BRAE_GATE_OUT:-$(mktemp -d /tmp/brae-rcf-wedge.XXXXXX)}
mkdir -p "$W"
export PYTHONDONTWRITEBYTECODE=1
BOUND=${BRAE_RCF_SOD_BOUND:-8e-14}
fails=0
passes=0
. "$ROOT/tests/rho_central_gate_lib.sh"

# stage <dir> <tutorial> [writePrecision] [variant] [early endTime]
stage()
{
    local d=$1 tut=$2 prec=${3:-} var=${4:-} early=${5:-}
    rm -rf "$d"
    mkdir -p "$d"
    cp -r "$TUT/$tut/system" "$TUT/$tut/constant" "$TUT/$tut/0" "$d/"
    if [ -n "$prec" ]
    then
        sed -i -E "s/^(writePrecision\s+)\S+;/\1$prec;/" "$d/system/controlDict"
        grep -qE "^writePrecision\s+$prec;" "$d/system/controlDict" || { fail "stage $d: writePrecision not set"; return 1; }
    fi
    [ -z "$var" ] || python3 "$CASES" "$var" "$d" || { fail "stage $d: variant $var"; return 1; }
    if [ -n "$early" ]
    then
        sed -i -E "s/^(writeControl\s+)\S+;/\1timeStep;/; s/^(writeInterval\s+)\S+;/\11;/; s/^(endTime\s+)\S+;/\1$early;/" \
            "$d/system/controlDict"
    fi
    of "cd '$d' && blockMesh > log.blockMesh 2>&1" < /dev/null || { fail "stage $d: blockMesh"; return 1; }
}
spread() { echo "$W/of_${1}_np2 $W/of_${1}_np4"; }   # OpenFOAM's own spread: its two decompositions

# ---- the cases and OpenFOAM ---------------------------------------------------------------------------------------
FULL="wedge17 oblique17 sym17 slipPT17 cone17"
EARLY="wedge17e oblique17e sym17e slipPT17e cone17e"
stage "$W/wedge" wedge15Ma5 || exit 1
stage "$W/oblique" obliqueShock || exit 1
stage "$W/wedge17" wedge15Ma5 17 || exit 1
stage "$W/oblique17" obliqueShock 17 || exit 1
stage "$W/sym17" wedge15Ma5 17 symmetry || exit 1
stage "$W/slipPT17" wedge15Ma5 17 slipPT || exit 1
stage "$W/cone17" wedge15Ma5 17 cone || exit 1
stage "$W/wedge17e" wedge15Ma5 17 "" 0.002 || exit 1
stage "$W/oblique17e" obliqueShock 17 "" 0.05 || exit 1
stage "$W/sym17e" wedge15Ma5 17 symmetry 0.002 || exit 1
stage "$W/slipPT17e" wedge15Ma5 17 slipPT 0.002 || exit 1
stage "$W/cone17e" wedge15Ma5 17 cone 0.002 || exit 1
for c in wedge oblique $FULL $EARLY; do copy "$W/$c" "$W/of_$c"; done
for c in wedge oblique $FULL $EARLY; do ofrun "$W/of_$c" & done
wait
# rs7: obliqueShock restarted from OpenFOAM's own t = 7, written at every (trimmed) step
rm -rf "$W/rs7"
mkdir -p "$W/rs7"
cp -r "$W/oblique17/system" "$W/oblique17/constant" "$W/of_oblique17/7" "$W/rs7/"
sed -i -E 's/^(startFrom\s+)\S+;/\1startTime;/; s/^(startTime\s+)\S+;/\17;/; s/^(writeInterval\s+)\S+;/\10.0025;/; s/^(endTime\s+)\S+;/\17.04;/' \
    "$W/rs7/system/controlDict"
[ -d "$W/rs7/7/uniform" ] && grep -qE '^startTime\s+7;' "$W/rs7/system/controlDict" || fail "stage rs7"
rcopy() { rm -rf "$2"; mkdir -p "$2"; cp -r "$1/system" "$1/constant" "$1/7" "$2/"; }
rcopy "$W/rs7" "$W/of_rs7"
ofrun "$W/of_rs7"
REF="$FULL $EARLY rs7"
for c in $REF; do
    if [ "$c" = rs7 ]; then rcopy "$W/rs7" "$W/of_${c}_np2"; rcopy "$W/rs7" "$W/of_${c}_np4"
    else copy "$W/$c" "$W/of_${c}_np2"; copy "$W/$c" "$W/of_${c}_np4"; fi
done
for c in $REF; do ofnp "$W/of_${c}_np2" 2 & done
wait
for c in $REF; do ofnp "$W/of_${c}_np4" 4 & [ "$c" = slipPT17 ] && wait; done
wait
# (the runs went in the background, where a fail() cannot count: each log must have reached its End)
for c in wedge oblique $FULL $EARLY rs7; do grep -qx End "$W/of_$c/log" || fail "OpenFOAM did not finish $c"; done
for c in $REF; do for n in 2 4; do grep -qx End "$W/of_${c}_np$n/log" || fail "OpenFOAM in $n did not finish $c"; done; done

# ---- brae ---------------------------------------------------------------------------------------------------------
bcase() { copy "$W/$1" "$W/$2"; }
for c in wedge oblique $FULL $EARLY; do bcase "$c" "b_$c"; braerun "$W/b_$c"; done
for c in wedge17 oblique17 cone17 $EARLY; do bcase "$c" "h_$c"; braerun "$W/h_$c" BRAE_RHOCENTRALFOAM_ARM=cpu; done
rcopy "$W/rs7" "$W/b_rs7"; braerun "$W/b_rs7"
rcopy "$W/rs7" "$W/h_rs7"; braerun "$W/h_rs7" BRAE_RHOCENTRALFOAM_ARM=cpu
bcase wedge17 cX_wedge17;    braerun "$W/cX_wedge17" BRAE_CONTROL_RCF=noTransform
bcase cone17e cX_cone17e;    braerun "$W/cX_cone17e" BRAE_CONTROL_RCF=noTransform
bcase oblique17 cL_oblique17; braerun "$W/cL_oblique17" BRAE_CONTROL_RCF=laggedRhoB
bcase cone17e cL_cone17e;    braerun "$W/cL_cone17e" BRAE_CONTROL_RCF=laggedRhoB

# ---- premises: the shocks are there, OpenFOAM's own angles near theory -------------------------------------------------
shockline() { python3 "$FIELDS" shock "$1" "$2" "$3" 2> /dev/null; }
field() { sed -n "s/.* $2 \([^ ]*\).*/\1/p" <<< "$1"; }
for spec in "wedge17 0.2 wedge angle beta" "cone17 0.2 cone angle beta" "oblique17 10 oblique incident incident" \
            "oblique17 10 oblique reflected reflected"
do
    read -r c t k m th <<< "$spec"
    o=$(shockline "$W/of_$c" "$t" "$k")
    want=$(python3 "$FIELDS" theory "$k" | sed -n "s/.* $th \([^ ]*\).*/\1/p")
    got=$(field "$o" "$m")
    if python3 -c "import sys; sys.exit(0 if abs(float('${got:-nan}') - float('${want:-nan}')) <= 1.0 else 1)" 2> /dev/null
    then
        ok "premise: OpenFOAM's $c $m shock angle $got degrees, inviscid theory $want (within 1 degree)"
    else
        fail "premise: OpenFOAM's $c $m shock angle ${got:-?} against theory ${want:-?}"
    fi
done

# ---- comparisons (arm, red, balance: tests/rho_central_gate_lib.sh) ---------------------------------------------
arm "S wedge15Ma5 (CUDA, as shipped)"      "$W/b_wedge"   "$W/of_wedge"   "$(spread wedge17)"   1e-6 fixed
arm "S obliqueShock (CUDA, as shipped)"    "$W/b_oblique" "$W/of_oblique" "$(spread oblique17)" 1e-6 exact
for c in $FULL; do
    dm=fixed; [ "$c" = oblique17 ] && dm=exact
    arm "$c (CUDA)" "$W/b_$c" "$W/of_$c" "$(spread $c)" "$BOUND" $dm
done
for c in wedge17 oblique17 cone17; do
    dm=fixed; [ "$c" = oblique17 ] && dm=exact
    arm "$c (host reference)" "$W/h_$c" "$W/of_$c" "$(spread $c)" "$BOUND" $dm
done
for c in $EARLY; do
    dm=fixed; [ "$c" = oblique17e ] && dm=bound
    arm "$c (CUDA, every step)" "$W/b_$c" "$W/of_$c" "$(spread $c)" "$BOUND" $dm
    arm "$c (host reference, every step)" "$W/h_$c" "$W/of_$c" "$(spread $c)" "$BOUND" $dm
done
arm "rs7 obliqueShock from OpenFOAM's t = 7 (CUDA, every step)" "$W/b_rs7" "$W/of_rs7" "$(spread rs7)" "$BOUND" exact
arm "rs7 obliqueShock from OpenFOAM's t = 7 (host reference, every step)" "$W/h_rs7" "$W/of_rs7" "$(spread rs7)" "$BOUND" exact

# ---- CPU against GPU (reported) ---------------------------------------------------------------------------------------
for c in $EARLY rs7; do
    t1=$(times "$W/b_$c" | awk '{print $1}'); tl=$(times "$W/b_$c" | awk '{print $NF}')
    a1=$(python3 "$FIELDS" maxrel "$W/h_$c/$t1" "$W/b_$c/$t1"); al=$(python3 "$FIELDS" maxrel "$W/h_$c/$tl" "$W/b_$c/$tl")
    echo "     CPU against GPU, $c: after one step ${a1#maxrel }, at t = $tl ${al#maxrel }"
done

# ---- shocks: brae's against OpenFOAM's, by the same measure ------------------------------------------------------------
for spec in "wedge17 0.2 wedge angle" "wedge17 0.2 wedge y_at_0.25" "sym17 0.2 wedge angle" "slipPT17 0.2 wedge angle" \
            "cone17 0.2 cone angle" "cone17 0.2 cone y_at_0.25" "oblique17 10 oblique incident" \
            "oblique17 10 oblique reflected" "oblique17 10 oblique impact_x_reflected" "wedge 0.2 wedge angle" \
            "oblique 10 oblique reflected"
do
    read -r c t k m <<< "$spec"
    rc=$c; [ "$c" = wedge ] && rc=wedge17; [ "$c" = oblique ] && rc=oblique17
    gb=$(field "$(shockline "$W/b_$c" "$t" "$k")" "$m")
    go=$(field "$(shockline "$W/of_$c" "$t" "$k")" "$m")
    sp=0
    for r in $(spread $rc)
    do
        gr=$(field "$(shockline "$r" "$t" "$k")" "$m")
        sp=$(python3 -c "print('%.3e' % max(float('$sp'), abs(float('${gr:-nan}') - float('${go:-nan}'))))")
    done
    sb=$(python3 -c "print('%.3e' % max(1e-6, 10*float('$sp')))")
    if python3 -c "import sys; sys.exit(0 if abs(float('${gb:-nan}') - float('${go:-nan}')) <= float('$sb') else 1)" 2> /dev/null
    then
        ok "shock $c $m: brae $gb, OpenFOAM $go (OpenFOAM's own spread $sp): within $sb"
    else
        fail "shock $c $m: brae ${gb:-?}, OpenFOAM ${go:-?} (OpenFOAM's own spread $sp), bound $sb"
    fi
done

# ---- conservation and stability -------------------------------------------------------------------------------------
for d in b_wedge b_oblique $(for c in $FULL; do echo b_$c; done) h_wedge17 h_oblique17 h_cone17; do balance "$d"; done
for c in wedge17 oblique17 cone17; do
    tl=$(times "$W/b_$c" | awk '{print $NF}'); tp=$(times "$W/b_$c" | awk '{print $(NF-1)}')
    sb=$(python3 "$FIELDS" stats "$W/b_$c" "$tl" "$tp" | awk '{print $6}'); so=$(python3 "$FIELDS" stats "$W/of_$c" "$tl" "$tp" | awk '{print $6}')
    echo "     steadiness, $c: max change of p from $tp to $tl, brae $sb, OpenFOAM $so"
done

# ---- controls --------------------------------------------------------------------------------------------------------
red "noTransform, wedge17" "$W/cX_wedge17" "$W/of_wedge17" "$(spread wedge17)"
red "noTransform, cone17e" "$W/cX_cone17e" "$W/of_cone17e" "$(spread cone17e)"
red "laggedRhoB, oblique17" "$W/cL_oblique17" "$W/of_oblique17" "$(spread oblique17)"
red "laggedRhoB, cone17e" "$W/cL_cone17e" "$W/of_cone17e" "$(spread cone17e)"

# ---- performance (reported) ------------------------------------------------------------------------------------------
for c in wedge17 oblique17 cone17; do
    echo "     runtime $c: OpenFOAM $(grep -h '^ExecutionTime' "$W/of_$c/log" | tail -n 1 | awk '{print $3}') s; brae CUDA $(sed -n 's/^diag steps \([^ ]*\) step-seconds \([^ ]*\) per-step-us \([^ ]*\).*/\2 s (\1 steps, \3 us a step)/p' "$W/b_$c/log"), $(sed -n 's/^diag device-memory/device memory/p' "$W/b_$c/log"); host $(sed -n 's/^diag steps [^ ]* step-seconds \([^ ]*\).*/\1 s/p' "$W/h_$c/log")"
done
echo "rho_central_wedge_vs_openfoam: $passes passed, $fails failed (work in $W)"
[ "$fails" = 0 ] && [ "$passes" -ge 127 ]
