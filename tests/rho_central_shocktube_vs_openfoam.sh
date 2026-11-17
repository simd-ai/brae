#!/usr/bin/env bash
# brae_rhoCentralFoam against OpenFOAM v2412 rhoCentralFoam on the shockTube tutorial AS SHIPPED: adjustTimeStep yes,
# maxCo 0.2, maxDeltaT 1, deltaT 1e-6 to start, writeControl adjustable every 0.001 s to endTime 0.007. Milestone B's
# gate: the adaptive time step (setDeltaT.H on the central Courant number, Time::adjustDeltaT's trimming to the write
# times, the accumulating clock) step by step, and the fields at every write time.
#
# CASES  each made by the tutorial's own Allrun steps (0.orig restored, blockMesh, setFields), by OpenFOAM's utilities
#   ship    the tutorial exactly as shipped (writePrecision 6: OpenFOAM's log and files carry 6 digits)
#   ship17  the same with writePrecision 17, the one change: it sets the files' digits and Info's (TimeIO.C:377-389), so
#           both logs print every step's deltaT and Courant numbers at round-trip precision. No numerics change.
#   tad17   ship17 with fluxScheme Tadmor
#   fine17  ship17 on 1000 cells
#   mdt17   ship17 with maxDeltaT 3e-5: the cap BINDS (premise: more steps than ship17, at least 50 of them at the cap)
#   rt17    ship17 with writeControl runTime: nothing trims deltaT to the write times, which land past them
#   ts17    ship17 with writeControl timeStep, writeInterval 20
# ARMS   (each brae against OpenFOAM serial on the same case)
#   S  CUDA on ship     B  CUDA on ship17     H  host reference on ship17     T  CUDA on tad17     F  CUDA on fine17
#   M  CUDA on mdt17    MH host reference on mdt17    R  CUDA on rt17    K  CUDA on ts17
#   (and OpenFOAM decomposed in two on fine17, F's reference: see BOUND). R and K: deltaT is not landed on a write
#   time, so it carries the Courant number's last bits (setDeltaT.H's maxCo/(CoNum + SMALL)); their deltaT is held to
#   BOUND at every step, not to the same digits.
# CHECKS, per arm: the same number of steps; every step's deltaT printed with the same digits (deltaT is landed on the
#   write times by adjustDeltaT, timeToNextWrite/round(nSteps), so it is the same number wherever the rounding agrees);
#   every step's time name the same; the max Courant number within BOUND at every step; the same time directories
#   written; U p T rho at every write time within BOUND (S: within 1e-6, its files' 6 digits).
# PREMISE  OpenFOAM's own step varies: at least 10 distinct deltaT values over a factor of 10, the first one trimmed
#   (not controlDict's 1e-6 times 1.2).
# CONTROLS, CUDA on ship17, each red against OpenFOAM's ship17:
#   untrimmed      setDeltaT.H without Time::adjustDeltaT: the time directories written are not OpenFOAM's
#   laggedCourant  setDeltaT.H fed the Courant number scaled by the step before's deltaT: the deltaT sequence parts
#                  from OpenFOAM's by more than 1e-3 and the fields at 0.007 by more than 100 x BOUND
# CONSERVATION and POSITIVITY as the Sod gate (brae's `diag balance` line; min rho, p, T > 0 at every write).
# BOUND  8e-14 for the 100-cell arms: the Sod gate's (tests/rho_central_sod_vs_openfoam.sh). First run, GB10 2026-11-17:
#   B 3.1e-15 (fields), 4.1e-15 (max Courant number); every deltaT and name identical, 142 steps.
#   F's bound is OpenFOAM's own: this run AMPLIFIES ROUND-OFF. On 1000 cells at maxCo 0.2 the star region carries
#   post-shock oscillations (U 291 to 298 m/s where the exact u* is 293) whose limiters switch on the last bits, and any
#   two runs that differ in rounding part by about three decades over the 7 ms. MEASURED, GB10 2026-11-17: OpenFOAM
#   decomposed in two (decomposePar, mpirun -np 2, reconstructPar; the processor faces' sums are the only change)
#   against OpenFOAM serial: 1.7e-15 at 0.001, 4.0e-13 at 0.004, 6.9e-12 at 0.007; brae's CUDA arm: 4.5e-15, 8.1e-14,
#   6.3e-12; the host reference 1.1e-11 at 0.007. The fixed-deltaT Sod gate's 1000-cell run (maxCo 0.035) does not
#   amplify (1.9e-14 after 7000 steps). So the gate runs that decomposed reference itself, and F's bound at each write
#   time (and for the max Courant number) is ten times OpenFOAM's serial-to-decomposed distance there, never below
#   the Sod gate's 2e-13: the decade rule of the Sod bounds, on OpenFOAM's own spread.
# F'S ABSOLUTE ERRORS  printed at every write time beside its relative bound: the largest |brae - OpenFOAM| per
#   field in its own units, and OpenFOAM decomposed's (tests/rho_central_fields.py absdiff).
# NOT CLAIMED  the mean Courant number (a sum: the device adds in another order; printed, not asserted), restart
#   (uniform/time), function objects' trimming, any other tutorial, patches the waves reach (they do not by 0.007).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BRAE_RCF_BIN:-$ROOT/build/brae_rhoCentralFoam}"
CHECK="$ROOT/tests/check_fields.py"
STEPS="$ROOT/tests/rho_central_steps.py"
SOD="$ROOT/tests/rho_central_sod.py"
FIELDS="$ROOT/tests/rho_central_fields.py"
OFBASHRC=${OF_BASHRC:-${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}}
TUT="${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}/compressible/rhoCentralFoam/shockTube"
[ -x "$BIN" ] || { echo "SKIP: no brae_rhoCentralFoam at $BIN"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: no OpenFOAM v2412 at $OFBASHRC"; exit 77; }
[ -d "$TUT/0.orig" ] || { echo "SKIP: no shockTube tutorial at $TUT"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
python3 -c 'import numpy' 2> /dev/null || { echo "SKIP: no numpy"; exit 77; }
W=${BRAE_GATE_OUT:-$(mktemp -d /tmp/brae-rcf-shocktube.XXXXXX)}
mkdir -p "$W"
export PYTHONDONTWRITEBYTECODE=1
BOUND=${BRAE_RCF_SOD_BOUND:-8e-14}
BOUND_F=${BRAE_RCF_SOD_BOUND_FINE:-2e-13}
fails=0
passes=0
fail()
{
    echo "FAIL $*"
    fails=$((fails + 1))
}
ok()
{
    passes=$((passes + 1))
    echo "ok   $*"
}
of()
{
    bash -c "set +u; source '$OFBASHRC' > /dev/null 2>&1; set -e; $*"
}
le() { python3 -c "import sys; sys.exit(0 if float('${1:-nan}') <= float('$2') else 1)" 2> /dev/null; }
ge() { python3 -c "import sys; sys.exit(0 if float('${1:-nan}') >= float('$2') else 1)" 2> /dev/null; }

# stage <dir> [cells] [writePrecision]: the tutorial's Allrun up to the solver
stage()
{
    rm -rf "$1"
    mkdir -p "$1"
    cp -r "$TUT/system" "$TUT/constant" "$TUT/0.orig" "$1/"
    cp -r "$TUT/0.orig" "$1/0"
    if [ -n "${2:-}" ]
    then
        sed -i -E "s/\(100 1 1\)/($2 1 1)/" "$1/system/blockMeshDict"
        grep -q "($2 1 1)" "$1/system/blockMeshDict" || { fail "stage: the tutorial's blockMeshDict no longer reads (100 1 1)"; return 1; }
    fi
    if [ -n "${3:-}" ]
    then
        sed -i -E "s/^(writePrecision\s+)\S+;/\1$3;/" "$1/system/controlDict"
        grep -qE "^writePrecision\s+$3;" "$1/system/controlDict" || { fail "stage: writePrecision not set"; return 1; }
    fi
    grep -qE '^adjustTimeStep\s+yes;' "$1/system/controlDict" || { fail "stage: the tutorial no longer says adjustTimeStep yes"; return 1; }
    of "cd '$1' && blockMesh > log.blockMesh 2>&1 && setFields > log.setFields 2>&1" < /dev/null \
        || { fail "stage: blockMesh/setFields in $1"; return 1; }
}
copy()   # copy <from> <to> [fluxScheme]
{
    rm -rf "$2"
    cp -r "$1" "$2"
    [ -z "${3:-}" ] || sed -i -E "s/^(fluxScheme\s+)\S+;/\1$3;/" "$2/system/fvSchemes"
}
ofrun()
{
    of "cd '$1' && rhoCentralFoam > log 2>&1" < /dev/null || fail "OpenFOAM rhoCentralFoam in $(basename "$1")"
}
braerun()   # braerun <dir> [env ...]
{
    local d=$1
    shift
    (cd "$d" && env "$@" timeout 600 "$BIN" -case . > log 2> err < /dev/null; echo $? > rc)
    [ "$(cat "$d/rc")" = 0 ] || { fail "$(basename "$d") exited $(cat "$d/rc"): $(head -c 300 "$d/err")"; return 1; }
}
times() { (cd "$1" && ls -d [0-9]* 2> /dev/null | grep -vx 0 | grep -vx '0.orig' | sort -g | tr '\n' ' '); }
dist()   # dist <time dir> <reference time dir>: the worst relative L2 over U p T rho
{
    local out=$1.$(basename "$(dirname "$2")").cmp
    python3 "$CHECK" fields "$1" "$2" 1e300 U p T rho > "$out" 2>&1
    sed -n 's/^worst \([^ ]*\).*/\1/p' "$out"
}

stage "$W/ship" || { echo "rho_central_shocktube_vs_openfoam: $passes passed, $fails failed"; exit 1; }
stage "$W/ship17" "" 17 || { echo "rho_central_shocktube_vs_openfoam: $passes passed, $fails failed"; exit 1; }
stage "$W/fine17" 1000 17 || { echo "rho_central_shocktube_vs_openfoam: $passes passed, $fails failed"; exit 1; }
copy "$W/ship17" "$W/tad17" Tadmor
copy "$W/ship17" "$W/mdt17"
sed -i -E 's/^(maxDeltaT\s+)\S+;/\13e-05;/' "$W/mdt17/system/controlDict"
copy "$W/ship17" "$W/rt17"
sed -i -E 's/^(writeControl\s+)\S+;/\1runTime;/' "$W/rt17/system/controlDict"
copy "$W/ship17" "$W/ts17"
sed -i -E 's/^(writeControl\s+)\S+;/\1timeStep;/; s/^(writeInterval\s+)\S+;/\120;/' "$W/ts17/system/controlDict"
grep -qE '^maxDeltaT\s+3e-05;' "$W/mdt17/system/controlDict" && grep -qE '^writeControl\s+runTime;' "$W/rt17/system/controlDict" \
    && grep -qE '^writeInterval\s+20;' "$W/ts17/system/controlDict" || fail "stage: the mdt17/rt17/ts17 edits"
for c in ship ship17 tad17 fine17 mdt17 rt17 ts17; do copy "$W/$c" "$W/of_$c"; ofrun "$W/of_$c"; done
# F's reference: OpenFOAM decomposed in two, the summation order at the processor faces its only change
copy "$W/fine17" "$W/of_fine17_np2"
printf '%s\n' 'FoamFile { version 2.0; format ascii; class dictionary; object decomposeParDict; }' \
    'numberOfSubdomains 2;' 'method simple;' 'coeffs { n (2 1 1); }' > "$W/of_fine17_np2/system/decomposeParDict"
of "cd '$W/of_fine17_np2' && decomposePar > log.decomposePar 2>&1 && mpirun -np 2 rhoCentralFoam -parallel > log 2>&1 \
    && reconstructPar > log.reconstructPar 2>&1" < /dev/null || fail "OpenFOAM decomposed run (decomposePar, mpirun -np 2, reconstructPar)"
copy "$W/ship" "$W/bS"
copy "$W/ship17" "$W/bB"
copy "$W/ship17" "$W/bH"
copy "$W/tad17" "$W/bT"
copy "$W/fine17" "$W/bF"
copy "$W/mdt17" "$W/bM"
copy "$W/mdt17" "$W/bMH"
copy "$W/rt17" "$W/bR"
copy "$W/ts17" "$W/bK"
copy "$W/ship17" "$W/cX"
copy "$W/ship17" "$W/cL"
braerun "$W/bS"
braerun "$W/bB"
braerun "$W/bH" BRAE_RHOCENTRALFOAM_ARM=cpu
braerun "$W/bT"
braerun "$W/bF"
braerun "$W/bM"
braerun "$W/bMH" BRAE_RHOCENTRALFOAM_ARM=cpu
braerun "$W/bR"
braerun "$W/bK"
braerun "$W/cX" BRAE_CONTROL_RCF=untrimmed
braerun "$W/cL" BRAE_CONTROL_RCF=laggedCourant

# premise: OpenFOAM's step varies
read -r _ n _ distinct _ dmin _ dmax _ dfirst <<< "$(python3 "$STEPS" summary "$W/of_ship17/log")"
if python3 -c "import sys; sys.exit(0 if int('$distinct') >= 10 and float('$dmax') > 10*float('$dmin') and float('$dfirst') != 1.2e-6 else 1)" 2> /dev/null
then
    ok "premise: OpenFOAM's shipped run takes $n steps, $distinct distinct deltaT from $dmin to $dmax; the first, $dfirst, is trimmed to the write time"
else
    fail "premise: OpenFOAM's deltaT does not vary as an adaptive run's does ($n steps, $distinct distinct, $dmin to $dmax, first $dfirst)"
fi

# premises of the three B-gap cases: the cap binds; runTime's writes land off the interval; timeStep's every 20 steps
read -r _ n17 _ <<< "$(python3 "$STEPS" summary "$W/of_ship17/log")"
read -r _ nm _ _ _ _ _ dmaxm _ <<< "$(python3 "$STEPS" summary "$W/of_mdt17/log")"
atcap=$(grep -E '^deltaT = ' "$W/of_mdt17/log" | awk '$3 >= 0.95*3e-5 {n++} END {print n+0}')
if [ "${nm:-0}" -gt "${n17:-0}" ] && [ "$atcap" -ge 50 ]
then
    ok "premise M: maxDeltaT 3e-5 binds: OpenFOAM takes $nm steps (as shipped $n17), $atcap of them at the cap (largest $dmaxm, trimmed)"
else
    fail "premise M: maxDeltaT does not bind ($nm steps against $n17, $atcap at the cap)"
fi
tr_=$(times "$W/of_rt17")
if python3 -c "
import sys
t = [float(x) for x in '$tr_'.split()]
sys.exit(0 if len(t) >= 5 and all(abs(x/1e-3 - round(x/1e-3)) > 1e-6 for x in t) else 1)" 2> /dev/null
then
    ok "premise R: OpenFOAM's runTime writes land past the interval, untrimmed: $tr_"
else
    fail "premise R: runTime write times '$tr_'"
fi
read -r _ nk _ <<< "$(python3 "$STEPS" summary "$W/of_ts17/log")"
if [ "$(times "$W/of_ts17" | wc -w)" = "$((nk / 20))" ] && [ $((nk % 20)) = 0 ]
then
    ok "premise K: OpenFOAM writes every 20 steps: $(times "$W/of_ts17" | wc -w) times over $nk steps"
else
    fail "premise K: $(times "$W/of_ts17" | wc -w) times over ${nk:-?} steps"
fi

# arm <name> <brae dir> <OF dir> <bound> <field bound> [<OpenFOAM reference run>|-] [exact|bound]: with a reference,
# each bound is ten times the reference's distance from <OF dir> at that write time (and over the steps for the
# Courant number), never below the given one. `bound`: every deltaT within <bound> instead of the same digits
arm()
{
    local name=$1 b=$2 o=$3 bound=$4 fbound=$5 ref=${6:-} dtmode=${7:-exact}
    [ "$ref" = - ] && ref=""
    python3 "$STEPS" compare "$b/log" "$o/log" > "$b/steps.cmp" 2>&1
    read -r nb no <<< "$(sed -n 's/^steps //p' "$b/steps.cmp")"
    read -r same wdt wdts <<< "$(sed -n 's/^deltaT //p' "$b/steps.cmp")"
    read -r names <<< "$(sed -n 's/^names //p' "$b/steps.cmp")"
    read -r wco wcos <<< "$(sed -n 's/^maxCo //p' "$b/steps.cmp")"
    read -r wme wmes <<< "$(sed -n 's/^meanCo //p' "$b/steps.cmp")"
    if [ -n "$ref" ]
    then
        python3 "$STEPS" compare "$ref/log" "$o/log" > "$b/ref.steps.cmp" 2>&1
        local rco
        read -r rco _ <<< "$(sed -n 's/^maxCo //p' "$b/ref.steps.cmp")"
        bound=$(python3 -c "print('%.3e' % max(float('$bound'), 10*float('${rco:-nan}')))")
    fi
    local dtok=0 dtnote="every deltaT and every time name the same"
    [ "$same" = "$nb" ] && dtok=1
    if [ "$dtmode" = bound ]
    then
        dtok=0
        le "$wdt" "$bound" && dtok=1
        dtnote="every time name the same, every deltaT within $bound (worst $wdt at step $wdts; the same digits at $same)"
    fi
    if [ -n "$nb" ] && [ "$nb" = "$no" ] && [ "$dtok" = 1 ] && [ "$names" = "$nb" ]
    then
        ok "$name: $nb steps as OpenFOAM's; $dtnote"
    else
        fail "$name: ${nb:-?} steps against OpenFOAM's ${no:-?}; deltaT the same at ${same:-?} (worst ${wdt:-?} at step ${wdts:-?}), names at ${names:-?}"
    fi
    if le "$wco" "$bound"
    then
        ok "$name: max Courant number within $bound at every step (worst $wco at step $wcos; mean $wme at $wmes, not asserted)"
    else
        fail "$name: max Courant number ${wco:-?} at step ${wcos:-?}, above $bound"
    fi
    local tb to
    tb=$(times "$b"); to=$(times "$o")
    if [ "$tb" = "$to" ] && [ -n "$to" ]
    then
        ok "$name: the same time directories written: $tb"
    else
        fail "$name: wrote '$tb', OpenFOAM '$to'"
        return
    fi
    local worst=0 where="" v over="" detail="" bt r
    for t in $to
    do
        v=$(dist "$b/$t" "$o/$t")
        [ -n "$v" ] || { fail "$name: no comparison at $t"; return; }
        bt=$fbound
        if [ -n "$ref" ]
        then
            r=$(dist "$ref/$t" "$o/$t")
            bt=$(python3 -c "print('%.3e' % max(float('$fbound'), 10*float('${r:-nan}')))")
            detail="$detail $t: $v (OpenFOAM decomposed ${r:-?}, bound $bt);"
        fi
        le "$v" "$bt" || over="$over $t ($v > $bt)"
        if python3 -c "import sys; sys.exit(0 if float('$v') > float('$worst') else 1)"; then worst=$v; where=$t; fi
    done
    [ -z "$detail" ] || echo "     $name, every write time:$detail"
    local bnote=$fbound
    [ -z "$ref" ] || bnote="ten times OpenFOAM's own spread at each write time"
    if [ -z "$over" ]
    then
        ok "$name: U p T rho against OpenFOAM at $(echo $to | wc -w) write times: worst $worst (t = $where, $(sed -n 's/^worst [^ ]* //p' "$b/$where.$(basename "$o").cmp")) within ${bnote}"
    else
        fail "$name: fields above the bound at$over"
        sed 's/^/     /' "$b/$where.$(basename "$o").cmp"
    fi
}
arm "S (CUDA, as shipped)" "$W/bS" "$W/of_ship" "$BOUND" 1e-6
arm "B (CUDA, writePrecision 17)" "$W/bB" "$W/of_ship17" "$BOUND" "$BOUND"
arm "H (host reference, writePrecision 17)" "$W/bH" "$W/of_ship17" "$BOUND" "$BOUND"
arm "T (CUDA, Tadmor)" "$W/bT" "$W/of_tad17" "$BOUND" "$BOUND"
arm "F (CUDA, 1000 cells)" "$W/bF" "$W/of_fine17" "$BOUND_F" "$BOUND_F" "$W/of_fine17_np2"
echo "     F absolute errors, max |brae - OpenFOAM| per field (OpenFOAM decomposed's in brackets):"
for t in $(times "$W/of_fine17")
do
    python3 "$FIELDS" absdiff "$W/bF/$t" "$W/of_fine17/$t" > "$W/bF/$t.abs" 2>&1
    python3 "$FIELDS" absdiff "$W/of_fine17_np2/$t" "$W/of_fine17/$t" > "$W/bF/$t.ref.abs" 2>&1
    echo "       t = $t:$(paste -d' ' "$W/bF/$t.abs" "$W/bF/$t.ref.abs" | awk '{printf " %s %s [%s]", $2, $3, $8}') (U m/s, p Pa, T K, rho kg/m3)"
done
arm "M (CUDA, maxDeltaT 3e-5)" "$W/bM" "$W/of_mdt17" "$BOUND" "$BOUND"
arm "MH (host reference, maxDeltaT 3e-5)" "$W/bMH" "$W/of_mdt17" "$BOUND" "$BOUND"
arm "R (CUDA, writeControl runTime)" "$W/bR" "$W/of_rt17" "$BOUND" "$BOUND" - bound
arm "K (CUDA, writeControl timeStep 20)" "$W/bK" "$W/of_ts17" "$BOUND" "$BOUND" - bound

# controls
tx=$(times "$W/cX"); to=$(times "$W/of_ship17")
if [ "$tx" != "$to" ]
then
    ok "control (untrimmed) red: it wrote '$tx' where OpenFOAM wrote '$to'"
else
    fail "control (untrimmed): wrote OpenFOAM's time directories"
fi
python3 "$STEPS" compare "$W/cL/log" "$W/of_ship17/log" > "$W/cL/steps.cmp" 2>&1
read -r _ wdt wdts <<< "$(sed -n 's/^deltaT //p' "$W/cL/steps.cmp")"
v=$(dist "$W/cL/0.007" "$W/of_ship17/0.007")
if python3 -c "import sys; sys.exit(0 if float('${wdt:-0}') > 1e-3 and float('${v:-0}') > 100*$BOUND else 1)" 2> /dev/null
then
    ok "control (laggedCourant) red: deltaT parts by $wdt (step $wdts), fields at 0.007 by $v"
else
    fail "control (laggedCourant): deltaT ${wdt:-?}, fields ${v:-?}: not red"
fi

# conservation and positivity
for d in bS bB bH bT bF bM bMH bR bK
do
    bal=$(sed -n 's/.*relative to the start totals: mass \([^ ]*\) energy \([^)]*\)).*/\1 \2/p' "$W/$d/log")
    mom=$(sed -n 's/^diag balance .* momentum \([^ ]*\) .*/\1/p' "$W/$d/log")
    imp=$(sed -n 's/^diag boundary-outflow .* momentum \([^ ]*\) .*/\1/p' "$W/$d/log")
    if python3 -c "
import sys
m, e = (float(x) for x in '$bal'.split())
r = abs(float('$mom'))/abs(float('$imp'))
print('     %s: mass %.1e, energy %.1e relative; momentum %.1e of the end-wall impulse %.6g' % ('$d', m, e, r, abs(float('$imp'))))
sys.exit(0 if m <= 1e-12 and e <= 1e-12 and r <= 1e-12 else 1)"
    then
        ok "$d conserves mass and energy, and balances momentum against the boundary, to 1e-12"
    else
        fail "$d conservation: '$bal' momentum ${mom:-none} impulse ${imp:-none}"
    fi
    neg=0
    for t in $(times "$W/$d")
    do
        python3 "$SOD" analyse "$W/$d" "$t" 2> /dev/null | awk '$1 == "min" { exit !($2 > 0 && $3 > 0 && $4 > 0) }' || neg=1
    done
    if [ "$neg" = 0 ]; then ok "$d positivity: rho, p, T > 0 at every written time"; else fail "$d: a non-positive rho, p or T"; fi
done

echo "     runtime: OpenFOAM $(grep -h ExecutionTime "$W/of_ship17/log" | tail -n 1 | cut -c1-40) (100 cells), $(grep -h ExecutionTime "$W/of_fine17/log" | tail -n 1 | cut -c1-40) (1000)"
for d in bB bH bF; do echo "     brae $d: $(sed -n 's/^diag steps //p' "$W/$d/log") $(sed -n 's/^diag device-memory/device memory/p' "$W/$d/log")"; done
echo "rho_central_shocktube_vs_openfoam: $passes passed, $fails failed (work in $W)"
[ "$fails" = 0 ] && [ "$passes" -ge 60 ]
