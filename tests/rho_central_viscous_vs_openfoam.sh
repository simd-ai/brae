#!/usr/bin/env bash
# brae_rhoCentralFoam against OpenFOAM v2412 rhoCentralFoam with mu > 0 (milestone G): the viscous correctors
# (rhoCentralFoam.C:218-219 tauMC, :234-244 the U corrector, :246-253 sigmaDotU, :273-282 the e corrector), laminar,
# constTransport, single GPU, static mesh.
#
# CASES (tests/rho_central_viscous.py; the tube is shockTube's ten times as long, 1000 cells, R = 1, gamma = 1.4)
#   shear       momentum diffusion alone: Stokes' first problem, U_y = +-0.1 across x = 0 on the tube one cell high
#               with its y faces a zeroGradient patch, p = T = 1; mu 0.01, Pr 1; t = 1
#   conduction  energy diffusion alone: the tube at rest, p = 1, T = 2 | 1; mu 0.01, Pr 1; t = 1
#   sod         Toro's test 1 with mu 0.002, Pr 1; t = 2
#   oblique     obliqueShock as shipped (2-D, Mach 2.9, 1800 cells, to t = 10) with mu 0.01, Pr 0.72
#   sod_conv    sod with both correctors converged: tolerance 1e-13, relTol 0
#   every case: fvSolution's `h` entry read for e too ("(e|h)"), writePrecision 17, adjustTimeStep as shipped
# LINEAR SOLVES, DECLARED BEFORE ANY COMPARISON (2026-11-18): the cases' fvSolution as the tutorials ship it -- U:
#   smoothSolver, GaussSeidel, nSweeps 2, tolerance 1e-9, relTol 0.01; e: the same with tolerance 1e-10, relTol 0 (the
#   tutorials' `h` entry) -- and sod_conv's 1e-13 and 0 for both. brae sweeps in OpenFOAM's sequential order, so the
#   requirement is OpenFOAM serial's: the same solves in the same order and the same iteration count at EVERY solve.
#   The residuals are reported, not bounded: a residual is sum|b - A x|, a difference of near-equal numbers whose last
#   digits round-off moves (the median and the worst relative difference are printed).
# REFERENCE  OpenFOAM serial; its spread in two (simple) and in four (the frozen scotch partitions; shear1000 is
#   shockTube1000's, the same cells) -- tests/rho_central_gate_lib.sh's envelope, unchanged -- and, a DIAGNOSTIC that
#   bounds nothing, OpenFOAM serial started with p one ulp up.
# CHECKS (the CUDA arm; the host reference arm refuses mu > 0, checked)
#   PREMISE      OpenFOAM with mu > 0 differs from OpenFOAM with mu = 0 by more than 1e-6 at the last write
#   per case     steps, every deltaT and name, Courant, directories, U p T rho at every write time (arm)
#   SOLVES       the same solves in the same order; every iteration count OpenFOAM serial's
#   FLUXES       the viscous stress mu*dU/dn and heat flux alphaEff*Cv*dT/dn on every internal face (the same arithmetic
#                for every run), at every write time within ten times OpenFOAM's own spread of them (floor BOUND)
#   CONSERVATION (acceptance, decided by the user 2026-11-18) the residual-accounted discrete balance, SOLVE-
#                CONSERVATION below, is the round-off closure test of an implicit linear solve; positivity is checked.
#                The RAW physical imbalance is always reported beside it, with its original round-off allowance, as a
#                historical result (`hist`, counted apart, neither pass nor FAIL): milestone G's five original checks
#                (FOUND ON THE FIRST RUN, 2026-11-18: momentum and energy do not close to round-off, because an implicit
#                corrector solved to a tolerance leaves b - A x, and the totals move by -deltaT*sum(b - A x), OpenFOAM's
#                algorithm the same; red, tolerance-limited, until the decision). brae prints the remainder
#                (`diag solve-residual`), and
#   SOLVE-CONSERVATION  momentum and energy balance PLUS the solves' remainder within the same round-off; it rests on
#                IDENTITY (the remainder is the solves' own, signs, volumes and patches checked) and TOLERANCE (tighter
#                solves lower the raw imbalance)
#   IDENTITY     (milestone H, declared before it ran on a gate) every U and e solve satisfies
#                sum_c rho V (x - x_p) = deltaT*(sum of its patch fluxes [+ Sf & tauMC_b] - sum(b - A x)) to within 1e-12
#                of sum_c rho V |U| (|e|): signs, volume weights and patch contributions, one solve at a time
#                (`diag solve-identity`; round-off is ~1e-16). Reported: the remainder signed and unsigned (sum|b - A x|).
#   TOLERANCE    (milestone H) Sod at four solve settings, the tutorials' (U relTol 0.01), A (U relTol 1e-3, e
#                tolerance 1e-11), B (U 1e-11, e 1e-12, relTol 0) and sod_conv's (1e-13): each tightening lowers the raw
#                momentum and energy imbalance unless it is already within brae's round-off allowance (`diag roundoff`):
#                there nothing is left to remove (fixed after a trial, before any gate ran it: energy at B, 2.7e-10, sat
#                below its 2.3e-08 allowance and moved to 3.7e-10 at sod_conv)
#   EXACT        (reported) shear and conduction against the erf solutions of their constant-density limits
#   MEMORY, RUNTIME (reported)
# CONTROLS (rhoCentralState.cuh), red against OpenFOAM by more than 100 x the bound:
#   noTauMC      tauMC left out (U corrector and sigmaDotU): sod, oblique, sod_conv
#   noSigmaDotU  - fvc::div(sigmaDotU) left out of the rhoE equation: shear, sod, sod_conv
#   (with the tutorials' relTol 0.01 OpenFOAM's decomposed runs part from serial by 1e-7 to 1e-3, so the envelope and
#   the controls' margins are wide there; sod_conv's converged solves keep that spread at round-off)
# NOT CLAIMED  turbulence (refused), the host reference arm (refused), non-orthogonal correction beyond the corrected
#   scheme's explicit part on these meshes, 3-D, multi-GPU. sutherland and wedge patches with mu > 0, refused when this
#   gate was written, are milestone H's (tests/rho_central_nozzle_vs_openfoam.sh).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BRAE_RCF_BIN:-$ROOT/build/brae_rhoCentralFoam}"
STEPS="$ROOT/tests/rho_central_steps.py"
FIELDS="$ROOT/tests/rho_central_fields.py"
VISC="$ROOT/tests/rho_central_viscous.py"
OFBASHRC=${OF_BASHRC:-${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}}
TUT="${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}/compressible/rhoCentralFoam"
[ -x "$BIN" ] || { echo "SKIP: no brae_rhoCentralFoam at $BIN"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: no OpenFOAM v2412 at $OFBASHRC"; exit 77; }
[ -d "$TUT/shockTube/0.orig" ] && [ -d "$TUT/obliqueShock/0" ] || { echo "SKIP: no shockTube/obliqueShock at $TUT"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
W=${BRAE_GATE_OUT:-$(mktemp -d /tmp/brae-rcf-viscous.XXXXXX)}
mkdir -p "$W"
export PYTHONDONTWRITEBYTECODE=1
BOUND=${BRAE_RCF_SOD_BOUND:-8e-14}
fails=0
passes=0
. "$ROOT/tests/rho_central_gate_lib.sh"
J=${BRAE_GATE_JOBS:-10}
throttle() { while [ "$(jobs -rp | wc -l)" -ge "$J" ]; do sleep 1; done; }

# ---- the cases ------------------------------------------------------------------------------------------------------
stage()   # stage <dir> <kind>
{
    local d=$1 k=$2
    rm -rf "$d"
    mkdir -p "$d"
    if [ "$k" = oblique ]
    then cp -r "$TUT/obliqueShock/system" "$TUT/obliqueShock/constant" "$TUT/obliqueShock/0" "$d/"
    else cp -r "$TUT/shockTube/system" "$TUT/shockTube/constant" "$TUT/shockTube/0.orig" "$d/"; cp -r "$TUT/shockTube/0.orig" "$d/0"
    fi
    python3 "$VISC" setup "$k" "$d" || { fail "stage $(basename "$d"): setup"; return 1; }
    of "cd '$d' && blockMesh > log.blockMesh 2>&1" < /dev/null || fail "stage $(basename "$d"): blockMesh"
    [ "$k" = sod ] && { of "cd '$d' && setFields > log.setFields 2>&1" < /dev/null || fail "stage $(basename "$d"): setFields"; }
    return 0
}
CASES="shear conduction sod oblique sod_conv"
kindof() { case $1 in sod_conv) echo sod ;; *) echo "$1" ;; esac; }
for c in $CASES; do stage "$W/$c" "$(kindof "$c")"; done
# sod_conv: both correctors converged
sed -i -E 's/^(\s*tolerance\s+)1e-09;/\11e-13;/; s/^(\s*tolerance\s+)1e-10;/\11e-13;/; s/^(\s*relTol\s+)0\.01;/\10;/' "$W/sod_conv/system/fvSolution"
grep -qE "relTol\s+0\.01" "$W/sod_conv/system/fvSolution" && fail "stage sod_conv: relTol"
# the tolerance sweep's two middle points (brae only: they measure brae's own conservation)
for t in A B; do copy "$W/sod" "$W/sod_tol$t"; done
sed -i -E 's/^(\s*relTol\s+)0\.01;/\11e-03;/; s/^(\s*tolerance\s+)1e-10;/\11e-11;/' "$W/sod_tolA/system/fvSolution"
sed -i -E 's/^(\s*relTol\s+)0\.01;/\10;/; s/^(\s*tolerance\s+)1e-09;/\11e-11;/; s/^(\s*tolerance\s+)1e-10;/\11e-12;/' "$W/sod_tolB/system/fvSolution"
# the premise's twins: the same cases with mu = 0
for c in shear conduction sod oblique; do
    copy "$W/$c" "$W/inv_$c"
    sed -i -E 's/^(\s*mu\s+)[^;]+;/\10;/' "$W/inv_$c/constant/thermophysicalProperties"
done
mupr() { sed -n 's/^\s*mu\s\+\([^;]*\);.*/\1/p' "$W/$1/constant/thermophysicalProperties"; }
prof() { sed -n 's/^\s*Pr\s\+\([^;]*\);.*/\1/p' "$W/$1/constant/thermophysicalProperties"; }

# ---- OpenFOAM ---------------------------------------------------------------------------------------------------------
for c in $CASES; do copy "$W/$c" "$W/of_$c"; copy "$W/$c" "$W/of_${c}_np2"; copy "$W/$c" "$W/of_${c}_np4"; done
for c in $CASES; do throttle; ofrun "$W/of_$c" & done
for c in shear conduction sod oblique; do throttle; ofrun "$W/inv_$c" & done
for c in $CASES; do throttle; ofulp "$W/$c" "$W/of_${c}_ulp" 0 & done
wait
for c in $CASES; do throttle; ofnp "$W/of_${c}_np2" 2 & done
wait
for c in $CASES; do throttle; ofnp "$W/of_${c}_np4" 4 & done
wait
for c in $CASES; do for r in "" _np2 _np4; do grep -qx End "$W/of_$c$r/log" || fail "OpenFOAM${r:+ $r} did not finish $c"; done; done

# ---- brae -------------------------------------------------------------------------------------------------------------
for c in $CASES sod_tolA sod_tolB; do copy "$W/$c" "$W/b_$c"; braerun "$W/b_$c"; done
copy "$W/sod" "$W/cT_sod";         ctlrun "$W/cT_sod" BRAE_CONTROL_RCF=noTauMC
copy "$W/oblique" "$W/cT_oblique"; ctlrun "$W/cT_oblique" BRAE_CONTROL_RCF=noTauMC
copy "$W/shear" "$W/cS_shear";     ctlrun "$W/cS_shear" BRAE_CONTROL_RCF=noSigmaDotU
copy "$W/sod" "$W/cS_sod";         ctlrun "$W/cS_sod" BRAE_CONTROL_RCF=noSigmaDotU
# the same two on sod_conv, whose converged solves leave OpenFOAM's own spread at round-off (the envelope tight)
copy "$W/sod_conv" "$W/cT_sod_conv"; ctlrun "$W/cT_sod_conv" BRAE_CONTROL_RCF=noTauMC
copy "$W/sod_conv" "$W/cS_sod_conv"; ctlrun "$W/cS_sod_conv" BRAE_CONTROL_RCF=noSigmaDotU
# the host reference arm refuses mu > 0
copy "$W/sod" "$W/h_sod"
(cd "$W/h_sod" && BRAE_RHOCENTRALFOAM_ARM=cpu "$BIN" -case . > log 2>&1 < /dev/null; echo $? > rc)
if [ "$(cat "$W/h_sod/rc")" != 0 ] && grep -q "host reference arm .* runs inviscid cases only" "$W/h_sod/log"
then ok "the host reference arm refuses mu > 0 by name: $(grep -o 'the host reference arm[^;]*' "$W/h_sod/log" | head -n 1)"
else fail "host reference arm with mu > 0: exit $(cat "$W/h_sod/rc")"; fi

# ---- premises: the viscous terms change OpenFOAM's solution ------------------------------------------------------------
for c in shear conduction sod oblique; do
    tl=$(times "$W/of_$c" | awk '{print $NF}')
    v=$(python3 "$FIELDS" maxrel "$W/of_$c/$tl" "$W/inv_$c/$tl" | awk '{print $2}')
    if python3 -c "import sys; sys.exit(0 if float('${v:-0}') > 1e-6 else 1)"
    then ok "premise: OpenFOAM's $c with mu $(mupr "$c") differs from its mu = 0 run by $v at t = $tl (the viscous terms act)"
    else fail "premise: OpenFOAM's $c is $v from its mu = 0 run: the viscous terms change nothing"; fi
done

# ---- comparisons ------------------------------------------------------------------------------------------------------
spread() { echo "$W/of_${1}_np2 $W/of_${1}_np4"; }
for c in $CASES; do
    arm "$c (CUDA)" "$W/b_$c" "$W/of_$c" "$(spread "$c")" "$BOUND" exact
    # the solves
    k=$(cmpdir "$W/b_$c")
    python3 "$STEPS" solves "$W/b_$c/log" "$W/of_$c/log" > "$k/solves.cmp" 2>&1
    read -r nb no same <<< "$(sed -n 's/^solves //p' "$k/solves.cmp")"
    read -r iok ifirst <<< "$(sed -n 's/^iterations //p' "$k/solves.cmp")"
    read -r iw iws iwf imed <<< "$(sed -n 's/^initial //p' "$k/solves.cmp")"
    if [ -n "$nb" ] && [ "$nb" = "$no" ] && [ "$same" = 1 ] && [ "$nb" -gt 0 ]
    then ok "$c (CUDA): the same $nb solves as OpenFOAM's, in the same order ($(grep -oE 'Solving for [A-Za-z]+' "$W/of_$c/log" | sort | uniq -c | awk '{printf "%s%s %s", (NR>1?", ":""), $1, $4}'))"
    else fail "$c (CUDA): solves ${nb:-?} against ${no:-?}, order the same: ${same:-?}"; fi
    if [ "${iok:-0}" = "$nb" ] && [ "${ifirst:-1}" = 0 ]
    then ok "$c (CUDA): every iteration count OpenFOAM serial's ($nb solves, $(grep -oE 'No Iterations [0-9]+' "$W/of_$c/log" | awk '{s+=$3; if ($3>m) m=$3} END {printf "%d sweeps in all, at most %d", s, m}'))"
    else fail "$c (CUDA): iteration counts differ from solve ${ifirst:-?} (${iok:-?} of $nb the same)"; fi
    echo "     $c (CUDA): initial residuals against OpenFOAM serial's: median relative difference $imed, worst $iw (solve $iws, $iwf)"
    for r in np2 np4 ulp; do
        python3 "$STEPS" solves "$W/of_${c}_$r/log" "$W/of_$c/log" > "$k/solves.$r" 2>&1
        echo "     $c: OpenFOAM's own $r against serial: $(sed -n 's/^iterations \([0-9]*\) .*/\1/p' "$k/solves.$r") of $(sed -n 's/^solves \([0-9]*\) .*/\1/p' "$k/solves.$r") iteration counts the same, initial residuals median $(awk '/^initial/ {print $5}' "$k/solves.$r"), worst $(awk '/^initial/ {print $2}' "$k/solves.$r")"
    done
    # the viscous fluxes
    mu=$(mupr "$c"); pr=$(prof "$c"); over=""; worst=-1; wl=""
    for t in $(times "$W/of_$c"); do
        read -r _ _ sb _ hb <<< "$(python3 "$VISC" fluxes "$W/b_$c" "$W/of_$c" "$t" "$mu" "$pr")"
        read -r _ _ s2 _ h2 <<< "$(python3 "$VISC" fluxes "$W/of_${c}_np2" "$W/of_$c" "$t" "$mu" "$pr")"
        read -r _ _ s4 _ h4 <<< "$(python3 "$VISC" fluxes "$W/of_${c}_np4" "$W/of_$c" "$t" "$mu" "$pr")"
        for q in "stress $sb $s2 $s4" "heat $hb $h2 $h4"; do
            read -r nm vb v2 v4 <<< "$q"
            bd=$(mx "$BOUND" "$(python3 -c "print(10*max(float('$v2'), float('$v4')))")")
            le "$vb" "$bd" || over="$over $t:$nm($vb>$bd)"
            r=$(python3 -c "print('%.3g' % (float('$vb')/float('$bd')))")
            python3 -c "import sys; sys.exit(0 if float('$r') > float('$worst') else 1)" && { worst=$r; wl="$nm $vb at t = $t (OpenFOAM in two $v2, in four $v4)"; }
        done
    done
    if [ -z "$over" ]; then ok "$c (CUDA): viscous stress and heat flux within ten times OpenFOAM's own spread at every write time; closest ($worst of the bound): $wl"
    else fail "$c (CUDA): viscous fluxes above the bound at$over"; fi
done

# ---- the exact solutions (reported) --------------------------------------------------------------------------------------
for c in shear conduction; do
    for t in $(times "$W/of_$c"); do
        echo "     exact, $c at t = $t: OpenFOAM $(python3 "$VISC" exact "$c" "$W/of_$c" "$t" | cut -d' ' -f2-); brae $(python3 "$VISC" exact "$c" "$W/b_$c" "$t" | cut -d' ' -f2-)"
    done
done

# ---- conservation and stability --------------------------------------------------------------------------------------------
for c in $CASES; do balance "b_$c" raw; done   # the raw imbalance: historical, reported (see CONSERVATION)

# the balance once the correctors' unconserved remainder is counted (see SOLVE-CONSERVATION above), and the identity,
# solve by solve (tests/rho_central_gate_lib.sh: vbalance, identity)
for c in $CASES; do vbalance "b_$c"; done
for c in $CASES; do identity "b_$c"; done
# the tolerance sweep
sweep=""
for c in sod sod_tolA sod_tolB sod_conv; do
    sweep="$sweep $(sed -n 's/^diag balance mass [^ ]* energy \([^ ]*\) momentum \([^ ]*\) .*/\2:\1/p' "$W/b_$c/log"):$(sed -n 's/^diag roundoff mass [^ ]* energy \([^ ]*\) momentum \([^ ]*\) .*/\2:\1/p' "$W/b_$c/log")"
    echo "     tolerance sweep, $c: raw imbalance $(sed -n 's/^diag balance mass [^ ]* energy \([^ ]*\) momentum \([^ ]*\) .*/momentum_x \2, energy \1/p' "$W/b_$c/log"); remainder $(sed -n 's/^diag solve-residual-l1 momentum \([^ ]*\) [^ ]* [^ ]* energy \([^ ]*\) .*/unsigned momentum_x \1, energy \2/p' "$W/b_$c/log")"
done
if python3 -c "
import sys
p = [tuple(abs(float(y)) for y in x.split(':')) for x in '$sweep'.split()]   # momentum, energy, their allowances
ok = len(p) == 4 and all((p[i + 1][j] < p[i][j] or p[i + 1][j] <= p[i + 1][j + 2]) for i in range(3) for j in (0, 1))
sys.exit(0 if ok else 1)" 2> /dev/null
then ok "tolerance sweep: tightening the U and e solves lowers the raw momentum and energy imbalance at every step until round-off (momentum:energy:their allowances$sweep)"
else fail "tolerance sweep: the raw imbalance does not fall at every tightening (momentum:energy:their allowances$sweep)"; fi

# ---- memory and runtime (reported) ------------------------------------------------------------------------------------------
for c in $CASES; do
    echo "     runtime $c: OpenFOAM $(grep -h '^ExecutionTime' "$W/of_$c/log" | tail -n 1 | awk '{print $3}') s; brae CUDA $(sed -n 's/^diag steps \([^ ]*\) step-seconds \([^ ]*\) per-step-us \([^ ]*\).*/\2 s (\1 steps, \3 us a step)/p' "$W/b_$c/log"), device memory $(sed -n 's/^diag device-memory \([^ ]*\) MiB/\1 MiB/p' "$W/b_$c/log")"
done

# ---- controls ---------------------------------------------------------------------------------------------------------
red "noTauMC, sod" "$W/cT_sod" "$W/of_sod" "$(spread sod)"
red "noTauMC, oblique" "$W/cT_oblique" "$W/of_oblique" "$(spread oblique)"
red "noSigmaDotU, shear" "$W/cS_shear" "$W/of_shear" "$(spread shear)"
red "noSigmaDotU, sod" "$W/cS_sod" "$W/of_sod" "$(spread sod)"
red "noTauMC, sod_conv" "$W/cT_sod_conv" "$W/of_sod_conv" "$(spread sod_conv)"
red "noSigmaDotU, sod_conv" "$W/cS_sod_conv" "$W/of_sod_conv" "$(spread sod_conv)"
opensummary
echo "rho_central_viscous_vs_openfoam: $passes passed, $fails failed, $historical historical (raw conservation, reported) (work in $W)"
[ "$fails" = 0 ] && [ "$passes" -ge "${BRAE_RCF_G_MIN_PASSES:-1}" ]
