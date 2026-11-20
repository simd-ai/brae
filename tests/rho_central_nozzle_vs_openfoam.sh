#!/usr/bin/env bash
# brae_rhoCentralFoam against OpenFOAM v2412 rhoCentralFoam on nozzle inflow and outflow boundaries (milestone H):
# totalPressure (the isentropic psi form), totalTemperature and waveTransmissive on p, with sutherland transport and the
# viscous correctors on wedge patches, which OpenFOAM's LadenburgJet60psi needs. Single GPU, static mesh, laminar.
#
# THE BOUNDARIES (OpenFOAM v2412 src/finiteVolume/fields/fvPatchFields/derived), updated where OpenFOAM updates them:
#   totalPressure     p_b = p0/pow(1 + 0.5*psi_b*gM1ByG*neg(phi_b)*magSqr(U_b), 1/gM1ByG), gM1ByG = (gamma - 1)/gamma, at
#                     p.correctBoundaryConditions() (rhoCentralFoam.C:287) on this step's phi, U_b and corrected psi_b
#   totalTemperature  T_b = T0/(1 + 0.5*psi_b*gM1ByG*neg(phi_b)*magSqr(U_b)), at every Tw.evaluate() a fixedEnergy he
#                     patch runs: e.correctBoundaryConditions() (:265) and, viscous, the e matrix's construction
#                     (not again after the e solve: the patch is then updated())
#   waveTransmissive  advective's mixed: w = max((phi_b/rho_b)/|Sf| + sqrt(gamma/psi_b), 0), alpha = w*deltaT*deltaCoeffs,
#                     k = w*deltaT/lInf; refValue = (p_b.oldTime + k*fieldInf)/(1 + k), valueFraction =
#                     (1 + k)/(1 + alpha + k) (lInf <= 0: p_b.oldTime and 1/(1 + alpha)), at :287 with rho_b before :288
#   sutherland        mu = As*sqrt(T)/(1 + Ts/T), kappa = mu*Cv*(1.32 + 1.77*R/Cv), alphahe = gamma*kappa/Cp: muEff from
#                     the T the step starts from, alphaEff from the T of the step's first thermo.correct()
# CASES (tests/rho_central_nozzle.py)
#   ladenburg        LadenburgJet60psi as shipped (1200 cells, axisymmetric wedge; startFrom its converged 0, endTime 2e-5,
#                    maxCo 0.5; U and e smoothSolver GaussSeidel to 1e-10, relTol 0), writePrecision 17
#   ladenburg_long   the same to 1e-4, writes every 2e-5
#   ladenburg_uzseed (milestone I) the shipped case with U_z seeded +-1e-3 in cells 25-39, to 3e-5, writes every 5e-6:
#                    OpenFOAM's out-of-plane mode at the Mach disk grown from a seed far above round-off
#   reservoir        the 1000-cell tube at rest fed through totalPressure/totalTemperature (p0 2, T0 1.2) into a
#                    waveTransmissive outlet (fieldInf 1, lInf 2); inviscid; t = 10
#   pulse            a pressure pulse leaving both ends through waveTransmissive (one with lInf, one without); inviscid; t = 6
#   backflow         the tube leaving through totalPressure/totalTemperature (their outflow branch); inviscid; t = 4
#   reservoir_visc   reservoir with sutherland (As 0.02, Ts 1), U and e as LadenburgJet60psi solves them
#   conduction_suth  milestone G's conduction (T = 2 | 1 at rest) with sutherland, the same solves; t = 1
#   (the two sutherland tube cases first ran with tolerance 1e-13, a trial on 2026-11-18 before any gate: both codes then
#   ended every e solve at maxIter 1000 with the residual at its floor, 1.5e-12, and one U solve's sweep count straddled
#   the tolerance; they solve as the tutorial does instead)
# REFERENCE  OpenFOAM serial; its spread in two (simple) and in four (the frozen scotch partitions: LadenburgJet60psi's,
#   frozen for this gate; the tube cases keep shockTube1000's faces, owner and neighbour, so its) --
#   tests/rho_central_gate_lib.sh's envelope, unchanged: ten times OpenFOAM's own spread, floor BOUND 8e-14 -- and, a
#   DIAGNOSTIC that bounds nothing, OpenFOAM serial started with p one ulp up. The inviscid tube cases amplify round-off
#   in OpenFOAM itself (its own runs part by 1e-9 at t = 2 and by 1e-4 to 1e-2 at their ends): their envelope is wide late.
# CHECKS, DECLARED BEFORE THE GATE FIRST RAN (2026-11-18), the CUDA arm
#   per case     steps, every deltaT and name, Courant, directories, U p T rho at every write time (arm)
#   SOLVES       viscous cases: the same solves in the same order; every iteration count OpenFOAM serial's
#   PROFILES     every nozzle patch's p, T and U (where OpenFOAM writes a value) at every write time within ten times
#                OpenFOAM's own spread of them (floor BOUND)
#   BRANCHES     OpenFOAM's totalPressure faces take both branches where the case is built for it (LadenburgJet60psi's
#                freestream in and out, reservoir's inlet in, backflow's inlet out), and brae's the same faces
#   CONSERVATION inviscid tube cases: mass, momentum and energy to round-off, the patches' fluxes counted; viscous cases:
#                momentum and energy to round-off once the solves' remainder is counted (vbalance) and every solve's
#                identity to 1e-12 (identity); the raw viscous balance is reported, as milestone G's five
#                tolerance-limited items are (open, criteria for review); positivity everywhere
#   SUTHERLAND   (reported) mu(T) over conduction_suth's final T, from OpenFOAM's field: the spread it takes
#   MEMORY, RUNTIME (reported)
# CONTROLS (rhoCentralState.cuh), red against OpenFOAM by more than 100 x the bound:
#   noDynamicHead        p_b = p0 and T_b = T0 on inflow too: reservoir, reservoir_visc, ladenburg
#   nozzleFlowInverted   the switch on -phi: reservoir, backflow, ladenburg
#   waveNoRelax          lInf ignored: reservoir, pulse, ladenburg
#   waveVolumeFlux       w from phi/|Sf| (not divided by rho_b): reservoir, pulse, ladenburg
#   totalTemperatureOnce T_b not re-evaluated at the e matrix's construction: reservoir_visc, ladenburg
#   residualUnfused      (milestone I) the viscous solves' loop residual rounded product by product, where OpenFOAM's
#                        aarch64 build fuses each multiply-subtract: reservoir_visc's stopping test parts in MORE solves
# REFUSED     the host reference arm (checked here); totalPressure's rho and incompressible forms, these patches on other
#   fields or other fluxes, waveTransmissive's lInf without fieldInf (tests/rho_central_refusals.sh)
# NOT CLAIMED  turbulence (refused), mesh motion, 3-D cases, multi-GPU, performance (GaussSeidel sweeps on the CPU).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BRAE_RCF_BIN:-$ROOT/build/brae_rhoCentralFoam}"
STEPS="$ROOT/tests/rho_central_steps.py"
FIELDS="$ROOT/tests/rho_central_fields.py"
NOZ="$ROOT/tests/rho_central_nozzle.py"
OFBASHRC=${OF_BASHRC:-${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}}
TUT="${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}/compressible/rhoCentralFoam"
[ -x "$BIN" ] || { echo "SKIP: no brae_rhoCentralFoam at $BIN"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: no OpenFOAM v2412 at $OFBASHRC"; exit 77; }
[ -d "$TUT/LadenburgJet60psi/0.orig" ] && [ -d "$TUT/shockTube/0.orig" ] || { echo "SKIP: no LadenburgJet60psi/shockTube at $TUT"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
W=${BRAE_GATE_OUT:-$(mktemp -d /tmp/brae-rcf-nozzle.XXXXXX)}
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
    local d=$1 k=$2 f
    rm -rf "$d"
    mkdir -p "$d/0"
    case $k in
        ladenburg*)
            cp -r "$TUT/LadenburgJet60psi/system" "$TUT/LadenburgJet60psi/constant" "$d/"
            for f in "$TUT/LadenburgJet60psi/0.orig/"*
            do
                case $f in *.gz) zcat "$f" > "$d/0/$(basename "$f" .gz)" ;; *) cp "$f" "$d/0/" ;; esac
            done ;;
        *)  cp -r "$TUT/shockTube/system" "$TUT/shockTube/constant" "$d/"; cp -r "$TUT/shockTube/0.orig/." "$d/0/" ;;
    esac
    python3 "$NOZ" setup "$k" "$d" || { fail "stage $(basename "$d"): setup"; return 1; }
    of "cd '$d' && blockMesh > log.blockMesh 2>&1" < /dev/null || fail "stage $(basename "$d"): blockMesh"
    return 0
}
CASES="ladenburg ladenburg_long ladenburg_uzseed reservoir pulse backflow reservoir_visc conduction_suth"
VCASES="ladenburg ladenburg_long ladenburg_uzseed reservoir_visc conduction_suth"
ICASES="reservoir pulse backflow"
for c in $CASES; do stage "$W/$c" "$c"; done
patchesof() { case $1 in ladenburg*) echo inlet,outlet,freestream,freestreamInlet ;; conduction_suth) echo "" ;; *) echo inlet,outlet ;; esac; }

# ---- OpenFOAM ---------------------------------------------------------------------------------------------------------
for c in $CASES; do copy "$W/$c" "$W/of_$c"; copy "$W/$c" "$W/of_${c}_np2"; copy "$W/$c" "$W/of_${c}_np4"; done
for c in $CASES; do throttle; ofrun "$W/of_$c" & done
for c in $CASES; do throttle; ofulp "$W/$c" "$W/of_${c}_ulp" 0 & done
wait
for c in $CASES; do throttle; ofnp "$W/of_${c}_np2" 2 & done
wait
for c in $CASES; do throttle; ofnp "$W/of_${c}_np4" 4 & done
wait
for c in $CASES; do for r in "" _np2 _np4 _ulp; do grep -qx End "$W/of_$c$r/log" || fail "OpenFOAM${r:+ $r} did not finish $c"; done; done

# ---- brae -------------------------------------------------------------------------------------------------------------
for c in $CASES; do copy "$W/$c" "$W/b_$c"; braerun "$W/b_$c"; done
control() { copy "$W/$2" "$W/c$1_$2"; ctlrun "$W/c$1_$2" BRAE_CONTROL_RCF="$1"; }
for c in reservoir reservoir_visc ladenburg; do control noDynamicHead "$c"; done
for c in reservoir backflow ladenburg; do control nozzleFlowInverted "$c"; done
for c in reservoir pulse ladenburg; do control waveNoRelax "$c"; control waveVolumeFlux "$c"; done
for c in reservoir_visc ladenburg; do control totalTemperatureOnce "$c"; done
control residualUnfused reservoir_visc
# the host reference arm refuses the nozzle boundaries
copy "$W/reservoir" "$W/h_reservoir"
(cd "$W/h_reservoir" && BRAE_RHOCENTRALFOAM_ARM=cpu "$BIN" -case . > log 2>&1 < /dev/null; echo $? > rc)
if [ "$(cat "$W/h_reservoir/rc")" != 0 ] && grep -q "host reference arm .* does not run totalPressure, totalTemperature or waveTransmissive" "$W/h_reservoir/log"
then ok "the host reference arm refuses the nozzle boundaries by name: $(grep -o 'the host reference arm[^;]*' "$W/h_reservoir/log" | head -n 1)"
else fail "host reference arm with nozzle boundaries: exit $(cat "$W/h_reservoir/rc")"; fi

# ---- comparisons ------------------------------------------------------------------------------------------------------
spread() { echo "$W/of_${1}_np2 $W/of_${1}_np4"; }
for c in $CASES; do arm "$c (CUDA)" "$W/b_$c" "$W/of_$c" "$(spread "$c")" "$BOUND" exact; done

# the solves of the viscous cases
for c in $VCASES; do
    k=$(cmpdir "$W/b_$c")
    python3 "$STEPS" solves "$W/b_$c/log" "$W/of_$c/log" > "$k/solves.cmp" 2>&1
    read -r nb no same <<< "$(sed -n 's/^solves //p' "$k/solves.cmp")"
    read -r iok ifirst <<< "$(sed -n 's/^iterations //p' "$k/solves.cmp")"
    read -r iw iws iwf imed <<< "$(sed -n 's/^initial //p' "$k/solves.cmp")"
    if [ -n "$nb" ] && [ "$nb" = "$no" ] && [ "$same" = 1 ] && [ "$nb" -gt 0 ]
    then ok "$c (CUDA): the same $nb solves as OpenFOAM's, in the same order ($(grep -oE 'Solving for [A-Za-z]+' "$W/of_$c/log" | grep -vE 'for (rho|rhoU|rhoE)' | sort | uniq -c | awk '{printf "%s%s %s", (NR>1?", ":""), $1, $4}'))"
    else fail "$c (CUDA): solves ${nb:-?} against ${no:-?}, order the same: ${same:-?}"; fi
    if [ "${iok:-0}" = "$nb" ] && [ "${ifirst:-1}" = 0 ]
    then ok "$c (CUDA): every iteration count OpenFOAM serial's ($nb solves, $(grep -E 'Solving for (U|e)' "$W/of_$c/log" | grep -oE 'No Iterations [0-9]+' | awk '{s+=$3; if ($3>m) m=$3} END {printf "%d sweeps in all, at most %d", s, m}'))"
    else fail "$c (CUDA): iteration counts differ from solve ${ifirst:-?} (${iok:-?} of $nb the same)"; fi
    echo "     $c (CUDA): initial residuals against OpenFOAM serial's: median relative difference $imed, worst $iw (solve $iws, $iwf)"
    # OpenFOAM's own sensitivity of the stopping test (reported, bounds nothing): its perturbed runs against serial
    own=""
    for r in np2:"in two" np4:"in four" ulp:"one ulp up"; do
        read -r _ rk rf <<< "$(python3 "$STEPS" solves "$W/of_${c}_${r%%:*}/log" "$W/of_$c/log" 2>/dev/null | sed -n 's/^iterations /x /p')"
        own="$own, ${r#*:} $(( nb - ${rk:-0} )) (first at solve ${rf:-?})"
    done
    echo "     $c: OpenFOAM's own runs against its serial, iteration counts that differ:${own#,}"
done

# the loop residual's rounding (milestone I): with the products rounded apart (x86-64 OpenFOAM's form) the stopping test
# parts from the reference's (OpenFOAM's aarch64 build fuses them) in more solves than brae's own does
k=$(cmpdir "$W/cresidualUnfused_reservoir_visc")
python3 "$STEPS" solves "$W/cresidualUnfused_reservoir_visc/log" "$W/of_reservoir_visc/log" > "$k/solves.cmp" 2>&1
read -r un _ <<< "$(sed -n 's/^iterations //p' "$k/solves.cmp")"
read -r fu _ <<< "$(sed -n 's/^iterations //p' "$(cmpdir "$W/b_reservoir_visc")/solves.cmp")"
read -r ns _ <<< "$(sed -n 's/^solves //p' "$k/solves.cmp")"
if [ -n "$un" ] && [ -n "$fu" ] && [ "$((ns - un))" -gt "$((ns - fu))" ]
then ok "control residualUnfused, reservoir_visc red: iteration counts differ from OpenFOAM serial's in $((ns - un)) of $ns solves, against $((ns - fu)) with the fused residual"
else fail "control residualUnfused, reservoir_visc: $((ns - ${un:-0})) differing solves, fused $((ns - ${fu:-0}))"; fi

# the boundary profiles, write time by write time
for c in $CASES; do
    pl=$(patchesof "$c"); [ -n "$pl" ] || continue
    over=""; worst=-1; wl=""; on=""
    for t in $(times "$W/of_$c"); do
        read -r _ _ pb _ Tb _ Ub _ on <<< "$(python3 "$NOZ" patches "$W/b_$c" "$W/of_$c" "$t" "$pl")"
        read -r _ _ p2 _ T2 _ U2 _ <<< "$(python3 "$NOZ" patches "$W/of_${c}_np2" "$W/of_$c" "$t" "$pl")"
        read -r _ _ p4 _ T4 _ U4 _ <<< "$(python3 "$NOZ" patches "$W/of_${c}_np4" "$W/of_$c" "$t" "$pl")"
        for q in "p ${pb:-nan} ${p2:-nan} ${p4:-nan}" "T ${Tb:-nan} ${T2:-nan} ${T4:-nan}" "U ${Ub:-nan} ${U2:-nan} ${U4:-nan}"; do
            read -r nm vb v2 v4 <<< "$q"
            [ "$vb" = none ] && continue   # no patch of this case writes a value of the field
            bd=$(mx "$BOUND" "$(python3 -c "print(10*max(float('$v2'), float('$v4')))")")
            le "$vb" "$bd" || over="$over $t:$nm($vb>$bd)"
            r=$(python3 -c "print('%.3g' % (float('$vb')/float('$bd')))")
            python3 -c "import sys; sys.exit(0 if float('$r') > float('$worst') else 1)" 2> /dev/null && { worst=$r; wl="$nm $vb at t = $t (OpenFOAM in two $v2, in four $v4)"; }
        done
    done
    if [ -z "$over" ]; then ok "$c (CUDA): the nozzle patches' values ($on) within ten times OpenFOAM's own spread at every write time; closest ($worst of the bound): $wl"
    else fail "$c (CUDA): nozzle patch values above the bound at$over"; fi
done

# the totalPressure branches, at the last write
branch()   # branch <case> <patch> <p0> <want: both|in|out>
{
    local c=$1 pt=$2 p0=$3 want=$4 t ob ib hb oo io ho
    t=$(times "$W/of_$c" | awk '{print $NF}')
    read -r _ _ oo _ io _ ho <<< "$(python3 "$NOZ" branches "$W/of_$c" "$t" "$pt" "$p0")"
    read -r _ _ ob _ ib _ hb <<< "$(python3 "$NOZ" branches "$W/b_$c" "$t" "$pt" "$p0")"
    local good=0
    case $want in
        both) [ "${oo:-0}" -gt 0 ] && [ "${io:-0}" -gt 0 ] && good=1 ;;
        in) [ "${io:-0}" -gt 0 ] && [ "${oo:-1}" = 0 ] && good=1 ;;
        out) [ "${oo:-0}" -gt 0 ] && [ "${io:-1}" = 0 ] && good=1 ;;
    esac
    if [ "$good" = 1 ] && [ "$ob" = "$oo" ] && [ "$ib" = "$io" ]
    then ok "$c: totalPressure on $pt at t = $t: OpenFOAM's faces $oo outflow (p_b = p0) and $io inflow (dynamic head up to $ho of p0), brae's the same ($ob, $ib; up to $hb)"
    else fail "$c: totalPressure branches on $pt at t = $t: OpenFOAM ${oo:-?} out ${io:-?} in, brae ${ob:-?} out ${ib:-?} in (wanted $want)"; fi
}
branch ladenburg freestream 101325 both
branch ladenburg_long freestream 101325 both

# ---- the out-of-plane mode at the Mach disk (milestone I; H's open item, kept red above) ---------------------------------
# U_z is round-off on a wedge (0 at t = 0 in every cell). OpenFOAM's discretization has a mode there that grows: seeded
# +-1e-9 (cells 25-39) it grows to 1.07e-5 by t = 1e-4 in OpenFOAM and in brae alike (milestone I). Unseeded,
# brae's own round-off excites it from about 4e-5; OpenFOAM's in two, in four and one ulp up do not within 1e-4, while
# OpenFOAM with a random U_z of 100 ulps of |U| does (7e-8 at 1e-4; 1 ulp does not). Covered here by
#   IN-PLANE  ladenburg_long's U_x and U_y at every write within ten times OpenFOAM's own spread of them (floor BOUND)
#   SEEDED    ladenburg_uzseed through the arm above (fields, steps, solves): the mode's growth from 1e-3, both codes
#   reported  max|U_z| over cells 25-39 at every write: brae, OpenFOAM serial, in two, in four, one ulp up
inplane()   # inplane <case>
{
    local c=$1 t vb v2 v4 bd over="" worst=-1 wl="" r
    for t in $(times "$W/of_$c"); do
        vb=$(python3 "$FIELDS" absdiff "$W/b_$c/$t" "$W/of_$c/$t" U:xy | awk '/^absdiff/ {print $4}')
        v2=$(python3 "$FIELDS" absdiff "$W/of_${c}_np2/$t" "$W/of_$c/$t" U:xy | awk '/^absdiff/ {print $4}')
        v4=$(python3 "$FIELDS" absdiff "$W/of_${c}_np4/$t" "$W/of_$c/$t" U:xy | awk '/^absdiff/ {print $4}')
        bd=$(mx "$BOUND" "$(python3 -c "print(10*max(float('${v2:-nan}'), float('${v4:-nan}')))")")
        le "${vb:-nan}" "$bd" || over="$over $t(${vb:-nan}>$bd)"
        r=$(python3 -c "print('%.3g' % (float('${vb:-nan}')/float('$bd')))")
        python3 -c "import sys; sys.exit(0 if float('$r') > float('$worst') else 1)" 2> /dev/null && { worst=$r; wl="$vb at t = $t (OpenFOAM in two $v2, in four $v4)"; }
    done
    if [ -z "$over" ]; then ok "$c (CUDA): U's in-plane components (x, y) within ten times OpenFOAM's own spread at every write time; closest ($worst of the bound): $wl"
    else fail "$c (CUDA): U's in-plane components above the bound at$over"; fi
}
inplane ladenburg_long
for c in ladenburg_long ladenburg_uzseed; do
    for t in $(times "$W/of_$c"); do
        echo "     $c, max|U_z| over cells 25-39 at t = $t: $(python3 -c "
import sys; sys.path.insert(0, '$ROOT/tests')
import numpy as np, rho_central_fields as rf
r = [('brae', '$W/b_$c'), ('OpenFOAM', '$W/of_$c'), ('in two', '$W/of_${c}_np2'), ('in four', '$W/of_${c}_np4'), ('one ulp up', '$W/of_${c}_ulp')]
print(', '.join('%s %.3e' % (n, np.abs(np.asarray(rf.internal(d + '/$t/U'), float)[25:40, 2]).max()) for n, d in r))")"
    done
done
branch reservoir inlet 2 in
branch reservoir_visc inlet 2 in
branch backflow inlet 1.1 out

# ---- sutherland (reported) ------------------------------------------------------------------------------------------------
t=$(times "$W/of_conduction_suth" | awk '{print $NF}')
echo "     sutherland, conduction_suth at t = $t: $(python3 -c "
import sys; sys.path.insert(0, '$ROOT/tests')
import numpy as np, rho_central_fields as rf
T = np.asarray(rf.internal('$W/of_conduction_suth/$t/T'), float)
mu = 0.02*np.sqrt(T)/(1 + 1/T)
print('OpenFOAM T from %.6g to %.6g, mu(T) from %.6g to %.6g (a factor %.4g)' % (T.min(), T.max(), mu.min(), mu.max(), mu.max()/mu.min()))")"

# ---- conservation and stability --------------------------------------------------------------------------------------------
for c in $ICASES; do balance "b_$c"; done
for c in $VCASES; do
    vbalance "b_$c"
    identity "b_$c"
    echo "     b_$c raw balance (reported; it carries the solves' remainder, milestone G's tolerance-limited items): $(sed -n 's/^diag balance \(.*\) (relative.*/\1/p' "$W/b_$c/log")"
done

# ---- memory and runtime (reported) ------------------------------------------------------------------------------------------
for c in $CASES; do
    echo "     runtime $c: OpenFOAM $(grep -h '^ExecutionTime' "$W/of_$c/log" | tail -n 1 | awk '{print $3}') s; brae CUDA $(sed -n 's/^diag steps \([^ ]*\) step-seconds \([^ ]*\) per-step-us \([^ ]*\).*/\2 s (\1 steps, \3 us a step)/p' "$W/b_$c/log"), device memory $(sed -n 's/^diag device-memory \([^ ]*\) MiB/\1 MiB/p' "$W/b_$c/log")"
done

# ---- controls ---------------------------------------------------------------------------------------------------------
for c in reservoir reservoir_visc ladenburg; do red "noDynamicHead, $c" "$W/cnoDynamicHead_$c" "$W/of_$c" "$(spread "$c")"; done
for c in reservoir backflow ladenburg; do red "nozzleFlowInverted, $c" "$W/cnozzleFlowInverted_$c" "$W/of_$c" "$(spread "$c")"; done
for c in reservoir pulse ladenburg; do
    red "waveNoRelax, $c" "$W/cwaveNoRelax_$c" "$W/of_$c" "$(spread "$c")"
    red "waveVolumeFlux, $c" "$W/cwaveVolumeFlux_$c" "$W/of_$c" "$(spread "$c")"
done
for c in reservoir_visc ladenburg; do red "totalTemperatureOnce, $c" "$W/ctotalTemperatureOnce_$c" "$W/of_$c" "$(spread "$c")"; done
opensummary
echo "rho_central_nozzle_vs_openfoam: $passes passed, $fails failed (work in $W)"
[ "$fails" = 0 ] && [ "$passes" -ge "${BRAE_RCF_H_MIN_PASSES:-1}" ]
