#!/usr/bin/env bash
# brae_rhoCentralFoam against OpenFOAM v2412 rhoCentralFoam on milestone I: local time stepping (ddtSchemes localEuler,
# setRDeltaT.H), janaf thermo, and the rarefied-gas walls of rhoCentralFoam/BCs (maxwellSlipU, smoluchowskiJumpT) --
# OpenFOAM's biconic25-55Run35 tutorial as shipped, and focused cases for each. Single GPU, static mesh, laminar.
#
# WHAT IS PORTED (OpenFOAM v2412), and where it updates
#   LTS          createRDeltaT.H (LTS = ddtSchemes default localEuler); setRDeltaT.H after centralCourantNo.H:
#                rDeltaT = max(1/maxDeltaT, surfaceSum(amaxSf)/((2*maxCo)*V)), correctBoundaryConditions, fvc::smooth
#                (rDeltaTSmoothingCoeff, default 0.02; brae's FaceCellWave port, to the bit of its host oracle), "Flow
#                time scale min/max" = gMin/gMax(1/rDeltaT); ++runTime after it, no setDeltaT.H; the diagonal solves and
#                the viscous correctors' fvm::ddt(rho, .) - fvc::ddt(rho, .) on localEulerDdtScheme's per-cell rDeltaT
#                (EulerDdtScheme's expressions with the field in the scalar's place); rDeltaT written (AUTO_WRITE,
#                extrapolatedCalculated)
#   janaf        janafThermoI.H: Cp, Ha, Hs = Ha - Hc, the coefficient switch at Tcommon, limit() clamping to
#                [Tlow, Thigh]; the coefficients times R (janafThermo.C); T(e) by thermoI.H's Newton loop on Cv(T); the
#                sound speed from thermo.Cp()/thermo.Cv() per cell and face; sutherland's kappa on Cv(T); alphahe() =
#                CpByCpv()*alpha_. OpenFOAM v2412 builds janaf psiThermo with sutherland transport only (refused
#                otherwise, as OpenFOAM does)
#   walls        maxwellSlipU (a partialSlip: valueFraction 1/(1 + deltaCoeffs*C1*nu), refValue Uwall - thermal creep
#                3 nu/(4 T_b) (I - n n).grad(T)_b - curvature C1/rho_b (I - n n).(n & tauMC_b)), updated at :232 and at
#                the U matrix's construction; smoluchowskiJumpT (a mixed patch: valueFraction 1/(1 + deltaCoeffs*C2),
#                refValue Twall) through e's mixedEnergy, at :265 and at the e matrix's construction; T_b kept through the
#                e solve's correctBoundaryConditions
#   THE e MATRIX'S ORDER  found on biconic25-55Run35 (2026-11-18, one step): OpenFOAM's build constructs fvm::ddt(rho, e)
#                -- whose updateCoeffs re-evaluates every T patch on the cells' T of the first thermo.correct() --
#                before it calls turbulence->alphaEff(), so alphahe()'s CpByCpv reads those T_b; with janaf this moves
#                e's matrix (control alphaheOldT)
# CASES (tests/rho_central_lts.py)
#   biconic        biconic25-55Run35 AS SHIPPED: its Allrun's mesh (blockMesh, datToFoam grid256.dat, collapseEdges, the
#                  wedge rename), 0.orig gzipped, writeFormat binary, endTime 1e-3 (10000 LTS steps); nothing edited
#   fs_lts         forwardStep with ddtSchemes localEuler, to 0.5, writes every 0.1 (inviscid hConst LTS)
#   ob_lts_visc    milestone G's viscous obliqueShock with ddtSchemes localEuler, to 0.5 (viscous LTS)
#   sod_janaf      the 1000-cell tube of N2 on biconic's janaf and sutherland, p 1e5 | 1e4 Pa, T 1500 | 600 K (shock and
#                  contact across Tcommon = 1000 K), to 3e-3
#   sod_janaf_visc the same with As 1000 times larger (diffusion over cells)
#   channel_slip   a rarefied channel (Knudsen about 0.04) on biconic's system/ and thermo, its floor maxwellSlipU and
#                  smoluchowskiJumpT (Twall 300 K, gas 200 K, U 1000 m/s), Euler, to 1e-4
# REFERENCE  OpenFOAM serial; its spread in two (simple) and in four (frozen scotch partitions: forwardStep's,
#   obliqueShock's, shockTube1000's, and biconic25-55Run35's and channelSlip's, frozen for this gate) -- the envelope of
#   tests/rho_central_gate_lib.sh, unchanged: ten times OpenFOAM's own spread, floor BOUND 8e-14 -- and, a DIAGNOSTIC that
#   bounds nothing, OpenFOAM serial with p one ulp up (biconic's p unpacked from its .gz for that run only)
# CHECKS, DECLARED BEFORE THE GATE FIRST RAN (2026-11-18), the CUDA arm
#   per case     steps, every deltaT and name, Courant, directories, U p T rho at every write time (arm)
#   SOLVES       viscous cases: the same solves in the same order; every iteration count OpenFOAM serial's
#   LTS          the Flow time scale min/max of every step, and rDeltaT at every write time, within ten times OpenFOAM's
#                own spread of them (floor BOUND)
#   JANAF        premise: OpenFOAM's T has cells on both sides of Tcommon at the last write (sod_janaf*, biconic)
#   WALLS        U and T on the cone (biconic) and the wall (channel_slip) at every write within ten times OpenFOAM's own
#                spread of them (floor BOUND)
#   AS SHIPPED   biconic: brae reads the tutorial's gzipped 0 and binary mesh, and says it writes ascii where the case asks
#                for binary (writeFormat: brae's shared writer is ascii only); OpenFOAM's binary output is what it is
#                compared with
#   CONSERVATION Euler viscous cases: positivity; the residual-accounted balance and every solve's identity (vbalance,
#                identity); the raw balance reported (hist). LTS cases: positivity only (a pseudo-time march)
#   MEMORY, RUNTIME (reported)
# CONTROLS (rhoCentralState.cuh), red against OpenFOAM by more than 100 x the bound:
#   ltsUnsmoothed fs_lts; janafLowOnly sod_janaf; alphaheOldT channel_slip; maxwellNoCreep, maxwellNoCurvature,
#   smoluchowskiNoJump channel_slip
# REFUSED (tests/rho_central_refusals.sh): janaf with const transport; a mix of localEuler and Euler; waveTransmissive
#   under LTS; maxwellSlipU/smoluchowskiJumpT inviscid or with nonuniform Uwall/Twall; the host reference arm with LTS
# NOT CLAIMED  turbulence, mesh motion, multi-GPU, binary output, performance (GaussSeidel sweeps on the CPU).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BRAE_RCF_BIN:-$ROOT/build/brae_rhoCentralFoam}"
STEPS="$ROOT/tests/rho_central_steps.py"
FIELDS="$ROOT/tests/rho_central_fields.py"
LTS="$ROOT/tests/rho_central_lts.py"
OFBASHRC=${OF_BASHRC:-${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}}
TUT="${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}"
RCF="$TUT/compressible/rhoCentralFoam"
[ -x "$BIN" ] || { echo "SKIP: no brae_rhoCentralFoam at $BIN"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: no OpenFOAM v2412 at $OFBASHRC"; exit 77; }
[ -d "$RCF/biconic25-55Run35/0.orig" ] && [ -f "$TUT/resources/geometry/grid256.dat.gz" ] \
    || { echo "SKIP: no biconic25-55Run35 or its grid256.dat at $TUT"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
W=${BRAE_GATE_OUT:-$(mktemp -d /tmp/brae-rcf-lts.XXXXXX)}
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
    case $k in
        biconic)
            cp -r "$RCF/biconic25-55Run35/system" "$RCF/biconic25-55Run35/constant" "$d/"
            cp -r "$RCF/biconic25-55Run35/0.orig" "$d/0"
            of "cd '$d' && blockMesh > log.blockMesh 2>&1 && cp '$TUT/resources/geometry/grid256.dat.gz' . \
                && gunzip -f grid256.dat.gz && datToFoam grid256.dat > log.datToFoam 2>&1 \
                && collapseEdges -overwrite > log.collapseEdges 2>&1" < /dev/null \
                || { fail "stage biconic: the Allrun's mesh"; return 1; }
            sed -i '/wedge1/,/nFaces/{s/patch/wedge/};/wedge2/,/nFaces/{s/patch/wedge/}' "$d/constant/polyMesh/boundary"
            return 0 ;;
        fs_lts) cp -r "$RCF/forwardStep/system" "$RCF/forwardStep/constant" "$RCF/forwardStep/0" "$d/" ;;
        ob_lts_visc) cp -r "$RCF/obliqueShock/system" "$RCF/obliqueShock/constant" "$RCF/obliqueShock/0" "$d/" ;;
        sod_janaf*) cp -r "$RCF/shockTube/system" "$RCF/shockTube/constant" "$d/"; cp -r "$RCF/shockTube/0.orig" "$d/0" ;;
        channel_slip) cp -r "$RCF/biconic25-55Run35/system" "$RCF/biconic25-55Run35/constant" "$d/" ;;
    esac
    python3 "$LTS" setup "$k" "$d" || { fail "stage $k: setup"; return 1; }
    of "cd '$d' && blockMesh > log.blockMesh 2>&1" < /dev/null || fail "stage $k: blockMesh"
    return 0
}
CASES="biconic fs_lts ob_lts_visc sod_janaf sod_janaf_visc channel_slip"
LCASES="biconic fs_lts ob_lts_visc"
VCASES="biconic ob_lts_visc sod_janaf sod_janaf_visc channel_slip"
ECASES="sod_janaf sod_janaf_visc channel_slip"   # viscous, Euler in time: the residual-accounted balance applies
for c in $CASES; do stage "$W/$c" "$c"; done

# ---- OpenFOAM ---------------------------------------------------------------------------------------------------------
# biconic's p is gzipped as shipped: the one-ulp run's copy is unpacked for the nudge (that run only)
ulp() { copy "$1" "$2"; [ -f "$2/0/p.gz" ] && { gunzip -f "$2/0/p.gz" || return 1; }; python3 "$FIELDS" nudge "$2/0/p" > /dev/null && ofrun "$2"; }
for c in $CASES; do copy "$W/$c" "$W/of_$c"; copy "$W/$c" "$W/of_${c}_np2"; copy "$W/$c" "$W/of_${c}_np4"; done
for c in $CASES; do throttle; ofrun "$W/of_$c" & done
for c in $CASES; do throttle; ulp "$W/$c" "$W/of_${c}_ulp" & done
wait
for c in $CASES; do throttle; ofnp "$W/of_${c}_np2" 2 & done
wait
for c in $CASES; do throttle; ofnp "$W/of_${c}_np4" 4 & done
wait
for c in $CASES; do for r in "" _np2 _np4 _ulp; do grep -qx End "$W/of_$c$r/log" || fail "OpenFOAM${r:+ $r} did not finish $c"; done; done

# ---- brae -------------------------------------------------------------------------------------------------------------
for c in $CASES; do copy "$W/$c" "$W/b_$c"; braerun "$W/b_$c"; done
control() { copy "$W/$2" "$W/c$1_$2"; ctlrun "$W/c$1_$2" BRAE_CONTROL_RCF="$1"; }
control ltsUnsmoothed fs_lts
control janafLowOnly sod_janaf
for k in alphaheOldT maxwellNoCreep maxwellNoCurvature smoluchowskiNoJump; do control "$k" channel_slip; done
# the host reference arm refuses LTS
copy "$W/fs_lts" "$W/h_fs_lts"
(cd "$W/h_fs_lts" && BRAE_RHOCENTRALFOAM_ARM=cpu "$BIN" -case . > log 2>&1 < /dev/null; echo $? > rc)
if [ "$(cat "$W/h_fs_lts/rc")" != 0 ] && grep -q "host reference arm .* does not run local time stepping" "$W/h_fs_lts/log"
then ok "the host reference arm refuses LTS by name: $(grep -o 'the host reference arm[^;]*' "$W/h_fs_lts/log" | head -n 1)"
else fail "host reference arm with localEuler: exit $(cat "$W/h_fs_lts/rc")"; fi

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
    then ok "$c (CUDA): the same $nb solves as OpenFOAM's, in the same order"
    else fail "$c (CUDA): solves ${nb:-?} against ${no:-?}, order the same: ${same:-?}"; fi
    if [ "${iok:-0}" = "$nb" ] && [ "${ifirst:-1}" = 0 ]
    then ok "$c (CUDA): every iteration count OpenFOAM serial's ($nb solves)"
    else fail "$c (CUDA): iteration counts differ from solve ${ifirst:-?} (${iok:-?} of $nb the same)"; fi
    echo "     $c (CUDA): initial residuals against OpenFOAM serial's: median relative difference $imed, worst $iw (solve $iws, $iwf)"
    own=""
    for r in np2:"in two" np4:"in four" ulp:"one ulp up"; do
        read -r _ rk rf <<< "$(python3 "$STEPS" solves "$W/of_${c}_${r%%:*}/log" "$W/of_$c/log" 2>/dev/null | sed -n 's/^iterations /x /p')"
        own="$own, ${r#*:} $(( nb - ${rk:-0} )) (first at solve ${rf:-?})"
    done
    echo "     $c: OpenFOAM's own runs against its serial, iteration counts that differ:${own#,}"
done

# LTS: the flow time scales step by step, and rDeltaT at every write
for c in $LCASES; do
    read -r _ nb no wb wbs <<< "$(python3 "$LTS" flowcmp "$W/b_$c/log" "$W/of_$c/log")"
    read -r _ _ _ w2 _ <<< "$(python3 "$LTS" flowcmp "$W/of_${c}_np2/log" "$W/of_$c/log")"
    read -r _ _ _ w4 _ <<< "$(python3 "$LTS" flowcmp "$W/of_${c}_np4/log" "$W/of_$c/log")"
    bd=$(mx "$BOUND" "$(python3 -c "print(10*max(float('${w2:-nan}'), float('${w4:-nan}')))")")
    if [ -n "$nb" ] && [ "$nb" = "$no" ] && [ "$nb" -gt 0 ] && le "$wb" "$bd"
    then ok "$c (CUDA): the Flow time scale min/max of all $nb steps within $bd (worst $wb at step $wbs; OpenFOAM in two $w2, in four $w4)"
    else fail "$c (CUDA): flow time scales: ${nb:-?} lines against ${no:-?}, worst ${wb:-?} at step ${wbs:-?} against $bd"; fi
    over=""; worst=-1; wl=""
    for t in $(times "$W/of_$c"); do
        vb=$(python3 "$FIELDS" absdiff "$W/b_$c/$t" "$W/of_$c/$t" rDeltaT | awk '/^absdiff/ {print $4}')
        v2=$(python3 "$FIELDS" absdiff "$W/of_${c}_np2/$t" "$W/of_$c/$t" rDeltaT | awk '/^absdiff/ {print $4}')
        v4=$(python3 "$FIELDS" absdiff "$W/of_${c}_np4/$t" "$W/of_$c/$t" rDeltaT | awk '/^absdiff/ {print $4}')
        bd=$(mx "$BOUND" "$(python3 -c "print(10*max(float('${v2:-nan}'), float('${v4:-nan}')))")")
        le "${vb:-nan}" "$bd" || over="$over $t(${vb:-nan}>$bd)"
        r=$(python3 -c "print('%.3g' % (float('${vb:-nan}')/float('$bd')))")
        python3 -c "import sys; sys.exit(0 if float('$r') > float('$worst') else 1)" 2> /dev/null && { worst=$r; wl="$vb at t = $t (OpenFOAM in two $v2, in four $v4)"; }
    done
    if [ -z "$over" ]; then ok "$c (CUDA): rDeltaT at every write time within ten times OpenFOAM's own spread; closest ($worst of the bound): $wl"
    else fail "$c (CUDA): rDeltaT above the bound at$over"; fi
done

# janaf's premise: both coefficient sets in use
for c in biconic sod_janaf sod_janaf_visc; do
    t=$(times "$W/of_$c" | awk '{print $NF}')
    read -r _ tmin tmax nlo nhi <<< "$(python3 "$LTS" janaf "$W/of_$c" "$t")"
    if [ "${nlo:-0}" -gt 0 ] && [ "${nhi:-0}" -gt 0 ]
    then ok "$c: premise, OpenFOAM's T at t = $t spans $tmin to $tmax K: $nlo cells below Tcommon (1000 K), $nhi at or above it"
    else fail "$c: premise, OpenFOAM's T at t = $t ($tmin to $tmax K) does not cross Tcommon"; fi
done

# the walls: U and T on their patch at every write
walls()   # walls <case> <patch>
{
    local c=$1 pt=$2 t f over="" worst=-1 wl="" vb v2 v4 bd r
    for t in $(times "$W/of_$c"); do
        for f in U T; do
            read -r vb v2 v4 <<< "$(python3 -c "
import sys; sys.path.insert(0, '$ROOT/tests')
import numpy as np, rho_central_fields as rf
o = np.asarray(rf.patch_value('$W/of_$c/$t/$f', '$pt'), float)
s = np.abs(o).max()
print(' '.join('%.3e' % (np.abs(np.asarray(rf.patch_value(d + '/$t/$f', '$pt'), float) - o).max()/s)
               for d in ('$W/b_$c', '$W/of_${c}_np2', '$W/of_${c}_np4')))")"
            bd=$(mx "$BOUND" "$(python3 -c "print(10*max(float('${v2:-nan}'), float('${v4:-nan}')))")")
            le "${vb:-nan}" "$bd" || over="$over $t:$f(${vb:-nan}>$bd)"
            r=$(python3 -c "print('%.3g' % (float('${vb:-nan}')/float('$bd')))")
            python3 -c "import sys; sys.exit(0 if float('$r') > float('$worst') else 1)" 2> /dev/null && { worst=$r; wl="$f $vb at t = $t (OpenFOAM in two $v2, in four $v4)"; }
        done
    done
    if [ -z "$over" ]; then ok "$c (CUDA): U and T on $pt within ten times OpenFOAM's own spread at every write time; closest ($worst of the bound): $wl"
    else fail "$c (CUDA): $pt values above the bound at$over"; fi
    t=$(times "$W/of_$c" | awk '{print $NF}')
    echo "     $c, $pt at t = $t: $(python3 -c "
import sys; sys.path.insert(0, '$ROOT/tests')
import numpy as np, rho_central_fields as rf
U = np.asarray(rf.patch_value('$W/of_$c/$t/U', '$pt'), float); T = np.asarray(rf.patch_value('$W/of_$c/$t/T', '$pt'), float)
print('OpenFOAM slip velocity up to %.6g m/s, wall temperature %.6g to %.6g K' % (np.sqrt((U**2).sum(axis=1)).max(), T.min(), T.max()))")"
}
walls biconic cone
walls channel_slip wall

# as shipped
if grep -q "writeFormat: 'binary' -- brae's field writer emits ascii only" "$W/b_biconic/err" \
    && grep -q "^writeFormat *binary;" "$W/b_biconic/system/controlDict" && ls "$W/b_biconic/0/"*.gz > /dev/null 2>&1
then ok "biconic as shipped: brae read the gzipped 0 and the binary mesh, and says it writes ascii where the case asks for binary"
else fail "biconic as shipped: the case or brae's notice is not what was shipped"; fi

# ---- conservation and stability ------------------------------------------------------------------------------------------
for c in $ECASES; do balance "b_$c" raw; vbalance "b_$c"; identity "b_$c"; done
for c in $LCASES; do
    bad=""
    for t in $(times "$W/b_$c"); do
        read -r _ r0 p0 T0 nan _ <<< "$(python3 "$FIELDS" stats "$W/b_$c" "$t")"
        python3 -c "import sys; sys.exit(0 if float('$r0') > 0 and float('$p0') > 0 and float('$T0') > 0 and int('$nan') == 0 else 1)" 2> /dev/null || bad="$bad $t"
    done
    if [ -z "$bad" ]; then ok "b_$c stable: rho, p, T positive and finite at all $(times "$W/b_$c" | wc -w) write times"; else fail "b_$c: non-positive or NaN at$bad"; fi
done

# ---- memory and runtime (reported) ------------------------------------------------------------------------------------------
for c in $CASES; do
    echo "     runtime $c: OpenFOAM $(grep -h '^ExecutionTime' "$W/of_$c/log" | tail -n 1 | awk '{print $3}') s; brae CUDA $(sed -n 's/^diag steps \([^ ]*\) step-seconds \([^ ]*\) per-step-us \([^ ]*\).*/\2 s (\1 steps, \3 us a step)/p' "$W/b_$c/log"), device memory $(sed -n 's/^diag device-memory \([^ ]*\) MiB/\1 MiB/p' "$W/b_$c/log")"
done

# ---- controls ---------------------------------------------------------------------------------------------------------
red "ltsUnsmoothed, fs_lts" "$W/cltsUnsmoothed_fs_lts" "$W/of_fs_lts" "$(spread fs_lts)"
red "janafLowOnly, sod_janaf" "$W/cjanafLowOnly_sod_janaf" "$W/of_sod_janaf" "$(spread sod_janaf)"
for k in alphaheOldT maxwellNoCreep maxwellNoCurvature smoluchowskiNoJump; do
    red "$k, channel_slip" "$W/c${k}_channel_slip" "$W/of_channel_slip" "$(spread channel_slip)"
done
opensummary
echo "rho_central_lts_vs_openfoam: $passes passed, $fails failed (work in $W)"
[ "$fails" = 0 ] && [ "$passes" -ge "${BRAE_RCF_I_MIN_PASSES:-1}" ]
