#!/usr/bin/env bash
# brae_rhoCentralFoam against OpenFOAM v2412 rhoCentralFoam on milestone J: compressible RAS, kOmegaSST, as
# rhoCentralFoam constructs and calls it. Single GPU, static mesh.
#
# WHAT IS PORTED (OpenFOAM v2412), and where it is called
#   construction createFields.H: compressible::turbulenceModel::New(rho, U, phi, thermo) -- constant/turbulenceProperties
#                (laminar | RAS), RASModel kOmegaSST, its coeffs (kOmegaSSTCoeffs, Prt for alphat), k omega nut alphat
#                MUST_READ with their patch classes; wallDist meshWave (F1, F2) and nearWallDist (the wall functions)
#   validate()   rhoCentralFoam.C:71, kOmegaSSTBase::correctNut() on the fields as read: nut = a1 k/max(a1 omega, b1 F2
#                sqrt(S2)), the wall functions' nut_w, alphat = rho nut/Prt (EddyDiffusivity), the constraint patches
#   muEff        :220, muEff = rho*nut + mu (CompressibleTurbulenceModel), on the rho and nut the step starts from, into
#                the U corrector's laplacian and tauMC; inviscid is still decided by the laminar mu (:216)
#   alphaEff     :278, thermo.alphaEff(alphat) = CpByCpv*(alpha + alphat) into the e corrector
#   correct()    :290, after p, p_b and rho_b: the omega and k equations (kOmegaSSTBase::correct; brae's shared kOmegaSST
#                closure, which every compressible solver of OpenFOAM's runs too) on this step's rho, U, the central
#                mass flux phi as phi/fvc::interpolate(rho), mu of the last thermo.correct(), fvm::ddt(rho, .) on
#                rho.oldTime() under the case's ddt scheme (Euler or localEuler), then correctNut and alphat
#   writes       k omega nut alphat (AUTO_WRITE)
# CASES (tests/rho_central_turb.py)
#   plate_sst      a supersonic turbulent flat plate (air, sutherland, Mach 1.9, a symmetry plane ahead of the leading
#                  edge), OpenFOAM's wall functions on the plate (nutkWallFunction, kqRWallFunction, omegaWallFunction,
#                  compressible::alphatWallFunction), 5500 cells graded to the wall, Euler, maxCo 0.2, to 3e-3 s
#                  (21585 steps), writes every 5e-4
#   plate_lts      the same with ddtSchemes localEuler, 1500 steps, writes every 500
#   jet_sst        LadenburgJet60psi as shipped plus kOmegaSST: an axisymmetric wedge with no wall-type patch (wall
#                  distance GREAT, F1 = 0: the k-epsilon branch in the shear layers), to 2e-5 s as shipped
# REFERENCE  OpenFOAM serial; its spread in two (simple) and in four (frozen scotch partitions: turbPlate, frozen for this
#   gate, and LadenburgJet60psi's) -- the envelope of tests/rho_central_gate_lib.sh, unchanged: ten times OpenFOAM's own
#   spread, floor BOUND 8e-14 -- and, a DIAGNOSTIC that bounds nothing, OpenFOAM serial with p one ulp up
# CHECKS, DECLARED BEFORE THE GATE FIRST RAN (2026-11-18), the CUDA arm
#   per case     steps, every deltaT and name, Courant, directories, U p T rho at every write time (arm)
#   TURBULENCE   k omega nut alphat at every write time within ten times OpenFOAM's own spread of them (floor BOUND)
#   WALL         plate_sst, plate_lts: nut alphat k omega on the plate at every write within ten times OpenFOAM's own
#                spread of them (floor BOUND)
#   SOLVES       the same solves in the same order (Ux Uy e, then omega and k); every iteration count OpenFOAM serial's;
#                OpenFOAM's own runs' differing counts reported
#   PREMISE      OpenFOAM's nut/nu above 10 somewhere at the last write (all cases), and the plate's wall function
#                active (nut_w > 0) on some of its faces
#   CONSERVATION plate_sst, jet_sst (Euler): positivity; the residual-accounted balance and every solve's identity
#                (vbalance, identity); the raw balance reported (hist). plate_lts: positivity only (a pseudo-time march)
#   MEMORY, RUNTIME (reported)
# CONTROLS (rhoCentralState.cuh), red against OpenFOAM by more than 100 x the bound, on jet_sst:
#   rasLaminarMuEff (:220 without mut), rasNoAlphat (:278 without alphat), rasCorrectFirst (correct() before the U
#   corrector, on the inviscid U), rasPhiUndivided (the model's phi() the mass flux itself)
# REFUSED (tests/rho_central_refusals.sh): RAS models other than kOmegaSST, LES, turbulence off, unported k|omega|nut|
#   alphat patch types and schemes, wallDist methods other than meshWave, a bounded div(phi,k) under LTS, a start
#   directory without k; here: the host reference arm with RAS
# NOT CLAIMED  other RAS models, LES, mesh motion, multi-GPU, performance (GaussSeidel sweeps on the CPU).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BRAE_RCF_BIN:-$ROOT/build/brae_rhoCentralFoam}"
STEPS="$ROOT/tests/rho_central_steps.py"
FIELDS="$ROOT/tests/rho_central_fields.py"
TURB="$ROOT/tests/rho_central_turb.py"
OFBASHRC=${OF_BASHRC:-${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}}
TUT="${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}"
RCF="$TUT/compressible/rhoCentralFoam"
[ -x "$BIN" ] || { echo "SKIP: no brae_rhoCentralFoam at $BIN"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: no OpenFOAM v2412 at $OFBASHRC"; exit 77; }
[ -d "$RCF/LadenburgJet60psi/0.orig" ] || { echo "SKIP: no LadenburgJet60psi at $RCF"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
W=${BRAE_GATE_OUT:-$(mktemp -d /tmp/brae-rcf-turb.XXXXXX)}
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
    cp -r "$RCF/LadenburgJet60psi/system" "$RCF/LadenburgJet60psi/constant" "$d/"
    if [ "$k" = jet_sst ]
    then
        cp -r "$RCF/LadenburgJet60psi/0.orig" "$d/0"
        gunzip -f "$d/0/"*.gz || { fail "stage $k: 0.orig"; return 1; }
    fi
    python3 "$TURB" setup "$k" "$d" || { fail "stage $k: setup"; return 1; }
    of "cd '$d' && blockMesh > log.blockMesh 2>&1" < /dev/null || fail "stage $k: blockMesh"
    return 0
}
CASES="plate_sst plate_lts jet_sst"
ECASES="plate_sst jet_sst"   # Euler in time: the residual-accounted balance applies
for c in $CASES; do stage "$W/$c" "$c"; done

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
for k in rasLaminarMuEff rasNoAlphat rasCorrectFirst rasPhiUndivided; do control "$k" jet_sst; done
# the host reference arm refuses RAS
copy "$W/jet_sst" "$W/h_jet_sst"
(cd "$W/h_jet_sst" && BRAE_RHOCENTRALFOAM_ARM=cpu "$BIN" -case . > log 2>&1 < /dev/null; echo $? > rc)
if [ "$(cat "$W/h_jet_sst/rc")" != 0 ] && grep -q "host reference arm .* does not run a RAS model" "$W/h_jet_sst/log"
then ok "the host reference arm refuses RAS by name: $(grep -o 'the host reference arm[^;]*' "$W/h_jet_sst/log" | head -n 1)"
else fail "host reference arm with RAS: exit $(cat "$W/h_jet_sst/rc")"; fi

# ---- comparisons ------------------------------------------------------------------------------------------------------
spread() { echo "$W/of_${1}_np2 $W/of_${1}_np4"; }
for c in $CASES; do arm "$c (CUDA)" "$W/b_$c" "$W/of_$c" "$(spread "$c")" "$BOUND" exact; done

# the turbulence fields at every write
for c in $CASES; do
    over=""; worst=-1; wl=""
    for t in $(times "$W/of_$c"); do
        for f in k omega nut alphat; do
            vb=$(python3 "$FIELDS" absdiff "$W/b_$c/$t" "$W/of_$c/$t" "$f" | awk '/^absdiff/ {print $4}')
            v2=$(python3 "$FIELDS" absdiff "$W/of_${c}_np2/$t" "$W/of_$c/$t" "$f" | awk '/^absdiff/ {print $4}')
            v4=$(python3 "$FIELDS" absdiff "$W/of_${c}_np4/$t" "$W/of_$c/$t" "$f" | awk '/^absdiff/ {print $4}')
            vu=$(python3 "$FIELDS" absdiff "$W/of_${c}_ulp/$t" "$W/of_$c/$t" "$f" | awk '/^absdiff/ {print $4}')
            bd=$(mx "$BOUND" "$(python3 -c "print(10*max(float('${v2:-nan}'), float('${v4:-nan}')))")")
            le "${vb:-nan}" "$bd" || over="$over $t:$f(${vb:-nan}>$bd)"
            r=$(python3 -c "print('%.3g' % (float('${vb:-nan}')/float('$bd')))")
            python3 -c "import sys; sys.exit(0 if float('$r') > float('$worst') else 1)" 2> /dev/null \
                && { worst=$r; wl="$f $vb at t = $t (OpenFOAM in two $v2, in four $v4; one ulp up, diagnostic, $vu)"; }
        done
    done
    if [ -z "$over" ]; then ok "$c (CUDA): k omega nut alphat at every write time within ten times OpenFOAM's own spread; closest ($worst of the bound): $wl"
    else fail "$c (CUDA): turbulence fields above the bound at$over"; fi
done

# the plate's wall: nut alphat k omega on its faces at every write
twall()   # twall <case> <patch>
{
    local c=$1 pt=$2 t f over="" worst=-1 wl="" vb v2 v4 bd r
    for t in $(times "$W/of_$c"); do
        for f in nut alphat k omega; do
            read -r vb v2 v4 <<< "$(python3 -c "
import sys; sys.path.insert(0, '$ROOT/tests')
import numpy as np, rho_central_fields as rf
def pv(d):
    v = np.asarray(rf.patch_value(d + '/$t/$f', '$pt'), float)
    return v
o = pv('$W/of_$c'); s = max(np.abs(o).max(), 1e-300)
out = []
for d in ('$W/b_$c', '$W/of_${c}_np2', '$W/of_${c}_np4'):
    v = pv(d)
    if v.size == 1 and o.size > 1: v = np.full(o.shape, v[0])
    oo = o if o.size == v.size else np.full(v.shape, o[0])
    out.append('%.3e' % (np.abs(v - oo).max()/s))
print(' '.join(out))")"
            bd=$(mx "$BOUND" "$(python3 -c "print(10*max(float('${v2:-nan}'), float('${v4:-nan}')))")")
            le "${vb:-nan}" "$bd" || over="$over $t:$f(${vb:-nan}>$bd)"
            r=$(python3 -c "print('%.3g' % (float('${vb:-nan}')/float('$bd')))")
            python3 -c "import sys; sys.exit(0 if float('$r') > float('$worst') else 1)" 2> /dev/null && { worst=$r; wl="$f $vb at t = $t (OpenFOAM in two $v2, in four $v4)"; }
        done
    done
    if [ -z "$over" ]; then ok "$c (CUDA): nut alphat k omega on $pt within ten times OpenFOAM's own spread at every write time; closest ($worst of the bound): $wl"
    else fail "$c (CUDA): $pt turbulence values above the bound at$over"; fi
}
twall plate_sst plate
twall plate_lts plate

# the solves
for c in $CASES; do
    k=$(cmpdir "$W/b_$c")
    python3 "$STEPS" solves "$W/b_$c/log" "$W/of_$c/log" > "$k/solves.cmp" 2>&1
    read -r nb no same <<< "$(sed -n 's/^solves //p' "$k/solves.cmp")"
    read -r iok ifirst <<< "$(sed -n 's/^iterations //p' "$k/solves.cmp")"
    read -r iw iws iwf imed <<< "$(sed -n 's/^initial //p' "$k/solves.cmp")"
    if [ -n "$nb" ] && [ "$nb" = "$no" ] && [ "$same" = 1 ] && [ "$nb" -gt 0 ] && grep -q "Solving for omega" "$W/b_$c/log"
    then ok "$c (CUDA): the same $nb solves as OpenFOAM's, in the same order (omega and k among them)"
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

# the premise: turbulent
for c in $CASES; do
    t=$(times "$W/of_$c" | awk '{print $NF}')
    wall=""; [ "$c" != jet_sst ] && wall=plate
    read -r _ rmax rmed won wn <<< "$(python3 "$TURB" premise "$W/of_$c" "$t" $wall)"
    if python3 -c "import sys; sys.exit(0 if float('${rmax:-0}') > 10 else 1)" && { [ -z "$wall" ] || [ "${won:-0}" -gt 0 ]; }
    then ok "$c: premise, OpenFOAM's nut/nu at t = $t up to $rmax (median $rmed)${wall:+; the wall function active on $won of $wn plate faces}"
    else fail "$c: premise, OpenFOAM's nut/nu at t = $t up to ${rmax:-?}${wall:+, wall function on ${won:-?} of ${wn:-?} faces}"; fi
done

# ---- conservation and stability ------------------------------------------------------------------------------------------
for c in $ECASES; do balance "b_$c" raw; vbalance "b_$c"; identity "b_$c"; done
bad=""
for t in $(times "$W/b_plate_lts"); do
    read -r _ r0 p0 T0 nan _ <<< "$(python3 "$FIELDS" stats "$W/b_plate_lts" "$t")"
    python3 -c "import sys; sys.exit(0 if float('$r0') > 0 and float('$p0') > 0 and float('$T0') > 0 and int('$nan') == 0 else 1)" 2> /dev/null || bad="$bad $t"
done
if [ -z "$bad" ]; then ok "b_plate_lts stable: rho, p, T positive and finite at all $(times "$W/b_plate_lts" | wc -w) write times"; else fail "b_plate_lts: non-positive or NaN at$bad"; fi

# ---- memory and runtime (reported) ------------------------------------------------------------------------------------------
for c in $CASES; do
    echo "     runtime $c: OpenFOAM $(grep -h '^ExecutionTime' "$W/of_$c/log" | tail -n 1 | awk '{print $3}') s; brae CUDA $(sed -n 's/^diag steps \([^ ]*\) step-seconds \([^ ]*\) per-step-us \([^ ]*\).*/\2 s (\1 steps, \3 us a step)/p' "$W/b_$c/log"), device memory $(sed -n 's/^diag device-memory \([^ ]*\) MiB/\1 MiB/p' "$W/b_$c/log")"
done

# ---- controls ---------------------------------------------------------------------------------------------------------
for k in rasLaminarMuEff rasNoAlphat rasCorrectFirst rasPhiUndivided; do
    red "$k, jet_sst" "$W/c${k}_jet_sst" "$W/of_jet_sst" "$(spread jet_sst)"
done
opensummary
echo "rho_central_turb_vs_openfoam: $passes passed, $fails failed (work in $W)"
[ "$fails" = 0 ] && [ "$passes" -ge "${BRAE_RCF_J_MIN_PASSES:-1}" ]
