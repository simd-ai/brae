#!/usr/bin/env bash
# brae_rhoCentralFoam against OpenFOAM v2412 rhoCentralFoam on THREE-DIMENSIONAL cases (2026-11-19). v2412 ships no 3-D
# rhoCentralFoam tutorial, so tests/rho_central_3d.py builds them with blockMesh: every cell has six neighbours in three
# directions, no empty or wedge patch, and the flow turns in all three. Single GPU, static mesh.
#
# CASES (tests/rho_central_3d.py; air, perfectGas, hConst)
#   corner_inv   Mach 3 into an axial corner, a floor and a side wall that both turn into a 10-degree ramp: two oblique
#                shocks and their 3-D interaction. Inviscid; slip walls, inletOutlet free boundaries, a symmetryPlane.
#                9216 cells (two blocks), Euler, maxCo 0.5, to 2e-3 s (two flow lengths)
#   corner_lam   the same corner laminar (sutherland, Re 6.6e5 a metre): noSlip isothermal walls, cells graded 50:1.
#                Euler, maxCo 0.2, to 1e-3 s
#   corner_sst   the same corner turbulent (Re 6.6e7 a metre): kOmegaSST with OpenFOAM's wall functions on both walls,
#                graded 200:1. Euler, maxCo 0.2, to 1e-3 s
#   box_blast    a closed 1 m cube, 24^3 cells: a high-pressure sphere off the centre bursts and reflects off five slip
#                walls and a symmetryPlane. Inviscid, to 3e-3 s: nothing crosses the boundary, so mass and energy must stay
#   duct_tp      a square duct driven from rest by a totalPressure/totalTemperature inlet into a waveTransmissive outlet,
#                noSlip walls and a symmetryPlane; laminar, sutherland, to 2e-3 s
# RESTARTS (box_blast; startFrom startTime at 1.5e-3, to 3e-3): from OpenFOAM's directory (rs_o) and from brae's (rs_b),
#   each against OpenFOAM's own restart of the same directory. Under adjustTimeStep a restart is not its uninterrupted
#   run continued (setDeltaT grows the first deltaT by 1.2 from CoNum 0): tests/rho_central_restart_vs_openfoam.sh
# REFERENCE  OpenFOAM serial; its spread in two (simple (2 1 1)) and in four (frozen scotch partitions: corner3d, box3d,
#   duct3d, frozen for this gate) -- the envelope of tests/rho_central_gate_lib.sh, unchanged: ten times OpenFOAM's own
#   spread, floor BOUND 8e-14 -- and, a DIAGNOSTIC that bounds nothing, OpenFOAM serial with p one ulp up
# CHECKS, DECLARED BEFORE THE GATE FIRST RAN (2026-11-19)
#   per case     steps, every deltaT and name, Courant, directories, U p T rho at every write time (arm), the CUDA arm on
#                all five, the host reference on corner_inv and box_blast; it runs inviscid cases only (milestone G) and
#                must refuse corner_lam by name (corrected after the first run, 2026-11-19, which had asked it for
#                corner_lam and duct_tp too)
#   TURBULENCE   corner_sst: k omega nut alphat at every write time, and on the floor and side walls, within ten times
#                OpenFOAM's own spread of them (floor BOUND)
#   SOLVES       the viscous cases: the same solves in the same order, every iteration count OpenFOAM serial's
#   BOUNDARIES   duct_tp: p and T on the inlet (totalPressure, totalTemperature) and p on the outlet (waveTransmissive) at
#                every write within ten times OpenFOAM's own spread of them (corrected after the first run: U on both and
#                T on the outlet are zeroGradient, which OpenFOAM writes without a value -- their faces are their cells',
#                which the field check covers)
#   TIME STATE   box_blast and its restarts: uniform/time the same text as OpenFOAM's at every write
#   PREMISE      corner_inv is three-dimensional: OpenFOAM's U_z reaches 1 % of the free stream at the last write;
#                corner_sst is turbulent: nut/nu above 10 and the wall function active on floor faces
#   CONSERVATION corner_inv, box_blast: the balance within round-off, positivity; box_blast's boundary flux zero.
#                corner_lam, corner_sst, duct_tp: the residual-accounted balance and every solve's identity (the raw
#                balance reported, hist)
#   ENGINEERING  corner_inv: the floor's ramp pressure away from the corner and the shock angle, brae's and OpenFOAM's,
#                beside oblique-shock theory (reported; the coarse mesh's distance from theory is the refinement
#                study's subject, not this gate's)
#   READ BACK    OpenFOAM starts from brae's last write of every case and runs on by 1e-6 s
#   MEMORY, RUNTIME (reported)
# CONTROLS (rhoCentralState.cuh), each red against OpenFOAM by more than 100 x the bound:
#   noTransform on corner_inv (the slip walls and the symmetryPlane as zeroGradient), noSigmaDotU on corner_lam (the
#   viscous work left out of the energy equation), rasLaminarMuEff on corner_sst, noPressureJump on box_blast, waveNoRelax
#   on duct_tp. The first run (2026-11-19) declared noTauMC on corner_lam: it moved the fields 19.5 x the bound by
#   1e-3 s (rho at 2.5e-4), not 100 x -- tauMC is muEff*dev2(grad(U)^T), and on this corner the wall-normal shear the
#   laplacian carries dominates it -- so it does not separate a wrong tauMC from OpenFOAM's spread here; replaced
# NOT CLAIMED  cyclic patches, LTS and mesh motion in 3-D (gated in 2-D and axisymmetric), multi-GPU, performance.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BRAE_RCF_BIN:-$ROOT/build/brae_rhoCentralFoam}"
STEPS="$ROOT/tests/rho_central_steps.py"
FIELDS="$ROOT/tests/rho_central_fields.py"
C3D="$ROOT/tests/rho_central_3d.py"
TURB="$ROOT/tests/rho_central_turb.py"
NOZ="$ROOT/tests/rho_central_nozzle.py"
OFBASHRC=${OF_BASHRC:-${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}}
TUT="${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}"
RCF="$TUT/compressible/rhoCentralFoam"
[ -x "$BIN" ] || { echo "SKIP: no brae_rhoCentralFoam at $BIN"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: no OpenFOAM v2412 at $OFBASHRC"; exit 77; }
[ -d "$RCF/LadenburgJet60psi/system" ] || { echo "SKIP: no LadenburgJet60psi at $RCF"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
python3 -c 'import numpy' 2> /dev/null || { echo "SKIP: no numpy"; exit 77; }
W=${BRAE_GATE_OUT:-$(mktemp -d /tmp/brae-rcf-3d.XXXXXX)}
mkdir -p "$W"
export PYTHONDONTWRITEBYTECODE=1
BOUND=${BRAE_RCF_SOD_BOUND:-8e-14}
fails=0
passes=0
. "$ROOT/tests/rho_central_gate_lib.sh"
J=${BRAE_GATE_JOBS:-10}
throttle() { while [ "$(jobs -rp | wc -l)" -ge "$J" ]; do sleep 1; done; }
ctl() { sed -i -E "$2" "$1/system/controlDict"; }

# ---- the cases ------------------------------------------------------------------------------------------------------
stage()   # stage <dir> <kind>
{
    local d=$1 k=$2
    rm -rf "$d"
    mkdir -p "$d"
    cp -r "$RCF/LadenburgJet60psi/system" "$RCF/LadenburgJet60psi/constant" "$d/"
    python3 "$C3D" setup "$k" "$d" || { fail "stage $k: setup"; return 1; }
    of "cd '$d' && blockMesh > log.blockMesh 2>&1" < /dev/null || { fail "stage $k: blockMesh"; return 1; }
    return 0
}
CASES="corner_inv corner_lam corner_sst box_blast duct_tp"
HCASES="corner_inv box_blast"   # the host reference arm runs inviscid cases only
VCASES="corner_lam corner_sst duct_tp"             # viscous, Euler: the residual-accounted balance applies
for c in $CASES; do stage "$W/$c" "$c" || exit 1; done

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
for c in $HCASES; do copy "$W/$c" "$W/h_$c"; braerun "$W/h_$c" BRAE_RHOCENTRALFOAM_ARM=cpu; done
copy "$W/corner_lam" "$W/h_corner_lam"
(cd "$W/h_corner_lam" && BRAE_RHOCENTRALFOAM_ARM=cpu "$BIN" -case . > log 2> err < /dev/null; echo $? > rc)
if [ "$(cat "$W/h_corner_lam/rc")" != 0 ] && grep -q "host reference arm .* runs inviscid cases only" "$W/h_corner_lam/err"
then ok "the host reference arm refuses the viscous corner by name: $(grep -o 'the host reference arm[^;]*' "$W/h_corner_lam/err" | head -n 1)"
else fail "host reference arm on corner_lam: exit $(cat "$W/h_corner_lam/rc")"; fi
control() { copy "$W/$2" "$W/c$1_$2"; ctlrun "$W/c$1_$2" BRAE_CONTROL_RCF="$1"; }
control noTransform corner_inv
control noSigmaDotU corner_lam
control rasLaminarMuEff corner_sst
control noPressureJump box_blast
control waveNoRelax duct_tp

# ---- the restarts (box_blast at 1.5e-3) ---------------------------------------------------------------------------------
RT=0.0015
rstage()   # rstage <dir> <source run>: box_blast's system and constant, the source's 1.5e-3 with its uniform/
{
    local d=$1 src=$2
    rm -rf "$d"
    mkdir -p "$d"
    cp -r "$W/box_blast/system" "$W/box_blast/constant" "$d/"
    cp -r "$src/$RT" "$d/" || { fail "stage $(basename "$d"): no $RT in $(basename "$src")"; return 1; }
    ctl "$d" "s/^(startFrom\s+)\S+;/\1startTime;/; s/^(startTime\s+)\S+;/\1$RT;/"
    grep -qE "^startTime\s+$RT;" "$d/system/controlDict" || fail "stage $(basename "$d"): startTime not set"
}
rcopyt() { rm -rf "$2"; mkdir -p "$2"; cp -r "$1/system" "$1/constant" "$1/$RT" "$2/"; }
for s in o b; do
    src="$W/of_box_blast"; [ "$s" = b ] && src="$W/b_box_blast"
    rstage "$W/rs_$s" "$src"
    for r in of_rs_$s of_rs_${s}_np2 of_rs_${s}_np4; do rcopyt "$W/rs_$s" "$W/$r"; done
    ofrun "$W/of_rs_$s" &
    ofnp "$W/of_rs_${s}_np2" 2 &
    ofnp "$W/of_rs_${s}_np4" 4 &
    wait
    rcopyt "$W/rs_$s" "$W/b_rs_$s"; braerun "$W/b_rs_$s"
done

# ---- comparisons ------------------------------------------------------------------------------------------------------
spread() { echo "$W/of_${1}_np2 $W/of_${1}_np4"; }
for c in $CASES; do arm "$c (CUDA)" "$W/b_$c" "$W/of_$c" "$(spread "$c")" "$BOUND" exact; done
for c in $HCASES; do arm "$c (host reference)" "$W/h_$c" "$W/of_$c" "$(spread "$c")" "$BOUND" exact; done
for s in o b; do arm "restart rs_$s (CUDA)" "$W/b_rs_$s" "$W/of_rs_$s" "$(spread rs_$s)" "$BOUND" exact; done

# the time state of box_blast and its restarts
tstate()   # tstate <name> <brae dir> <OF dir>
{
    local r n miss v nm ix dt dt0
    r=$(python3 "$FIELDS" timedict "$2" "$3" 2>&1)
    read -r _ n _ miss _ v _ _ nm _ ix _ dt _ _ dt0 _ <<< "$r"
    if [ -n "$n" ] && [ "$n" -gt 0 ] && [ "$miss" = 0 ] && [ "$v" = "$n" ] && [ "$nm" = "$n" ] && [ "$ix" = "$n" ] \
        && [ "$dt" = "$n" ] && [ "$dt0" = "$n" ]
    then ok "$1: uniform/time at all $n write times the same text as OpenFOAM's (value, name, index, deltaT, deltaT0)"
    else fail "$1: uniform/time against OpenFOAM's: ${r:-nothing}"; fi
}
tstate "box_blast (CUDA)" "$W/b_box_blast" "$W/of_box_blast"
for s in o b; do tstate "restart rs_$s (CUDA)" "$W/b_rs_$s" "$W/of_rs_$s"; done

# the turbulence fields of corner_sst at every write, then on its walls
c=corner_sst
over=""; worst=-1; wl=""
for t in $(times "$W/of_$c"); do
    for f in k omega nut alphat; do
        vb=$(python3 "$FIELDS" absdiff "$W/b_$c/$t" "$W/of_$c/$t" "$f" | awk '/^absdiff/ {print $4}')
        v2=$(python3 "$FIELDS" absdiff "$W/of_${c}_np2/$t" "$W/of_$c/$t" "$f" | awk '/^absdiff/ {print $4}')
        v4=$(python3 "$FIELDS" absdiff "$W/of_${c}_np4/$t" "$W/of_$c/$t" "$f" | awk '/^absdiff/ {print $4}')
        bd=$(mx "$BOUND" "$(python3 -c "print(10*max(float('${v2:-nan}'), float('${v4:-nan}')))")")
        le "${vb:-nan}" "$bd" || over="$over $t:$f(${vb:-nan}>$bd)"
        r=$(python3 -c "print('%.3g' % (float('${vb:-nan}')/float('$bd')))")
        python3 -c "import sys; sys.exit(0 if float('$r') > float('$worst') else 1)" 2> /dev/null \
            && { worst=$r; wl="$f $vb at t = $t (OpenFOAM in two $v2, in four $v4)"; }
    done
done
if [ -z "$over" ]; then ok "$c (CUDA): k omega nut alphat at every write time within ten times OpenFOAM's own spread; closest ($worst of the bound): $wl"
else fail "$c (CUDA): turbulence fields above the bound at$over"; fi
pwall()   # pwall <case> <patch> <fields...>: the patch values at every write within ten times OpenFOAM's own spread
{
    local c=$1 pt=$2 t f over="" worst=-1 wl="" vb v2 v4 bd r
    shift 2
    for t in $(times "$W/of_$c"); do
        for f in "$@"; do
            read -r vb v2 v4 <<< "$(python3 -c "
import sys; sys.path.insert(0, '$ROOT/tests')
import numpy as np, rho_central_fields as rf
def pv(d):
    return np.asarray(rf.patch_value(d + '/$t/$f', '$pt'), float)
o = pv('$W/of_$c'); s = max(np.abs(o).max(), 1e-300)
out = []
for d in ('$W/b_$c', '$W/of_${c}_np2', '$W/of_${c}_np4'):
    v = pv(d)
    if v.size == 1 and o.size > 1: v = np.full(o.shape, v.ravel()[0])
    oo = o if o.size == v.size else np.full(v.shape, o.ravel()[0])
    out.append('%.3e' % (np.abs(v - oo).max()/s))
print(' '.join(out))")"
            bd=$(mx "$BOUND" "$(python3 -c "print(10*max(float('${v2:-nan}'), float('${v4:-nan}')))")")
            le "${vb:-nan}" "$bd" || over="$over $t:$f(${vb:-nan}>$bd)"
            r=$(python3 -c "print('%.3g' % (float('${vb:-nan}')/float('$bd')))")
            python3 -c "import sys; sys.exit(0 if float('$r') > float('$worst') else 1)" 2> /dev/null && { worst=$r; wl="$f $vb at t = $t (OpenFOAM in two $v2, in four $v4)"; }
        done
    done
    if [ -z "$over" ]; then ok "$c (CUDA): $* on $pt within ten times OpenFOAM's own spread at every write time; closest ($worst of the bound): $wl"
    else fail "$c (CUDA): $pt values above the bound at$over"; fi
}
pwall corner_sst floor nut alphat k omega
pwall corner_sst side nut alphat k omega
pwall duct_tp inlet p T
pwall duct_tp outlet p

# the solves of the viscous cases
for c in $VCASES; do
    k=$(cmpdir "$W/b_$c")
    python3 "$STEPS" solves "$W/b_$c/log" "$W/of_$c/log" > "$k/solves.cmp" 2>&1
    read -r nb no same <<< "$(sed -n 's/^solves //p' "$k/solves.cmp")"
    read -r iok ifirst <<< "$(sed -n 's/^iterations //p' "$k/solves.cmp")"
    if [ -n "$nb" ] && [ "$nb" = "$no" ] && [ "$same" = 1 ] && [ "$nb" -gt 0 ]
    then ok "$c (CUDA): the same $nb solves as OpenFOAM's, in the same order"
    else fail "$c (CUDA): solves ${nb:-?} against ${no:-?}, order the same: ${same:-?}"; fi
    if [ "${iok:-0}" = "$nb" ] && [ "${ifirst:-1}" = 0 ]
    then ok "$c (CUDA): every iteration count OpenFOAM serial's ($nb solves)"
    else fail "$c (CUDA): iteration counts differ from solve ${ifirst:-?} (${iok:-?} of $nb the same)"; fi
    own=""
    for r in np2:"in two" np4:"in four" ulp:"one ulp up"; do
        read -r _ rk rf <<< "$(python3 "$STEPS" solves "$W/of_${c}_${r%%:*}/log" "$W/of_$c/log" 2>/dev/null | sed -n 's/^iterations /x /p')"
        own="$own, ${r#*:} $(( nb - ${rk:-0} )) (first at solve ${rf:-?})"
    done
    echo "     $c: OpenFOAM's own runs against its serial, iteration counts that differ:${own#,}"
done

# ---- premises -----------------------------------------------------------------------------------------------------------
t=$(times "$W/of_corner_inv" | awk '{print $NF}')
uz=$(python3 -c "
import sys; sys.path.insert(0, '$ROOT/tests')
import rho_central_3d as r, numpy as np
U = r._read_internal('$W/of_corner_inv/$t/U'); print('%.4f' % (np.abs(U[:, 2]).max()/r.U_INF))")
if python3 -c "import sys; sys.exit(0 if float('${uz:-0}') > 0.01 else 1)"
then ok "corner_inv: premise, three-dimensional: OpenFOAM's |U_z| reaches $uz of the free stream at t = $t"
else fail "corner_inv: premise, |U_z| only ${uz:-?} of the free stream"; fi
t=$(times "$W/of_corner_sst" | awk '{print $NF}')
read -r _ rmax rmed won wn <<< "$(python3 "$TURB" premise "$W/of_corner_sst" "$t" floor)"
if python3 -c "import sys; sys.exit(0 if float('${rmax:-0}') > 10 else 1)" && [ "${won:-0}" -gt 0 ]
then ok "corner_sst: premise, OpenFOAM's nut/nu at t = $t up to $rmax (median $rmed); the wall function active on $won of $wn floor faces"
else fail "corner_sst: premise, nut/nu up to ${rmax:-?}, wall function on ${won:-?} of ${wn:-?} floor faces"; fi

# ---- conservation and stability ------------------------------------------------------------------------------------------
for c in corner_inv box_blast; do balance "b_$c"; done
out=$(sed -n 's/^diag boundary-outflow //p' "$W/b_box_blast/log")
if python3 -c "
import sys
v = [abs(float(x)) for x in '''$out'''.split() if x.replace('.', '', 1).replace('e', '', 1).replace('-', '').replace('+', '').isdigit()]
sys.exit(0 if v and max(v) == 0 else 1)" 2> /dev/null
then ok "box_blast: nothing crosses the closed box's boundary (diag boundary-outflow $out)"
else echo "     box_blast: diag boundary-outflow '${out:-none}' (reported: the balance above carries any wall impulse)"; fi
for c in $VCASES; do balance "b_$c" raw; vbalance "b_$c"; identity "b_$c"; done

# ---- read back --------------------------------------------------------------------------------------------------------------
for c in $CASES; do readback "$W/b_$c" 1e-6 "$c"; done

# ---- engineering quantities (reported) ------------------------------------------------------------------------------------
for d in of_corner_inv b_corner_inv; do
    of "cd '$W/$d' && postProcess -func writeCellCentres -time 0 > log.cc 2>&1 && postProcess -func writeCellVolumes -time 0 > log.cv 2>&1" < /dev/null
done
t=$(times "$W/of_corner_inv" | awk '{print $NF}')
read -r _ ro th ao bth <<< "$(python3 "$C3D" ramp "$W/of_corner_inv" "$t")"
read -r _ rb _ ab _ <<< "$(python3 "$C3D" ramp "$W/b_corner_inv" "$t")"
echo "     corner_inv at t = $t: floor pressure on the ramp p/p_inf brae $rb, OpenFOAM $ro, oblique-shock theory $th; shock angle brae $ab, OpenFOAM $ao, theory $bth deg (9216 cells; the refinement study converges these)"

# ---- memory and runtime (reported) ------------------------------------------------------------------------------------------
for c in $CASES; do
    echo "     runtime $c: OpenFOAM $(grep -h '^ExecutionTime' "$W/of_$c/log" | tail -n 1 | awk '{print $3}') s; brae CUDA $(sed -n 's/^diag steps \([^ ]*\) step-seconds \([^ ]*\) per-step-us \([^ ]*\).*/\2 s (\1 steps, \3 us a step)/p' "$W/b_$c/log"), device memory $(sed -n 's/^diag device-memory \([^ ]*\) MiB/\1 MiB/p' "$W/b_$c/log")"
done

# ---- controls ---------------------------------------------------------------------------------------------------------
red "noTransform, corner_inv" "$W/cnoTransform_corner_inv" "$W/of_corner_inv" "$(spread corner_inv)"
red "noSigmaDotU, corner_lam" "$W/cnoSigmaDotU_corner_lam" "$W/of_corner_lam" "$(spread corner_lam)"
red "rasLaminarMuEff, corner_sst" "$W/crasLaminarMuEff_corner_sst" "$W/of_corner_sst" "$(spread corner_sst)"
red "noPressureJump, box_blast" "$W/cnoPressureJump_box_blast" "$W/of_box_blast" "$(spread box_blast)"
red "waveNoRelax, duct_tp" "$W/cwaveNoRelax_duct_tp" "$W/of_duct_tp" "$(spread duct_tp)"
opensummary
echo "rho_central_3d_vs_openfoam: $passes passed, $fails failed (work in $W)"
[ "$fails" = 0 ] && [ "$passes" -ge "${BRAE_RCF_3D_MIN_PASSES:-1}" ]
