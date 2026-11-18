#!/usr/bin/env bash
# brae_rhoCentralFoam against OpenFOAM v2412 rhoCentralFoam on Riemann problems and the reconstruction limiters
# (milestone F): interpolationSchemes reconstruct(rho), reconstruct(T) and reconstruct(U) with each of OpenFOAM's TVD
# limiters brae ports -- vanLeer, vanAlbada, Minmod, MUSCL, SuperBee and limitedLinear 1, their V forms on U -- in 1-D
# and in 2-D, and against the exact solution.
#
# CASES (tests/rho_central_riemann.py: OpenFOAM's shockTube on 1000 cells, x in [-5, 5], a perfect gas with R = 1 and
#   gamma = 1.4, adjustTimeStep as shipped, writePrecision 17, four adjustable writes)
#   <problem>_<limiter>  problem sod (Toro 1, t = 2), lax (t = 1.3), 123 (Toro 2, near vacuum, t = 1.5), strong (Toro 3,
#                        p 1000 against 0.01, t = 0.12); limiter on rho and T, its V form on U
#   lax_Minmod_T, strong_vanAlbada_T   the same with fluxScheme Tadmor
#   ob_<limiter>         obliqueShock (2-D, Mach 2.9, 1800 cells, writePrecision 17, to t = 10) with the limiter: the V
#                        forms on a velocity with two components
#   lax_SuperBee_fixed, ob_MUSCL_fixed, ob_SuperBee_fixed   the three cases whose adaptive step sequence parts from
#                        OpenFOAM serial's (open items), with adjustTimeStep off: deltaT 5e-4 (lax) and the tutorial's
#                        0.0025 (obliqueShock), every 50th step written. Checked like every other arm (milestone G)
# REFERENCE  OpenFOAM serial; its spread decomposed in two (simple) and in four (the frozen scotch partitions,
#   tests/rho_central_gate_lib.sh), each reported on its own. A case OpenFOAM itself does not finish is not compared: the
#   gate asserts that brae stops on it too, and says where each stopped.
# ENVELOPE  tests/rho_central_gate_lib.sh's, unchanged.
# DIAGNOSTIC (reported, never a bound; milestone G) OpenFOAM serial started with p one ulp up on every case with an open
#   check: each arm reports brae's distance from serial against the one-ulp run's. Open checks are listed, with their
#   category, in tests/data/rho_central_open_checks.tsv and stay FAILs. For the divergent cases the gate reports the first
#   step each run parts from serial (adaptive), the step each run moves 1e-12, 1e-8, 1e-4 away (fixed deltaT), and the
#   shock each run carries at the last write (tests/rho_central_divergence.py).
# CHECKS
#   per arm      steps, every deltaT (the same digits: adjustable writes), names, Courant, directories, U p T rho
#   EXACT        the exact Riemann solution (Toro ch. 4, every wave pattern) against brae's and OpenFOAM's final fields:
#                L1 of rho, u, p for both (reported), and the star state
#   PREMISE      each limiter changes the solution: OpenFOAM's run with it differs from OpenFOAM's vanLeer run by more
#                than 1e-6 (a limiter brae ignored could not pass)
# CONTROLS (rhoCentralState.cuh), red against OpenFOAM by more than 100 x the bound:
#   limiterVanLeer     vanLeer and vanLeerV whatever the case names: sod_Minmod, ob_SuperBee
#   limiterScalarOnV   the V limiter built from rhoU_x alone: ob_Minmod. NOT visible in 1-D, where rhoU has one component
# NOT CLAIMED  the scalar-form limiters on U (limitFuncs::magSqr; refused), the other limitedSchemes (QUICK, UMIST,
#   Gamma, SFCD, filteredLinear, limitedCubic... refused), linear reconstruction, viscous flow, 3-D.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BRAE_RCF_BIN:-$ROOT/build/brae_rhoCentralFoam}"
STEPS="$ROOT/tests/rho_central_steps.py"
FIELDS="$ROOT/tests/rho_central_fields.py"
RIEMANN="$ROOT/tests/rho_central_riemann.py"
DIVERGE="$ROOT/tests/rho_central_divergence.py"
OFBASHRC=${OF_BASHRC:-${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}}
TUT="${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}/compressible/rhoCentralFoam"
[ -x "$BIN" ] || { echo "SKIP: no brae_rhoCentralFoam at $BIN"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: no OpenFOAM v2412 at $OFBASHRC"; exit 77; }
[ -d "$TUT/shockTube/0.orig" ] && [ -d "$TUT/obliqueShock/0" ] || { echo "SKIP: no shockTube/obliqueShock at $TUT"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
python3 -c 'import numpy' 2> /dev/null || { echo "SKIP: no numpy"; exit 77; }
W=${BRAE_GATE_OUT:-$(mktemp -d /tmp/brae-rcf-riemann.XXXXXX)}
mkdir -p "$W"
export PYTHONDONTWRITEBYTECODE=1
BOUND=${BRAE_RCF_SOD_BOUND:-8e-14}
fails=0
passes=0
. "$ROOT/tests/rho_central_gate_lib.sh"
J=${BRAE_GATE_JOBS:-10}
throttle() { while [ "$(jobs -rp | wc -l)" -ge "$J" ]; do sleep 1; done; }
LIMS="vanLeer vanAlbada Minmod MUSCL SuperBee limitedLinear_1"
PROBS="sod lax 123 strong"
schemes()   # schemes <dir> <limiter, _ for a space> [Tadmor]
{
    local lim=${2//_/ } base=${2%%_*} rest=""
    [ "$base" != "$2" ] && rest=" ${2#*_}"
    sed -i -E "s/^(\s*reconstruct\(rho\)\s+)[^;]+;/\1$lim;/; s/^(\s*reconstruct\(T\)\s+)[^;]+;/\1$lim;/; s/^(\s*reconstruct\(U\)\s+)[^;]+;/\1${base}V$rest;/" "$1/system/fvSchemes"
    [ -z "${3:-}" ] || sed -i -E "s/^(fluxScheme\s+)\S+;/\1$3;/" "$1/system/fvSchemes"
    grep -qE "reconstruct\(U\)\s+${base}V$rest;" "$1/system/fvSchemes" || fail "stage $(basename "$1"): schemes"
}

# ---- the cases --------------------------------------------------------------------------------------------------
tube()   # tube <dir> <problem> <limiter> [Tadmor]
{
    local d=$1
    rm -rf "$d"
    mkdir -p "$d"
    cp -r "$TUT/shockTube/system" "$TUT/shockTube/constant" "$TUT/shockTube/0.orig" "$d/"
    cp -r "$TUT/shockTube/0.orig" "$d/0"
    sed -i -E 's/\(100 1 1\)/(1000 1 1)/' "$d/system/blockMeshDict"
    sed -i -E 's/^(writePrecision\s+)\S+;/\117;/' "$d/system/controlDict"
    python3 "$RIEMANN" setup "$2" "$d" || { fail "stage $(basename "$d"): setup"; return 1; }
    schemes "$d" "$3" "${4:-}"
    of "cd '$d' && blockMesh > log.blockMesh 2>&1 && setFields > log.setFields 2>&1" < /dev/null || fail "stage $(basename "$d"): mesh"
}
oblique()   # oblique <dir> <limiter>
{
    local d=$1
    rm -rf "$d"
    mkdir -p "$d"
    cp -r "$TUT/obliqueShock/system" "$TUT/obliqueShock/constant" "$TUT/obliqueShock/0" "$d/"
    sed -i -E 's/^(writePrecision\s+)\S+;/\117;/' "$d/system/controlDict"
    schemes "$d" "$2"
    of "cd '$d' && blockMesh > log.blockMesh 2>&1" < /dev/null || fail "stage $(basename "$d"): mesh"
}
fixdt()   # fixdt <dir> <deltaT> <n>: adjustTimeStep off, deltaT fixed, every n-th step written
{
    sed -i -E "s/^(adjustTimeStep\s+)\S+;/\1no;/; s/^(deltaT\s+)\S+;/\1$2;/; s/^(writeControl\s+)\S+;/\1timeStep;/; s/^(writeInterval\s+)\S+;/\1$3;/" "$1/system/controlDict"
    grep -qE "^adjustTimeStep\s+no;" "$1/system/controlDict" && grep -qE "^writeInterval\s+$3;" "$1/system/controlDict" || fail "stage $(basename "$1"): fixed deltaT"
}
ALL=""
for p in $PROBS; do for l in $LIMS; do tube "$W/${p}_$l" "$p" "$l"; ALL="$ALL ${p}_$l"; done; done
tube "$W/lax_Minmod_T" lax Minmod Tadmor
tube "$W/strong_vanAlbada_T" strong vanAlbada Tadmor
ALL="$ALL lax_Minmod_T strong_vanAlbada_T"
for l in $LIMS; do oblique "$W/ob_$l" "$l"; ALL="$ALL ob_$l"; done
# the cases whose adaptive step sequence parts from OpenFOAM serial's, again with the step fixed (obliqueShock: the
# tutorial's own deltaT), every 50th step written: the fields compared with no deltaT sequence to part
FIXED="lax_SuperBee_fixed ob_MUSCL_fixed ob_SuperBee_fixed"
tube "$W/lax_SuperBee_fixed" lax SuperBee && fixdt "$W/lax_SuperBee_fixed" 5e-04 50
oblique "$W/ob_MUSCL_fixed" MUSCL && fixdt "$W/ob_MUSCL_fixed" 0.0025 50
oblique "$W/ob_SuperBee_fixed" SuperBee && fixdt "$W/ob_SuperBee_fixed" 0.0025 50
ALL="$ALL $FIXED"
for c in $ALL; do copy "$W/$c" "$W/of_$c"; done
# OpenFOAM serial, quietly: a case it does not finish is a finding, checked below, not a failure of the gate
for c in $ALL; do throttle; of "cd '$W/of_$c' && rhoCentralFoam > log 2>&1" < /dev/null & done
wait
# the cases OpenFOAM finishes are compared; the others are checked to stop in brae too
DONE=""; STOPPED=""
for c in $ALL; do if grep -qx End "$W/of_$c/log"; then DONE="$DONE $c"; else STOPPED="$STOPPED $c"; fi; done
for c in $DONE; do copy "$W/$c" "$W/of_${c}_np2"; copy "$W/$c" "$W/of_${c}_np4"; done
for c in $DONE; do throttle; ofnp "$W/of_${c}_np2" 2 & done
wait
for c in $DONE; do throttle; ofnp "$W/of_${c}_np4" 4 & done
wait
for c in $DONE; do for n in 2 4; do grep -qx End "$W/of_${c}_np$n/log" || fail "OpenFOAM in $n did not finish $c"; done; done
# OpenFOAM serial with p one ulp up on the cases with open checks: a DIAGNOSTIC reference beside serial, in two and in
# four (arm() reports brae against it; it bounds nothing)
ULP=""; for c in $DONE; do case $c in strong_*|lax_SuperBee*|ob_MUSCL*|ob_SuperBee*) ULP="$ULP $c" ;; esac; done
for c in $ULP; do throttle; ofulp "$W/$c" "$W/of_${c}_ulp" 0 & done
wait
for c in $ULP; do grep -qx End "$W/of_${c}_ulp/log" || echo "     diagnostic: OpenFOAM with p one ulp up did not finish $c"; done

# ---- brae ---------------------------------------------------------------------------------------------------------
HOST="sod_vanLeer sod_vanAlbada sod_Minmod sod_MUSCL sod_SuperBee sod_limitedLinear_1 lax_Minmod_T ob_Minmod ob_SuperBee
      strong_vanLeer strong_vanAlbada strong_Minmod strong_MUSCL strong_SuperBee strong_limitedLinear_1 strong_vanAlbada_T
      $FIXED"
for c in $DONE; do copy "$W/$c" "$W/b_$c"; braerun "$W/b_$c"; done
for c in $HOST; do case " $DONE " in *" $c "*) copy "$W/$c" "$W/h_$c"; braerun "$W/h_$c" BRAE_RHOCENTRALFOAM_ARM=cpu ;; esac; done
for c in $STOPPED; do copy "$W/$c" "$W/b_$c"; ctlrun "$W/b_$c"; done
copy "$W/sod_Minmod" "$W/cV_sod_Minmod"; ctlrun "$W/cV_sod_Minmod" BRAE_CONTROL_RCF=limiterVanLeer
copy "$W/ob_SuperBee" "$W/cV_ob_SuperBee"; ctlrun "$W/cV_ob_SuperBee" BRAE_CONTROL_RCF=limiterVanLeer
copy "$W/ob_Minmod" "$W/cS_ob_Minmod"; ctlrun "$W/cS_ob_Minmod" BRAE_CONTROL_RCF=limiterScalarOnV

# ---- premises: each limiter changes the solution ---------------------------------------------------------------------
for c in $DONE; do
    ref=${c%%_*}_vanLeer; case $c in ob_*) ref=ob_vanLeer ;; esac
    [ "$c" = "$ref" ] && continue
    case $c in *_fixed) continue ;; esac
    case " $DONE " in *" $ref "*) ;; *) continue ;; esac
    tl=$(times "$W/of_$c" | awk '{print $NF}')
    v=$(python3 "$FIELDS" maxrel "$W/of_$c/$tl" "$W/of_$ref/$tl" | awk '{print $2}')
    if python3 -c "import sys; sys.exit(0 if float('${v:-0}') > 1e-6 else 1)"
    then ok "premise: OpenFOAM's $c differs from its $ref by $v at t = $tl (the limiter is exercised)"
    else fail "premise: OpenFOAM's $c is $v from its $ref: the limiter changes nothing here"; fi
done

# ---- where OpenFOAM itself stops ------------------------------------------------------------------------------------
for c in $STOPPED; do
    no=$(grep -c "^Time = " "$W/of_$c/log"); why=$(grep -m 1 -o "Negative initial temperature[^\n]*\|Floating point exception\|sigFpe" "$W/of_$c/log" | head -n 1)
    nb=$(grep -c "^Time = " "$W/b_$c/log")
    if [ "$(cat "$W/b_$c/rc")" != 0 ] && grep -q "unphysical state" "$W/b_$c/err"
    then ok "$c: OpenFOAM v2412 stops at its step $no (${why:-aborts}); brae stops too, at its step $((nb + 1)) (an unphysical state: brae checks positivity a step before OpenFOAM's temperature seed)"
    else fail "$c: OpenFOAM stops at step $no and brae does not (exit $(cat "$W/b_$c/rc"))"; fi
done

# ---- comparisons ------------------------------------------------------------------------------------------------------
spread() { echo "$W/of_${1}_np2 $W/of_${1}_np4"; }
for c in $DONE; do
    m=exact; case $c in *_fixed) m=fixed ;; esac
    arm "$c (CUDA)" "$W/b_$c" "$W/of_$c" "$(spread "$c")" "$BOUND" $m
    [ -d "$W/h_$c" ] && arm "$c (host reference)" "$W/h_$c" "$W/of_$c" "$(spread "$c")" "$BOUND" $m
done

# ---- OpenFOAM's own reproducibility (reported, NOT a bound) --------------------------------------------------------------
# (1) how many steps OpenFOAM's decomposed runs keep serial's deltaT sequence, beside brae's: on the cases where the steps
# part, OpenFOAM's own runs part too. (2) OpenFOAM serial started with p one unit in the last place up (a fixed,
# reproducible perturbation: OpenFOAM's response to the last bit of its own state), against serial, beside brae.
for c in $DONE; do
    sb=$(python3 "$STEPS" compare "$W/b_$c/log" "$W/of_$c/log" | sed -n 's/^deltaT \([0-9]*\).*/\1/p')
    s2=$(python3 "$STEPS" compare "$W/of_${c}_np2/log" "$W/of_$c/log" | sed -n 's/^deltaT \([0-9]*\).*/\1/p')
    s4=$(python3 "$STEPS" compare "$W/of_${c}_np4/log" "$W/of_$c/log" | sed -n 's/^deltaT \([0-9]*\).*/\1/p')
    ns=$(grep -c "^Time = " "$W/of_$c/log")
    [ "$sb" = "$ns" ] && [ "$s2" = "$ns" ] && [ "$s4" = "$ns" ] && continue
    echo "     reproducibility, $c ($ns steps): OpenFOAM serial's deltaT kept by OpenFOAM in two for $s2 steps, in four for $s4, by brae CUDA for $sb"
done
for c in $ULP; do
    case $c in *_fixed) continue ;; esac
    case $c in ob_*|lax_*) echo "     reproducibility, $c: OpenFOAM serial with p one ulp up keeps serial's deltaT for $(python3 "$STEPS" compare "$W/of_${c}_ulp/log" "$W/of_$c/log" | sed -n 's/^deltaT \([0-9]*\).*/\1/p') steps of $(grep -c "^Time = " "$W/of_$c/log")" ;; esac
    for t in $(times "$W/of_$c" | cut -d' ' -f1-2); do
        if [ -f "$W/of_${c}_ulp/$t/U" ]; then
            read -r _ cu fu ru _ <<< "$(python3 "$FIELDS" worst "$W/of_${c}_ulp/$t" "$W/of_$c/$t")"
        else   # said, not left blank
            ru="NOT RUN"; fu="no $t/U in of_${c}_ulp"; cu="-"
        fi
        read -r _ cb fb rb _ <<< "$(python3 "$FIELDS" worst "$W/b_$c/$t" "$W/of_$c/$t")"
        hh=""; [ -d "$W/h_$c" ] && hh=", host $(python3 "$FIELDS" worst "$W/h_$c/$t" "$W/of_$c/$t" | awk '{print $4}')"
        echo "     sensitivity, $c at t = $t: OpenFOAM with p one ulp up $ru ($fu, cell $cu); brae CUDA $rb ($fb, cell $cb)$hh; OpenFOAM in two $(python3 "$FIELDS" maxrel "$W/of_${c}_np2/$t" "$W/of_$c/$t" | awk '{print $2}'), in four $(python3 "$FIELDS" maxrel "$W/of_${c}_np4/$t" "$W/of_$c/$t" | awk '{print $2}')"
    done
done

# ---- the divergent cases (reported, NOT a bound): where the runs part, and the shocks they carry ---------------------------
# adaptive: the first step whose deltaT or time name parts from OpenFOAM serial's, for brae and for OpenFOAM's own runs;
# fixed deltaT: the first step (every 50th written) where each run is above 1e-12, 1e-8 and 1e-4 of the field's largest
# |value| away from serial, and its largest distance; then the shock at the last write, by every run (and the exact one)
for c in lax_SuperBee ob_MUSCL ob_SuperBee; do
    case " $DONE " in *" $c "*) ;; *) continue ;; esac
    line=""
    for r in "brae CUDA=b_$c" "brae host=h_$c" "OpenFOAM in two=of_${c}_np2" "OpenFOAM in four=of_${c}_np4" "OpenFOAM, p one ulp up=of_${c}_ulp"; do
        [ -f "$W/${r#*=}/log" ] || continue
        read -r _ k tk <<< "$(python3 "$STEPS" compare "$W/${r#*=}/log" "$W/of_$c/log" | grep '^firstdiff')"
        line="$line; ${r%%=*} at step $k (after t = $tk)"
    done
    echo "     divergence, $c, adaptive: the first step parting from OpenFOAM serial's${line#;}"
    f=${c}_fixed
    case " $DONE " in *" $f "*) ;; *) continue ;; esac
    dt=$(sed -n 's/^deltaT[[:space:]]*\([^;]*\);.*/\1/p' "$W/$f/system/controlDict")
    runs=""; for r in brae_CUDA=b_$f brae_host=h_$f OpenFOAM_in_two=of_${f}_np2 OpenFOAM_in_four=of_${f}_np4 OpenFOAM_one_ulp_up=of_${f}_ulp; do
        [ -d "$W/${r#*=}" ] && runs="$runs ${r%%=*}=$W/${r#*=}"; done
    # shellcheck disable=SC2086
    python3 "$DIVERGE" trace "$W/of_$f" "$dt" $runs > "$W/cmp/divergence_$f.txt" 2>&1
    sed -n 's/^trace \([^ ]*\) \([^ ]*\) \([^ ]*\) \([^ ]*\) \([^ ]*\) \([^ ]*\) \([^ ]*\)$/     divergence, '"$f"' (deltaT '"$dt"'): \1 above 1e-12 from step \2, 1e-8 from \3, 1e-4 from \4; largest \5 (\7, step \6)/p' "$W/cmp/divergence_$f.txt"
    for g in $c $f; do
        tl=$(times "$W/of_$g" | awk '{print $NF}')
        for r in "OpenFOAM serial=of_$g" "brae CUDA=b_$g" "brae host=h_$g" "OpenFOAM in two=of_${g}_np2" "OpenFOAM in four=of_${g}_np4" "OpenFOAM, p one ulp up=of_${g}_ulp"; do
            [ -f "$W/${r#*=}/$tl/p" ] || continue
            case $g in
                ob_*) q=$(python3 "$FIELDS" shock "$W/${r#*=}" "$tl" oblique | awk '{printf "incident %s deg, reflected %s deg, impact x %s", $3, $5, $7}') ;;
                *)    q=$(python3 "$DIVERGE" shock1d "$W/${r#*=}" "$tl" | awk '{printf "shock x %s, p2 %s, rho2 %s (exact %s, %s, %s)", $3, $5, $7, $9, $11, $13}') ;;
            esac
            echo "     shock, $g at t = $tl, ${r%%=*}: $q"
        done
    done
done

# ---- the exact solution (reported) ---------------------------------------------------------------------------------------
for c in $DONE; do
    case $c in ob_*) continue ;; esac
    tl=$(times "$W/of_$c" | awk '{print $NF}')
    eo=$(python3 "$RIEMANN" exact "$W/of_$c" "$tl" | tr '\n' ' '); eb=$(python3 "$RIEMANN" exact "$W/b_$c" "$tl" | tail -n 1)
    echo "     exact, $c at t = $tl: $(echo "$eo" | cut -d' ' -f2-5); OpenFOAM $(echo "$eo" | sed 's/.*L1 /L1 /'); brae $eb"
done

# ---- conservation and stability ---------------------------------------------------------------------------------------
for c in sod_Minmod lax_SuperBee strong_MUSCL ob_vanAlbada; do case " $DONE " in *" $c "*) balance "b_$c" ;; esac; done

# ---- controls ---------------------------------------------------------------------------------------------------------
red "limiterVanLeer, sod_Minmod" "$W/cV_sod_Minmod" "$W/of_sod_Minmod" "$(spread sod_Minmod)"
red "limiterVanLeer, ob_SuperBee" "$W/cV_ob_SuperBee" "$W/of_ob_SuperBee" "$(spread ob_SuperBee)"
red "limiterScalarOnV, ob_Minmod" "$W/cS_ob_Minmod" "$W/of_ob_Minmod" "$(spread ob_Minmod)"
opensummary
echo "rho_central_riemann_vs_openfoam: $passes passed, $fails failed (work in $W)"
[ "$fails" = 0 ] && [ "$passes" -ge 213 ]
