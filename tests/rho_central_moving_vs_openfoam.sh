#!/usr/bin/env bash
# brae_rhoCentralFoam against OpenFOAM v2412 rhoCentralFoam on milestone K: mesh motion -- dynamicMotionSolverFvMesh with
# velocityComponentLaplacian, OpenFOAM's movingCone tutorial. Single GPU, laminar, Euler.
#
# WHAT IS PORTED (OpenFOAM v2412), in rhoCentralFoam.C's order
#   mesh.update()   after setDeltaT.H and ++runTime (:91-97): dynamicMotionSolverFvMesh::update = fvMesh::movePoints(
#                   motionSolver::newPoints()) then U.correctBoundaryConditions(). velocityComponentLaplacian (brae's
#                   shared src/fvMotionSolver/fvMotionSolvers/componentVelocity/laplacian): the directional diffusivity,
#                   pointMotionU<c>'s fixed values written in patch order, cellMotion's face average of the point field,
#                   laplacian(diffusivity,cellMotionU) as the case asks (Gauss linear uncorrected), PCG/DIC as fvSolution
#                   asks, volPointInterpolation, points[c] += deltaT*pointMotionU<c>, twoDCorrectPoints (a wedge:
#                   snapToWedge); fvMesh::movePoints (brae's shared DynamicMotionSolverFvMesh): V0 once a time index,
#                   meshPhi = face::sweptVol*rDeltaT, the geometry rebuilt
#   fluxes          :126-133 phiv_pos/neg -= meshPhi; :213-218 phiEp += meshPhi*(a_pos*p_pos + a_neg*p_neg)
#   ddt             fvm::ddt on the moving mesh: the old-time source on V0 (rho, rhoU, rhoE; the U and e correctors'
#                   fvm::ddt(rho, .) - fvc::ddt(rho, .), the latter rDeltaT*(rho*vf - rho0*vf0*V0/V))
#   BCs             movingWallVelocity (Up from face::centre at the old and new points, Un = meshPhi_b/(magSf + VSMALL)),
#                   pressureInletOutletVelocity (directionMixed, valueFraction neg(phi)*(I - nn)), totalPressure's rho
#                   form (p0 - 0.5*rho_b*neg(phi)*|U_b|^2: psi none)
#   writes          polyMesh/points, meshPhi, cellMotionU<c>, pointMotionU<c> at every write (brae writes ascii)
#   restart         from a time directory with its polyMesh/points (polyMesh's points instance) and cellMotionU<c>
# CASES
#   mc_shipped   movingCone AS SHIPPED (writeFormat binary, writePrecision 6, purgeWrite 5; U and e symGaussSeidel
#                1e-5 relTol 0.1, cellMotionUx PCG/DIC 1e-8): the last five writes compared
#   mc_tight     the same with U and e at tolerance 1e-10 relTol 0 and cellMotionUx at 1e-14, writes every 2e-6
#                (ascii, 17 digits, purgeWrite 0): the shipped solves stop at relTol 0.1, and OpenFOAM's own parallel
#                runs then part from its serial by 1e-5 to 3e-4 -- too wide to bound the motion or show a control red
#   mc_restart   mc_tight restarted from OpenFOAM's own 1e-5 (its points, fields, cellMotionUx), against OpenFOAM's own
#                restart from the same directory
# REFERENCE the envelope of tests/rho_central_gate_lib.sh: ten times OpenFOAM's own spread (two, simple; four, frozen
#   scotch `movingCone`), floor BOUND; OpenFOAM serial with p one ulp up a diagnostic
# CHECKS, DECLARED BEFORE THE GATE FIRST RAN, the CUDA arm
#   per case     steps, deltaT, names, Courant, directories, U p T rho at every write (arm)
#   MESH         the points, meshPhi, cellMotionUx and pointMotionUx at every write within ten times OpenFOAM's own
#                spread of them (floor BOUND)
#   SOLVES       mc_tight: the same solves in the same order (cellMotionUx among them); every iteration count OpenFOAM
#                serial's
#   GCL          brae's discrete space-conservation residual max |V - V0 - deltaT*sum(meshPhi)|/V over every cell and
#                step, at round-off (reported by brae, `diag gcl`)
#   CONSERVATION mc_tight: positivity; the residual-accounted balance and every solve's identity (vbalance, identity);
#                the raw balance reported (hist)
#   MEMORY, RUNTIME (reported)
# CONTROLS (rhoCentralState.cuh), red against OpenFOAM by more than 100 x the bound, on mc_tight:
#   motionFluxAbsolute (no meshPhi in the fluxes), motionDdtStatic (ddt on V, not V0), motionWallStatic
#   (movingWallVelocity's Uwall 0), motionNoSnap (no twoDCorrectPoints)
# REFUSED (tests/rho_central_refusals.sh): other dynamicFvMesh and motionSolver types, LTS with motion, a
#   pointMotionU<c> condition or Function1 not ported; here: the host reference arm with motion
# NOT CLAIMED  other motion solvers in rhoCentralFoam, multi-GPU, binary output, performance.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BRAE_RCF_BIN:-$ROOT/build/brae_rhoCentralFoam}"
STEPS="$ROOT/tests/rho_central_steps.py"
FIELDS="$ROOT/tests/rho_central_fields.py"
MOV="$ROOT/tests/rho_central_moving.py"
OFBASHRC=${OF_BASHRC:-${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}}
TUT="${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}"
RCF="$TUT/compressible/rhoCentralFoam"
[ -x "$BIN" ] || { echo "SKIP: no brae_rhoCentralFoam at $BIN"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: no OpenFOAM v2412 at $OFBASHRC"; exit 77; }
[ -d "$RCF/movingCone/0" ] || { echo "SKIP: no movingCone at $RCF"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
W=${BRAE_GATE_OUT:-$(mktemp -d /tmp/brae-rcf-moving.XXXXXX)}
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
    cp -r "$RCF/movingCone/0" "$RCF/movingCone/constant" "$RCF/movingCone/system" "$d/"
    python3 "$MOV" setup "$k" "$d" || { fail "stage $k: setup"; return 1; }
    of "cd '$d' && blockMesh > log.blockMesh 2>&1" < /dev/null || fail "stage $k: blockMesh"
    return 0
}
CASES="mc_shipped mc_tight"
for c in $CASES; do stage "$W/$c" "$c"; done

# ---- OpenFOAM ---------------------------------------------------------------------------------------------------------
oset() { for c in "$@"; do copy "$W/$c" "$W/of_$c"; copy "$W/$c" "$W/of_${c}_np2"; copy "$W/$c" "$W/of_${c}_np4"; done; }
orun()   # orun <case> <start time>
{
    local c=$1 st=$2
    throttle; ofrun "$W/of_$c" &
    throttle; ofulp "$W/$c" "$W/of_${c}_ulp" "$st" &
    wait
    throttle; ofnp "$W/of_${c}_np2" 2 &
    throttle; ofnp "$W/of_${c}_np4" 4 &
    wait
}
oset $CASES
for c in $CASES; do orun "$c" 0; done
# the restart: OpenFOAM's own mc_tight at 1e-5 (its moved points, its fields, cellMotionUx, pointMotionUx)
python3 "$MOV" restart "$W/of_mc_tight" 1e-05 "$W/mc_restart" || fail "stage mc_restart"
copyr() { rm -rf "$2"; cp -r "$1" "$2"; }
for r in "" _np2 _np4; do copyr "$W/mc_restart" "$W/of_mc_restart$r"; done
copyr "$W/mc_restart" "$W/of_mc_restart_ulp"
python3 "$FIELDS" nudge "$W/of_mc_restart_ulp/1e-05/p" > /dev/null
throttle; ofrun "$W/of_mc_restart" &
throttle; ofrun "$W/of_mc_restart_ulp" &
wait
throttle; ofnp "$W/of_mc_restart_np2" 2 &
throttle; ofnp "$W/of_mc_restart_np4" 4 &
wait
ALL="mc_shipped mc_tight mc_restart"
for c in $ALL; do for r in "" _np2 _np4 _ulp; do grep -qx End "$W/of_$c$r/log" || fail "OpenFOAM${r:+ $r} did not finish $c"; done; done

# ---- brae -------------------------------------------------------------------------------------------------------------
for c in mc_shipped mc_tight; do copy "$W/$c" "$W/b_$c"; braerun "$W/b_$c"; done
copyr "$W/mc_restart" "$W/b_mc_restart"; braerun "$W/b_mc_restart"
control() { copy "$W/mc_tight" "$W/c$1_mc_tight"; ctlrun "$W/c$1_mc_tight" BRAE_CONTROL_RCF="$1"; }
for k in motionFluxAbsolute motionDdtStatic motionWallStatic motionPiovStatic; do control "$k"; done
copy "$W/mc_tight" "$W/h_mc_tight"
(cd "$W/h_mc_tight" && BRAE_RHOCENTRALFOAM_ARM=cpu "$BIN" -case . > log 2>&1 < /dev/null; echo $? > rc)
if [ "$(cat "$W/h_mc_tight/rc")" != 0 ] && grep -q "host reference arm .* does not move a mesh" "$W/h_mc_tight/log"
then ok "the host reference arm refuses mesh motion by name: $(grep -o 'the host reference arm[^;]*' "$W/h_mc_tight/log" | head -n 1)"
else fail "host reference arm with mesh motion: exit $(cat "$W/h_mc_tight/rc")"; fi

# ---- comparisons ------------------------------------------------------------------------------------------------------
spread() { echo "$W/of_${1}_np2 $W/of_${1}_np4"; }
for c in $ALL; do arm "$c (CUDA)" "$W/b_$c" "$W/of_$c" "$(spread "$c")" "$BOUND" exact; done

# the mesh: points, meshPhi, cellMotionUx, pointMotionUx at every write
for c in $ALL; do
    over=""; worst=-1; wl=""
    for t in $(times "$W/of_$c"); do
        for f in points meshPhi cellMotionUx pointMotionUx; do
            if [ "$f" = points ]; then
                rel() { python3 "$FIELDS" pointsdiff "$1/$t/polyMesh/points" "$W/of_$c/$t/polyMesh/points" | awk '{print $3}'; }
            else
                rel() { python3 "$FIELDS" absdiff "$1/$t" "$W/of_$c/$t" "$f" | awk '/^absdiff/ {print $4}'; }
            fi
            vb=$(rel "$W/b_$c"); v2=$(rel "$W/of_${c}_np2"); v4=$(rel "$W/of_${c}_np4"); vu=$(rel "$W/of_${c}_ulp")
            bd=$(mx "$BOUND" "$(python3 -c "print(10*max(float('${v2:-nan}'), float('${v4:-nan}')))")")
            le "${vb:-nan}" "$bd" || over="$over $t:$f(${vb:-nan}>$bd)"
            r=$(python3 -c "print('%.3g' % (float('${vb:-nan}')/float('$bd')))")
            python3 -c "import sys; sys.exit(0 if float('$r') > float('$worst') else 1)" 2> /dev/null \
                && { worst=$r; wl="$f $vb at t = $t (OpenFOAM in two $v2, in four $v4; one ulp up, diagnostic, $vu)"; }
        done
    done
    if [ -z "$over" ]; then ok "$c (CUDA): points, meshPhi, cellMotionUx and pointMotionUx at every write within ten times OpenFOAM's own spread; closest ($worst of the bound): $wl"
    else fail "$c (CUDA): mesh-motion fields above the bound at$over"; fi
done

# the solves
for c in mc_tight mc_restart; do
    k=$(cmpdir "$W/b_$c")
    python3 "$STEPS" solves "$W/b_$c/log" "$W/of_$c/log" > "$k/solves.cmp" 2>&1
    read -r nb no same <<< "$(sed -n 's/^solves //p' "$k/solves.cmp")"
    read -r iok ifirst <<< "$(sed -n 's/^iterations //p' "$k/solves.cmp")"
    read -r iw iws iwf imed <<< "$(sed -n 's/^initial //p' "$k/solves.cmp")"
    if [ -n "$nb" ] && [ "$nb" = "$no" ] && [ "$same" = 1 ] && [ "$nb" -gt 0 ] && grep -q "Solving for cellMotionUx" "$W/b_$c/log"
    then ok "$c (CUDA): the same $nb solves as OpenFOAM's, in the same order (cellMotionUx among them)"
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

# the discrete space-conservation law of the swept volumes
for c in $ALL; do
    g=$(sed -n 's/^diag gcl max .*|\/V \([^ ]*\) .*/\1/p' "$W/b_$c/log")
    if le "${g:-nan}" 1e-12
    then ok "b_$c: |V - V0 - deltaT*sum(meshPhi)|/V at most $g over every cell and step (the swept volumes close each cell's volume change to round-off)"
    else fail "b_$c: the swept volumes leave ${g:-?} of a cell's volume change unaccounted"; fi
done

# ---- conservation and stability ------------------------------------------------------------------------------------------
balance "b_mc_tight" raw; vbalance "b_mc_tight"; identity "b_mc_tight"

# ---- memory and runtime (reported) ------------------------------------------------------------------------------------------
for c in $ALL; do
    echo "     runtime $c: OpenFOAM $(grep -h '^ExecutionTime' "$W/of_$c/log" | tail -n 1 | awk '{print $3}') s; brae CUDA $(sed -n 's/^diag steps \([^ ]*\) step-seconds \([^ ]*\) per-step-us \([^ ]*\).*/\2 s (\1 steps, \3 us a step)/p' "$W/b_$c/log"), device memory $(sed -n 's/^diag device-memory \([^ ]*\) MiB/\1 MiB/p' "$W/b_$c/log")"
done

# ---- controls ---------------------------------------------------------------------------------------------------------
for k in motionFluxAbsolute motionDdtStatic motionWallStatic motionPiovStatic; do
    red "$k, mc_tight" "$W/c${k}_mc_tight" "$W/of_mc_tight" "$(spread mc_tight)"
done
opensummary
echo "rho_central_moving_vs_openfoam: $passes passed, $fails failed (work in $W)"
[ "$fails" = 0 ] && [ "$passes" -ge "${BRAE_RCF_K_MIN_PASSES:-1}" ]
