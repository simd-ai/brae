#!/usr/bin/env bash
# pcorr solved on the GPU ACROSS a coupled pair, on RAS/mixerVesselAMI (894,950 cells, a cyclicAMI pair), brae's
# default pressure path. DevicePcorrSolver declined any mesh with a coupled patch, so pcorr there was the host's,
# with the case's own solver. It now puts the pair's faces -- own cell, neighbour slots and weights, the matrix's
# interface coefficient -- on the matrix view, where the product applies them and the hierarchy carries them on
# every grid (amg_pcg/coupled_pair_in_hierarchy.sh).
# MEASURED 2026-10-04 on the tutorial, 30 steps: CorrectPhi 364.5 -> 68.1 ms a step, the step 2,093.3 -> 1,804.4.
# Here, every solve pinned at 1e-13: pcorr converges in 137 148 132 iterations where OpenFOAM's DICPCG stops at
# its 1,000-iteration limit twice, and the worst file is 1.2e-09 from OpenFOAM (p), as with pcorr on the host.
# The CONTROL stops pcorr's solve after one iteration: the fields leave the bound, so the gate sees pcorr.
. "$(dirname "$0")/../lib.sh"
unset BRAE_PRESSURE_CASE_SOLVER
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase mixerVesselAMI of > "$W/pp_stage.txt" 2>&1
o="$W/w_of_mixerVesselAMI"
[ -d "$o" ] || { say "mixerVesselAMI did not stage" FAIL; finish "pcorr across a coupled pair"; }
# arm <name> [env...]: brae on a copy of the staged case; never ends the gate
arm()
{
    local e="$W/pp_$1"
    shift
    rm -rf "${e:?}"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    ( cd "$e" && env BRAE_PRINT_TURB_SOLVES=1 "$@" "$BIN" -case . -device > log.brae 2>&1; echo $? > exit.txt )
}
its() { grep -a "Solving for pcorr" "$W/pp_$1/log.brae" | sed -E 's/.*No Iterations ([0-9]+).*/\1/' | tr '\n' ' '; }
arm gpu BRAE_X=1
arm one BRAE_CONTROL_AMG_PCG_PCORR_ONE_ITERATION=1
what="[device] pcorr runs the AMG-PCG across the cyclicAMI pair and converges: $(its gpu)iterations"
[ "$(cat "$W/pp_gpu/exit.txt")" = 0 ] && [ -n "$(its gpu)" ] \
    && grep -aq "pcorr: system/fvSolution asks for .*; brae runs its AMG-preconditioned PCG" "$W/pp_gpu/log.brae" \
    && ! grep -aq "pcorr keeps the case's own solver" "$W/pp_gpu/log.brae" \
    && [ "$(its gpu | tr ' ' '\n' | sort -n | tail -1)" -lt 1000 ] && say "$what" ok || say "$what" FAIL
python3 "$CMP" "$o" "$W/pp_gpu" $(timedirs "$o") > "$W/cmp_pp_gpu.txt" 2>&1
judge "mixerVesselAMI with pcorr on the GPU" "$W/cmp_pp_gpu.txt" 1.2e-08 "$o/log.interFoam" \
    && say "[device] ...and every file within 1.2e-08 of OpenFOAM" ok \
    || say "[device] ...and every file within 1.2e-08 of OpenFOAM" FAIL
python3 "$CMP" "$o" "$W/pp_one" $(timedirs "$o") > "$W/cmp_pp_one.txt" 2>&1
python3 - "$W/cmp_pp_one.txt" 1.2e-08 > "$W/pp_one.txt" <<'PY'
import json, sys
r = json.loads([l for l in open(sys.argv[1]) if l.startswith('RESULT ')][-1][7:])
f = max(((k, v['rel']) for k, v in r['files'].items() if 'cumulativeContErr' not in k), key=lambda kv: kv[1])
print('%s %.1e' % f)
sys.exit(0 if f[1] > float(sys.argv[2]) else 1)
PY
rc=$?
what="CONTROL  pcorr stopped after one iteration puts the fields over 1.2e-08 ($(cat "$W/pp_one.txt"))"
[ $rc -eq 0 ] && say "$what" ok || say "$what" FAIL
finish "pcorr is solved on the GPU across a coupled pair"
