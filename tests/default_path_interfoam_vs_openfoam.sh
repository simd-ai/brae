#!/usr/bin/env bash
# interFoam AS A CUSTOMER RUNS IT: `brae -case <dir>` (the front door hands the case to brae_interFoam with -device,
# solvers/common/solver_dispatch.cuh) with no BRAE_* variable, so p_rgh and pcorr run brae's AMG-preconditioned PCG
# in place of the case's own solver -- the pressure rule (device_inter_pressure_step.cu), which every other test
# turns off (BRAE_PRESSURE_CASE_SOLVER=1, tests/CMakeLists.txt).
#
# THE CASE AND THE BOUND are the pressure rule's own gate's, reused, not set here: laminar/capillaryRise as
# tests/interfoam_write/lib.sh stages its W row (meshed as its Allrun does, two steps, every solve pinned) against
# OpenFOAM v2412 at the same staging, every written file within 3.8e-12 -- tests/interfoam_write/amg_pcg/
# capillary_rise.sh:8 (measured worst 3.8e-13, phi at the first step, 2026-10-03). That gate runs brae_interFoam
# -device directly with the turbulence rule's variable still set; this one goes through `brae` with neither, which
# changes nothing on a laminar case but the way in. capillaryRise is laminar: the turbulence rule's default
# (colour-ordered Gauss-Seidel on k, epsilon, omega) is held by tests/interfoam_write/turb_colour/, still with the
# pressure rule's variable set, and is not covered here (no bound for the two rules together has been measured).
#
#   ARM      every file brae writes within 3.8e-12 of OpenFOAM's (tools/foam_time_compare.py, lib.sh's judge)
#   ARM      the log says what ran in the case's place: "brae runs its AMG-preconditioned PCG"
#
# Run through tests/default_path_run.sh (no BRAE_* variable); lib.sh exports the two rule variables for its own
# gates, and they are unset again below before brae runs.
BRAE="${1:?brae binary (the front door)}"
shift
. "$(dirname "$0")/interfoam_write/lib.sh"
unset BRAE_PRESSURE_CASE_SOLVER BRAE_TURBULENCE_CASE_SOLVER
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
BRAE="$(cd "$(dirname "$BRAE")" && pwd)/$(basename "$BRAE")"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU: the front door runs interFoam on the device"; exit 77; }
BOUND=3.8e-12   # tests/interfoam_write/amg_pcg/capillary_rise.sh:8

wcase capillaryRise of > "$W/stage.txt" 2>&1
o="$W/w_of_capillaryRise"
[ -d "$o" ] || { cat "$W/stage.txt"; say "capillaryRise did not stage" FAIL; finish "staging"; }
e="$W/default"
mkdir -p "$e"
cp -r "$o/0" "$o/constant" "$o/system" "$e/"
leaked=$(env | grep -E '^BRAE_(PRESSURE|TURBULENCE|SIMPLE)_CASE_SOLVER=' | tr '\n' ' ')
[ -z "$leaked" ] || { say "no case-solver variable reaches brae ($leaked)" FAIL; finish "environment"; }
( cd "$e" && "$BRAE" -case . > log.brae 2>&1 ) < /dev/null \
    || { tail -5 "$e/log.brae"; say "brae -case ran capillaryRise" FAIL; finish "the run"; }
python3 "$CMP" "$o" "$e" $(timedirs "$o") > "$W/cmp.txt" 2>&1
judge "capillaryRise, brae -case" "$W/cmp.txt" "$BOUND" "$o/log.interFoam" \
    && say "capillaryRise through brae -case: every file within $BOUND of OpenFOAM" ok \
    || say "capillaryRise through brae -case: every file within $BOUND of OpenFOAM" FAIL
grep -q "brae runs its AMG-preconditioned PCG" "$e/log.brae" \
    && say "the log announces the AMG-PCG in the case's solver's place" ok \
    || say "the log announces the AMG-PCG in the case's solver's place" FAIL
finish "interFoam's default path, through brae -case, is within the pressure rule's bound of OpenFOAM"
