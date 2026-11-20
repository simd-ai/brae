#!/usr/bin/env bash
# THE DEFAULT-PATH RUNNER (tests/cmake/default_path.cmake): a gate run in the environment a customer's `brae -case`
# runs in, with none of brae's BRAE_* variables.
#
#   default_path_run.sh [NAME=VALUE ...] <gate> [args ...]
#
#   1. The suite's three case-solver variables must be ABSENT. tests/CMakeLists.txt appends BRAE_PRESSURE_CASE_SOLVER,
#      BRAE_TURBULENCE_CASE_SOLVER and BRAE_SIMPLE_CASE_SOLVER to every test and unsets them again (ENVIRONMENT_
#      MODIFICATION) for a test labelled `default-path`. One of them here means that arrangement did not apply, and
#      the test would run the case-solver path under a default-path name: it stops, saying so.
#   2. Every other BRAE_* variable it was started with (a developer's shell, a runner's) is removed, but the test
#      harness's own, which say where things are and which no brae source reads (`grep -rnw <name> src/` finds
#      none): BRAE_OF_TUTORIALS (where OpenFOAM's tutorials are, set on a machine whose install is elsewhere),
#      BRAE_OF_CACHE (tests/of_oracle_cache.sh), BRAE_GATE_OUT and BRAE_GATE_JOBS, and tools/ctest_cached.py's
#      BRAE_TEST_CACHE, BRAE_TEST_CACHE_SALT and BRAE_CTEST.
#   3. NAME=VALUE arguments are then set for the gate: the locator a gate reads to find its binary (BRAE_RCF_BIN, so
#      that a gate defaulting to a solver executable runs the `brae` front door instead). No brae source reads it
#      (`grep -rnw BRAE_RCF_BIN src/` finds nothing), so what brae itself does is unchanged by it.
# A gate's own NAME=value controls (BRAE_CONTROL_..., a host arm's selector) are set by the gate for its control runs
# and are not touched: only the arms the gate runs without one are the default path.
set -u
for v in BRAE_PRESSURE_CASE_SOLVER BRAE_TURBULENCE_CASE_SOLVER BRAE_SIMPLE_CASE_SOLVER
do
    if [ -n "${!v+set}" ]
    then
        echo "FAIL: $v is set (${!v}): a default-path test runs without it -- register the test with the default-path"
        echo "      label (tests/cmake/default_path.cmake), whose ENVIRONMENT_MODIFICATION unsets it (tests/CMakeLists.txt)"
        exit 1
    fi
done
KEEP=" BRAE_OF_TUTORIALS BRAE_OF_CACHE BRAE_GATE_OUT BRAE_GATE_JOBS BRAE_TEST_CACHE BRAE_TEST_CACHE_SALT BRAE_CTEST "
for v in $(compgen -e | grep '^BRAE_')
do
    [[ $KEEP == *" $v "* ]] || unset "$v"
done
while [ $# -gt 0 ] && [[ $1 =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]
do
    export "${1?}"
    shift
done
[ $# -gt 0 ] || { echo "usage: default_path_run.sh [NAME=VALUE ...] <gate> [args ...]"; exit 2; }
kept=$(compgen -e | grep '^BRAE_' | tr '\n' ' ')
gate=$(basename "$1")
case $gate in bash|sh|python3) gate=$(basename "${2:-$1}") ;; esac
echo "default_path_run: $gate with no BRAE_* variable brae reads${kept:+ (set, read by no brae source: $kept)}"
exec "$@"
