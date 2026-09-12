#!/usr/bin/env bash
# Solver selection ctest: `brae` routes a case to the brae solver its controlDict `application` asks for, and
# refuses the cases it has no solver for instead of running them with the wrong one (solver_dispatch.cuh).
# Only the SELECTION is under test here -- it happens before any mesh read or CUDA init, so these minimal
# fake cases (system/ dicts only, no polyMesh) never reach the solver and the test needs no GPU. The happy
# path -- the transient case actually solving through `brae` -- is the pimple_dispatch ctest.
set -u
BIN="${1:?brae binary}"
WORK="${2:?work dir}"

fail=0
mkcase()   # mkcase <dir> <application-line> <ddt-scheme>
{
    mkdir -p "$1/system"
    printf 'FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\n%s\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\nendTime 1;\ndeltaT 1;\n' "$2" > "$1/system/controlDict"
    printf 'FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }\nddtSchemes { default %s; }\ngradSchemes { default Gauss linear; }\ndivSchemes { default none; div(phi,U) bounded Gauss upwind; }\nlaplacianSchemes { default Gauss linear corrected; }\n' "$3" > "$1/system/fvSchemes"
}
check()    # check <name> <case dir> <expected text in output>
{
    local log="$WORK/$1.log"
    "$BIN" -case "$2" > "$log" 2>&1
    if grep -qF -e "$3" "$log"; then
        echo "ok:   $1"
    else
        echo "FAIL: $1 -- output does not contain '$3'"; sed -n '1,15p' "$log"; fail=1
    fi
}

rm -rf "$WORK"; mkdir -p "$WORK"

# 1. A solver brae does not implement: stop, name it, and list what brae does have. Never silently solve it
#    with simpleFoam -- a wrong solver is a wrong answer that looks right.
mkcase "$WORK/unsupported" "application     interFoam;" "Euler"
check unsupported_application "$WORK/unsupported" "'interFoam' is not a solver brae implements yet"
grep -qF "pimpleFoam" "$WORK/unsupported_application.log" || { echo "FAIL: error does not list brae's solvers"; fail=1; }

# 2. STEADY solver + transient ddtSchemes: NOT an error. OpenFOAM's steady solvers construct no ddt term
#    at all (simpleFoam's UEqn is div(phi,U) + MRF.DDt(U) + divDevSigma(U) == fvOptions(U)), so the entry
#    is inert there. OF's OWN squareBend tutorial ships `application simpleFoam` with
#    `ddtSchemes { default Euler; }` and runs; brae used to REFUSE it, which was a false positive against
#    a shipped case. It must now report the entry as ignored and run steady.
mkcase "$WORK/mismatch" "application     simpleFoam;" "Euler"
check application_ddt_ignored "$WORK/mismatch" "'Euler' on the steady solver"

# 2b. The REVERSE is still an error: a transient solver with steadyState would drop the time derivative
#     entirely -- a different equation, not an unused entry.
mkcase "$WORK/mismatch2" "application     pimpleFoam;" "steadyState"
check transient_with_steadyState "$WORK/mismatch2" "would drop the time derivative"

# 3. Hand-written case with no `application`: fall back to the ddt scheme. Euler -> the transient solver, which
#    is reached (the notice) and then fails on the absent mesh -- selection is what matters here.
mkcase "$WORK/noapp" "" "Euler"
check no_application_transient "$WORK/noapp" "-> pimpleFoam"

# 4. Same, steadyState: stays in this executable (no hand-over notice), and gets as far as the missing case
#    dictionary that the steady driver reads next.
mkcase "$WORK/noapp_steady" "" "steadyState"
"$BIN" -case "$WORK/noapp_steady" > "$WORK/no_application_steady.log" 2>&1
if grep -qF -e "-> pimpleFoam" "$WORK/no_application_steady.log"; then
    echo "FAIL: no_application_steady -- steady case was handed to the transient solver"; fail=1
else
    echo "ok:   no_application_steady"
fi

# 5. application pimpleFoam is handed over even though this binary is `brae`.
mkcase "$WORK/pimple" "application     pimpleFoam;" "backward"
check application_pimplefoam "$WORK/pimple" "controlDict application pimpleFoam -> pimpleFoam"

# 6. application simpleFoam: the registry names `brae` for it, which IS the running binary, so it must solve in
#    this process -- no hand-over, and above all no exec loop (the guard would report one).
mkcase "$WORK/simple" "application     simpleFoam;" "steadyState"
"$BIN" -case "$WORK/simple" > "$WORK/application_simplefoam.log" 2>&1
if grep -qE -e "-> simpleFoam|dispatch loop" "$WORK/application_simplefoam.log"; then
    echo "FAIL: application_simplefoam -- brae handed the steady case to another process"
    sed -n '1,10p' "$WORK/application_simplefoam.log"; fail=1
else
    echo "ok:   application_simplefoam"
fi

# 7. application rhoSimpleFoam is handed over to brae_rhoSimpleFoam. The registry has carried this row
#    since the compressible solver was added and no test ever exercised it -- and the target was missing
#    from `add_dependencies(brae ...)`, so on a fresh build the hand-over exec'd a binary that did not
#    exist. This arm fails on both: the routing notice, and the sibling actually being there to run.
mkcase "$WORK/rho" "application     rhoSimpleFoam;" "steadyState"
check application_rhosimplefoam "$WORK/rho" "controlDict application rhoSimpleFoam -> rhoSimpleFoam"
if [ -x "$(dirname "$BIN")/brae_rhoSimpleFoam" ]; then
    echo "ok:   rhosimplefoam_sibling_built"
else
    echo "FAIL: rhosimplefoam_sibling_built -- the registry routes to a binary that was not built"; fail=1
fi

[ "$fail" -eq 0 ] && echo "PASS: solver selection routes and refuses correctly"
exit "$fail"
