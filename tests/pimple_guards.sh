#!/usr/bin/env bash
# The transient solver's refusals: physics the STEADY driver applies and this one does not yet must stop the run,
# not be silently skipped (gpuPimpleFoam.cu, "Physics the STEADY driver applies"). A dropped MRF rotation or
# fvOptions source produces a plausible-looking field that is simply wrong, which is the failure mode brae exists
# to avoid -- so each refusal gets a test, and the control case proves the guards do not fire on a clean case.
#
# The guards run after the dictionary parse and BEFORE the mesh is read, so these fixtures are dicts only: no
# polyMesh, no GPU. The control case is expected to fail too -- but on the missing mesh, further in.
set -u
BIN="${1:?brae_pimpleFoam binary}"
WORK="${2:?work dir}"

fail=0
mkcase()   # mkcase <dir> [extra controlDict lines]
{
    mkdir -p "$1/system" "$1/constant"
    printf 'FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\napplication pimpleFoam;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\nendTime 1;\ndeltaT 0.1;\n%s\n' "${2:-}" > "$1/system/controlDict"
    printf 'FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }\nddtSchemes { default Euler; }\ngradSchemes { default Gauss linear; }\ndivSchemes { default none; div(phi,U) bounded Gauss upwind; }\nlaplacianSchemes { default Gauss linear corrected; }\n' > "$1/system/fvSchemes"
    printf 'FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\nsolvers { p { solver PCG; tolerance 1e-7; } U { solver PBiCGStab; tolerance 1e-8; } }\nPIMPLE { nOuterCorrectors 2; nCorrectors 2; nNonOrthogonalCorrectors 0; }\n' > "$1/system/fvSolution"
    printf 'FoamFile { version 2.0; format ascii; class dictionary; object transportProperties; }\ntransportModel Newtonian;\nnu 1e-05;\n' > "$1/constant/transportProperties"
    printf 'FoamFile { version 2.0; format ascii; class dictionary; object turbulenceProperties; }\nsimulationType laminar;\n' > "$1/constant/turbulenceProperties"
}
# The mirror of `refuses`: the entry must no longer STOP the run at dictionary-read time.
#
# These fixtures carry no mesh -- they exist to check what the dictionary stage accepts or rejects, and
# they always die later on the absent polyMesh. So the assertion is "got PAST the dict stage", evidenced
# by reaching the mesh error, not "produced a Courant number" (which needs a mesh and a flux). Asserting
# the latter here would be asserting something the fixture cannot show.
accepts_past_dict() {   # accepts_past_dict <name> <case dir> [env assignments...]
    local name=$1 dir=$2
    shift 2
    env "$@" "$BIN" -case "$dir" > "$WORK/$name.log" 2>&1
    if grep -qF "not supported yet" "$WORK/$name.log"; then
        echo "FAIL: $name -- still refused at the dictionary stage, but the feature is implemented"; fail=1; return
    fi
    if ! grep -qF "constant/polyMesh/points" "$WORK/$name.log"; then
        echo "FAIL: $name -- did not reach the mesh stage; it stopped on something else"
        tail -2 "$WORK/$name.log"; fail=1; return
    fi
    echo "ok:   $name"
}

refuses()  # refuses <name> <case dir> <expected text>
{
    local log="$WORK/$1.log"
    "$BIN" -case "$2" > "$log" 2>&1
    local rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "FAIL: $1 -- the solver accepted a case it cannot solve correctly"; fail=1; return
    fi
    if grep -qF -e "$3" "$log"; then
        echo "ok:   $1"
    else
        echo "FAIL: $1 -- refused, but not for the stated reason (wanted '$3')"; sed -n '1,8p' "$log"; fail=1
    fi
}

rm -rf "$WORK"; mkdir -p "$WORK"

# 1. adjustTimeStep's Courant control (OF readTimeControls/CourantNo/setInitialDeltaT/setDeltaT, via the shared
#    cfdTools module) on brae's own clock. Refused by default until M4 (issue #53; tests/pimple_setup_refusals.sh
#    holds the refusal), it is reached here through its experimental override: the control loop must still run past
#    the dictionaries -- a case that silently stopped at them would mean the override no longer opens it.
mkcase "$WORK/adjustdt" "adjustTimeStep yes;
maxCo 0.9;"
accepts_past_dict adjust_time_step "$WORK/adjustdt" BRAE_ALLOW_ADJUSTTIMESTEP=1

# 2. An active MRF zone: dropping the rotation leaves a converged-looking field of the wrong flow.
mkcase "$WORK/mrf"
printf 'FoamFile { version 2.0; format ascii; class dictionary; object MRFProperties; }\nMRF1\n{\n    cellZone rotor;\n    active yes;\n    origin (0 0 0);\n    axis (0 0 1);\n    omega 104.72;\n}\n' > "$WORK/mrf/constant/MRFProperties"
refuses active_mrf_zone "$WORK/mrf" "does not apply MRF yet"

# 3. fvOptions. The blanket "transient does not apply fvOptions" refusal is GONE -- the momentum sources
#    live in solveMomentumPredictor, which pimpleStep already calls, so the driver reads them and hands
#    them over (planarPoiseuille agrees with OpenFOAM to 7.9e-08 on a flow that IS the source). What must
#    still refuse is a source brae cannot apply, and a TIME WINDOW it cannot honour -- brae bakes every
#    source into one set of buffers, so it cannot switch one on mid-run.
mkcase "$WORK/fvopt_unsupported"
printf 'FoamFile { version 2.0; format ascii; class dictionary; object fvOptions; }\nheat { type semiImplicitSourceNotAThing; }\n' > "$WORK/fvopt_unsupported/system/fvOptions"
refuses fvoptions_unsupported_source "$WORK/fvopt_unsupported" "SILENTLY dropped"

# A vectorSemiImplicitSource with NEITHER `sources` nor `injectionRateSuSp`: nothing to apply, so loud.
mkcase "$WORK/fvopt_nokeys"
printf 'FoamFile { version 2.0; format ascii; class dictionary; object fvOptions; }\nmomentumSource { type vectorSemiImplicitSource; selectionMode all; }\n' > "$WORK/fvopt_nokeys/constant/fvOptions"
refuses fvoptions_no_source_keys "$WORK/fvopt_nokeys" "neither a \`sources\`"

# A window that does not cover the run must refuse rather than apply the source throughout.
mkcase "$WORK/fvopt_window"
printf 'FoamFile { version 2.0; format ascii; class dictionary; object fvOptions; }\nmomentumSource { type vectorSemiImplicitSource; selectionMode all; volumeMode specific; timeStart 5000; duration 1; sources { U ((5 0 0) 0); } }\n' > "$WORK/fvopt_window/constant/fvOptions"
refuses fvoptions_time_window "$WORK/fvopt_window" "does not cover this run"

# ...and a SUPPORTED, always-on source must NOT be refused -- the mirror-image bug of the old blanket stop.
mkcase "$WORK/fvopt_ok"
printf 'FoamFile { version 2.0; format ascii; class dictionary; object fvOptions; }\nmomentumSource { type vectorSemiImplicitSource; selectionMode all; volumeMode specific; sources { U ((5 0 0) 0); } }\n' > "$WORK/fvopt_ok/constant/fvOptions"
"$BIN" -case "$WORK/fvopt_ok" > "$WORK/fvopt_ok.log" 2>&1
if grep -qE -e "fvOptions|SILENTLY dropped" "$WORK/fvopt_ok.log"; then
    echo "FAIL: fvoptions_supported_runs -- a supported always-on source was refused"
    grep -m2 -E "fvOptions|SILENTLY" "$WORK/fvopt_ok.log"; fail=1
else
    echo "ok:   fvoptions_supported_runs"
fi

# 4. An INACTIVE MRF zone is not a refusal -- OpenFOAM cases routinely ship one switched off, and refusing those
#    would be the mirror-image bug (stopping a case brae can solve perfectly well).
mkcase "$WORK/mrf_off"
printf 'FoamFile { version 2.0; format ascii; class dictionary; object MRFProperties; }\nMRF1\n{\n    cellZone rotor;\n    active no;\n    omega 104.72;\n}\n' > "$WORK/mrf_off/constant/MRFProperties"
"$BIN" -case "$WORK/mrf_off" > "$WORK/inactive_mrf.log" 2>&1
if grep -qF -e "does not apply MRF yet" "$WORK/inactive_mrf.log"; then
    echo "FAIL: inactive_mrf -- a switched-off MRF zone was treated as active"; fail=1
else
    echo "ok:   inactive_mrf"
fi

# 5. Control: a clean case must get PAST every guard. It still fails -- on the mesh these fixtures do not have --
#    which is exactly how we know the guards let it through rather than the run being stopped early.
mkcase "$WORK/clean"
"$BIN" -case "$WORK/clean" > "$WORK/clean.log" 2>&1
if grep -qE -e "does not apply MRF yet|SILENTLY dropped|adjustTimeStep yes is not supported" "$WORK/clean.log"; then
    echo "FAIL: clean_case -- a guard fired on a case with none of the unsupported features"
    sed -n '1,8p' "$WORK/clean.log"; fail=1
else
    echo "ok:   clean_case"
fi

# 6. --help is part of the CLI surface: it must work without a case and exit 0.
if "$BIN" --help > "$WORK/help.log" 2>&1 && grep -qF "brae_pimpleFoam" "$WORK/help.log"; then
    echo "ok:   help"
else
    echo "FAIL: help -- --help did not print usage and exit 0"; sed -n '1,5p' "$WORK/help.log"; fail=1
fi

[ "$fail" -eq 0 ] && echo "PASS: the transient solver refuses what it cannot solve, and only that"
exit "$fail"
