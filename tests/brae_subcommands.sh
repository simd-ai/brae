#!/usr/bin/env bash
# `brae benchmark` and `brae node`: the two reserved words, and the guarantee that they are the ONLY two.
#
# The risk this test exists for is regression by generalisation -- someone later treats any leading word as a
# subcommand, and `brae myCase` silently stops working for every existing user. Half of these cases are that
# negative: ordinary case directories must keep winning.
#
# Builds a LOCAL git repository as the benchmark template repo (BRAE_BENCH_REPO), so the pull path is exercised
# for real without touching the network or github.com/simd-ai/brae-bench.
set -u
BIN="${1:?brae binary}"
WORK="${2:?work dir}"

fail=0
say_ok()   { echo "ok:   $1"; }
say_fail() { echo "FAIL: $1 -- $2"; fail=1; }

rm -rf "$WORK"; mkdir -p "$WORK"
export BRAE_BENCH_CACHE="$WORK/cache"
export HOME="$WORK/home"; mkdir -p "$HOME"          # keep the real ~/.cache out of it
git config --global --get user.email > /dev/null 2>&1 || export GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@t GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@t

# ---- a local template repo with two sample branches -------------------------------------------------------
TEMPLATE="$WORK/brae-bench"
mkcase()   # mkcase <dir> <application> <ddt>
{
    mkdir -p "$1/system" "$1/constant"
    printf 'FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\napplication %s;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\nendTime 1;\ndeltaT 0.1;\n' "$2" > "$1/system/controlDict"
    printf 'FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }\nddtSchemes { default %s; }\ngradSchemes { default Gauss linear; }\ndivSchemes { default none; div(phi,U) bounded Gauss upwind; }\nlaplacianSchemes { default Gauss linear corrected; }\n' "$3" > "$1/system/fvSchemes"
    printf 'FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\nsolvers { p { solver PCG; tolerance 1e-7; } }\nSIMPLE { nNonOrthogonalCorrectors 0; }\n' > "$1/system/fvSolution"
    printf 'FoamFile { version 2.0; format ascii; class dictionary; object transportProperties; }\ntransportModel Newtonian;\nnu 1e-05;\n' > "$1/constant/transportProperties"
    printf 'FoamFile { version 2.0; format ascii; class dictionary; object turbulenceProperties; }\nsimulationType laminar;\n' > "$1/constant/turbulenceProperties"
}

git init -q "$TEMPLATE"
mkcase "$TEMPLATE" simpleFoam steadyState
printf '{\n  "sample": "tiny-steady",\n  "description": "fixture",\n  "cells": 1234,\n  "min_vram_mb": 512,\n  "solver": "simpleFoam"\n}\n' > "$TEMPLATE/brae-bench.json"
git -C "$TEMPLATE" add -A && git -C "$TEMPLATE" commit -qm "tiny-steady"
git -C "$TEMPLATE" branch -M tiny-steady
# a second branch that carries a coded BC -- the case a benchmark must refuse to run
git -C "$TEMPLATE" checkout -q -b evil-coded
mkdir -p "$TEMPLATE/0"
printf 'FoamFile { version 2.0; format ascii; class volVectorField; object U; }\nboundaryField\n{\n    inlet\n    {\n        type            codedFixedValue;\n        value           uniform (0 0 0);\n        name            attack;\n        code            #{ /* device code brae would compile and run */ #};\n    }\n}\n' > "$TEMPLATE/0/U"
git -C "$TEMPLATE" add -A && git -C "$TEMPLATE" commit -qm "coded"
git -C "$TEMPLATE" checkout -q tiny-steady
export BRAE_BENCH_REPO="$TEMPLATE"

# ---- reserved words are reserved -------------------------------------------------------------------------
"$BIN" benchmark --list > "$WORK/list.log" 2>&1
if grep -q "tiny-steady" "$WORK/list.log" && grep -q "evil-coded" "$WORK/list.log"; then
    say_ok "benchmark_list"
else
    say_fail "benchmark_list" "did not list the fixture branches"; sed -n '1,8p' "$WORK/list.log"
fi

# Nothing brae shells out to may write to brae's own stdout -- `git --version` leaked there once, and anything
# parsing the output would have picked it up.
if grep -qi "git version" "$WORK/list.log"; then
    say_fail "no_subprocess_noise_on_stdout" "a helper command's output leaked into brae's"
else
    say_ok "no_subprocess_noise_on_stdout"
fi

"$BIN" benchmark --help > "$WORK/bhelp.log" 2>&1 \
    && grep -q "brae benchmark \[sample\]" "$WORK/bhelp.log" \
    && say_ok "benchmark_help" || say_fail "benchmark_help" "no usage / non-zero exit"

# `brae node <verb>` must reach brae-agent. When the agent is built it answers for real; when it is not (no
# libcurl), brae must still say clearly which binary is missing rather than failing obscurely.
BRAE_IDENTITY_PATH="$WORK/no-such-identity.json" "$BIN" node status > "$WORK/node.log" 2>&1
if grep -qE "Not registered|brae-agent" "$WORK/node.log"; then
    say_ok "node_routes_to_agent"
else
    say_fail "node_routes_to_agent" "did not reach brae-agent"; sed -n '1,8p' "$WORK/node.log"
fi

# The hand-over must forward arguments, not swallow them.
BRAE_IDENTITY_PATH="$WORK/no-such-identity.json" "$BIN" node --help > "$WORK/nodehelp.log" 2>&1
if grep -qE "brae node register|brae-agent" "$WORK/nodehelp.log"; then
    say_ok "node_forwards_arguments"
else
    say_fail "node_forwards_arguments" "--help did not reach the agent"; sed -n '1,6p' "$WORK/nodehelp.log"
fi

# ---- and nothing else is ---------------------------------------------------------------------------------
# A case directory whose name could be mistaken for a verb must still be treated as a case.
for word in run check solvers capabilities register status list bench nodes benchmarks; do
    mkcase "$WORK/$word" simpleFoam steadyState
    ( cd "$WORK" && "$BIN" "$word" > "$WORK/word_$word.log" 2>&1 )
    if grep -qE "brae-agent|benchmark sample|fetching sample" "$WORK/word_$word.log"; then
        say_fail "case_named_$word" "was taken as a subcommand instead of a case directory"
        sed -n '1,5p' "$WORK/word_$word.log"
    else
        say_ok "case_named_$word"
    fi
done

# The documented escape hatch for a case that really is called 'node'.
mkcase "$WORK/node_case" simpleFoam steadyState
"$BIN" -case "$WORK/node_case" > "$WORK/node_case.log" 2>&1
if grep -q "brae-agent" "$WORK/node_case.log"; then
    say_fail "case_option_beats_subcommand" "-case was overridden by the subcommand word"
else
    say_ok "case_option_beats_subcommand"
fi

"$BIN" --help > "$WORK/help.log" 2>&1
grep -q "brae benchmark" "$WORK/help.log" && grep -q "brae node" "$WORK/help.log" \
    && grep -q "brae -case node" "$WORK/help.log" \
    && say_ok "help_documents_subcommands_and_caveat" \
    || say_fail "help_documents_subcommands_and_caveat" "--help omits a subcommand or the reserved-word caveat"

[ "$fail" -eq 0 ] && echo "PASS: the reserved words route, and a case directory of any name stays a case"
exit "$fail"
