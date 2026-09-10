#!/usr/bin/env bash
# `brae benchmark <sample>` end to end: pull a template branch, refuse the ones carrying code, solve the case,
# write the result. The template repository is a LOCAL git repo built here (BRAE_BENCH_REPO), so nothing touches
# the network -- but the fetch, cache, manifest, guard and run paths are the real ones.
#
# The sample is the committed pitzDaily, cut to a handful of iterations, so the happy path is a genuine solve
# rather than a mock.
#
# The test that matters most is `refuses_coded_case`: brae compiles codedFixedValue bodies with NVRTC and runs
# them on the device, so a benchmark sample that may ship code is a remote-code-execution path onto every
# contributor's machine. It must be refused BEFORE the case is read.
set -u
BIN="${1:?brae binary}"
SRC="${2:?committed pitzDaily case dir}"
WORK="${3:?work dir}"

if [ ! -f "$SRC/constant/polyMesh/points" ] || [ ! -f "$SRC/0/U" ]; then
    echo "SKIP: fixture '$SRC' not present"; exit 125
fi

fail=0
say_ok()   { echo "ok:   $1"; }
say_fail() { echo "FAIL: $1 -- $2"; fail=1; }

rm -rf "$WORK"; mkdir -p "$WORK"
export BRAE_BENCH_CACHE="$WORK/cache"
export HOME="$WORK/home"; mkdir -p "$HOME"
export GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@t GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@t

# ---- the template repo: one runnable sample, one that carries code ----------------------------------------
TEMPLATE="$WORK/brae-bench"
git init -q "$TEMPLATE"
cp -r "$SRC/constant" "$SRC/system" "$SRC/0" "$TEMPLATE/"
sed -i 's/^endTime .*/endTime 5;/; s/^writeInterval .*/writeInterval 5;/' "$TEMPLATE/system/controlDict" 2>/dev/null || true
printf '{\n  "sample": "pitz-tiny",\n  "description": "pitzDaily, 5 iterations",\n  "cells": 12225,\n  "min_vram_mb": 512,\n  "solver": "simpleFoam"\n}\n' > "$TEMPLATE/brae-bench.json"
git -C "$TEMPLATE" add -A && git -C "$TEMPLATE" commit -qm "pitz-tiny"
# Grouped by solver, as the real repo is: the name is a branch AND a cache path, so the slash has to work in both.
git -C "$TEMPLATE" branch -M pimplefoam/pitz-tiny

git -C "$TEMPLATE" checkout -q -b pimplefoam/coded-attack
printf 'FoamFile { version 2.0; format ascii; class dictionary; object U; }\nboundaryField { inlet { type codedFixedValue; code #{ #}; } }\n' > "$TEMPLATE/0/U.coded"
git -C "$TEMPLATE" add -A && git -C "$TEMPLATE" commit -qm "coded"

# The Function1 form, branched from the CLEAN sample so it trips nothing else. It carries no distinctive
# word of its own -- `type coded;` inside a massFlowRate is the whole marker -- and brae compiles that body
# with the HOST compiler, so a benchmark carrying one is arbitrary code on a contributor's machine just as
# a codedFixedValue is. The spacing is deliberately not the writer's, since the guard matches a pattern.
git -C "$TEMPLATE" checkout -q pimplefoam/pitz-tiny
git -C "$TEMPLATE" checkout -q -b pimplefoam/coded-function1-attack
printf 'FoamFile { version 2.0; format ascii; class dictionary; object U; }\nboundaryField { inlet { type flowRateInletVelocity; massFlowRate { type   coded ; name r; code #{ return 5; #}; } } }\n' > "$TEMPLATE/0/U.f1"
git -C "$TEMPLATE" add -A && git -C "$TEMPLATE" commit -qm "coded Function1"
git -C "$TEMPLATE" checkout -q pimplefoam/pitz-tiny

# branched from the CLEAN sample, so the only thing wrong with it is the missing manifest -- otherwise it would
# trip the code guard first and this would not be testing what it claims to test.
git -C "$TEMPLATE" checkout -q pimplefoam/pitz-tiny
git -C "$TEMPLATE" checkout -q -b pimplefoam/no-manifest
git -C "$TEMPLATE" rm -q brae-bench.json && git -C "$TEMPLATE" commit -qm "no manifest"
git -C "$TEMPLATE" checkout -q pimplefoam/pitz-tiny
export BRAE_BENCH_REPO="$TEMPLATE"

cd "$WORK"

# ---- 1. the happy path: pull, solve, write the result with NO flags asked for -----------------------------
"$BIN" benchmark pimplefoam/pitz-tiny > "$WORK/run.log" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then
    say_fail "benchmark_runs" "exited $rc"; sed -n '1,20p' "$WORK/run.log"
elif [ ! -f "$WORK/brae-benchmark.json" ]; then
    say_fail "benchmark_runs" "no brae-benchmark.json was written"
else
    say_ok "benchmark_runs"
fi

# The machine-readable result is produced by default -- there is no --json to remember.
if python3 -c "
import json,sys
r = json.load(open('$WORK/brae-benchmark.json'))
assert r['sample'] == 'pitz-tiny', r
assert r['success'] is True, r
assert r['cells'] == 12225, r
assert r['runtime_s'] > 0, r
assert len(r['sample_commit']) == 40, r
" 2>"$WORK/json.err"; then
    say_ok "result_json_is_valid_and_complete"
else
    say_fail "result_json_is_valid_and_complete" "$(head -3 "$WORK/json.err")"
fi

grep -q "fetching sample" "$WORK/run.log" && say_ok "first_run_fetches" \
    || say_fail "first_run_fetches" "did not report fetching the sample"

# The result carries evidence the run computed something, not just that it finished. Deliberately scalar
# invariants rather than a hash: two GPUs sum in different orders, so a digest would differ between machines
# that are both correct.
if python3 -c "
import json
r = json.load(open('$WORK/brae-benchmark.json'))
inv = r['invariants']
assert inv['cells'] == 12225, inv
assert inv['u_max'] > inv['u_mean'] > 0, inv
assert inv['p_min'] < inv['p_max'], inv
" 2>"$WORK/inv.err"; then
    say_ok "result_carries_invariants"
else
    say_fail "result_carries_invariants" "$(head -3 "$WORK/inv.err")"
fi

cp "$WORK/brae-benchmark.json" "$WORK/first.json"

# ---- 2. the cache: a second run must not re-clone ----------------------------------------------------------
"$BIN" benchmark pimplefoam/pitz-tiny > "$WORK/run2.log" 2>&1
if grep -q "fetching sample" "$WORK/run2.log"; then
    say_fail "second_run_uses_cache" "re-fetched an already-cached sample"
else
    say_ok "second_run_uses_cache"
fi

# Same sample, same machine, same numbers -- to a tolerance, not exactly.
#
# The GPU reduces in whatever order the scheduler produces, so two identical runs differ in the last few digits:
# measured here at ~4e-9 relative on u_max. That is why the result carries invariants rather than a hash, and
# why whoever compares them across nodes must use a tolerance looser still. 1e-6 sits comfortably above the
# observed floor while remaining far tighter than any real error.
if python3 -c "
import json
a = json.load(open('$WORK/first.json'))['invariants']
b = json.load(open('$WORK/brae-benchmark.json'))['invariants']
for k in a:
    lhs, rhs = float(a[k]), float(b[k])
    scale = max(abs(lhs), abs(rhs), 1e-12)
    assert abs(lhs - rhs) / scale < 1e-6, (k, lhs, rhs)
" 2>"$WORK/repro.err"; then
    say_ok "invariants_are_reproducible"
else
    say_fail "invariants_are_reproducible" "$(head -3 "$WORK/repro.err")"
fi

# ---- 3. SECURITY: a sample carrying coded BCs is refused, and never solved ---------------------------------
rm -f "$WORK/brae-benchmark.json"
"$BIN" benchmark pimplefoam/coded-attack > "$WORK/coded.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ]; then
    say_fail "refuses_coded_case" "ran a sample carrying device code"
elif ! grep -q "executable content" "$WORK/coded.log"; then
    say_fail "refuses_coded_case" "refused, but not for carrying code"; sed -n '1,10p' "$WORK/coded.log"
elif [ -f "$WORK/brae-benchmark.json" ]; then
    say_fail "refuses_coded_case" "the case was run anyway (a result was written)"
else
    say_ok "refuses_coded_case"
fi

# ...and the `type coded;` Function1, which brae compiles with the host C++ compiler. Its own branch, so the
# clean sample above is the control: the guard refuses these two and still runs pitz-tiny.
rm -f "$WORK/brae-benchmark.json"
"$BIN" benchmark pimplefoam/coded-function1-attack > "$WORK/coded_f1.log" 2>&1
rc=$?
if [ "$rc" -eq 0 ]; then
    say_fail "refuses_coded_function1_case" "ran a sample carrying a coded Function1"
elif ! grep -q "type coded" "$WORK/coded_f1.log"; then
    say_fail "refuses_coded_function1_case" "refused, but not for the Function1"; sed -n '1,10p' "$WORK/coded_f1.log"
elif [ -f "$WORK/brae-benchmark.json" ]; then
    say_fail "refuses_coded_function1_case" "the case was run anyway (a result was written)"
else
    say_ok "refuses_coded_function1_case"
fi

# ---- 4. bad names and bad samples --------------------------------------------------------------------------
# Samples are grouped as <solver>/<case>, so '/' is legal in a name. Traversal therefore has to be excluded by
# rule rather than by banning the separator -- these are the cases that would escape the cache directory.
for bad in "../../etc" "pimplefoam/../../etc" "/etc/passwd" "pimplefoam//x" "pimplefoam/" "-rf" "pimplefoam/.hidden"; do
    "$BIN" benchmark "$bad" > "$WORK/traversal.log" 2>&1
    # A leading '-' is refused one layer earlier, by the option parser, so accept either rejection.
    if [ "$?" -eq 0 ] || ! grep -qE "may only contain|may not start|may not contain|empty path component|component starting|unknown option" "$WORK/traversal.log"; then
        say_fail "rejects_bad_name[$bad]" "accepted a name that escapes or is malformed"
        sed -n '1,4p' "$WORK/traversal.log"
    else
        say_ok "rejects_bad_name[$bad]"
    fi
done

"$BIN" benchmark pimplefoam/no-such-sample > "$WORK/missing.log" 2>&1
if [ "$?" -eq 0 ] || ! grep -q -- "--list" "$WORK/missing.log"; then
    say_fail "unknown_sample_points_at_list" "no usable error for an unknown sample"
else
    say_ok "unknown_sample_points_at_list"
fi

"$BIN" benchmark pimplefoam/no-manifest > "$WORK/nomanifest.log" 2>&1
if [ "$?" -eq 0 ] || ! grep -q "brae-bench.json" "$WORK/nomanifest.log"; then
    say_fail "requires_manifest" "a branch without a manifest was treated as a benchmark"
else
    say_ok "requires_manifest"
fi

[ "$fail" -eq 0 ] && echo "PASS: benchmark pulls, verifies and runs a template sample"
exit "$fail"
