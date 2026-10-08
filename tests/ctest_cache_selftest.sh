#!/usr/bin/env bash
# tools/ctest_cached.py held to what it promises, on a project of four tests made here: a test that passed
# is left out while its inputs stand, and runs again when one of them changes -- whichever kind of input.
#   alpha   a script that sources a helper            beta    a script whose exit status is a file's content
#   gamma   a script handed a data directory          delta   a script that WRITES into the directory it reads
# Arms, each the summary line's counts: (1) a first run runs all four, and beta's failure is the exit status;
# (2) again: alpha and gamma are held, beta (it failed) and delta (its run changed its own inputs) run;
# (3) beta mended: it runs and passes, (4) and is then held; (5) the helper alpha sources edited: alpha runs;
# (6) a data file touched, same bytes: gamma is still held; (7) its bytes changed: gamma runs; (8) --no-cache
# runs all four; (9) another BRAE_TEST_CACHE_SALT holds none. No GPU, no solver.
set -u
TOOL="${1:?tools/ctest_cached.py}"
WORK="${2:?work directory}"
rm -rf "$WORK"
mkdir -p "$WORK/proj/tests" "$WORK/proj/data" "$WORK/proj/grown" "$WORK/store"
P="$WORK/proj"
cat > "$P/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.24)
project(cachedSelftest NONE)
enable_testing()
add_test(NAME alpha COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/alpha.sh)
add_test(NAME beta COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/beta.sh)
add_test(NAME gamma COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/gamma.sh ${CMAKE_CURRENT_SOURCE_DIR}/data)
add_test(NAME delta COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tests/delta.sh ${CMAKE_CURRENT_SOURCE_DIR}/grown)
EOF
printf '#!/usr/bin/env bash\n. "$(dirname "$0")/helper.sh"\nhelped\n' > "$P/tests/alpha.sh"
printf 'helped() { return 0; }\n' > "$P/tests/helper.sh"
printf '#!/usr/bin/env bash\nexit "$(cat "$(dirname "$0")/beta.rc")"\n' > "$P/tests/beta.sh"
echo 1 > "$P/tests/beta.rc"
printf '#!/usr/bin/env bash\n[ -f "$1/values.txt" ]\n' > "$P/tests/gamma.sh"
echo "1 2 3" > "$P/data/values.txt"
printf '#!/usr/bin/env bash\ndate +%%s%%N >> "$1/log.txt"\n' > "$P/tests/delta.sh"
chmod +x "$P"/tests/*.sh
cmake -S "$P" -B "$P/build" > "$WORK/configure.log" 2>&1 || { echo "FAIL: the project did not configure"; exit 1; }

fail=0
# arm <name> <held> <to run> <exit status: 0 | nonzero> [arguments or NAME=value ...]
arm()
{
    local name="$1" held="$2" run="$3" status="$4" log="$WORK/$1.log"
    shift 4
    local envs=()
    while [ $# -gt 0 ] && [[ "$1" == *=* ]] && [[ "$1" != --* ]]; do
        envs+=("$1")
        shift
    done
    env "${envs[@]}" python3 "$TOOL" --test-dir "$P/build" --store "$WORK/store" "$@" > "$log" 2>&1
    local rc=$?
    local got
    got=$(sed -n -E 's/^ctest_cached: [0-9]+ tests selected, ([0-9]+) passed before.*, ([0-9]+) to run.*/\1 \2/p' "$log")
    local ok=1
    [ "$got" = "$held $run" ] || ok=0
    if [ "$status" = 0 ]; then [ $rc -eq 0 ] || ok=0; else [ $rc -ne 0 ] || ok=0; fi
    if [ $ok = 1 ]; then
        echo "ok:   $name ($held held, $run run, exit $rc)"
    else
        echo "FAIL: $name -- expected $held held, $run to run, exit $status; got '$got', exit $rc"
        sed -n '1,12p' "$log"
        fail=1
    fi
}
ran() { grep -qE "Test +#[0-9]+: $2 " "$WORK/$1.log"; }

arm first_run 0 4 nonzero
arm second_run 2 2 nonzero
ran second_run beta && ran second_run delta && ! ran second_run alpha && ! ran second_run gamma \
    || { echo "FAIL: second_run -- the two that ran are not beta and delta"; fail=1; }
grep -q "NOT recorded.*delta" "$WORK/first_run.log" \
    || { echo "FAIL: first_run -- a test that changed its own inputs was not named"; fail=1; }
echo 0 > "$P/tests/beta.rc"
arm beta_mended 2 2 0
arm beta_held 3 1 0
printf 'helped() { true; return 0; }\n' > "$P/tests/helper.sh"
arm helper_edited 2 2 0
ran helper_edited alpha || { echo "FAIL: helper_edited -- alpha did not run"; fail=1; }
touch "$P/data/values.txt"
arm data_touched 3 1 0
echo "1 2 4" > "$P/data/values.txt"
arm data_changed 2 2 0
ran data_changed gamma || { echo "FAIL: data_changed -- gamma did not run"; fail=1; }
arm no_cache 0 4 0 --no-cache
arm salted 0 4 0 BRAE_TEST_CACHE_SALT=another
python3 "$TOOL" --test-dir "$P/build" --store "$WORK/store" --list > "$WORK/list.log" 2>&1
[ "$(grep -c '^held ' "$WORK/list.log")" = 3 ] && grep -q '^run  delta$' "$WORK/list.log" \
    && echo "ok:   list (three held, delta to run)" || { echo "FAIL: list"; sed -n '1,8p' "$WORK/list.log"; fail=1; }
[ "$fail" -eq 0 ] && echo "PASS: a passed test is held while its inputs stand, and runs when one changes"
exit "$fail"
