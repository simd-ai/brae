# Sourced by the gates. `requireFresh <binary> [<binary> ...]` fails when a binary is OLDER than the
# newest source file under src/, i.e. when the gate is about to measure code that is not the code in
# the tree.
#
# WHY THIS EXISTS, with the measurement. On 2026-09-22 two scoped `ctest` runs reported
# `interfoam_refusals` green while SIX of its arms named refusals that had just been lifted -- the
# gate runs build/brae_interFoam and only the `test_*` targets had been rebuilt. The arms were
# correct; they were reading a binary from before the change. A gate that silently measures a stale
# binary is worse than one that does not run: it reports a pass nobody earned.
#
# It FAILS rather than SKIPS, and it says what to run. A skip is what a missing OpenFOAM gets; a
# stale binary is a mistake in the loop the developer is standing in.
requireFresh()
{
    local root newest b
    root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    # the newest thing the binaries are built FROM. -newer is a per-file test, so the comparison is
    # done by find rather than by parsing timestamps, which differ in resolution across filesystems.
    newest=$(find "$root/src" -name '*.cu' -o -name '*.cuh' -o -name '*.H' -o -name '*.C' \
             2>/dev/null | head -1)
    [ -n "$newest" ] || return 0
    for b in "$@"; do
        [ -x "$b" ] || continue
        local stale
        stale=$(find "$root/src" \( -name '*.cu' -o -name '*.cuh' -o -name '*.H' -o -name '*.C' \) \
                -newer "$b" -print -quit 2>/dev/null)
        if [ -n "$stale" ]; then
            echo "FAIL: $(basename "$b") is older than $stale"
            echo "      The gate would measure a binary built before that edit. Build first:"
            echo "        ( cd ${BUILD:-$root/build} && make -j )"
            return 1
        fi
    done
    return 0
}
