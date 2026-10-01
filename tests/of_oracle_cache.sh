# Sourced by the gates. Caches the ORACLE -- the real-OpenFOAM run a profile is compared against --
# so a gate re-run does not pay for it again.
#
# WHY. A gate's wall clock is not brae and it is not the other solvers' tests: on 2026-09-22 the
# scoped set took 956 s and `interfoam_moving_vs_openfoam` alone took 955.90 of it, because it stages
# 24 profiles and each needs blockMesh (+ snappyHexMesh) + setFields + a full serial interFoam before
# brae runs at all. Every other test in that set finished inside it; the shared-file consumers cost
# 0.4 to 17 s each. So the lever is the oracle, not the file scope.
#
# WHAT MAKES IT SAFE. A cached oracle that does not match its case is the worst defect this gate could
# have -- it would compare brae against someone else's answer and pass. So:
#
#   * the KEY is a sha256 of EVERY BYTE of the staged case at the moment of the call (after the
#     profile's edits, before blockMesh), plus the caller's own tag, plus the identity of the
#     interFoam binary. Anything that can change OpenFOAM's output is in the hash, because the whole
#     input is in the hash;
#   * the key is STORED INSIDE the archive and re-checked on restore, so a collision or a corrupted
#     file is caught rather than used;
#   * a restore that does not produce the expected end-time directory is a MISS, not a pass -- the
#     caller re-runs OpenFOAM.
#
# `BRAE_OF_CACHE=off` disables it. `BRAE_OF_CACHE=<dir>` moves it; the default is outside the repo.
: "${BRAE_OF_CACHE:=$HOME/.cache/brae/of-oracle}"

# oracleKey <caseDir> <tag...>  -- the hash of the staged case and everything else that decides the run
oracleKey()
{
    local C="$1"
    shift
    local ofbin
    ofbin=$(command -v interFoam 2>/dev/null)
    {
        # every staged byte, in a stable order, contents AND path -- EXCEPT the solver LOGS. A log is
        # pure output and carries `Date`, `Time`, `Host` and `PID`, so a case hashed after blockMesh
        # gets a new key on every run and the cache can never hit. MEASURED: the three gates that hook
        # AFTER meshing (les, baffle, ami) reused 0 of 14 oracles until this exclusion went in; the
        # moving gate hooks before blockMesh, where no log exists yet, which is why it hit from the
        # start. Excluding them weakens nothing -- no log is an input to OpenFOAM.
        find "$C" -type f ! -name 'log.*' ! -name '.brae-oracle-key' -print0 \
            | LC_ALL=C sort -z | xargs -0 sha256sum 2>/dev/null \
            | sed "s|$C/||"
        # ...and which OpenFOAM would run it
        [ -n "$ofbin" ] && stat -c '%s %Y %n' "$ofbin" 2>/dev/null
        printf '%s\n' "$@"
    } | sha256sum | cut -c1-40
}

# oracleRestore <caseDir> <key> <endDir>  -- 0 on a verified hit, 1 on a miss
oracleRestore()
{
    local C="$1" key="$2" end="$3"
    [ "$BRAE_OF_CACHE" = off ] && return 1
    local f="$BRAE_OF_CACHE/$key.tar.zst"
    [ -f "$f" ] || f="$BRAE_OF_CACHE/$key.tar.gz"
    [ -f "$f" ] || return 1
    local tmp
    tmp=$(mktemp -d) || return 1
    if ! tar -xf "$f" -C "$tmp" 2>/dev/null; then
        rm -rf "$tmp"
        return 1
    fi
    # the key travels INSIDE the archive: a name that matches is not the same as a case that matches
    if [ "$(cat "$tmp/.brae-oracle-key" 2>/dev/null)" != "$key" ]; then
        echo "  oracle cache: $key.tar.* does not carry its own key -- ignoring it"
        rm -rf "$tmp"
        return 1
    fi
    rm -f "$tmp/.brae-oracle-key"
    # ...and it has to hold the answer the caller is about to read: the end-time directory, or the file
    # the caller names as its marker of a finished run
    if [ ! -e "$tmp/$end" ]; then
        rm -rf "$tmp"
        return 1
    fi
    rm -rf "$C"
    mv "$tmp" "$C" || return 1
    return 0
}

# oracleStore <caseDir> <key>
oracleStore()
{
    local C="$1" key="$2"
    [ "$BRAE_OF_CACHE" = off ] && return 0
    mkdir -p "$BRAE_OF_CACHE" 2>/dev/null || return 0
    printf '%s' "$key" > "$C/.brae-oracle-key"
    local tmp
    if command -v zstd > /dev/null 2>&1; then
        tmp="$BRAE_OF_CACHE/.$key.$$.tar.zst"
        tar -C "$C" -cf - . 2>/dev/null | zstd -3 -q -o "$tmp" 2>/dev/null \
            && mv -f "$tmp" "$BRAE_OF_CACHE/$key.tar.zst"
    else
        tmp="$BRAE_OF_CACHE/.$key.$$.tar.gz"
        tar -C "$C" -czf "$tmp" . 2>/dev/null && mv -f "$tmp" "$BRAE_OF_CACHE/$key.tar.gz"
    fi
    rm -f "$tmp" "$C/.brae-oracle-key"
    return 0
}

# oracleRun <caseDir> <tag...>  -- real interFoam on the staged case, cached on every staged byte. Returns
# non-zero when interFoam itself failed; the caller reads log.interFoam either way.
oracleRun()
{
    local C="$1"
    shift
    local key
    key=$(oracleKey "$C" "$@" "run")
    if oracleRestore "$C" "$key" "log.interFoam"; then
        echo "  OpenFOAM's run reused from the oracle cache   [$(basename "$C")]"
        return 0
    fi
    ( cd "$C" && interFoam > log.interFoam 2>&1 ) || return 1
    oracleStore "$C" "$key"
}

# oracleMesh <caseDir> <tag> <function> [extra key...]  -- the case's meshing, cached. <function> does the
# meshing (in the caller's shell); the key is the case as it stands BEFORE it, the function's own text and
# whatever else the caller says decides the mesh (a geometry file outside the case).
oracleMesh()
{
    local C="$1" tag="$2" fn="$3"
    shift 3
    local key
    key=$(oracleKey "$C" "$tag" "mesh" "$(declare -f "$fn" | sha256sum | cut -c1-16)" "$@")
    if oracleRestore "$C" "$key" ".brae-mesh-done"; then
        echo "  the mesh reused from the oracle cache   [$(basename "$C")]"
        return 0
    fi
    "$fn" || return 1
    : > "$C/.brae-mesh-done"
    oracleStore "$C" "$key"
}
