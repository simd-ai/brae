#!/usr/bin/env bash
# The legacy simpleFoam driver WRITES its AMG hierarchy cache by default and reads it on the next run, as
# interFoam's driver does (BRAE_AMG_CACHE=0 opts out of both). It wrote one only under -partition or
# BRAE_MESH_CACHE=1, so every run on the boxy5 UAV (4.94 M cells) agglomerated again at start-up.
#   ARM 1    a fresh copy of validation/pitzDailyTurb: the log says "built and written", the file is there.
#   ARM 2    the same directory again: the log says "read from the cache", and the fields it writes are the
#            first run's to the byte -- the hierarchy read is the one that was built.
#   CONTROL  BRAE_AMG_CACHE=0 on a fresh copy: no file written, and neither line.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="$ROOT/validation/pitzDailyTurb"
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$SRC" ] || { echo "SKIP: fixture $SRC missing"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
W=$(mktemp -d)
[ -n "${KEEP_W:-}" ] && { W="$KEEP_W"; rm -rf "${W:?}"; mkdir -p "$W"; } || trap 'rm -rf "$W"' EXIT
fail=0
say()
{
    printf '  %-66s %s\n' "$1" "$2"
    [ "$2" = FAIL ] && fail=1
    return 0
}
stage()
{
    mkdir -p "$1"
    cp -r "$SRC/constant" "$SRC/system" "$SRC/0" "$1/"
    sed -i -e 's/^endTime .*/endTime         20;/' -e 's/^writeInterval .*/writeInterval   20;/' "$1/system/controlDict"
}
cachefiles()
{
    ls -A "$1/constant/polyMesh" | grep -c "^\.brae_amgcache"
}

stage "$W/a"
[ "$(cachefiles "$W/a")" = 0 ] || say "PREMISE the fixture carries no AMG cache" FAIL
( cd "$W/a" && env -u BRAE_AMG_CACHE "$BRAE" -case . > log1 2> err1 ) || { say "ARM 1 ran" FAIL; tail -3 "$W/a/err1"; }
grep -q "AMG hierarchy: built and written to the cache" "$W/a/log1" && say "ARM 1 says it built and wrote the hierarchy" ok \
    || say "ARM 1 says it built and wrote the hierarchy" FAIL
[ "$(cachefiles "$W/a")" -ge 1 ] && say "ARM 1 left a cache file in constant/polyMesh" ok || say "ARM 1 left a cache file in constant/polyMesh" FAIL
mv "$W/a/20" "$W/a20_first"
( cd "$W/a" && env -u BRAE_AMG_CACHE "$BRAE" -case . > log2 2> err2 ) || { say "ARM 2 ran" FAIL; tail -3 "$W/a/err2"; }
grep -q "AMG hierarchy: read from the cache" "$W/a/log2" && say "ARM 2 says it read the hierarchy" ok || say "ARM 2 says it read the hierarchy" FAIL
same=1
for f in U p k epsilon nut
do
    cmp -s "$W/a20_first/$f" "$W/a/20/$f" || { same=0; echo "    $f differs"; }
done
[ $same = 1 ] && say "ARM 2 writes ARM 1's U p k epsilon nut to the byte" ok || say "ARM 2 writes ARM 1's U p k epsilon nut to the byte" FAIL

stage "$W/c"
( cd "$W/c" && BRAE_AMG_CACHE=0 "$BRAE" -case . > log 2> err ) || { say "CONTROL ran" FAIL; tail -3 "$W/c/err"; }
[ "$(cachefiles "$W/c")" = 0 ] && ! grep -q "AMG hierarchy: \(built and written\|read from\)" "$W/c/log" \
    && say "CONTROL BRAE_AMG_CACHE=0 writes no file and says neither" ok || say "CONTROL BRAE_AMG_CACHE=0 writes no file and says neither" FAIL

[ $fail -eq 0 ] && echo PASS || echo FAIL
exit $fail
