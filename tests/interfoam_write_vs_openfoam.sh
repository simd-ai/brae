#!/usr/bin/env bash
# The whole write gate in one go, for a run by hand: every file under tests/interfoam_write/<arm>/ and every
# tutorial row, one after another. ctest does not call this -- it registers each file and each tutorial as
# its own test (CMakeLists.txt, `interfoam_write_*`; tests/interfoam_write/lib.sh has the gate's header,
# helpers, bounds and cache).
#   WRITE_ONLY="core/when.sh tutorial:damBreakLeakage" runs the named ones alone.
here="$(cd "$(dirname "$0")" && pwd)/interfoam_write"
keys=$(sed -n '/^W_CASES="/,/^"/p' "$here/lib.sh" | grep ':' | sed 's/:.*//; s#.*/##')
all=$(cd "$here" && ls */*.sh | grep -v '/_common.sh$')
for k in $keys; do
    all="$all tutorial:$k"
done
rc=0
for a in ${WRITE_ONLY:-$all}; do
    echo "== $a"
    if [ "${a#tutorial:}" != "$a" ]; then
        bash "$here/tutorial.sh" "${a#tutorial:}" || rc=1
    else
        bash "$here/$a" || rc=1
    fi
done
exit $rc
