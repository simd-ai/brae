#!/usr/bin/env bash
# The write gate: hooks_empty_on_device.sh on a MOVING 2-D mesh, waveMakerFlap, whose two empty patches split
# the other patches into runs (a wrong run offset shows here and nowhere on a mesh with its empty patch last)
# and whose mesh update reads the host's fields between two steps. With an empty patch's entries kept on the
# GPU the host no longer evaluates alpha there nor builds the mixture's boundary there: its lists on those
# patches are stale. THE CLAIM that nothing reads them stale is held by NaN written into them at every call
# (BRAE_CONTROL_HOOKS_EMPTY_POISON=1), on the mesh where the most readers run.
# IT FOUND ONE the first time it ran (2026-10-06): the mesh update's curvature pass, calculateK whole, reads
# alpha's value on every patch -- p, U, phi, rAU and Uf were NaN in the second step. The driver now evaluates
# alpha's empty patches before the update; the CONTROL leaves them stale under the poison and has to show NaN.
# (Stale and finite, the two pinned steps of this row write the same bytes: alpha has not moved there yet. The
# poison is what sees the reader, not the files.)
# The oracle is brae's own host build, byte for byte on the written files, and the in-code bitwise check.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase waveMakerFlap of > "$W/hm_stage.txt" 2>&1
o="$W/w_of_waveMakerFlap"
[ -d "$o" ] || { say "waveMakerFlap did not stage" FAIL; finish "hooks' empty faces on a moving mesh identity"; }
for v in host check plain poison stale; do
    e="$W/hm_$v"
    mkdir -p "$e"
    cp -r "$o/0" "$o/constant" "$o/system" "$e/"
    case $v in
        host)   runbrae "$e" device BRAE_CONTROL_HOOKS_EMPTY_FROM_HOST=1 ;;
        check)  runbrae "$e" device BRAE_CONTROL_HOOKS_EMPTY_CHECK=1 ;;
        plain)  runbrae "$e" device ;;
        poison) runbrae "$e" device BRAE_CONTROL_HOOKS_EMPTY_POISON=1 ;;
        stale)  runbrae "$e" device BRAE_CONTROL_HOOKS_EMPTY_POISON=1 \
                    BRAE_CONTROL_HOOKS_EMPTY_STALE_AT_MESH_UPDATE=1 ;;
    esac
done
differs()
{
    local n=0 t f
    for t in $(timedirs "$o"); do
        for f in $(cd "$1/$t" && find . -type f | sort); do
            cmp -s "$1/$t/$f" "$2/$t/$f" || n=$((n + 1))
        done
    done
    echo $n
}
nans()
{
    grep -raiwl "nan" $(for t in $(timedirs "$o"); do echo "$1/$t"; done) | wc -l
}
arrays=$(grep -a "hooks empty check:" "$W/hm_check/log.brae" | sed -E 's/.*check: (.*), [0-9]+ boundary.*/\1/' \
    | sort -u | wc -l)
n=$(differs "$W/hm_host" "$W/hm_check")
m=$(differs "$W/hm_host" "$W/hm_plain")
what="[device] $arrays arrays the host's whole build at every call, bitwise; files differ: checked $n, plain $m"
grep -q "hooks \[CHECKED" "$W/hm_check/log.brae" && grep -q "hooks: an empty patch's entries" "$W/hm_plain/log.brae" \
    && ! grep -q "hooks.*an empty patch's entries" "$W/hm_host/log.brae" \
    && [ "$arrays" -ge 5 ] && [ "$n" = 0 ] && [ "$m" = 0 ] && [ "$(timedirs "$W/hm_plain")" = "$(timedirs "$o")" ] \
    && say "$what" ok || say "$what" FAIL
n=$(differs "$W/hm_host" "$W/hm_poison")
nan=$(nans "$W/hm_poison")
what="[device] NaN in the host's lists on the empty patches at every call: $n files differ, $nan hold a nan"
grep -q "hooks \[the host's lists there POISONED" "$W/hm_poison/log.brae" \
    && [ "$n" = 0 ] && [ "$nan" = 0 ] && say "$what" ok || say "$what" FAIL
nan=$(nans "$W/hm_stale")
what="CONTROL  the mesh update left on alpha's poisoned values there: $nan written files hold a nan"
grep -q "CONTROL MODE: the mesh update reads alpha's stale values" "$W/hm_stale/log.brae" \
    && [ "$nan" != 0 ] && say "$what" ok || say "$what" FAIL
finish "on a moving mesh the hooks' arrays kept on the GPU are the host's, and nothing reads the host's stale lists"
