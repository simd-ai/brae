#!/usr/bin/env bash
# The write gate on damBreak: OpenFOAM restarts from what brae wrote (arm E), with its control.
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
core_base
# E: OpenFOAM restarts from brae's 0.1, and from its own, to 0.11
restart()   # restart <dir> <from case> [drop uniform/time]
{
    mkdir -p "$1"
    cp -r "$W/of/constant" "$W/of/system" "$1/"
    cp -r "$2/0.1" "$1/"
    [ "${3:-}" = drop ] && rm -f "$1/0.1/uniform/time"
    sed -i 's/^startFrom .*/startFrom       latestTime;/; s/^endTime .*/endTime         0.11;/' "$1/system/controlDict"
    ( cd "$1" && interFoam > log.interFoam 2>&1 )
}
firsts() { python3 - "$1" <<'PY'
import re, sys
s = open(sys.argv[1]).read()
dt = re.findall(r'^deltaT = (\S+)', s, flags=re.M)
cu = re.findall(r'cumulative = (\S+)', s)
print(dt[0] if dt else 'none', cu[0] if cu else 'none')
PY
}
restart "$W/rs_of" "$W/of" || { say "ARM E  OpenFOAM restarts from its own 0.1" FAIL; }
ref=$(firsts "$W/rs_of/log.interFoam")
for arm in $ARMS; do
    if restart "$W/rs_$arm" "$W/br_$arm" && grep -q "^End" "$W/rs_$arm/log.interFoam"; then
        got=$(firsts "$W/rs_$arm/log.interFoam")
        python3 - $ref $got "$BOUND_TIME" "$BOUND_FIELDS" <<'PY' && say "ARM E  [$arm] OpenFOAM restarts from brae's 0.1: first deltaT and cumulative continue brae's" ok \
                                          || say "ARM E  [$arm] OpenFOAM restarts from brae's 0.1: first deltaT and cumulative continue brae's" FAIL
import sys
dto, cuo, dtb, cub, bound, boundc = sys.argv[1:7]
rd = abs(float(dtb) - float(dto)) / float(dto)
rc = abs(float(cub) - float(cuo)) / max(abs(float(cuo)), 1e-300)
print('      first deltaT OF-from-OF %s, OF-from-brae %s (%.3e); first cumulative %s vs %s (%.3e)'
      % (dto, dtb, rd, cuo, cub, rc))
sys.exit(0 if rd < float(bound) and rc < float(boundc) else 1)
PY
    else
        say "ARM E  [$arm] OpenFOAM restarts from brae's 0.1 and runs to 0.11" FAIL
        tail -5 "$W/rs_$arm/log.interFoam" | sed 's/^/      /'
    fi
done
restart "$W/rs_drop" "$W/br_host" drop
got=$(firsts "$W/rs_drop/log.interFoam")
python3 - $ref $got <<'PY' && say "CONTROL  brae's 0.1 without uniform/time moves E's first deltaT" ok \
                           || say "CONTROL  brae's 0.1 without uniform/time moves E's first deltaT" FAIL
import sys
dto, dtb = sys.argv[1], sys.argv[3]
print('      first deltaT OF-from-OF %s, from brae without uniform/time %s' % (dto, dtb))
sys.exit(0 if dtb == 'none' or abs(float(dtb) - float(dto)) / float(dto) > 1e-3 else 1)
PY
finish "arm E: OpenFOAM continues from brae's write"
