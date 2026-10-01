#!/usr/bin/env bash
# The write gate: the wave models' list entries are byte-identical to OpenFOAM's, with the raw-token control.
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
wcase irregularMultiDirection
wcase streamFunction
# V: the wave models' list entries (irregularMultiDirection's 57-row wavePeriods/waveHeights/wavePhases/
# waveDirs, streamFunction's Bjs/Ejs) are written as primitiveEntry writes them -- each token re-emitted at
# writePrecision, joined by single spaces on one line. Arm W's comparer reads values and is blind to that
# form (`15.367` and `15.367000000000001` are one number to it), so the files are held BYTE-identical to
# OpenFOAM's after the banner, on every arm arm W ran. FAIL-PROOF (2026-09-30): the tokens joined raw, not
# re-emitted, wrote irregularMultiDirection's `rampTime 18.0;` where OpenFOAM writes `18` -- 2 of 4 files red.
for key in irregularMultiDirection streamFunction; do
    for arm in $ARMS; do
        d="$W/w_br_${key}_$arm"
        [ -d "$d" ] || { say "ARM V  [$arm] $key did not run in arm W" FAIL; continue; }
        python3 - "$W/w_of_$key" "$d" <<'EOF_V' && say "ARM V  [$arm] $key: every waveProperties file byte-identical to OpenFOAM's after the banner" ok \
                                           || say "ARM V  [$arm] $key: every waveProperties file byte-identical to OpenFOAM's after the banner" FAIL
import os, re, sys
of, br = sys.argv[1], sys.argv[2]
times = sorted([t for t in os.listdir(of) if re.match(r'^[0-9.e+-]+$', t) and t != '0'], key=float)
n = bad = 0
for t in times:
    for f in sorted(os.listdir('%s/%s/uniform' % (of, t))):
        if not f.startswith('waveProperties.'):
            continue
        n += 1
        so = open('%s/%s/uniform/%s' % (of, t, f)).read()
        try:
            sb = open('%s/%s/uniform/%s' % (br, t, f)).read()
        except OSError:
            sb = ''
        so, sb = so[so.find('// * * *'):], sb[sb.find('// * * *'):]
        if so != sb:
            bad += 1
            i = next((k for k in range(min(len(so), len(sb))) if so[k] != sb[k]), min(len(so), len(sb)))
            print('      %s/%s differs at byte %d: OpenFOAM %r, brae %r' % (t, f, i, so[i:i + 40], sb[i:i + 40]))
print('      %d files, %d differ' % (n, bad))
sys.exit(0 if n and not bad else 1)
EOF_V
    done
done
# CONTROL for V: brae's own irregularMultiDirection output with ONE token put back raw (`18` -> `18.0`,
# what joining the tokens unre-emitted wrote) must fail the byte check
if [ -d "$W/w_br_irregularMultiDirection_host" ]; then
    d="$W/w_ctl_vraw"
    rm -rf "$d"
    cp -r "$W/w_br_irregularMultiDirection_host" "$d"
    first=$(timedirs "$d" | awk '{print $1}')
    sed -i -E 's/^(rampTime +)18;/\118.0;/' "$d/$first/uniform/waveProperties.inlet"
    grep -q "^rampTime *18.0;" "$d/$first/uniform/waveProperties.inlet" || say "CONTROL  V: the raw token was not staged" FAIL
    vbytes "$W/w_of_irregularMultiDirection" "$d" > /dev/null \
        && say "CONTROL  one raw token (\`rampTime 18.0\`) fails arm V's byte check" FAIL \
        || say "CONTROL  one raw token (\`rampTime 18.0\`) fails arm V's byte check" ok
fi
finish "arm V: every waveProperties file is byte-identical"
