#!/usr/bin/env bash
# The write gate, a rigid body's state file: the state file at writePrecision 6.
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
q_of_case
# C: t and deltaT are written at writePrecision, not at uniform/time's maximum (TimeIO.C:528) -- which
# writePrecision 17 cannot tell apart. At 6 over three steps of 0.01 the file must be OpenFOAM's byte for
# byte after the banner. FAIL-PROOF (2026-09-30): t written at 17 digits put `0.029999999999999999`
# where OpenFOAM writes `0.03`.
for side in of br; do
    d="$W/q_p6_$side"
    rscase "$d"
    sed -i -E 's/^(writePrecision\s+)[^;]*;/\16;/; s/^(endTime\s+)[^;]*;/\10.03;/' "$d/system/controlDict"
done
runof "$W/q_p6_of"
runbrae "$W/q_p6_br" host
python3 - "$W/q_p6_of" "$W/q_p6_br" <<'EOF_P6' && say "ARM Q  [host] floatingObject at writePrecision 6: rigidBodyMotionState byte-identical after the banner" ok \
                                              || say "ARM Q  [host] floatingObject at writePrecision 6: rigidBodyMotionState byte-identical after the banner" FAIL
import os, re, sys
of, br = sys.argv[1], sys.argv[2]
times = sorted([t for t in os.listdir(of) if re.match(r'^[0-9.e+-]+$', t) and t != '0'], key=float)
bad = len(times) != 3
for t in times:
    so = open('%s/%s/uniform/rigidBodyMotionState' % (of, t)).read()
    so = so[so.find('// * * *'):]
    try:
        sb = open('%s/%s/uniform/rigidBodyMotionState' % (br, t)).read()
        sb = sb[sb.find('// * * *'):]
    except OSError:
        sb = ''
    tl = re.search(r'^t\s+(\S+);', so, re.M)
    print('      %s: OpenFOAM t %s; %s' % (t, tl.group(1) if tl else '?', 'identical' if so == sb else 'DIFFERS'))
    bad += so != sb
sys.exit(1 if bad else 0)
EOF_P6
finish "arm Q: the state file is byte-identical at writePrecision 6"
