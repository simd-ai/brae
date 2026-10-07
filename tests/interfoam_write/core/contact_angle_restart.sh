#!/usr/bin/env bash
# The write gate: A RESTART OF A CONTACT-ANGLE WALL. alphaContactAngleTwoPhaseFvPatchScalarField's dictionary
# constructor reads the patch's `gradient` and evaluates the fixed gradient on it -- value = cell value +
# gradient/deltaCoeffs -- or, with no such entry, takes the cells' value and a zero gradient; `value` is never
# read (alphaContactAngleTwoPhaseFvPatchScalarField.C:61-83). The first curvature pass then reads that
# gradient through grad(alpha)'s patch correction. brae took the file's `value` and a zero gradient, so a
# restart began with a cold contact line.
# THE FIXTURE is laminar/capillaryRise's pinned row: real interFoam's own second directory, from which OpenFOAM
# and both of brae's loops take two more steps. ORACLE: OpenFOAM's restart. The directory holds alpha.water_0
# too (alpha is sub-cycled), whose wall OpenFOAM constructs the same way and whose gradient every later
# alpha.water_0 carries: brae REFUSED to write that file on such a restart, as not read, and reads it now.
# Three checks. (1) THE PREMISES, on OpenFOAM, as numbers: restarted with the two walls' `value` overwritten
# it writes the same bytes (the entry is not read), and restarted with their `gradient` REMOVED it does not
# (the entry is) -- how far apart, in its worst file. (2) Each loop within the bound of OpenFOAM's restart,
# alpha.water_0 among the files. (3) CONTROL: BRAE_CONTROL_CONTACT_ANGLE_GRADIENT_IGNORED=1 drops the files'
# gradient: over the bound against the oracle, and within it against OpenFOAM's own restart without the entry
# -- the control IS that run.
# MEASURED 2026-10-07: OpenFOAM's two restarts are 1.0e+00 apart in alpha.water_0 (its wall gradient) and
# 9.0e-02 in U; brae's host loop is 1.5e-14 from the oracle and its device loop 1.7e-14 (U), so the bound is
# 2e-13; the control is 9.0e-02 from the oracle in U and 1.3e-14 from OpenFOAM's restart without the entry.
. "$(dirname "$0")/../lib.sh"
[ $GPU -eq 1 ] || { echo "SKIP: no GPU for the device arm"; exit 77; }
wcase capillaryRise of > "$W/ca_stage.txt" 2>&1
o="$W/w_of_capillaryRise"
[ -d "$o" ] || { say "capillaryRise did not stage" FAIL; finish "a contact-angle wall's restart"; }
last=$(timedirs "$o" | awk '{print $NF}')
restage()   # restage <dir> <asWritten|noGradient|otherValue>: OpenFOAM's last directory, two more steps
{
    rm -rf "${1:?}"
    mkdir -p "$1"
    cp -r "$o/constant" "$o/system" "$o/$last" "$1/"
    python3 - "$1" "$last" "$2" <<'PY'
import re, sys
d, last, kind = sys.argv[1:4]
c = d + '/system/controlDict'
t = open(c).read()
t = re.sub(r'^startFrom\s.*', 'startFrom       latestTime;', t, flags=re.M)
t = re.sub(r'^endTime\s.*', 'endTime         %.12g;' % (2*float(last)), t, flags=re.M)
open(c, 'w').write(t)
# the wall of alpha AND of its stored old level: OpenFOAM constructs both patches from their files
for name in ('alpha.water', 'alpha.water_0'):
    p = d + '/' + last + '/' + name
    t = open(p).read()
    m = re.search(r'\n    walls\n    \{.*?\n    \}', t, re.S)
    block = m.group(0)
    assert 'constantAlphaContactAngle' in block
    g = re.search(r'\n\s*gradient\s+nonuniform List<scalar>\s*\d+\s*\(.*?\)\s*;', block, re.S)
    v = re.search(r'\n\s*value\s+nonuniform List<scalar>\s*\d+\s*\(.*?\)\s*;', block, re.S)
    assert g and v, 'the wall of %s holds no gradient list and value list' % name
    assert max(abs(float(x)) for x in re.search(r'\((.*)\)', g.group(0), re.S).group(1).split()) > 0
    if kind == 'noGradient':
        block2 = block.replace(g.group(0), '', 1)
    elif kind == 'otherValue':
        block2 = block.replace(v.group(0), '\n        value           uniform 0.5;', 1)
    else:
        block2 = block
    open(p, 'w').write(t.replace(block, block2, 1))
PY
}
for k in asWritten noGradient otherValue; do
    restage "$W/ca_of_$k" $k || { say "the restart did not stage [$k]" FAIL; finish "a contact-angle wall's restart"; }
    runof "$W/ca_of_$k"
done
rt=$(timedirs "$W/ca_of_asWritten" | tr ' ' '\n' | grep -vx "$last" | tr '\n' ' ')
n=0
for t in $rt; do
    for f in $(cd "$W/ca_of_asWritten/$t" && find . -type f | sort); do
        cmp -s "$W/ca_of_asWritten/$t/$f" "$W/ca_of_otherValue/$t/$f" || n=$((n + 1))
    done
done
# worst <result file>: the worst file's relative distance, cumulativeContErr left out
worst()
{
    python3 - "$1" <<'PY'
import json, sys
try:
    r = json.loads([l for l in open(sys.argv[1]) if l.startswith('RESULT ')][-1][7:])
    print('%.1e' % max(v['rel'] for k, v in r['files'].items() if 'cumulativeContErr' not in k))
except Exception:
    print('-')
PY
}
python3 "$CMP" "$W/ca_of_asWritten" "$W/ca_of_noGradient" $rt > "$W/cmp_ca_premise.txt" 2>&1
apart=$(worst "$W/cmp_ca_premise.txt")
what="PREMISES OpenFOAM restarted to [$rt]: with the wall's value overwritten $n files differ; with its gradient"
what="$what removed the worst file is $apart away"
[ "$(echo $rt | wc -w)" = 2 ] && [ "$n" = 0 ] && above "$apart" 1e-8 && say "$what" ok || say "$what" FAIL
for v in host device control; do
    restage "$W/ca_$v" asWritten
    case $v in
        host)    runbrae "$W/ca_$v" host BRAE_X=1 ;;
        device)  runbrae "$W/ca_$v" device BRAE_X=1 ;;
        control) runbrae "$W/ca_$v" host BRAE_CONTROL_CONTACT_ANGLE_GRADIENT_IGNORED=1 ;;
    esac
    python3 "$CMP" "$W/ca_of_asWritten" "$W/ca_$v" $rt > "$W/cmp_ca_$v.txt" 2>&1
done
B=2e-13
judge "the contact-angle restart, host" "$W/cmp_ca_host.txt" "$B" "$W/ca_of_asWritten/log.interFoam" \
    && judge "the contact-angle restart, device" "$W/cmp_ca_device.txt" "$B" "$W/ca_of_asWritten/log.interFoam" \
    && say "[host, device] every written file within $B of OpenFOAM's restart" ok \
    || say "[host, device] every written file within $B of OpenFOAM's restart" FAIL
python3 "$CMP" "$W/ca_of_noGradient" "$W/ca_control" $rt > "$W/cmp_ca_control_cold.txt" 2>&1
judge "control against the oracle" "$W/cmp_ca_control.txt" "$B" "$W/ca_of_asWritten/log.interFoam" > "$W/ca_c1.txt"
c1=$?
judge "control against the cold restart" "$W/cmp_ca_control_cold.txt" "$B" "$W/ca_of_noGradient/log.interFoam" \
    > "$W/ca_c2.txt"
c2=$?
what="CONTROL  the file's gradient dropped: $(worst "$W/cmp_ca_control.txt") from the oracle, and"
what="$what $(worst "$W/cmp_ca_control_cold.txt") from OpenFOAM's own restart without the entry"
[ $c1 != 0 ] && [ $c2 = 0 ] && say "$what" ok || say "$what" FAIL
finish "a contact-angle wall restarts with the gradient the run was written with, as in OpenFOAM"
