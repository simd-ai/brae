#!/usr/bin/env bash
# THE PRESSURE REFERENCE on the transient driver, against REAL OpenFOAM pimpleFoam.
#
# WHY: OpenFOAM's pimpleFoam runs setRefCell(p, pimple.dict(), pRefCell, pRefValue) at createFields.H:34-36
# and drives adjustPhi + pEqn.setReference off p.needReference() (pEqn.H:13-16, :46). brae's transient
# driver never set DeviceSimpleControls::needRef, so the shared DeviceSimpleSolver -- which reads it at
# four points (device_simple_foam.cu:2347, 2704, 3164-3170, 3214) -- took the default `false` and solved
# the singular all-Neumann system with neither the flux adjustment nor the reference, on every case whose
# pressure fixes no value. Nine of OpenFOAM's own pimpleFoam tutorials are such cases (LES/decayIsoTurb,
# LES/periodicHill, laminar/mixerVesselAMI2D, laminar/sloshing2D, laminar/planarPoiseuille ...). The
# steady driver has always set it; this is what carrying the two side by side costs. Found by
# tools/default_audit.py, not by a case.
#
# THE CASE: the committed pitzDaily fixture made TRANSIENT exactly as tests/pimple_run.sh makes it, with
# ONE edit -- the outlet's p from fixedValue to zeroGradient, so no p patch fixes a value and OpenFOAM's
# own p.needReference() is true. Both codes run the same fixed steps from the same start; brae is compared
# to OpenFOAM's own fields, cell by cell.
#
# THE CONTROL is brae with the reference disabled (BRAE_NO_PREF=1, the fail-proof hook this gate exists
# for): it must be FAR further from OpenFOAM than brae is, or the gate proves nothing. MEASURED: it
# DIVERGES at step two (non-finite residual at t = 4e-05), where the fixed driver runs the ten steps and
# reads U 6.9e-04 and p 2.8e-03 relative against OpenFOAM's own fields.
#
# NOT CLAIMED: those two numbers are the transient driver's agreement on this case (it reads 0.07% on U
# at steady on the shipped pitzDaily), not the reference's accuracy. This gate says the reference is
# APPLIED and that the run depends on it; closing the driver's own gap is another unit.
set -u
BIN="${1:?brae_pimpleFoam binary}"
SRC="${2:?committed pitzDaily case dir}"
W="${3:?work dir}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
STEPS=${STEPS:-10}
DT=${DT:-2e-5}

[ -f "$SRC/constant/polyMesh/points" ] || { echo "SKIP: fixture '$SRC' not present"; exit 125; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 125; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v pimpleFoam > /dev/null 2>&1 || { echo "SKIP: pimpleFoam not on PATH"; exit 125; }

END=$(python3 -c "print('%.10g' % ($STEPS*float('$DT')))")

stage()   # stage <dir>
{
    rm -rf "$1"; mkdir -p "$1"
    cp -r "$SRC/constant" "$SRC/system" "$SRC/0" "$1/"
    rm -rf "$1"/[1-9]* "$1"/log.*
    cat > "$1/system/controlDict" <<EOF
FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application     pimpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         $END;
deltaT          $DT;
writeControl    timeStep;
writeInterval   $STEPS;
purgeWrite      0;
writeFormat     ascii;
writePrecision  15;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable false;
adjustTimeStep  no;
EOF
    # transient schemes + a PIMPLE block, as tests/pimple_run.sh patches them
    python3 - "$1" <<'PYEOF' || return 1
import re, sys
d = sys.argv[1]
s = open(d + '/system/fvSchemes').read()
s = re.sub(r'ddtSchemes\s*\{[^}]*\}', 'ddtSchemes\n{\n    default         Euler;\n}', s, flags=re.S)
open(d + '/system/fvSchemes', 'w').write(s)
s = open(d + '/system/fvSolution').read()
# the algorithm block may hold a sub-dictionary (residualControl), so cut it by MATCHING braces --
# a non-greedy [^}]* stops at the first inner close and leaves a stray `}` behind
m = re.search(r'\n(SIMPLE|PIMPLE)\s*\{', s)
if m:
    i, depth = m.end() - 1, 0
    while i < len(s):
        if s[i] == '{':
            depth += 1
        elif s[i] == '}':
            depth -= 1
            if depth == 0:
                break
        i += 1
    s = s[:m.start()] + s[i + 1:]
s += ('\nPIMPLE\n{\n    nOuterCorrectors 2;\n    nCorrectors     2;\n'
      '    nNonOrthogonalCorrectors 0;\n    pRefCell        0;\n    pRefValue       0;\n}\n')
# `nOuterCorrectors 2` makes the last outer corrector select <field>Final (fvMatrix::solve through
# GeometricField::select), and OpenFOAM stops when the entry is absent. The steady fixture has none,
# so each solved field's own block is copied under its Final name -- the same settings, so the two
# codes solve the same systems and this gate stays about the pressure reference.
def block(text, key):
    m = re.search(r'\n(\s*)%s\s*\n?\s*\{' % re.escape(key), text)   # key may be a quoted regex
    if not m:
        return None
    i, depth = text.index('{', m.start()), 0
    while i < len(text):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[text.index('{', m.start()):i + 1]
        i += 1
    return None
# the fixture keys its momentum and turbulence solvers through ONE regex, so the Final copies are
# keyed the same way -- a literal `UFinal` beside a `"(U|k|epsilon...)"` entry would leave k and
# epsilon without theirs, which is the error OpenFOAM reports next
finals = ''
for key in ('p', '"(U|k|epsilon|omega|f|v2)"'):
    b = block(s, key)
    if b:
        finals += '\n    %s\n    %s\n' % (key[:-1] + 'Final"' if key.startswith('"') else key + 'Final', b)
assert finals.count('Final') >= 2, 'the fixture\'s fvSolution has no p and momentum solver blocks to copy'
m = re.search(r'\nsolvers\s*\n?\s*\{', s)
assert m, 'the fixture has no solvers dictionary'
i, depth = s.index('{', m.start()), 0
while i < len(s):
    if s[i] == '{':
        depth += 1
    elif s[i] == '}':
        depth -= 1
        if depth == 0:
            break
    i += 1
s = s[:i] + finals + s[i:]
open(d + '/system/fvSolution', 'w').write(s)
# THE EDIT: the outlet's p zeroGradient, so no p patch fixes a value
p = d + '/0/p'
t = open(p).read()
m = re.search(r'(\n    outlet\s*\{)([^}]*)(\})', t, re.S)
assert m, 'the fixture no longer has an `outlet` p patch'
assert 'fixedValue' in m.group(2), 'the fixture\'s outlet p is no longer fixedValue'
t = t[:m.start(2)] + '\n        type            zeroGradient;\n    ' + t[m.end(2):]
open(p, 'w').write(t)
assert 'fixedValue' not in open(p).read(), 'another p patch still fixes a value -- the case would not need a reference'
# ...and the outlet's U zeroGradient with it. adjustPhi needs a patch that fixes NEITHER: an
# inletOutlet U is a mixed field and mixedFvPatchField::fixesValue() is TRUE, so OpenFOAM itself
# stops with "Adjustable mass outflow: 0" while the inlet keeps pushing mass in.
u = d + '/0/U'
t = open(u).read()
m = re.search(r'(\n    outlet\s*\{)([^}]*)(\})', t, re.S)
assert m, 'the fixture no longer has an `outlet` U patch'
t = t[:m.start(2)] + '\n        type            zeroGradient;\n    ' + t[m.end(2):]
# ...and the field STARTED at the inlet velocity, not at rest: adjustPhi scales the outflow to the
# inflow and cannot start from zero outflow (adjustPhi.C:100-116, "Continuity error cannot be removed
# by adjusting the outflow"), which is OpenFOAM refusing, not brae. Both codes get this same start.
t, n = re.subn(r'^internalField\s+uniform\s+\([^)]*\);', 'internalField   uniform (10 0 0);', t, flags=re.M)
assert n == 1, 'U has no uniform internalField to start from the inlet value'
open(u, 'w').write(t)
PYEOF
}

stage "$W/of" || { echo "FAIL: staging"; exit 1; }
( cd "$W/of" && pimpleFoam > log.pimpleFoam 2>&1 ) \
    || { echo "FAIL: OpenFOAM pimpleFoam"; tail -20 "$W/of/log.pimpleFoam"; exit 1; }
[ -d "$W/of/$END" ] || { echo "FAIL: OpenFOAM wrote no $END"; exit 1; }
echo "OpenFOAM ran $STEPS steps of $DT to t = $END"

compare()   # compare <braeTimeDir> <label> -> prints "U <rel> p <rel>"
{
    python3 - "$W/of/$END" "$1" "$2" <<'PYEOF'
import re, sys
def read(path, vec):
    s = open(path).read()
    kind = 'vector' if vec else 'scalar'
    m = re.search(r'internalField\s+nonuniform List<%s>\s*(\d+)\s*\((.*?)\n\)\s*;' % kind, s, re.S)
    if not m:
        u = re.search(r'internalField\s+uniform\s+(\S+|\([^)]*\))\s*;', s)
        raise SystemExit('FAIL: %s has a uniform internalField (%s)' % (path, u.group(1) if u else '?'))
    if vec:
        return [tuple(float(x) for x in v.split()) for v in re.findall(r'\(([^()]*)\)', m.group(2))]
    return [(float(x),) for x in m.group(2).split()]
of_dir, br_dir, label = sys.argv[1], sys.argv[2], sys.argv[3]
out = []
for f, vec in (('U', True), ('p', False)):
    a, b = read(of_dir + '/' + f, vec), read(br_dir + '/' + f, vec)
    if len(a) != len(b):
        raise SystemExit('FAIL: %s cell counts differ (%d vs %d)' % (f, len(a), len(b)))
    d = max(max(abs(x - y) for x, y in zip(u, v)) for u, v in zip(a, b))
    ref = max(max(abs(x) for x in u) for u in a)
    out.append(d / ref)
print('%s %.4e %.4e' % (label, out[0], out[1]))
PYEOF
}

rc=0
stage "$W/brae" || { echo "FAIL: staging brae"; exit 1; }
( cd "$W/brae" && "$BIN" -case . > log.brae 2>&1 ) || { echo "FAIL: brae_pimpleFoam"; tail -20 "$W/brae/log.brae"; exit 1; }
grep -q "pressure needs reference" "$W/brae/log.brae" \
    || { echo "FAIL: brae did not report that this case needs a pressure reference"; rc=1; }
read -r _ gotU gotP < <(compare "$W/brae/$END" brae) || exit 1
echo "  brae vs OpenFOAM:            U $gotU   p $gotP"

# THE CONTROL: the same binary with the reference suppressed. It may end either way -- far from
# OpenFOAM, or not at all -- and a run that DIVERGES is the strongest form of "far", so it is asserted
# as its own outcome rather than read as a broken script. MEASURED: it diverges at step two.
stage "$W/noref" || { echo "FAIL: staging control"; exit 1; }
ctlDiverged=0
( cd "$W/noref" && BRAE_NO_PREF=1 "$BIN" -case . > log.brae 2>&1 ) || ctlDiverged=1
grep -q "pressure needs reference" "$W/noref/log.brae" \
    && { echo "FAIL: BRAE_NO_PREF did not suppress the reference"; rc=1; }
grep -q "BRAE_NO_PREF: the pressure reference is SUPPRESSED" "$W/noref/log.brae" \
    || { echo "FAIL: the control did not announce itself"; rc=1; }
if [ $ctlDiverged = 1 ]; then
    if grep -q "diverged" "$W/noref/log.brae"; then
        echo "  CONTROL (no reference):      DIVERGED -- $(grep -o 'solution diverged.*' "$W/noref/log.brae" | head -1)"
        ctlU=inf; ctlP=inf
    else
        echo "FAIL: the control run stopped for a reason other than divergence"; tail -5 "$W/noref/log.brae"; exit 1
    fi
else
    read -r _ ctlU ctlP < <(compare "$W/noref/$END" control) || exit 1
    echo "  CONTROL (no reference):      U $ctlU   p $ctlP"
fi

python3 - "$gotU" "$gotP" "$ctlU" "$ctlP" <<'PYEOF' || rc=1
import sys
gotU, gotP, ctlU, ctlP = (float(x) for x in sys.argv[1:5])
ok = True
# THE BOUNDS ARE THE MEASUREMENT, 6.9e-04 and 2.8e-03 at ten steps, rounded up by a little. They are
# this DRIVER's agreement with OpenFOAM on pitzDaily -- the transient driver reads 0.07% on U at steady
# on the shipped case too -- not the reference's: what this gate asserts about the reference is that
# WITHOUT it the run does not survive two steps. A tighter number here is a different unit's work.
for name, got, ctl, bound in (('U', gotU, ctlU, 1e-3), ('p', gotP, ctlP, 5e-3)):
    if got > bound:
        print('  FAIL: %s is %.3e from OpenFOAM, above the bound %.1e' % (name, got, bound)); ok = False
    else:
        print('  ok:   %s agrees with OpenFOAM (%.3e <= %.1e)' % (name, got, bound))
    if ctl == float('inf'):
        print('  ok:   without the reference the run DIVERGES, which no tolerance can excuse')
        continue
    if ctl < 10 * max(got, 1e-12):
        print('  FAIL: the control is %.3e -- it does not move %s far enough to discriminate' % (ctl, name)); ok = False
    else:
        print('  ok:   without the reference %s is %.3e, %.0fx further' % (name, ctl, ctl / max(got, 1e-30)))
raise SystemExit(0 if ok else 1)
PYEOF

echo "pimple_pressure_reference_vs_openfoam: rc $rc"
exit $rc
