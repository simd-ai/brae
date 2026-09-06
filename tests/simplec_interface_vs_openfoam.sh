#!/usr/bin/env bash
# SIMPLEC (`consistent true`) across a COUPLED interface, on OpenFOAM's own basic/simpleFoam/implicitAMI
# tutorial, against real OpenFOAM.
#
# WHAT WAS WRONG. SIMPLEC replaces rAU with rAtU = 1/(1/rAU - H1) and adds a matching flux term
# (simpleFoam/pEqn.H):
#     rAtU     = 1.0/(1.0/rAU - UEqn.H1());
#     phiHbyA += fvc::interpolate(rAtU() - rAU)*fvc::snGrad(p)*mesh.magSf();
#     HbyA    -= (rAU - rAtU())*fvc::grad(p);
# Both of the first two reach a coupled patch in OpenFOAM and neither did in brae:
#
#   1. fvMatrix::H1() sums boundaryCoeffs over EVERY coupled patch. brae's row sum took the cyclic
#      interface and stopped there, so on a mesh coupled only by a cyclicAMI it was the internal faces
#      alone and rAtU came out too large at every interface cell.
#   2. The flux term is a surfaceScalarField whose coupled boundary values are evaluated like any other
#      patch's. brae built it with deviceMatrixFluxInternal (internal faces) and deviceMatrixFluxBoundary
#      (NON-coupled patches) -- a cyclic or cyclicAMI face is in neither list, so the interface flux never
#      received it at all.
#
# They are two halves of one consistency relation and neither is right alone: with only the first, the
# converged U on this case got WORSE (8.253e-02 -> 9.282e-02); with both, 1.692e-02.
#
# WHY IT SURVIVED. No case in validation/ combined SIMPLEC with a coupled interface -- every cyclic and
# AMI case there runs plain SIMPLE, and every SIMPLEC case is uncoupled. The whole path was unreachable.
#
# THE SHARP TEST is check 3, and it needs no converged comparison at all: seed brae with OpenFOAM's own
# converged U, p and phi and assemble ONE iteration. At OpenFOAM's steady state the true divergence is
# zero, so brae's continuity error there is a direct read of how inconsistent its pressure equation is --
# 0.128 before, 1.7e-03 after, while the momentum residual was 2.8e-07 the whole time and said so.
#
# Seeding phi matters and is not a detail: without it brae recomputes the flux from U and the comparison
# measures that instead. The first run of this investigation reported a bogus interface-coefficient
# mismatch for exactly that reason -- with phi seeded, brae's interface off-diagonal matches OpenFOAM's
# boundaryCoeffs to every digit.
set -u
SRC="${1:?implicitAMI case dir}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
OFBIN=/usr/lib/openfoam/openfoam2412/platforms/linuxARM64GccDPInt32Opt
[ -x "$BRAE" ]                 || { echo "SKIP: no brae binary at $BRAE"; exit 77; }
[ -x "$OFBIN/bin/simpleFoam" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
SRC="$(cd "$SRC" && pwd)"
grep -q 'cyclicAMI' "$SRC/constant/polyMesh/boundary" \
    || { echo "FAIL: $SRC has no coupled interface -- this gate would test nothing"; exit 1; }
grep -qE 'consistent\s+(true|yes|on)' "$SRC/system/fvSolution" \
    || { echo "FAIL: $SRC does not ask for SIMPLEC, so nothing below exercises it"; exit 1; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
set +u
# shellcheck disable=SC1091
source /usr/lib/openfoam/openfoam2412/etc/bashrc > /dev/null 2>&1 || true
set -u

# <dir> <on|off>. div(phi,U) is forced to upwind in every run: the interface's div-scheme weight is a
# SEPARATE concern with its own gate (coupledinterfacescheme_vs_openfoam), and leaving the case's
# `Gauss linear` in would mix the two.
mkcase()
{
    cp -r "$SRC" "$1"
    rm -rf "$1"/[1-9]* "$1"/0.[0-9]* "$1"/log.* "$1"/postProcessing
    python3 - "$1" "$2" <<'PY'
import re, sys
d, simplec = sys.argv[1], sys.argv[2]
p = d + '/system/fvSchemes'; s = open(p).read()
s = re.sub(r'div\(phi,U\)\s+[^;]+;', 'div(phi,U)      bounded Gauss upwind;', s)
open(p, 'w').write(s)
p = d + '/system/fvSolution'; s = open(p).read()
s = re.sub(r'consistent\s+\w+;', 'consistent      %s;' % ('true' if simplec == 'on' else 'false'), s)
open(p, 'w').write(s)
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'^endTime .*',        'endTime         20000;', s, flags=re.M)
s = re.sub(r'^writeInterval .*',  'writeInterval   20000;', s, flags=re.M)
s = re.sub(r'^writeFormat .*',    'writeFormat     ascii;', s, flags=re.M)
s = re.sub(r'^writePrecision .*', 'writePrecision  15;', s, flags=re.M)
open(c, 'w').write(s)
PY
}

for s in on off; do
    mkcase "$W/of_$s" "$s"
    ( cd "$W/of_$s" && timeout 1800 simpleFoam > log.of 2>&1 ) \
        || { echo "FAIL: OpenFOAM did not run with SIMPLEC $s"; tail -5 "$W/of_$s/log.of"; exit 1; }
    mkcase "$W/brae_$s" "$s"
    ( cd "$W/brae_$s" && "$BRAE" . > run.log 2>&1 ) \
        || { echo "FAIL: brae refused or crashed with SIMPLEC $s"; tail -15 "$W/brae_$s/run.log"; exit 1; }
done

# CHECK 3's case: brae seeded with OpenFOAM's converged state, ONE iteration, SIMPLEC on.
mkcase "$W/seed" "on"
cp "$W"/of_on/20000/U "$W"/of_on/20000/p "$W"/of_on/20000/phi "$W/seed/0/"
python3 - "$W/seed" <<'PY'
import re, sys
c = sys.argv[1] + '/system/controlDict'; s = open(c).read()
s = re.sub(r'^endTime .*',       'endTime         1;', s, flags=re.M)
s = re.sub(r'^writeInterval .*', 'writeInterval   1;', s, flags=re.M)
open(c, 'w').write(s)
PY
( cd "$W/seed" && "$BRAE" . > run.log 2>&1 ) \
    || { echo "FAIL: brae crashed on the seeded state"; tail -15 "$W/seed/run.log"; exit 1; }

rc=0
python3 - "$W" <<'PY'
import os, re, sys
import numpy as np
W = sys.argv[1]

def lastTime(d):
    ts = [x for x in os.listdir(d) if re.fullmatch(r'[0-9]+(\.[0-9]+)?', x) and x != '0']
    return None if not ts else max(ts, key=float)

def read(d, f):
    b = open(os.path.join(d, f), 'rb').read()
    m = re.search(rb'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n(\d+)\s*\n\(', b)
    if not m:
        m2 = re.search(rb'internalField\s+uniform\s+([^;]+);', b)
        return None if not m2 else np.array([[float(x) for x in re.findall(rb'[-+0-9.eE]+', m2.group(1))]])
    typ = m.group(1).decode(); n = int(m.group(2)); start = m.end()
    nc = 3 if typ == 'vector' else 1
    txt = b[start:].decode('latin-1')
    vals = re.findall(r'[-+0-9.eE]+', txt.split(')\n;')[0] if ')\n;' in txt else txt)
    return np.array([float(x) for x in vals[:n*nc]]).reshape(n, nc)

def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))

dirs = {}
for k in ('of_on', 'of_off', 'brae_on', 'brae_off'):
    t = lastTime(os.path.join(W, k))
    if t is None:
        print("  FAIL: %s wrote no result" % k); sys.exit(1)
    dirs[k] = os.path.join(W, k, t)

rc = 0
# 1. SIMPLEC ON -- the path that was broken. The bound is loose because this is a coarse 24-cell case run
#    to steady state by two different codes, but the pre-fix run sat at U 8.253e-02 and fails it 2.5x over.
print("  1. brae vs OpenFOAM, SIMPLEC on")
for f, bnd in (('U', 3.5e-02), ('p', 8e-02)):
    e = rel(read(dirs['brae_on'], f), read(dirs['of_on'], f))
    ok = e < bnd
    print("     %-8s L2 rel %.3e   %s" % (f, e, "ok" if ok else "FAIL (> %.0e)" % bnd))
    if not ok: rc = 1
# 2. SIMPLEC OFF -- untouched by this work and already agreed. Keeps a regression in the shared SIMPLE
#    path from being read as a SIMPLEC result.
print("  2. brae vs OpenFOAM, SIMPLEC off (the path that was never broken)")
for f, bnd in (('U', 5e-03), ('p', 1e-02)):
    e = rel(read(dirs['brae_off'], f), read(dirs['of_off'], f))
    ok = e < bnd
    print("     %-8s L2 rel %.3e   %s" % (f, e, "ok" if ok else "FAIL (> %.0e)" % bnd))
    if not ok: rc = 1
# 2b. ...and OpenFOAM's own two runs must DIFFER, or checks 1 and 2 are the same measurement twice.
e = rel(read(dirs['of_on'], 'U'), read(dirs['of_off'], 'U'))
ok = e > 1e-04
print("  2b. OpenFOAM SIMPLEC on vs off: U L2 rel %.3e   %s"
      % (e, "ok" if ok else "FAIL: identical, so SIMPLEC is not exercised"))
if not ok: rc = 1
sys.exit(rc)
PY
rc=$?

# 3. THE SHARP ONE. At OpenFOAM's converged state the divergence is zero, so brae's continuity error is a
#    direct measure of its own pressure-equation inconsistency -- no bound borrowed from a field diff.
CONT=$(grep 'continuity errors' "$W/seed/run.log" | head -1 | sed -n 's/.*sum local = \([0-9.e+-]*\).*/\1/p')
PRES=$(grep -E 'Solving for p' "$W/seed/run.log" | head -1 | sed -n 's/.*Initial residual = \([0-9.e+-]*\).*/\1/p')
MOM=$(grep -E 'Solving for Ux' "$W/seed/run.log" | head -1 | sed -n 's/.*Initial residual = \([0-9.e+-]*\).*/\1/p')
echo "  3. seeded at OpenFOAM's converged state: momentum $MOM  pressure $PRES  continuity $CONT"
python3 -c "
import sys
mom, pres, cont = float('$MOM'), float('$PRES'), float('$CONT')
ok = True
# The momentum equation was never the problem and must stay that way -- if it drifts, checks 1-2 would
# blame SIMPLEC for something that is not SIMPLEC.
if not mom < 1e-05:  print('     FAIL: momentum residual %.3e -- the UEqn no longer matches OpenFOAM' % mom); ok = False
else:                print('     momentum   %.3e  ok  (brae is at OpenFOAM steady state)' % mom)
if not pres < 5e-03: print('     FAIL: pressure residual %.3e (>5e-03; it was 3.4e-02 before this work)' % pres); ok = False
else:                print('     pressure   %.3e  ok' % pres)
if not cont < 1e-02: print('     FAIL: continuity error %.3e (>1e-02; it was 1.3e-01 before this work)' % cont); ok = False
else:                print('     continuity %.3e  ok' % cont)
sys.exit(0 if ok else 1)
" || rc=1

[ $rc -eq 0 ] && echo "  ok:   brae's SIMPLEC reaches the coupled interface"
exit $rc
