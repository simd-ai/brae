#!/usr/bin/env bash
# rhoSimpleFoam's pcEqn.H -- the SIMPLEC pressure equation -- against REAL OpenFOAM, BOTH branches.
#
# OpenFOAM is run TWICE on the same fixture, once `transonic no` and once `transonic yes`, and the binary
# is told which branch it is looking at. That is the only way to gate both: `simple.transonic()` selects
# between two pressure equations that differ in the matrix (`fvm::div(phid, p)` or not), in what happens
# to phiHbyA (a psi*p subtraction or adjustPhi), and in whether pEqn.relax() is called at all.
#
# BOTH RUNS FORCE `consistent yes`, which is what sends OpenFOAM's driver to pcEqn.H rather than pEqn.H.
# sbMatched ships that already; it is set explicitly so the gate does not depend on the fixture keeping it.
#
# THE ORACLE is tools/dumpPEqn's pcEqn.H. Its transonic branch already dumped the assembled system; the
# SUBSONIC branch had no dump at all, so the SIMPLEC pressure equation could only ever be compared in its
# transonic form. That dump was added, along with stage_phiHbyA0/phiHbyAc, stage_HbyAc, stage_rhorAtU and
# stage_psi, so every SIMPLEC intermediate has an oracle rather than only the final matrix.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_rho_pceqn_cpp"
SRC="$ROOT/validation/sbMatched"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u

DUMP="$(command -v dumpPEqn || true)"
[ -z "$DUMP" ] && [ -n "${FOAM_USER_APPBIN:-}" ] && [ -x "$FOAM_USER_APPBIN/dumpPEqn" ] \
    && DUMP="$FOAM_USER_APPBIN/dumpPEqn"
[ -n "$DUMP" ] || { echo "SKIP: dumpPEqn not built -- (cd tools/dumpPEqn && wmake)"; exit 77; }

rc=0
for MODE in subsonic transonic; do
    C="$W/$MODE"
    rm -rf "$C"; cp -r "$SRC" "$C" || exit 1
    rm -rf "$C"/[1-9]* "$C"/0 "$C"/processor* "$C"/log.*
    [ -d "$C/constant/polyMesh" ] || { echo "SKIP: fixture ships no mesh"; exit 77; }
    cp -r "$C/0.orig" "$C/0"

    # The inlet is neutralised for the same reason as in the momentum gate: sbMatched's
    # flowRateInletVelocity disagrees with OpenFOAM by ~2.4e-01 (see PORT.md), and leaving it in would put
    # a boundary-condition error inside a number that is supposed to be about pEqn.H.
    python3 - "$C/0/U" <<'PYEOF'
import re, sys
p = sys.argv[1]
s = open(p).read()
s = re.sub(r'(inlet\s*\{)[^}]*\}',
           r'\1\n        type            fixedValue;\n        value           uniform (1 2 3);\n    }',
           s, count=1)
open(p, 'w').write(s)
PYEOF

    TRANS=no;  [ "$MODE" = transonic ] && TRANS=yes
    MODE="$MODE" TRANS="$TRANS" python3 - "$C" <<'PYEOF'
import os, re, sys
d = sys.argv[1]
c = os.path.join(d, 'system/controlDict')
s = open(c).read()
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
s = re.sub(r'^writeFormat .*',    'writeFormat     ascii;',    s, flags=re.M)
s = re.sub(r'^writePrecision .*', 'writePrecision  15;',       s, flags=re.M)
s = re.sub(r'^endTime .*',        'endTime         1;',        s, flags=re.M)
s = re.sub(r'^writeInterval .*',  'writeInterval   1;',        s, flags=re.M)
s = re.sub(r'^writeControl .*',   'writeControl    timeStep;', s, flags=re.M)
open(c, 'w').write(s)

# `consistent no` forces the driver to pEqn.H rather than pcEqn.H; `transonic` selects the branch.
f = os.path.join(d, 'system/fvSolution')
s = open(f).read()
s = re.sub(r'^(\s*)consistent\s+\w+;', r'\1consistent      yes;', s, flags=re.M)
s = re.sub(r'^(\s*)transonic\s+\w+;',  r'\1transonic       %s;' % os.environ['TRANS'], s, flags=re.M)
assert 'consistent      yes;' in s, 'could not force consistent yes'

# THE PRECONDITIONER HAS TO CHANGE WITH THE BRANCH, and this is a property of the two equations rather
# than a convenience. The transonic branch adds fvm::div(phid, p), so its pressure matrix is ASYMMETRIC
# and takes DILU (which is what the fixture ships). The subsonic branch is a pure laplacian, so the matrix
# is SYMMETRIC and OpenFOAM refuses DILU on it outright -- "Unknown symmetric matrix preconditioner type
# DILU". Only the p solver is touched; U and e stay asymmetric and keep DILU. None of this reaches the
# assembled matrix, which is what the gate compares -- it is dumped before the solve.
if os.environ['TRANS'] == 'no':
    s = re.sub(r'(^\s*p\s*\{[^}]*preconditioner\s+)DILU', r'\1DIC', s, flags=re.M)
    assert 'preconditioner DIC' in s, 'could not switch the p preconditioner to DIC'
assert 'transonic       %s;' % os.environ['TRANS'] in s, 'could not set transonic'
open(f, 'w').write(s)
PYEOF

    # TWO RUNS, AND THE FIRST ONE IS WHAT MAKES THIS GATE MEAN ANYTHING.
    #
    # sbMatched starts from `p uniform 110000` with zeroGradient walls and a fixedValue outlet at the same
    # value -- so p is uniform everywhere and fvc::grad(p) is ANALYTICALLY ZERO. At iteration 1 the SIMPLEC
    # corrections `HbyA -= (rAU - rAtU)*fvc::grad(p)` and the snGrad flux term are therefore no-ops, and
    # comparing them measures nothing but round-off: OpenFOAM's own correction moves HbyA by 9.0e-10 there,
    # which is (1/V)*sum(Sf) failing to cancel exactly at p ~ 1.1e5. Since those two corrections ARE
    # SIMPLEC, a gate taken at iteration 1 would report machine precision while testing none of it.
    #
    # So OpenFOAM is run to iteration DEV first, and the comparison is taken from a RESTART at DEV -- one
    # iteration, dumped as its iteration 1 -- where p varies and the corrections are real terms.
    DEV=${DEV:-20}
    DEV="$DEV" python3 - "$C" <<'PYDEV'
import os, re, sys
c = os.path.join(sys.argv[1], 'system/controlDict')
s = open(c).read()
s = re.sub(r'^endTime .*',       'endTime         %s;' % os.environ['DEV'], s, flags=re.M)
s = re.sub(r'^writeInterval .*', 'writeInterval   %s;' % os.environ['DEV'], s, flags=re.M)
open(c, 'w').write(s)
PYDEV
    ( cd "$C" && BRAE_DUMP_STAGE_ITER=999999 "$DUMP" > develop.log 2>&1 ) \
        || { echo "FAIL[$MODE]: the developing run failed"; tail -20 "$C/develop.log"; exit 1; }
    [ -d "$C/$DEV" ] || { echo "FAIL[$MODE]: no $DEV/ written"; tail -20 "$C/develop.log"; exit 1; }
    python3 - "$C/$DEV/p" <<'PYCHK'
import re, sys
s = open(sys.argv[1]).read()
assert 'nonuniform' in s, 'p is still uniform after developing -- the SIMPLEC corrections stay no-ops'
PYCHK

    NEXT=$((DEV + 1))
    DEV="$DEV" python3 - "$C" <<'PYNEXT'
import os, re, sys
c = os.path.join(sys.argv[1], 'system/controlDict')
s = open(c).read()
s = re.sub(r'^startFrom .*',     'startFrom       startTime;',              s, flags=re.M)
s = re.sub(r'^startTime .*',     'startTime       %s;' % os.environ['DEV'], s, flags=re.M)
s = re.sub(r'^endTime .*',       'endTime         %d;' % (int(os.environ['DEV']) + 1), s, flags=re.M)
s = re.sub(r'^writeInterval .*', 'writeInterval   1;',                      s, flags=re.M)
open(c, 'w').write(s)
PYNEXT
    # BRAE_DUMP_STAGE_ITER is matched against runTime.timeIndex(), which on a RESTART continues from the
    # start time -- it is not 1 just because this run performs one iteration.
    ( cd "$C" && BRAE_DUMP_STAGE_ITER="$NEXT" "$DUMP" > dump.log 2>&1 ) \
        || { echo "FAIL[$MODE]: dumpPEqn did not run"; tail -20 "$C/dump.log"; exit 1; }

    for fld in stage_rAU stage_rAtU stage_rhorAtU stage_HbyA stage_HbyAc stage_phiHbyA0 stage_phiHbyAc \
               stage_muEff stage_Uass stage_Upred stage_psi stage_rhoP stage_pD stage_pSrc; do
        [ -f "$C/$NEXT/$fld" ] \
            || { echo "FAIL[$MODE]: dumpPEqn wrote no $NEXT/$fld -- did the $MODE branch actually run?"; \
                 tail -20 "$C/dump.log"; exit 1; }
    done
    if [ "$MODE" = transonic ]; then
        [ -f "$C/$NEXT/stage_phid" ] || { echo "FAIL: no stage_phid; the transonic branch did not run"; exit 1; }
    fi

    T=0; [ "$MODE" = transonic ] && T=1
    "$BIN" "$C" "$DEV" "$NEXT" "$T" || rc=1
    echo
done
exit $rc
