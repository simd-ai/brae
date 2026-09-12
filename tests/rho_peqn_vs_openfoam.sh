#!/usr/bin/env bash
# rhoSimpleFoam's pEqn.H against REAL OpenFOAM -- BOTH branches.
#
# OpenFOAM is run TWICE on the same fixture, once `transonic no` and once `transonic yes`, and the binary
# is told which branch it is looking at. That is the only way to gate both: `simple.transonic()` selects
# between two pressure equations that differ in the matrix (`fvm::div(phid, p)` or not), in what happens
# to phiHbyA (a psi*p subtraction or adjustPhi), and in whether pEqn.relax() is called at all.
#
# BOTH RUNS FORCE `consistent no`. sbMatched ships `consistent yes`, which sends OpenFOAM's driver to
# pcEqn.H instead of pEqn.H -- a different file, and a different manifest component. Gating pEqn against a
# run that never executed it would be measuring nothing.
#
# THE ORACLE is tools/dumpPEqn. Its subsonic branch already carried a stage harness; the transonic branch
# had none and one was added (stage_phid, stage_phiHbyA, stage_pICt/pBCt/pDt/pSrct), so the branch that
# owns the convective pressure term could be compared at all.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_rho_peqn_cpp"
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
s = re.sub(r'^(\s*)consistent\s+\w+;', r'\1consistent      no;',  s, flags=re.M)
s = re.sub(r'^(\s*)transonic\s+\w+;',  r'\1transonic       %s;' % os.environ['TRANS'], s, flags=re.M)
assert 'consistent      no;' in s, 'could not force consistent no'

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

    ( cd "$C" && BRAE_DUMP_STAGE_ITER=1 "$DUMP" > dump.log 2>&1 ) \
        || { echo "FAIL[$MODE]: dumpPEqn did not run"; tail -20 "$C/dump.log"; exit 1; }

    SFX=""; [ "$MODE" = transonic ] && SFX="t"
    for fld in stage_rAU stage_rhorAUf stage_HbyA stage_phiHbyA0 stage_phiHbyA \
               stage_muEff stage_Uass stage_Upred stage_psi stage_rhoP "stage_pD$SFX" "stage_pSrc$SFX"; do
        [ -f "$C/1/$fld" ] \
            || { echo "FAIL[$MODE]: dumpPEqn wrote no 1/$fld -- did the $MODE branch actually run?"; \
                 tail -20 "$C/dump.log"; exit 1; }
    done
    if [ "$MODE" = transonic ]; then
        [ -f "$C/1/stage_phid" ] || { echo "FAIL: no stage_phid; the transonic branch did not run"; exit 1; }
    fi

    T=0; [ "$MODE" = transonic ] && T=1
    "$BIN" "$C" 0 1 "$T" || rc=1
    echo
done
exit $rc
