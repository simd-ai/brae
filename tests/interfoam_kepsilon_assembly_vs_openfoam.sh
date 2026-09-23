#!/usr/bin/env bash
# brae's kEpsilon closure ASSEMBLED SYSTEM against real OpenFOAM's own, at the FIRST closure call, on
# RAS/damBreak under `Gauss limitedLinear 1` for div(rhoPhi,(k|epsilon)) -- host arm and DEVICE arm.
# The kEpsilon twin of tests/interfoam_sst_assembly_vs_openfoam.sh, sharing its comparator
# (tests/assembly_compare.py) and its reasoning; that gate's header has the long version.
#
# WHY AT THE ASSEMBLY. The same reason as the SST gate: a ten-step field comparison on an interFoam
# RAS case cannot witness the closure's convection scheme, because the last bit of k and epsilon is
# amplified out of all proportion (MEASURED on RAS/waterChannel's SST twin: one ulp of the initial
# omega field is worth 4.3e-02 after ten steps). The matrix is where the scheme IS, and it is compared
# at the one call both sides reach with identical inputs.
#
# THE ORACLE is tools/dumpKEpsilon -- OpenFOAM's own kEpsilon with writes added, `RASModel
# kEpsilonDump` -- which writes `stage_epsD`/`epsSrc`/`epsDUpper`/`epsDLower` and the k four at the
# time index BRAE_DUMP_STAGE_ITER, from the fvScalarMatrix OpenFOAM is about to solve (after relax(),
# constrain() and boundaryManipulate()). It had to be REGISTERED for interFoam's lineage first:
# RAS/damBreak builds PhaseIncompressibleTurbulenceModel<transportModel>, a third selection table made
# by VoFphaseTurbulentTransportModels.H, and a library registering only the compressible one is
# invisible there -- the run dies with "Unknown RASModel type kEpsilonDump"
# (tools/dumpKEpsilon/kEpsilonDumpVoFModels.C).
#
# brae's two arms write the same eight columns at one latched call: the DEVICE closure through
# solveScalarEqn's dumpPrefix (kEpsilon.cu:1240, which has always done it) and the HOST through
# interFoam's driver (inter_turbulence_cpp.cu), which is new -- the capture existed
# (KEResiduals::captureStages) and nothing wrote it for this solver.
#
# IT RUNS FROM A SPUN-UP FIELD, and that is not a convenience -- see stage(). From the tutorial's own
# start the two schemes assemble the IDENTICAL system (1.3e-14 apart), because k and epsilon are
# uniform on every row the matrix keeps; after 50 steps they are 2.0e-01 apart on the off-diagonals.
# The first version of this gate ran from the tutorial's start, and its controls failed it.
#
# A, B and C must match (bound 1e-10 in the shared comparator); D and E must differ (floor 1e-3). C is
# what makes D and E name the LIMITER rather than a broken upwind path.
#
# NO ORACLE CACHE here, unlike the SST gate: OpenFOAM's whole contribution is 51 steps of a 2,268-cell
# case, well under a second, so the cache would add a way to be wrong for nothing.
#
# FAIL-PROOFS, the shared assembler broken once each in a scratch build linked ahead of
# libbrae_core.a (the tree untouched), this gate red in arm B alone both times:
#
#   the device ignores the limiter (`if (false)` on the limited branch)
#       -> B: upper 2.022e-01, lower 2.594e-01, D 4.613e-02, k upper 1.583e-01; A and C green
#   the case's coefficient doubled (twoByk built from 2*k)
#       -> B: upper 9.165e-02, lower 1.176e-01, D 1.629e-02, k D 1.067e-02
#
# The second one moves the OFF-DIAGONALS here, where on the SST gate's case it moved only D and Src --
# there every face whose weight changed belonged to a row setValues eliminates. Two cases, two shapes
# of the same defect; between them the shared comparator's columns are all exercised.
#
# WHAT THIS LIFTS. The device kEpsilon closure refused `Gauss limitedLinear` by name -- "ungated",
# which it was: the kOmegaSST half was lifted by its own assembly gate and this half had no oracle
# until tools/dumpKEpsilon reached interFoam's lineage. Both closures share the assembler
# (turbulence_transport.cu), so this gate and the SST one hold the same code from two directions.
#
# NOT CLAIMED: the second and later closure calls (from call two the arms hold different fields, so
# their systems differ by what the fields differ by); `limitedLinear` with a coefficient other than 1;
# any other case.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BUILD:-$ROOT/build}/brae_interFoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/RAS/damBreak/damBreak"
END=0.001
# steps of deltaT 0.001 before the comparison -- see stage() for why a spun-up field is the whole
# point. 50 is where epsilon's interior stops being uniform; the gate asserts that it did.
SPIN=${SPIN:-50}

[ -x "$BRAE" ]     || { echo "SKIP: $BRAE not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: RAS/damBreak tutorial not found at $SRC"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

. "$(dirname "$0")/require_fresh_binary.sh"
requireFresh "$BRAE" || exit 1

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v blockMesh > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }
command -v setFields > /dev/null 2>&1 || { echo "SKIP: setFields not on PATH"; exit 77; }
command -v interFoam > /dev/null 2>&1 || { echo "SKIP: interFoam not on PATH"; exit 77; }

# every arm this gate asserts is a device arm; with no GPU there is nothing left to run
command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi > /dev/null 2>&1 \
    || { echo "SKIP: no GPU -- this gate's subject is the device closure"; exit 77; }

DUMPLIB="${FOAM_USER_LIBBIN:-}/libdumpKEpsilon.so"
[ -f "$DUMPLIB" ] || { echo "SKIP: the oracle library is not built -- (cd $ROOT/tools/dumpKEpsilon && wmake libso)"; exit 77; }

W=${KEEP_W:-$(mktemp -d)}
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"

rc=0
pass() { echo "  ok:   $1"; }
fail() { echo "  FAIL: $1"; rc=1; }

# stage <name> <scheme text>
stage()
{
    local name="$1" scheme="$2"
    local C="$W/$name"
    rm -rf "$C"
    cp -r "$SRC" "$C" || return 1
    rm -rf "$C"/[1-9]* "$C"/0.[0-9]* "$C"/0 "$C"/processor* "$C"/log.*
    cp -r "$C/0.orig" "$C/0"
    grep -q "RASModel  *kEpsilon;" "$C/constant/turbulenceProperties" \
        || { echo "FAIL: the tutorial no longer names kEpsilon"; return 1; }

    SCHEME="$scheme" python3 - "$C" <<'PYEOF' || { echo "FAIL: staging $name"; return 1; }
import os, re, sys
d, scheme = sys.argv[1], os.environ['SCHEME']
p = os.path.join(d, 'system/fvSchemes')
s = open(p).read()
n = 0
for eqn in ('k', 'epsilon'):
    s, k = re.subn(r'^(\s*div\(rhoPhi,%s\)\s+).*$' % eqn,
                   lambda m: m.group(1) + 'Gauss ' + scheme + ';', s, flags=re.M)
    assert k == 1, 'the tutorial no longer carries one div(rhoPhi,%s) entry' % eqn
    n += k
open(p, 'w').write(s)

c = os.path.join(d, 'system/controlDict')
s = open(c).read()
s = re.sub(r'\nfunctions\s*\{.*\n\}\s*\n', '\n', s, flags=re.S)
for key, val in [('adjustTimeStep', 'no'), ('deltaT', '0.001'), ('endTime', '0.001'),
                 ('writeControl', 'timeStep'), ('writeInterval', '1'), ('writeFormat', 'ascii'),
                 ('writePrecision', '15')]:
    s, k = re.subn(r'^%s\s.*' % key, '%s %s;' % (key.ljust(15), val), s, flags=re.M)
    assert k == 1, key
s = re.sub(r'^writeCompression\s.*', 'writeCompression off;', s, flags=re.M)
open(c, 'w').write(s)
PYEOF
    grep -q "Gauss $scheme;" "$C/system/fvSchemes" \
        || { echo "FAIL: \`Gauss $scheme\` did not reach div(rhoPhi,(k|epsilon)) [$name]"; return 1; }

    ( cd "$C" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh [$name]"; tail -20 "$C/log.blockMesh"; return 1; }
    ( cd "$C" && setFields > log.setFields 2>&1 ) || { echo "FAIL: setFields [$name]"; tail -20 "$C/log.setFields"; return 1; }

    # SPIN UP FIRST, and the gate is worth nothing without it. From the tutorial's own start, k and
    # epsilon are UNIFORM everywhere the matrix keeps: the only cells carrying anything else are the
    # ones epsilonWallFunction pins, whose rows setValues eliminates. With a uniform field NVDTVD's
    # gradf is zero on every surviving face, the limiter lands on the same weight upwind would give,
    # and `Gauss limitedLinear 1` assembles the IDENTICAL system to `Gauss upwind` -- MEASURED, the two
    # oracles 1.3e-14 apart, and the controls below caught it and failed the gate, as they should.
    # After $SPIN steps the field is developed and the two schemes are 2.0e-01 apart on the
    # off-diagonals. The developed state becomes the new `0` so that both OpenFOAM's dump run and
    # brae's runs start at time zero from the same fields, and the oracle's `timeIndex == 1` latch is
    # the first closure call for all of them.
    SPIN="$SPIN" python3 - "$C" <<'PYEOF' || { echo "FAIL: spin-up controlDict [$name]"; return 1; }
import os, re, sys
c = os.path.join(sys.argv[1], 'system/controlDict')
s = open(c).read()
n = int(os.environ['SPIN'])
for key, val in [('endTime', '%.10g' % (n*0.001)), ('writeInterval', str(n))]:
    s, k = re.subn(r'^%s\s.*' % key, '%s %s;' % (key.ljust(15), val), s, flags=re.M)
    assert k == 1, key
open(c, 'w').write(s)
PYEOF
    ( cd "$C" && interFoam > log.spinup 2>&1 ) || { echo "FAIL: the spin-up run [$name]"; tail -30 "$C/log.spinup"; return 1; }
    local dev
    dev=$(python3 -c "print('%.10g' % ($SPIN*0.001))")
    [ -d "$C/$dev" ] || { echo "FAIL: the spin-up wrote no $dev directory [$name]"; ls "$C"; return 1; }
    rm -rf "$C/0"
    mv "$C/$dev" "$C/0"
    rm -rf "$C/0/uniform"   # the time index/value, which would make this a restart rather than a start
    # ...and the field must actually have developed, or the spin-up is doing nothing and the controls
    # below are the only thing standing between this gate and a vacuous pass
    python3 - "$C/0/epsilon" <<'PYEOF' || { echo "FAIL: epsilon is still uniform after the spin-up [$name]"; return 1; }
import sys
s = open(sys.argv[1]).read()
i = s.index('internalField')
assert 'nonuniform' in s[i:s.index('\n', i)], 'internalField is still uniform'
PYEOF

    python3 - "$C" <<'PYEOF' || { echo "FAIL: the one-step controlDict [$name]"; return 1; }
import re, sys
c = sys.argv[1] + '/system/controlDict'
s = open(c).read()
for key, val in [('endTime', '0.001'), ('writeInterval', '1')]:
    s, k = re.subn(r'^%s\s.*' % key, '%s %s;' % (key.ljust(15), val), s, flags=re.M)
    assert k == 1, key
open(c, 'w').write(s)
PYEOF
    sed -i 's/RASModel  *kEpsilon;/RASModel        kEpsilonDump;/' "$C/constant/turbulenceProperties"
    printf '\nlibs            ("libdumpKEpsilon.so");\n' >> "$C/system/controlDict"
    ( cd "$C" && BRAE_DUMP_STAGE_ITER=1 interFoam > log.interFoam 2>&1 ) \
        || { echo "FAIL: interFoam [$name]"; tail -30 "$C/log.interFoam"; return 1; }
    sed -i 's/RASModel  *kEpsilonDump;/RASModel        kEpsilon;/' "$C/constant/turbulenceProperties"
    sed -i '/libdumpKEpsilon/d' "$C/system/controlDict"
    echo "OpenFOAM ran $SPIN steps, then one more with the dump model   [$name]"

    local f
    for f in stage_epsD stage_epsSrc stage_epsDUpper stage_epsDLower \
             stage_kD  stage_kSrc  stage_kDUpper  stage_kDLower; do
        [ -f "$C/$END/$f" ] || { echo "FAIL: the oracle wrote no $f [$name]"; return 1; }
    done
    grep -q "RASModel  *kEpsilon;" "$C/constant/turbulenceProperties" \
        || { echo "FAIL: the case was left naming the dump model [$name]"; return 1; }
}

stage limitedLinear "limitedLinear 1" || { echo "interfoam_kepsilon_assembly_vs_openfoam: staging failed"; exit 1; }
stage upwind        "upwind"          || { echo "interfoam_kepsilon_assembly_vs_openfoam: staging failed"; exit 1; }

# runBrae <case> <arm> -- both arms write the SAME eight column names, so each gets its own directory
runBrae()
{
    local C="$W/$1" arm="$2"
    local d="$C/dump.$arm"
    rm -rf "$d"
    mkdir -p "$d"
    local flag=""
    [ "$arm" = device ] && flag="-device"
    ( cd "$C" && BRAE_STAGE_DUMP_DIR="$d" BRAE_STAGE_DUMP_ITER=1 "$BRAE" -case "$C" $flag > "log.brae.$arm" 2>&1 ) \
        || { echo "FAIL: brae $arm did not run [$1]"; tail -20 "$C/log.brae.$arm"; return 1; }
    if [ "$arm" = device ]; then
        grep -q "\[device\]" "$C/log.brae.$arm" \
            || { echo "FAIL: brae's run does not say it was on the device [$1]"; return 1; }
        grep -qi "BRAE_KE_DIAG_LIMITED" "$C/log.brae.$arm" \
            && { echo "FAIL: the device arm ran under the diagnostic bypass, not on its own [$1]"; return 1; }
    fi
    return 0
}

compare()
{
    local ofc="$W/$1" brc="$W/$2" arm="$3" expect="$4" label="$5"
    python3 "$(dirname "$0")/assembly_compare.py" \
        "$ofc/$END" "$brc/dump.$arm" "$ofc/constant/polyMesh" ke "$expect" "$label"
}

echo "== brae's kEpsilon assembled system vs OpenFOAM's own, first closure call =="

runBrae limitedLinear host   || rc=1
runBrae limitedLinear device || rc=1
runBrae upwind       host    || rc=1
runBrae upwind       device  || rc=1

if [ $rc = 0 ]; then
    compare limitedLinear limitedLinear host   match  "A host  limitedLinear vs OpenFOAM's own" \
        && pass "the HOST's assembled system is OpenFOAM's under \`Gauss limitedLinear 1\`" \
        || fail "the HOST's assembled system is OpenFOAM's under \`Gauss limitedLinear 1\`"

    compare limitedLinear limitedLinear device match  "B DEVICE limitedLinear vs OpenFOAM's own" \
        && pass "...and the DEVICE's is, which is what the refusal was waiting for" \
        || fail "...and the DEVICE's is, which is what the refusal was waiting for"

    compare upwind upwind device match  "C DEVICE upwind vs OpenFOAM's own" \
        && pass "...and the DEVICE's upwind is OpenFOAM's too, so D and E below name the LIMITER" \
        || fail "...and the DEVICE's upwind is OpenFOAM's too, so D and E below name the LIMITER"

    compare upwind limitedLinear device differ "D DEVICE limitedLinear vs OpenFOAM's UPWIND" \
        && pass "CONTROL: the comparison can witness the scheme -- limitedLinear is not upwind" \
        || fail "CONTROL: the comparison can witness the scheme -- limitedLinear is not upwind"

    compare limitedLinear upwind device differ "E DEVICE upwind vs OpenFOAM's limitedLinear" \
        && pass "CONTROL: ...and the other way round" \
        || fail "CONTROL: ...and the other way round"
fi

echo "interfoam_kepsilon_assembly_vs_openfoam: rc $rc"
exit $rc
