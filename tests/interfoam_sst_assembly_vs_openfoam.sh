#!/usr/bin/env bash
# brae's kOmegaSST closure ASSEMBLED SYSTEM against real OpenFOAM's own, at the FIRST closure call,
# on RAS/waterChannel under `Gauss limitedLinear 1` for div(phi,(k|omega)) -- host arm and DEVICE arm.
#
# WHY AT THE ASSEMBLY, AND NOT ON THE FIELDS. Every other interFoam gate compares fields at an end
# time. On this case that comparison cannot witness the closure's convection scheme, because the case
# AMPLIFIES round-off in the turbulence fields by about fourteen orders over ten steps. MEASURED on
# the HOST, perturbing only the initial `0/omega` and changing nothing else:
#
#   one ulp (4.3e-19) in ONE interior cell   ->  k 3.1e-07, omega 1.1e-07, nut 1.6e-07
#   one ulp in EVERY cell, sign at random    ->  k 4.5e-02, omega 4.3e-02, nut 1.7e-02
#                                                (and U 5.0e-05, alpha 2.3e-06)
#
# The device arm reads omega 1.78e-04 at t = 1 there -- two orders BELOW the whole-field ulp figure,
# and the SAME number it read when it was silently convecting with upwind. So no field bound on that
# profile can tell the scheme from the last bit: a bound loose enough to pass cannot fail for the
# substitution either. The host passes at 7.2e-12 only because it is bitwise OpenFOAM's (one ulp in
# one cell would have put it at 1.1e-07). This gate compares the object the scheme is IN.
#
# THE ORACLE is OpenFOAM's own kOmegaSST with writes added -- tools/dumpKOmegaSST, `RASModel
# kOmegaSSTDump` -- which writes `stage_sstOmD`/`OmSrc`/`OmDUpper`/`OmDLower` and the k four, at the
# time index BRAE_DUMP_STAGE_ITER, from the fvScalarMatrix OpenFOAM is about to solve: after relax(),
# fvOptions.constrain() and boundaryManipulate(), with D() carrying internalCoeffs and the source
# carrying boundaryCoeffs. brae's two arms write the same four at the same point, in the same
# convention (`BRAE_SST_DUMP_DIR`, `BRAE_SST_DUMP_ITER`), so the columns diff directly.
#
# WHAT IS COMPARED, and what each part is worth. `setValues` (the omegaWallFunction's
# manipulateMatrix) eliminates 6,310 of the 28,000 rows -- every off-diagonal touching them is zero --
# and on such a row the answer is Src/D whatever the diagonal is. So:
#
#   * omega's and k's OFF-DIAGONALS, and D and Src on the 21,690 rows that survive: the discriminator.
#   * the 6,310 eliminated rows' IMPLIED VALUE (Src/D): a completeness check, not a discriminator --
#     the controls below move it by NOTHING (1.8e-14 in every arm, scheme or no scheme), because a
#     pinned row carries no convection. It is asserted so that a change to the wall constraint cannot
#     pass unnoticed; it is not what proves the scheme.
#
# MEASURED, 28,000 cells / 79,800 internal faces, first closure call:
#
#   arm                                        omega upper   omega D(free)   omega Src(free)   k upper
#   A  DEVICE limitedLinear vs OF same          3.4e-14        3.9e-14         5.4e-13         1.2e-13
#   B  host   limitedLinear vs OF same          3.2e-14        3.3e-14         8.0e-13         7.8e-14
#   C  DEVICE upwind        vs OF same          3.6e-14        6.0e-14         5.3e-13         5.9e-14
#   D  DEVICE limitedLinear vs OF UPWIND        5.0e-01        4.2e-01         1.2e-01         5.0e-01
#   E  DEVICE upwind        vs OF limitedLinear 5.0e-01        3.4e-01         1.2e-01         1.0e+00
#
# A, B and C must MATCH (bound 1e-10, two orders above the worst measured 8.0e-13). D and E must
# DIFFER (floor 1e-3, two orders below the smallest control 1.2e-01). C is what makes D and E mean
# something: without it, "brae's limitedLinear is not OpenFOAM's upwind" could be a broken upwind
# path rather than a limiter that works. The margin between a match and a control is thirteen orders.
#
# FAIL-PROOFS, each broken once in a scratch build of the shared assembler
# (src/TurbulenceModels/turbulenceModels/turbulenceModel/turbulence_transport.cu, linked ahead of
# libbrae_core.a so the tree was never touched), the gate red every time and red in the RIGHT arm:
#
#   the device ignores the limiter (`if (false)` on the limited branch)
#       -> arm B red on all eight columns, worst 9.989e-01; A and C still green, so the gate names
#          the arm that broke rather than falling over
#   the case's coefficient doubled (twoByk built from 2*k)
#       -> arm B red on omega D 7.301e-04, omega Src 6.450e-03, k D 8.068e-07, k Src 9.216e-06 --
#          and the OFF-DIAGONALS do not move at all. MEASURED why: all 10,932 faces whose weight
#          changed belong to rows setValues eliminates, so the matrix keeps none of their
#          coefficients; the defect reaches the answer through relax(), which runs BEFORE setValues
#          and folds those coefficients into the neighbours' diagonal dominance, and through the
#          elimination's source transfer. That is why D and Src on the surviving rows are asserted
#          and not only the off-diagonals: on this case they are the only columns that can see a
#          wrong limiter COEFFICIENT.
#
# WHAT THIS LIFTS. The device closure refused `Gauss limitedLinear` by name. Its assembled system is
# OpenFOAM's; the refusal was for want of a gate, and this is the gate. `tests/interfoam_refusals.sh`
# holds the kEpsilon half, which is still refused (no oracle here writes its system).
#
# NOT CLAIMED: the second and later closure calls (from call two the two arms hold different fields --
# omega at the assembly is 6.1e-14 apart at call 2 and 4.7e-11 at call 5 -- so the systems differ by
# what the fields differ by, and no assembly comparison can separate the two); the device kEpsilon
# closure under the same scheme; `limitedLinear` with a coefficient other than 1; any other case.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BUILD:-$ROOT/build}/brae_interFoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/RAS/waterChannel"
DT=0.1
END=0.1

[ -x "$BRAE" ]     || { echo "SKIP: $BRAE not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: RAS/waterChannel tutorial not found at $SRC"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

# the binary must be newer than the sources it is supposed to hold -- a stale one has reported a false
# green here before (tests/require_fresh_binary.sh)
. "$(dirname "$0")/require_fresh_binary.sh"
requireFresh "$BRAE" || exit 1

# the oracle cache, and the loud failure if it did not load: an uncached gate is slow, but a gate that
# silently lost `oracleKey` would also have lost the key CHECK that makes a restore safe
. "$(dirname "$0")/of_oracle_cache.sh"
command -v oracleKey > /dev/null \
    || { echo "FAIL: of_oracle_cache.sh did not define oracleKey -- the gate would run uncached"; exit 1; }

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v blockMesh   > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }
command -v extrudeMesh > /dev/null 2>&1 || { echo "SKIP: extrudeMesh not on PATH"; exit 77; }
command -v interFoam   > /dev/null 2>&1 || { echo "SKIP: interFoam not on PATH"; exit 77; }

# EVERY arm this gate asserts is a device arm -- the host arm is here to keep the two honest, not as
# the claim -- so with no GPU there is nothing left to run and the gate skips rather than passing on
# the host alone.
command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi > /dev/null 2>&1 \
    || { echo "SKIP: no GPU -- this gate's subject is the device closure"; exit 77; }

DUMPLIB="${FOAM_USER_LIBBIN:-}/libdumpKOmegaSST.so"
[ -f "$DUMPLIB" ] || { echo "SKIP: the oracle library is not built -- (cd $ROOT/tools/dumpKOmegaSST && wmake libso)"; exit 77; }

W=${KEEP_W:-$(mktemp -d)}
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"

rc=0
pass() { echo "  ok:   $1"; }
fail() { echo "  FAIL: $1"; rc=1; }

# stage <name> <scheme text> -- the tutorial, meshed as Allrun.pre does, ONE step of deltaT 0.1, with
# the closure's div scheme replaced. The OpenFOAM run that follows writes its assembled system at the
# first time index and is cached on a hash of every staged byte.
stage()
{
    local name="$1" scheme="$2"
    local C="$W/$name"
    rm -rf "$C"
    cp -r "$SRC" "$C" || return 1
    rm -rf "$C"/[1-9]* "$C"/0 "$C"/processor* "$C"/log.*
    cp -r "$C/0.orig" "$C/0"
    grep -q "RASModel  *kOmegaSST;" "$C/constant/turbulenceProperties" \
        || { echo "FAIL: the tutorial no longer names kOmegaSST"; return 1; }

    SCHEME="$scheme" python3 - "$C" <<'PYEOF' || { echo "FAIL: staging $name"; return 1; }
import os, re, sys
d, scheme = sys.argv[1], os.environ['SCHEME']
p = os.path.join(d, 'system/fvSchemes')
s = open(p).read()
pat = r'^(\s*)"div\\\(phi,\(k\|omega\)\\\)"(\s+).*$'
if scheme == 'split':
    # ONE PATTERN KEY BECOMES TWO LITERAL ENTRIES. `fvm::div(phi, psi)` resolves its entry by the FIELD's
    # name, and a literal beats a pattern in OpenFOAM's lookup, so writing both literals is what makes the
    # two equations two schemes. k keeps the tutorial's upwind; omega takes limitedLinear.
    s, k = re.subn(pat,
                   lambda m: m.group(1) + 'div(phi,k)' + m.group(2) + 'Gauss upwind;\n'
                           + m.group(1) + 'div(phi,omega)' + m.group(2) + 'Gauss limitedLinear 1;', s,
                   flags=re.M)
else:
    s, k = re.subn(pat, lambda m: m.group(1) + '"div\\(phi,(k|omega)\\)"' + m.group(2)
                                  + 'Gauss ' + scheme + ';', s, flags=re.M)
assert k == 1, 'the tutorial no longer carries one div(phi,(k|omega)) entry'
open(p, 'w').write(s)

c = os.path.join(d, 'system/controlDict')
s = open(c).read()
# the function objects write nothing this gate reads, and `s` is a scalarTransport brae does not run
s = re.sub(r'\nfunctions\s*\{.*\n\}\s*\n', '\n', s, flags=re.S)
for key, val in [('adjustTimeStep', 'no'), ('deltaT', '0.1'), ('endTime', '0.1'),
                 ('writeControl', 'timeStep'), ('writeInterval', '1'), ('writeFormat', 'ascii'),
                 ('writePrecision', '15')]:
    s, k = re.subn(r'^%s\s.*' % key, '%s %s;' % (key.ljust(15), val), s, flags=re.M)
    assert k == 1, key
s = re.sub(r'^writeCompression\s.*', 'writeCompression off;', s, flags=re.M)
open(c, 'w').write(s)
PYEOF
    if [ "$scheme" = split ]; then
        grep -q "div(phi,k)" "$C/system/fvSchemes" && grep -q "div(phi,omega)" "$C/system/fvSchemes" \
            && grep -q "Gauss limitedLinear 1;" "$C/system/fvSchemes" \
            && grep -q "Gauss upwind;" "$C/system/fvSchemes" \
            || { echo "FAIL: the split entries did not reach fvSchemes [$name]"; return 1; }
    else
        grep -q "Gauss $scheme;" "$C/system/fvSchemes" \
            || { echo "FAIL: \`Gauss $scheme\` did not reach div(phi,(k|omega)) [$name]"; return 1; }
    fi

    ( cd "$C" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh [$name]"; tail -20 "$C/log.blockMesh"; return 1; }
    local i
    for i in 1 2; do
        cp "$C/system/extrudeMeshDict.$i" "$C/system/extrudeMeshDict"
        ( cd "$C" && extrudeMesh > "log.extrudeMesh.$i" 2>&1 ) \
            || { echo "FAIL: extrudeMesh $i [$name]"; tail -20 "$C/log.extrudeMesh.$i"; return 1; }
    done
    ( cd "$C" && setFields > log.setFields 2>&1 ) || { echo "FAIL: setFields [$name]"; tail -20 "$C/log.setFields"; return 1; }

    # OpenFOAM's own closure with the writes, for ONE step. The model name goes back to kOmegaSST
    # afterwards, because brae refuses `kOmegaSSTDump` by name -- as it should.
    local key
    key=$(oracleKey "$C" "interfoam_sst_assembly" "$name" "$(sha256sum "$DUMPLIB" | cut -d' ' -f1)")
    if oracleRestore "$C" "$key" "$END"; then
        echo "OpenFOAM's assembled system reused from the oracle cache   [$name]"
    else
        sed -i 's/RASModel  *kOmegaSST;/RASModel        kOmegaSSTDump;/' "$C/constant/turbulenceProperties"
        printf '\nlibs            ("libdumpKOmegaSST.so");\n' >> "$C/system/controlDict"
        ( cd "$C" && BRAE_DUMP_STAGE_ITER=1 interFoam > log.interFoam 2>&1 ) \
            || { echo "FAIL: interFoam [$name]"; tail -30 "$C/log.interFoam"; return 1; }
        sed -i 's/RASModel  *kOmegaSSTDump;/RASModel        kOmegaSST;/' "$C/constant/turbulenceProperties"
        sed -i '/libdumpKOmegaSST/d' "$C/system/controlDict"
        oracleStore "$C" "$key"
        echo "OpenFOAM ran one step of deltaT $DT to t = $END with the dump model   [$name]"
    fi
    local f
    for f in stage_sstOmD stage_sstOmSrc stage_sstOmDUpper stage_sstOmDLower \
             stage_sstKD  stage_sstKSrc  stage_sstKDUpper  stage_sstKDLower; do
        [ -f "$C/$END/$f" ] || { echo "FAIL: the oracle wrote no $f [$name]"; return 1; }
    done
    grep -q "RASModel  *kOmegaSST;" "$C/constant/turbulenceProperties" \
        || { echo "FAIL: the case was left naming the dump model [$name]"; return 1; }
}

stage limitedLinear "limitedLinear 1" || { echo "interfoam_sst_assembly_vs_openfoam: staging failed"; exit 1; }
stage upwind        "upwind"          || { echo "interfoam_sst_assembly_vs_openfoam: staging failed"; exit 1; }
# THE SPLIT: div(phi,k) upwind beside div(phi,omega) limitedLinear 1. Two schemes in one closure call,
# which the device arm refused until KOmegaSSTInput::omegaDiv/omegaGrad carried the second equation's
# entries into schemeOf(). The assembled system is the right oracle here for the same reason the
# whole gate exists: at the first call omega is uniform, so limitedLinear's limiter is a 0/0 ratio
# and no FIELD bound can separate the scheme from the last bit (the RAS/damBreak `splitDivSST` device
# arm reads omega 1.275e-07 on that first solve's initial residual where its fields are at the SST
# device floor, and the host reads 2.742e-15 -- the same coin-flip, landing the other way).
stage split         "split"           || { echo "interfoam_sst_assembly_vs_openfoam: staging failed"; exit 1; }

# runBrae <case> <arm: host|device> -- writes <case>/dump/<host|cuda>/... at the first closure call
runBrae()
{
    local C="$W/$1" arm="$2"
    local d="$C/dump.$arm"
    rm -rf "$d"
    local flag=""
    [ "$arm" = device ] && flag="-device"
    ( cd "$C" && BRAE_SST_DUMP_DIR="$d" BRAE_SST_DUMP_ITER=1 "$BRAE" -case "$C" $flag > "log.brae.$arm" 2>&1 ) \
        || { echo "FAIL: brae $arm did not run [$1]"; tail -20 "$C/log.brae.$arm"; return 1; }
    if [ "$arm" = device ]; then
        grep -q "\[device\]" "$C/log.brae.$arm" \
            || { echo "FAIL: brae's run does not say it was on the device [$1]"; return 1; }
        grep -qi "BRAE_SST_DIAG_LIMITED" "$C/log.brae.$arm" \
            && { echo "FAIL: the device arm ran under the diagnostic bypass, not on its own [$1]"; return 1; }
    fi
    return 0
}

# compare <oracle case> <brae case> <arm> <expect: match|differ> <label> -- through the comparator
# every assembly gate shares (tests/assembly_compare.py), so the rule about which rows carry the claim
# is written once and cannot drift between the kOmegaSST gate and the kEpsilon one.
compare()
{
    local ofc="$W/$1" brc="$W/$2" arm="$3" expect="$4" label="$5"
    local sub=host flavour=sstHost
    [ "$arm" = device ] && { sub=cuda; flavour=sstCuda; }
    python3 "$(dirname "$0")/assembly_compare.py" \
        "$ofc/$END" "$brc/dump.$arm/$sub" "$ofc/constant/polyMesh" "$flavour" "$expect" "$label"
}

echo "== brae's kOmegaSST assembled system vs OpenFOAM's own, first closure call =="

runBrae limitedLinear host   || rc=1
runBrae limitedLinear device || rc=1
runBrae upwind       host    || rc=1
runBrae upwind       device  || rc=1
runBrae split        host    || rc=1
runBrae split        device  || rc=1

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

    # THE SPLIT, host and device, and TWO controls. F/G hold each arm's assembled system against
    # OpenFOAM's own for the same split; H and I are what make F/G mean the SPLIT rather than either
    # uniform scheme -- a device arm that silently ran ONE scheme for both equations would match one of
    # them instead of the split, which is exactly what this arm was refused for.
    compare split split host   match  "F host  SPLIT vs OpenFOAM's own split" \
        && pass "the HOST's split assembled system is OpenFOAM's" \
        || fail "the HOST's split assembled system is OpenFOAM's"

    compare split split device match  "G DEVICE SPLIT vs OpenFOAM's own split" \
        && pass "...and the DEVICE's is, which is what its refusal was waiting for" \
        || fail "...and the DEVICE's is, which is what its refusal was waiting for"

    compare limitedLinear split device differ "H DEVICE SPLIT vs OpenFOAM's BOTH-limitedLinear" \
        && pass "CONTROL: the split is not both equations limited -- k really took upwind" \
        || fail "CONTROL: the split is not both equations limited -- k really took upwind"

    compare upwind split device differ "I DEVICE SPLIT vs OpenFOAM's BOTH-upwind" \
        && pass "CONTROL: ...and not both upwind either -- omega really took the limiter" \
        || fail "CONTROL: ...and not both upwind either -- omega really took the limiter"
fi

echo "interfoam_sst_assembly_vs_openfoam: rc $rc"
exit $rc
