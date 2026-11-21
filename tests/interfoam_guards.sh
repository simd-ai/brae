#!/usr/bin/env bash
# brae interFoam's start-up guards of 2026-11-21, on laminar/damBreak, one input changed at a time. What each
# arm holds is where the case stood before: ACCEPTED and run under another name, or refused LATE.
#
#   UNWRITTEN TYPES (inter_writer_cpp.cu, refuseUntranscribedType). A condition type the writer has no branch
#     for -- mixed, fixedGradient, uniformTotalPressure, ... -- was named at start-up and refused at the FIRST
#     WRITE TIME, so the run spent its steps on a boundary no interFoam gate has run and then stopped with
#     nothing written. Refused before step 1 now: the arms assert the message comes before any step line.
#   p0 TABLES (inter_case_cpp.cu, buildInterFields). `p0 table (...)` on uniformTotalPressure was held at its
#     t = 0 value (neither loop refreshes it), and on totalPressure -- where OpenFOAM reads p0 as a field and
#     stops on a table (totalPressureFvPatchScalarField.C:67) -- it RAN, frozen, and wrote p0 out.
#   fvSchemes PATTERNS AND A ONE-LINE BLOCK. OpenFOAM looks a scheme up as a dictionary does: the literal key,
#     else the last quoted (regular-expression) key that matches, else `default` (schemesLookupDetail.C:76-89,
#     dictionarySearch.C). interFoam's readers searched the text: a pattern was never seen, a name inside a
#     quoted key was taken as the key, and `interpolationSchemes { default cubic; }` on one line was skipped.
#     div(rhoPhi,U) and the two alpha fluxes now RESOLVE as OpenFOAM does (pattern and `default` alike: the
#     runs must equal the literal case's bytes); gradients, laplacians and snGrads that a pattern would
#     resolve differently are refused by name.
#   THE PIMPLE BLOCK (preflightInterPimple). An entry OpenFOAM's interFoam ignores -- `cAlpha` in PIMPLE, as
#     two waveMaker tutorials ship it -- must RUN, and be named by the unread-entry audit.
#   THE UNREAD-ENTRY AUDIT (auditInterDictionaries, dict_audit.cuh): fvSolution, turbulenceProperties and
#     fvSchemes each name an entry no reader asked for, as `brae NOTICE [unread]`; interFoam printed none.
#   THE DEVICE V-SCHEMES (inter_driver_device.cu): `Gauss vanLeerV` or `limitedLinearV` with a cellLimited
#     grad(U) ran the GPU limiter on an unlimited gradient. Refused on -device; the host arm runs it.
#
# Every refused arm sits beside the nearest form that must still run. CI runs every test with
# BRAE_PRESSURE_CASE_SOLVER=1 and BRAE_TURBULENCE_CASE_SOLVER=1; nothing here depends on either. The divergence
# stop and minIter have their own gates (interfoam_divergence_vs_openfoam.sh, interfoam_miniter_vs_openfoam.sh).
# No OpenFOAM solver is run -- only blockMesh and setFields, because damBreak ships no mesh.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/brae_interFoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}
SRC="$TUT/multiphase/interFoam/laminar/damBreak/damBreak"
SRCR="$TUT/multiphase/interFoam/RAS/damBreak/damBreak"

# shellcheck disable=SC1091
. "$(dirname "$0")/require_fresh_binary.sh"
[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: damBreak tutorial not found at $SRC"; exit 77; }
[ -d "$SRCR" ]     || { echo "SKIP: RAS/damBreak tutorial not found at $SRCR"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available (blockMesh is needed)"; exit 77; }
requireFresh "$BIN" || exit 1

W=${KEEP_W:-$(mktemp -d)}
[ -n "${KEEP_W:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v blockMesh > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }

# stage <src> <dir>: the tutorial meshed as its Allrun does, two fixed steps, the last one written
stage()
{
    rm -rf "$2"
    cp -r "$1" "$2" || return 1
    cp -r "$2/0.orig" "$2/0"
    ( cd "$2" && blockMesh > log.blockMesh 2>&1 && setFields > log.setFields 2>&1 ) || return 1
    python3 - "$2/system/controlDict" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
for key, val in [('endTime', '0.0002'), ('deltaT', '1e-4'), ('adjustTimeStep', 'no'), ('writeControl', 'timeStep'),
                 ('writeInterval', '2'), ('writePrecision', '17'), ('writeFormat', 'ascii')]:
    s, n = re.subn(r'^%s\s.*' % key, '%s %s;' % (key.ljust(15), val), s, flags=re.M)
    assert n == 1, key
s = re.sub(r'\nfunctions\s*\{.*\n\}\s*\n', '\n', s, flags=re.S)
open(p, 'w').write(s)
PY
}
B="$W/base"
BR="$W/baseRAS"
stage "$SRC" "$B"   || { echo "SKIP: blockMesh/setFields failed on damBreak"; exit 77; }
stage "$SRCR" "$BR" || { echo "SKIP: blockMesh/setFields failed on RAS/damBreak"; exit 77; }

HAVE_GPU=0
if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi > /dev/null 2>&1; then HAVE_GPU=1; fi
fails=0

# arm <name> <expect: refused|runs> <needle or -> <flags> <edit>: the edit runs in a copy of $BASE (default
# damBreak). A refused arm must carry the needle in the message and stop BEFORE its first step line; a `runs`
# arm with a needle must carry it as a notice.
arm()
{
    local name="$1" expect="$2" needle="$3" flags="$4"
    shift 4
    local C="$W/$name"
    rm -rf "$C"
    cp -r "${BASE:-$B}" "$C"
    ( cd "$C" && eval "$@" ) || { echo "  FAIL: $name -- the staging edit itself failed"; fails=$((fails+1)); return; }
    local out rc
    # shellcheck disable=SC2086
    out=$(cd "$C" && stdbuf -oL -eL "$BIN" -case . $flags 2>&1); rc=$?
    printf '%s\n' "$out" > "$C/log.brae"
    local got=refused
    { [ $rc -eq 0 ] && printf '%s\n' "$out" | grep -q "^End: t"; } && got=runs
    local ok=1
    [ "$got" = "$expect" ] || ok=0
    if [ "$needle" != "-" ]; then
        printf '%s\n' "$out" | grep -qF -- "$needle" || ok=0
    fi
    if [ "$expect" = refused ] && [ $ok = 1 ]; then
        # ...at START-UP: the message before any step line, and no time directory written
        python3 - "$C/log.brae" "$needle" <<'PY' || ok=0
import re, sys
log, needle = open(sys.argv[1]).read(), sys.argv[2]
at = log.find(needle) if needle != '-' else len(log)
first = re.search(r'^\s*t = ', log, flags=re.M)
sys.exit(0 if (first is None or at < first.start()) else 1)
PY
        [ -z "$(cd "$C" && ls -d 0.0* 2>/dev/null)" ] || ok=0
    fi
    if [ $ok = 1 ]; then
        printf "  ok:   %-30s %s\n" "$name" "$got"
    else
        printf "  FAIL: %-30s expected %s%s, got %s: %s\n" "$name" "$expect" \
               "$([ "$needle" != "-" ] && echo " naming \`$needle\`")" "$got" \
               "$(printf '%s\n' "$out" | grep -E '^brae interFoam|^End:' | tail -1 | cut -c1-160)"
        fails=$((fails+1))
    fi
}

# same <a> <b>: two arms' written last step byte-identical (a resolution that must not move a number)
same()
{
    if diff -r -q "$W/$1/0.0002" "$W/$2/0.0002" > /dev/null 2>&1; then
        printf "  ok:   %-30s writes %s's bytes\n" "$2" "$1"
    else
        printf "  FAIL: %-30s does not write %s's bytes\n" "$2" "$1"
        fails=$((fails+1))
    fi
}

# patch <file> <patch> <body>: replace one patch's entry of a field file (body on one shell line, \n allowed)
cat > "$W/patch.py" <<'PY'
import re, sys
path, patch, body = sys.argv[1], sys.argv[2], sys.argv[3].replace('\\n', '\n')
t = open(path).read()
t2, n = re.subn(r'(\n\s*' + re.escape(patch) + r'\s*\n\s*\{)[^}]*\}', lambda m: m.group(1) + '\n' + body + '\n    }', t, count=1)
assert n == 1, (path, patch)
open(path, 'w').write(t2)
PY
PATCH="python3 $W/patch.py"
# sub <file> <python regex> <replacement>: one substitution that must change the file
cat > "$W/sub.py" <<'PY'
import re, sys
path, pat, rep = sys.argv[1], sys.argv[2], sys.argv[3].replace('\\n', '\n')
t = open(path).read()
t2 = re.sub(pat, lambda m: rep, t, count=1, flags=re.S)
assert t2 != t, (path, pat)
open(path, 'w').write(t2)
PY
SUB="python3 $W/sub.py"

echo "== brae interFoam: the start-up guards =="
arm baseline                    runs    -   "" true

echo "-- condition types the writer has no branch for: refused before step 1"
arm unwritten_mixedU            refused "is \`mixed\`, a condition interFoam is not ported for" "" \
    "$PATCH 0/U atmosphere '        type mixed;\n        refValue uniform (0 0 0);\n        refGradient uniform (0 0 0);\n        valueFraction uniform 1;\n        value uniform (0 0 0);'"
arm unwritten_fixedGradientA    refused "is \`fixedGradient\`, a condition interFoam is not ported for" "" \
    "$PATCH 0/alpha.water atmosphere '        type fixedGradient;\n        gradient uniform 0;\n        value uniform 0;'"
arm unwritten_uniformTP_const   refused "is \`uniformTotalPressure\`, a condition interFoam is not ported for" "" \
    "$PATCH 0/p_rgh atmosphere '        type uniformTotalPressure;\n        p0 constant 0;\n        value uniform 0;'"
echo "-- a p0 table: refused before step 1 on both total-pressure conditions"
arm p0table_uniformTP           refused "uniformTotalPressure with a p0 table" "" \
    "$PATCH 0/p_rgh atmosphere '        type uniformTotalPressure;\n        p0 table ((0 0) (1 1000));\n        value uniform 0;'"
arm p0table_totalPressure       refused "totalPressure with \`p0\` given as a table" "" \
    "$PATCH 0/p_rgh atmosphere '        type totalPressure;\n        p0 table ((0 0) (1 1000));\n        value uniform 0;'"
arm p0uniform_totalPressure     runs    -   "" \
    "$PATCH 0/p_rgh atmosphere '        type totalPressure;\n        p0 uniform 0;\n        value uniform 0;'"

echo "-- interpolationSchemes on one line"
arm interp_oneline_cubic        refused "names \`default cubic\`" "" \
    "$SUB system/fvSchemes 'interpolationSchemes\s*\{[^}]*\}' 'interpolationSchemes { default cubic; }'"
arm interp_oneline_linear       runs    -   "" \
    "$SUB system/fvSchemes 'interpolationSchemes\s*\{[^}]*\}' 'interpolationSchemes { default linear; }'"
same baseline interp_oneline_linear

echo "-- divSchemes resolved as OpenFOAM resolves them: pattern, then default"
arm div_alpha_pattern           runs    -   "" \
    "$SUB system/fvSchemes '\n\s*div\(phi,alpha\)\s[^;]*;' '\n    \"div\\(phi,alpha\\)\" Gauss vanLeer;'"
same baseline div_alpha_pattern
arm div_U_pattern               runs    -   "" \
    "$SUB system/fvSchemes '\n\s*div\(rhoPhi,U\)\s[^;]*;' '\n    \"div\\(rhoPhi,.*\\)\" Gauss linearUpwind grad(U);'"
same baseline div_U_pattern
arm div_U_default               runs    -   "" \
    "$SUB system/fvSchemes '\n\s*div\(rhoPhi,U\)\s[^;]*;' '\n    default Gauss linearUpwind grad(U);'"
same baseline div_U_default
arm div_U_missing               refused "has no \`div(rhoPhi,U)\` and no default" "" \
    "$SUB system/fvSchemes '\n\s*div\(rhoPhi,U\)\s[^;]*;' ''"
# a QUOTED key is a regular expression to OpenFOAM: `"div(phi,alpha)"` matches `divphi,alpha`, not the name
arm div_alpha_quoted_literal    refused "has no default to take its place" "" \
    "$SUB system/fvSchemes '\n\s*div\(phi,alpha\)\s[^;]*;' '\n    \"div(phi,alpha)\" Gauss vanLeer;'"
arm stress_pattern              refused "through the pattern" "" \
    "$SUB system/fvSchemes '\n\s*div\(\(\(rho\*nuEff\)\*dev2\(T\(grad\(U\)\)\)\)\)\s[^;]*;' '\n    \"div\\(\\(\\(rho.*\" Gauss linear;\n    default none;'"

echo "-- gradSchemes: a name a pattern answers otherwise than brae's reader, refused"
arm grad_pattern_limited        refused "resolves \`grad(p_rgh)\` to \`cellLimited Gauss linear 1\` through the pattern" "" \
    "$SUB system/fvSchemes '(gradSchemes\s*\{)' 'gradSchemes\n{\n    \"grad\\(p_rgh\\)\" cellLimited Gauss linear 1;'"
arm grad_quoted_literal         refused "resolves \`grad(U)\` to \`Gauss linear\` through the entry default" "" \
    "$SUB system/fvSchemes '(gradSchemes\s*\{)' 'gradSchemes\n{\n    \"grad(U)\" cellLimited Gauss linear 1;'"
arm grad_pattern_same           runs    -   "" \
    "$SUB system/fvSchemes '(gradSchemes\s*\{)' 'gradSchemes\n{\n    \"grad\\(.*\\)\" Gauss linear;'"
same baseline grad_pattern_same

echo "-- laplacianSchemes and snGradSchemes: a pattern beside the default"
arm lap_pattern                 refused "names the pattern" "" \
    "$SUB system/fvSchemes '(laplacianSchemes\s*\{)' 'laplacianSchemes\n{\n    \"laplacian\\(.*p_rgh\\)\" Gauss linear uncorrected;'"
arm lap_pattern_same            runs    -   "" \
    "$SUB system/fvSchemes '(laplacianSchemes\s*\{)' 'laplacianSchemes\n{\n    \"laplacian\\(.*p_rgh\\)\" Gauss linear corrected;'"
arm sngrad_pattern              refused "names the pattern" "" \
    "$SUB system/fvSchemes '(snGradSchemes\s*\{)' 'snGradSchemes\n{\n    \"snGrad\\(.*\\)\" uncorrected;'"

echo "-- PIMPLE entries OpenFOAM's interFoam ignores: they run, and the audit names the ones nothing reads"
arm pimple_cAlpha               runs    "brae NOTICE [unread]   PIMPLE/cAlpha" "" \
    "$SUB system/fvSolution '(PIMPLE\s*\{)' 'PIMPLE\n{\n    cAlpha 1;'"
arm pimple_ddtCorr_inert        runs    -   "" \
    "$SUB system/fvSolution '(PIMPLE\s*\{)' 'PIMPLE\n{\n    ddtCorr yes;\n    consistent yes;'"
if grep -q 'PIMPLE/ddtCorr\|PIMPLE/consistent' "$W/pimple_ddtCorr_inert/log.brae"; then
    echo "  FAIL: pimple_ddtCorr_inert            the inert entries were reported unread"; fails=$((fails+1))
fi

echo "-- the unread-entry audit, file by file"
arm audit_fvSolution            runs    "brae NOTICE [unread]   PIMPLE/nCorectors" "" \
    "$SUB system/fvSolution '(PIMPLE\s*\{)' 'PIMPLE\n{\n    nCorectors 2;'"
arm audit_fvSchemes             runs    "brae NOTICE [unread]   fluxRequired/ (whole block)" "" \
    "printf 'fluxRequired\n{\n    default no;\n}\n' >> system/fvSchemes"
BASE="$BR"
arm audit_turbulence            runs    "brae NOTICE [unread]   RAS/bogusEntry" "" \
    "$SUB constant/turbulenceProperties '(RAS\s*\{)' 'RAS\n{\n    bogusEntry 1;'"
arm audit_turbulence_clean      runs    -   "" true
if grep -q 'NOTICE \[unread\] system/fvSolution\|NOTICE \[unread\] system/fvSchemes' "$W/audit_turbulence_clean/log.brae"; then
    echo "  FAIL: audit_turbulence_clean          RAS/damBreak as shipped reports an unread fvSolution or fvSchemes entry"
    fails=$((fails+1))
fi
BASE="$B"

echo "-- the GPU loop's V-scheme limiter under a cellLimited grad(U)"
VLV="$SUB system/fvSchemes 'div\(rhoPhi,U\)\s[^;]*;' 'div(rhoPhi,U)   Gauss vanLeerV;'"
LIMU="$SUB system/fvSchemes '(gradSchemes\s*\{)' 'gradSchemes\n{\n    grad(U) cellLimited Gauss linear 1;'"
arm vanLeerV_limitedU_host      runs    -   "" "$VLV && $LIMU"
if [ $HAVE_GPU = 1 ]; then
    arm vanLeerV_limitedU_device        refused "names \`div(rhoPhi,U) Gauss vanLeerV\` and grad(U) resolves to \`cellLimited" "-device" "$VLV && $LIMU"
    arm limitedLinearV_limitedU_device  refused "Gauss limitedLinearV\` and grad(U) resolves to \`cellLimited" "-device" \
        "$SUB system/fvSchemes 'div\(rhoPhi,U\)\s[^;]*;' 'div(rhoPhi,U)   Gauss limitedLinearV 1;' && $LIMU"
    arm vanLeerV_device                 runs    -   "-device" "$VLV"
    arm limitedU_device                 runs    -   "-device" "$LIMU"
else
    echo "  (no GPU: the -device arms are not run)"
fi

echo "interfoam_guards: $fails failure(s)"
[ $fails -eq 0 ]
