#!/usr/bin/env bash
# linearUpwindV AND LUST READ A GRADIENT NAME, AND IT COMES OFF div(phi,U)'s OWN STREAM.
#
# All three name-taking schemes read the word straight after themselves: linearUpwind.H:105/:117,
# linearUpwindV.H:117-135, and LUST.H, whose constructor hands the stream to linearUpwind unchanged. An
# unresolved name falls back to gradSchemes `default` (schemesLookupDetail.C:76-88), which is a perfectly
# valid scheme -- just not the one the case asked for, which is why the miss was silent.
#
# brae extracted the name with `std::regex(R"(linearUpwind\s+([^\s;]+))")` searched over the WHOLE
# divSchemes block. Three holes: it cannot match `linearUpwindV` (the next character is V, not
# whitespace), cannot match `LUST` at all, and takes whichever statement comes first rather than
# div(phi,U)'s own. Measured on staged pitzDaily copies sharing `grad(U) cellLimited Gauss linear 1`:
# the linearUpwind copy reported the coefficient, the linearUpwindV and LUST copies reported nothing --
# character-for-character identical to a case with no entry at all.
#
#   ARM 1  each of linearUpwind / linearUpwindV / LUST resolves grad(U), and each names ITSELF in the log.
#   ARM 2  `upwind` resolves NO gradient and prints no such line -- it reads none off its stream, and a
#          line claiming otherwise would be the same class of lie in the other direction.
#   ARM 3  ORDERING: with `div(phi,k) ... linearUpwind grad(k)` written ABOVE `div(phi,U) ... linearUpwind
#          grad(U)` and only grad(k) limited, brae must report NOTHING for div(phi,U).
#   ARM 4  PROXIMITY, against real OpenFOAM. Two OpenFOAM runs differing ONLY in the gradient
#          linearUpwindV names -- `grad(U)` (cellLimited 1) against a `Gauss linear` alias -- with
#          gradSchemes grad(U) itself untouched in both, so divDevReff and the closures are identical and
#          the arm isolates the deferred correction. brae must land nearer the named oracle.
#   CONTROL the two oracles must differ by more than REL_MIN, or ARM 4 is choosing between two identical
#           answers.
#
# Fail-proof, 2026-09-04: restoring the block-wide regex makes ARM 1 red on linearUpwindV and LUST, ARM 3
# red (it reports grad(k)'s coefficient for div(phi,U)), and flips ARM 4.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="${1:-$ROOT/validation/pitzDaily}"
PROX="${2:-$ROOT/validation/windAroundBuildingsBox}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
N=${N:-20}
REL_MIN=${REL_MIN:-1e-04}
MARGIN=${MARGIN:-1.05}

[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$SRC" ]  || { echo "SKIP: fixture $SRC missing"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
SRC="$(cd "$SRC" && pwd)"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-70s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

# ---- ARMS 1-3: what the driver resolves, from its own log -------------------------------------------
probe()   # probe <divSchemes body override> ; echoes the named-gradient line or ""
{
    rm -rf "$W/p"; mkdir -p "$W/p"
    cp -r "$SRC/constant" "$SRC/system" "$W/p/"
    [ -d "$SRC/0.orig" ] && cp -r "$SRC/0.orig" "$W/p/0" || cp -r "$SRC/0" "$W/p/0"
    python3 - "$W/p" "$1" "$2" <<'PY'
import re, sys
d, divbody, gradbody = sys.argv[1], sys.argv[2], sys.argv[3]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'endTime\s+\S+;', 'endTime         1;', s)
s = re.sub(r'functions\s*\{.*\n\}', 'functions\n{\n}', s, flags=re.S)
open(c, 'w').write(s)
f = d + '/system/fvSchemes'; s = open(f).read()
s = re.sub(r'divSchemes\s*\{[^}]*\}', 'divSchemes { ' + divbody + ' }', s, flags=re.S)
s = re.sub(r'gradSchemes\s*\{[^}]*\}', 'gradSchemes { ' + gradbody + ' }', s, flags=re.S)
open(f, 'w').write(s)
PY
    ( cd "$W/p" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" "$W/p" 2>&1 | grep "named gradient" || true )
}
GRADLIM='default Gauss linear; grad(U) cellLimited Gauss linear 1;'
DIVREST='div((nuEff*dev2(T(grad(U))))) Gauss linear; turbulence bounded Gauss upwind; div(phi,k) bounded Gauss upwind; div(phi,epsilon) bounded Gauss upwind;'
for sch in linearUpwind linearUpwindV LUST; do
    out=$(probe "default none; div(phi,U) bounded Gauss $sch grad(U); $DIVREST" "$GRADLIM")
    echo "$out" | grep -q "$sch's named gradient: cellLimited Gauss linear 1" \
        && say "ARM 1  $sch resolves grad(U) and names itself" ok \
        || { say "ARM 1  $sch resolves grad(U) and names itself" FAIL; echo "        got: ${out:-<nothing>}"; }
done
out=$(probe "default none; div(phi,U) bounded Gauss upwind; $DIVREST" "$GRADLIM")
[ -z "$out" ] && say "ARM 2  upwind resolves no gradient and prints no such line" ok \
              || { say "ARM 2  upwind resolves no gradient and prints no such line" FAIL; echo "        got: $out"; }
out=$(probe "default none; div(phi,k) bounded Gauss linearUpwind grad(k); div(phi,epsilon) bounded Gauss upwind; div(phi,U) bounded Gauss linearUpwind grad(U); div((nuEff*dev2(T(grad(U))))) Gauss linear;" \
            "default Gauss linear; grad(U) Gauss linear; grad(k) cellLimited Gauss linear 0.5;")
[ -z "$out" ] && say "ARM 3  div(phi,U)'s own entry wins over an earlier div(phi,k)" ok \
              || { say "ARM 3  div(phi,U)'s own entry wins over an earlier div(phi,k)" FAIL; echo "        got: $out"; }

# ---- ARM 4: proximity against real OpenFOAM ---------------------------------------------------------
if [ -f "$OFBASHRC" ] && [ -d "$PROX" ]; then
    PROX="$(cd "$PROX" && pwd)"
    set +u
    # shellcheck disable=SC1091
    source "$OFBASHRC" > /dev/null 2>&1 || true
    set -u
    if command -v simpleFoam > /dev/null 2>&1; then
        pstage()  # pstage <dir> <named|alias>
        {
            rm -rf "$1"; cp -r "$PROX" "$1"; rm -rf "$1"/[1-9]* "$1"/log.* "$1"/postProcessing
            [ -d "$1/0.orig" ] && { rm -rf "$1/0"; cp -r "$1/0.orig" "$1/0"; }
            python3 - "$1" "$2" "$N" <<'PY'
import re, sys
d, mode, n = sys.argv[1], sys.argv[2], sys.argv[3]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*\n\}', 'functions\n{\n}', s, flags=re.S)
s = re.sub(r'endTime\s+\S+;', 'endTime         %s;' % n, s)
s = re.sub(r'writeInterval\s+\S+;', 'writeInterval   %s;' % n, s)
s = re.sub(r'writeControl\s+\S+;', 'writeControl    timeStep;', s)
s = re.sub(r'writeFormat\s+\S+;', 'writeFormat     ascii;', s)
s = re.sub(r'writePrecision\s+\S+;', 'writePrecision  15;', s)
s = re.sub(r'writeCompression\s+\S+;', 'writeCompression off;', s)
open(c, 'w').write(s)
f = d + '/system/fvSolution'; s = open(f).read()
s = re.sub(r'residualControl\s*\{[^}]*\}', 'residualControl { }', s, flags=re.S)
s = re.sub(r'tolerance\s+[0-9.eE+-]+;', 'tolerance       1e-12;', s)
s = re.sub(r'relTol\s+[0-9.eE+-]+;', 'relTol          0;', s)
open(f, 'w').write(s)
g = d + '/system/fvSchemes'; s = open(g).read()
# linearUpwindV so the arm exercises the scheme the old regex could not see. gradSchemes grad(U) is
# UNTOUCHED in both arms, so divDevReff and the closures are identical and only the deferred correction
# moves. `unlimGrad` is an alias entry, resolved by schemesLookup exactly like any other name.
s = re.sub(r'div\(phi,U\)\s*[^;]*;',
           'div(phi,U)      bounded Gauss linearUpwindV %s;' % ('grad(U)' if mode == 'named' else 'unlimGrad'), s)
s = re.sub(r'(gradSchemes\s*\{)', r'\1\n    unlimGrad       Gauss linear;', s)
open(g, 'w').write(s)
PY
        }
        pstage "$W/of_named" named; pstage "$W/of_alias" alias; pstage "$W/br" named
        ( cd "$W/of_named" && simpleFoam > run.log 2>&1 ) || { echo "FAIL: OpenFOAM (named)"; exit 1; }
        ( cd "$W/of_alias" && simpleFoam > run.log 2>&1 ) || { echo "FAIL: OpenFOAM (alias)"; exit 1; }
        ( cd "$W/br" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" "$W/br" > run.log 2>&1 ) \
            || { echo "FAIL: brae crashed on the proximity arm"; tail -10 "$W/br/run.log"; exit 1; }
        python3 - "$W" "$N" "$REL_MIN" "$MARGIN" <<'PY'
import gzip, os, re, sys
import numpy as np
W, n, relmin, margin = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
def rd(p):
    if os.path.exists(p):         s = open(p, 'rb').read()
    elif os.path.exists(p+'.gz'): s = gzip.open(p+'.gz', 'rb').read()
    else: return None
    s = s.decode('utf-8', 'replace'); t = s[s.index('internalField'):]
    m = re.match(r'internalField\s+nonuniform\s+List<(\w+)>\s*\n?\s*(\d+)\s*\(', t)
    if not m: return None
    kind, cnt = m.group(1), int(m.group(2)); b = t[m.end():]; b = b[:b.index('\n)')]
    if kind == 'scalar': return np.fromstring(b.replace('\n', ' '), sep=' ')
    return np.fromstring(b.replace('(', ' ').replace(')', ' ').replace('\n', ' '), sep=' ').reshape(cnt, 3)
def rel(a, b): return float(np.linalg.norm(a-b) / max(np.linalg.norm(a), 1e-300))
rc = 0; decided = 0
for f in ('U', 'p', 'k', 'epsilon', 'nut'):
    a, b, c = (rd(os.path.join(W, d, n, f)) for d in ('of_named', 'of_alias', 'br'))
    if a is None or b is None or c is None: continue
    sep = rel(a, b)
    if sep < relmin:
        print('  %-8s the two oracles agree to %.3e -- the named gradient does not reach it' % (f, sep)); continue
    dn, da = rel(a, c), rel(b, c)
    if min(dn, da) >= sep:
        print('  %-8s INCONCLUSIVE: brae %.3e / %.3e from oracles only %.3e apart' % (f, dn, da, sep)); continue
    decided += 1
    ok = da > margin * dn
    print('  %-70s %s' % ('ARM 4  %-8s brae is %.3e from NAMED, %.3e from the alias (sep %.3e)'
                          % (f, dn, da, sep), 'ok' if ok else 'FAIL'))
    if not ok: rc = 1
print('  %-70s %s' % ('CONTROL  at least one field is decidable', 'ok' if decided else 'FAIL'))
sys.exit(rc if decided else 1)
PY
        [ $? -eq 0 ] || fail=1
    else
        echo "  ARM 4    skipped: simpleFoam not on PATH"
    fi
else
    echo "  ARM 4    skipped: no OpenFOAM or no proximity fixture"
fi

[ $fail -eq 0 ] && echo "PASS: linearUpwind, linearUpwindV and LUST all resolve the gradient they name"
exit $fail
