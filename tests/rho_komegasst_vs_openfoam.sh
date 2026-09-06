#!/usr/bin/env bash
# The COMPRESSIBLE kOmegaSST closure against OpenFOAM's OWN kOmegaSST, instrumented.
#
# TWO INSTRUMENTS, and they are different things. tools/dumpPEqn is rhoSimpleFoam with writes added; it
# brackets the turbulence call so the model's inputs -- rho, mu, phi, U -- can be read as OpenFOAM had
# them. tools/dumpKOmegaSST is OpenFOAM's OWN kOmegaSSTBase.C with writes added and its equations
# untouched, registered as `kOmegaSSTDump`; it writes the model's internals -- divU, gradU, S2, GbyNu0, G,
# CDkOmega, F1, F23, both assembled systems before and after relax, and their off-diagonals. Running the
# first with the second selected gives both at the same iteration.
#
# WHY A MODEL INSTRUMENT AND NOT JUST THE OUTPUTS. A closure that is 3e-02 out in k cannot tell you which
# of a dozen terms did it. When this was done for kEpsilon it turned one unexplained gap into four named
# defects in an afternoon: a stale inletOutlet on U, k and epsilon contributing nothing to their own
# systems, turbulent inlets frozen at the case file's value, and the diffusion assembled orthogonally
# where the case said corrected.
#
# THE FIXTURE is sbMatched switched to kOmegaSST in place, rather than a second copy of a 112000-cell
# mesh. Every step of that switch is asserted below, because a gate that silently failed to switch the
# model would compare kEpsilon to kEpsilon and pass. omega is derived from the fixture's own k and
# epsilon by omega = epsilon/(Cmu*k), which is the definition, so the start state is the same turbulence.
#
# THE INLET IS NEUTRALISED here, as in the other closure gates: this measures the CLOSURE against
# OpenFOAM's own on OpenFOAM's own inputs, and sbMatched's flowRateInletVelocity is a separate component
# with its own gate (rho_ueqn_vs_openfoam, which runs the fixture's real inlet).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_rho_komegasst_cpp"
SRC="$ROOT/validation/sbMatched"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
ITERS=${ITERS:-1}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
cp -r "$SRC" "$W/case" || exit 1
rm -rf "$W"/case/[1-9]* "$W"/case/0 "$W"/case/processor* "$W"/case/log.*
[ -d "$W/case/constant/polyMesh" ] || { echo "SKIP: fixture ships no mesh"; exit 77; }
cp -r "$W/case/0.orig" "$W/case/0"

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
DUMP="$(command -v dumpPEqn || true)"
[ -z "$DUMP" ] && [ -n "${FOAM_USER_APPBIN:-}" ] && [ -x "$FOAM_USER_APPBIN/dumpPEqn" ] \
    && DUMP="$FOAM_USER_APPBIN/dumpPEqn"
[ -n "$DUMP" ] || { echo "SKIP: dumpPEqn not built -- (cd tools/dumpPEqn && wmake)"; exit 77; }
[ -f "${FOAM_USER_LIBBIN:-}/libdumpKOmegaSST.so" ] \
    || { echo "SKIP: libdumpKOmegaSST.so not built -- (cd tools/dumpKOmegaSST && wmake libso)"; exit 77; }

# ---- switch the fixture to kOmegaSST, asserting every step -------------------------------------------
sed -i 's/RASModel  *kEpsilon;/RASModel            kOmegaSSTDump;/' "$W/case/constant/turbulenceProperties"
grep -q "kOmegaSSTDump" "$W/case/constant/turbulenceProperties" \
    || { echo "FAIL: could not select the instrumented model"; exit 1; }

python3 - "$W/case" <<'PYEOF'
import re, sys, os
d = sys.argv[1]
# omega = epsilon/(Cmu*k), the definition. The fixture's k and epsilon are uniform, so this is exact.
k = open(os.path.join(d, '0/k')).read()
e = open(os.path.join(d, '0/epsilon')).read()
kv = float(re.search(r'internalField\s+uniform\s+([0-9.eE+-]+)', k).group(1))
ev = float(re.search(r'internalField\s+uniform\s+([0-9.eE+-]+)', e).group(1))
om = ev / (0.09 * kv)
s = e.replace('epsilon', 'omega')
s = re.sub(r'internalField\s+uniform\s+[0-9.eE+-]+', 'internalField   uniform %.15g' % om, s)
s = re.sub(r'\$internalField', 'uniform %.15g' % om, s)
s = s.replace('epsilonWallFunction', 'omegaWallFunction')
s = s.replace('turbulentMixingLengthDissipationRateInlet', 'turbulentMixingLengthFrequencyInlet')
s = re.sub(r'dimensions.*;', 'dimensions      [0 0 -1 0 0 0 0];', s)
open(os.path.join(d, '0/omega'), 'w').write(s)

# the solver, relaxation and scheme entries omega needs
f = os.path.join(d, 'system/fvSolution')
s = open(f).read()
s = s.replace('"(U|e|k|epsilon)"', '"(U|e|k|epsilon|omega)"')
s = s.replace('"(k|epsilon)" 1e-3;', '"(k|epsilon|omega)" 1e-3;')
s = re.sub(r'^(\s*)epsilon(\s+)0\.9;', r'\1epsilon\g<2>0.9;\n\1omega\g<2>0.9;', s, flags=re.M)
open(f, 'w').write(s)

f = os.path.join(d, 'system/fvSchemes')
s = open(f).read()
s = s.replace('div(phi,epsilon)', 'div(phi,omega)')
# kOmegaSST needs a wall-distance method; the kEpsilon fixture has none.
if 'wallDist' not in s:
    s += '\nwallDist\n{\n    method meshWave;\n}\n'
open(f, 'w').write(s)

c = os.path.join(d, 'system/controlDict')
s = open(c).read()
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
s = re.sub(r'^writeFormat .*',    'writeFormat     ascii;',    s, flags=re.M)
s = re.sub(r'^writePrecision .*', 'writePrecision  15;',       s, flags=re.M)
s = re.sub(r'^endTime .*',        'endTime         1;',        s, flags=re.M)
s = re.sub(r'^writeInterval .*',  'writeInterval   1;',        s, flags=re.M)
s = re.sub(r'^writeControl .*',   'writeControl    timeStep;', s, flags=re.M)
s += '\nlibs ("libdumpKOmegaSST.so");\n'
open(c, 'w').write(s)
print('omega uniform %.15g' % om)
PYEOF

[ -f "$W/case/0/omega" ] || { echo "FAIL: omega was not written"; exit 1; }
grep -q "omegaWallFunction" "$W/case/0/omega" \
    || { echo "FAIL: omega carries no wall function"; exit 1; }
grep -q "div(phi,omega)" "$W/case/fvSchemes" 2>/dev/null || grep -q "div(phi,omega)" "$W/case/system/fvSchemes" \
    || { echo "FAIL: no div(phi,omega) scheme"; exit 1; }

# The inlet, neutralised: this gate is about the closure. See the header.
python3 - "$W/case/0/U" <<'PYEOF'
import re, sys
p = sys.argv[1]
s = open(p).read()
s = re.sub(r'(inlet\s*\{)[^}]*\}',
           r'\1\n        type            fixedValue;\n        value           uniform (1 2 3);\n    }',
           s, count=1)
open(p, 'w').write(s)
PYEOF
grep -q "fixedValue" "$W/case/0/U" || { echo "FAIL: could not neutralise the inlet"; exit 1; }

( cd "$W/case" && BRAE_DUMP_STAGE_ITER="$ITERS" "$DUMP" > dump.log 2>&1 ) \
    || { echo "FAIL: dumpPEqn did not run"; tail -25 "$W/case/dump.log"; exit 1; }
[ -d "$W/case/$ITERS" ] \
    || { echo "FAIL: OpenFOAM wrote no $ITERS/"; tail -25 "$W/case/dump.log"; exit 1; }

# The MODEL's own dumps must be there: without them this would silently degrade to an outputs-only
# comparison, which is the thing the header says is not enough.
for fld in stage_sstDivU stage_sstGradU stage_sstS2 stage_sstGbyNu0 stage_sstG stage_sstCDkOmega \
           stage_sstF1 stage_sstF23 stage_sstKIn stage_sstOmegaIn stage_sstNutIn stage_sstU \
           stage_sstOmD stage_sstOmSrc stage_sstKD stage_sstKSrc \
           stage_sstKOut stage_sstOmegaOut stage_sstNutOut \
           stage_rhoTurb stage_muTurb stage_phiTurb; do
    [ -f "$W/case/$ITERS/$fld" ] \
        || { echo "FAIL: no $ITERS/$fld -- is libdumpKOmegaSST selected?"; tail -20 "$W/case/dump.log"; exit 1; }
done

"$BIN" "$W/case" "$ITERS"
