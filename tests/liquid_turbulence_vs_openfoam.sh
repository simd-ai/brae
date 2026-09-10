#!/usr/bin/env bash
# A TURBULENT liquid, end to end on the host mirror arm, against real OpenFOAM v2412 -- stage H3.5.
#
# What a liquid thermo changes inside a RAS closure is exactly one input: the laminar kinematic
# viscosity nu = mu(T)/rho(T), which for water falls ~30% between the 300 K inlet and the 350 K walls.
# It enters twice -- the closure's own diffusivities (DkEff = nut/sigmak + nu, and its SST twins), and
# every wall function, which builds y+ and the viscous branch from the WALL FACE's nu (nutkWallFunction,
# epsilonWallFunction, omegaWallFunction all call turbModel.nu(patchi)). Both reach the closure through
# thermoMuOf, the same accessor the momentum and energy equations use, so this stage needed no new code:
# the gate is the evidence that the accessor refactor (H3.1) made the path generic rather than a claim.
#
# BOTH CLOSURES, on purpose. squareBendLiq runs kEpsilon; this gate also runs kOmegaSST on the same
# fields (validation/liqBoxRAS ships 0.orig/epsilon AND 0.orig/omega), because nothing about a liquid is
# specific to one model and a liquid path proven on one closure only would be a patch for a tutorial.
# The fixture carries squareBendLiq's own boundary set: turbulentIntensityKineticEnergyInlet,
# turbulentMixingLengthDissipationRateInlet (FrequencyInlet for omega), kqRWallFunction,
# epsilonWallFunction / omegaWallFunction, nutkWallFunction, compressible::alphatWallFunction (Prt 0.85).
#
# FOUR ARMS: each closure after ONE iteration (the first evaluation of every property, before anything
# has fed back) and after 200 (residuals at ~1e-11, i.e. converged), every field and the wall-function
# patch values of nut and alphat, which are what the wall functions themselves write.
#
# NOT VACUOUS: OpenFOAM's rho must be a liquid's (~993 kg/m3); the case must be genuinely turbulent
# (max nut at least 10x the laminar nu, else the closure contributes nothing measurable); and the wall
# faces must be present in both outputs before their values are compared. Two fail-proofs, measured
# through this gate by editing rhoSimpleFoam_cpp.cu and re-running the kEpsilon arm at 200 iterations:
#     the closure's nu with mu frozen at 300 K (no T dependence):
#         k 1.19e-01, epsilon 2.55e-01, nut 3.38e-02, U 5.66e-03, T 2.38e-03
#     the WALL nu at the owner cell's T instead of the face's (wall functions blind to the wall):
#         k 7.02e-02, epsilon 1.42e-01, nut 2.60e-02, U 3.43e-03, T 1.57e-03
# against an implemented path at ~1e-12 on every field -- ten orders apart.
#
# Measured, OpenFOAM v2412:
#     kEpsilon  it 1   U 5.75e-13 p 3.90e-12 T 9.61e-13 rho 3.03e-13 k 1.54e-12 eps   9.00e-13 nut 7.29e-13
#     kEpsilon  it 200 U 6.70e-13 p 6.23e-12 T 9.26e-13 rho 2.95e-13 k 2.00e-12 eps   1.23e-12 nut 1.76e-12
#     kOmegaSST it 1   U 6.48e-13 p 5.04e-12 T 9.63e-13 rho 3.12e-13 k 1.38e-12 omega 7.59e-13 nut 5.25e-13
#     kOmegaSST it 200 U 8.02e-13 p 3.07e-12 T 9.23e-13 rho 2.93e-13 k 2.01e-12 omega 1.29e-12 nut 1.33e-12
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae_rhoSimpleFoam}"
SRC="$ROOT/validation/liqBoxRAS"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}

[ -x "$BRAE" ]     || { echo "SKIP: no brae binary at $BRAE"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0

# stage <dir> <RASModel> <endTime>. The fixture's dictionaries are single-line, so the edits are
# token-wise; an anchored sed silently rewrites a whole line of unrelated keys.
stage() {
    rm -rf "$1"; cp -r "$SRC" "$1"
    rm -rf "$1"/[1-9]* "$1"/0 "$1"/log.*
    cp -r "$1/0.orig" "$1/0"
    python3 - "$1" "$2" "$3" <<'PYEOF'
import re, sys
d, model, n = sys.argv[1:4]
p = d + '/constant/turbulenceProperties'
s = open(p).read()
s = re.sub(r'RASModel\s+\S+;', 'RASModel        %s;' % model, s)
open(p, 'w').write(s)
p = d + '/system/controlDict'
s = open(p).read()
s = re.sub(r'\bendTime\s+\S+;',       'endTime %s;' % n,       s)
s = re.sub(r'\bwriteInterval\s+\S+;', 'writeInterval %s;' % n, s)
open(p, 'w').write(s)
PYEOF
}

for MODEL in kEpsilon kOmegaSST; do
    for END in 1 200; do
        TAG="${MODEL}_${END}"
        stage "$W/of_$TAG" "$MODEL" "$END"
        ( cd "$W/of_$TAG" && blockMesh > log.blockMesh 2>&1 && rhoSimpleFoam > log.rhoSimpleFoam 2>&1 ) || {
            echo "FAIL: $TAG -- OpenFOAM did not run"; tail -20 "$W/of_$TAG/log.rhoSimpleFoam"; exit 1; }
        stage "$W/br_$TAG" "$MODEL" "$END"
        cp -r "$W/of_$TAG/constant/polyMesh" "$W/br_$TAG/constant/"
        BRAE_RHOSIMPLEFOAM_MIRROR=1 "$BRAE" -case "$W/br_$TAG" > "$W/br_$TAG/log.brae" 2>&1 || {
            echo "FAIL: $TAG -- brae did not run"; grep -v '^brae NOTICE' "$W/br_$TAG/log.brae" | tail -8
            fail=1; continue; }

        echo "== $MODEL, $END iteration(s) =="
        MODEL="$MODEL" python3 - "$W/br_$TAG/$END" "$W/of_$TAG/$END" <<'PYEOF' || fail=1
import math, os, re, sys

brae, of = sys.argv[1], sys.argv[2]
second = 'epsilon' if os.environ['MODEL'] == 'kEpsilon' else 'omega'
bad = 0

def say(what, verdict):
    global bad
    print('     %-58s %s' % (what, verdict))
    if verdict == 'FAIL': bad = 1

def report(what, got, bound):
    global bad
    ok = got < bound
    print('     %-58s %.6e   %s' % (what, got, 'ok' if ok else 'FAIL (bound %g)' % bound))
    if not ok: bad = 1

def text(path): return open(path).read()

def internal(path):
    s = text(path)
    m = re.search(r'internalField\s+nonuniform[^(]*\((.*?)\)\s*;', s, re.S)
    if not m: raise SystemExit('%s: internalField is not a nonuniform list' % path)
    return [float(x) for x in m.group(1).split()]

def internalVec(path):
    s = text(path)
    m = re.search(r'internalField\s+nonuniform[^(]*\(\s*(.*?)\s*\)\s*;\s*\n\s*boundaryField', s, re.S)
    return [float(x) for x in re.findall(r'-?[\d.]+(?:[eE][-+]?\d+)?', m.group(1))]

def patchValues(path, patch):
    # The `value` list of one patch. uniform values are expanded to nothing (returned as a 1-list) --
    # a wall function writes a nonuniform list once it has run, and the caller asserts the length.
    s = text(path)
    m = re.search(r'\b' + re.escape(patch) + r'\s*\{(.*?)\n\s*\}', s[s.index('boundaryField'):], re.S)
    if not m: return []
    body = m.group(1)
    v = re.search(r'\bvalue\s+nonuniform[^(]*\((.*?)\)\s*;', body, re.S)
    if v: return [float(x) for x in v.group(1).split()]
    v = re.search(r'\bvalue\s+uniform\s+([-\d.eE+]+)\s*;', body)
    return [float(v.group(1))] if v else []

def rel(a, b):
    n = sum((x - y) ** 2 for x, y in zip(a, b))
    d = sum(y * y for y in b)
    return math.sqrt(n / d) if d > 0 else math.sqrt(n)

rho = internal(of + '/rho')
say('OpenFOAM rho is a LIQUID (980..1010 kg/m3): %.2f..%.2f' % (min(rho), max(rho)),
    'ok' if 980.0 < min(rho) and max(rho) < 1010.0 else 'FAIL')
# Genuinely turbulent: water's nu is ~1e-6 m2/s here, so max(nut) must clear 1e-5 or the closure is
# contributing nothing a laminar run would not also produce.
nutOF = internal(of + '/nut')
say('the case is TURBULENT: max nut %.3e >= 10 x laminar nu' % max(nutOF),
    'ok' if max(nutOF) > 1e-5 else 'FAIL')

for f, bound in (('T', 1e-11), ('p', 1e-10), ('rho', 1e-11), ('k', 1e-11), (second, 1e-11),
                 ('nut', 1e-11), ('alphat', 1e-11)):
    report('%-7s vs OpenFOAM (L2 rel)' % f, rel(internal(brae + '/' + f), internal(of + '/' + f)), bound)
report('%-7s vs OpenFOAM (L2 rel)' % 'U', rel(internalVec(brae + '/U'), internalVec(of + '/U')), 1e-11)

# THE WALL FUNCTIONS' OWN OUTPUT: nut and alphat on the two walls, which is what nutkWallFunction and
# compressible::alphatWallFunction write, and what the wall-face nu reaches first.
for f in ('nut', 'alphat'):
    for patch in ('hotWall', 'coldWall'):
        b, o = patchValues(brae + '/' + f, patch), patchValues(of + '/' + f, patch)
        if len(o) < 2 or len(b) != len(o):
            say('%s on %s: both codes wrote the wall faces (%d, %d)' % (f, patch, len(b), len(o)), 'FAIL')
            continue
        report('%s on %s (%d faces, L2 rel)' % (f, patch, len(o)), rel(b, o), 1e-11)

sys.exit(bad)
PYEOF
    done
done

[ "$fail" -eq 0 ] && echo "PASSED" || echo "FAILED"
exit $fail
