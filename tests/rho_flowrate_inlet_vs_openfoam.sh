#!/usr/bin/env bash
# flowRateInletVelocity ON THE DEVICE: the value the MOMENTUM ASSEMBLY sees, not just the one written.
#
# OpenFOAM's flowRateInletVelocity computes avgU = -flowRate/gSum(rho*magSf) and then does
# `operator==(avgU*n)` (flowRateInletVelocityFvPatchVectorField.C:194-196) -- an ASSIGNMENT to the patch
# VALUE, inside updateCoeffs, which fvMatrix.C:396 runs while constructing the momentum matrix. So
# OpenFOAM's fvc::grad(U) sees the value updateCoeffs just wrote.
#
# THE DEVICE ARM KEPT TWO COPIES OF THAT VALUE AND UPDATED ONE. deviceUpdateFlowRateInlet wrote
# dbU.comp[k].refValue; the driver also keeps f.UxBnd/UyBnd/UzBnd, which rhoUEqn hands to
# deviceDivDevReff as UbStored and which fvc::grad(U) inside dev2(T(grad(U))) reads directly. Only the
# refValue moved, so the assembly differentiated against 0/U's SEED until the post-solve refresh --
# and the written fields were right, because the velocity correction refreshes the arrays before the
# write. That is why this was invisible to every field-vs-field gate in the suite.
#
# WHY THE TOLERANCES ARE PINNED AND WHY BRAE_U_SOLVER=ofOrder. At the case's own relTol two solvers stop
# at different iterates and the difference swamps everything measured here; the CUDA arm also substitutes
# a multicolour Gauss-Seidel smoothSolver for momentum by default. Both arms assert from their own log
# that no substitution ran.
#
#   ARM 1  massFlowRate,       device + host vs OpenFOAM, t=1 and t=20   -> the round-off floor
#   ARM 2  volumetricFlowRate, device + host vs OpenFOAM, t=1 and t=20   -> the round-off floor
#   ARM 3  `rho none;` on a massFlowRate: OpenFOAM's volumetric branch   -> inlet 58.85, not 50.6878
#   ARM 4  SEED SWEEP on ARM 1: 0/U's `value` is a seed OpenFOAM overwrites, so the answer must not
#          depend on it. Seeds 10 and 50 both at the floor; the seed that already EQUALS the computed
#          value is run too, as the arm that cannot fail -- if it is the only one passing, the defect
#          is back.
#   ARM 5  CONTROL that the comparison is live: OpenFOAM's OWN t=1 answers at seed 10 and seed 50 must
#          differ by far more than the bound (createFields builds phi from the seeded U), so the arms
#          are genuinely distinguishable rather than trivially equal.
#   ARM 6  REFUSAL: `rho <someOtherField>;` names a density field brae does not look up -> refuse.
#
# FAIL-PROOF, measured in-session with each fix reverted:
#   stale UbStored           ARM 1 device U 1.212e-05 at t=1 (seed 50), 1.511e-04 at seed 10,
#                            4.624e-07 at t=20. Host 5.67e-13 throughout, which is what made it a
#                            device-only defect. On validation/rhoTI the same defect read U 8.945e-06.
#   volumetric never updated ARM 2 device U 4.328e-03 at t=1 and 4.823e-02 at t=20, inlet frozen at the
#                            seed 50 against OpenFOAM's 50.687834608 -- bcCategory() reports 9 for the
#                            mass form and a plain fixedValue 1 for the volumetric one, so the driver
#                            built no flow-rate mask for it and never called the updater at all.
#   `rho` unparsed           ARM 3 BOTH arms read 50.687834608 where OpenFOAM writes 58.85: the entry
#                            was silently dropped and the mass form ran. A 16% inlet error, no message.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/brae_rhoSimpleFoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
N=${N:-20}
FLOOR=${FLOOR:-1e-11}
CONTROL_RATIO=${CONTROL_RATIO:-1e3}

[ -x "$BIN" ] || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$ROOT/validation/rhoSST" ] || { echo "SKIP: fixture rhoSST missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }
command -v blockMesh     > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-72s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

# rhoSST, whose U inlet is a plain `fixedValue`, mutated one line at a time. Using it rather than
# validation/rhoTI keeps the k and omega inlets plain too, so nothing from the turbulence closure can
# reach these numbers -- rhoTI's own turbulent inlets carry a separate, open gap.
cp -r "$ROOT/validation/rhoSST" "$W/base"
rm -rf "$W"/base/[1-9]* "$W"/base/0; cp -r "$W/base/0.orig" "$W/base/0"
( cd "$W/base" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh"; exit 1; }

stage()   # $1 dst  $2 the inlet entry for 0/U  $3 endTime
{
    INLET="$2" python3 - "$W/base" "$1" "$3" <<'PYEOF'
import os, re, shutil, sys
src, dst, iters = sys.argv[1], sys.argv[2], sys.argv[3]
shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst)
c = os.path.join(dst, 'system/controlDict'); s = open(c).read()
for k, v in [('writeFormat', 'ascii'), ('writePrecision', '15'), ('endTime', iters), ('writeInterval', '1'),
             ('writeControl', 'timeStep'), ('startFrom', 'startTime'), ('startTime', '0'), ('deltaT', '1')]:
    s = re.sub(r'\b%s\s+[^;]*;' % k, '%s %s;' % (k, v), s)
s = re.sub(r'\bfunctions\s*\{.*\}\s*$', '', s, flags=re.S)
open(c, 'w').write(s)
f = os.path.join(dst, 'system/fvSolution'); s = open(f).read()
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { }', s)
s = re.sub(r'tolerance\s+[0-9.eE+-]+;', 'tolerance 1e-14;', s)
s = re.sub(r'relTol\s+[0-9.eE+-]+;', 'relTol 0;', s)
open(f, 'w').write(s)
u = os.path.join(dst, '0/U'); s = open(u).read()
s, n = re.subn(r'\n(\s*)inlet\s*\{[^}]*\}', '\n\\1inlet { %s }' % os.environ['INLET'], s, count=1)
assert n == 1, 'the 0/U inlet mutation did not apply'
open(u, 'w').write(s)
PYEOF
}

MASS='type flowRateInletVelocity; massFlowRate constant 0.5885; rhoInlet 1.0; value uniform (50 0 0);'
run()   # $1 tag  $2 inlet  $3 endTime  $4 arms ("of host cuda")
{
    for a in $4; do
        stage "$W/$1_$a" "$2" "$3"
        case $a in
          of)   ( cd "$W/$1_$a" && rhoSimpleFoam > log 2>&1 ) || true ;;   # some variants diverge later; t=1 is written
          host) ( cd "$W/$1_$a" && BRAE_U_SOLVER=ofOrder BRAE_RHOSIMPLEFOAM_MIRROR=1 \
                    "$BIN" -case "$W/$1_$a" > log 2>&1 ) || true ;;
          cuda) ( cd "$W/$1_$a" && BRAE_U_SOLVER=ofOrder BRAE_RHOSIMPLEFOAM_MIRROR=cuda \
                    "$BIN" -case "$W/$1_$a" > log 2>&1 ) || true ;;
        esac
    done
}

run mass "$MASS"                                                                      "$N"  "of host cuda"
run vol  'type flowRateInletVelocity; volumetricFlowRate constant 0.506878346077879; value uniform (50 0 0);' "$N" "of host cuda"
run none 'type flowRateInletVelocity; massFlowRate constant 0.5885; rho none; value uniform (50 0 0);'        1    "of host cuda"
run s10  'type flowRateInletVelocity; massFlowRate constant 0.5885; rhoInlet 1.0; value uniform (10 0 0);'    1    "of cuda"
run sx   'type flowRateInletVelocity; massFlowRate constant 0.5885; rhoInlet 1.0; value uniform (50.687834607787899 0 0);' 1 "of cuda"

for c in mass_cuda mass_host vol_cuda vol_host none_cuda s10_cuda sx_cuda; do
    [ -d "$W/$c/1" ] || { say "$c produced no output" FAIL; }
    grep -q "solvers/U solver" "$W/$c/log" && say "$c: no momentum-solver substitution" FAIL \
                                           || say "$c: no momentum-solver substitution" ok
done

W="$W" N="$N" FLOOR="$FLOOR" CONTROL_RATIO="$CONTROL_RATIO" python3 - <<'PYEOF' || fail=1
import os, re, sys
import numpy as np
W, N = os.environ['W'], os.environ['N']
FLOOR, RATIO = float(os.environ['FLOOR']), float(os.environ['CONTROL_RATIO'])

def read(p):
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n?(\d+)\s*\n\(\n(.*?)\n\)\s*;', s, re.S)
    if not m:
        u = re.search(r'internalField\s+uniform\s+(\(.*?\)|[-+0-9.eE]+)\s*;', s); v = u.group(1)
        return np.array([float(x) for x in v.strip('()').split()]) if v.startswith('(') else np.array([float(v)])
    if m.group(1) == 'scalar':
        return np.array([float(x) for x in m.group(3).split()])
    return np.array([[float(c) for c in v.split()] for v in re.findall(r'\(([^)]*)\)', m.group(3))])

def rel(a, b, t, f):
    x, y = read(os.path.join(W, a, t, f)), read(os.path.join(W, b, t, f))
    return float(np.linalg.norm(x - y) / np.linalg.norm(y))

def inletUx(case, t):
    s = open(os.path.join(W, case, t, 'U')).read()
    b = s[s.find('boundaryField'):]
    blk = re.search(r'\n    inlet\s*\n    \{(.*?)\n    \}', b, re.S).group(1)
    u = re.search(r'value\s+uniform\s+\(([^)]*)\)', blk)
    if u: return float(u.group(1).split()[0])
    v = re.search(r'value\s+nonuniform\s+List<vector>\s*\n?\d+\s*\n?\(\s*\(([^)]*)\)', blk, re.S)
    return float(v.group(1).split()[0])

ok = True
FIELDS = ('U', 'p', 'T', 'k', 'omega', 'nut')

# ARMS 1 and 2 -- both flow-rate forms, both arms, at the floor, at t=1 AND t=N.
# t=N matters on its own: a fix that only refreshed the value at construction still passes t=1.
for tag, label in (('mass', 'massFlowRate      '), ('vol', 'volumetricFlowRate')):
    for arm in ('cuda', 'host'):
        for t in ('1', N):
            worst, wf = max(((rel('%s_%s' % (tag, arm), '%s_of' % tag, t, f), f) for f in FIELDS))
            good = worst < FLOOR
            print('     %s %-4s t=%-3s worst field %-6s %.4e   (bound %.1e)   %s'
                  % (label, arm, t, wf, worst, FLOOR, 'ok' if good else 'FAIL'))
            ok = ok and good

# ARM 3 -- `rho none` takes OpenFOAM's volumetric branch: 0.5885/0.01 = 58.85, not 0.5885/(rho*0.01).
for arm in ('cuda', 'host'):
    a, b = inletUx('none_%s' % arm, '1'), inletUx('none_of', '1')
    good = abs(a - b) / abs(b) < 1e-12
    print('     rho none: %-4s inlet Ux %.9f  vs OpenFOAM %.9f            %s'
          % (arm, a, b, 'ok' if good else 'FAIL'))
    ok = ok and good
    worst, wf = max(((rel('none_%s' % arm, 'none_of', '1', f), f) for f in FIELDS))
    good = worst < FLOOR
    print('     rho none: %-4s t=1   worst field %-6s %.4e   (bound %.1e)   %s'
          % (arm, wf, worst, FLOOR, 'ok' if good else 'FAIL'))
    ok = ok and good

# ARM 4 -- the answer must not depend on 0/U's seed, which OpenFOAM overwrites before the assembly.
for tag, note in (('s10', 'seed 10   (far from the computed value)'),
                  ('sx',  'seed exact (the arm that cannot fail)  ')):
    worst, wf = max(((rel('%s_cuda' % tag, '%s_of' % tag, '1', f), f) for f in FIELDS))
    good = worst < FLOOR
    print('     %s worst field %-6s %.4e   (bound %.1e)   %s'
          % (note, wf, worst, FLOOR, 'ok' if good else 'FAIL'))
    ok = ok and good

# ARM 5 -- CONTROL: the seed is a live variable in OpenFOAM's own answer, so the arms above are
# genuinely distinguishable and are not passing because every run is the same run.
d = max(rel('s10_of', 'mass_of', '1', f) / FLOOR for f in ('U', 'p'))
good = d > RATIO
print('     control: OpenFOAM\'s own seed-10 and seed-50 answers differ by %.0e x the bound   %s'
      % (d, 'ok' if good else 'FAIL (the seed is inert; ARM 4 proves nothing)'))
ok = ok and good
sys.exit(0 if ok else 1)
PYEOF

# ARM 6 -- a density field brae does not look up must be refused BY NAME, not run against rho.
stage "$W/badrho" 'type flowRateInletVelocity; massFlowRate constant 0.5885; rho rhoLiquid; value uniform (50 0 0);' 1
out=$( cd "$W/badrho" && BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BIN" -case "$W/badrho" 2>&1 || true )
echo "$out" | grep -q "rhoLiquid" && ! [ -d "$W/badrho/1" ] \
    && say "an unsupported flowRateInletVelocity 'rho' field is refused by name" ok \
    || { echo "$out" | tail -3; say "an unsupported flowRateInletVelocity 'rho' field is refused by name" FAIL; }

say "flowRateInletVelocity reaches the momentum assembly, in both forms" "$([ $fail = 0 ] && echo ok || echo FAIL)"
exit $fail
