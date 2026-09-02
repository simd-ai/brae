#!/usr/bin/env bash
# kOmegaSST ON THE HOST MIRROR takes its coefficients, its Prt and its omega relaxation FROM THE CASE.
#
# The SST branch of rhoSimpleFoam_cpp.cu took all three from StepInput fields that nothing assigned --
# sstCoeffs{} (model defaults), Prt = 1.0, relaxOmega = 1.0 -- while the kEpsilon branch beside it read
# the field set's keCoeffs and Prt. Invisible on every fixture: none names kOmegaSSTCoeffs or Prt (so the
# defaults equalled OpenFOAM's), and every SST gate compares at CONVERGENCE, where a relaxation factor
# changes nothing. And where the case names NO omega factor OpenFOAM's fvMatrix::relax() does nothing
# (fvMatrix.C:1250-1263), while brae's relaxMatrix at 1.0 still applied the dominance clamp.
#
# Host arm only: the device refuses kOmegaSST by name (rho_mirror_solver_vs_openfoam arm 10). Every
# linear solver is pinned to 1e-12/0 on every run so the TRAJECTORIES are comparable, and the comparison
# is taken at N iterations, mid-transient -- the relaxation arms have no converged signature at all.
#
#   ARM 1  kOmegaSSTCoeffs { betaStar 0.1; a1 0.4; gamma1 0.6; Prt 0.85; }  -> brae matches OpenFOAM
#   ARM 2  equations { omega 0.4; }                                          -> brae matches OpenFOAM
#   ARM 3  no omega factor named at all                                      -> brae matches OpenFOAM
#   CONTROLS  OpenFOAM's own answers for each arm must differ from its default run by >= 100x the bound.
#
# TAKEN AT ONE ITERATION, on purpose. The closure runs last in the iteration, so every coefficient,
# Prt (through validate()'s alphat) and the omega factor already decide the fields written at t=1 --
# and at t=1 the host mirror sits on OpenFOAM at the floor (omega/T/U ~7e-13) except for a 2e-06
# residual in k that is a separate, queued lead (wall rows, both models); by t=10 that residual has
# grown into a 1e-03 trajectory drift on this model, which would swallow the arms. Measured at t=1:
#   fixed       every arm: omega 5.5e-13..9.1e-13, T 7.4e-13, U 5.8e-13, k 1.9e-06 (the baseline)
#   controls    OpenFOAM's own answers move by 1e5..1e8 x the omega bound on each arm
#   fail-proof  step plumbing reverted (model defaults, Prt 1.0, omega factor 1.0, clamp forced): arm 1
#               omega 5.5e-01 / k 3.0e-02, arm 2 omega 6.1e-01 / k 3.5e-02. Arm 3 PASSED in that
#               fail-proof: relaxMatrix's dominance clamp at 1.0 is inert on rhoSST at t=1 (the omega
#               diagonal is already dominant), so the "no factor -> no relax()" half of the fix stands on
#               fvMatrix.C:1250-1263 rather than on a difference this case can show. The arm still
#               pins the reading path: a factor the case does not name must not be invented.
# The coefficient arm also found that brae handed the wall functions the MODEL's betaStar as their
# Cmu (OpenFOAM's read wallCoeffs_.Cmu(), default 0.09): with betaStar 0.1 it read omega 2.5e-02 at
# t=1, all in the wall rows. Fixed with KOmegaSSTCoeffs::CmuWall (and KEpsilonCoeffs::CmuWall).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/brae_rhoSimpleFoam"
SRC="$ROOT/validation/rhoSST"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
N=${N:-1}
K_BOUND=${K_BOUND:-1e-05}; OMEGA_BOUND=${OMEGA_BOUND:-1e-10}; T_BOUND=${T_BOUND:-1e-10}; U_BOUND=${U_BOUND:-1e-10}
CONTROL_RATIO=100

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }
command -v blockMesh     > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-70s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

# One meshed base; every variant is a copy of it.
cp -r "$SRC" "$W/base"; rm -rf "$W"/base/[1-9]* "$W"/base/0; cp -r "$W/base/0.orig" "$W/base/0"
( cd "$W/base" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh on rhoSST"; exit 1; }

stage()   # $1 dst  $2 coeffs block ("" = none)  $3 omega factor ("" = keep 0.7, "none" = remove the entry)
{
    COEFFS="$2" OMEGA="$3" python3 - "$W/base" "$1" "$N" <<'PYEOF'
import os, re, shutil, sys
src, dst, iters = sys.argv[1], sys.argv[2], sys.argv[3]
shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst)
c = os.path.join(dst, 'system/controlDict'); s = open(c).read()
for k, v in [('writeFormat', 'ascii'), ('writePrecision', '15'), ('endTime', iters), ('writeInterval', iters),
             ('writeControl', 'timeStep'), ('startFrom', 'startTime'), ('startTime', '0'), ('deltaT', '1')]:
    s = re.sub(r'\b%s\s+[^;]*;' % k, '%s %s;' % (k, v), s)
open(c, 'w').write(s)
f = os.path.join(dst, 'system/fvSolution'); s = open(f).read()
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { }', s)
s = re.sub(r'tolerance\s+[0-9.eE+-]+;', 'tolerance 1e-12;', s)
s = re.sub(r'relTol\s+[0-9.eE+-]+;', 'relTol 0;', s)
om = os.environ['OMEGA']
if om == 'none':
    s, n = re.subn(r'\bomega\s+[0-9.]+\s*;', '', s); assert n == 1
elif om:
    s, n = re.subn(r'\bomega\s+[0-9.]+\s*;', 'omega %s;' % om, s); assert n == 1
open(f, 'w').write(s)
co = os.environ['COEFFS']
if co:
    t = os.path.join(dst, 'constant/turbulenceProperties'); s = open(t).read()
    s, n = re.subn(r'(RAS\s*\{)', r'\1 kOmegaSSTCoeffs { %s } ' % co, s, count=1); assert n == 1
    open(t, 'w').write(s)
PYEOF
}
COEFFS='betaStar 0.1; a1 0.4; gamma1 0.6; Prt 0.85;'
stage "$W/of_def"  ""        ""
stage "$W/of_co"   "$COEFFS" ""
stage "$W/of_om"   ""        "0.4"
stage "$W/of_none" ""        "none"
for c in of_def of_co of_om of_none; do
    ( cd "$W/$c" && rhoSimpleFoam > of.log 2>&1 ) || { tail -5 "$W/$c/of.log"; echo "FAIL: OpenFOAM did not run ($c)"; exit 1; }
done
stage "$W/br_co"   "$COEFFS" ""
stage "$W/br_om"   ""        "0.4"
stage "$W/br_none" ""        "none"
for c in br_co br_om br_none; do
    ( cd "$W/$c" && BRAE_RHOSIMPLEFOAM_MIRROR=1 "$BIN" -case "$W/$c" > run.log 2>&1 ) \
        || { tail -5 "$W/$c/run.log"; echo "FAIL: the host mirror did not run ($c)"; exit 1; }
done
grep -q "kOmegaSSTCoeffs (case): betaStar=0.1 a1=0.4 gamma1=0.6 beta1=0.075 Prt=0.85" "$W/br_co/run.log" \
    && say "the case's kOmegaSSTCoeffs and Prt are read and printed" ok \
    || { grep "kOmegaSSTCoeffs" "$W/br_co/run.log"; say "the case's kOmegaSSTCoeffs and Prt are read and printed" FAIL; }

W="$W" N="$N" K_BOUND="$K_BOUND" OMEGA_BOUND="$OMEGA_BOUND" T_BOUND="$T_BOUND" U_BOUND="$U_BOUND" CONTROL_RATIO="$CONTROL_RATIO" \
python3 - <<'PYEOF' || fail=1
import os, re, sys
import numpy as np
W, N = os.environ['W'], os.environ['N']
def read(p):
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n?(\d+)\s*\n\(\n(.*?)\n\)\s*;', s, re.S)
    if m.group(1) == 'scalar':
        return np.array([float(x) for x in m.group(3).split()])
    return np.array([[float(c) for c in v.split()] for v in re.findall(r'\(([^)]*)\)', m.group(3))])
def rel(a, b, f):
    x = read(os.path.join(W, a, N, f)); y = read(os.path.join(W, b, N, f))
    return float(np.linalg.norm(x - y) / np.linalg.norm(y))
bounds = {'k': 'K_BOUND', 'omega': 'OMEGA_BOUND', 'T': 'T_BOUND', 'U': 'U_BOUND'}
ratio = float(os.environ['CONTROL_RATIO'])
ok = True
for arm, tag in (('co', 'coefficients + Prt'), ('om', 'omega 0.4'), ('none', 'no omega factor')):
    best = max((rel('of_' + arm, 'of_def', f) / float(os.environ[k]), f) for f, k in bounds.items())
    good = best[0] > ratio
    print('     control: OpenFOAM %-18s vs default differ by %.0fx the %s bound   %s' % (tag, best[0], best[1], 'ok' if good else 'FAIL (inert)'))
    ok = ok and good
    for f, k in bounds.items():
        r = rel('br_' + arm, 'of_' + arm, f); b = float(os.environ[k]); good = r < b
        print('     %-18s %-5s brae vs OpenFOAM %.4e   (bound %.1e)   %s' % (tag, f, r, b, 'ok' if good else 'FAIL'))
        ok = ok and good
sys.exit(0 if ok else 1)
PYEOF
say "the SST coefficients, Prt and omega relaxation reach the closure from the case" "$([ $fail = 0 ] && echo ok || echo FAIL)"
[ $fail = 0 ] && echo PASS || { echo FAIL; exit 1; }
