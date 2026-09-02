#!/usr/bin/env bash
# turbulence->validate() AT CONSTRUCTION, both models, against OpenFOAM's FIRST iteration.
#
# eddyViscosity::validate() runs correctNut() before the first solve: the interior nut from the model
# (Cmu*k^2/epsilon; a1*k/max(a1*omega, b1*F2*sqrt(S2))), nut.correctBoundaryConditions() -- which
# evaluates nutkWallFunction from the initial k -- and EddyDiffusivity's alphat = rho*nut/Prt with ITS
# boundary, wall functions included. brae's createFields ran the kEpsilon interior only: kOmegaSST
# entered its first momentum solve on the case file's nut and alphat, and on either model the wall nut
# stayed at the file's `uniform 0`, so the first momentum matrix carried mu at the wall instead of
# mu + rho*nut_w. The kOmegaSST branch of the step also returned before the alphat wall-function pass.
#
# NONE of that is visible at convergence, which is where every end-to-end gate compared -- and the
# mirror's kOmegaSST had no end-to-end gate at all (ctl/ti/pm/sst all run the legacy binary). So this
# compares the FIRST iteration, linear solvers pinned to 1e-12/0, on the plain fixtures.
#
# Measured, host mirror vs OpenFOAM at t=1 -- before: rhoKE U 2.1e-03 / p 4.6e-04 / nut 4.8e-04;
# rhoSST nut 1.6e-01 / omega 1.2e-01 / U 1.7e-03 / alphat wall faces 1.0 (identically zero).
# After: p 2.9e-12, T 7.4e-13, U 5.8e-13, epsilon 1.1e-12 / omega 7.3e-13 on both; k 1.3e-06 (KE) and
# 2.0e-06 (SST), a residual concentrated in the wall rows and queued as its own lead. The k bound is
# set at that residual's order, the rest at ~100x the floor; the fail-proof (the old kEpsilon-only,
# interior-only block) fails every field on rhoSST and U/p on rhoKE by 1e3..1e8 x.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/brae_rhoSimpleFoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
P_BOUND=1e-10; T_BOUND=1e-10; U_BOUND=1e-10; SECOND_BOUND=1e-10; K_BOUND=1e-05; NUT_BOUND=1e-05

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }
command -v blockMesh     > /dev/null 2>&1 || { echo "SKIP: blockMesh not on PATH"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-66s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

stage()   # $1 fixture  $2 dst -- one iteration, tight solvers, residualControl off
{
    rm -rf "$2"; cp -r "$ROOT/validation/$1" "$2"; rm -rf "$2"/[1-9]* "$2"/0; cp -r "$2/0.orig" "$2/0"
    ( cd "$2" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh on $1"; exit 1; }
    python3 - "$2" <<'PYEOF'
import os, re, sys
d = sys.argv[1]
c = os.path.join(d, 'system/controlDict'); s = open(c).read()
for k, v in [('writeFormat', 'ascii'), ('writePrecision', '15'), ('endTime', '1'), ('writeInterval', '1'),
             ('writeControl', 'timeStep'), ('startFrom', 'startTime'), ('startTime', '0'), ('deltaT', '1')]:
    s = re.sub(r'\b%s\s+[^;]*;' % k, '%s %s;' % (k, v), s)
open(c, 'w').write(s)
f = os.path.join(d, 'system/fvSolution'); s = open(f).read()
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { }', s)
s = re.sub(r'tolerance\s+[0-9.eE+-]+;', 'tolerance 1e-12;', s)
s = re.sub(r'relTol\s+[0-9.eE+-]+;', 'relTol 0;', s)
open(f, 'w').write(s)
PYEOF
}

for fx in rhoKE rhoSST; do
    [ -d "$ROOT/validation/$fx" ] || { say "fixture $fx missing" SKIP; continue; }
    second=$([ "$fx" = rhoKE ] && echo epsilon || echo omega)
    stage "$fx" "$W/of_$fx"
    ( cd "$W/of_$fx" && rhoSimpleFoam > of.log 2>&1 ) || { tail -5 "$W/of_$fx/of.log"; echo "FAIL: OpenFOAM did not run ($fx)"; exit 1; }
    arms="1"; [ "$fx" = rhoKE ] && arms="1 cuda"    # the device refuses kOmegaSST by name
    for arm in $arms; do
        stage "$fx" "$W/br_${fx}_$arm"
        ( cd "$W/br_${fx}_$arm" && BRAE_RHOSIMPLEFOAM_MIRROR=$arm "$BIN" -case "$W/br_${fx}_$arm" > run.log 2>&1 ) \
            || { tail -5 "$W/br_${fx}_$arm/run.log"; echo "FAIL: the mirror (arm $arm) did not run $fx"; exit 1; }
        BR="$W/br_${fx}_$arm/1" OF="$W/of_$fx/1" SECOND="$second" ARM="$arm" FX="$fx" \
        P_BOUND="$P_BOUND" T_BOUND="$T_BOUND" U_BOUND="$U_BOUND" SECOND_BOUND="$SECOND_BOUND" K_BOUND="$K_BOUND" NUT_BOUND="$NUT_BOUND" \
        python3 - <<'PYEOF' || fail=1
import os, re, sys
import numpy as np
def read(p):
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n?(\d+)\s*\n\(\n(.*?)\n\)\s*;', s, re.S)
    if m.group(1) == 'scalar':
        return np.array([float(x) for x in m.group(3).split()])
    return np.array([[float(c) for c in v.split()] for v in re.findall(r'\(([^)]*)\)', m.group(3))])
ok = True
sec = os.environ['SECOND']
for f, key in (('p', 'P_BOUND'), ('T', 'T_BOUND'), ('U', 'U_BOUND'), (sec, 'SECOND_BOUND'), ('k', 'K_BOUND'), ('nut', 'NUT_BOUND'), ('alphat', 'NUT_BOUND')):
    a = read(os.path.join(os.environ['BR'], f)); b = read(os.path.join(os.environ['OF'], f))
    r = float(np.linalg.norm(a - b) / np.linalg.norm(b)); bnd = float(os.environ[key]); good = r < bnd
    print('     %-6s arm %-4s %-7s vs OpenFOAM at t=1 %.4e   (bound %.1e)   %s' % (os.environ['FX'], os.environ['ARM'], f, r, bnd, 'ok' if good else 'FAIL'))
    ok = ok and good
sys.exit(0 if ok else 1)
PYEOF
        say "$fx arm $arm: the first iteration sits on OpenFOAM's (validate() done as OpenFOAM does it)" "$([ $fail = 0 ] && echo ok || echo FAIL)"
    done
done
[ $fail = 0 ] && echo PASS || { echo FAIL; exit 1; }
