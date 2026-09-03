#!/usr/bin/env bash
# relaxationFactors `default` REACHES BOTH ARMS of the rhoSimpleFoam mirror, as OpenFOAM's predicate.
#
# DRAFT -- UNVERIFIED. Written against the tip's driver without a buildable worktree (see the item 8
# report): it has NOT been run, no floor has been measured here, and the bounds below are INHERITED
# from rho_mirror_solver_vs_openfoam (host) and rho_mirror_compare.py (CUDA), which were pinned at
# t=200 on the same fixture. Before registering: run it, read the floors at t=1..10, pin ~30x them,
# and record the fail-proof numbers in this header.
#
# OpenFOAM relaxes an equation iff solution::relaxEquation(name) -- eqnRelaxDict_.found(name) ||
# found("default") (solution.C:330-334) -- and a field iff relaxField(name), the same on fieldRelaxDict_
# (solution.C:320-327); the factor is the named entry else the default (solution.C:337-416). The
# mirror's buildStepInput read only the name, so a case relaxing through `default` ran UNRELAXED on
# both arms. Transient by nature: invisible at convergence, so every arm is taken at t=1..10 from 0/
# with every linear solver pinned to 1e-12/0/2000.
#
#   ARM A  fields { default 0.3; } equations { default 0.7; }         both arms match OpenFOAM
#   ARM B  fields { p 0.3; default 0.9; } equations { U 0.5; default 0.7; }
#          the NAMED entry wins over the default, as in OpenFOAM
#   ARM C  equations { default 0.7; } only: fields has no default, so p is NOT relaxed
#   CONTROLS  OpenFOAM's own answers: A vs NO relaxationFactors at all (what the unfixed mirror ran)
#             differ by >= CONTROL_RATIO x the bound; A vs C and A vs B likewise.
#   LOG    the mirror SAYS the factor it applies and where it came from.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILDDIR="${BUILD:-$ROOT/build}"
BIN="$BUILDDIR/brae_rhoSimpleFoam"
SRC="$ROOT/validation/rhoBox"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
N=${N:-10}
# Arms A and B, both mirror arms, measured at t=1..10 (max): p 3.0e-12, T 1.0e-12, U 2.0e-11, rho 2.6e-12.
# Bounds ~30x, the same for the host and the device arm -- they sit on the same floor here.
P_BOUND=${P_BOUND:-1e-10}; T_BOUND=${T_BOUND:-3e-11}; U_BOUND=${U_BOUND:-6e-10}; RHO_BOUND=${RHO_BOUND:-1e-10}
# Arm C leaves p UNRELAXED, and an unrelaxed pressure amplifies the linear-solver floor by ~1.8x per
# iteration on both arms alike (measured host p 2.8e-12 at t=1, 4.2e-12 at t=5, 6.3e-11 at t=10; a
# second run 1.8e-10 at t=10; host-vs-device 1.6e-11 at t=10). So arm C is taken to N_C=5, where the
# floor is still ~1e-11 (max over t=1..5: p 4.2e-12, U 1.3e-11, rho 4.2e-12), at bounds ~30x that.
N_C=${N_C:-5}
C_P_BOUND=${C_P_BOUND:-3e-10}; C_T_BOUND=${C_T_BOUND:-3e-11}; C_U_BOUND=${C_U_BOUND:-6e-10}; C_RHO_BOUND=${C_RHO_BOUND:-3e-10}
CONTROL_RATIO=${CONTROL_RATIO:-100}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-72s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

# One meshed base; every variant is a copy of it with its own relaxationFactors block.
cp -r "$SRC" "$W/base"; rm -rf "$W"/base/[1-9]* "$W"/base/0 "$W"/base/log.*
cp -r "$W/base/0.orig" "$W/base/0"
if [ -f "$W/base/system/blockMeshDict" ] && command -v blockMesh > /dev/null 2>&1; then
    ( cd "$W/base" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh on rhoBox"; exit 1; }
fi

stage()   # $1 dst  $2 relaxationFactors block ("none" = delete the whole entry)  $3 endTime (default N)
{
    RELAX="$2" python3 - "$W/base" "$1" "${3:-$N}" <<'RELAXSTAGE'
import os, re, shutil, sys
src, dst, n = sys.argv[1], sys.argv[2], sys.argv[3]
shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst)
c = os.path.join(dst, 'system/controlDict'); s = open(c).read()
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
for k, v in [('writeFormat', 'ascii'), ('writePrecision', '15'), ('endTime', n), ('writeInterval', '1'),
             ('writeControl', 'timeStep'), ('startFrom', 'startTime'), ('startTime', '0'), ('deltaT', '1')]:
    s = re.sub(r'\b%s\s+[^;]*;' % k, '%s %s;' % (k, v), s)
open(c, 'w').write(s)
f = os.path.join(dst, 'system/fvSolution'); s = open(f).read()
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { }', s)
s = re.sub(r'tolerance\s+[0-9.eE+-]+;', 'tolerance 1e-12;', s)
s = re.sub(r'relTol\s+[0-9.eE+-]+;', 'relTol 0;', s)
# rhoBox's entries carry no maxIter: append one after relTol so the pin check below has something to see.
s = re.sub(r'maxIter\s+[0-9]+;', 'maxIter 2000;', s)
s = re.sub(r'relTol 0;(?![^}]*maxIter)', 'relTol 0; maxIter 2000;', s)
# rhoBox writes the block on one line: relaxationFactors { fields { ... } equations { ... } }
s, k = re.subn(r'relaxationFactors\s*\{(?:[^{}]|\{[^{}]*\})*\}', '', s)
assert k == 1, 'expected exactly one relaxationFactors block, found %d' % k
blk = os.environ['RELAX']
if blk != 'none':
    s += '\nrelaxationFactors %s\n' % blk
open(f, 'w').write(s)
RELAXSTAGE
}
A='{ fields { default 0.3; } equations { default 0.7; } }'
B='{ fields { p 0.3; default 0.9; } equations { U 0.5; default 0.7; } }'
C='{ equations { default 0.7; } }'
# The UNRELAXED control stops at N_NONE: real OpenFOAM itself diverges on rhoBox without relaxation
# (measured: "Negative initial temperature T0: -25.43" at iteration 6), so its answer exists to t=5 only.
N_NONE=3
stage "$W/of_A" "$A";    stage "$W/of_B" "$B";    stage "$W/of_C" "$C";    stage "$W/of_none" none "$N_NONE"
for c in of_A of_B of_C of_none; do
    grep -q "maxIter 2000" "$W/$c/system/fvSolution" || { echo "FAIL: the solver pin did not apply ($c)"; exit 1; }
    ( cd "$W/$c" && rhoSimpleFoam > of.log 2>&1 ) || { tail -5 "$W/$c/of.log"; echo "FAIL: OpenFOAM did not run ($c)"; exit 1; }
    last=$N; [ "$c" = of_none ] && last=$N_NONE
    [ -d "$W/$c/$last" ] || { echo "FAIL: OpenFOAM wrote no $last/ ($c)"; exit 1; }
done
for arm in A B C; do
    eval "blk=\$$arm"
    stage "$W/host_$arm" "$blk"; stage "$W/cuda_$arm" "$blk"
    ( cd "$W/host_$arm" && BRAE_RHOSIMPLEFOAM_MIRROR=1    "$BIN" -case "$W/host_$arm" > run.log 2>&1 ) \
        || { tail -5 "$W/host_$arm/run.log"; echo "FAIL: the host mirror did not run (arm $arm)"; exit 1; }
    ( cd "$W/cuda_$arm" && BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BIN" -case "$W/cuda_$arm" > run.log 2>&1 ) \
        || { tail -5 "$W/cuda_$arm/run.log"; echo "FAIL: the CUDA mirror did not run (arm $arm)"; exit 1; }
done

# The log SAYS what it applies (buildStepInput prints under verbose, which the solver path is).
for side in host cuda; do
    grep -q "relaxation: equations U 0.7 (default) | h 0.7 (default) | p 0.7 (default)" "$W/${side}_A/run.log" \
        && grep -q "fields p 0.3 (default) | rho 0.3 (default)" "$W/${side}_A/run.log" \
        && say "arm A ($side): the log reports every factor as coming from default" ok \
        || { grep "relaxation:" "$W/${side}_A/run.log"; say "arm A ($side): the log reports every factor as coming from default" FAIL; }
    grep -q "relaxation: equations U 0.5 | h 0.7 (default)" "$W/${side}_B/run.log" \
        && grep -q "fields p 0.3 | rho 0.9 (default)" "$W/${side}_B/run.log" \
        && say "arm B ($side): the named entry is reported over the default" ok \
        || { grep "relaxation:" "$W/${side}_B/run.log"; say "arm B ($side): the named entry is reported over the default" FAIL; }
    grep -q "fields p none | rho none" "$W/${side}_C/run.log" \
        && say "arm C ($side): no field default -> the log reports p unrelaxed" ok \
        || { grep "relaxation:" "$W/${side}_C/run.log"; say "arm C ($side): no field default -> the log reports p unrelaxed" FAIL; }
done

W="$W" N="$N" N_NONE="$N_NONE" N_C="$N_C" P_BOUND="$P_BOUND" T_BOUND="$T_BOUND" U_BOUND="$U_BOUND" RHO_BOUND="$RHO_BOUND" \
C_P_BOUND="$C_P_BOUND" C_T_BOUND="$C_T_BOUND" C_U_BOUND="$C_U_BOUND" C_RHO_BOUND="$C_RHO_BOUND" \
CONTROL_RATIO="$CONTROL_RATIO" python3 - <<'RELAXCMP' || fail=1
import os, re, sys
import numpy as np
W, N, N_NONE, N_C = os.environ['W'], int(os.environ['N']), int(os.environ['N_NONE']), int(os.environ['N_C'])
def read(p):
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n?(\d+)\s*\n\(\n(.*?)\n\)\s*;', s, re.S)
    if m.group(1) == 'scalar':
        return np.array([float(x) for x in m.group(3).split()])
    return np.array([[float(c) for c in v.split()] for v in re.findall(r'\(([^)]*)\)', m.group(3))])
def rel(a, b, t, f):
    x = read(os.path.join(W, a, str(t), f)); y = read(os.path.join(W, b, str(t), f))
    return float(np.linalg.norm(x - y) / np.linalg.norm(y))
fields = ('p', 'T', 'U', 'rho')
host = {f: float(os.environ[f.upper() + '_BOUND']) for f in fields}
boundC = {f: float(os.environ['C_' + f.upper() + '_BOUND']) for f in fields}
ratio = float(os.environ['CONTROL_RATIO'])
ok = True
# Controls first: OpenFOAM's own answers must separate the arms, or a mirror ignoring the block passes.
for a, b, t, tag in (('of_A', 'of_none', N_NONE, 'A vs no relaxationFactors (the unfixed mirror)'),
                     ('of_A', 'of_C', N, 'A vs C (field default present vs absent)'),
                     ('of_A', 'of_B', N, 'A vs B (named U/p over default)')):
    best = max((rel(a, b, t, f) / host[f], f) for f in fields)
    good = best[0] > ratio
    print('     control: OpenFOAM %-48s %8.0fx the %s bound at t=%d   %s' % (tag, best[0], best[1], t, 'ok' if good else 'FAIL (inert)'))
    ok = ok and good
for arm in 'ABC':
    bounds = boundC if arm == 'C' else host
    horizon = N_C if arm == 'C' else N
    for side in ('host', 'cuda'):
        worst = {f: 0.0 for f in fields}
        for t in range(1, horizon + 1):
            for f in fields:
                worst[f] = max(worst[f], rel('%s_%s' % (side, arm), 'of_%s' % arm, t, f))
        for f in fields:
            good = worst[f] < bounds[f]
            print('     arm %s %-4s %-4s max over t=1..%d vs OpenFOAM %.4e   (bound %.1e)   %s'
                  % (arm, side, f, horizon, worst[f], bounds[f], 'ok' if good else 'FAIL'))
            ok = ok and good
sys.exit(0 if ok else 1)
RELAXCMP
say "relaxationFactors default / named-over-default / no-field-default match OpenFOAM at t=1..$N" \
    "$([ $fail = 0 ] && echo ok || echo FAIL)"

[ $fail = 0 ] && echo PASS || { echo FAIL; exit 1; }
