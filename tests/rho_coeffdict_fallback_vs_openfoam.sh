#!/usr/bin/env bash
# A COEFFICIENT WRITTEN FLAT IN `RAS { }` REACHES THE MODEL, as OpenFOAM's optionalSubDict -- both arms.
#
# RASModel.C:72 builds every model's coeffDict_ as RASDict_.optionalSubDict(type + "Coeffs"), and
# dictionary::optionalSubDict (dictionary.C:566-591) returns the RAS dictionary ITSELF when the
# sub-dictionary is absent; EddyDiffusivity reads Prt from the same coeffDict() (EddyDiffusivity.C:37) and
# LESModel.C:72 does the same for LES. So `RAS { RASModel kEpsilon; Cmu 0.12; Prt 0.7; }` runs OpenFOAM's
# kEpsilon at Cmu 0.12 and alphat at Prt 0.7. brae read the sub-dictionary only, so that case ran the
# defaults on both arms (queue item 21). FoamDict::optionalSubDict now mirrors OpenFOAM's at the three reads
# (kEpsilon coefficients, kOmegaSST coefficients, Prt).
#
#   ARM KE  validation/rhoKE  (kEpsilon, blockMesh): FLAT `Cmu 0.12; Prt 0.7;` and NESTED
#           `kEpsilonCoeffs { Cmu 0.12; Prt 0.7; }`, host and device, at t=1..10 vs real rhoSimpleFoam.
#   ARM SST validation/rhoSST (kOmegaSST, host arm; the device refuses SST): FLAT `betaStar 0.11; Prt 0.7;`
#           and NESTED `kOmegaSSTCoeffs { betaStar 0.11; Prt 0.7; }`.
#   IDENTITY  OpenFOAM's own flat and nested answers must be identical (< 1e-14): the semantic control that
#           the fallback IS what OpenFOAM does, measured rather than read.
#   CONTROL OpenFOAM flat vs the unmodified fixture must differ by >= 100x the bound at t=10, or the flat
#           coefficients changed nothing and the arm is inert.
#
# Every linear solver pinned to 1e-12 / 0 / 2000; residualControl emptied; from 0/.
# Measured floors (max over t=1..10, 2026-09-03), identical for flat and nested: host k 8.6e-13, epsilon
# 2.1e-12, U 6.9e-13, p 2.9e-12, T 9.7e-13, alphat 7.0e-13; device k 3.1e-12, epsilon 5.8e-12, alphat
# 6.5e-11 (the device's alphat write path; its k/epsilon sit with the host's); SST host omega 7.4e-13.
# Bounds ~30x the larger arm's floor. OpenFOAM's flat-vs-nested identity read 0.000e+00 on every field.
# Fail-proof: FoamDict::optionalSubDict returning the sub-dictionary only (null when absent) -> every FLAT
# arm FAILS (brae runs the defaults) while the nested arms and the identities stay ok.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILDDIR="${BUILD:-$ROOT/build}"
BIN="$BUILDDIR/brae_rhoSimpleFoam"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
N=${N:-10}
K_BOUND=${K_BOUND:-1e-10}; E_BOUND=${E_BOUND:-2e-10}; W_BOUND=${W_BOUND:-3e-11}
U_BOUND=${U_BOUND:-3e-11}; P_BOUND=${P_BOUND:-1e-10}; T_BOUND=${T_BOUND:-3e-11}; A_BOUND=${A_BOUND:-2e-09}
CONTROL_RATIO=${CONTROL_RATIO:-100}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$ROOT/validation/rhoKE" ] && [ -d "$ROOT/validation/rhoSST" ] || { echo "SKIP: fixtures missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-76s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

base()   # $1 fixture name -> $W/base_<name>, meshed
{
    local src="$ROOT/validation/$1" dst="$W/base_$1"
    cp -r "$src" "$dst"; rm -rf "$dst"/[1-9]* "$dst"/0 "$dst"/log.*
    cp -r "$dst/0.orig" "$dst/0"
    ( cd "$dst" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh on $1"; exit 1; }
}
base rhoKE; base rhoSST

stage()   # $1 dst  $2 fixture  $3 RAS-block body ("" = the fixture's own)
{
    RASBODY="$3" python3 - "$W/base_$2" "$1" "$N" <<'CDFSTAGE'
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
s = re.sub(r'maxIter\s+[0-9]+;', 'maxIter 2000;', s)
s = re.sub(r'relTol\s+[0-9.eE+-]+;', 'relTol 0;', s)
s = re.sub(r'relTol 0;(?![^}]*maxIter)', 'relTol 0; maxIter 2000;', s)
open(f, 'w').write(s)
body = os.environ['RASBODY']
if body:
    t = os.path.join(dst, 'constant/turbulenceProperties'); s = open(t).read()
    s, k = re.subn(r'RAS\s*\{.*?\n\}', 'RAS\n{\n%s\n}' % body, s, flags=re.S)
    assert k == 1, 'expected one RAS block'
    open(t, 'w').write(s)
CDFSTAGE
}
run()   # $1 dir  $2 of|1|cuda
{
    case "$2" in
        of) ( cd "$1" && rhoSimpleFoam > run.log 2>&1 ) ;;
        *)  ( cd "$1" && BRAE_RHOSIMPLEFOAM_MIRROR=$2 "$BIN" -case "$1" > run.log 2>&1 ) ;;
    esac || { tail -6 "$1/run.log"; echo "FAIL: $2 did not run in $1"; exit 1; }
    [ -d "$1/$N" ] || { echo "FAIL: $2 wrote no $N/ in $1"; exit 1; }
}

KE_HEAD='    RASModel        kEpsilon;
    turbulence      on;
    printCoeffs     on;'
SST_HEAD='    RASModel        kOmegaSST;
    turbulence      on;
    printCoeffs     on;'
stage "$W/ke_of_default" rhoKE ""
stage "$W/ke_of_flat"    rhoKE "$KE_HEAD
    Cmu             0.12;
    Prt             0.7;"
stage "$W/ke_of_nested"  rhoKE "$KE_HEAD
    kEpsilonCoeffs { Cmu 0.12; Prt 0.7; }"
stage "$W/sst_of_default" rhoSST ""
stage "$W/sst_of_flat"    rhoSST "$SST_HEAD
    betaStar        0.11;
    Prt             0.7;"
stage "$W/sst_of_nested"  rhoSST "$SST_HEAD
    kOmegaSSTCoeffs { betaStar 0.11; Prt 0.7; }"
for c in ke_of_default ke_of_flat ke_of_nested sst_of_default sst_of_flat sst_of_nested; do run "$W/$c" of; done
# OpenFOAM's printCoeffs writes the coefficients it RUNS: the flat form must print Cmu 0.12 / betaStar 0.11.
grep -qE "^\s*Cmu\s+0\.12;" "$W/ke_of_flat/run.log"      && say "OpenFOAM's printCoeffs shows the flat Cmu 0.12 in use (kEpsilon)" ok \
    || say "OpenFOAM's printCoeffs shows the flat Cmu 0.12 in use (kEpsilon)" FAIL
grep -qE "^\s*betaStar\s+0\.11;" "$W/sst_of_flat/run.log" && say "OpenFOAM's printCoeffs shows the flat betaStar 0.11 in use (kOmegaSST)" ok \
    || say "OpenFOAM's printCoeffs shows the flat betaStar 0.11 in use (kOmegaSST)" FAIL

for v in flat nested; do
    cp -r "$W/ke_of_$v"  "$W/ke_host_$v";  rm -rf "$W/ke_host_$v"/[1-9]*;  run "$W/ke_host_$v" 1
    cp -r "$W/ke_of_$v"  "$W/ke_cuda_$v";  rm -rf "$W/ke_cuda_$v"/[1-9]*;  run "$W/ke_cuda_$v" cuda
    cp -r "$W/sst_of_$v" "$W/sst_host_$v"; rm -rf "$W/sst_host_$v"/[1-9]*; run "$W/sst_host_$v" 1
done

W="$W" N="$N" K_BOUND="$K_BOUND" E_BOUND="$E_BOUND" W_BOUND="$W_BOUND" U_BOUND="$U_BOUND" P_BOUND="$P_BOUND" \
T_BOUND="$T_BOUND" A_BOUND="$A_BOUND" CONTROL_RATIO="$CONTROL_RATIO" python3 - <<'CDFCMP' || fail=1
import os, re, sys
import numpy as np
W, N = os.environ['W'], int(os.environ['N'])
ratio = float(os.environ['CONTROL_RATIO'])
def read(p):
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n?(\d+)\s*\n\(\n(.*?)\n\)\s*;', s, re.S)
    if m:
        if m.group(1) == 'scalar':
            return np.array([float(x) for x in m.group(3).split()])
        return np.array([[float(c) for c in v.split()] for v in re.findall(r'\(([^)]*)\)', m.group(3))])
    u = re.search(r'internalField\s+uniform\s+\(?([^);]+)\)?;', s)
    return np.array([float(x) for x in u.group(1).split()])
def rel(a, b, t, f):
    x = read(os.path.join(W, a, str(t), f)); y = read(os.path.join(W, b, str(t), f))
    return float(np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-300))
B = {'k': 'K_BOUND', 'epsilon': 'E_BOUND', 'omega': 'W_BOUND', 'U': 'U_BOUND', 'p': 'P_BOUND', 'T': 'T_BOUND', 'alphat': 'A_BOUND'}
bounds = {f: float(os.environ[k]) for f, k in B.items()}
ok = True
# IDENTITY: OpenFOAM's flat and nested runs are the same run.
for fx, fields in (('ke', ('k', 'epsilon', 'U', 'p', 'T', 'alphat')), ('sst', ('k', 'omega', 'U', 'p', 'T', 'alphat'))):
    worst = max(rel('%s_of_flat' % fx, '%s_of_nested' % fx, t, f) for t in range(1, N + 1) for f in fields)
    good = worst < 1e-14
    print('     identity: OpenFOAM flat == nested (%s) max relL2 %.3e   %s' % (fx, worst, 'ok' if good else 'FAIL'))
    ok = ok and good
    # CONTROL: the flat coefficients must have changed OpenFOAM's answer, or nothing here discriminates.
    best = max((rel('%s_of_flat' % fx, '%s_of_default' % fx, N, f) / bounds[f], f) for f in fields)
    good = best[0] > ratio
    print('     control: OpenFOAM flat vs default (%s) %8.0fx the %s bound at t=%d   %s' % (fx, best[0], best[1], N, 'ok' if good else 'FAIL (inert)'))
    ok = ok and good
# THE ARMS: brae flat and nested against OpenFOAM, per field, max over t.
for arm, fx, fields in (('ke_host', 'ke', ('k', 'epsilon', 'U', 'p', 'T', 'alphat')),
                        ('ke_cuda', 'ke', ('k', 'epsilon', 'U', 'p', 'T', 'alphat')),
                        ('sst_host', 'sst', ('k', 'omega', 'U', 'p', 'T', 'alphat'))):
    for v in ('flat', 'nested'):
        for f in fields:
            worst = max(rel('%s_%s' % (arm, v), '%s_of_%s' % (fx, v), t, f) for t in range(1, N + 1))
            good = worst < bounds[f]
            print('     %-9s %-6s %-7s max over t=1..%d vs OpenFOAM %.4e   (bound %.1e)   %s' % (arm, v, f, N, worst, bounds[f], 'ok' if good else 'FAIL'))
            ok = ok and good
sys.exit(0 if ok else 1)
CDFCMP
say "flat RAS coefficients reach both arms' models and Prt as OpenFOAM's optionalSubDict" "$([ $fail = 0 ] && echo ok || echo FAIL)"
[ $fail = 0 ] && echo PASS || { echo FAIL; exit 1; }
