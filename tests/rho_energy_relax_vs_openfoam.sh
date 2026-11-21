#!/usr/bin/env bash
# The ENERGY equation's relaxation on rhoSimpleFoam's DEFAULT driver, held to OpenFOAM v2412.
#
# OpenFOAM relaxes EEqn only when solution::relaxEquation finds the energy field's own name -- "h" under
# sensibleEnthalpy, "e" under sensibleInternalEnergy -- or a `default` in relaxationFactors/equations
# (fvMatrix.C:1250-1263, solution.C:379-418, regex keys included); otherwise relax() is never called. The
# default driver relaxed at 0.7 whenever the case named neither, and also took the OTHER energy name
# (rho_simple_controls.cuh). Transient by nature -- a converged field cannot see a relaxation factor -- so
# every arm is compared at t=1..N from 0/ with every linear solver pinned (tolerance 1e-13, relTol 0).
#
# Fixture validation/rhoBox (laminar, hePsiThermo, sensibleEnthalpy; 700 K / 300 K walls, so the energy
# equation carries the flow): mesh in the repository; only relaxationFactors/equations changes per arm.
#
#   ARM none     equations { U 0.7; }                 OpenFOAM does not relax h
#   ARM other    equations { U 0.7; e 0.7; }          the other energy name: still not relaxed
#   ARM default  equations { U 0.7; default 0.5; }    h relaxed at 0.5 through `default`
#   ARM regex    equations { "(U|h)" 0.6; }           h relaxed at 0.6 through a pattern key
#   ARM named    equations { U 0.7; h 0.7; }          the fixture as shipped (positive control: unchanged)
#   CONTROL      OpenFOAM's own `none` and `named` answers differ by >= CONTROL_RATIO x the bound, so the
#                fixture can see the factor; and brae's log names the factor it applied (`e|h=none`).
#
# TOLERANCES, declared before this gate's first run. The default driver's agreement with OpenFOAM on this
# fixture with the shipped factor, the same pins and t=1..5, measured on main 9b7b1e58 (the floor this
# change does not touch): T 1.0e-12, p 3.0e-12, U 1.1e-11, rho 2.5e-12 (relative L2 per time, max). Bounds
# ~30x that floor. The pre-fix driver on ARM none sat at T 2.6e-02, rho 2.3e-02, U 7.9e-03 (measured, the
# same run): this gate's negative control, run with BRAE_BIN pointing at that binary.
#
# Labels: slow (OpenFOAM runs). Honours OFBASHRC; exits 77 without OpenFOAM, a GPU or the binary.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae_rhoSimpleFoam}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
N=${N:-5}
T_BOUND=${T_BOUND:-3e-11}; P_BOUND=${P_BOUND:-1e-10}; U_BOUND=${U_BOUND:-3e-10}; RHO_BOUND=${RHO_BOUND:-1e-10}
CONTROL_RATIO=${CONTROL_RATIO:-100}

[ -x "$BRAE" ] || { echo "SKIP: $BRAE not built"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: OpenFOAM not available ($OFBASHRC)"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
# shellcheck disable=SC1090
source "$OFBASHRC" > /dev/null 2>&1 || true   # before set -u: the bashrc reads unset variables
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }
SRC="$ROOT/validation/rhoBox"
[ -f "$SRC/constant/polyMesh/owner" ] || { echo "SKIP: fixture rhoBox has no mesh"; exit 77; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-74s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

# stage <dir> <equations body>
stage() {
    rm -rf "$1"; cp -r "$SRC" "$1"; rm -rf "$1"/[1-9]* "$1"/0 "$1"/log.*; cp -r "$1/0.orig" "$1/0"
    python3 - "$1" "$N" "$2" <<'PY'
import re, sys
d, n, eq = sys.argv[1:4]
c = d + '/system/controlDict'; s = open(c).read()
for k, v in (('endTime', n), ('writeInterval', '1'), ('writePrecision', '17')):
    s, k2 = re.subn(r'\b%s\s+[^;]+;' % k, '%s %s;' % (k, v), s); assert k2 == 1, k
open(c, 'w').write(s)
p = d + '/system/fvSolution'; s = open(p).read()
s = re.sub(r'tolerance\s+[^;]+;', 'tolerance 1e-13;', s)
s = re.sub(r'relTol\s+[^;]+;', 'relTol 0;', s)
s = re.sub(r'residualControl\s*\{[^{}]*\}', '', s)
s, k2 = re.subn(r'equations\s*\{[^{}]*\}', 'equations { %s }' % eq, s); assert k2 == 1
open(p, 'w').write(s)
PY
}
# cmp <A> <B>: per-field max over t=1..N of the relative L2 of B against A, printed as "T p U rho"
cmpf() {
    python3 - "$1" "$2" "$N" <<'PY'
import math, os, re, sys
A, B, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
def rd(p):
    t = open(p).read()
    m = re.search(r'internalField\s+nonuniform\s+List<(\w+)>\s*(\d+)\s*\(', t)
    k, cnt, s = m.group(1), int(m.group(2)), t[m.end():]
    if k == 'scalar': return [float(x) for x in s.split(None, cnt)[:cnt]]
    v = []
    for mm in re.finditer(r'\(([^()]*)\)', s):
        v.extend(float(x) for x in mm.group(1).split())
        if len(v) >= 3 * cnt: break
    return v
out = []
for f in ('T', 'p', 'U', 'rho'):
    worst = 0.0
    for t in range(1, n + 1):
        a, b = rd('%s/%d/%s' % (A, t, f)), rd('%s/%d/%s' % (B, t, f))
        den = math.sqrt(sum(x * x for x in a)) or 1.0
        worst = max(worst, math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))) / den)
    out.append('%.3e' % worst)
print(' '.join(out))
PY
}
within() {  # within "<T p U rho>" -> 0 when every field is inside its bound
    python3 - "$1" "$T_BOUND" "$P_BOUND" "$U_BOUND" "$RHO_BOUND" <<'PY'
import sys
v = [float(x) for x in sys.argv[1].split()]; b = [float(x) for x in sys.argv[2:6]]
sys.exit(0 if all(x <= y for x, y in zip(v, b)) else 1)
PY
}

echo "rho_energy_relax_vs_openfoam: default driver vs OpenFOAM, t=1..$N, bounds T $T_BOUND p $P_BOUND U $U_BOUND rho $RHO_BOUND"
for spec in "none:U 0.7;" "other:U 0.7; e 0.7;" "default:U 0.7; default 0.5;" 'regex:"(U|h)" 0.6;' "named:U 0.7; h 0.7;"; do
    arm=${spec%%:*}; eq=${spec#*:}
    stage "$W/of_$arm" "$eq"; stage "$W/br_$arm" "$eq"
    ( cd "$W/of_$arm" && nice -n 10 rhoSimpleFoam > log 2>&1 ) || { say "$arm: OpenFOAM failed" FAIL; continue; }
    ( cd "$W/br_$arm" && "$BRAE" -case "$W/br_$arm" > log 2>&1 ) || { say "$arm: brae failed" FAIL; tail -3 "$W/br_$arm/log"; continue; }
    [ -f "$W/of_$arm/$N/T" ] && [ -f "$W/br_$arm/$N/T" ] || { say "$arm: missing time $N" FAIL; continue; }
    d=$(cmpf "$W/of_$arm" "$W/br_$arm")
    if within "$d"; then say "$arm  (T p U rho) $d" PASS; else say "$arm  (T p U rho) $d" FAIL; fi
done
# the log names what it applied
grep -q 'e|h=none' "$W/br_none/log" && say "LOG none: brae says e|h=none" PASS || say "LOG none: brae does not say e|h=none" FAIL
grep -q 'e|h=0.5' "$W/br_default/log" && say "LOG default: brae says e|h=0.5" PASS || say "LOG default: brae does not say e|h=0.5" FAIL
# CONTROL: the fixture discriminates -- OpenFOAM's unrelaxed and relaxed answers are far apart
ctl=$(cmpf "$W/of_named" "$W/of_none")
python3 - "$ctl" "$T_BOUND" "$CONTROL_RATIO" <<'PY' && say "CONTROL OpenFOAM none vs named (T p U rho) $ctl" PASS || say "CONTROL OpenFOAM none vs named does not discriminate: $ctl" FAIL
import sys
t = float(sys.argv[1].split()[0]); sys.exit(0 if t >= float(sys.argv[3]) * float(sys.argv[2]) else 1)
PY

[ "$fail" -eq 0 ] && echo "rho_energy_relax_vs_openfoam: PASS" || echo "rho_energy_relax_vs_openfoam: FAIL"
exit "$fail"
