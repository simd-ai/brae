#!/usr/bin/env bash
# generalizedNewtonian (powerLaw) on the rhoSimpleFoam mirror -- OpenFOAM's own squareBendLiqNoNewtonian
# tutorial against real OpenFOAM v2412, iteration by iteration. Stage S1 Half B.
#
# THE TUTORIAL RUNS UNMODIFIED except for what every gate pins: linear solvers at 1e-14 relTol 0,
# residualControl and the `abort` function object out, ascii at 15 digits, a write every iteration.
#
# WHY THREE ARMS, AND WHY TWO OF THEM RESTART. The case starts from rest, so the model's constructor sees
# grad(U) = 0 in every cell the inlet has not reached and puts nu_ at nuMax = 1 there; iteration 1 then
# runs a momentum equation a thousand times more viscous than the developed one, and its pressure
# correction leaves a PLUG in the straight sections -- |U| = 0.6204 in 21120 cells, uniform to ~2e-12.
# The strain rate there is the round-off in U (~1e-9 1/s), and nu_ = nu0*strainRate^(n-1) turns it into
# a viscosity anywhere from nuMin to 0.3. That is OpenFOAM's own floor, not brae's -- OpenFOAM against
# ITSELF with only the p solver swapped (PCG for PBiCGStab, both 1e-14):
#     iteration 1  U 4.34e-12   time-1 nu_ 6.21e-02   iteration 2  U 1.04e-03
# which is brae against OpenFOAM to three digits (U 4.45e-12, nu_ 6.21e-02, U 1.04e-03). So:
#   A  from rest, ITERATION 1 ONLY: the constructor's nu_ (the nuMax clamp over the quiescent field, the
#      inlet-driven cells), rho*nu_ in the momentum equation, the pMax clamp that follows.
#   B  restart from OpenFOAM's own iteration 5, the tutorial's coefficients: nu_ is at nuMin in every
#      cell (strain rate > 8e-6 1/s everywhere, far above the plug's round-off) -- the model as the
#      tutorial actually runs it, correct() every iteration, compared with its written nu_.
#   C  the same restart with the UNCLAMPED branch live: nuMin 1e-9, so nu_ = nu0(T)*strainRate^(n-1) in
#      every cell and on every patch face, varying with the shear and with mu(T)/rho (walls at 350 K).
#      The coefficients sit in `generalizedNewtonianCoeffs { powerLawCoeffs { } }` -- OpenFOAM's two
#      optionalSubDict levels -- NEXT TO the tutorial's flat entries, which OpenFOAM then ignores and a
#      reader that looked only at the flat spelling would take. grad(U) is `cellLimited Gauss linear 1`,
#      the gradient strainRate() and divDevRhoReff both take.
# B alone cannot see most of the model: at nuMin everywhere, a frozen nu_, an unlagged correct() or a
# boundary nu_ copied from the face cell all give the same number. C is where those live.
#
# THE ONE LOOSER BOUND, C's interior nu_ at 5e-9, and why it is OpenFOAM's floor rather than slack. nu_
# goes as strainRate^(n-1), so it is LARGEST where the shear is smallest and the L2 norm is weighted
# towards the worst-conditioned gradients in the field. OpenFOAM against itself on arm C, p solver
# swapped (PCG for PBiCGStab, both 1e-14), U agreeing to 3e-14..5.7e-13:
#     nu_  it 6  2.22e-10   it 7  6.04e-10   it 8  1.13e-09
# and brae against OpenFOAM, U at ~1.5e-12:
#     nu_  it 6  2.76e-10   it 7  4.65e-10   it 8  1.01e-09
# Every quantity the model feeds -- U, p, T, rho -- and nu_ on every patch stay at 1e-10 (T, rho 1e-11).
#
# THE RESTART DEFECT THIS GATE FOUND FIRST, in brae's inletOutlet/outletInlet/freestream constructors
# (fv_patch_field.cuh): the patch value was taken from `inletValue` where OpenFOAM keeps the file's
# `value` (extrapolating the face cells when there is none). A start from rest cannot see it. Before the
# fix, the restart WITHOUT any viscosity model (`model Stokes`) at iteration 6: U 1.55e-05, T 5.64e-06,
# rho 4.60e-07 against OpenFOAM; after it U 1.44e-12, T 9.47e-13, rho 2.90e-13.
#
# FAIL-PROOFS, each defect put back in the source and the gate re-run against that binary (host arm);
# every one turns it red, on the arm and at the iteration where it lives:
#   the model ignored in muEff (Stokes)          A it 1  U 1.24e-01  p 3.32e-01;  B it 6  U 2.19e-01
#   correct() never called (nu_ frozen)          C it 6  nu_ 2.11e+00;  C it 7  U 3.54e-04        (B green)
#   correct() before the pressure corrector      C it 6  nu_ 5.48e+03;  C it 7  U 1.48e-02
#   boundary strainRate from the face cell's
#     gradient, gaussGrad's correction dropped   C it 6  nu_ walls 2.73e+00  U 1.15e-05          (B green)
#   the reader ignores generalizedNewtonianCoeffs C it 6  nu_ 9.16e+02  U 2.01e-01                (B green)
#   strainRate ignores grad(U)'s cellLimited     C it 6  nu_ 4.03e-01  U 4.10e-05                (B green)
#   inletOutlet's value taken from inletValue    B it 6  U 3.01e-03  T 5.64e-06
# THE CUDA ARM (same staging, same bounds) matches OpenFOAM at the same floors with its own arithmetic --
# C's interior nu_ 2.55e-10 / 4.71e-10 / 1.21e-09 against the host's 2.76e-10 / 4.65e-10 / 1.01e-09, every
# other field and every patch at ~1e-12. Five device fail-proofs, each module broken in the source and the
# CUDA arm re-run; every one turns it red:
#   device muEff on the molecular viscosity      A it 1  U 1.24e-01  p 3.32e-01
#   ...on the boundary faces only                A it 1  U 1.43e-01;  B it 6  U 9.42e-02
#   the correct() hook never set                 C it 6  nu_ 2.11e+00;  C it 7  U 3.54e-04
#   the device strainRate drops the limiter      C it 6  nu_ 4.05e-01;  C it 7  U 6.01e-05
#   the constructor's nu_ never uploaded         A it 1  U 1.24e-01;  B it 6  U 2.19e-01;  C it 6  U 8.07e-04
# and one that LOOKS like a defect and is not, recorded so nobody adds it as a fail-proof: correct() moved
# to the START of the step leaves U, p, T and rho at the floor on every arm -- SIMPLE's state at the end of
# iteration n is its state at the start of n+1 -- and only the WRITTEN nu_ is an iteration stale (C it 6
# 2.11e+00).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae_rhoSimpleFoam}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
ARMS=${ARMS:-"1 cuda"}
RESTART=5
ITERS=3

[ -x "$BRAE" ]     || { echo "SKIP: no brae binary at $BRAE"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }
SRC="${FOAM_TUTORIALS:-}/compressible/rhoSimpleFoam/squareBendLiqNoNewtonian"
[ -d "$SRC" ] || { echo "SKIP: tutorial $SRC not found"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0

cp -r "$SRC" "$W/mesh" || exit 1
( cd "$W/mesh" && rm -rf 0 [1-9]* processor* log.* && cp -r 0.orig 0 \
    && blockMesh > log.blockMesh 2>&1 && topoSet > log.topoSet 2>&1 ) \
    || { echo "FAIL: blockMesh/topoSet did not run"; tail -20 "$W/mesh/log.blockMesh"; exit 1; }

# stage <dir> <endTime> <variant>: the pins, and for variant C the dictionary and scheme above.
stage() {
    ENDT="$2" VARIANT="$3" python3 - "$1" <<'PYEOF'
import os, re, sys
d = sys.argv[1]; n = os.environ['ENDT']; variant = os.environ['VARIANT']
def edit(rel, fn):
    p = os.path.join(d, rel); s = open(p).read(); s2 = fn(s)
    if s2 == s: raise SystemExit('stage: no change made to ' + rel)
    open(p, 'w').write(s2)
def fixSol(s):
    s = re.sub(r'solvers\s*\{.*?\n\}', 'solvers\n{\n    p { solver PBiCGStab; preconditioner DIC; '
               'tolerance 1e-14; relTol 0; }\n    "(U|e|k|epsilon)" { solver PBiCGStab; '
               'preconditioner DILU; tolerance 1e-14; relTol 0; }\n}', s, count=1, flags=re.S)
    return re.sub(r'residualControl\s*\{[^}]*\}', '', s)
edit('system/fvSolution', fixSol)
def fixCtl(s):
    s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
    for k, v in (('startFrom', 'latestTime'), ('endTime', n), ('writeInterval', '1'),
                 ('writeFormat', 'ascii'), ('writePrecision', '15')):
        s = re.sub(r'^%s\s+.*' % k, '%-15s %s;' % (k, v), s, flags=re.M)
    return s
edit('system/controlDict', fixCtl)
if variant == 'C':
    edit('constant/turbulenceProperties', lambda s: s.replace(
        '    n                   0.4;\n',
        '    n                   0.4;\n\n    generalizedNewtonianCoeffs\n    {\n'
        '        viscosityModel  powerLaw;\n        powerLawCoeffs\n        {\n'
        '            n       0.4;\n            nuMin   1e-09;\n            nuMax   1;\n        }\n    }\n'))
    edit('system/fvSchemes', lambda s: s.replace(
        '    limited             cellLimited Gauss linear 1;',
        '    limited             cellLimited Gauss linear 1;\n    grad(U)             cellLimited Gauss linear 1;'))
PYEOF
}

# restart <dir> <variant>: OpenFOAM's own iteration-RESTART state, run on for ITERS more.
restart() {
    rm -rf "$1"; cp -r "$W/base" "$1"
    ( cd "$1" && rm -rf 0 log.* && for t in [1-9]*; do [ "$t" = "$RESTART" ] || rm -rf "$t"; done )
    cp "$W/mesh/system/controlDict" "$W/mesh/system/fvSchemes" "$W/mesh/system/fvSolution" "$1/system/"
    cp "$W/mesh/constant/turbulenceProperties" "$1/constant/"
    stage "$1" $((RESTART + ITERS)) "$2"
}

# compare <brae> <of> <first> <last> <label> <withNu> [<interior nu_ bound>]
compare() {
    FIRST="$3" LAST="$4" LABEL="$5" WITHNU="$6" NUBOUND="${7:-1e-10}" python3 - "$1" "$2" <<'PYEOF'
import math, os, re, sys
brae, of = sys.argv[1], sys.argv[2]
first, last = int(os.environ['FIRST']), int(os.environ['LAST'])
label, withNu = os.environ['LABEL'], os.environ['WITHNU'] == '1'
nuBound = float(os.environ['NUBOUND'])
nuName = 'generalizedNewtonian:nu'
def nums(t):
    return [float(x) for x in re.findall(r'-?[\d.]+(?:[eE][-+]?\d+)?', t)]
def internal(p, ncell):
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform[^(]*\((.*?)\)\s*;\s*\n\s*boundaryField', s, re.S)
    if m: return nums(m.group(1))
    m = re.search(r'internalField\s+uniform\s+(\([^)]*\)|[^;]+);', s)
    v = nums(m.group(1))
    return v * ncell
def patchValues(p):
    s = open(p).read(); bf = s[s.index('boundaryField'):]; out = {}
    for m in re.finditer(r'\n    ([^\s{]+)\s*\{(.*?)\n    \}', bf, re.S):
        body = m.group(2)
        v = re.search(r'value\s+nonuniform\s+List<scalar>\s*(\d+)\s*\((.*?)\)\s*;', body, re.S)
        if v: out[m.group(1)] = nums(v.group(2)); continue
        u = re.search(r'value\s+uniform\s+([^;]+);', body)
        if u: out[m.group(1)] = ('uniform', float(u.group(1)))
    return out
def rel(a, b):
    num = sum((x - y) ** 2 for x, y in zip(a, b)); den = sum(y * y for y in b)
    return math.sqrt(num / den) if den > 0 else math.sqrt(num)
# The mesh is BINARY (the tutorial's writeFormat); only the header note is read, as bytes.
ncell = int(re.search(rb'nCells:\s*(\d+)', open(of + '/constant/polyMesh/owner', 'rb').read(4096)).group(1))
bad = 0
for it in range(first, last + 1):
    fields = [('U', 1e-10), ('p', 1e-10), ('T', 1e-11), ('rho', 1e-11)]
    if withNu: fields.append((nuName, nuBound))
    for f, bound in fields:
        a = internal('%s/%d/%s' % (brae, it, f), ncell * (3 if f == 'U' else 1))
        b = internal('%s/%d/%s' % (of, it, f), ncell * (3 if f == 'U' else 1))
        e = rel(a, b) if len(a) == len(b) else float('inf')
        ok = e < bound
        print('     %s it %d  %-24s vs OpenFOAM (L2 rel)  %.6e   %s'
              % (label, it, f, e, 'ok' if ok else 'FAIL (bound %g)' % bound)); bad |= not ok
    if withNu:
        pa = patchValues('%s/%d/%s' % (brae, it, nuName)); pb = patchValues('%s/%d/%s' % (of, it, nuName))
        for k, vb in pb.items():
            va = pa.get(k)
            if va is None: print('     %s it %d  nu_ patch %s missing from brae   FAIL' % (label, it, k)); bad = 1; continue
            n = len(va) if isinstance(va, list) else (len(vb) if isinstance(vb, list) else 1)
            xa = va if isinstance(va, list) else [va[1]] * n
            xb = vb if isinstance(vb, list) else [vb[1]] * n
            e = rel(xa, xb) if len(xa) == len(xb) else float('inf')
            ok = e < 1e-10
            print('     %s it %d  nu_ on patch %-12s vs OpenFOAM (L2 rel)  %.6e   %s'
                  % (label, it, k, e, 'ok' if ok else 'FAIL (bound 1e-10)')); bad |= not ok
sys.exit(1 if bad else 0)
PYEOF
}

# The model must actually be selected, by OpenFOAM, on every arm -- a staging that fell back to Stokes
# would pass the comparisons below against itself.
selected() {
    grep -q "Selecting laminar stress model generalizedNewtonian" "$1/log.rhoSimpleFoam" \
        && grep -q "Selecting generalized Newtonian model powerLaw" "$1/log.rhoSimpleFoam"
}

runBrae() {
    BRAE_U_SOLVER=ofOrder BRAE_RHOSIMPLEFOAM_MIRROR=$2 "$BRAE" -case "$1" > "$1/log.brae" 2>&1
}

# ---- A: from rest, iteration 1 --------------------------------------------------------------------
rm -rf "$W/of_A"; cp -r "$W/mesh" "$W/of_A"; stage "$W/of_A" 1 A
( cd "$W/of_A" && rhoSimpleFoam > log.rhoSimpleFoam 2>&1 ) || { echo "FAIL: A -- OpenFOAM did not run"; exit 1; }
selected "$W/of_A" || { echo "FAIL: A -- OpenFOAM did not select generalizedNewtonian/powerLaw"; fail=1; }
for MIRROR in $ARMS; do
    BR="$W/br_A_$MIRROR"; rm -rf "$BR"; cp -r "$W/mesh" "$BR"; stage "$BR" 1 A
    runBrae "$BR" "$MIRROR" || { echo "FAIL: A ($MIRROR) -- brae did not run"; grep -v '^brae NOTICE' "$BR/log.brae" | tail -5; fail=1; continue; }
    echo "== A: squareBendLiqNoNewtonian from rest -- $([ "$MIRROR" = cuda ] && echo 'CUDA arm' || echo 'host arm') =="
    compare "$BR" "$W/of_A" 1 1 A 0 || fail=1
done

# ---- the developed state: OpenFOAM's own iteration RESTART ------------------------------------------
rm -rf "$W/base"; cp -r "$W/mesh" "$W/base"; stage "$W/base" "$RESTART" A
( cd "$W/base" && rhoSimpleFoam > log.rhoSimpleFoam 2>&1 ) || { echo "FAIL: base -- OpenFOAM did not run"; exit 1; }

for V in B C; do
    restart "$W/of_$V" "$V" || { echo "FAIL: $V -- staging"; fail=1; continue; }
    ( cd "$W/of_$V" && rhoSimpleFoam > log.rhoSimpleFoam 2>&1 ) || {
        echo "FAIL: $V -- OpenFOAM did not run"; tail -15 "$W/of_$V/log.rhoSimpleFoam"; fail=1; continue; }
    selected "$W/of_$V" || { echo "FAIL: $V -- OpenFOAM did not select generalizedNewtonian/powerLaw"; fail=1; }
    # What the arm claims about OpenFOAM's own nu_, checked on OpenFOAM's file: B at nuMin everywhere,
    # C strictly between nuMin and nuMax everywhere and spread over the field.
    V="$V" RS=$((RESTART + 1)) python3 - "$W/of_$V" <<'PYEOF' || fail=1
import os, re, sys
d = sys.argv[1]; v = os.environ['V']
s = open('%s/%s/generalizedNewtonian:nu' % (d, os.environ['RS'])).read()
m = re.search(r'internalField\s+nonuniform[^(]*\((.*?)\)\s*;', s, re.S)
nu = [float(x) for x in m.group(1).split()] if m else None
if v == 'B':
    u = re.search(r'internalField\s+uniform\s+([^;]+);', s)
    ok = (u is not None and abs(float(u.group(1)) - 1e-3) < 1e-15) or (nu and max(abs(x - 1e-3) for x in nu) < 1e-15)
    print('     B  OpenFOAM nu_ is nuMin = 1e-3 in every cell                       %s' % ('ok' if ok else 'FAIL'))
else:
    inside = sum(1 for x in nu if 1e-9 * (1 + 1e-9) < x < 1 - 1e-9) if nu else 0
    ok = nu is not None and inside == len(nu) and max(nu) / min(nu) > 10
    print('     C  OpenFOAM nu_ unclamped in %d of %d cells, spread %.3g..%.3g        %s'
          % (inside, len(nu) if nu else 0, min(nu) if nu else 0, max(nu) if nu else 0, 'ok' if ok else 'FAIL'))
sys.exit(0 if ok else 1)
PYEOF
    for MIRROR in $ARMS; do
        BR="$W/br_${V}_$MIRROR"; restart "$BR" "$V" || { fail=1; continue; }
        runBrae "$BR" "$MIRROR" || { echo "FAIL: $V ($MIRROR) -- brae did not run"; grep -v '^brae NOTICE' "$BR/log.brae" | tail -5; fail=1; continue; }
        echo "== $V: restart from OpenFOAM's iteration $RESTART -- $([ "$MIRROR" = cuda ] && echo 'CUDA arm' || echo 'host arm') =="
        compare "$BR" "$W/of_$V" $((RESTART + 1)) $((RESTART + ITERS)) "$V" 1 \
            "$([ "$V" = C ] && echo 5e-9 || echo 1e-10)" || fail=1
    done
done

[ "$fail" -eq 0 ] && echo "PASSED" || echo "FAILED"
exit $fail
