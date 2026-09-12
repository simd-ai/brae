#!/usr/bin/env bash
# `gradSchemes default leastSquares` reaching the turbulence CLOSURE -- kOmegaSST's CDkOmega, and the
# k/omega|epsilon corrected-laplacian corrections -- on the rhoSimpleFoam OF-mirror host arm, end to end
# on validation/rhoSST against real OpenFOAM v2412. The gradient itself is gated by
# tests/leastsquares_grad_vs_openfoam.sh (a 2x2 against OpenFOAM's own grad(T)); this gates the CONSUMERS.
#
# WHY. OpenFOAM's fvc::grad(vf) resolves `grad(<name>)` through gradSchemes (fvcGrad.C:149), so a case
# whose default is leastSquares builds CDkOmega = 2*alphaOmega2*(grad(k) & grad(omega))/omega
# (kOmegaSSTBase.C:555-558) from the least-squares fit, and F1 -- the blend between the k-omega and
# k-epsilon branches -- from that. The host closures took Gauss linear whatever the case said and the
# mirror REFUSED such a case by name; now they take fvc::leastSquaresGrad under gradKLeastSq, from each
# field's STORED patch values, with a cellLimited <k> still applied on top when named. The path this is
# on: gasMixing/injectorPipe (kEpsilon, `default leastSquares`, limitedLinear on div(phi,e|k|epsilon)).
#
# FOUR ARMS, EACH THE OTHERS' CONTROL:
#   ship   rhoSST as shipped (Gauss linear everywhere), both arms vs OpenFOAM at the mirror's usual floor
#          -- proves the fixture and the staging discriminate at all.
#   lsq    grad(k) and grad(omega) `leastSquares` EXPLICITLY (a `default` would also switch OpenFOAM's
#          grad(U) and grad(p), which the momentum path does not), divSchemes UNCHANGED (upwind, so the limiter
#          path is not involved and the only leastSquares consumer is CDkOmega; rhoSST's laplacians are
#          `orthogonal`, so the correction path is not exercised here and is stated as such).
#          BOTH arms vs OpenFOAM at LSQBOUND (measured: host 2.900e-12, CUDA 2.901e-12, worst p at
#          iteration 1 -- the shipped control's own floor). The CUDA closures compute grad(k)/grad(omega|epsilon)
#          leastSquares since the device port (kOmegaSST.cu's CDkOmega, turbulence_transport.cu's
#          corrected-laplacian correction; the device twin test_rho_kepsilon_cuda holds the kEpsilon
#          closure to 1e-13 against the host under it, on a `corrected` fixture where this one is
#          orthogonal). The two closure entries are switched EXPLICITLY here so that this arm isolates
#          CDkOmega; the restart arms below run `default leastSquares` (grad(U), grad(p), the limiters) on
#          BOTH mirror arms: lsq_none/CUDA p 2.85e-12, lsq_all/CUDA omega 2.46e-11 (host 2.45e-11).
#   wrong  the control that lsq is not trivially passable: brae's host arm run with fvSchemes saying
#          `Gauss linear` (so it computes the Gauss gradient) compared against OpenFOAM's leastSquares
#          run -- must DIFFER by far more than LSQBOUND, or the fixture cannot tell the two schemes apart
#          and the lsq arm proves nothing.
#   lim    `bounded Gauss limitedLinear 1` on div(phi,k|omega) and on div(phi,h|K), each RESTARTED from
#          OpenFOAM's own iteration 5 -- see the restart arms for why a from-rest comparison of a limiter
#          on a uniform field is decided by round-off sign and cannot be asserted.
#
# EVERY CONSUMER a `default leastSquares` switches is now ported, and the arms below were what named them
# one at a time (restarted, worst field over iterations 6-8, before -> after):
#   grad(k)/grad(omega)  the closure's CDkOmega and corrected-laplacian gradients   exact from the start
#   the limiter's own gradient for div(phi,k|omega)   k 3.1e-06 -> 8.7e-12   (LimitedScheme.C:51-55
#                        resolves grad(<field>); both closures' divWithScheme took Gauss regardless)
#   grad(p)              U = HbyA - rAtU*grad(p) (pEqn.H:86, pcEqn.H:99), SIMPLEC's HbyA correction
#                        (pcEqn.H:30,65) and each branch's non-orth correction   1.7e-05 -> 2.9e-12
#   grad(U)              divDevRhoReff's dev2 term, the closure's production, validate()'s correctNut
#                        3.6e-06 -> 2.9e-12, via the VECTOR fvc::leastSquaresGrad (gated in its own right
#                        by tests/leastsquares_grad_vs_openfoam.sh against OpenFOAM's own grad(U))
# WHAT THIS GATE DOES NOT CLAIM: rhoSST's laplacianSchemes are `orthogonal` and its divSchemes name no
# linearUpwind, so the non-orthogonal grad(p)/grad(U) corrections and the gradient linearUpwind NAMES are
# ported but NOT exercised here. Recorded in PORT.md rather than asserted.
#
# BOUNDS -- every one measured, and recorded beside its measurement below.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae_rhoSimpleFoam}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
ITERS=3
SHIPBOUND=${SHIPBOUND:-1e-11}     # measured: every field <= 2.9e-12 at every iteration, both arms
LSQBOUND=${LSQBOUND:-1e-11}       # measured: every field <= 1.3e-12 at every iteration -- the shipped control's own numbers
DIFFER=${DIFFER:-1e-09}           # wrong-scheme control must exceed this (100x LSQBOUND); measured 6.3e-08 on nut at it 3, 1.5e-08 omega
LIMBOUND=${LIMBOUND:-1e-11}       # the limiter arms RESTART from OpenFOAM's iteration 5; measured <= 2.9e-12 (gauss), see the lim arm
LSQLIMBOUND=${LSQLIMBOUND:-1e-11}        # explicit k/omega leastSquares + k/omega limiter, restarted; measured 8.7e-12
LSQALLBOUND=${LSQALLBOUND:-1e-11}        # `default leastSquares` (grad(U), grad(p)), upwind divs, restarted; measured 2.9e-12
# ...and the same WITH limitedLinear on all four equations: leastSquares in every consumer and a limiter
# on k, omega, h and K, three iterations of accumulation from the restart. The loosest arm in the gate
# and the only one not at the fixture's 3e-12 floor; measured omega 2.5e-11 at iteration 8, nut 1.4e-11.
LSQALLLIMBOUND=${LSQALLLIMBOUND:-1e-10}

[ -x "$BRAE" ]     || { echo "SKIP: no brae binary at $BRAE"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }
[ -d "$ROOT/validation/rhoSST" ] || { echo "SKIP: fixture rhoSST missing"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0

# stage <dest> <variant: ship|lsq|lim|wrong>. `wrong` is the lsq case with fvSchemes put back to Gauss
# linear AFTER staging, so brae computes the Gauss gradient on a case OpenFOAM ran with leastSquares.
stage() {
    local d="$1" v="$2"
    rm -rf "$d"; mkdir -p "$d"
    cp -r "$ROOT/validation/rhoSST/constant" "$ROOT/validation/rhoSST/system" "$d/"
    cp -r "$ROOT/validation/rhoSST/0.orig" "$d/0"
    ITERS="$ITERS" V="$v" python3 - "$d" <<'PYEOF'
import os, re, sys
d, n, v = sys.argv[1], os.environ['ITERS'], os.environ['V']
p = os.path.join(d, 'system/controlDict'); s = open(p).read()
for k, val in (('startFrom', 'latestTime'), ('endTime', n), ('writeInterval', '1'),
               ('writeFormat', 'ascii'), ('writePrecision', '15'), ('writeCompression', 'off')):
    if re.search(r'^%s\s+' % k, s, re.M):
        s = re.sub(r'^%s\s+.*' % k, '%-15s %s;' % (k, val), s, flags=re.M)
    else:
        s += '\n%-15s %s;\n' % (k, val)
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
open(p, 'w').write(s)
p = os.path.join(d, 'system/fvSolution'); s = open(p).read()
s = re.sub(r'\btolerance\s+[0-9.eE+-]+\s*;', 'tolerance 1e-14;', s)
s = re.sub(r'\brelTol\s+[0-9.eE+-]+\s*;', 'relTol 0;', s)
s = re.sub(r'residualControl\s*\{[^}]*\}', '', s)
open(p, 'w').write(s)
if v in ('lsq', 'lsqp', 'lim'):
    p = os.path.join(d, 'system/fvSchemes'); s = open(p).read()
    if v == 'lsqp':
        # grad(p) ALONE, explicit: its five consumers (the momentum source, U = HbyA - rAtU*grad(p),
        # SIMPLEC's HbyA correction, each pressure branch's non-orth correction) on both arms, with
        # grad(U) and the closure untouched. The CUDA arm computes grad(p) leastSquares since its port
        # (tests/rho_step_cuda_lsq.sh is its driver-level twin against the host).
        s2 = re.sub(r'gradSchemes\s*\{\s*default\s+Gauss linear;',
                    'gradSchemes     { default Gauss linear;\n    grad(p)         leastSquares;', s)
    elif v == 'lsq':
        # EXPLICIT entries, not `default`: a leastSquares default also reaches OpenFOAM's grad(U) and
        # grad(p), which brae's momentum path takes as Gauss -- measured U 3.3e-06 / p 3.8e-07 at
        # iteration 1 under `default leastSquares` with upwind divergence. Naming only grad(k) and
        # grad(omega) isolates the closure: CDkOmega is then the ONLY least-squares consumer.
        s2 = re.sub(r'gradSchemes\s*\{\s*default\s+Gauss linear;',
                    'gradSchemes     { default Gauss linear;\n    grad(k)         leastSquares;\n    grad(omega)     leastSquares;', s)
    else:
        s2 = re.sub(r'gradSchemes\s*\{\s*default\s+Gauss linear;', 'gradSchemes     { default leastSquares;', s)
    if v == 'lim':
        for f in ('k', 'omega', 'h', 'K'):
            s2 = re.sub(r'div\(phi,%s\)\s+bounded Gauss upwind;' % f,
                        'div(phi,%s)      bounded Gauss limitedLinear 1;' % f, s2)
    if s2 == s: raise SystemExit('staging changed nothing for ' + v)
    open(p, 'w').write(s2)
PYEOF
    ( cd "$d" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh"; return 1; }
}

# compare <brae> <of> <label> <bound> <mustDiffer: 0|1>
compare() {
    LABEL="$3" BOUND="$4" MUSTDIFFER="$5" ITERS="$ITERS" python3 - "$1" "$2" <<'PYEOF'
import math, os, re, sys
b, o = sys.argv[1], sys.argv[2]
label, bound, must, n = os.environ['LABEL'], float(os.environ['BOUND']), os.environ['MUSTDIFFER'] == '1', int(os.environ['ITERS'])
def internal(p, m=None):
    s = open(p).read()
    mm = re.search(r'internalField\s+nonuniform[^(]*\((.*?)\)\s*;\s*\n\s*boundaryField', s, re.S)
    if mm: return [float(x) for x in re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', mm.group(1))]
    mm = re.search(r'internalField\s+uniform\s+([^;]+);', s)
    v = [float(x) for x in re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', mm.group(1))]
    return v * (m // len(v)) if m and m % len(v) == 0 else v
def rel(a, c):
    den = sum(y * y for y in c); return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, c)) / den) if den else 0.0
bad, worst, wname = 0, 0.0, ''
for it in range(1, n + 1):
    for f in ('U', 'p', 'T', 'k', 'omega', 'nut'):
        pb, po = '%s/%d/%s' % (b, it, f), '%s/%d/%s' % (o, it, f)
        if not (os.path.exists(pb) and os.path.exists(po)): continue
        a = internal(pb); c = internal(po, len(a))
        e = rel(a, c) if len(a) == len(c) else float('inf')
        if e > worst: worst, wname = e, '%s it %d' % (f, it)
        if not must and e >= bound:
            print('     %s it %d  %-6s vs OpenFOAM (L2 rel)  %.6e   FAIL (bound %g)' % (label, it, f, e, bound)); bad = 1
if must:
    ok = worst > bound
    print('     %s CONTROL: worst %s %.3e must exceed %g   %s' % (label, wname, worst, bound, 'ok' if ok else 'FAIL (the fixture cannot tell the schemes apart)'))
    bad = 0 if ok else 1
else:
    print('     %s worst %s %.3e  (bound %g)   %s' % (label, wname, worst, bound, 'FAIL' if bad else 'ok'))
sys.exit(1 if bad else 0)
PYEOF
}

runOF() { ( cd "$1" && rhoSimpleFoam > log.rhoSimpleFoam 2>&1 ); [ -d "$1/$ITERS" ]; }
runBrae() {   # runBrae <dir> <mirror> [<env>]
    env ${3:-} BRAE_RHOSIMPLEFOAM_MIRROR=$2 "$BRAE" -case "$1" > "$1/log.brae" 2>&1 && [ -d "$1/$ITERS" ]
}

echo "== ship: rhoSST as shipped, both arms =="
stage "$W/of_ship" ship || fail=1
runOF "$W/of_ship" || { echo "SKIP: OpenFOAM did not run the fixture"; exit 77; }
for M in 1 cuda; do
    stage "$W/br_ship_$M" ship || { fail=1; continue; }
    runBrae "$W/br_ship_$M" $M || { echo "     $M: brae did not run the shipped case   FAIL"; tail -2 "$W/br_ship_$M/log.brae"; fail=1; continue; }
    compare "$W/br_ship_$M" "$W/of_ship" "$([ $M = cuda ] && echo CUDA || echo host)" "$SHIPBOUND" 0 || fail=1
done

echo "== lsq: default leastSquares, CDkOmega from the least-squares gradient =="
stage "$W/of_lsq" lsq || fail=1
runOF "$W/of_lsq" || { echo "SKIP: OpenFOAM did not run the leastSquares variant"; exit 77; }
stage "$W/br_lsq_1" lsq || fail=1
if runBrae "$W/br_lsq_1" 1; then
    compare "$W/br_lsq_1" "$W/of_lsq" "host" "$LSQBOUND" 0 || fail=1
else
    echo "     host: did not run the leastSquares variant   FAIL"; grep -v '^brae NOTICE' "$W/br_lsq_1/log.brae" | tail -3; fail=1
fi
stage "$W/br_lsq_cuda" lsq || fail=1
if runBrae "$W/br_lsq_cuda" cuda; then
    compare "$W/br_lsq_cuda" "$W/of_lsq" "CUDA" "$LSQBOUND" 0 || fail=1
else
    echo "     CUDA: did not run the leastSquares variant   FAIL"; grep -v '^brae NOTICE' "$W/br_lsq_cuda/log.brae" | tail -3; fail=1
fi

echo "== lsqp: grad(p) leastSquares, explicit -- the pressure gradient's five consumers, both arms =="
stage "$W/of_lsqp" lsqp || fail=1
runOF "$W/of_lsqp" || { echo "SKIP: OpenFOAM did not run the grad(p) leastSquares variant"; exit 77; }
for M in 1 cuda; do
    stage "$W/br_lsqp_$M" lsqp || { fail=1; continue; }
    if runBrae "$W/br_lsqp_$M" $M; then
        compare "$W/br_lsqp_$M" "$W/of_lsqp" "$([ $M = cuda ] && echo CUDA || echo host)" "$LSQBOUND" 0 || fail=1
    else
        echo "     $M: did not run the grad(p) leastSquares variant   FAIL"; grep -v '^brae NOTICE' "$W/br_lsqp_$M/log.brae" | tail -3; fail=1
    fi
done
# ...and its control: the shipped (Gauss grad(p)) brae run against OpenFOAM's grad(p) leastSquares run
# must differ, or the fixture cannot tell a Gauss pressure gradient from a least-squares one.
compare "$W/br_ship_1" "$W/of_lsqp" "lsqp wrong-scheme" "$DIFFER" 1 || fail=1
echo "== wrong: the Gauss gradient against OpenFOAM's leastSquares run -- must differ =="
stage "$W/br_wrong" ship || fail=1     # brae computes Gauss linear...
if runBrae "$W/br_wrong" 1; then
    compare "$W/br_wrong" "$W/of_lsq" "wrong-scheme" "$DIFFER" 1 || fail=1    # ...against OpenFOAM's leastSquares
else
    echo "     wrong-scheme control did not run   FAIL"; fail=1
fi

# ---- the limiter arms, RESTARTED. From rest, rhoSST's k is uniform, so on every interior face limitedLinear's
# r sits in NVDTVD's 0/0 branch: gradf is exactly 0 and gradcf is the round-off of a Gauss sum, and
# r = 2000*sign(gradcf)*sign(0) - 1 is +1999 (central) or -2001 (upwind) by the SIGN of that round-off --
# i.e. by summation order. Both codes are right and part at iteration 1 (measured k 3.9e-03, T 4.2e-04 from
# rest with Gauss gradients), the same phenomenon as squareBendLiqNoNewtonian's plug. Restarted from
# OpenFOAM's own iteration 5 -- non-uniform k, omega and h -- the limiter paths are exact, and that is
# what is asserted: measured 6.0e-13 .. 2.9e-12 on every field at iterations 6-8, both limiter sets.
restartArm() {   # restartArm <tag> <grad: gauss|lsq> <lim: komega|hK|all> <bound> [<env>]
    local tag="$1" gr="$2" lim="$3" bound="$4" envs="${5:-}"
    local of="$W/of_r_$tag" br="$W/br_r_$tag"
    stageR "$of" "$gr" "$lim" 8
    ( cd "$of" && rhoSimpleFoam > log.rhoSimpleFoam 2>&1 )
    [ -d "$of/8" ] || { echo "     $tag: OpenFOAM did not reach iteration 8   FAIL"; fail=1; return; }
    # BOTH mirror arms from the same OpenFOAM restart: the CUDA arm computes every leastSquares
    # consumer these arms switch (grad(U)'s vector form last, deviceLeastSquaresGradU).
    for M in 1 cuda; do
        local b="${br}_$M" label="$tag/$([ $M = cuda ] && echo CUDA || echo host)"
        stageR "$b" "$gr" "$lim" 8
        rm -rf "$b/0"; cp -r "$of/5" "$b/5"
        if env $envs BRAE_RHOSIMPLEFOAM_MIRROR=$M "$BRAE" -case "$b" > "$b/log.brae" 2>&1 && [ -d "$b/8" ]; then
            FIRST=6 LAST=8 compareR "$b" "$of" "$label" "$bound" || fail=1
        else
            echo "     $label: brae did not run from the restart   FAIL"; grep -v '^brae NOTICE' "$b/log.brae" | tail -2; fail=1
        fi
    done
}
stageR() {   # stageR <dest> <grad> <lim> <endTime>
    local d="$1" gr="$2" lim="$3" et="$4"
    rm -rf "$d"; mkdir -p "$d"
    cp -r "$ROOT/validation/rhoSST/constant" "$ROOT/validation/rhoSST/system" "$d/"
    cp -r "$ROOT/validation/rhoSST/0.orig" "$d/0"
    ET="$et" GR="$gr" LIM="$lim" python3 - "$d" <<'PYEOF'
import os, re, sys
d, et, gr, lim = sys.argv[1], os.environ['ET'], os.environ['GR'], os.environ['LIM']
p = os.path.join(d, 'system/controlDict'); s = open(p).read()
for k, val in (('startFrom', 'latestTime'), ('endTime', et), ('writeInterval', '1'),
               ('writeFormat', 'ascii'), ('writePrecision', '15'), ('writeCompression', 'off')):
    s = re.sub(r'^%s\s+.*' % k, '%-15s %s;' % (k, val), s, flags=re.M) if re.search(r'^%s\s+' % k, s, re.M) else s + '\n%-15s %s;\n' % (k, val)
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S); open(p, 'w').write(s)
p = os.path.join(d, 'system/fvSolution'); s = open(p).read()
s = re.sub(r'\btolerance\s+[0-9.eE+-]+\s*;', 'tolerance 1e-14;', s); s = re.sub(r'\brelTol\s+[0-9.eE+-]+\s*;', 'relTol 0;', s)
s = re.sub(r'residualControl\s*\{[^}]*\}', '', s); open(p, 'w').write(s)
p = os.path.join(d, 'system/fvSchemes'); s = open(p).read()
if gr == 'lsq': s = re.sub(r'gradSchemes\s*\{\s*default\s+Gauss linear;', 'gradSchemes     { default leastSquares;', s)
if gr == 'lsqko': s = re.sub(r'gradSchemes\s*\{\s*default\s+Gauss linear;', 'gradSchemes     { default Gauss linear;\n    grad(k)         leastSquares;\n    grad(omega)     leastSquares;', s)
for f in {'none': (), 'komega': ('k', 'omega'), 'hK': ('h', 'K'), 'all': ('k', 'omega', 'h', 'K')}[lim]:
    s = re.sub(r'div\(phi,%s\)\s+bounded Gauss upwind;' % f, 'div(phi,%s)      bounded Gauss limitedLinear 1;' % f, s)
open(p, 'w').write(s)
PYEOF
    ( cd "$d" && blockMesh > log.blockMesh 2>&1 )
}
compareR() {   # compareR <brae> <of> <label> <bound>; FIRST/LAST from the environment
    LABEL="$3" BOUND="$4" python3 - "$1" "$2" <<'PYEOF'
import math, os, re, sys
b, o, label, bound = sys.argv[1], sys.argv[2], os.environ['LABEL'], float(os.environ['BOUND'])
def internal(p, m=None):
    s = open(p).read()
    mm = re.search(r'internalField\s+nonuniform[^(]*\((.*?)\)\s*;\s*\n\s*boundaryField', s, re.S)
    if mm: return [float(x) for x in re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', mm.group(1))]
    mm = re.search(r'internalField\s+uniform\s+([^;]+);', s)
    v = [float(x) for x in re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', mm.group(1))]
    return v * (m // len(v)) if m and m % len(v) == 0 else v
def rel(a, c):
    den = sum(y * y for y in c); return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, c)) / den) if den else 0.0
bad, worst, wname = 0, 0.0, ''
for it in range(int(os.environ['FIRST']), int(os.environ['LAST']) + 1):
    for f in ('U', 'p', 'T', 'k', 'omega', 'nut'):
        pb, po = '%s/%d/%s' % (b, it, f), '%s/%d/%s' % (o, it, f)
        if not (os.path.exists(pb) and os.path.exists(po)): continue
        a = internal(pb); c = internal(po, len(a)); e = rel(a, c) if len(a) == len(c) else float('inf')
        if e > worst: worst, wname = e, '%s it %d' % (f, it)
        if e >= bound:
            print('     %s it %d  %-6s vs OpenFOAM (L2 rel)  %.6e   FAIL (bound %g)' % (label, it, f, e, bound)); bad = 1
print('     %s (restart, it 6-8) worst %s %.3e  (bound %g)   %s' % (label, wname, worst, bound, 'FAIL' if bad else 'ok'))
sys.exit(1 if bad else 0)
PYEOF
}

echo "== limiters, restarted from OpenFOAM's iteration 5: k/omega and h/K under Gauss linear =="
restartArm gauss_komega gauss komega "$LIMBOUND"
restartArm gauss_hK     gauss hK     "$LIMBOUND"

# ---- the two remaining consumers of a leastSquares default, each restarted the same way ----
#   lsqko+limiter  grad(k)/grad(omega) leastSquares EXPLICITLY, limitedLinear on div(phi,k|omega): the
#                  LIMITER's own gradient (LimitedScheme.C:51-55 resolves grad(<field>)), which both
#                  closures' divWithScheme took as Gauss whatever the case said. Before the port: exact at
#                  iteration 6, k 3.1e-06 / omega 6.4e-05 at iteration 7.
#   lsq default    `default leastSquares`, upwind divergence: grad(U) (production, the momentum's
#                  limitedLinearV/linearUpwind gradients) and grad(p) -- the vector fvc::leastSquaresGrad.
#                  Before the port: U 3.1e-05, k 2.4e-04, nut 1.3e-03 at iteration 6.
#   lsq+all        `default leastSquares` with limitedLinear on div(phi,k|omega|h|K): gasMixing's own
#                  configuration, everything above at once. Before: U 2.8e-05, k 1.8e-04, nut 1.2e-03.
echo "== leastSquares consumers, restarted from OpenFOAM's iteration 5 =="
restartArm lsqko_komega lsqko komega "$LSQLIMBOUND"
restartArm lsq_none     lsq   none   "$LSQALLBOUND"
restartArm lsq_all      lsq   all    "$LSQALLLIMBOUND"

[ "$fail" -eq 0 ] && echo "PASSED" || echo "FAILED"
exit $fail
