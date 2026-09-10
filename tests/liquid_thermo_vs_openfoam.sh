#!/usr/bin/env bash
# `properties liquid` END TO END on the host mirror arm, against real OpenFOAM v2412 -- stage H3.4.
#
# This is the first rhoSimpleFoam case brae runs whose thermo is NOT a perfect gas. Everything the
# solver asks of a thermo is a different function here: mu, Cp and kappa are NSRDS correlations in T,
# rho is a correlation and not p/(RT), psi is identically ZERO, Es carries -p/rho(T), and T comes back
# out of he through OpenFOAM's own do-while inversion rather than a closed form. Nothing about the
# momentum or pressure equations changed; what changed is that the eight accessors in
# liquid_thermo.cuh are now the only thing the solver asks, so ONE branch decides which thermo runs.
#
# THE FIXTURE, validation/liqBox, is 1200 cells and laminar on purpose: a closure between the thermo and
# the answer is one more thing that has to be right before the measurement means anything. The duct is
# narrow so it develops ~8.1e+02 Pa across the domain -- ARM B needs that, see below.
#
# ARM 1  the thermo, whole and unclamped, after ONE iteration -- the first evaluation of every
#        property, before anything has fed back.
# ARM A  the same, at iteration 200 with the residuals at ~1e-10. T, p, rho and U against OpenFOAM's.
# ARM B  the same run with fv::limitTemperature active from both sides (min 302, max 305, and the
#        unclamped field runs 300.0 to 306.5 so BOTH bite). This is the arm that gates stage H3.4's
#        second half: OpenFOAM builds limitTemperature's he bounds as FIELDS over the cell pressure,
#            heMin = thermo.he(thermo.p(), Tmin, cells_)     (limitTemperature.C:156-157)
#            heMinp = thermo.he(pp, Tminp, patchi)                            :243-247
#        and for a liquid he depends on p, so a single scalar bound clamps to a temperature that is NOT
#        Tmin. The discriminating assertion is a COUNT taken from OpenFOAM's own output: how many cells
#        sit at EXACTLY Tmin and at exactly Tmax. OpenFOAM has 848 and 52; a scalar bound built at cell
#        0's pressure measured 2 and 0, because a clamped cell lands at Tmin + (pRef - p_c)/(rho*Cpv)
#        instead -- up to 1.9e-04 K here, five orders above the 1e-09 K the count is taken at -- and only
#        the cells whose pressure happens to equal the reference one land on Tmin at all.
#
# NOT VACUOUS: ARM A asserts rho is around 993 kg/m3, which a perfect-gas path cannot produce at these
# p and T (it gives ~1.1), so the arm cannot silently be running the gas closed forms; and ARM B asserts
# the two clamp counts are NON-ZERO before comparing them, so a limit that never bit could not pass.
#
# Measured, OpenFOAM v2412:
#     ARM 1   iteration 1:   T 6.00e-13   p 2.86e-12   rho 3.10e-13   U 5.98e-13
#     ARM A   iteration 200: T 9.69e-13   p 3.66e-12   rho 2.83e-13   U 7.18e-13
#     ARM B   iteration 200: T 4.83e-13   p 6.12e-12   rho 2.62e-13   clamped 848 low / 52 high, both codes
# THE CUDA ARM (stage H3.6) runs every arm against the same OpenFOAM runs. Measured: T 5.1e-13 to
# 9.7e-13 and rho 2.6e-13 to 3.1e-13 -- the host's floor -- the clamp counts 848/52 and the report lines
# exact; p 2.9e-11 and U 1.5e-11 at worst, the device linear solvers' floor (see the U bound below).
# CUDA fail-proofs, each device module broken in the source and the gate re-run -- every one red on the
# CUDA arm and green on the host arm:
#   the device he -> T inversion seeded with 300 K   -> ARM A T 1.03e-10; ARM B Tmin count 0 of 848
#   the live energy boundary update skipped          -> T 6.39e-07, rho 5.06e-08
#   limitTemperature's device bounds at ONE pressure -> ARM B T 3.21e-07
#
# Host fail-proofs measured through this gate by editing the source and re-running:
#   limitTemperature's he bounds taken as ONE scalar at cell 0's pressure -- ARM B goes red on seven
#     checks at once: T 3.21e-07, p 5.77e-09, rho 2.57e-08, U 2.56e-08, the Tmin count 2 against 848,
#     the Tmax count 0 against 52, and OpenFOAM's own report lines (774 LimitedCells against 848,
#     UnlimitedTmin 301.999807757389 against 302).
#   the he -> T inversion seeded with a fixed T0 = 300 instead of the cell's own -- ARM A's T reads
#     1.03e-10 against the 1e-11 bound and ARM B's clamp counts fall to 0 of 848 and 0 of 52. Note the
#     MARGIN: OpenFOAM's inversion stops on a temperature STEP measured against its initial guess, so
#     the seed is worth about 1e-10 relative and no more -- one order over the bound in the field
#     comparison, and decisive only through the clamp counts. That is the honest size of it; a gate
#     claiming more from the field alone would be claiming resolution it does not have.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae_rhoSimpleFoam}"
SRC="$ROOT/validation/liqBox"
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
say() { printf '     %-56s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

# stage <dir> <fvOptions body, or "" for none>
stage() {
    rm -rf "$1"; cp -r "$SRC" "$1"
    rm -rf "$1"/[1-9]* "$1"/0 "$1"/log.*
    cp -r "$1/0.orig" "$1/0"
    if [ -n "$2" ]; then printf '%s\n' "$2" > "$1/system/fvOptions"; else rm -f "$1/system/fvOptions"; fi
}

# The fixture ships endTime 200 in a single-line controlDict, so the edit is token-wise; an anchored
# sed silently rewrites the whole line of unrelated keys.
pinEnd() {
    python3 - "$1/system/controlDict" "$2" <<'PYEOF'
import re, sys
p, n = sys.argv[1], sys.argv[2]
s = open(p).read()
s = re.sub(r'\bendTime\s+\S+;',       'endTime %s;' % n,       s)
s = re.sub(r'\bwriteInterval\s+\S+;', 'writeInterval %s;' % n, s)
open(p, 'w').write(s)
PYEOF
}

FVOPT='FoamFile { version 2.0; format ascii; class dictionary; object fvOptions; }
limitT { type limitTemperature; min 302; max 305; selectionMode all; }'

# ARM 1 comes first and is the SAME case as ARM A stopped after ONE iteration. The liquid work was
# agreed to be validated incrementally -- init, one iteration, several, converged -- because a
# temperature-dependent property that is subtly wrong is nearly invisible at iteration 1 and only shows
# once Cp(T), mu(T), kappa(T) and rho(T) feed back; and conversely a defect in the FIRST evaluation of
# any of them is clearest before anything has fed back at all. ARM A is the "several" (200 iterations,
# residuals at ~1e-10), and the two together bracket the trajectory.
for ARM in 1 A B; do
    OPT=""
    [ "$ARM" = B ] && OPT="$FVOPT"
    END=200
    [ "$ARM" = 1 ] && END=1

    stage "$W/of$ARM" "$OPT"
    pinEnd "$W/of$ARM" "$END"
    ( cd "$W/of$ARM" && blockMesh > log.blockMesh 2>&1 && rhoSimpleFoam > log.rhoSimpleFoam 2>&1 ) || {
        echo "FAIL: ARM $ARM -- OpenFOAM did not run"; tail -20 "$W/of$ARM/log.rhoSimpleFoam"; exit 1; }

    case "$ARM" in
        1) DESC='thermo alone, ONE iteration' ;;
        A) DESC='thermo alone, 200 iterations' ;;
        B) DESC='limitTemperature min 302 max 305, 200 iterations' ;;
    esac

    # BOTH ARMS against the same OpenFOAM run: the host step, and since stage H3.6 the CUDA arm, whose
    # kernels now ask the same liquid_thermo.cuh accessors for every property, invert he -> T with the
    # same seeded loop, rebuild the energy boundary conditions live and clamp limitTemperature per cell.
    # BRAE_U_SOLVER=ofOrder so the momentum solve is the case's own and not the colour-GS substitute --
    # at tolerance 1e-14 relTol 0 both converge, but the arms should not differ in more than one thing.
    for MIRROR in 1 cuda; do
    BR="$W/br${ARM}_$MIRROR"
    stage "$BR" "$OPT"
    pinEnd "$BR" "$END"
    cp -r "$W/of$ARM/constant/polyMesh" "$BR/constant/"
    BRAE_U_SOLVER=ofOrder BRAE_RHOSIMPLEFOAM_MIRROR=$MIRROR "$BRAE" -case "$BR" > "$BR/log.brae" 2>&1 || {
        echo "FAIL: ARM $ARM ($MIRROR) -- brae did not run"; grep -v '^brae NOTICE' "$BR/log.brae" | tail -8
        fail=1; continue; }

    echo "== ARM $ARM ($DESC) -- $([ "$MIRROR" = cuda ] && echo 'CUDA arm' || echo 'host arm') =="
    MIRROR="$MIRROR" ARM="$ARM" python3 - "$BR/$END" "$W/of$ARM/$END" <<'PYEOF' || fail=1
import math, os, re, sys

brae, of = sys.argv[1], sys.argv[2]
arm = os.environ['ARM']
bad = 0

def say(what, verdict):
    global bad
    print('     %-56s %s' % (what, verdict))
    if verdict == 'FAIL': bad = 1

def scalars(path):
    s = open(path).read()
    m = re.search(r'internalField\s+nonuniform[^(]*\((.*?)\)\s*;', s, re.S)
    if m: return [float(x) for x in m.group(1).split()]
    m = re.search(r'internalField\s+uniform\s+([-\d.eE+]+)\s*;', s)
    raise SystemExit('%s: internalField is uniform -- the field carries no information' % path)

def vectors(path):
    s = open(path).read()
    m = re.search(r'internalField\s+nonuniform[^(]*\(\s*(.*?)\s*\)\s*;\s*\n\s*boundaryField', s, re.S)
    return [float(x) for x in re.findall(r'-?[\d.]+(?:[eE][-+]?\d+)?', m.group(1))]

def rel(a, b):
    n = sum((x - y) ** 2 for x, y in zip(a, b))
    d = sum(y * y for y in b)
    return math.sqrt(n / d) if d > 0 else math.sqrt(n)

def report(what, got, bound):
    print('     %-56s %.6e   %s' % (what, got, 'ok' if got < bound else 'FAIL (bound %g)' % bound))
    global bad
    if not got < bound: bad = 1

fields = {}
for f in ('T', 'p', 'rho'):
    fields[f] = (scalars(brae + '/' + f), scalars(of + '/' + f))
fields['U'] = (vectors(brae + '/U'), vectors(of + '/U'))

# THE ARM IS A LIQUID, asserted before anything is compared. At this p and T a perfect gas is ~1.1
# kg/m3; H2O's NSRDS rho correlation gives ~993. Without this the whole gate could be green with the
# thermo silently running the gas closed forms on a case that happens to be insensitive.
rmin, rmax = min(fields['rho'][1]), max(fields['rho'][1])
say('OpenFOAM rho is a LIQUID (980..1010 kg/m3): %.2f..%.2f' % (rmin, rmax),
    'ok' if 980.0 < rmin and rmax < 1010.0 else 'FAIL')

# THE THERMO IS HELD TO THE SAME BOUND ON BOTH ARMS -- T and rho are what the port is about, and the
# CUDA arm reaches the host's floor on both (5e-13, 3e-13). U gets a CUDA bound of its own for the
# reason every CUDA arm in the tree does (rho_mirror_solver_vs_openfoam: "looser than the host arm by
# exactly the linear solvers between them"): each fully converged device solve carries its own round-off
# floor -- the device pressure solve takes 108 BiCGStab iterations where OpenFOAM's takes 58 -- and U
# inherits it. Measured on the CUDA arm: p 2.9e-11 and U 1.5e-11 at worst, against the host's 6e-12 and
# 1.6e-12; the thermo-sensitive checks below (the clamp counts, the report lines) are exact on both.
cuda = os.environ.get('MIRROR') == 'cuda'
report('T   vs OpenFOAM (L2 rel)',   rel(*fields['T']),   1e-11)
report('p   vs OpenFOAM (L2 rel)',   rel(*fields['p']),   1e-10)
report('rho vs OpenFOAM (L2 rel)',   rel(*fields['rho']), 1e-11)
report('U   vs OpenFOAM (L2 rel)',   rel(*fields['U']),   1e-10 if cuda else 1e-11)

if arm == 'B':
    # limitTemperature's bounds are FIELDS over p. A clamped cell therefore lands on EXACTLY Tmin (or
    # Tmax) -- that is only true when the bound was built at that cell's own pressure. Counted from
    # OpenFOAM's output first, so the number brae must match is OpenFOAM's and not this script's.
    TMIN, TMAX, EPS = 302.0, 305.0, 1e-9
    def count(v, t): return sum(1 for x in v if abs(x - t) < EPS)
    ofLo, ofHi = count(fields['T'][1], TMIN), count(fields['T'][1], TMAX)
    brLo, brHi = count(fields['T'][0], TMIN), count(fields['T'][0], TMAX)
    say('the clamp BITES from both sides in OpenFOAM (%d low, %d high)' % (ofLo, ofHi),
        'ok' if ofLo > 0 and ofHi > 0 else 'FAIL')
    say('cells at exactly Tmin: brae %d, OpenFOAM %d' % (brLo, ofLo), 'ok' if brLo == ofLo else 'FAIL')
    say('cells at exactly Tmax: brae %d, OpenFOAM %d' % (brHi, ofHi), 'ok' if brHi == ofHi else 'FAIL')
    # The pressure has to VARY, or a scalar bound and a per-cell one are the same number and the two
    # counts above discriminate nothing.
    pv = fields['p'][1]
    say('p varies across the domain by %.1f Pa (>100 needed)' % (max(pv) - min(pv)),
        'ok' if (max(pv) - min(pv)) > 100.0 else 'FAIL')

sys.exit(bad)
PYEOF

    if [ "$ARM" = B ]; then
        # limitTemperature's own report lines, brae's against OpenFOAM's, verbatim.
        grep -h "^limitTemperature=" "$W/ofB/log.rhoSimpleFoam" | tail -2 > "$W/of.lim"
        grep -h "^limitTemperature=" "$BR/log.brae"             | tail -2 > "$W/br.lim"
        if diff -q "$W/of.lim" "$W/br.lim" > /dev/null; then
            say "limitTemperature report lines match OpenFOAM's" ok
        else
            say "limitTemperature report lines match OpenFOAM's" FAIL
            diff "$W/of.lim" "$W/br.lim" || true
        fi
    fi
    done
done

[ "$fail" -eq 0 ] && echo "PASSED" || echo "FAILED"
exit $fail
