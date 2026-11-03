#!/usr/bin/env bash
# fvc::grad(U) TAKES THE CASE'S OWN gradSchemes ENTRY -- in divDevReff and in every turbulence closure.
#
# OpenFOAM builds its viscous stress from fvc::grad(U) (linearViscousStress.C:114) and each closure's
# strain and production from the same call: kOmegaSSTBase.C:522 `tgradU = fvc::grad(U)` feeding S2 (:142)
# and GbyNu0 (:176-177), kEpsilon.C:237/:241, SpalartAllmarasBase.C:461 feeding Omega (:103).
# fvc::grad resolves the gradSchemes `grad(U)` entry, falling back to `default`
# (schemesLookupDetail.C:76-88).
#
# The V2 driver passed a LITERAL 0.0 to the SST closure, let the kEpsilon and divDevReff arguments fall
# through to their defaults, and had no such parameter on SpalartAllmaras at all -- while printing the
# case's cellLimited coefficient on its setup line. It is not the same lookup as the one linearUpwind
# names: on validation/rotorDisk the case says `linearUpwind unlimited` beside `grad(U) $limited`.
#
# Measured on this fixture at OpenFOAM's own converged t=400, limited against unlimited through
# OpenFOAM's own `postProcess -func grad(U)`: |gradU| relative L2 3.21e+01; the production term
# GbyNu = gradU && devTwoSymm(gradU) peaks at 1.05e-03 limited against 5.10e-01 unlimited, a factor 487;
# SpalartAllmaras's Omega 22x; 34% of the 5000 cells carry a clipped gradient and 66% are untouched.
#
# WHY THIS IS A PROXIMITY GATE AND NOT AN ABSOLUTE BOUND. The obvious construction -- require OpenFOAM's
# limited and unlimited answers to separate by 100x brae's floor -- is unreachable here and was measured
# to be: the two OpenFOAM answers differ by U 3.8e-03 / k 7.0e-03 while brae's own floor against
# OpenFOAM is U 7.2e-03 / k 1.7e-02, so no absolute bound can tell them apart. What CAN be asserted, and
# is the whole content of the fix, is which of the two answers brae is CLOSER to.
#
# WHICH FIELDS CAN BE JUDGED is decided BEFORE brae's answer on the limited case is read, from the two
# OpenFOAM answers and the comparison's own uncertainty. Per field, as relative L2 distances:
#   S   = |OF_lim - OF_unl|                    what the limiter does to OpenFOAM's answer
#   u   = max(|brae_unl - OF_unl|,             brae against OpenFOAM where the limiter is not in play
#             |OF_lim - OF_lim'|, |OF_unl - OF_unl'|)   ...and OpenFOAM against itself, k's start one ulp up
#   judgeable  iff  S > (1 + MARGIN) u
# If brae's error on the limited case is within u, as it is on the unlimited one, a CORRECT brae has
# dl <= u and du >= S - u > MARGIN u >= MARGIN dl: the arm passes. A brae that drops the limiter sits within
# u of OF_unl: du <= u < (S - u)/MARGIN <= dl/MARGIN, and the arm fails. The rule fixes both outcomes in
# advance; on a field where S is not that large, the question has no answer at this resolution.
# Until 2026-11-03 a field was judged when brae's OWN answer sat inside the oracles' separation, so a
# better brae could turn a field from skipped to judged: patch 0018 brought nut from 1.95e-02 to 1.52e-02
# of OpenFOAM, nut became judged, and missed the margin at 1.049 against 1.05 while every field moved
# closer to OpenFOAM.
#
# THE FIRST-ITERATION PASS. After N = 20 iterations (JUPITER job 2663824, the 18-patch candidate) U alone
# was judgeable, S/u 8.23 against 2.05; k 0.75, epsilon 1.04, nut 0.95 and p 1.98 were not, because OpenFOAM
# moved k by 7.6e-03 from a start one ulp up against the limiter's 5.7e-03. So the same rule is applied again
# after N_TIGHT iterations, its judgeable fields again decided from OpenFOAM and brae's unlimited run alone.
# MEASURED there (JUPITER job 2666385): U and p judgeable at S/u 1.3e+11 and 5.5e+09, brae 1.2e-13 from the
# limited oracle, and the limiter dropped everywhere (1) or in divDevReff alone (divdev) fails the arm on both.
# k 0.27, epsilon 0.40 and nut 0.09 are not judgeable: OpenFOAM's own one-ulp k start moves k by 2.5e-02 and
# nut by 4.7e-02 in that one iteration, and brae's unlimited epsilon sits 1.9e-02 from OpenFOAM's where
# OpenFOAM's twin moves it 5.7e-15 -- but that twin moves k's start alone, and with epsilon's uniform start one
# ulp up OpenFOAM moves its own epsilon 2.4e-02 (job 2667846). Both fields start uniform under limitedLinear
# here, where grad() is round-off and round-off picks the limiter's branch (public 9998d262,
# tests/limitedlinear_robustness_vs_openfoam.sh). A brae that drops the limiter in the closures alone
# (closure) is NOT gated by either pass: after 20 iterations it kept U at 10.47 against 1.05, and in the first
# iteration it changes only k, epsilon and nut, which are solved after U and p. A pass one iteration from
# OpenFOAM's converged 400/, where nothing is uniform, was tried (job 2667459) and judged p alone (3.65): the V2
# driver this gate runs sits 7.6e-05 (U) and 2.9e-03 (p) from OpenFOAM there, as shipped, where the legacy
# driver sits 2.6e-16 and 1.9e-12 (job 2667755), so the closures stay unjudged until the V2 driver's restart
# matches. The proof above covers 1; divdev drops the limiter at one site, so its failing the arm is asserted,
# not derived, and a control that passes is a FAIL of the gate.
#
#   ARM      on every judgeable field, rel(brae, OF_unlimited) > MARGIN rel(brae, OF_limited).
#   CONTROL  at least one field is judgeable, or the arm asserts nothing.
#   CONTROL2 brae must print the gradSchemes grad(U) line, i.e. it read the entry at all.
#   CONTROL3 BRAE_CONTROL_GRADU_LIMIT_IGNORED=<each of GATED_CONTROLS after N iterations, of
#            GATED_CONTROLS_TIGHT after N_TIGHT> -- brae with the case's grad(U) limiter dropped -- must FAIL
#            the arm on the same judgeable fields.
#
# Fail-proof, 2026-09-04: with gradUSchemeUnsupported returning 0.0, the inequality flips on U and k.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="${1:-$ROOT/validation/windAroundBuildingsBox}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
N=${N:-20}
N_TIGHT=${N_TIGHT:-1}              # the first-iteration pass
REL_MIN=${REL_MIN:-1e-04}          # below this the two oracles are the same answer
MARGIN=${MARGIN:-1.05}             # brae must be at least this much closer to the limited oracle
# BRAE_CONTROL_GRADU_LIMIT_IGNORED values that must fail the arm; set and empty means none (a binary
# without the control)
GATED_CONTROLS=${GATED_CONTROLS-1}
GATED_CONTROLS_TIGHT=${GATED_CONTROLS_TIGHT-1 divdev}

[ -x "$BRAE" ]     || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
SRC="$(cd "$SRC" && pwd)"
set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
command -v simpleFoam > /dev/null 2>&1 || { echo "SKIP: simpleFoam not on PATH"; exit 77; }

W=$(mktemp -d); [ -n "${KEEP_W:-}" ] && echo "work dir $W" || trap 'rm -rf "$W"' EXIT

stage()   # stage <dir> <limited|unlimited> <iterations> [twin]
{
    # This fixture ships 0/ directly (no 0.orig) and a committed 400/ reference; both are kept out of
    # the way so the run starts where the dictionary says and writes exactly one time directory.
    rm -rf "$1"; cp -r "$SRC" "$1"; rm -rf "$1"/[1-9]* "$1"/log.* "$1"/postProcessing
    [ -d "$1/0.orig" ] && { rm -rf "$1/0"; cp -r "$1/0.orig" "$1/0"; }
    [ -d "$1/0" ] || { echo "SKIP: fixture ships no 0/"; exit 77; }
    python3 - "$1" "$2" "$3" "${4:-}" <<'PY'
import math, re, sys
d, mode, n, twin = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*\n\}', 'functions\n{\n}', s, flags=re.S)
s = re.sub(r'endTime\s+\S+;',        'endTime         %s;' % n, s)
s = re.sub(r'writeInterval\s+\S+;',  'writeInterval   %s;' % n, s)
s = re.sub(r'writeControl\s+\S+;',   'writeControl    timeStep;', s)
s = re.sub(r'writeFormat\s+\S+;',    'writeFormat     ascii;', s)
s = re.sub(r'writePrecision\s+\S+;', 'writePrecision  15;', s)
s = re.sub(r'writeCompression\s+\S+;', 'writeCompression off;', s)
open(c, 'w').write(s)
f = d + '/system/fvSolution'; s = open(f).read()
# Every linear solve pinned, so the arms differ by the GRADIENT and not by where a loose solve stopped.
s = re.sub(r'residualControl\s*\{[^}]*\}', 'residualControl { }', s, flags=re.S)
s = re.sub(r'tolerance\s+[0-9.eE+-]+;', 'tolerance       1e-12;', s)
s = re.sub(r'relTol\s+[0-9.eE+-]+;',    'relTol          0;', s)
open(f, 'w').write(s)
if mode == 'unlimited':
    g = d + '/system/fvSchemes'; s = open(g).read()
    s = re.sub(r'grad\(U\)\s+\S+;', 'grad(U)         Gauss linear;', s)
    open(g, 'w').write(s)
if twin:
    # OpenFOAM's own spread: k's starting value one ulp up, the boundary values as shipped
    f = d + '/0/k'; s = open(f).read()
    v = float(re.search(r'^kInlet\s+([0-9.eE+-]+);', s, flags=re.M).group(1))
    s, nsub = re.subn(r'internalField\s+uniform\s+\$kInlet;', 'internalField   uniform %r;' % math.nextafter(v, 2*v), s)
    if nsub != 1: sys.exit('twin: 0/k has no internalField uniform $kInlet')
    open(f, 'w').write(s)
PY
}

cat > "$W/judge.py" <<'PY'
import gzip, os, re, sys
import numpy as np
W, n, relmin, margin = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
controls = sys.argv[5].split()
fail = 0
def say(msg, ok):
    global fail
    print('  %-70s %s' % (msg, 'ok' if ok else 'FAIL'))
    if not ok: fail = 1
def rd(p):
    if os.path.exists(p):       s = open(p, 'rb').read()
    elif os.path.exists(p+'.gz'): s = gzip.open(p+'.gz', 'rb').read()
    else: return None
    s = s.decode('utf-8', 'replace')
    t = s[s.index('internalField'):]
    m = re.match(r'internalField\s+uniform\s+([^;]+);', t)
    if m:
        tok = m.group(1).strip()
        return np.array([[float(x) for x in tok.strip('()').split()]]) if tok.startswith('(') \
               else np.array([float(tok)])
    m = re.match(r'internalField\s+nonuniform\s+List<(\w+)>\s*\n?\s*(\d+)\s*\(', t)
    if not m: return None
    kind, cnt = m.group(1), int(m.group(2))
    b = t[m.end():]; b = b[:b.index('\n)')]
    if kind == 'scalar':
        return np.fromstring(b.replace('\n', ' '), sep=' ')
    return np.fromstring(b.replace('(', ' ').replace(')', ' ').replace('\n', ' '), sep=' ').reshape(cnt, 3)
def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-300))

print('  brae ran with the case as shipped; the two OpenFOAM runs differ ONLY in the grad(U) entry.')
FIELDS = ('U', 'k', 'epsilon', 'nut', 'p')
def fields(arm):
    return {f: rd(os.path.join(W, arm, n, f)) for f in FIELDS}
lim, unl, limt, unlt, bru = (fields(a) for a in ('of_lim', 'of_unl', 'of_limt', 'of_unlt', 'br_unl'))
# 1. The judgeable fields, from OpenFOAM and brae's UNLIMITED run only. brae's limited answer is not read yet.
judged = []
for f in FIELDS:
    if any(x[f] is None for x in (lim, unl, limt, unlt, bru)):
        print('  %-8s missing on one side -- skipped' % f); continue
    sep = rel(lim[f], unl[f])
    if sep < relmin:
        print('  %-8s the two oracles agree to %.3e -- the limiter does not reach this field' % (f, sep))
        continue
    floor, twin = rel(unl[f], bru[f]), max(rel(lim[f], limt[f]), rel(unl[f], unlt[f]))
    u = max(floor, twin)
    ok = sep > (1.0 + margin) * u
    print('  %-8s S %.3e  u %.3e (brae floor %.3e, OpenFOAM twin %.3e)  S/u %7.2f against %.2f: %s'
          % (f, sep, u, floor, twin, sep / u if u > 0 else float('inf'), 1.0 + margin,
             'judgeable' if ok else 'NOT judgeable'))
    if ok: judged.append(f)
say('CONTROL  at least one field is judgeable (S > (1 + MARGIN) u)', len(judged) > 0)
# 2. The arm, then each control, on those fields only.
def verdict(arm, label):
    br = fields(arm)
    passed = True
    for f in FIELDS:
        if any(x[f] is None for x in (lim, unl, br)): continue
        dl, du = rel(lim[f], br[f]), rel(unl[f], br[f])
        good = du > margin * dl
        if f in judged: passed = passed and good
        print('  %-8s %-12s %.3e from LIMITED, %.3e from unlimited, ratio %7.3f against %.2f  %s'
              % (f, label, dl, du, du / dl if dl > 0 else float('inf'), margin,
                 ('ok' if good else 'fails') if f in judged else '(not judged)'))
    return passed
say('ARM      brae is closer to the LIMITED oracle on every judgeable field', verdict('br', 'brae'))
for c in controls:
    log = open(os.path.join(W, 'ctl_' + c, 'run.log')).read()
    say('CONTROL3 BRAE_CONTROL_GRADU_LIMIT_IGNORED=%s reached brae' % c,
        'BRAE_CONTROL_GRADU_LIMIT_IGNORED=%s' % c in log)
    say('CONTROL3 BRAE_CONTROL_GRADU_LIMIT_IGNORED=%s fails the arm' % c,
        bool(judged) and not verdict('ctl_' + c, 'control ' + c))
say('CONTROL2 brae read the gradSchemes grad(U) entry',
    'gradSchemes grad(U): cellLimited' in open(os.path.join(W, 'br', 'run.log')).read())
sys.exit(1 if fail else 0)
PY

pass()   # pass <iterations> <controls>: the whole comparison after that many iterations, in $W/n<iterations>
{
    local n=$1 controls=$2 P=$W/n$1 o b c
    mkdir -p "$P"
    stage "$P/of_lim"  limited   "$n"
    stage "$P/of_unl"  unlimited "$n"
    stage "$P/of_limt" limited   "$n" twin
    stage "$P/of_unlt" unlimited "$n" twin
    stage "$P/br_unl"  unlimited "$n"
    stage "$P/br"      limited   "$n"
    for o in of_lim of_unl of_limt of_unlt; do
        ( cd "$P/$o" && simpleFoam > run.log 2>&1 ) || { echo "FAIL: OpenFOAM ($o) did not run"; tail -8 "$P/$o/run.log"; exit 1; }
    done
    for b in br_unl br; do
        ( cd "$P/$b" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" "$P/$b" > run.log 2>&1 ) \
            || { echo "FAIL: brae crashed ($b)"; tail -15 "$P/$b/run.log"; exit 1; }
    done
    for c in $controls; do
        stage "$P/ctl_$c" limited "$n"
        ( cd "$P/ctl_$c" && BRAE_SIMPLEFOAM_V2=1 BRAE_CONTROL_GRADU_LIMIT_IGNORED=$c "$BRAE" "$P/ctl_$c" > run.log 2>&1 ) \
            || { echo "FAIL: brae crashed (control $c)"; tail -15 "$P/ctl_$c/run.log"; exit 1; }
    done
    python3 "$W/judge.py" "$P" "$n" "$REL_MIN" "$MARGIN" "$controls"
}

echo "== after $N iterations from the case's start"
pass "$N" "$GATED_CONTROLS"
rc=$?
echo "== after $N_TIGHT iteration(s) from the case's start"
pass "$N_TIGHT" "$GATED_CONTROLS_TIGHT"
rc_tight=$?
[ $rc -eq 0 ] && [ $rc_tight -eq 0 ] && echo "PASS: fvc::grad(U) takes the case's gradSchemes entry in divDevReff and the closures"
[ $rc -eq 0 ] && [ $rc_tight -eq 0 ]
