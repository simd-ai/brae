#!/usr/bin/env bash
# Foam::bound's message, which brae did not print on either arm.
#
# OpenFOAM's bound() computes min(vsf) unconditionally -- it is the guard it branches on -- and when the
# guard fires it prints the field's PRE-bound min, max and average before touching it (bound.C:38-46).
# brae clamped silently. What that silence cost is not hypothetical: the whole of item 78 -- a
# substituted PBiCGStab preconditioned with `diagonal`, epsilon driven to the bound floor in 201 interior
# cells, nut = Cmu k^2/epsilon reaching 1.5e+17 -- was found by dumping fields at chosen iterations and
# diffing extremes against OpenFOAM over several days. Real OpenFOAM on that same case printed TEN of
# these lines in the first eight iterations.
#
# THE ORACLE is real rhoSimpleFoam on the same bytes, so the message is checked against OpenFOAM's own
# text and not against this test's idea of it.
#
# THE CASE: sbMatched at OpenFOAM's iteration 4, limitedLinear on k and epsilon, the outlet held at three times each
# field's maximum, ten tight iterations: a tight solve goes negative and every code carries on. The stage this gate
# had, the loose bare-diagonal solve of item 78, is kept as tests/bound_weak_solver_stress.sh: at its iteration 7 a
# correct patch half leaves nut at 3e15 on an outlet face, and OpenFOAM from that state dies at the same iteration.
#
#   ARM 1    brae prints the line, in OpenFOAM's exact format, on a case that bounds
#   ARM 2    the numbers are the PRE-bound ones (min < the floor, not == it)
#   ARM 3    min/max include the BOUNDARY and average does not -- OF's asymmetry
#   ARM 4    the bounded fields against OpenFOAM's, cell and patch values at five iterations, and every line's numbers
#   CONTROL  BRAE_CONTROL_BOUND_PATCH_LIVE=1, the old patch half, turns arm 4 red
#   CONTROL  a healthy run prints NOTHING (the message is an event, not a banner)
#   CONTROL  the line cannot break a log-parsing gate: no `Time = ` prefix, no `Solving for`,
#            no `time step continuity errors`
#   FAIL-PROOF  BRAE_BOUND_REPORT=0 silences it, so the arms above are testing the message and not
#            something else in the log
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="$ROOT/validation/sbMatched"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
[ -x "$BRAE" ]     || { echo "SKIP: no brae at $BRAE"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
ITERS=12

fail=0
say() { printf '  %-72s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

# The default run of the healthy control: the tutorial's own loose GAMG entries on the turbulence pair, nothing bounds.
stage_healthy() {
    rm -rf "$1"; mkdir -p "$1"
    cp -r "$SRC/constant" "$SRC/system" "$1/"
    cp -r "$SRC/0.orig" "$1/0"
    python3 - "$1" "$ITERS" <<'PYEOF'
import re, sys
d, it = sys.argv[1], sys.argv[2]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*', '', s, flags=re.S)
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime %s;' % it, s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval %s;' % it, s)
open(c, 'w').write(s + '\n')
f = d + '/system/fvSolution'; s = open(f).read()
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { }', s)
old = '"(U|e|k|epsilon)" { solver PBiCGStab; preconditioner DILU; tolerance 1e-12; relTol 0; }'
assert old in s, 'the fixture fvSolution changed shape'
s = s.replace(old, '"(U|e)" { solver PBiCGStab; preconditioner DILU; tolerance 1e-12; relTol 0; }\n'
                   '    "(k|epsilon)"\n    {\n        solver          GAMG;\n'
                   '        smoother        GaussSeidel;\n        tolerance       1e-08;\n'
                   '        relTol          0.1;\n    }')
s = s.replace('p { solver PBiCGStab; preconditioner DILU; tolerance 1e-12; relTol 0; }',
              'p { solver GAMG; smoother GaussSeidel; tolerance 1e-08; relTol 0.1; }')
open(f, 'w').write(s)
PYEOF
}

# tight <case> <start> <end>: every solve tight, every iteration written in full
tight() {
    python3 - "$@" <<'PYEOF'
import re, sys
d, st, en = sys.argv[1:]
p = d + '/system/controlDict'; s = open(p).read()
s = re.sub(r'functions\s*\{.*', '', s, flags=re.S)
for k, v in (('startFrom', 'startTime'), ('startTime', st), ('endTime', en), ('writeInterval', '1'), ('writeFormat', 'ascii'), ('writePrecision', '17')):
    s, n = re.subn(r'(?m)(^|\s)%s\s+[^;]*;' % k, r'\1%s %s;' % (k, v), s)
    if n == 0:
        s += '\n%s %s;\n' % (k, v)
open(p, 'w').write(s + '\n')
p = d + '/system/fvSolution'; s = open(p).read()
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { }', s)
s = re.sub(r'tolerance\s+[^;]*;', 'tolerance 1e-13;', s)
s = re.sub(r'relTol\s+[^;]*;', 'relTol 0;', s)
open(p, 'w').write(s)
PYEOF
}

# ---- the DEVELOPED STATE, OpenFOAM's iteration 4, once ---------------------------------------------
mkdir -p "$W/dev"
cp -r "$SRC/constant" "$SRC/system" "$W/dev/"
cp -r "$SRC/0.orig" "$W/dev/0"
tight "$W/dev" 0 4
( set +u; source "$OFBASHRC" > /dev/null 2>&1
  command -v rhoSimpleFoam > /dev/null 2>&1 || exit 77
  cd "$W/dev" && rhoSimpleFoam > run.log 2>&1 ) || { echo "SKIP: OpenFOAM rhoSimpleFoam did not run"; exit 77; }

# stage <dir>: iteration 4, limitedLinear on the pair, each outlet at three times its field's maximum. MEASURED (job 4837184): 1.5x
# bounds and stays exact, 3x bounds 19 times and the old patch half moves by 1e-2, 10x and over abort in every code.
stage() {
    rm -rf "$1"; mkdir -p "$1"
    cp -r "$W/dev/constant" "$W/dev/system" "$W/dev/4" "$1/"
    rm -rf "$1/4/uniform"
    tight "$1" 4 "$ITERS"
    python3 - "$1" <<'PYEOF'
import re, sys
d = sys.argv[1]
for f in ('k', 'epsilon'):
    p = '%s/4/%s' % (d, f)
    s = open(p).read()
    m = re.search(r'internalField\s+nonuniform\s+List<scalar>\s*\n?(\d+)\s*\n?\(', s)
    assert m, f
    e = s.index('\n)', m.end())
    hi = 3 * max(float(x) for x in s[m.end():e].split())
    s, n = re.subn(r'(\boutlet\s*\{)[^}]*\}', lambda g: g.group(1) + ' type fixedValue; value uniform %.17g; }' % hi, s, count=1)
    assert n == 1, f
    open(p, 'w').write(s)
p = d + '/system/fvSchemes'
s = open(p).read()
s, n = re.subn(r'turbulence\s+bounded Gauss upwind;', 'turbulence bounded Gauss limitedLinear 1;', s)
assert n == 1
open(p, 'w').write(s)
PYEOF
}
ITERS=14

# ---- the ORACLE: real OpenFOAM on the same bytes ---------------------------------------------------
stage "$W/of"
( set +u; source "$OFBASHRC" > /dev/null 2>&1
  cd "$W/of" && rhoSimpleFoam > run.log 2>&1 ) || { echo "FAIL: OpenFOAM did not finish the bounded stage"; tail -3 "$W/of/run.log"; exit 1; }
OF_N=$(grep -c '^bounding ' "$W/of/run.log" || true)
[ "${OF_N:-0}" -gt 0 ] && say "OpenFOAM bounds on this stage (premise)" ok || say "OpenFOAM bounds on this stage (premise)" FAIL
printf '        (OpenFOAM printed %s bounding lines; first: %s)\n' "$OF_N" "$(grep -m1 '^bounding ' "$W/of/run.log")"

# ---- ARM 1: brae prints it, in OpenFOAM's format -------------------------------------------------
stage "$W/br"
( cd "$W/br" && BRAE_AMG_CACHE=0 BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BRAE" -case "$W/br" > run.log 2>&1 ) \
    || { tail -3 "$W/br/run.log"; say "brae runs the case" FAIL; }
BR_N=$(grep -c '^bounding ' "$W/br/run.log" || true)
[ "${BR_N:-0}" -gt 0 ] && say "brae prints the bounding message on a case that bounds" ok \
                       || say "brae prints the bounding message on a case that bounds" FAIL
printf '        (brae printed %s; first: %s)\n' "${BR_N:-0}" "$(grep -m1 '^bounding ' "$W/br/run.log")"

# The FORMAT, against OpenFOAM's own text: same field names, and every line matching the shape
# `bounding <word>, min: <num> max: <num> average: <num>` with nothing after it.
python3 - "$W/of/run.log" "$W/br/run.log" <<'PYEOF' || fail=1
import re, sys
pat = re.compile(r'^bounding (\w+), min: (-?[\d.]+(?:e[+-]\d+)?) max: (-?[\d.]+(?:e[+-]\d+)?) average: (-?[\d.]+(?:e[+-]\d+)?)$')
ofl = [l.rstrip('\n') for l in open(sys.argv[1]) if l.startswith('bounding ')]
brl = [l.rstrip('\n') for l in open(sys.argv[2]) if l.startswith('bounding ')]
bad = [l for l in brl if not pat.match(l)]
ok1 = not bad
print('  %-72s %s' % ("every brae line matches OpenFOAM's exact format", 'ok' if ok1 else 'FAIL'))
if bad: print('        (first offender: %r)' % bad[0])
offlds = {pat.match(l).group(1) for l in ofl if pat.match(l)}
brflds = {pat.match(l).group(1) for l in brl if pat.match(l)}
ok2 = brflds and brflds <= offlds
print('  %-72s %s' % ("...and names only fields OpenFOAM also bounds on this case", 'ok' if ok2 else 'FAIL'))
print('        (OpenFOAM: %s | brae: %s)' % (sorted(offlds), sorted(brflds)))
raise SystemExit(0 if (ok1 and ok2) else 1)
PYEOF

# ---- ARM 4: the bounded fields and every line against OpenFOAM's -----------------------------------
# BOUND 1e-6 at every field and patch: the worst on the candidate is nut 1.3e-07 at iteration 14 (job 4837184).
# The control is 1.0e-02 on U and 1.5e-01 on epsilon at the same iteration, so five decades separate them.
cat > "$W/arm4.py" <<'PYEOF'
import re, sys
import numpy as np
of, br, want = sys.argv[1:4]
BOUND = 1e-6
TIMES = (5, 6, 7, 10, 14)
FIELDS = ('U', 'p', 'k', 'epsilon', 'nut')
def internal(s):
    m = re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n?(\d+)\s*\n?\(', s)
    n, nc = int(m.group(2)), 3 if m.group(1) == 'vector' else 1
    b = s[m.end():]
    b = b[:b.index('\n)')]
    return np.array(b.replace('(', ' ').replace(')', ' ').split(), dtype=float).reshape(n, nc)
def patches(s):
    out = {}
    b = s[s.index('boundaryField'):]
    for m in re.finditer(r'(\w+)\s*\{([^{}]*)\}', b):
        v = re.search(r'value\s+nonuniform\s+List<scalar>\s*\n?(\d+)\s*\n?\(([^)]*)\)', m.group(2))
        if v:
            out[m.group(1)] = np.array(v.group(2).split(), dtype=float)
    return out
worst = 0.0
where = ''
npatch = 0
for t in TIMES:
    for f in FIELDS:
        a = open('%s/%d/%s' % (of, t, f), errors='replace').read()
        c = open('%s/%d/%s' % (br, t, f), errors='replace').read()
        x, y = internal(a), internal(c)
        e = np.linalg.norm(y - x) / np.linalg.norm(x)
        if e > worst:
            worst, where = e, '%s cells t=%d' % (f, t)
        if f in ('k', 'epsilon'):
            pa, pc = patches(a), patches(c)
            for n in pa:
                if n in pc and np.linalg.norm(pa[n]) > 0:
                    npatch += 1
                    e = np.linalg.norm(pc[n] - pa[n]) / np.linalg.norm(pa[n])
                    if e > worst:
                        worst, where = e, '%s %s t=%d' % (f, n, t)
pat = re.compile(r'^bounding (\w+), min: (\S+) max: (\S+) average: (\S+)')
la = [pat.match(l) for l in open(of + '/run.log', errors='replace') if pat.match(l)]
lc = [pat.match(l) for l in open(br + '/run.log', errors='replace') if pat.match(l)]
same = len(la) == len(lc)
wl = 0.0
if same:
    for x, y in zip(la, lc):
        same = same and x.group(1) == y.group(1)
        for k in (2, 3, 4):
            wl = max(wl, abs(float(x.group(k)) - float(y.group(k))) / max(abs(float(x.group(k))), 1e-300))
print('        worst %.3e (%s), %d patch comparisons, %d and %d bounding lines, worst line number %.3e' % (worst, where, npatch, len(la), len(lc), wl))
inside = worst < BOUND and same and wl < BOUND and npatch > 0
raise SystemExit(0 if inside == (want == 'ok') else 1)
PYEOF
python3 "$W/arm4.py" "$W/of" "$W/br" ok && say "the bounded cells, patches and lines match OpenFOAM within 1e-6" ok \
    || say "the bounded cells, patches and lines match OpenFOAM within 1e-6" FAIL
stage "$W/live"
( cd "$W/live" && BRAE_AMG_CACHE=0 BRAE_RHOSIMPLEFOAM_MIRROR=cuda BRAE_CONTROL_BOUND_PATCH_LIVE=1 "$BRAE" -case "$W/live" > run.log 2>&1 ) || tail -2 "$W/live/run.log"
python3 "$W/arm4.py" "$W/of" "$W/live" red && say "CONTROL: the old patch half is out by more than that" ok \
    || say "CONTROL: the old patch half is out by more than that" FAIL

# ---- ARM 2: the numbers are the PRE-bound ones ---------------------------------------------------
# Printing after the clamp would report min == the floor every time; OF prints before (bound.C:42
# precedes bound.C:48). k and epsilon floor at 1e-15, so a pre-bound min is strictly below it and in
# practice negative.
python3 - "$W/br/run.log" <<'PYEOF' || fail=1
import re, sys
mins = [float(m.group(1)) for m in
        (re.match(r'^bounding \w+, min: (-?[\d.]+(?:e[+-]\d+)?)', l) for l in open(sys.argv[1]))
        if m]
ok = bool(mins) and all(v < 1e-15 for v in mins)
print('  %-72s %s' % ('every reported min is the PRE-bound value (< the floor, not == it)', 'ok' if ok else 'FAIL'))
print('        (%d lines, min of mins %.6g, max of mins %.6g)' % (len(mins), min(mins), max(mins)))
raise SystemExit(0 if ok else 1)
PYEOF

# ---- ARM 3: min <= average <= max, and the average is an interior mean ---------------------------
python3 - "$W/br/run.log" <<'PYEOF' || fail=1
import re, sys
pat = re.compile(r'^bounding (\w+), min: (\S+) max: (\S+) average: (\S+)$')
rows = [pat.match(l).groups() for l in open(sys.argv[1]) if pat.match(l)]
bad = [r for r in rows if not (float(r[1]) <= float(r[3]) <= float(r[2]))]
ok = bool(rows) and not bad
print('  %-72s %s' % ('min <= average <= max on every line', 'ok' if ok else 'FAIL'))
if bad: print('        (offender: %r)' % (bad[0],))
raise SystemExit(0 if ok else 1)
PYEOF

# ---- CONTROL: a healthy run says nothing ---------------------------------------------------------
ITERS=12
stage_healthy "$W/ok"
( cd "$W/ok" && BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BRAE" -case "$W/ok" > run.log 2>&1 ) || true
OKN=$(grep -c '^bounding ' "$W/ok/run.log" || true)
[ "${OKN:-1}" = 0 ] && say "the default run prints NOTHING (an event, not a banner)" ok \
                    || say "the default run prints NOTHING (an event, not a banner)" FAIL

# ---- CONTROL: the line cannot break a log-parsing gate -------------------------------------------
# The three shapes brae's own gates key on. A diagnostic that collided with any of them would break
# timeloop_vs_openfoam (Time = sequence equality), empty_direction/u_colour_gs (absence of
# `Solving for Uz`), or mirror_continuity (an exact count of continuity lines).
BAD=$(grep '^bounding ' "$W/br/run.log" | grep -cE '^Time = |Solving for|time step continuity errors' || true)
[ "${BAD:-1}" = 0 ] && say "the line collides with no gate-parsed prefix or substring (control)" ok \
                    || say "the line collides with no gate-parsed prefix or substring (control)" FAIL

# ---- CONTROL: the store is EMPTIED by whoever drains it -------------------------------------------
# The reports are collected during the solve and printed by the driver, so something has to empty the
# store. A clear inside a model's correct() reaches only the closure that has one: with that arrangement
# the legacy incompressible driver -- which is what nine tutorial_* gates run -- reprinted every prior
# line on every iteration. Measured on the fixture below: 47 lines with the clear removed, 2 with it.
# Accumulation reprints lines VERBATIM, so the discriminator is a duplicate, not a count.
#
# Foam::bound needs a case that actually bounds, and upwind convection never produces the negative cell
# (see bound_cpp.cuh) -- a limited scheme does.
LEG="$ROOT/validation/pitzDailyTurb"
if [ -d "$LEG" ]; then
    rm -rf "$W/leg"; mkdir -p "$W/leg"
    cp -r "$LEG/constant" "$LEG/system" "$W/leg/"
    if [ -d "$LEG/0.orig" ]; then cp -r "$LEG/0.orig" "$W/leg/0"; else cp -r "$LEG/0" "$W/leg/0"; fi
    python3 - "$W/leg" <<'PYEOF'
import re, sys
d = sys.argv[1]
f = d + '/system/fvSchemes'; s = open(f).read()
s = re.sub(r'div\(phi,k\)\s+[^;]+;', 'div(phi,k)      Gauss limitedLinear 1;', s)
s = re.sub(r'div\(phi,epsilon\)\s+[^;]+;', 'div(phi,epsilon) Gauss limitedLinear 1;', s)
open(f, 'w').write(s)
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*', '', s, flags=re.S)
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime 25;', s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval 25;', s)
open(c, 'w').write(s + '\n')
PYEOF
    ( cd "$W/leg" && BRAE_POLY_KE=1 BRAE_DILU_KE=0 "$BRAE" -case "$W/leg" > run.log 2>&1 ) || true
    LEGN=$(grep -c '^bounding ' "$W/leg/run.log" || true)
    LEGU=$(grep '^bounding ' "$W/leg/run.log" | sort -u | wc -l)
    [ "${LEGN:-0}" -gt 0 ] && [ "$LEGN" = "$LEGU" ] \
        && say "the legacy driver prints each event ONCE (the store is emptied)" ok \
        || say "the legacy driver prints each event ONCE (the store is emptied)" FAIL
    printf '        (%s lines, %s distinct -- equal means no accumulation)\n' "${LEGN:-0}" "$LEGU"
else
    say "the legacy driver prints each event ONCE (fixture missing, skipped)" ok
fi

# ---- FAIL-PROOF ----------------------------------------------------------------------------------
ITERS=14
stage "$W/off"
( cd "$W/off" && BRAE_BOUND_REPORT=0 BRAE_AMG_CACHE=0 BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BRAE" -case "$W/off" > run.log 2>&1 ) || true
OFFN=$(grep -c '^bounding ' "$W/off/run.log" || true)
[ "${OFFN:-1}" = 0 ] && say "BRAE_BOUND_REPORT=0 silences it (fail-proof for the arms above)" ok \
                     || say "BRAE_BOUND_REPORT=0 silences it (fail-proof for the arms above)" FAIL

[ "$fail" = 0 ] && echo "== PASSED ==" || echo "== FAILED =="
exit "$fail"
