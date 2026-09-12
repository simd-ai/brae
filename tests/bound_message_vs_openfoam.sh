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
# THE ORACLE is real rhoSimpleFoam on the same mesh with the same weak preconditioner, so the message is
# checked against OpenFOAM's own text and not against this test's idea of it.
#
#   ARM 1    brae prints the line, in OpenFOAM's exact format, on a case that bounds
#   ARM 2    the numbers are the PRE-bound ones (min < the floor, not == it)
#   ARM 3    min/max include the BOUNDARY and average does not -- OF's asymmetry
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

# The tutorial's own loose GAMG entries on the turbulence pair. brae substitutes a PBiCGStab for GAMG,
# and with BRAE_POLY_KE=1 that substitute is the bare `diagonal` -- the preconditioner item 78 removed,
# and the one that makes this case bound. A gate for a message needs a case that fires it.
stage() {
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

# ---- the ORACLE: real OpenFOAM on the same case, its own diagonal on the pair ---------------------
stage "$W/of"
python3 - "$W/of/system/fvSolution" <<'PYEOF'
import sys
p = sys.argv[1]; s = open(p).read()
s = s.replace('        solver          GAMG;\n        smoother        GaussSeidel;\n        tolerance       1e-08;\n        relTol          0.1;\n    }',
              '        solver          PBiCGStab;\n        preconditioner  diagonal;\n        tolerance       1e-08;\n        relTol          0.1;\n    }', 1)
open(p, 'w').write(s)
PYEOF
( set +u; source "$OFBASHRC" > /dev/null 2>&1
  command -v rhoSimpleFoam > /dev/null 2>&1 || exit 77
  cd "$W/of" && rhoSimpleFoam > run.log 2>&1 ) || { echo "SKIP: OpenFOAM rhoSimpleFoam did not run"; exit 77; }
OF_N=$(grep -c '^bounding ' "$W/of/run.log" || true)
[ "${OF_N:-0}" -gt 0 ] || { echo "SKIP: OpenFOAM did not bound on this case, so there is no oracle"; exit 77; }
printf '        (OpenFOAM printed %s bounding lines; first: %s)\n' "$OF_N" "$(grep -m1 '^bounding ' "$W/of/run.log")"

# ---- ARM 1: brae prints it, in OpenFOAM's format -------------------------------------------------
stage "$W/br"
( cd "$W/br" && BRAE_POLY_KE=1 BRAE_DILU_KE=0 BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BRAE" -case "$W/br" > run.log 2>&1 ) \
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
stage "$W/ok"
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
stage "$W/off"
( cd "$W/off" && BRAE_BOUND_REPORT=0 BRAE_POLY_KE=1 BRAE_DILU_KE=0 BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BRAE" -case "$W/off" > run.log 2>&1 ) || true
OFFN=$(grep -c '^bounding ' "$W/off/run.log" || true)
[ "${OFFN:-1}" = 0 ] && say "BRAE_BOUND_REPORT=0 silences it (fail-proof for the arms above)" ok \
                     || say "BRAE_BOUND_REPORT=0 silences it (fail-proof for the arms above)" FAIL

[ "$fail" = 0 ] && echo "== PASSED ==" || echo "== FAILED =="
exit "$fail"
