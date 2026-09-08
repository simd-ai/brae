#!/bin/bash
# Item 78. A case that asks for GAMG on k and epsilon gets brae's PBiCGStab instead -- brae has no GAMG
# solver -- and the question this gate settles is which PRECONDITIONER that substitute carries. It used
# to be `diagonal`, because the case names no preconditioner (OpenFOAM's GAMG takes none) and brae only
# switched to DILU when it saw the word. That picks OpenFOAM's WEAKEST preconditioner where the case
# asked for its strongest solver, and at the tutorial's relTol 0.1 the two stop in very different
# places: epsilon is driven non-positive in interior cells, bound() floors it at 1e-15, and
# nut = Cmu k^2/epsilon explodes. Reported k residuals then collapse to ~1e-14 and climb 10x per
# iteration, because normFactor is inflated by the same runaway cells.
#
# The blank is now filled by a degree-10 TRUNCATED NEUMANN SERIES, not by DILU: both fix it, and the
# series costs 9 SpMVs against DILU's launch-per-dependency-level walk (measured on squareBend at 307k,
# turbulence block: diagonal 12.2 ms/it, this 13.4, DILU 33.5). BRAE_POLY_KE=1 restores the bare
# diagonal and is this gate's fail-proof; BRAE_DILU_KE=1 selects DILU.
#
# The gate is against REAL OpenFOAM on the same case at the same iteration, because "epsilon looks small"
# is not a criterion -- what epsilon should be at outer iteration 12 is a number only OpenFOAM has.
#
# Fixture: validation/sbMatched (112k), the compressible squareBend mesh, with the TUTORIAL's linear
# solver entries (GAMG, tolerance 1e-8, relTol 0.1) written in place of the fixture's own matched-tight
# ones -- the loose relTol is the condition, and the fixture exists at tight tolerance for other gates.
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
SRC="$ROOT/validation/sbMatched"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
[ -x "$BRAE" ] || { echo "SKIP: no brae binary"; exit 77; }
[ -d "$SRC/constant" ] || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: no OpenFOAM at $OFBASHRC"; exit 77; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT

fail=0
say() { printf '  %-70s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

# stage() <dir> [<k/epsilon preconditioner entry>]
stage() {
    rm -rf "$1"; mkdir -p "$1"
    cp -r "$SRC/constant" "$SRC/system" "$1/"
    cp -r "$SRC/0.orig" "$1/0"
    sed -i 's/^endTime.*/endTime         12;/; s/^writeInterval.*/writeInterval   12;/' "$1/system/controlDict"
    python3 - "$1/system/fvSolution" "${2:-}" <<'PYEOF'
import sys
path, pc = sys.argv[1], sys.argv[2]
s = open(path).read()
extra = ("\n        preconditioner  %s;" % pc) if pc else ""
s = s.replace('p { solver PBiCGStab; preconditioner DILU; tolerance 1e-12; relTol 0; }',
              'p { solver GAMG; smoother GaussSeidel; tolerance 1e-08; relTol 0.1; }')
old = '"(U|e|k|epsilon)" { solver PBiCGStab; preconditioner DILU; tolerance 1e-12; relTol 0; }'
assert old in s, "fixture fvSolution changed shape"
s = s.replace(old, '"(U|e|k|epsilon)"\n    {\n        solver          GAMG;\n'
                   '        smoother        GaussSeidel;\n        tolerance       1e-08;\n'
                   '        relTol          0.1;%s\n    }' % extra)
open(path, 'w').write(s)
PYEOF
}

# minmax() <field file> -> "<min> <max> <n floored>"
minmax() {
    python3 - "$1" <<'PYEOF'
import re, sys
b = open(sys.argv[1], 'rb').read()
m = re.search(rb'internalField\s+nonuniform\s+List<scalar>\s*\n?(\d+)\s*\n\(', b)
if not m:
    print("nan nan -1"); raise SystemExit
v = [float(x) for x in b[m.end():].split(b')\n', 1)[0].split()]
print("%.6e %.6e %d" % (min(v), max(v), sum(1 for x in v if x <= 1e-14)))
PYEOF
}

# ---- the ORACLE: real OpenFOAM, 12 outer iterations on the same case ------------------------------
stage "$W/of"
( set +e; source "$OFBASHRC" > /dev/null 2>&1
  command -v rhoSimpleFoam > /dev/null 2>&1 || exit 77
  cd "$W/of" && rhoSimpleFoam > run.log 2>&1 ) || { echo "SKIP: OpenFOAM rhoSimpleFoam did not run"; exit 77; }
[ -f "$W/of/12/epsilon" ] || { echo "SKIP: OpenFOAM wrote no 12/ directory"; exit 77; }
read OF_E_MIN OF_E_MAX OF_E_FLOOR <<< "$(minmax "$W/of/12/epsilon")"
read OF_N_MIN OF_N_MAX OF_N_FLOOR <<< "$(minmax "$W/of/12/nut")"
printf '        (OpenFOAM @12: epsilon min %s max %s | nut min %s max %s)\n' \
       "$OF_E_MIN" "$OF_E_MAX" "$OF_N_MIN" "$OF_N_MAX"

# check() <label> <epsMin> <nutMin> <nutMax> <expect: pass|fail>: the three bounds, together.
# A factor of 10 either side of OpenFOAM's own extremes. It is loose ON PURPOSE -- the substituted
# solver takes a different path and the extremes of a transient turbulence field are not expected to
# match to a few percent -- and it still separates the two arms by orders of magnitude (measured:
# DILU is 4.0x / 4.6x / 2.7x inside it, diagonal is 1641x / 6.6e29x / 23x outside).
check() {
    python3 - "$1" "$2" "$3" "$4" "$5" "$OF_E_MIN" "$OF_N_MIN" "$OF_N_MAX" <<'PYEOF'
import sys
label, eMin, nMin, nMax, expect, ofE, ofN, ofNmax = sys.argv[1:9]
eMin, nMin, nMax = float(eMin), float(nMin), float(nMax)
ofE, ofN, ofNmax = float(ofE), float(ofN), float(ofNmax)
tests = [("epsilon min", eMin, ofE / 10.0, eMin >= ofE / 10.0),
         ("nut min",     nMin, ofN / 10.0, nMin >= ofN / 10.0),
         ("nut max",     nMax, ofNmax * 10.0, nMax <= ofNmax * 10.0)]
ok = all(t[3] for t in tests)
hit = (ok if expect == 'pass' else not ok)
print("%s|%s" % ("ok" if hit else "FAIL",
                 "  ".join("%s %.4e (bound %.4e) %s" % (n, v, b, "ok" if p else "OUT") for n, v, b, p in tests)))
PYEOF
}

# ---- arm 1: brae as it now runs -- the substituted PBiCGStab carries DILU -------------------------
stage "$W/dilu"
( cd "$W/dilu" && BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BRAE" -case "$W/dilu" > log 2>&1 ) \
    || { tail -3 "$W/dilu/log"; say "brae runs the case" FAIL; }
read B_E_MIN B_E_MAX B_E_FLOOR <<< "$(minmax "$W/dilu/12/epsilon")"
read B_N_MIN B_N_MAX B_N_FLOOR <<< "$(minmax "$W/dilu/12/nut")"
IFS='|' read -r verdict detail <<< "$(check poly "$B_E_MIN" "$B_N_MIN" "$B_N_MAX" pass)"
say "brae's k/epsilon extremes track OpenFOAM's at outer iteration 12" "$verdict"
printf '        (%s)\n' "$detail"

# ---- arm 2 (FAIL-PROOF): the old default must NOT clear those bounds ------------------------------
# Without this the bounds above could be anything: a gate that no wrong answer fails is not a gate.
stage "$W/diag"
( cd "$W/diag" && BRAE_POLY_KE=1 BRAE_DILU_KE=0 BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BRAE" -case "$W/diag" > log 2>&1 ) \
    || { tail -3 "$W/diag/log"; say "brae runs the case with the bare diagonal" FAIL; }
read D_E_MIN D_E_MAX D_E_FLOOR <<< "$(minmax "$W/diag/12/epsilon")"
read D_N_MIN D_N_MAX D_N_FLOOR <<< "$(minmax "$W/diag/12/nut")"
IFS='|' read -r verdict detail <<< "$(check diag "$D_E_MIN" "$D_N_MIN" "$D_N_MAX" fail)"
say "...and the bare diagonal FAILS them (fail-proof)" "$verdict"
printf '        (%s)\n' "$detail"

# ---- arm 3: nut collapses to the floor under diagonal and not under DILU --------------------------
# The mechanism, stated as its own number: epsilon goes non-positive, bound() floors it at 1e-15, and
# nut = Cmu k^2/epsilon loses every significant digit in those cells.
[ "$B_N_FLOOR" = 0 ] && [ "$D_N_FLOOR" -gt 0 ] \
    && say "no nut cell is at the floor under the series, and some are under diagonal" ok \
    || say "no nut cell is at the floor under the series, and some are under diagonal" FAIL
printf '        (nut cells <= 1e-14: series %s, diagonal %s, OpenFOAM %s)\n' "$B_N_FLOOR" "$D_N_FLOOR" "$OF_N_FLOOR"

# ---- arm 4: the notice says what it runs ----------------------------------------------------------
# The notice has to name what RUNS. Printing `diagonal` over a Neumann-preconditioned solve would be
# the shared-capability-notice-lies defect, so the notice and the policy read the same function.
grep -q "solvers/k solver: case asks 'GAMG', brae runs PBiCGStab preconditioned with a degree-10 truncated Neumann series" "$W/dilu/log" \
    && grep -q "solvers/epsilon solver: case asks 'GAMG', brae runs PBiCGStab preconditioned with a degree-10 truncated Neumann series" "$W/dilu/log" \
    && say "the notice names the Neumann series for both k and epsilon" ok \
    || { grep -m2 "solvers/k solver\|solvers/epsilon solver" "$W/dilu/log"; say "the notice names the Neumann series for both k and epsilon" FAIL; }

# ---- arm 5: an UNRELAXED pair falls to DILU, not to the bare diagonal -----------------------------
# The series' convergence ratio is bounded by the RELAXATION FACTOR, because fvMatrix::relax clamps
# D >= sum|offdiag| and then divides by alpha (fvMatrix.C:105-113). Measured on this mesh family, the row
# bound is exactly alpha at 24k, 112k, 307k, 896k, 1.75M and 3.02M cells -- flat, because it is alpha and
# not the mesh that pins it. Strip the factor and the bound becomes 1 (measured rho 0.9986 at 112k,
# 1.0010 at 3.02M): the series stops preconditioning, silently. So that case takes DILU instead.
stage "$W/norelax"
python3 - "$W/norelax/system/fvSolution" <<'PYEOF'
import re, sys
p = sys.argv[1]
s = open(p).read()
s2 = re.sub(r'^\s*(k|epsilon)\s+[0-9.]+\s*;\s*$', '', s, flags=re.M)
assert s2 != s, 'the fixture no longer relaxes k/epsilon, so this arm tests nothing'
open(p, 'w').write(s2)
PYEOF
( cd "$W/norelax" && BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BRAE" -case "$W/norelax" > log 2>&1 ) || true
grep -q "solvers/k solver: case asks 'GAMG', brae runs PBiCGStab preconditioned with DILU (the case names none, and relaxes k by 1 or not at all" "$W/norelax/log" \
    && say "an unrelaxed pair takes DILU, and the notice says why" ok \
    || { grep -m1 "solvers/k solver" "$W/norelax/log"; say "an unrelaxed pair takes DILU, and the notice says why" FAIL; }
# ...and the RELAXED fixture must not, or the arm above proves nothing about relaxation
grep -q "preconditioned with a degree-10 truncated Neumann series" "$W/dilu/log" \
    && say "...and the relaxed fixture keeps the series (control)" ok \
    || say "...and the relaxed fixture keeps the series (control)" FAIL

# ---- arm 6 (control): a NAMED preconditioner is still honoured ------------------------------------
# The rule fills the blank a substitution leaves; it does not override a case that states its choice.
# (`preconditioner diagonal` beside `solver GAMG` is unread by OpenFOAM, which is what makes it a clean
# way to state the choice without changing what the oracle would do.)
stage "$W/named" diagonal
( cd "$W/named" && BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BRAE" -case "$W/named" > log 2>&1 ) || true
grep -q "solvers/k solver: case asks 'GAMG', brae runs PBiCGStab preconditioned with diagonal" "$W/named/log" \
    && say "a case that names \`preconditioner diagonal\` still gets diagonal (control)" ok \
    || { grep -m1 "solvers/k solver" "$W/named/log"; say "a case that names \`preconditioner diagonal\` still gets diagonal (control)" FAIL; }

[ "$fail" = 0 ] && echo "== PASSED ==" || echo "== FAILED =="
exit "$fail"
