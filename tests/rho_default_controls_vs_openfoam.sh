#!/usr/bin/env bash
# Two Time/solution controls of rhoSimpleFoam's DEFAULT driver, held to OpenFOAM v2412: the pressure
# reference cell named by a POINT, and a non-integer endTime.
#
# PART P -- pRefPoint (findRefCell.C:31-115). With no p patch fixing a value OpenFOAM pins p at pRefCell,
# else at the cell polyMesh::findCell locates pRefPoint in. The default driver took pRefCell with a
# default of 0, so a case naming pRefPoint was pinned in cell 0. Fixture validation/rhoBoxP with U and p
# zeroGradient at the outlet, so no p patch fixes a value and p needs a reference; pRefCell replaced by
# pRefPoint. (Not rhoBoxP as shipped: the default driver refuses its fixedFluxPressure outlet -- it never
# runs constrainPressure -- and with that p made zeroGradient but U left inletOutlet, the driver's
# deviceAdjustPhi counts the inletOutlet outflow as FIXED where OpenFOAM's adjustPhi.C counts it adjustable,
# and stops: a separate default-driver finding, reported, not this gate's.)
#   ORACLE   OpenFOAM with pRefPoint and OpenFOAM with pRefCell <the cell brae reports> write the same
#            bytes -- the cell brae found IS the cell OpenFOAM found -- and OpenFOAM with pRefCell 0 does not.
#   ARM      brae with pRefPoint agrees within the bounds with brae given pRefCell <that cell> (not bytes:
#            the default driver is not bitwise reproducible run to run on a case whose p needs a reference,
#            on main as well), and with OpenFOAM's pRefPoint run.
#   CONTROL  brae with pRefCell 0 (what the pre-fix driver ran for this case) is >= CONTROL_RATIO x the
#            bound away from OpenFOAM's pRefPoint run.
#
# PART T -- endTime (TimeIO.C:340-356, Time.C:785). The driver cast endTime to int: `endTime 4.7` ran 4
# iterations where OpenFOAM's `value() < endTime - 0.5*deltaT` runs 5, and `deltaT 0.5; endTime 2.5` ran 4
# where OpenFOAM runs 5. Fixture validation/rhoBox, writeInterval 1.
#   ARM      brae writes exactly OpenFOAM's time directories, and every field at every one of them agrees
#            with OpenFOAM's within the bounds.
#
# TOLERANCES, declared before this gate's first run: the bounds of rho_energy_relax_vs_openfoam (~30x the
# default driver's measured agreement with OpenFOAM on the rhoBox family at t=1..5 with every linear solver
# pinned: T 1.0e-12, p 3.0e-12, U 1.1e-11, rho 2.5e-12). Same driver, same 1200-cell duct, same pins.
# Labels: slow (OpenFOAM runs). Honours OFBASHRC; exits 77 without OpenFOAM, a GPU or the binary.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae_rhoSimpleFoam}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
N=${N:-5}
T_BOUND=${T_BOUND:-3e-11}; P_BOUND=${P_BOUND:-1e-10}; U_BOUND=${U_BOUND:-3e-10}; RHO_BOUND=${RHO_BOUND:-1e-10}
CONTROL_RATIO=${CONTROL_RATIO:-100}
REFPOINT=${REFPOINT:-1.51 0.135 0.025}   # inside cell (45, 13) of rhoBoxP's 60x20 duct, far from cell 0

[ -x "$BRAE" ] || { echo "SKIP: $BRAE not built"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: OpenFOAM not available ($OFBASHRC)"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
# shellcheck disable=SC1090
source "$OFBASHRC" > /dev/null 2>&1 || true   # before set -u: the bashrc reads unset variables
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }
for f in rhoBox rhoBoxP; do
    [ -f "$ROOT/validation/$f/constant/polyMesh/owner" ] || { echo "SKIP: fixture $f has no mesh"; exit 77; }
done
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-74s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

# stage <fixture> <dir> <python edits on s (controlDict) and f (fvSolution)>
stage() {
    rm -rf "$2"; cp -r "$ROOT/validation/$1" "$2"; rm -rf "$2"/[1-9]* "$2"/0 "$2"/log.*; cp -r "$2/0.orig" "$2/0"
    python3 - "$2" "$N" "$3" <<'PY'
import re, sys
d, n, code = sys.argv[1:4]
def sub(s, pat, rep):
    s, k = re.subn(pat, rep, s); assert k >= 1, pat; return s
c = d + '/system/controlDict'; s = open(c).read()
s = sub(s, r'\bendTime\s+[^;]+;', 'endTime %s;' % n)
s = sub(s, r'\bwriteInterval\s+[^;]+;', 'writeInterval 1;')
s = sub(s, r'\bwritePrecision\s+[^;]+;', 'writePrecision 17;')
if d.split('/')[-1].startswith(('br_pt', 'of_pt', 'of_cell', 'of_0', 'br_cell', 'br_0')):
    for q in (d + '/0/p', d + '/0/U'):
        ps = open(q).read()
        ps, k = re.subn(r'outlet\s*\{[^{}]*\}', 'outlet { type zeroGradient; }', ps); assert k == 1
        open(q, 'w').write(ps)
p = d + '/system/fvSolution'; f = open(p).read()
f = re.sub(r'tolerance\s+[^;]+;', 'tolerance 1e-13;', f)
f = re.sub(r'relTol\s+[^;]+;', 'relTol 0;', f)
f = re.sub(r'residualControl\s*\{[^{}]*\}', '', f)
exec(code)
open(c, 'w').write(s); open(p, 'w').write(f)
PY
}
ofrun() { ( cd "$1" && nice -n 10 rhoSimpleFoam > log 2>&1 ); }
brrun() { ( cd "$1" && "$BRAE" -case "$1" > log 2>&1 ); }
times() { (cd "$1" && ls -d [0-9]* 2>/dev/null | grep -E '^[0-9.e+-]+$' | grep -vx 0 | sort -g | tr '\n' ' '); }
samebytes() {  # every field file of every written time
    local t f
    [ "$(times "$1")" = "$(times "$2")" ] || return 1
    for t in $(times "$1"); do for f in U p T rho phi; do cmp -s "$1/$t/$f" "$2/$t/$f" || return 1; done; done
}
# cmpf <A> <B>: per-field max over every time A wrote of the relative L2 of B against A ("T p U rho")
cmpf() {
    python3 - "$1" "$2" <<'PY'
import math, os, re, sys
A, B = sys.argv[1], sys.argv[2]
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
ts = sorted([x for x in os.listdir(A) if re.fullmatch(r'[0-9.e+-]+', x) and x != '0'], key=float)
out = []
for f in ('T', 'p', 'U', 'rho'):
    worst = 0.0
    for t in ts:
        a, b = rd('%s/%s/%s' % (A, t, f)), rd('%s/%s/%s' % (B, t, f))
        den = math.sqrt(sum(x * x for x in a)) or 1.0
        worst = max(worst, math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))) / den)
    out.append('%.3e' % worst)
print(' '.join(out))
PY
}
within() {
    python3 - "$1" "$T_BOUND" "$P_BOUND" "$U_BOUND" "$RHO_BOUND" <<'PY'
import sys
v = [float(x) for x in sys.argv[1].split()]; b = [float(x) for x in sys.argv[2:6]]
sys.exit(0 if all(x <= y for x, y in zip(v, b)) else 1)
PY
}
farOutside() {  # farOutside "<T p U rho>": some field >= CONTROL_RATIO x its bound
    python3 - "$1" "$T_BOUND" "$P_BOUND" "$U_BOUND" "$RHO_BOUND" "$CONTROL_RATIO" <<'PY'
import sys
v = [float(x) for x in sys.argv[1].split()]; b = [float(x) for x in sys.argv[2:6]]; r = float(sys.argv[6])
sys.exit(0 if any(x >= r * y for x, y in zip(v, b)) else 1)
PY
}

echo "rho_default_controls_vs_openfoam: bounds T $T_BOUND p $P_BOUND U $U_BOUND rho $RHO_BOUND"

# ---- PART P: pRefPoint ----------------------------------------------------------------------------------
PT="f = sub(f, r'pRefCell\s+\d+;', 'pRefPoint ($REFPOINT);')"
stage rhoBoxP "$W/br_pt" "$PT"; brrun "$W/br_pt" || { say "P: brae failed on pRefPoint" FAIL; tail -3 "$W/br_pt/log"; }
cell=$(sed -n 's/.*pRefPoint .* is in cell \([0-9][0-9]*\).*/\1/p' "$W/br_pt/log" | head -1)
if [ -z "$cell" ]; then
    say "P: brae did not report the pRefPoint cell" FAIL
else
    echo "  brae locates pRefPoint ($REFPOINT) in cell $cell"
    stage rhoBoxP "$W/of_pt" "$PT"; ofrun "$W/of_pt"
    stage rhoBoxP "$W/of_cell" "f = sub(f, r'pRefCell\s+\d+;', 'pRefCell $cell;')"; ofrun "$W/of_cell"
    stage rhoBoxP "$W/of_0" "";  ofrun "$W/of_0"
    stage rhoBoxP "$W/br_cell" "f = sub(f, r'pRefCell\s+\d+;', 'pRefCell $cell;')"; brrun "$W/br_cell"
    stage rhoBoxP "$W/br_0" "";  brrun "$W/br_0"
    samebytes "$W/of_pt" "$W/of_cell" && say "P ORACLE OpenFOAM pRefPoint == OpenFOAM pRefCell $cell (bytes)" PASS \
        || say "P ORACLE OpenFOAM pRefPoint != OpenFOAM pRefCell $cell: brae found another cell" FAIL
    samebytes "$W/of_pt" "$W/of_0" && say "P ORACLE OpenFOAM pRefCell 0 == pRefPoint: the fixture cannot see the cell" FAIL \
        || say "P ORACLE OpenFOAM pRefCell 0 differs from pRefPoint (the fixture sees the cell)" PASS
    # Not a byte comparison on brae's side: on this case (p needs a reference) the default driver is not
    # bitwise reproducible run to run -- measured on main 9b7b1e58 too, 2 distinct outputs in 4 identical
    # runs, the spread ~1e-11 from iteration 2 on U -- so the two runs are held to the bounds instead.
    d=$(cmpf "$W/br_cell" "$W/br_pt")
    within "$d" && say "P ARM    brae pRefPoint vs brae pRefCell $cell (T p U rho) $d" PASS \
        || say "P ARM    brae pRefPoint vs brae pRefCell $cell (T p U rho) $d" FAIL
    d=$(cmpf "$W/of_pt" "$W/br_pt")
    within "$d" && say "P ARM    brae vs OpenFOAM, pRefPoint (T p U rho) $d" PASS \
        || say "P ARM    brae vs OpenFOAM, pRefPoint (T p U rho) $d" FAIL
    c=$(cmpf "$W/of_pt" "$W/br_0")
    farOutside "$c" && say "P CONTROL brae pinned in cell 0 vs OpenFOAM pRefPoint (T p U rho) $c" PASS \
        || say "P CONTROL cell 0 is not far from OpenFOAM's answer: $c" FAIL
fi

# ---- PART T: endTime as a scalar ------------------------------------------------------------------------
for spec in "4.7:1" "2.5:0.5"; do
    et=${spec%%:*}; dt=${spec#*:}
    ED="s = sub(s, r'\bendTime\s+[^;]+;', 'endTime $et;'); s = sub(s, r'\bdeltaT\s+[^;]+;', 'deltaT $dt;')"
    stage rhoBox "$W/of_t$et" "$ED"; ofrun "$W/of_t$et"
    stage rhoBox "$W/br_t$et" "$ED"; brrun "$W/br_t$et" || { say "T endTime $et: brae failed" FAIL; tail -3 "$W/br_t$et/log"; continue; }
    to=$(times "$W/of_t$et"); tb=$(times "$W/br_t$et")
    [ -n "$to" ] && [ "$to" = "$tb" ] && say "T endTime $et deltaT $dt: brae writes OpenFOAM's times ($to)" PASS \
        || say "T endTime $et deltaT $dt: OpenFOAM wrote [$to], brae [$tb]" FAIL
    if [ "$to" = "$tb" ]; then
        d=$(cmpf "$W/of_t$et" "$W/br_t$et")
        within "$d" && say "T endTime $et: every field at every time (T p U rho) $d" PASS \
            || say "T endTime $et: every field at every time (T p U rho) $d" FAIL
    fi
done

[ "$fail" -eq 0 ] && echo "rho_default_controls_vs_openfoam: PASS" || echo "rho_default_controls_vs_openfoam: FAIL"
exit "$fail"
