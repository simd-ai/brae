#!/usr/bin/env bash
# RESTART through the files rhoSimpleFoam's DEFAULT driver writes: alphat and writePrecision.
#
# The default driver wrote U, p, T, rho, k, omega|epsilon, nut and phi -- not alphat -- and every one at a
# fixed 12 digits (phi 17) whatever controlDict's writePrecision said. alphat is MUST_READ for a RAS model
# (EddyDiffusivity.C:73-82), so OpenFOAM could not restart from brae's output at all, and brae's own
# restart lost the patch types that carry each wall's Prt (alphatWallFunction's 0.85 against the model's
# 1.0). gpuRhoSimpleFoam.cu now writes alphat = rho*nut/Prt as it computes it (cells: the model's Prt;
# faces: the boundary rho and nut and that face's Prt, Prt echoed where it is not OpenFOAM's 0.85), and
# writes every field at writePrecision (OpenFOAM's default 6).
#
# PART A -- validation/naca0012 (the aerofoilNACA0012 tutorial: kOmegaSST, compressible::alphatWallFunction
# on the aerofoil), meshed as its Allrun.pre does, limitTemperature removed as naca_vs_openfoam and
# restart_vs_openfoam remove it. writePrecision 17 for the restart arms, so no digit is lost on the way.
#   WRITE    brae 0 -> 2N writes alphat at N and 2N: the start file's patch types (the wall patch keeps
#            compressible::alphatWallFunction), and values that ARE rho*nut/Prt of the rho and nut written
#            beside them (cells with the model's Prt 1.0, wall faces with the patch's 0.85): relative 1e-12.
#   PRECISION the same case at the shipped writePrecision 8: no number in any written field carries more
#            than 8 significant digits, and some carry exactly 8.
#   brae<-OF brae restarts from OpenFOAM's N/ (alphat read for its patch Prt) and, RESTART_STEPS (5) iterations
#            on, agrees with OpenFOAM's own restart from it within RESTART_TOL on U p T rho: the fields, steps and
#            bound validation/restart_vs_openfoam.sh holds the same driver to on the same case (k, omega, nut
#            reported: the default driver's legacy closures do not track OpenFOAM's start-up transient closely).
#   OF<-brae REPORTED here, not asserted: OpenFOAM stops on brae's p before alphat -- the shared writer
#            writes freestreamPressure's free-stream value as `inletValue`, OpenFOAM reads `freestreamValue`
#            (on main too). Asserted on rhoKE (PART B). Turns into a check when the writer is fixed.
#   #54      reported, not asserted: brae 0 -> 2N against brae N -> 2N from its own N/, and OpenFOAM against
#            itself the same way (issue #54: the default driver's own restart drifts where OpenFOAM's does
#            not). This item does not claim to change it.
# PART B -- validation/rhoKE (kEpsilon, a 700 K wall with compressible::alphatWallFunction Prt 0.85, so the
# wall alphat carries a heat flux; the aerofoil's T wall is adiabatic and cannot show it).
#   WRITE    as PART A, on both walls.
#   OF<-brae OpenFOAM restarts from brae's N/ (before: no alphat, FatalError) and reaches 2N, and lands as
#            close to brae's own restart from the same N/ as brae restarted from OpenFOAM's N/ lands to
#            OpenFOAM's own restart (both are "the two codes, N iterations from one state"): within
#            AGREE_RATIO x, so OpenFOAM reads brae's write as brae does.
#   CONTROL  brae restarted from OpenFOAM's N/ with alphat's wall entries turned into `calculated` -- the
#            model's Prt 1.0 at the wall, which is what the old restart (no alphat file) ran -- lands at
#            least CONTROL_RATIO x further from OpenFOAM's own restart than with the patches as written.
#
# TOLERANCES, declared before this gate's first run: WRITE 1e-12 (above); RESTART_TOL 1e-3
# (restart_vs_openfoam.sh's bound, naca0012 restarted from OpenFOAM's write, default driver); AGREE_RATIO 10;
# CONTROL_RATIO 10. Labels: slow. Honours OFBASHRC; exits 77 without OpenFOAM, a GPU or the binary.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae_rhoSimpleFoam}"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
N=${N:-50}
NKE=${NKE:-20}
WRITE_TOL=${WRITE_TOL:-1e-12}
RESTART_TOL=${RESTART_TOL:-1e-3}
RESTART_STEPS=${RESTART_STEPS:-5}
AGREE_RATIO=${AGREE_RATIO:-10}
CONTROL_RATIO=${CONTROL_RATIO:-10}
[ -x "$BRAE" ] || { echo "SKIP: $BRAE not built"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: OpenFOAM not available ($OFBASHRC)"; exit 77; }
# shellcheck disable=SC1090
source "$OFBASHRC" > /dev/null 2>&1 || true   # before set -u: the bashrc reads unset variables
set -u
command -v rhoSimpleFoam > /dev/null 2>&1 || { echo "SKIP: rhoSimpleFoam not on PATH"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
GEOM="${FOAM_TUTORIALS:-}/resources/geometry/NACA0012.obj.gz"
[ -f "$GEOM" ] || { echo "SKIP: $GEOM not found"; exit 77; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-80s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }

ctl() {  # ctl <dir> <endTime> <writeInterval> <writePrecision>
    python3 - "$1/system/controlDict" "$2" "$3" "$4" <<'PY'
import re, sys
c, et, wi, wp = sys.argv[1:5]; s = open(c).read()
s = re.sub(r'\nfunctions\s*\{.*', '\n', s, flags=re.S)
for k, v in (('startFrom', 'latestTime'), ('endTime', et), ('writeInterval', wi), ('writePrecision', wp)):
    s, n = re.subn(r'^%s\s+[^;]+;' % k, '%-16s%s;' % (k, v), s, flags=re.M); assert n == 1, k
open(c, 'w').write(s.rstrip() + '\n')
f = c.replace('controlDict', 'fvSolution'); s = open(f).read()   # no early stop: every run takes its steps
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { }', s)
open(f, 'w').write(s)
PY
}
ofrun() { ( cd "$1" && nice -n 10 rhoSimpleFoam > log 2>&1 ); }
brrun() { ( cd "$1" && "$BRAE" -case "$1" > log 2>&1 ); }
restart() {  # restart <base> <from-run> <time> <new dir>: the base case (0/ only) plus <from-run>/<time>
    rm -rf "$4"; cp -r "$1" "$4"; cp -r "$2/$3" "$4/$3"; rm -f "$4/log"
}
# relative L2 per field at one time: rel <A dir> <B dir> <fields...>  -> "f=v ..." and max
rel() {
    python3 - "$@" <<'PY'
import math, re, sys
A, B, fields = sys.argv[1], sys.argv[2], sys.argv[3:]
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
out, worst = [], 0.0
for f in fields:
    a, b = rd(A + '/' + f), rd(B + '/' + f)
    e = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))) / (math.sqrt(sum(x * x for x in a)) or 1.0)
    out.append('%s %.2e' % (f, e)); worst = max(worst, e)
print('%.3e  ' % worst + '  '.join(out))
PY
}
FIELDS="U p T rho k omega nut"

echo "rho_default_restart_vs_openfoam: N=$N, 2N=$((2 * N)) (naca0012); rhoKE $NKE -> $((2 * NKE))"
# ---- PART A: naca0012 ------------------------------------------------------------------------------------
B="$W/base"
cp -r "$ROOT/validation/naca0012" "$B"; rm -rf "$B"/[1-9]* "$B"/0 "$B"/log.* "$B"/system/fvOptions
mkdir -p "$B/constant/geometry"; cp -f "$GEOM" "$B/constant/geometry/"; cp -r "$B/0.orig" "$B/0"
( cd "$B" && blockMesh > log.blockMesh 2>&1 && transformPoints -scale '(1 0 1)' > log.transformPoints 2>&1 \
  && extrudeMesh > log.extrudeMesh 2>&1 ) || { echo "FAIL: naca0012 meshing"; exit 1; }
rm -f "$B"/log.*
ctl "$B" $((2 * N)) "$N" 17

cp -r "$B" "$W/OFc"; ofrun "$W/OFc" || say "OpenFOAM 0 -> $((2 * N)) failed" FAIL
cp -r "$B" "$W/Bc";  brrun "$W/Bc"  || say "brae 0 -> $((2 * N)) failed" FAIL

# WRITE: alphat at N and 2N, types as the start file's, values = rho*nut/Prt of the files beside it
checkWrite() {  # checkWrite <run dir> <time> <model Prt> -> 0 when the written alphat is what brae computes
    python3 - "$1" "$2" "$3" "$WRITE_TOL" <<'PY'
import re, sys
d, t, prtModel, tol = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
def body(p): return open(p).read()
def internal(s):
    m = re.search(r'internalField\s+nonuniform\s+List<scalar>\s*(\d+)\s*\(', s)
    n = int(m.group(1)); return [float(x) for x in s[m.end():].split(None, n)[:n]]
def patches(s):
    s = s[s.index('boundaryField'):]
    out = {}
    for m in re.finditer(r'\n    (\w+)\n    \{(.*?)\n    \}', s, re.S):
        e = m.group(2); ty = re.search(r'type\s+([\w:]+);', e).group(1)
        v = re.search(r'value\s+nonuniform\s+List<scalar>\s*(\d+)\s*\(', e)
        vals = [float(x) for x in e[v.end():].split(None, int(v.group(1)))[:int(v.group(1))]] if v else None
        prt = re.search(r'Prt\s+([-0-9.eE+]+);', e)
        out[m.group(1)] = (ty, vals, float(prt.group(1)) if prt else 0.85)
    return out
bnd = body(d + '/constant/polyMesh/boundary')
meshType = {m.group(1): m.group(2) for m in re.finditer(r'\n\s*(\w+)\s*\{[^{}]*?\btype\s+(\w+);', bnd)}
a, r, n = (body('%s/%s/%s' % (d, t, f)) for f in ('alphat', 'rho', 'nut'))
ok = True
for x, y, z in zip(internal(a), internal(r), internal(n)):
    if abs(x - y * z / prtModel) > tol * max(abs(x), 1e-300): print('  cells: alphat', x, 'rho*nut/Prt', y * z / prtModel); ok = False; break
pa, pr, pn = patches(a), patches(r), patches(n)
seen = 0
for name, (ty, vals, prtPatch) in pa.items():
    mt = meshType.get(name, '?')
    if mt in ('empty', 'symmetryPlane', 'symmetry', 'wedge', 'cyclic', 'cyclicAMI'): continue
    want = 'compressible::alphatWallFunction' if mt == 'wall' else 'calculated'
    if ty != want: print('  patch', name, '(' + mt + ') type', ty, 'want', want); ok = False; continue
    prt = prtPatch if ty == 'compressible::alphatWallFunction' else prtModel
    for x, y, z in zip(vals, pr[name][1], pn[name][1]):
        seen += 1
        if abs(x - y * z / prt) > tol * max(abs(x), 1e-300): print('  patch', name, x, y * z / prt); ok = False; break
print('  checked %d cells and %d boundary faces' % (len(internal(a)), seen))
sys.exit(0 if ok and seen > 0 else 1)
PY
}
for t in "$N" "$((2 * N))"; do
    if [ ! -f "$W/Bc/$t/alphat" ]; then say "WRITE    brae wrote no $t/alphat" FAIL; continue; fi
    checkWrite "$W/Bc" "$t" 1.0 && say "WRITE    $t/alphat: the start file's types, = rho*nut/Prt (cells 1.0, walls 0.85)" PASS \
        || say "WRITE    $t/alphat is not what brae computes" FAIL
done

# PRECISION: the shipped writePrecision 8
cp -r "$B" "$W/Bp"; ctl "$W/Bp" 2 1 8; brrun "$W/Bp"
python3 - "$W/Bp/2" <<'PY' && say "PRECISION writePrecision 8: no written number carries more than 8 digits" PASS || say "PRECISION writePrecision 8 is not what brae wrote" FAIL
import os, re, sys
d = sys.argv[1]; most = 0; bad = 0
for f in os.listdir(d):
    if not os.path.isfile(d + '/' + f): continue
    for tok in re.findall(r'(?<![\w.])-?\d+\.\d+(?:e[-+]?\d+)?', open(d + '/' + f).read()):
        mant = tok.lstrip('-').split('e')[0].replace('.', '').lstrip('0')
        most = max(most, len(mant)); bad += len(mant) > 8
print('  most significant digits seen:', most)
sys.exit(0 if bad == 0 and most == 8 else 1)
PY

# the four restarts from N
restart "$B" "$W/OFc" "$N" "$W/OFr"; ofrun "$W/OFr" || say "OpenFOAM restart from its own $N/ failed" FAIL
restart "$B" "$W/Bc"  "$N" "$W/Bb";  brrun "$W/Bb"  || say "brae restart from its own $N/ failed" FAIL
restart "$B" "$W/OFc" "$N" "$W/Bo";  brrun "$W/Bo"  || { say "brae<-OF  brae does not restart from OpenFOAM's $N/" FAIL; tail -3 "$W/Bo/log"; }
T2=$((2 * N))
# agree <label> <OF-from-brae dir> <brae-from-brae dir> <brae-from-OF dir> <OF-from-OF dir> <fields...>
agree() {
    local lbl="$1" ob="$2" bb="$3" bo="$4" ofr="$5"; shift 5
    local dBO dOB
    dBO=$(rel "$ofr" "$bo" "$@"); dOB=$(rel "$bb" "$ob" "$@")
    echo "           $lbl brae<-OF vs OpenFOAM's own restart : $dBO"
    echo "           $lbl OF<-brae vs brae's own restart     : $dOB"
    python3 - "${dOB%% *}" "${dBO%% *}" "$AGREE_RATIO" <<'PY' && say "OF<-brae  $lbl: OpenFOAM reads brae's write as brae does (${dOB%% *} <= $AGREE_RATIO x ${dBO%% *})" PASS || say "OF<-brae  $lbl: ${dOB%% *} > $AGREE_RATIO x ${dBO%% *}" FAIL
import sys; sys.exit(0 if float(sys.argv[1]) <= float(sys.argv[3]) * max(float(sys.argv[2]), 1e-12) else 1)
PY
}
# brae<-OF, as restart_vs_openfoam.sh measures it: RESTART_STEPS iterations from OpenFOAM's N/, both codes,
# U p T rho against RESTART_TOL (that gate's bound and fields); the turbulence fields reported beside them.
TR=$((N + RESTART_STEPS))
restart "$B" "$W/OFc" "$N" "$W/OFr5"; ctl "$W/OFr5" "$TR" "$RESTART_STEPS" 17; ofrun "$W/OFr5"
restart "$B" "$W/OFc" "$N" "$W/Bo5";  ctl "$W/Bo5"  "$TR" "$RESTART_STEPS" 17; brrun "$W/Bo5"
if [ -d "$W/Bo5/$TR" ] && [ -d "$W/OFr5/$TR" ]; then
    dBO=$(rel "$W/OFr5/$TR" "$W/Bo5/$TR" U p T rho)
    python3 - "${dBO%% *}" "$RESTART_TOL" <<'PY' && say "brae<-OF  naca0012: $RESTART_STEPS iterations from OpenFOAM's $N/, U p T rho within $RESTART_TOL" PASS || say "brae<-OF  naca0012: ${dBO%% *} > RESTART_TOL $RESTART_TOL" FAIL
import sys; sys.exit(0 if float(sys.argv[1]) <= float(sys.argv[2]) else 1)
PY
    echo "           ($dBO;  turbulence: $(rel "$W/OFr5/$TR" "$W/Bo5/$TR" k omega nut))"
else
    say "brae<-OF  naca0012: a run did not reach $TR" FAIL; tail -3 "$W/Bo5/log"
fi
# OpenFOAM from brae's naca0012 write. REPORTED, not asserted: the shared field writer writes
# freestreamPressure/freestreamVelocity's free-stream value as `inletValue` (foam_field_writer.cuh
# writePatchEntry), and OpenFOAM's dictionary constructor reads `freestreamValue`
# (freestreamPressureFvPatchScalarField.C:59) -- it stops on brae's p before it reaches alphat, on main as
# well. Asserted on rhoKE below, whose patches the writer round-trips. Turns into a check by itself the day
# the writer emits freestreamValue.
restart "$B" "$W/Bc"  "$N" "$W/Ob"
if ofrun "$W/Ob" && [ -d "$W/Ob/$T2" ]; then
    say "OF<-brae naca0012: OpenFOAM restarts from brae's $N/ and reaches $T2" PASS
    [ -d "$W/Bb/$T2" ] && [ -d "$W/Bo/$T2" ] && agree naca0012 "$W/Ob/$T2" "$W/Bb/$T2" "$W/Bo/$T2" "$W/OFr/$T2" $FIELDS
elif grep -q "Required entry 'freestreamValue' missing" "$W/Ob/log"; then
    echo "  OPEN     OF<-brae naca0012: OpenFOAM stops on brae's freestreamPressure entry (no freestreamValue; the"
    echo "           shared writer's, not this driver's) -- the rhoKE arm below carries the assertion"
else
    say "OF<-brae naca0012: OpenFOAM does not restart from brae's $N/" FAIL; grep -A3 'FOAM FATAL' "$W/Ob/log" | head -5
fi
if [ -d "$W/Bb/$T2" ] && [ -d "$W/OFr/$T2" ]; then
    echo "  #54 (reported, not asserted):"
    echo "           naca0012 brae 0->$T2 vs brae $N->$T2 (own write) : $(rel "$W/Bc/$T2" "$W/Bb/$T2" $FIELDS)"
    echo "           naca0012 OF   0->$T2 vs OF   $N->$T2 (own write) : $(rel "$W/OFc/$T2" "$W/OFr/$T2" $FIELDS)"
fi

# ---- PART B: the wall Prt that alphat's patch types carry (rhoKE, heated wall) ----------------------------
K="$W/kbase"
cp -r "$ROOT/validation/rhoKE" "$K"; rm -rf "$K"/[1-9]* "$K"/0 "$K"/log.*; cp -r "$K/0.orig" "$K/0"
( cd "$K" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: rhoKE meshing"; exit 1; }
rm -f "$K"/log.*
ctl "$K" $((2 * NKE)) "$NKE" 17   # OpenFOAM writes only at write times: 2N is one
cp -r "$K" "$W/kOFc"; ofrun "$W/kOFc"
cp -r "$K" "$W/kBcont"; brrun "$W/kBcont"
restart "$K" "$W/kOFc"   "$NKE" "$W/kOFr"; ofrun "$W/kOFr"
restart "$K" "$W/kOFc"   "$NKE" "$W/kBo";  brrun "$W/kBo"
restart "$K" "$W/kBcont" "$NKE" "$W/kBb";  brrun "$W/kBb"
restart "$K" "$W/kBcont" "$NKE" "$W/kOb"
KT=$((2 * NKE))
checkWrite "$W/kBcont" "$NKE" 1.0 && say "WRITE    rhoKE $NKE/alphat: the start file's types, = rho*nut/Prt (hot and cold walls 0.85)" PASS \
    || say "WRITE    rhoKE $NKE/alphat is not what brae computes" FAIL
if ofrun "$W/kOb" && [ -d "$W/kOb/$KT" ]; then
    say "OF<-brae rhoKE: OpenFOAM restarts from brae's $NKE/ and reaches $KT" PASS
    agree rhoKE "$W/kOb/$KT" "$W/kBb/$KT" "$W/kBo/$KT" "$W/kOFr/$KT" U p T rho k epsilon nut
else
    say "OF<-brae rhoKE: OpenFOAM does not restart from brae's $NKE/" FAIL; grep -A3 'FOAM FATAL' "$W/kOb/log" | head -5
fi
echo "  #54 (reported, not asserted):"
echo "           rhoKE brae 0->$KT vs brae $NKE->$KT (own write) : $(rel "$W/kBcont/$KT" "$W/kBb/$KT" U p T rho k epsilon nut)"
echo "           rhoKE OF   0->$KT vs OF   $NKE->$KT (own write) : $(rel "$W/kOFc/$KT" "$W/kOFr/$KT" U p T rho k epsilon nut)"
restart "$K" "$W/kOFc" "$NKE" "$W/kBc"
python3 - "$W/kBc/$NKE/alphat" <<'PY'
import re, sys
p = sys.argv[1]; s = open(p).read()
s, n = re.subn(r'compressible::alphatWallFunction;', 'calculated;', s); assert n >= 1
open(p, 'w').write(s)
PY
brrun "$W/kBc"
if [ -d "$W/kOFr/$KT" ] && [ -d "$W/kBo/$KT" ] && [ -d "$W/kBc/$KT" ]; then
    good=$(rel "$W/kOFr/$KT" "$W/kBo/$KT" T U p); ctlr=$(rel "$W/kOFr/$KT" "$W/kBc/$KT" T U p)
    echo "           with the patch Prt  : $good"
    echo "           with the model Prt  : $ctlr"
    python3 - "${ctlr%% *}" "${good%% *}" "$CONTROL_RATIO" <<'PY' && say "CONTROL  the wall Prt alphat carries is read: model Prt lands >= $CONTROL_RATIO x further" PASS || say "CONTROL  alphat's wall types make no measurable difference here" FAIL
import sys; sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[3]) * max(float(sys.argv[2]), 1e-14) else 1)
PY
else
    say "CONTROL  a rhoKE run did not reach $KT" FAIL
fi

[ "$fail" -eq 0 ] && echo "rho_default_restart_vs_openfoam: PASS" || echo "rho_default_restart_vs_openfoam: FAIL"
exit "$fail"
