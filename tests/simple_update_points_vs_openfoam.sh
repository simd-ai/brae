#!/usr/bin/env bash
# simpleFoam's boundary updates at OpenFOAM's points, against OpenFOAM v2412.
#
# WHAT IT HOLDS. Each boundary condition updates in the constructor or evaluate of its own field's equation
# (fvMatrix.C:396, fvPatchField.C:336-344): totalPressure and p inletOutlet at the pressure assembly
# (simpleFoam pEqn.H), pressureInletOutletVelocity after the momentum solve and after the velocity correction, the turbulence
# fields' inletOutlet and turbulent inlets at their equations in turbulence->correct(). p.relax() then blends the
# pressure's patch values with prevIter as well as its cells (GeometricField::relax), so a totalPressure face holds the relaxed
# value until the next pressure assembly sets it again, and U = HbyA - rAtU*grad(p) reads that; the pressure laplacian's
# non-orthogonal correction reads fvc::grad(p) after the assembly's updateCoeffs, totalPressure's new value included. brae's
# steady step took every update at the momentum assembly, from the flux the iteration started with, and left totalPressure's
# value unrelaxed.
#
# THE CASE. incompressible/simpleFoam/windAroundBuildings (totalPressure p and pressureInletOutletVelocity U on its outlet,
# inletOutlet k and epsilon there), meshed as the tutorial does (serial snappyHexMesh), run by OpenFOAM with the tutorial's
# own settings to iteration 20 and written at writePrecision 17. Both codes continue from that write for three iterations,
# every solve pinned (tolerance 1e-12, relTol 0), written every iteration.
#
# ARMS, every entry of p U k epsilon nut on its own scale (tests/pimple_entry_compare.py) within BOUND at 21-23:
#   W1  p relaxation 1: the update points alone
#   W2  as shipped, p relaxation 0.3: the relaxed totalPressure value as well
# phi is held entry by entry on its own scale, but for an entry whose scale is round-off of the field's (frontAndBack's
# flux, ~1e-15 against ~3e+03), which is held on the field's (worstphi).
# PREMISE: OpenFOAM's W2 differs from its W1, so the relaxation reaches the fields W2 holds.
# CONTROLS: BRAE_CONTROL_STEADY_UPDATE_POINTS_LEGACY=1 (the momentum assembly's points) red on W1;
#           BRAE_CONTROL_TOTAL_PRESSURE_UNRELAXED=1 (the value the assembly set) red on W2.
#           BRAE_CONTROL_PASS0_GRAD_BEFORE_TOTAL_PRESSURE=1 (the first pressure pass's non-orthogonal correction from the
#           momentum predictor's gradient, the totalPressure faces as stored) red on W1.
# MEASURED (Arrhenius, one GH200, OpenFOAM v2412 serial, 2026-11-28, jobs 5870999, 5872345): worst entry at 21-23
#   before (develop e4c67534): W1 1.3e-03 (U outlet), W2 4.1 (p outlet: 0 written where OpenFOAM's p.relax() leaves 0.7 x -0.231)
#   after: W1 5.3e-11 (p internal), W2 1.9e-11 (epsilon internal); phi 3.3e-12, 1.6e-12. OpenFOAM's own run with p on PCG
#   rather than GAMG is 9.4e-11 from its GAMG run (5872073), so BOUND 1e-9 sits above that floor and twenty times brae's worst.
#   Premise: inf (p outlet). Controls: 1.3e-03 (U outlet), 1.0 (p outlet), 3.4e-07 (U outlet).
# Labels: slow. Honours OFBASHRC; exits 77 without OpenFOAM, a GPU or the binary.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BRAE_BIN:-$ROOT/build/brae}"
OFBASHRC=${OF_BASHRC:-${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}}
CMP="$ROOT/tests/pimple_entry_compare.py"
BOUND=${BRAE_SU_BOUND:-1e-9}
PREMISE=1e-4
[ -x "$BIN" ] || { echo "SKIP: $BIN not built"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: no OpenFOAM v2412 at $OFBASHRC"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
set --
set +u   # the bashrc reads unset variables
# shellcheck disable=SC1090
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
TUTS="${BRAE_OF_TUTORIALS:-${FOAM_TUTORIALS:-}}"
TUT="$TUTS/incompressible/simpleFoam/windAroundBuildings"
[ -d "$TUT/system" ] || { echo "SKIP: no simpleFoam/windAroundBuildings tutorial at $TUT"; exit 77; }
[ -f "$TUTS/resources/geometry/buildings.obj.gz" ] || { echo "SKIP: no buildings.obj.gz under $TUTS/resources"; exit 77; }
command -v simpleFoam > /dev/null 2>&1 && command -v snappyHexMesh > /dev/null 2>&1 || { echo "SKIP: simpleFoam/snappyHexMesh not on PATH"; exit 77; }
W=${BRAE_SU_KEEP:-$(mktemp -d)}
[ -n "${BRAE_SU_KEEP:-}" ] || trap 'rm -rf "$W"' EXIT
mkdir -p "$W"
fail=0
say() { printf '  %-96s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
echo "simple_update_points_vs_openfoam: windAroundBuildings from OpenFOAM's iteration 20, three pinned iterations"

# the mesh, as Allrun.pre makes it, and OpenFOAM's iteration 20 with the tutorial's own settings
B=$W/base
rm -rf "$B"
mkdir -p "$B/constant/triSurface"
cp -r "$TUT/0.orig" "$B/0"
cp -r "$TUT/constant/." "$B/constant/"
cp -r "$TUT/system" "$B/"
cp "$TUTS/resources/geometry/buildings.obj.gz" "$B/constant/triSurface/"
( cd "$B" && surfaceFeatureExtract > log.sfe 2>&1 && blockMesh > log.blockMesh 2>&1 && snappyHexMesh -overwrite > log.shm 2>&1 ) \
    || { echo "FAIL: meshing"; tail -5 "$B"/log.*; exit 1; }
# setcontrol <case> <endTime> <writeInterval>: the controls both codes take; the tutorial's function objects are not part of it
setcontrol()
{
    python3 - "$1" "$2" "$3" << 'PY'
import re
import sys
d, end, wi = sys.argv[1:4]
p = d + '/system/controlDict'
c = open(p).read()
i = c.find('\nfunctions')
if i >= 0:
    j = c.index('{', i)
    depth, k = 0, j
    while True:
        depth += {'{': 1, '}': -1}.get(c[k], 0)
        k += 1
        if depth == 0:
            break
    c = c[:i] + '\n' + c[k:]
for key, v in (('startFrom', 'latestTime'), ('endTime', end), ('writeControl', 'timeStep'), ('writeInterval', wi),
               ('writeFormat', 'ascii'), ('writePrecision', '17'), ('writeCompression', 'off'), ('timePrecision', '6')):
    c, n = re.subn(r'^%s\s+[^;]*;' % key, '%-16s%s;' % (key, v), c, flags=re.M)
    if n == 0:
        c += '\n%-16s%s;\n' % (key, v)
open(p, 'w').write(c)
PY
}
setcontrol "$B" 20 20
( cd "$B" && simpleFoam > log.seed 2>&1 ) || { echo "FAIL: OpenFOAM to iteration 20"; tail -5 "$B/log.seed"; exit 1; }
[ -f "$B/20/p" ] || { echo "FAIL: OpenFOAM wrote no iteration 20"; exit 1; }
rm -rf "$B/0"

# stage <dir> <arm>
stage()
{
    rm -rf "$1"
    mkdir -p "$1"
    cp -r "$B/20" "$B/constant" "$B/system" "$1/"
    setcontrol "$1" 23 1
    python3 - "$1" "$2" << 'PY'
import re
import sys
d, arm = sys.argv[1:3]
s = open(d + '/system/fvSolution').read()
i = s.index('solvers')
j = s.index('SIMPLE')
blk = re.sub(r'(\btolerance\s+)[^;]+;', r'\g<1>1e-12;', s[i:j])
blk = re.sub(r'(\brelTol\s+)[^;]+;', r'\g<1>0;', blk)
blk = re.sub(r'(\bsolver\s+[^;]+;)', r'\g<1>\n        maxIter         5000;', blk)
s = s[:i] + blk + s[j:]
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl\n    {\n    }', s)
if arm == 'W1':
    s, n = re.subn(r'(\n\s*p\s+)0\.3;', r'\g<1>1;', s, count=1)
    assert n == 1, 'p relaxation'
open(d + '/system/fvSolution', 'w').write(s)
PY
}
ofrun() { ( cd "$1" && simpleFoam > log 2>&1 ) || { echo "FAIL: OpenFOAM in $1"; tail -5 "$1/log"; exit 1; }; }
brrun() { local d=$1; shift; ( cd "$d" && env "$@" "$BIN" -case . > log 2>&1 ) || { echo "FAIL: brae in $d"; tail -5 "$d/log"; exit 1; }; }
TL=21,22,23
FIELDS="p U k epsilon nut"
worst() { python3 "$CMP" "$@" | awk '/^MAX/{print $2, $3, $4, $5}'; }
# phi's worst entry, each entry on its own scale unless that scale is round-off of the field's (below 1e-12 of its largest
# entry at that time), which is held on the field's scale: frontAndBack's flux is ~1e-15 against faces of ~3e+03
worstphi()
{
    python3 "$CMP" "$@" phi | python3 -c '
import sys
rows = []
for line in sys.stdin:
    f = line.split()
    if len(f) >= 10 and f[0] != "MAX":
        rows.append((f[0], f[2], float(f[5]), float(f[7])))
best = (0.0, "", "")
for t in set(r[0] for r in rows):
    mine = [r for r in rows if r[0] == t]
    top = max(r[2] for r in mine)
    for _, entry, s, d in mine:
        scale = s if s >= 1e-12 * top else top
        rel = d / scale if scale > 0 else (0.0 if d == 0 else float("inf"))
        if rel >= best[0]:
            best = (rel, t, entry)
print("%.3e %s phi %s" % best)'
}
le() { python3 -c "import sys; sys.exit(0 if float('$1') <= float('$2') else 1)"; }
gt() { python3 -c "import sys; sys.exit(0 if float('$1') > float('$2') else 1)"; }

for a in W1 W2; do
    stage "$W/$a-of" "$a"
    ofrun "$W/$a-of"
    stage "$W/$a-br" "$a"
    brrun "$W/$a-br"
    w=$(worst "$W/$a-of" "$W/$a-br" "$TL" $FIELDS)
    wp=$(worstphi "$W/$a-of" "$W/$a-br" "$TL")
    if [ "$BOUND" = report ]; then
        say "$a: worst entry at 21-23 $w; phi $wp" INFO
        continue
    fi
    le "${w%% *}" "$BOUND" && say "$a: every entry of $FIELDS at 21-23 within $BOUND (worst $w)" PASS \
                           || say "$a: every entry of $FIELDS at 21-23 within $BOUND (worst $w)" FAIL
    le "${wp%% *}" "$BOUND" && say "$a: phi at 21-23 within $BOUND (worst ${wp%% *})" PASS \
                            || say "$a: phi at 21-23 within $BOUND (worst ${wp%% *})" FAIL
done

# PREMISE: the relaxation reaches what W2 holds
w=$(worst "$W/W1-of" "$W/W2-of" "$TL" $FIELDS)
gt "${w%% *}" "$PREMISE" && say "premise: OpenFOAM at p relaxation 0.3 differs from 1 ($w)" PASS \
                         || say "premise: OpenFOAM at p relaxation 0.3 differs from 1 ($w)" FAIL

# CONTROLS
CB=$BOUND
[ "$CB" = report ] && CB=1e-9
for c in W1:BRAE_CONTROL_STEADY_UPDATE_POINTS_LEGACY W2:BRAE_CONTROL_TOTAL_PRESSURE_UNRELAXED W1:BRAE_CONTROL_PASS0_GRAD_BEFORE_TOTAL_PRESSURE; do
    a=${c%%:*}
    v=${c#*:}
    stage "$W/$a-ctl" "$a"
    brrun "$W/$a-ctl" "$v=1"
    w=$(worst "$W/$a-of" "$W/$a-ctl" "$TL" $FIELDS)
    gt "${w%% *}" "$CB" && say "control $v=1: $a goes red ($w)" PASS \
                        || say "control $v=1: $a goes red ($w)" FAIL
done

[ $fail = 0 ] && echo "PASS" || echo "FAIL"
exit $fail
