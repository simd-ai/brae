#!/usr/bin/env bash
# brae solves the momentum components OpenFOAM solves, and no others.
#
# fvMatrix<vector>::solveSegregated walks the three components and `continue`s on every one whose
# validComponents entry is -1 (fvMatrixSolve.C:157-164); fvMesh::validComponents<vector>() IS
# polyMesh::solutionD() (fvMeshTemplates.C:32-44), which knocks out the directions normal to a non-empty
# EMPTY patch (polyMesh.C:75-118). So on a 2D case OpenFOAM never forms the out-of-plane momentum system:
# no solve, no `Solving for Uz` line, a SolverPerformance left default-constructed at Zero, and
# fvMatrix::H() replaces that component with Zero as its closing act (fvMatrix.C's validComponents loop).
# Wedge patches knock out geometricD_ only, so an axisymmetric case still solves all three.
#
# brae solved it. The system it solved has a ~0 source and a ~0 field, so its normFactor-scaled residual
# is meaningless -- on T3A it read 6.83e-01 every single iteration, next to 5.7e-07 for Ux at convergence
# -- and it was a third of the momentum linear algebra for an answer of zero.
#
# THE ORACLE IS OPENFOAM'S OWN LOG, committed with the fixtures: validation/t3a_of/log.sf is a real
# simpleFoam run of this mesh and carries 269 `Solving for Ux`, 269 `Solving for Uy` and ZERO
# `Solving for Uz`; validation/duct3d_of/log.sf, a 3D case, carries 147 of the Uz line. The gate reads
# both rather than hard-coding "two components on a 2D mesh", so a fixture swap cannot quietly change
# what is being asserted.
#
#   ARM 1   legacy driver (the default, and the path the tutorial gates run) on the 2D T3A.
#   ARM 2   the V2 mirror on the same case.
#   ARM 3   CONTROL, and the one that fails if the mask is too eager: the 3D duct must still solve all
#           three, and must print no knocked-out-direction notice.
#   Each arm: the components brae solves per iteration are exactly the ones OpenFOAM's log shows, and
#   on the 2D case the written Uz is at round-off against Ux (OpenFOAM's is exactly zero, brae's z
#   quantities are round-off nonzero -- see the header of solveMomentumPredictor).
#
# THE 2D ARMS RUN 200 ITERATIONS, and that number is the second half of the gate, not a convenience.
# Skipping the solve is only safe because fvMatrix::H()'s closing block is ported with it: with the
# zeroing removed, the knocked-out direction is fed by round-off and AMPLIFIES about 1.15 per iteration.
# Measured on this fixture: max|Uz|/max|Ux| is 1.1e-15 at 20 iterations -- which would sail through any
# sane bound -- and 7.7e+00 at 200, i.e. Uz larger than Ux. Twenty iterations cannot see the defect the
# field check exists to catch.
#   CONTROL  the OpenFOAM logs must actually carry momentum lines, or the shape is read off nothing.
#
# FAIL-PROOFS, both RUN by hand, one per half:
#   (a) the skip removed (delete `if (!sd_.valid(kk)) continue;` in device_simple_foam.cu): ARM 1 read
#       200 Uz solves against OpenFOAM's 0 and the gate exited 1, while ARM 2 stayed clean -- which is
#       the arms doing their job, since only the legacy driver's skip was removed. NOTE what this
#       fail-proof first caught: the report mask used to be copied from solutionD independently of the
#       solve, so removing the skip left the log unchanged and the gate passed a driver that solved a
#       component OpenFOAM does not. The driver now sets solvedU where the solve happens, so the log is
#       a witness of the solve rather than a second opinion about it.
#   (b) H()'s zeroing removed (delete the memset loop after the interface contributions): the counts
#       stay right and max|Uz|/max|Ux| reads 7.699e+00 at 200 iterations against 5.4e-19, so the field
#       check fails alone. Neither half of the fix is redundant.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
TWOD="$ROOT/validation/T3A";      TWOD_OF="$ROOT/validation/t3a_of/log.sf"
THREED="$ROOT/validation/duct3d_cf"; THREED_OF="$ROOT/validation/duct3d_of/log.sf"
[ -x "$BRAE" ] || { echo "SKIP: no brae at $BRAE"; exit 77; }
for f in "$TWOD" "$THREED" "$TWOD_OF" "$THREED_OF"; do
    [ -e "$f" ] || { echo "SKIP: fixture $f missing"; exit 77; }
done
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-74s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
ofsolves() {   # ofsolves <log> -> "x y z" counts OpenFOAM printed
    printf '%s %s %s' "$(grep -c 'Solving for Ux' "$1")" "$(grep -c 'Solving for Uy' "$1")" \
                      "$(grep -c 'Solving for Uz' "$1")"
}
run() {   # run <dir> <src> <iterations> <env>
    local d="$1" src="$2" n="$3" envs="$4"
    rm -rf "$d"; mkdir -p "$d"
    cp -r "$src/constant" "$src/system" "$d/"
    if [ -d "$src/0.orig" ]; then cp -r "$src/0.orig" "$d/0"; else cp -r "$src/0" "$d/0"; fi
    python3 - "$d" "$n" <<'PY'
import re, sys
d, n = sys.argv[1:3]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime %s;' % n, s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval %s;' % n, s)
# residualControl must not stop the run early: the counts below are per iteration.
s = re.sub(r'\bstopAt\s+[^;]*;', 'stopAt endTime;', s)
open(c, 'w').write(s)
f = d + '/system/fvSolution'; s = open(f).read()
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { }', s)
open(f, 'w').write(s)
PY
    ( cd "$d" && env $envs "$BRAE" "$d" > log 2>&1 ) || true
}
counts() { printf '%s %s %s' "$(grep -c 'Solving for Ux' "$1/log")" "$(grep -c 'Solving for Uy' "$1/log")" \
                             "$(grep -c 'Solving for Uz' "$1/log")"; }
uzrel() {   # uzrel <dir> <time> -> max|Uz| / max|Ux| of the written field
    python3 - "$1/$2/U" <<'PY'
import sys, re
s = open(sys.argv[1]).read()
m = re.search(r'internalField\s+nonuniform\s+List<vector>\s*\n(\d+)\s*\n\(\s*\n(.*?)\n\)\s*;', s, re.S)
v = [l.strip().strip('()').split() for l in m.group(2).split('\n') if l.strip()]
z = max(abs(float(t[2])) for t in v); x = max(abs(float(t[0])) for t in v)
print(f'{(z/x if x else z):.3e}')
PY
}
N2D=200   # see the header: the field check needs the amplification to have run
N3D=20
read -r ofx ofy ofz <<<"$(ofsolves "$TWOD_OF")"
read -r o3x o3y o3z <<<"$(ofsolves "$THREED_OF")"
printf '  OpenFOAM %s: Ux %s, Uy %s, Uz %s\n' "$(basename "$(dirname "$TWOD_OF")")" "$ofx" "$ofy" "$ofz"
printf '  OpenFOAM %s: Ux %s, Uy %s, Uz %s\n' "$(basename "$(dirname "$THREED_OF")")" "$o3x" "$o3y" "$o3z"
[ "$ofx" -gt 0 ] && [ "$o3x" -gt 0 ] && say "the OpenFOAM logs carry momentum solves to read the shape from" ok \
                                     || say "the OpenFOAM logs carry momentum solves to read the shape from" FAIL
# OpenFOAM's own shape: which components it solved on each mesh.
of2d_z=$([ "$ofz" -gt 0 ] && echo yes || echo no)
of3d_z=$([ "$o3z" -gt 0 ] && echo yes || echo no)
[ "$of2d_z" = no ] && [ "$of3d_z" = yes ] \
    && say "OpenFOAM solves z on the 3D mesh and not on the 2D one (the fixtures still differ)" ok \
    || say "OpenFOAM solves z on the 3D mesh and not on the 2D one (the fixtures still differ)" FAIL

arm2d() {   # arm2d <label> <env>
    local label="$1" envs="$2" d="$W/${1// /_}"
    run "$d" "$TWOD" "$N2D" "$envs"
    local bx by bz; read -r bx by bz <<<"$(counts "$d")"
    printf '  %s  brae: Ux %s, Uy %s, Uz %s over %s iterations\n' "$label" "$bx" "$by" "$bz" "$N2D"
    [ "$bx" -eq "$N2D" ] && [ "$by" -eq "$N2D" ] \
        && say "$label  solves the components OpenFOAM solves, every iteration" ok \
        || say "$label  solves the components OpenFOAM solves, every iteration" FAIL
    [ "$bz" -eq 0 ] && say "$label  solves no component OpenFOAM knocked out" ok \
                    || say "$label  solves no component OpenFOAM knocked out" FAIL
    grep -q "empty patches knock out a solution direction" "$d/log" \
        && say "$label  says so, rather than differing from OpenFOAM in silence" ok \
        || say "$label  says so, rather than differing from OpenFOAM in silence" FAIL
    local r; r=$(uzrel "$d" "$N2D")
    printf '  %s  written field: max|Uz|/max|Ux| = %s (OpenFOAM: exactly 0)\n' "$label" "$r"
    python3 -c "import sys; sys.exit(0 if $r < 1e-12 else 1)" \
        && say "$label  the knocked-out direction stays at round-off" ok \
        || say "$label  the knocked-out direction stays at round-off" FAIL
}
arm2d "ARM 1  legacy" ""
arm2d "ARM 2  mirror" "BRAE_SIMPLEFOAM_V2=1"
# ARM 3 -- the control. A mask that fired on a 3D mesh would stop the flow dead in one direction.
d3="$W/ARM3"
run "$d3" "$THREED" "$N3D" ""
read -r cx cy cz <<<"$(counts "$d3")"
printf '  ARM 3  3D: brae Ux %s, Uy %s, Uz %s over %s iterations\n' "$cx" "$cy" "$cz" "$N3D"
[ "$cx" -eq "$N3D" ] && [ "$cy" -eq "$N3D" ] && [ "$cz" -eq "$N3D" ] \
    && say "ARM 3  a mesh with no empty patch still solves all three" ok \
    || say "ARM 3  a mesh with no empty patch still solves all three" FAIL
grep -q "empty patches knock out a solution direction" "$d3/log" \
    && say "ARM 3  ...and claims no knocked-out direction" FAIL \
    || say "ARM 3  ...and claims no knocked-out direction" ok
[ $fail -eq 0 ] && echo "PASS: brae solves the momentum components OpenFOAM solves, and no others"
exit $fail
