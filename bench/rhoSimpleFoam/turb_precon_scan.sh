#!/usr/bin/env bash
# ============================================================================
#  WHICH PRECONDITIONER a substituted PBiCGStab should carry on k and epsilon, measured across MESH
#  SCALE rather than argued from one case (item 78).
#
#  A case that names `solver GAMG` on the turbulence pair names no `preconditioner` -- GAMG takes none --
#  so brae has to choose one for the PBiCGStab it substitutes. The two candidates are OpenFOAM's own:
#
#    diagonal  one divide per cell. Fully parallel; the weakest operator OpenFOAM has.
#    series    the truncated Neumann series, sum_{j<10} (I - D^-1 A)^j D^-1: nine sparse matrix-vector
#              products and nothing else -- no factorisation, no ordering, no dependency between cells.
#              brae's default on this seam since item 78, and what BRAE_POLY_KE=1 turns off.
#    DILU      an incomplete LU. OpenFOAM's default for an asymmetric matrix, and SEQUENTIAL: cell c
#              cannot be updated before its upstream neighbours, so the apply is a level-scheduled walk,
#              one kernel launch per dependency level. The level count grows with the mesh, which is
#              why this has to be measured against SIZE and not asserted from one of them.
#
#  TWO TABLES, because the two candidates differ on both counts:
#    SPEED   the turbulence block, ms per outer iteration, brae under each preconditioner.
#    HEALTH  nut and epsilon at outer iteration 8 against REAL OpenFOAM on the same mesh. The failure
#            being looked for is epsilon driven non-positive, floored at 1e-15 by bound(), and
#            nut = Cmu k^2/epsilon exploding. OpenFOAM is run three ways -- its own GAMG, and PBiCGStab
#            under each preconditioner -- because the question is whether brae's diagonal is FAITHFUL
#            (it is: OpenFOAM collapses with it too) or defective.
#
#  Usage:  ./turb_precon_scan.sh
#  Env:  BRAE, OFBASHRC, SIZES (blockMesh scale factors, default "0.6 1 1.4 2" ~= 24k/112k/306k/896k),
#        ITERS (timed iterations, default 100), HEALTH_IT (default 8), WORK
#  Nothing here goes into validation/: these meshes are generated and thrown away.
# ============================================================================
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
BRAE="${BRAE:-$HERE/../../build/brae}"
SIZES="${SIZES:-0.6 1 1.4 2}"; ITERS="${ITERS:-100}"; HEALTH_IT="${HEALTH_IT:-8}"
WORK="${WORK:-/tmp/brae_turb_precon}"
OFBASHRC="${OFBASHRC:-$(ls /usr/lib/openfoam/openfoam*/etc/bashrc /opt/openfoam*/etc/bashrc 2>/dev/null | head -1)}"
set +u; source "$OFBASHRC" >/dev/null 2>&1; set -u
[ -x "$BRAE" ] || { echo "ERROR: no brae at '$BRAE'"; exit 1; }
command -v rhoSimpleFoam >/dev/null || { echo "ERROR: OpenFOAM not sourced"; exit 1; }
TUT="$FOAM_TUTORIALS/compressible/rhoSimpleFoam/squareBend"
[ -d "$TUT" ] || { echo "ERROR: tutorial not at $TUT"; exit 1; }
mkdir -p "$WORK"

# mkgrid <scale> <dir> <endTime> [<k/epsilon solver block>]
mkgrid(){ local M="$1" d="$2" ET="$3" KE="${4:-}"
  rm -rf "$d"; mkdir -p "$d"
  cp -r "$TUT/constant" "$TUT/system" "$d"/; cp -r "$TUT/0.orig" "$d/0"
  python3 - "$d" "$M" "$ET" "$KE" <<'PY'
import re, sys
d, M, et, ke = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
f = d + '/system/blockMeshDict'; s = open(f).read()
s = re.sub(r'\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)\s*simpleGrading',
           lambda m: '(%d %d %d) simpleGrading' % tuple(max(1, int(round(int(m[k])*M))) for k in (1, 2, 3)), s)
open(f, 'w').write(s)
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*', '', s, flags=re.S)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval %s;' % et, s)
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime %s;' % et, s)
open(c, 'w').write(s + '\n')
f = d + '/system/fvSolution'; s = open(f).read()
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { }', s)
if ke:
    # split the tutorial's "(U|e|k|epsilon)" block so the pair can be given its own solver
    old = re.search(r'"\(U\|e\|k\|epsilon\)"\s*\{[^{}]*\}', s)
    assert old, 'the tutorial no longer writes one block for (U|e|k|epsilon)'
    body = old.group(0)
    s = s.replace(body, body.replace('"(U|e|k|epsilon)"', '"(U|e)"') + '\n\n    "(k|epsilon)"\n    {\n' + ke + '\n    }')
open(f, 'w').write(s)
PY
  ( cd "$d"; blockMesh > log.blockMesh 2>&1 ) || { echo "ERROR: blockMesh failed in $d"; tail -5 "$d/log.blockMesh"; exit 1; }
}

# health <field file> -> "<min> <max> <n at or below 1e-14>"
health(){ python3 - "$1" <<'PY'
import re, sys
try: b = open(sys.argv[1], 'rb').read()
except OSError: print("- - -"); raise SystemExit
m = re.search(rb'internalField\s+nonuniform\s+List<scalar>\s*\n?(\d+)\s*\n\(', b)
if not m: print("- - -"); raise SystemExit
v = [float(x) for x in b[m.end():].split(b')\n', 1)[0].split()]
print("%.3e %.3e %d" % (min(v), max(v), sum(1 for x in v if x <= 1e-14)))
PY
}
turbms(){ grep -oE 'turbulence [0-9.]+ s \([0-9.]+ ms/it\)' "$1" | grep -oE '\([0-9.]+' | tr -d '(' | tail -1; }

echo "brae=$BRAE | sizes=$SIZES | timed iterations=$ITERS | health at iteration $HEALTH_IT"
echo
echo "SPEED -- brae, the turbulence block, ms per outer iteration"
printf "%10s %8s %10s %10s %10s %14s\n" "nCells" "levels" "diagonal" "series" "DILU" "4 phases series"
printf "%10s %8s %10s %10s %10s %14s\n" "------" "------" "--------" "------" "----" "---------------"
SPEED=""
for M in $SIZES; do
  d="$WORK/speed_$M"; mkgrid "$M" "$d" "$ITERS"
  NC=$(grep -aoE 'nCells:?[[:space:]]*[0-9]+' "$d/constant/polyMesh/owner" | grep -oE '[0-9]+' | head -1)
  ( cd "$d" && BRAE_PHASE_TIME=1 BRAE_POLY_KE=1 BRAE_DILU_KE=0 BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BRAE" -case "$d" > diag.log 2>&1 )
  ( cd "$d" && BRAE_PHASE_TIME=1 BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BRAE" -case "$d" > poly.log 2>&1 )
  ( cd "$d" && BRAE_PHASE_TIME=1 BRAE_DILU_KE=1 BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BRAE" -case "$d" > dilu.log 2>&1 )
  DG=$(turbms "$d/diag.log"); PS=$(turbms "$d/poly.log"); DL=$(turbms "$d/dilu.log")
  # the DILU level count: how many kernel launches one apply costs
  LV=$(grep -aoE 'DILU[^.]*[0-9]+ levels' "$d/dilu.log" | grep -oE '[0-9]+ levels' | grep -oE '[0-9]+' | head -1)
  FOUR=$(grep -oE 'the four total [0-9.]+ s \([0-9.]+ ms/it\)' "$d/poly.log" | grep -oE '\([0-9.]+' | tr -d '(' | tail -1)
  printf "%10s %8s %10s %10s %10s %14s\n" "${NC:-?}" "${LV:--}" "${DG:--}" "${PS:--}" "${DL:--}" "${FOUR:--}"
  SPEED="$SPEED $NC:$DG:$PS:$DL"
done

echo
echo "HEALTH -- nut and epsilon at outer iteration $HEALTH_IT, against real OpenFOAM on the same mesh"
printf "%10s  %-26s %12s %12s %10s\n" "nCells" "arm" "epsilon min" "nut max" "nut floored"
printf "%10s  %-26s %12s %12s %10s\n" "------" "---" "-----------" "-------" "-----------"
for M in $SIZES; do
  for arm in "brae diagonal" "brae series" "brae DILU" "OF GAMG" "OF diagonal" "OF DILU"; do
    case "$arm" in
      "brae diagonal") d="$WORK/h_${M}_bd"; mkgrid "$M" "$d" "$HEALTH_IT"
                       ( cd "$d" && BRAE_POLY_KE=1 BRAE_DILU_KE=0 BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BRAE" -case "$d" > run.log 2>&1 ) ;;
      "brae series")   d="$WORK/h_${M}_bp"; mkgrid "$M" "$d" "$HEALTH_IT"
                       ( cd "$d" && BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BRAE" -case "$d" > run.log 2>&1 ) ;;
      "brae DILU")     d="$WORK/h_${M}_bl"; mkgrid "$M" "$d" "$HEALTH_IT"
                       ( cd "$d" && BRAE_DILU_KE=1 BRAE_RHOSIMPLEFOAM_MIRROR=cuda "$BRAE" -case "$d" > run.log 2>&1 ) ;;
      "OF GAMG")       d="$WORK/h_${M}_og"; mkgrid "$M" "$d" "$HEALTH_IT"
                       ( cd "$d" && rhoSimpleFoam > run.log 2>&1 ) ;;
      "OF diagonal")   d="$WORK/h_${M}_od"; mkgrid "$M" "$d" "$HEALTH_IT" \
                       "        solver          PBiCGStab;
        preconditioner  diagonal;
        tolerance       1e-08;
        relTol          0.1;"
                       ( cd "$d" && rhoSimpleFoam > run.log 2>&1 ) ;;
      "OF DILU")       d="$WORK/h_${M}_ol"; mkgrid "$M" "$d" "$HEALTH_IT" \
                       "        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-08;
        relTol          0.1;"
                       ( cd "$d" && rhoSimpleFoam > run.log 2>&1 ) ;;
    esac
    NC=$(grep -aoE 'nCells:?[[:space:]]*[0-9]+' "$d/constant/polyMesh/owner" | grep -oE '[0-9]+' | head -1)
    read E_MIN E_MAX E_FL <<< "$(health "$d/$HEALTH_IT/epsilon")"
    read N_MIN N_MAX N_FL <<< "$(health "$d/$HEALTH_IT/nut")"
    printf "%10s  %-26s %12s %12s %10s\n" "$NC" "$arm" "$E_MIN" "$N_MAX" "$N_FL"
  done
  echo
done
