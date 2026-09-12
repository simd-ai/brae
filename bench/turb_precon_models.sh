#!/usr/bin/env bash
# ============================================================================
#  The SECOND axis of the same question (item 78): does the preconditioner a substituted PBiCGStab
#  carries on the transported turbulence scalars behave the same way across TURBULENCE MODELS, or was
#  the answer specific to kEpsilon on one compressible case?
#
#  The compressible mirror's CUDA arm is kEpsilon-only (it refuses kOmegaSST by name), so the model
#  axis is measured on the INCOMPRESSIBLE driver, which reads the same policy out of the same
#  linear_solver_setup.cuh. Each case here names `solver GAMG` on its turbulence pair, which is the
#  condition for the substitution: GAMG takes no `preconditioner`, so brae has to choose one.
#
#  Reported per case: the wall time for a fixed iteration count under each preconditioner, and the
#  turbulent viscosity's extremes at the end of it against real OpenFOAM on the same case. nut is the
#  quantity that shows the failure -- it is Cmu k^2/epsilon (or k/omega), so it explodes when the
#  dissipation scalar is driven to the bound floor.
#
#  Usage:  ./turb_precon_models.sh
#  Env:  BRAE, OFBASHRC, ITERS (default 60), CASES (default: one per model), WORK
#  Nothing here goes into validation/: every case is copied to WORK and thrown away.
# ============================================================================
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
BRAE="${BRAE:-$ROOT/build/brae}"
ITERS="${ITERS:-60}"; WORK="${WORK:-/tmp/brae_turb_models}"
OFBASHRC="${OFBASHRC:-$(ls /usr/lib/openfoam/openfoam*/etc/bashrc /opt/openfoam*/etc/bashrc 2>/dev/null | head -1)}"
set +u; source "$OFBASHRC" >/dev/null 2>&1; set -u
[ -x "$BRAE" ] || { echo "ERROR: no brae at '$BRAE'"; exit 1; }
# case:solver -- one per turbulence model brae carries on this path
# polyDual is deliberately not here: its fixture ships no 0/ or 0.orig, so neither code can start it.
CASES="${CASES:-pitzDailyTurb pitzDailyTurbBig pitzDailySST pitzDailyRKE lmFlatPlate}"
mkdir -p "$WORK"

stage(){ local name="$1" d="$2"
  rm -rf "$d"; mkdir -p "$d"
  cp -r "$ROOT/validation/$name/constant" "$ROOT/validation/$name/system" "$d"/ 2>/dev/null || return 1
  if   [ -d "$ROOT/validation/$name/0.orig" ]; then cp -r "$ROOT/validation/$name/0.orig" "$d/0"
  elif [ -d "$ROOT/validation/$name/0" ];      then cp -r "$ROOT/validation/$name/0" "$d/0"
  else return 1; fi
  python3 - "$d" "$ITERS" <<'PY'
import re, sys
d, it = sys.argv[1], sys.argv[2]
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*', '', s, flags=re.S)
s = re.sub(r'\bendTime\s+[^;]*;', 'endTime %s;' % it, s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval %s;' % it, s)
s = re.sub(r'\bwriteControl\s+[^;]*;', 'writeControl timeStep;', s)
open(c, 'w').write(s + '\n')
f = d + '/system/fvSolution'; s = open(f).read()
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { }', s)
# THE CONDITION THE QUESTION IS ABOUT, imposed identically on every case and on both codes: the
# turbulence pair asks for GAMG at the tutorial's loose relTol. The validation fixtures ship tight
# (`relTol 0; tolerance 1e-10`) because they are gates, and at that setting the solve is converged, so
# the preconditioner cannot reach the answer at all -- which is what the first run of this scan showed
# and why it said nothing. GAMG names no `preconditioner`, so brae must choose one; that choice is the
# measurement.
GAMG = ('{\n        solver          GAMG;\n        smoother        GaussSeidel;\n'
        '        tolerance       1e-08;\n        relTol          0.1;\n    }')
n = 0
for f2 in ('k', 'epsilon', 'omega', 'nuTilda', 'ReThetat', 'gammaInt'):
    s, k2 = re.subn(r'(^\s*"?\(?[^"\n{}]*\b%s\b[^"\n{}]*\)?"?\s*\n?\s*)\{[^{}]*\}' % f2,
                    lambda m: m.group(1) + GAMG, s, flags=re.M)
    n += k2
assert n, 'no turbulence solver block found to rewrite'
open(f, 'w').write(s)
PY
  return 0
}
nutx(){ python3 - "$1" <<'PY'
import re, sys
try: b = open(sys.argv[1], 'rb').read()
except OSError: print("- - -"); raise SystemExit
m = re.search(rb'internalField\s+nonuniform\s+List<scalar>\s*\n?(\d+)\s*\n\(', b)
if not m: print("- - -"); raise SystemExit
v = [float(x) for x in b[m.end():].split(b')\n', 1)[0].split()]
print("%.3e %.3e %d" % (min(v), max(v), sum(1 for x in v if x <= 1e-14)))
PY
}
wall(){ local a b; a=$(date +%s.%N); eval "$1" >/dev/null 2>&1; b=$(date +%s.%N); python3 -c "print('%.2f'%($b-$a))"; }

echo "brae=$BRAE | iterations=$ITERS | cases: $CASES"
echo
printf "%-20s %-13s %7s  %8s %8s %6s   %-34s\n" "case" "model" "cells" "diag_s" "DILU_s" "ratio" "nut max at iteration $ITERS (floored)"
printf "%-20s %-13s %7s  %8s %8s %6s   %-34s\n" "----" "-----" "-----" "------" "------" "-----" "----------------------------------"
for name in $CASES; do
  d="$WORK/$name"
  stage "$name" "$d" || { printf "%-20s %s\n" "$name" "SKIP (no fixture)"; continue; }
  model=$(grep -oE "RASModel[[:space:]]+[A-Za-z]+" "$d/constant/turbulenceProperties" | awk '{print $2}' | head -1)
  app=$(grep -oE "^application[[:space:]]+[A-Za-z]+" "$d/system/controlDict" | awk '{print $2}' | head -1)
  [ -f "$d/constant/polyMesh/owner" ] || ( cd "$d" && blockMesh > log.blockMesh 2>&1 )
  NC=$(grep -aoE 'nCells:?[[:space:]]*[0-9]+' "$d/constant/polyMesh/owner" | grep -oE '[0-9]+' | head -1)
  rm -rf "$d"/[1-9]*
  TD=$(wall "cd '$d' && BRAE_DILU_KE=0 '$BRAE' -case '$d'")
  read a b c <<< "$(nutx "$d/$ITERS/nut")"; NDG="$b"; FDG="$c"
  rm -rf "$d"/[1-9]*
  TL=$(wall "cd '$d' && BRAE_DILU_KE=1 '$BRAE' -case '$d'")
  read a b c <<< "$(nutx "$d/$ITERS/nut")"; NDL="$b"; FDL="$c"
  NOF="-"; FOF="-"
  if command -v "$app" >/dev/null 2>&1; then
    rm -rf "$d"/[1-9]*
    ( cd "$d" && "$app" > of.log 2>&1 )
    read a b c <<< "$(nutx "$d/$ITERS/nut")"; NOF="$b"; FOF="$c"
  fi
  R=$(python3 -c "print('%.2f'%(${TL:-0}/${TD:-1}))" 2>/dev/null || echo "-")
  printf "%-20s %-13s %7s  %8s %8s %6s   diag %s(%s)  DILU %s(%s)  OF %s(%s)\n" \
         "$name" "${model:-?}" "${NC:-?}" "$TD" "$TL" "$R" "$NDG" "$FDG" "$NDL" "$FDL" "$NOF" "$FOF"
done
