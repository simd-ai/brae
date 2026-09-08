#!/usr/bin/env bash
# ============================================================================
#  brae rhoSimpleFoam (the OF-mirror, CUDA arm) against OpenFOAM's rhoSimpleFoam on N CPU cores,
#  on the stock compressible squareBend tutorial scaled past 500k cells.
#
#  TWO NUMBERS PER MESH, and they answer different questions:
#    wall      total wall time for a FIXED number of SIMPLE iterations, prep (blockMesh, decompose)
#              excluded on both sides -- the same rule as ../run_benchmark.sh.
#    agreement relative L2 of U, p, T, k, epsilon at that iteration, brae vs OpenFOAM. This is a
#              TRAJECTORY comparison: brae solves p with AMG-PCG where OpenFOAM's tutorial names GAMG
#              (a substitution brae announces), so the two take different paths to the same fixed
#              point and the number here says how far apart the paths are at iteration ITERS -- not
#              how far apart the converged answers are. Run with MODE=converged for that (both codes
#              to the tutorial's residualControl; slower).
#
#  The tutorial ships with sampling functionObjects that need triSurfaces built by its Allrun.pre;
#  `functions` is stripped from BOTH copies (same physics, no dependence on the surfaces).
#  residualControl is removed in the fixed-iteration mode so both codes run exactly ITERS.
#
#  Usage:   ./run_benchmark.sh
#  Env:
#     BRAE       brae binary                (default: ../../build/brae)
#     OFBASHRC   OpenFOAM etc/bashrc        (default: autodetect)
#     CORES      OpenFOAM CPU cores         (default: 20)
#     SIZES      blockMesh scale factors, ALL THREE directions, fractions allowed (default: "1 2"  ~= 112k / 896k cells; 1.65 ~= 503k;
#                3 ~= 3.0M, 4 ~= 7.2M)
#     ITERS      SIMPLE iterations timed    (default: 100)
#     MODE       fixed (default) | converged
#     TRANSONIC  yes (default, the tutorial) | no: symmetric pressure, AMG-PCG path in brae, GAMG in OF
#     MASSFLOW   inlet massFlowRate in kg/s for both codes (default: the tutorial's 0.5; 0.1 is subsonic)
#     WORK       scratch dir                (default: /tmp/brae_bench_rho)
#  Nothing here goes into validation/: these meshes are generated and thrown away.
# ============================================================================
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
BRAE="${BRAE:-$HERE/../../build/brae}"
CORES="${CORES:-20}"; ITERS="${ITERS:-100}"; SIZES="${SIZES:-1 2}"; WORK="${WORK:-/tmp/brae_bench_rho}"; MODE="${MODE:-fixed}"
# TRANSONIC=no flips the tutorial's `transonic yes` in BOTH codes: the pressure matrix is then symmetric and
# brae's p goes through AMG-PCG instead of the diagonal BiCGStab -- the subsonic compressible path, measured
# so the crossover question has an answer on it too. Same mesh, same everything else; not the tutorial.
TRANSONIC="${TRANSONIC:-yes}"
# MASSFLOW=<kg/s> replaces the inlet's `massFlowRate constant 0.5` in BOTH codes. The tutorial is transonic
# at 0.5; flipping `transonic no` on it is NOT a subsonic measurement -- OpenFOAM itself aborts within 7
# iterations there (measured). At 0.1 kg/s the flow is subsonic and `transonic no` is the right setting.
MASSFLOW="${MASSFLOW:-}"
OFBASHRC="${OFBASHRC:-$(ls /usr/lib/openfoam/openfoam*/etc/bashrc /opt/openfoam*/etc/bashrc 2>/dev/null | head -1)}"
set +u; source "$OFBASHRC" >/dev/null 2>&1; set -u
[ -x "$BRAE" ] || { echo "ERROR: brae binary not found at '$BRAE' (set BRAE=...)"; exit 1; }
command -v rhoSimpleFoam >/dev/null || { echo "ERROR: OpenFOAM not sourced (set OFBASHRC=...)"; exit 1; }
TUT="$FOAM_TUTORIALS/compressible/rhoSimpleFoam/squareBend"
[ -d "$TUT" ] || { echo "ERROR: tutorial not found at $TUT"; exit 1; }
echo "brae=$BRAE (OF-mirror, cuda) | OF cores=$CORES | iters=$ITERS | mode=$MODE | sizes=$SIZES | transonic=$TRANSONIC | massflow=${MASSFLOW:-tutorial}"

mkgrid(){ local M="$1" d="$2"; rm -rf "$d"; mkdir -p "$d"
  cp -r "$TUT/constant" "$TUT/system" "$d"/; cp -r "$TUT/0.orig" "$d/0"
  python3 - "$d" "$M" "$ITERS" "$MODE" "$TRANSONIC" "$MASSFLOW" <<'PY'
import re, sys
d, M, iters, mode, transonic, massflow = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
if massflow:
    f = d + '/0/U'; s = open(f).read()
    s, k = re.subn(r'\bmassFlowRate\s+constant\s+[-0-9.eE]+\s*;', 'massFlowRate constant %s;' % massflow, s)
    assert k == 1, 'expected one massFlowRate constant entry in 0/U'
    open(f, 'w').write(s)
f = d + '/system/blockMeshDict'; s = open(f).read()
# every hex block's (nx ny nz), scaled in all three directions; a fractional factor rounds each count
# (1.65 puts the 112k tutorial at ~503k cells)
s = re.sub(r'\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)\s*simpleGrading',
           lambda m: '(%d %d %d) simpleGrading' % tuple(max(1, int(round(int(m[k])*M))) for k in (1, 2, 3)), s)
open(f, 'w').write(s)
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*', '', s, flags=re.S)            # the sampling FOs need surfaces Allrun.pre builds
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval %s;' % iters, s)
if mode == 'fixed':
    s = re.sub(r'\bendTime\s+[^;]*;', 'endTime %s;' % iters, s)
open(c, 'w').write(s + '\n')
f = d + '/system/fvSolution'; s = open(f).read()
if mode == 'fixed':
    s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { }', s)   # exactly ITERS iterations, both codes
if transonic == 'no':
    s, k = re.subn(r'\btransonic\s+yes\s*;', 'transonic no;', s)
    assert k == 1, 'expected one `transonic yes;` to flip'
open(f, 'w').write(s)
PY
  ( cd "$d"; blockMesh > log.blockMesh 2>&1 ) || { echo "ERROR: blockMesh failed in $d"; tail -5 "$d/log.blockMesh"; exit 1; }; }
wall(){ local a b; a=$(date +%s.%N); eval "$1" >/dev/null 2>&1; b=$(date +%s.%N); echo "$b - $a"|bc; }
fmt(){ [ "${1:-}" = "-" ] || [ -z "${1:-}" ] && echo "-" || printf "%.1f" "$1"; }
agree2(){   # agree2 <braeTimeDir> <ofTimeDir> -> "U p T k epsilon" relative L2 values
  python3 - "$1" "$2" <<'PY'
import re, sys, os, numpy as np
b, o = sys.argv[1:3]
def read(fn):
    if not os.path.exists(fn): return None
    raw = open(fn, 'rb').read()
    m = re.search(rb'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n(\d+)\s*\n\(', raw)
    if not m: return None
    typ, n, start = m.group(1).decode(), int(m.group(2)), m.end()
    nc = 3 if typ == 'vector' else 1
    fm = re.search(r'format\s+(\w+)', raw[:1024].decode('latin-1'))
    if fm and fm.group(1) == 'binary': return np.frombuffer(raw[start:start+n*nc*8], dtype='<f8').reshape(n, nc)
    txt = raw[start:].decode('latin-1'); vals = re.findall(r'[-+0-9.eE]+', txt.split(')\n;')[0] if ')\n;' in txt else txt)
    return np.array([float(x) for x in vals[:n*nc]]).reshape(n, nc)
out = []
for f in ('U', 'p', 'T', 'k', 'epsilon'):
    x, y = read('%s/%s' % (b, f)), read('%s/%s' % (o, f))
    out.append('-' if x is None or y is None or x.shape != y.shape else '%.2e' % (np.linalg.norm(x - y) / np.linalg.norm(y)))
print(' '.join(out))
PY
}
mkdir -p "$WORK"; RES="$WORK/results.csv"
echo "nCells,brae_s,of${CORES}core_s,iters_brae,iters_of,relL2_U,relL2_p,relL2_T,relL2_k,relL2_epsilon" > "$RES"
printf "\n%10s %9s %11s %7s %7s  %-45s\n" "nCells" "brae" "OF-${CORES}c" "it_brae" "it_OF" "rel L2 at the last iteration: U p T k epsilon"
printf "%10s %9s %11s %7s %7s  %-45s\n" "------" "----" "------" "-------" "-----" "-------------------------------------------"
for M in $SIZES; do
  SRC="$WORK/mesh_$M"; mkgrid "$M" "$SRC"
  NC=$(grep -aoE 'nCells:?[[:space:]]*[0-9]+' "$SRC/constant/polyMesh/owner" 2>/dev/null | grep -oE '[0-9]+' | head -1)
  [ -n "$NC" ] || NC=$(grep -iE 'nCells' "$SRC/log.blockMesh" 2>/dev/null | grep -oE '[0-9]+' | tail -1)
  # --- brae, the OF-mirror on the device ---
  BW="$WORK/brae_$M"; rm -rf "$BW"; cp -r "$SRC" "$BW"
  tB=$(wall "BRAE_RHOSIMPLEFOAM_MIRROR=cuda '$BRAE' -case '$BW' > '$BW/log.brae' 2>&1")
  iB=$(grep -c '^Time = ' "$BW/log.brae"); tLastB=$(ls -d "$BW"/[1-9]* 2>/dev/null | sed 's|.*/||' | sort -n | tail -1)
  # --- OpenFOAM, N cores ---
  OW="$WORK/of_$M"; rm -rf "$OW"; cp -r "$SRC" "$OW"
  printf 'FoamFile{version 2.0;format ascii;class dictionary;object decomposeParDict;}\nnumberOfSubdomains %d;method scotch;\n' "$CORES" > "$OW/system/decomposeParDict"
  ( cd "$OW"; decomposePar -force > log.decomposePar 2>&1 )                       # decompose (EXCLUDED)
  tO=$(wall "( cd '$OW'; mpirun -np $CORES rhoSimpleFoam -parallel > log.of 2>&1 )")
  ( cd "$OW"; reconstructPar -latestTime > log.reconstructPar 2>&1 )              # for the comparison (EXCLUDED)
  iO=$(grep -c '^Time = ' "$OW/log.of"); tLastO=$(ls -d "$OW"/[1-9]* 2>/dev/null | sed 's|.*/||' | sort -n | tail -1)
  # fixed mode: both wrote ITERS. converged mode: each wrote its own last iteration, and those are the
  # two converged states -- compare them regardless of the iteration count they were reached at.
  AG="-"; [ -n "${tLastB:-}" ] && [ -n "${tLastO:-}" ] && AG=$(agree2 "$BW/$tLastB" "$OW/$tLastO")
  printf "%10s %9s %11s %7s %7s  %-45s\n" "$NC" "$(fmt "$tB")" "$(fmt "$tO")" "$iB" "$iO" "$AG"
  echo "$NC,$(fmt "$tB"),$(fmt "$tO"),$iB,$iO,$(echo "$AG" | tr ' ' ',')" >> "$RES"
  rm -rf "$OW"/processor* 2>/dev/null
done
echo; echo "CSV -> $RES   (work dirs kept under $WORK for inspection; delete when done)"
