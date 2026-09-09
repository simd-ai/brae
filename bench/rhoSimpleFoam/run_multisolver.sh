#!/usr/bin/env bash
# ============================================================================
#  rhoSimpleFoam GPU shootout -- brae against every other way to put this solver on a GPU.
#
#  This is the compressible counterpart of ../H100/run_benchmark.sh (which is simpleFoam on scaled
#  pitzDaily). The case is OpenFOAM's own compressible/rhoSimpleFoam/squareBend, scaled in all three
#  directions: scale 1 = 112k cells, 2 = 896k, 3 = 3.02M, 4 = 7.17M.
#
#  Runners (auto-detected; a back-end that is not built is simply blank, never substituted):
#    brae          the OF-mirror, CUDA arm, whole loop device-resident
#    OF-CPU        stock OpenFOAM rhoSimpleFoam on N cores (native GAMG, MPI)
#    SPUMA         CINECA's device-resident OpenFOAM-v2412 fork      (SPUMA_BIN=.../bin/rhoSimpleFoam)
#    OF+AMGX       stock OpenFOAM with the PRESSURE solve offloaded to AMGX   (libamgxFoam.so)
#    OF+PETSc      stock OpenFOAM with the PRESSURE solve offloaded to PETSc/cuSPARSE (libpetscFoam.so)
#
#  THREE THINGS THIS HARNESS DOES THAT THE simpleFoam ONE DOES NOT NEED TO, and they are the reason it
#  is a separate script rather than a flag on that one:
#
#  1. THE PRESSURE MATRIX IS ASYMMETRIC HERE. squareBend ships `transonic yes`, so pEqn carries
#     fvm::div(phid, p) and p's matrix has lower != upper (rhoSimpleFoam/pEqn.H). CG and its
#     preconditioners are only defined for a symmetric operator, so the offload arms are configured
#     with BiCGStab on the transonic case and CG on the subsonic one -- announced in the header line.
#     Handing an asymmetric matrix to `ksp_type cg` produces a number, and the number means nothing.
#
#  2. AN ARM THAT DID NOT FINISH IS NOT TIMED. Compressible runs abort: the thermo's Newton solve
#     fatals ("Maximum number of iterations exceeded ... when starting from T0") on a bad partition or
#     a diverging pressure, and OpenFOAM exits mid-run. A wall time from a run that stopped at
#     iteration 4 is faster than every honest number in the table. Every arm's completed iteration
#     count is counted from its own log and printed; short arms report `N/ITERS it`, never a time.
#
#  2b. THE OFFLOAD ARMS ARE CONFIGURED FROM MEASUREMENT, not from the defaults. squareBend ships
#     massFlowRate 0.5 kg/s, which is near-choked, and an AGGREGATION-AMG preconditioner does not
#     survive that pressure operator. Measured on GB10 at 112k, 20 iterations:
#       PETSc  pc_type gamg   -> diverges (5/20; p residual 26.7 after 537 its at outer iteration 2),
#              on `mat_type aijcusparse` AND on the CPU matrix type, so it is the preconditioner.
#       PETSc  pc_type ilu    -> 20/20 on cuSPARSE.   pc_type jacobi -> 20/20 on cuSPARSE.
#       OF     PBiCGStab/DILU -> 20/20 (160, then 9, then 3 inner iterations).
#     So the PETSc arm runs `ksp_type <bcgs|cg>; mat_type aijcusparse; pc_type ilu` -- the configuration
#     that solves the case, which is what a benchmark owes a competitor. PETSC_PC= overrides it.
#
#  2c. THE AMGX ARM DOES NOT CONVERGE ON THIS CASE, and the evidence says the fault is in the wrapper
#     (bench/amgxFoam/amgxSolver.C, brae's own) rather than in AMGX or in the matrix:
#       - EVERY config fails at the same outer iteration: PBICGSTAB and FGMRES crossed with
#         BLOCK_JACOBI, MULTICOLOR_DILU, MULTICOLOR_GS, aggregation AMG, and NOSOLVER (no
#         preconditioner at all) -- nine configurations, all 4/20.
#       - The tolerance is not being honoured: at main:tolerance 1e-1, 2e-2, 5e-3 and 1e-3 the solve
#         stops at 13 inner iterations every time and OpenFOAM measures the same ~0.27 final residual
#         against a requested relTol of 0.1. AMGX reports convergence on a system whose residual
#         OpenFOAM does not see fall, so the two are not looking at the same system.
#       - The asymmetric matrix is NOT the trigger: the same `transonic yes` case at massFlowRate 0.1
#         runs 10/10 through the same wrapper.
#     The column is therefore reported as not-completing rather than silently dropped, and
#     AMGX_CFG=<file.json> takes a hand-written config for anyone who wants to try. Fixing the wrapper
#     is a separate unit of work, filed in REFUSALS.md.
#
#  3. ONLY p IS OFFLOADED on the AMGX and PETSc arms, because that is all those libraries plug into.
#     squareBend asks GAMG for (U|e|k|epsilon) too and those stay on OpenFOAM's CPU GAMG. The column
#     is "OpenFOAM with its pressure solve on the GPU", not "OpenFOAM on the GPU"; brae and SPUMA are
#     the two whole-loop columns. Reading the table without that distinction overstates the offloads.
#
#  Prep (blockMesh, decomposePar, brae's -partition cache) is EXCLUDED from every timed number, the
#  same rule as the other harnesses.
#
#  Usage:   ./run_multisolver.sh
#  Env:
#     BRAE        brae binary                     (default: ../../build/brae)
#     OFBASHRC    OpenFOAM etc/bashrc             (default: autodetect)
#     CORES       OpenFOAM CPU cores              (default: 24)
#     SIZES       blockMesh scale factors         (default: "1 2"; 3 = 3.0M, 4 = 7.2M)
#     ITERS       SIMPLE iterations timed         (default: 100)
#     TRANSONIC   yes (default, the tutorial) | no  -- flips it in EVERY arm, and picks the Krylov
#     MASSFLOW    inlet kg/s in every arm         (default: the tutorial's 0.5; use 0.1 with TRANSONIC=no)
#     WORK        scratch dir                     (default: /tmp/brae_bench_rho_multi)
#     SPUMA_BIN   SPUMA's rhoSimpleFoam           SPUMA_POOL (GB, default 24)  NVARCH (default: autodetect)
#     AMGX_DIR    (default $HOME/opt/amgx)   PETSC_DIR (default $HOME/petsc)  PETSC_ARCH (default arch-cuda)
#     PETSC_PC    PETSc preconditioner            (default: ilu -- see note 2b; gamg diverges here)
#     AMGX_CFG    an AMGX JSON config file        (default: the one written below)
#     KEEP=1      keep the scratch dirs for inspection
#
#  Nothing here is written into validation/: the meshes are generated and thrown away.
# ============================================================================
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
BRAE="${BRAE:-$HERE/../../build/brae}"
CORES="${CORES:-24}"; ITERS="${ITERS:-100}"; SIZES="${SIZES:-1 2}"
WORK="${WORK:-/tmp/brae_bench_rho_multi}"
TRANSONIC="${TRANSONIC:-yes}"; MASSFLOW="${MASSFLOW:-}"; KEEP="${KEEP:-0}"
SPUMA_BIN="${SPUMA_BIN:-}"; SPUMA_POOL="${SPUMA_POOL:-24}"
# ilu and not gamg: measured, see note 2b. gamg is PETSc's headline AMG and it diverges on this case's
# pressure, so using it would report "PETSc cannot run this" when PETSc can.
PETSC_PC="${PETSC_PC:-ilu}"; AMGX_CFG="${AMGX_CFG:-}"
NVARCH="${NVARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d .)}"
AMGX_DIR="${AMGX_DIR:-$HOME/opt/amgx}"; PETSC_DIR="${PETSC_DIR:-$HOME/petsc}"; PETSC_ARCH="${PETSC_ARCH:-arch-cuda}"
OFBASHRC="${OFBASHRC:-$(ls /usr/lib/openfoam/openfoam*/etc/bashrc /opt/openfoam*/etc/bashrc 2>/dev/null | head -1)}"
set +u; source "$OFBASHRC" > /dev/null 2>&1; set -u

command -v rhoSimpleFoam > /dev/null || { echo "ERROR: OpenFOAM not sourced (set OFBASHRC=...)"; exit 1; }
TUT="$FOAM_TUTORIALS/compressible/rhoSimpleFoam/squareBend"
[ -d "$TUT" ] || { echo "ERROR: tutorial not found at $TUT"; exit 1; }
[ -x "$BRAE" ] || echo "WARNING: no brae at '$BRAE' -- that column stays blank"

HAVE_BRAE=0;  [ -x "$BRAE" ] && HAVE_BRAE=1
HAVE_AMGX=0;  [ -f "${FOAM_USER_LIBBIN:-}/libamgxFoam.so"  ] && HAVE_AMGX=1
HAVE_PETSC=0; [ -f "${FOAM_USER_LIBBIN:-}/libpetscFoam.so" ] && HAVE_PETSC=1
HAVE_SPUMA=0
if [ -n "$SPUMA_BIN" ]; then
    # REFUSED rather than substituted: SPUMA is a full OpenFOAM fork and ships every solver, so a
    # SPUMA_BIN pointing at simpleFoam would run the WRONG SOLVER on a compressible case and report a
    # time for it. The basename has to be the solver this benchmark is about.
    if [ ! -x "$SPUMA_BIN" ]; then
        echo "ERROR: SPUMA_BIN='$SPUMA_BIN' is not executable."; exit 1
    elif [ "$(basename "$SPUMA_BIN")" != "rhoSimpleFoam" ]; then
        echo "ERROR: SPUMA_BIN must be SPUMA's OWN rhoSimpleFoam, not '$(basename "$SPUMA_BIN")'."
        echo "       (build it with: cd \$SPUMA_SRC && wmake applications/solvers/compressible/rhoSimpleFoam)"
        exit 1
    else
        HAVE_SPUMA=1
        SP_SRC="$(cd "$(dirname "$SPUMA_BIN")/../../.." && pwd)"
    fi
fi
# The offload plugins are PROVEN LOADABLE, not assumed loadable, and the test is the same dlopen
# OpenFOAM does. This is not pedantry: a `libs ("libpetscFoam.so")` that fails to load is only a
# WARNING in OpenFOAM (dlLibraryTable.C:188). The run continues, fatals later at the p solver lookup,
# and lands in this table as an arm that ran and aborted -- a far more flattering claim than "the
# plugin never loaded". Measured here: PETSc's arch-linux-c-opt is built against a different MPI than
# OpenFOAM's, so it dlopens with `undefined symbol: ompi_instance_count` while arch-sysmpi loads. An
# `ldd` check does NOT catch that -- every file resolves; only an actual load does.
dlopens(){   # dlopens <libpath> -- with the CURRENT LD_LIBRARY_PATH
    python3 -c 'import ctypes,sys; ctypes.CDLL(sys.argv[1])' "$1" 2>&1
}
export PETSC_OPTIONS="${PETSC_OPTIONS:--use_gpu_aware_mpi 0}"
if [ $HAVE_PETSC = 1 ]; then
    # Candidate arches, the caller's first. PETSc builds several arches side by side and only the one
    # built against OpenFOAM's own MPI will load.
    PETSC_OK=0
    for cand in "$PETSC_DIR/$PETSC_ARCH/lib/libpetsc.so" \
                "$HOME"/petsc/*/lib/libpetsc.so "$HOME"/space/*/petsc/*/lib/libpetsc.so \
                /opt/petsc/*/lib/libpetsc.so; do
        [ -e "$cand" ] || continue
        d="$(dirname "$(dirname "$(dirname "$cand")")")"; a="$(basename "$(dirname "$(dirname "$cand")")")"
        err=$(LD_LIBRARY_PATH="$d/$a/lib:$AMGX_DIR/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}" \
              dlopens "${FOAM_USER_LIBBIN:-}/libpetscFoam.so")
        if [ -z "$err" ]; then PETSC_DIR="$d"; PETSC_ARCH="$a"; PETSC_OK=1; break; fi
        PETSC_WHY="$a: $(echo "$err" | tail -1 | sed 's/.*: //')"
    done
    if [ $PETSC_OK = 0 ]; then
        echo "  OF+PETSc DISABLED: no PETSc build loads libpetscFoam.so (${PETSC_WHY:-none found})"
        HAVE_PETSC=0
    fi
fi
export PETSC_DIR PETSC_ARCH
export LD_LIBRARY_PATH="$PETSC_DIR/$PETSC_ARCH/lib:$AMGX_DIR/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
if [ $HAVE_AMGX = 1 ]; then
    err=$(dlopens "${FOAM_USER_LIBBIN:-}/libamgxFoam.so")
    [ -n "$err" ] && { echo "  OF+AMGX DISABLED: $(echo "$err" | tail -1 | sed 's/.*: //') (set AMGX_DIR=)"; HAVE_AMGX=0; }
fi

# The Krylov method for the offload arms follows the MATRIX, not the habit. transonic yes => p carries
# fvm::div(phid, p) and is asymmetric => BiCGStab. transonic no => symmetric => CG.
if [ "$TRANSONIC" = yes ]; then
    PETSC_KSP=bcgs;  AMGX_MAIN=PBICGSTAB;  MATSYM="asymmetric (transonic: pEqn has div(phid,p))"
else
    PETSC_KSP=cg;    AMGX_MAIN=PCG;        MATSYM="symmetric (subsonic pEqn)"
fi

echo "rhoSimpleFoam GPU shootout | squareBend | iters=$ITERS | sizes=$SIZES | transonic=$TRANSONIC | massflow=${MASSFLOW:-tutorial}"
echo "  p matrix: $MATSYM -> AMGX main=$AMGX_MAIN, PETSc ksp_type=$PETSC_KSP pc_type=$PETSC_PC"
echo "  brae=$([ $HAVE_BRAE = 1 ] && echo "$BRAE" || echo none) | OF cores=$CORES | SPUMA=$([ $HAVE_SPUMA = 1 ] && echo "$SPUMA_BIN" || echo no)"
echo "  AMGX=$([ $HAVE_AMGX = 1 ] && echo "$AMGX_DIR" || echo no) | PETSc=$([ $HAVE_PETSC = 1 ] && echo "$PETSC_DIR/$PETSC_ARCH" || echo no)"
echo "  whole-loop-on-GPU columns: brae, SPUMA.  pressure-offload-only columns: OF+AMGX, OF+PETSc."

# `simple` needs an (nx ny nz) that multiplies out to CORES exactly; decomposePar fatals otherwise. The
# 2x2 cross-section suits squareBend's duct, so nx is the free factor and CORES has to be a multiple of 4.
NX=$((CORES / 4))
if [ "$((NX * 4))" != "$CORES" ]; then
    echo "ERROR: CORES=$CORES cannot be laid out as (nx 2 2) for the fixed simple decomposition."
    echo "       Use a multiple of 4 (20, 24, 32, 48...). scotch is NOT substituted here: half of its"
    echo "       partitions diverge on this case, and a reference number must come from a run that"
    echo "       finished -- see the note at the OpenFOAM arm."
    exit 1
fi

mkgrid(){   # mkgrid <scale> <dir>
  local M="$1" d="$2"; rm -rf "$d"; mkdir -p "$d"
  cp -r "$TUT/constant" "$TUT/system" "$d"/; cp -r "$TUT/0.orig" "$d/0"
  python3 - "$d" "$M" "$ITERS" "$TRANSONIC" "$MASSFLOW" <<'PY'
import re, sys
d, M, iters, transonic, massflow = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
if massflow:
    f = d + '/0/U'; s = open(f).read()
    s, k = re.subn(r'\bmassFlowRate\s+constant\s+[-0-9.eE]+\s*;', 'massFlowRate constant %s;' % massflow, s)
    assert k == 1, 'expected one massFlowRate constant entry in 0/U'
    open(f, 'w').write(s)
f = d + '/system/blockMeshDict'; s = open(f).read()
s = re.sub(r'\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)\s*simpleGrading',
           lambda m: '(%d %d %d) simpleGrading' % tuple(max(1, int(round(int(m[k])*M))) for k in (1, 2, 3)), s)
open(f, 'w').write(s)
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'functions\s*\{.*', '', s, flags=re.S)      # the sampling FOs need surfaces Allrun.pre builds
s = re.sub(r'\bendTime\s+[^;]*;',       'endTime %s;' % iters, s)
s = re.sub(r'\bwriteInterval\s+[^;]*;', 'writeInterval %s;' % iters, s)
open(c, 'w').write(s + '\n')
f = d + '/system/fvSolution'; s = open(f).read()
s = re.sub(r'residualControl\s*\{[^{}]*\}', 'residualControl { }', s)   # exactly ITERS in every arm
if transonic == 'no':
    s, k = re.subn(r'\btransonic\s+yes\s*;', 'transonic no;', s)
    assert k == 1, 'expected one `transonic yes;` to flip'
open(f, 'w').write(s)
PY
  ( cd "$d" && blockMesh > log.blockMesh 2>&1 ) || { echo "ERROR: blockMesh failed in $d"; tail -5 "$d/log.blockMesh"; exit 1; }
}

wall(){ local a b; a=$(date +%s.%N); eval "$1" > /dev/null 2>&1; b=$(date +%s.%N); echo "$b - $a" | bc; }
# An arm's number is its time ONLY if its own log shows ITERS completed iterations. Anything else is
# reported as the count, so a run that aborted can never masquerade as a fast one -- see note 2.
score(){   # score <seconds> <log> -> "ms/iter" or "n/ITERS it" or "-"
  local t="$1" log="$2" n
  [ -f "$log" ] || { echo "-"; return; }
  n=$(grep -c '^Time = ' "$log" 2> /dev/null || echo 0)
  if [ "${n:-0}" -ge "$ITERS" ]; then printf "%.1f" "$(echo "scale=4; $t/$ITERS*1000" | bc)"
  else echo "${n:-0}/${ITERS}it"; fi
}
rat(){   # rat <brae ms> <other ms> -> "x" or "-"
  case "${1:-}${2:-}" in *it*|*-*|"") echo "-"; return;; esac
  [ -z "${1:-}" ] || [ -z "${2:-}" ] && { echo "-"; return; }
  printf "%.1f" "$(echo "scale=3; $2/$1" | bc)"
}

mkdir -p "$WORK"; RES="$WORK/results.csv"
echo "nCells,brae_ms,ofcpu_ms,spuma_ms,amgx_ms,petsc_ms" > "$RES"
printf "\n%10s %11s %11s %11s %11s %11s\n" "cells" "brae" "OF-${CORES}c" "SPUMA" "OF+AMGX" "OF+PETSc"
printf   "%10s %11s %11s %11s %11s %11s\n" "-----" "----" "------" "-----" "-------" "--------"
RATLINES=""
for M in $SIZES; do
  SRC="$WORK/mesh_$M"; mkgrid "$M" "$SRC"
  NC=$(grep -aoE 'nCells:?[[:space:]]*[0-9]+' "$SRC/constant/polyMesh/owner" 2>/dev/null | grep -oE '[0-9]+' | head -1)
  [ -n "$NC" ] || NC=$(grep -iE 'nCells' "$SRC/log.blockMesh" 2>/dev/null | grep -oE '[0-9]+' | tail -1)
  sB="-"; sO="-"; sS="-"; sA="-"; sP="-"

  # ---- brae: the OF-mirror, CUDA arm ----
  if [ $HAVE_BRAE = 1 ]; then
    BW="$WORK/brae_$M"; rm -rf "$BW"; cp -r "$SRC" "$BW"
    t=$(wall "BRAE_RHOSIMPLEFOAM_MIRROR=cuda '$BRAE' -case '$BW' > '$BW/log.run' 2>&1")
    sB=$(score "$t" "$BW/log.run")
  fi

  # ---- OpenFOAM, N cores ----
  # `simple` and not `scotch`: scotch re-partitions differently on every decomposePar and about half of
  # those partitions DIVERGE on this case (measured 2026-09-08: 2 of 4 scotch runs aborted in the
  # thermo Newton solve at iteration 4, 3 of 3 `simple` decompositions ran to 100). A reference number
  # has to come from a run that finished, so the decomposition is fixed.
  OW="$WORK/of_$M"; rm -rf "$OW"; cp -r "$SRC" "$OW"
  printf 'FoamFile{version 2.0;format ascii;class dictionary;object decomposeParDict;}\nnumberOfSubdomains %d;method simple;coeffs{n (%d 2 2);}\n' \
         "$CORES" "$NX" > "$OW/system/decomposeParDict"
  ( cd "$OW" && decomposePar -force > log.decomposePar 2>&1 )                    # EXCLUDED
  t=$(wall "( cd '$OW' && mpirun --allow-run-as-root -np $CORES rhoSimpleFoam -parallel > log.run 2>&1 )")
  sO=$(score "$t" "$OW/log.run")

  # ---- SPUMA: whole loop on the device, its own binary and its own environment ----
  if [ $HAVE_SPUMA = 1 ]; then
    SW="$WORK/spuma_$M"; rm -rf "$SW"; cp -r "$SRC" "$SW"
    # SPUMA's GPU smoothers, the config its own paper benchmarks with. The default dummyMemoryPool is a
    # perf-killer on a GPU, so the fixed pool is not optional (setup_spuma.sh, trap 4).
    foamDictionary -entry 'solvers/p/smoother' -set twoStageGaussSeidel "$SW/system/fvSolution" > /dev/null 2>&1
    foamDictionary -entry 'solvers/"(U|e|k|epsilon)"/smoother' -set twoStageSymGaussSeidel "$SW/system/fvSolution" > /dev/null 2>&1
    t=$(wall "( export have_cuda=true NVARCH=$NVARCH FOAM_SIGFPE=false; set +u; source '$SP_SRC/etc/bashrc' > /dev/null 2>&1; set -u; cd '$SW' && '$SPUMA_BIN' -pool fixedSizeMemoryPool -poolSize $SPUMA_POOL > log.run 2>&1 )")
    sS=$(score "$t" "$SW/log.run")
  fi

  # ---- OF + AMGX: the PRESSURE solve only ----
  if [ $HAVE_AMGX = 1 ]; then
    AW="$WORK/amgx_$M"; rm -rf "$AW"; cp -r "$SRC" "$AW"
    # amgxFoam's built-in default config is solver(main)=PCG, which is undefined on this case's
    # asymmetric p. The config file below names the right outer method for the matrix (see the header),
    # keeping the same aggregation AMG preconditioner so only the Krylov changes with `transonic`.
    if [ -n "$AMGX_CFG" ]; then cp "$AMGX_CFG" "$AW/amgx_p.json"; else
    cat > "$AW/amgx_p.json" <<JSON
{
    "config_version": 2,
    "solver": {
        "solver": "$AMGX_MAIN",
        "max_iters": 1000,
        "tolerance": 0.1,
        "norm": "L2",
        "convergence": "RELATIVE_INI",
        "monitor_residual": 1,
        "preconditioner": {
            "solver": "AMG",
            "algorithm": "AGGREGATION",
            "selector": "SIZE_2",
            "smoother": "BLOCK_JACOBI",
            "presweeps": 2,
            "postsweeps": 2,
            "relaxation_factor": 0.75,
            "coarsest_sweeps": 2,
            "max_iters": 1,
            "cycle": "V",
            "max_levels": 50,
            "min_coarse_rows": 2
        }
    }
}
JSON
    fi
    foamDictionary -entry solvers/p/solver -set amgx "$AW/system/fvSolution" > /dev/null 2>&1
    foamDictionary -entry libs         -set '("libamgxFoam.so")' "$AW/system/controlDict" > /dev/null 2>&1
    foamDictionary -entry amgxConfig   -set "\"$AW/amgx_p.json\"" "$AW/system/controlDict" > /dev/null 2>&1
    t=$(wall "( cd '$AW' && rhoSimpleFoam > log.run 2>&1 )")
    sA=$(score "$t" "$AW/log.run")
  fi

  # ---- OF + PETSc: the PRESSURE solve only ----
  if [ $HAVE_PETSC = 1 ]; then
    PW="$WORK/petsc_$M"; rm -rf "$PW"; cp -r "$SRC" "$PW"
    foamDictionary -entry solvers/p -set \
      "{solver petsc; petsc{options{ksp_type $PETSC_KSP; mat_type aijcusparse; pc_type $PETSC_PC;}} tolerance 1e-08; relTol 0.1;}" \
      "$PW/system/fvSolution" > /dev/null 2>&1
    foamDictionary -entry libs -set '("libpetscFoam.so")' "$PW/system/controlDict" > /dev/null 2>&1
    t=$(wall "( cd '$PW' && rhoSimpleFoam > log.run 2>&1 )")
    sP=$(score "$t" "$PW/log.run")
  fi

  printf "%10s %11s %11s %11s %11s %11s\n" "$NC" "$sB" "$sO" "$sS" "$sA" "$sP"
  echo "$NC,$sB,$sO,$sS,$sA,$sP" >> "$RES"
  RATLINES="$RATLINES$(printf "\n%10s %11s %11s %11s %11s %11s" "$NC" "1.0" "$(rat "$sB" "$sO")" "$(rat "$sB" "$sS")" "$(rat "$sB" "$sA")" "$(rat "$sB" "$sP")")"
  [ "$KEEP" = 1 ] || rm -rf "$SRC" "$WORK"/{brae,of,spuma,amgx,petsc}_$M 2> /dev/null
done

echo
echo "ms per SIMPLE iteration, lower is better. 'n/${ITERS}it' means that arm ABORTED after n"
echo "iterations and has no time -- it is not a fast run (see note 2 in the header)."
echo
echo "how many times faster brae is (x):"
printf "%10s %11s %11s %11s %11s %11s\n" "cells" "brae" "OF-${CORES}c" "SPUMA" "OF+AMGX" "OF+PETSc"
printf "%b\n" "$RATLINES"
echo
echo "CSV -> $RES"
[ "$KEEP" = 1 ] && echo "scratch kept under $WORK"
exit 0
