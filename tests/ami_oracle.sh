#!/usr/bin/env bash
# Distributed cyclicAMI regression: run the SAME case through the single-GPU device path (brae -case) and the
# distributed path (mpirun -np N brae -case -parallel), then compare the written fields. This is the comparison
# that CATCHES real AMI-coupling bugs -- both my fixes (bounded-div missing the AMI-face flux; deferred-rotation
# leaking into UEqn.H()) lived in shared distributed code, so a np2-vs-np1 oracle passes with both wrong; only
# distributed-vs-single-GPU exposes them.
#
# Args: <brae_binary> <source_case_dir> <work_root> <np> <uTol> <pTol> <compare_py>
# Both solvers are forced to FULL inner convergence (relTol 0) so a short fixed run is solver-algorithm-independent
# and thus bit-exact at np=1 (single-GPU smoothSolver vs distributed BiCGStab otherwise stop at different residuals).
set -u
BRAE="$1"; SRC="$2"; WORK="$3"; NP="$4"; UTOL="$5"; PTOL="$6"; CMP="$7"
END=400   # generous ceiling: tc_wedge_ami converges (residualControl) in ~148; a BUGGY AMI limit-cycles instead and
          # hits this endTime with a very different field -> the comparison fails (that is the regression signal).

prep() {   # copy the case, force FULL inner convergence (relTol 0) so single-GPU + distributed reach the SAME fixed
           # point, then compare the CONVERGED write (the two paths use different p solvers so they do NOT track
           # bit-exactly mid-run -- only the converged solution matches, to ~roundoff at np=1).
    local dst="$1"
    rm -rf "$dst"; cp -r "$SRC" "$dst"
    rm -rf "$dst"/processor* "$dst"/[1-9]* "$dst"/0.0* 2>/dev/null
    sed -i "s/relTol[[:space:]]*[0-9.eE+-]*;/relTol 0;/g; s/tolerance[[:space:]]*[0-9.eE+-]*;/tolerance 1e-10;/g" "$dst/system/fvSolution"
    sed -i "s/endTime[[:space:]].*/endTime $END;/; s/writeInterval[[:space:]].*/writeInterval $END;/" "$dst/system/controlDict"
}

last_time() {   # newest written time dir that is not 0/ (sort by the numeric basename)
    ls -d "$1"/[0-9]*/ 2>/dev/null | grep -vE "/0/$" \
        | while read -r d; do printf '%s %s\n' "$(basename "$d")" "$d"; done \
        | sort -n | tail -1 | cut -d' ' -f2-
}

mkdir -p "$WORK"
SGL="$WORK/single"; DST="$WORK/dist_np$NP"
prep "$SGL"; prep "$DST"

echo "== single-GPU =="
"$BRAE" -case "$SGL" > "$WORK/single.log" 2>&1 || { echo "single-GPU run FAILED"; tail -5 "$WORK/single.log"; exit 2; }
echo "== distributed np=$NP =="
mpirun -np "$NP" "$BRAE" -case "$DST" -parallel > "$WORK/dist.log" 2>&1 || { echo "distributed run FAILED"; tail -5 "$WORK/dist.log"; exit 2; }

LS="$(last_time "$SGL")"; LD="$(last_time "$DST")"
[ -n "$LS" ] && [ -n "$LD" ] || { echo "missing output dir (single=$LS dist=$LD)"; exit 2; }
echo "comparing single-GPU $(basename "$LS") vs distributed-np$NP $(basename "$LD")"
python3 "$CMP" "${LS%/}" "${LD%/}" "$UTOL" "$PTOL"
