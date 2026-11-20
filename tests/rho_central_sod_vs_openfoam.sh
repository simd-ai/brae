#!/usr/bin/env bash
# brae_rhoCentralFoam (first stage) against OpenFOAM v2412 rhoCentralFoam and against the exact Riemann solution, on
# the Sod shock tube OpenFOAM ships (tutorials/compressible/rhoCentralFoam/shockTube: x in [-5, 5] m, 100 cells,
# diaphragm at 0, p 1e5/1e4 Pa, T 348.432/278.746 K, so rho 1.0/0.125 and gamma 1.4002; inviscid, perfectGas, hConst).
#
# CASE   the tutorial with ONE change both codes see: `adjustTimeStep no` and deltaT 1e-5 (Courant number <= 0.035),
#        because this stage runs a fixed deltaT and refuses adjustTimeStep by name; plus writePrecision 17 so the
#        comparison is not floored by the written digits. 700 steps to the shipped endTime 0.007, written every 0.001.
#        A refined copy: 1000 cells, deltaT 1e-6, 7000 steps.
# ARMS   (each brae against OpenFOAM serial on the same case, U p T rho, internal and patch values, every write time)
#   K    the CUDA arm, fluxScheme Kurganov (the tutorial's)
#   H    the host reference (BRAE_RHOCENTRALFOAM_ARM=cpu), a transcription of rhoCentralFoam.C
#   T    the CUDA arm, fluxScheme Tadmor
#   F    the CUDA arm on the 1000-cell tube
# PREMISE  OpenFOAM's Kurganov and Tadmor runs at least 100 x BOUND apart: the case tells the two flux schemes apart
# CONTROLS deliberately wrong numerics in the CUDA arm (BRAE_CONTROL_RCF), each asserted red against OpenFOAM's K at
#          more than 100 x BOUND and 1000 x K's distance: `upwind` (limiter 0, first order), `noPressureJump` (phiEp
#          without aSf*(p_pos - p_neg)); and arm T held against OpenFOAM's Kurganov (the other flux scheme), red.
# EXACT  (tests/rho_central_sod.py, Toro ch. 4) at t = 0.007, on K and F: L1 of rho, p, u the same as OpenFOAM's to 1e-9
#        relative (the same fields); the shock (where rho crosses (rho*R + rho_R)/2) and the contact (where it crosses
#        (rho*L + rho*R)/2) within 0.5 cells of the exact positions; F's L1(rho) below a quarter of K's (convergence).
#        These are engineering checks of the scheme OpenFOAM and brae share, not of brae against OpenFOAM.
# CONSERVATION  from brae's own `diag balance` line: the change of the total V*rho and V*rhoE plus the time integral of
#        the boundary outflow, relative to the start total, within 1e-12; the momentum balance (the end walls push)
#        within 1e-12 of the boundary impulse. POSITIVITY: min rho, p, T > 0 at every written time of every brae arm.
# BOUND  8e-14 for the 100-cell arms (one decade above the first gate run's worst, GB10 2026-11-16: K 7.7e-15,
#        H 7.8e-15, T 7.1e-15, all on U); 2e-13 for F (7000 steps; first run 1.9e-14). Controls on that run: upwind
#        9.8e-02, noPressureJump 6.2e-03, T against Kurganov 8.8e-03. The exact-solution figures on that run: shock 0.21
#        cells, contact 0.24 cells off at 100 cells; 0.17 and 0.03 at 1000; L1(rho) 6.3e-03 -> 7.5e-04.
# NOT CLAIMED  viscous flow, adjustTimeStep, boundary types other than zeroGradient and empty (fixedValue is
#        implemented, not gated here: the waves do not reach the ends by t = 0.007), 2-D/3-D waves, any other
#        tutorial.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BRAE_RCF_BIN:-$ROOT/build/brae_rhoCentralFoam}"
CHECK="$ROOT/tests/check_fields.py"
SOD="$ROOT/tests/rho_central_sod.py"
OFBASHRC=${OF_BASHRC:-${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}}
TUT="${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}/compressible/rhoCentralFoam/shockTube"
[ -x "$BIN" ] || { echo "SKIP: no brae_rhoCentralFoam at $BIN"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: no OpenFOAM v2412 at $OFBASHRC"; exit 77; }
[ -d "$TUT/0.orig" ] || { echo "SKIP: no shockTube tutorial at $TUT"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
python3 -c 'import numpy' 2> /dev/null || { echo "SKIP: no numpy"; exit 77; }
W=${BRAE_GATE_OUT:-$(mktemp -d /tmp/brae-rcf-sod.XXXXXX)}
mkdir -p "$W"
export PYTHONDONTWRITEBYTECODE=1
BOUND=${BRAE_RCF_SOD_BOUND:-8e-14}
BOUND_F=${BRAE_RCF_SOD_BOUND_FINE:-2e-13}
TIMES="0.001 0.002 0.003 0.004 0.005 0.006 0.007"
fails=0
passes=0
fail()
{
    echo "FAIL $*"
    fails=$((fails + 1))
}
ok()
{
    passes=$((passes + 1))
    echo "ok   $*"
}
of()
{
    bash -c "set +u; source '$OFBASHRC' > /dev/null 2>&1; set -e; $*"
}
le() { python3 -c "import sys; sys.exit(0 if float('${1:-nan}') <= float('$2') else 1)" 2> /dev/null; }
ge() { python3 -c "import sys; sys.exit(0 if float('${1:-nan}') >= float('$2') else 1)" 2> /dev/null; }

# stage <dir> <cells> <deltaT>: the tutorial, restored, meshed and set by OpenFOAM's own utilities
stage()
{
    rm -rf "$1"
    mkdir -p "$1"
    cp -r "$TUT/system" "$TUT/constant" "$TUT/0.orig" "$1/"
    cp -r "$TUT/0.orig" "$1/0"
    sed -i -E "s/\(100 1 1\)/($2 1 1)/" "$1/system/blockMeshDict"
    grep -q "($2 1 1)" "$1/system/blockMeshDict" || { fail "stage: the tutorial's blockMeshDict no longer reads (100 1 1)"; return 1; }
    sed -i -E "s/^(adjustTimeStep\s+)\S+;/\1no;/; s/^(deltaT\s+)\S+;/\1$3;/; s/^(writePrecision\s+)\S+;/\117;/" \
        "$1/system/controlDict"
    grep -qE '^adjustTimeStep\s+no;' "$1/system/controlDict" || { fail "stage: adjustTimeStep not found in controlDict"; return 1; }
    of "cd '$1' && blockMesh > log.blockMesh 2>&1 && setFields > log.setFields 2>&1" < /dev/null \
        || { fail "stage: blockMesh/setFields in $1"; return 1; }
}
copy()   # copy <from> <to> [fluxScheme]
{
    rm -rf "$2"
    cp -r "$1" "$2"
    [ -z "${3:-}" ] || sed -i -E "s/^(fluxScheme\s+)\S+;/\1$3;/" "$2/system/fvSchemes"
}
ofrun()
{
    of "cd '$1' && rhoCentralFoam > log 2>&1" < /dev/null || fail "OpenFOAM rhoCentralFoam in $(basename "$1")"
}
braerun()   # braerun <dir> [env ...]
{
    local d=$1
    shift
    (cd "$d" && env "$@" timeout 600 "$BIN" -case . > log 2> err < /dev/null; echo $? > rc)
    [ "$(cat "$d/rc")" = 0 ] || { fail "$(basename "$d") exited $(cat "$d/rc"): $(head -c 300 "$d/err")"; return 1; }
}
dist()   # dist <time dir> <reference time dir>: the worst relative L2 over U p T rho
{
    local out=$1.$(basename "$(dirname "$2")").cmp
    python3 "$CHECK" fields "$1" "$2" 1e300 U p T rho > "$out" 2>&1
    sed -n 's/^worst \([^ ]*\).*/\1/p' "$out"
}

stage "$W/base" 100 1e-05 || { echo "rho_central_sod_vs_openfoam: $passes passed, $fails failed"; exit 1; }
stage "$W/fine" 1000 1e-06 || { echo "rho_central_sod_vs_openfoam: $passes passed, $fails failed"; exit 1; }
copy "$W/base" "$W/ofK"
copy "$W/base" "$W/ofT" Tadmor
copy "$W/fine" "$W/ofF"
for d in ofK ofT ofF; do ofrun "$W/$d"; done
copy "$W/base" "$W/bK"
copy "$W/base" "$W/bH"
copy "$W/base" "$W/bT" Tadmor
copy "$W/fine" "$W/bF"
copy "$W/base" "$W/cU"
copy "$W/base" "$W/cP"
braerun "$W/bK"
braerun "$W/bH" BRAE_RHOCENTRALFOAM_ARM=cpu
braerun "$W/bT"
braerun "$W/bF"
braerun "$W/cU" BRAE_CONTROL_RCF=upwind
braerun "$W/cP" BRAE_CONTROL_RCF=noPressureJump

# premise
p=$(dist "$W/ofT/0.007" "$W/ofK/0.007")
if ge "$p" "$(python3 -c "print(100*$BOUND)")"
then
    ok "premise: OpenFOAM Tadmor against OpenFOAM Kurganov at 0.007: $p: the case tells the flux schemes apart"
else
    fail "premise: OpenFOAM's two flux schemes ${p:-none} apart, under 100 x $BOUND"
fi

# arms, every write time
arm()   # arm <name> <brae dir> <OF dir> <bound> <times>
{
    local worst=0 where="" v
    for t in $5
    do
        v=$(dist "$2/$t" "$3/$t")
        [ -n "$v" ] || { fail "$1: no comparison at $t ($(tail -n 2 "$2/$t.$(basename "$3").cmp" 2> /dev/null))"; return; }
        if python3 -c "import sys; sys.exit(0 if float('$v') > float('$worst') else 1)"; then worst=$v; where=$t; fi
    done
    if le "$worst" "$4"
    then
        ok "$1 against OpenFOAM at $(echo $5 | wc -w) write times: worst $worst (t = $where, $(sed -n 's/^worst [^ ]* //p' "$2/$where.$(basename "$3").cmp")) within $4"
    else
        fail "$1 against OpenFOAM: worst $worst at t = $where above $4"
        sed 's/^/     /' "$2/$where.$(basename "$3").cmp"
    fi
}
arm "K (CUDA, Kurganov)" "$W/bK" "$W/ofK" "$BOUND" "$TIMES"
arm "H (host reference, Kurganov)" "$W/bH" "$W/ofK" "$BOUND" "$TIMES"
arm "T (CUDA, Tadmor)" "$W/bT" "$W/ofT" "$BOUND" "$TIMES"
arm "F (CUDA, 1000 cells, 7000 steps)" "$W/bF" "$W/ofF" "$BOUND_F" "$TIMES"

# controls
k=$(dist "$W/bK/0.007" "$W/ofK/0.007")
for c in "upwind:$W/cU:$W/ofK" "noPressureJump:$W/cP:$W/ofK" "Tadmor against OpenFOAM's Kurganov:$W/bT:$W/ofK"
do
    IFS=: read -r name d ref <<< "$c"
    v=$(dist "$d/0.007" "$ref/0.007")
    if python3 -c "import sys; v = float('${v:-0}'); k = float('${k:-1e300}'); sys.exit(0 if v > 100*$BOUND and v >= 1000*k else 1)"
    then
        ok "control ($name) red: $v ($(sed -n 's/^worst [^ ]* //p' "$d/0.007.$(basename "$ref").cmp")), K $k"
    else
        fail "control ($name): ${v:-none}, not above 100 x $BOUND and 1000 x K's ${k:-none}"
    fi
done

# the exact Riemann solution
sod()   # sod <dir>: the analysis lines into <dir>/sod.txt
{
    python3 "$SOD" analyse "$1" 0.007 > "$1/sod.txt" 2>&1 || fail "rho_central_sod.py on $(basename "$1"): $(tail -n 1 "$1/sod.txt")"
}
for d in ofK bK ofF bF cU; do sod "$W/$d"; done
field() { awk -v k="$2" -v i="$3" '$1 == k { print $(i + 1) }' "$1/sod.txt"; }
for pair in "K:bK:ofK:100 cells" "F:bF:ofF:1000 cells"
do
    IFS=: read -r name b o what <<< "$pair"
    lb=$(field "$W/$b" L1 1); lo=$(field "$W/$o" L1 1)
    if python3 -c "
import sys
b = [float(x) for x in '$(awk '$1 == "L1" { print $2, $3, $4 }' "$W/$b/sod.txt")'.split()]
o = [float(x) for x in '$(awk '$1 == "L1" { print $2, $3, $4 }' "$W/$o/sod.txt")'.split()]
sys.exit(0 if len(b) == 3 and all(abs(x - y) <= 1e-9*abs(y) for x, y in zip(b, o)) else 1)"
    then
        ok "$name exact solution, $what: L1 rho p u $(awk '$1 == "L1" { print $2, $3, $4 }' "$W/$b/sod.txt"), OpenFOAM's the same to 1e-9"
    else
        fail "$name exact solution, $what: L1 $(awk '$1 == "L1" { print $2, $3, $4 }' "$W/$b/sod.txt") against OpenFOAM's $(awk '$1 == "L1" { print $2, $3, $4 }' "$W/$o/sod.txt")"
    fi
    for wave in shock contact
    do
        read -r xb xe h <<< "$(awk -v k=$wave '$1 == k { print $2, $3, $4 }' "$W/$b/sod.txt")"
        if python3 -c "import sys; sys.exit(0 if abs(float('$xb') - float('$xe')) <= 0.5*float('$h') else 1)" 2> /dev/null
        then
            ok "$name $wave at $xb m, exact $xe m: $(python3 -c "print('%.2f' % (abs(float('$xb') - float('$xe'))/float('$h')))") cells off ($what)"
        else
            fail "$name $wave at ${xb:-none} m, exact ${xe:-none} m: more than half a cell off ($what)"
        fi
    done
done
lK=$(field "$W/bK" L1 1); lF=$(field "$W/bF" L1 1); lU=$(field "$W/cU" L1 1)
if python3 -c "import sys; sys.exit(0 if float('$lF') < 0.25*float('$lK') else 1)" 2> /dev/null
then
    ok "convergence: L1(rho) $lK at 100 cells -> $lF at 1000"
else
    fail "convergence: L1(rho) ${lK:-none} at 100 cells -> ${lF:-none} at 1000, not below a quarter"
fi
echo "     (the upwind control's L1(rho) is ${lU:-none} against K's $lK: the exact-solution check sees it too)"

# conservation and positivity, from brae's own diagnostics
for d in bK bH bT bF
do
    bal=$(sed -n 's/.*relative to the start totals: mass \([^ ]*\) energy \([^)]*\)).*/\1 \2/p' "$W/$d/log")
    mom=$(sed -n 's/^diag balance .* momentum \([^ ]*\) .*/\1/p' "$W/$d/log")
    imp=$(sed -n 's/^diag boundary-outflow .* momentum \([^ ]*\) .*/\1/p' "$W/$d/log")
    if python3 -c "
import sys
m, e = (float(x) for x in '$bal'.split())
r = abs(float('$mom'))/abs(float('$imp'))
print('     %s: mass %.1e, energy %.1e relative; momentum %.1e of the end-wall impulse %.6g' % ('$d', m, e, r, abs(float('$imp'))))
sys.exit(0 if m <= 1e-12 and e <= 1e-12 and r <= 1e-12 else 1)"
    then
        ok "$d conserves mass and energy, and balances momentum against the boundary, to 1e-12"
    else
        fail "$d conservation: '$bal' momentum ${mom:-none} impulse ${imp:-none}"
    fi
    neg=0
    for t in $TIMES
    do
        python3 "$SOD" analyse "$W/$d" "$t" 2> /dev/null | awk '$1 == "min" { exit !($2 > 0 && $3 > 0 && $4 > 0) }' || neg=1
    done
    if [ "$neg" = 0 ]; then ok "$d positivity: rho, p, T > 0 at every written time"; else fail "$d: a non-positive rho, p or T"; fi
done

echo "     runtime: OpenFOAM $(grep -h ExecutionTime "$W/ofK/log" | tail -n 1 | cut -c1-40) (100 cells), $(grep -h ExecutionTime "$W/ofF/log" | tail -n 1 | cut -c1-40) (1000)"
for d in bK bH bF; do echo "     brae $d: $(sed -n 's/^diag steps //p' "$W/$d/log") $(sed -n 's/^diag device-memory/device memory/p' "$W/$d/log")"; done
echo "rho_central_sod_vs_openfoam: $passes passed, $fails failed (work in $W)"
[ "$fails" = 0 ] && [ "$passes" -ge 20 ]
