#!/usr/bin/env bash
# brae_rhoCentralFoam against OpenFOAM v2412 rhoCentralFoam on inletOutlet and purgeWrite (milestone D): the forwardStep
# tutorial AS SHIPPED, and variants made by tests/rho_central_cases.py for what it does not exercise.
#
# WHAT forwardStep EXERCISES. Mach 3 over a step (Woodward-Colella's forward-facing step): U and T `inletOutlet` on the
#   outlet, purgeWrite 10 with adjustable writes every 0.1 to 4, adjustTimeStep (maxCo 0.2), symmetryPlane top and
#   bottom, slip on the step. Its outlet flux leaves on every face at every step (brae's `diag inletOutlet`: 0 inflow
#   face-steps of 428,416), so the switch's inflow branch and a change of branch need the variants.
# CASES (blockMesh, and setFields for the shockTube ones, on the tutorial's files; nothing else changed unless named)
#   fs        forwardStep as shipped (writePrecision 6: its files and log carry 6 digits)
#   fs17      the same at writePrecision 17 and purgeWrite 0, so every write time is compared (40 of them)
#   fs17e     fs17 written EVERY STEP for its first 20 steps (writeControl timeStep, interval 1; endTime 0.0158)
#   fsIn17    fs17 with the INLET's U, p and T inletOutlet on the shipped values: the inflow branch on every face of
#             every step, on all three fields (cases.py inletIO)
#   rev17     shockTube (100 cells) with both ends flowing OUT at 50 m/s and U, T inletOutlet there (inletValue (50 0 0),
#             348.4, no value), to t = 0.02: the expansion turns the left end's flux inward between t = 0.013 and
#             0.014, at a resolved velocity, and the inflow then holds (cases.py sodRevSet, sodRev)
#   rsRev     rev17 RESTARTED from OpenFOAM's own t = 0.013 (its inletOutlet `value` as OpenFOAM wrote it) and written
#             every step (writeControl timeStep, 1) to 0.0145: the step the switch changes branch, step by step
#   sodP      shockTube as shipped with purgeWrite 2 and writeControl timeStep, interval 40
#   sodPr     shockTube as shipped with purgeWrite 3 and writeControl runTime, interval 0.001
# REFERENCE  OpenFOAM serial on every case; OpenFOAM's own spread (decomposed in two, simple (2 1 1), and in four,
#   scotch, the partition frozen: tests/rho_central_gate_lib.sh) on fs17, fs17e, fsIn17, rev17 and rsRev.
# ENVELOPE (tests/rho_central_gate_lib.sh, approved with milestone C, declared here before the first run): every field
#   at every write time within ten times the larger of OpenFOAM's two spreads, never below 8e-14 (6-digit arms: 1e-6);
#   each decomposition alone reported beside it. The Courant number likewise. deltaT: the same digits where
#   Time::adjustDeltaT lands it (adjustable writes), within 8e-14 where nothing does (timeStep writes).
# CHECKS beyond the envelope
#   PURGE      every time directory a run leaves -- 0 included -- the same as OpenFOAM's, and purgeWrite + 1 of them
#   PATCH      the inletOutlet patches' written values (U, T, rho) within the envelope of the same measure, every time
#   BRANCHES   premises, read off OpenFOAM's own written patch values: fs's outlet takes its cells' values (outflow),
#              fsIn17's inlet the inletValue (inflow), rev17's left end its cell's before the switch and (50 0 0) after
#   SHOCKS     forwardStep's bow-shock standoff (bottom row, and the row under the step's corner), where the shock meets
#              the top wall, and the pitot pressure on the step's face (rho_central_fields.py shock step): brae's
#              against OpenFOAM's within ten times OpenFOAM's spread of the same measure (never below 1e-6). Premise:
#              OpenFOAM's pitot within 2 % of Rayleigh's (12.061 at Mach 3) -- the resolution, not a bound on brae.
#   CONSERVATION  `diag balance` within `diag roundoff` (four ulps a cell a step); STABILITY  rho, p, T > 0, finite
#   CPU/GPU    the two arms against each other (reported); PERFORMANCE and device memory (reported)
# CONTROLS (rhoCentralState.cuh), each red by more than 100 x its bound against OpenFOAM:
#   inletOutletInverted  the switch fed -phi: fs17 (its outlet held at the free stream)
#   inletOutletNewT      T's inletOutlet evaluated on the cells' new T (Tw.evaluate() after thermo.correct()): fs17, rev17
#   laggedPhi            the switch on the step before's phi: rev17 and rsRev. NOT on forwardStep: its outlet never
#                        changes branch, so a lagged switch decides the same
#   noPurge              purgeWrite ignored (what stage 1 did in silence): fs's directories are not OpenFOAM's
# NOT CLAIMED  inletOutlet on a `phi` other than rhoCentralFoam's mass flux (refused), outletInlet and the other mixed
#   types (refused), restart from brae's own output (rsRev restarts from OpenFOAM's), viscous flow, 3-D.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BRAE_RCF_BIN:-$ROOT/build/brae_rhoCentralFoam}"
STEPS="$ROOT/tests/rho_central_steps.py"
FIELDS="$ROOT/tests/rho_central_fields.py"
CASES="$ROOT/tests/rho_central_cases.py"
OFBASHRC=${OF_BASHRC:-${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}}
TUT="${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}/compressible/rhoCentralFoam"
[ -x "$BIN" ] || { echo "SKIP: no brae_rhoCentralFoam at $BIN"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: no OpenFOAM v2412 at $OFBASHRC"; exit 77; }
[ -d "$TUT/forwardStep/0" ] && [ -d "$TUT/shockTube/0.orig" ] || { echo "SKIP: no forwardStep/shockTube at $TUT"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
python3 -c 'import numpy' 2> /dev/null || { echo "SKIP: no numpy"; exit 77; }
W=${BRAE_GATE_OUT:-$(mktemp -d /tmp/brae-rcf-forwardstep.XXXXXX)}
mkdir -p "$W"
export PYTHONDONTWRITEBYTECODE=1
BOUND=${BRAE_RCF_SOD_BOUND:-8e-14}
fails=0
passes=0
. "$ROOT/tests/rho_central_gate_lib.sh"
ctl() { sed -i -E "$2" "$1/system/controlDict"; }

# fstage <dir> [writePrecision] [variant]: forwardStep, blockMesh
fstage()
{
    local d=$1 prec=${2:-} var=${3:-}
    rm -rf "$d"
    mkdir -p "$d"
    cp -r "$TUT/forwardStep/system" "$TUT/forwardStep/constant" "$TUT/forwardStep/0" "$d/"
    if [ -n "$prec" ]
    then
        ctl "$d" "s/^(writePrecision\s+)\S+;/\1$prec;/; s/^(purgeWrite\s+)\S+;/\10;/"
        grep -qE "^writePrecision\s+$prec;" "$d/system/controlDict" && grep -qE "^purgeWrite\s+0;" "$d/system/controlDict" \
            || { fail "stage $d: writePrecision/purgeWrite not set"; return 1; }
    fi
    [ -z "$var" ] || python3 "$CASES" "$var" "$d" || { fail "stage $d: variant $var"; return 1; }
    of "cd '$d' && blockMesh > log.blockMesh 2>&1" < /dev/null || { fail "stage $d: blockMesh"; return 1; }
}
# sstage <dir> [writePrecision] [rev]: shockTube by its Allrun steps (0.orig restored, blockMesh, setFields)
sstage()
{
    local d=$1 prec=${2:-} rev=${3:-}
    rm -rf "$d"
    mkdir -p "$d"
    cp -r "$TUT/shockTube/system" "$TUT/shockTube/constant" "$TUT/shockTube/0.orig" "$d/"
    cp -r "$TUT/shockTube/0.orig" "$d/0"
    [ -z "$prec" ] || ctl "$d" "s/^(writePrecision\s+)\S+;/\1$prec;/"
    [ -z "$rev" ] || python3 "$CASES" sodRevSet "$d" || { fail "stage $d: sodRevSet"; return 1; }
    of "cd '$d' && blockMesh > log.blockMesh 2>&1 && setFields > log.setFields 2>&1" < /dev/null \
        || { fail "stage $d: blockMesh/setFields"; return 1; }
    [ -z "$rev" ] || python3 "$CASES" sodRev "$d" || { fail "stage $d: sodRev"; return 1; }
}
rcopy() { rm -rf "$2"; mkdir -p "$2"; cp -r "$1/system" "$1/constant" "$1/0.013" "$2/"; }
spread() { echo "$W/of_${1}_np2 $W/of_${1}_np4"; }

# ---- the cases and OpenFOAM ---------------------------------------------------------------------------------------
fstage "$W/fs" || exit 1
fstage "$W/fs17" 17 || exit 1
fstage "$W/fs17e" 17 || exit 1
ctl "$W/fs17e" 's/^(writeControl\s+)\S+;/\1timeStep;/; s/^(writeInterval\s+)\S+;/\11;/; s/^(endTime\s+)\S+;/\10.0158;/'
fstage "$W/fsIn17" 17 inletIO || exit 1
sstage "$W/rev17" 17 rev || exit 1
ctl "$W/rev17" 's/^(endTime\s+)\S+;/\10.02;/'
sstage "$W/sodP" || exit 1
ctl "$W/sodP" 's/^(purgeWrite\s+)\S+;/\12;/; s/^(writeControl\s+)\S+;/\1timeStep;/; s/^(writeInterval\s+)\S+;/\140;/'
sstage "$W/sodPr" || exit 1
ctl "$W/sodPr" 's/^(purgeWrite\s+)\S+;/\13;/; s/^(writeControl\s+)\S+;/\1runTime;/'
grep -qE '^purgeWrite\s+2;' "$W/sodP/system/controlDict" && grep -qE '^purgeWrite\s+3;' "$W/sodPr/system/controlDict" \
    && grep -qE '^endTime\s+0.02;' "$W/rev17/system/controlDict" && grep -qE '^endTime\s+0.0158;' "$W/fs17e/system/controlDict" \
    || fail "stage: a controlDict edit did not take"
ALL="fs fs17 fs17e fsIn17 rev17 sodP sodPr"
for c in $ALL; do copy "$W/$c" "$W/of_$c"; done
for c in $ALL; do ofrun "$W/of_$c" & done
wait
# rsRev: rev17 restarted from OpenFOAM's own t = 0.013, every step to 0.0145
rm -rf "$W/rsRev"
mkdir -p "$W/rsRev"
cp -r "$W/rev17/system" "$W/rev17/constant" "$W/of_rev17/0.013" "$W/rsRev/"
ctl "$W/rsRev" 's/^(startFrom\s+)\S+;/\1startTime;/; s/^(startTime\s+)\S+;/\10.013;/; s/^(writeControl\s+)\S+;/\1timeStep;/; s/^(writeInterval\s+)\S+;/\11;/; s/^(endTime\s+)\S+;/\10.0145;/'
[ -d "$W/rsRev/0.013/uniform" ] && grep -qE '^startTime\s+0.013;' "$W/rsRev/system/controlDict" || fail "stage rsRev"
rcopy "$W/rsRev" "$W/of_rsRev"
ofrun "$W/of_rsRev"
REF="fs17 fs17e fsIn17 rev17 rsRev"
for c in $REF; do
    if [ "$c" = rsRev ]; then rcopy "$W/rsRev" "$W/of_${c}_np2"; rcopy "$W/rsRev" "$W/of_${c}_np4"
    else copy "$W/$c" "$W/of_${c}_np2"; copy "$W/$c" "$W/of_${c}_np4"; fi
done
for c in $REF; do ofnp "$W/of_${c}_np2" 2 & done
wait
for c in $REF; do ofnp "$W/of_${c}_np4" 4 & done
wait
for c in $ALL rsRev; do grep -qx End "$W/of_$c/log" || fail "OpenFOAM did not finish $c"; done
for c in $REF; do for n in 2 4; do grep -qx End "$W/of_${c}_np$n/log" || fail "OpenFOAM in $n did not finish $c"; done; done

# ---- brae -----------------------------------------------------------------------------------------------------------
bcase() { copy "$W/$1" "$W/$2"; }
for c in $ALL; do bcase "$c" "b_$c"; braerun "$W/b_$c"; done
for c in fs fs17 fs17e rev17; do bcase "$c" "h_$c"; braerun "$W/h_$c" BRAE_RHOCENTRALFOAM_ARM=cpu; done
rcopy "$W/rsRev" "$W/b_rsRev"; braerun "$W/b_rsRev"
rcopy "$W/rsRev" "$W/h_rsRev"; braerun "$W/h_rsRev" BRAE_RHOCENTRALFOAM_ARM=cpu
bcase fs17 cI_fs17;   ctlrun "$W/cI_fs17" BRAE_CONTROL_RCF=inletOutletInverted
bcase fs17 cT_fs17;   ctlrun "$W/cT_fs17" BRAE_CONTROL_RCF=inletOutletNewT
bcase rev17 cT_rev17; ctlrun "$W/cT_rev17" BRAE_CONTROL_RCF=inletOutletNewT
bcase rev17 cL_rev17; ctlrun "$W/cL_rev17" BRAE_CONTROL_RCF=laggedPhi
rcopy "$W/rsRev" "$W/cL_rsRev"; ctlrun "$W/cL_rsRev" BRAE_CONTROL_RCF=laggedPhi
bcase fs cP_fs;       braerun "$W/cP_fs" BRAE_CONTROL_RCF=noPurge

# ---- premises: what the cases exercise, read off OpenFOAM's own output -------------------------------------------------
pv() { python3 "$FIELDS" patchvalue "$1" "$2" "$3" "$4" 2> /dev/null | cut -d' ' -f2-; }
cellU() { python3 -c "
import sys; sys.path.insert(0, '$ROOT/tests'); import rho_central_fields as f
print('%.17g' % f.internal('$1/U')[$2][0])" 2> /dev/null; }
# fs: on its outlet every written U is its face cell's (zeroGradient: the outflow branch) -- the largest |U_b - U_P|
outflow=$(python3 -c "
import sys, numpy as np; sys.path.insert(0, '$ROOT/tests'); import rho_central_fields as f
case = '$W/of_fs17'
import re
s = open(case + '/constant/polyMesh/boundary').read()
m = re.search(r'outlet\s*\{[^}]*nFaces\s+(\d+);\s*startFace\s+(\d+);', s)
n, st = int(m.group(1)), int(m.group(2))
own = f._list(case + '/constant/polyMesh/owner', int)
o = np.array(own[1][own[2]:own[1].index(')', own[2])].split(), dtype=int)[st:st + n]
w = 0.0
for t in '$(times "$W/of_fs17")'.split():
    w = max(w, float(np.abs(f.patch_value(f'{case}/{t}/U', 'outlet') - f.internal(f'{case}/{t}/U')[o]).max()))
print('%.3e' % w)" 2> /dev/null)
if [ "${outflow:-x}" = "0.000e+00" ]
then
    ok "premise: OpenFOAM's fs17 outlet holds its cells' U at all $(times "$W/of_fs17" | wc -w) write times (the outflow branch: inletOutlet's switch never takes inflow on forwardStep)"
else
    fail "premise: OpenFOAM's fs17 outlet U differs from its cells' by ${outflow:-?}"
fi
inl=$(pv "$W/of_fsIn17/4" inlet U 0)
if [ "$inl" = "3 0 0" ]; then ok "premise: OpenFOAM's fsIn17 inlet holds the inletValue (3 0 0) at t = 4 (the inflow branch)"; else fail "premise: fsIn17 inlet U ${inl:-?}"; fi
# rev17: the left end is face 1 of `sides` (face 0 the right end), its cell 0
l12=$(pv "$W/of_rev17/0.012" sides U 1 | cut -d' ' -f1); c12=$(cellU "$W/of_rev17/0.012" 0)
l14=$(pv "$W/of_rev17/0.014" sides U 1 | cut -d' ' -f1); c14=$(cellU "$W/of_rev17/0.014" 0)
if [ -n "$l12" ] && [ "$l12" = "$c12" ] && python3 -c "import sys; sys.exit(0 if float('$l12') < 0 else 1)" \
   && [ "$l14" = 50 ] && python3 -c "import sys; sys.exit(0 if float('$c14') > 0 else 1)"
then
    ok "premise: OpenFOAM's rev17 left end flows out at t = 0.012 (U_b = its cell's $l12) and in at 0.014 (U_b = the inletValue 50, its cell $c14): the switch changes branch between"
else
    fail "premise: rev17 left end U_b ${l12:-?} (cell ${c12:-?}) at 0.012, ${l14:-?} (cell ${c14:-?}) at 0.014"
fi
for c in fs17 rev17 rsRev; do echo "     brae $c: $(sed -n 's/^diag inletOutlet //p' "$W/b_$c/log")"; done
o=$(python3 "$FIELDS" shock "$W/of_fs17" 4 step 2> /dev/null); field() { sed -n "s/.* $2 \([^ ]*\).*/\1/p" <<< "$1"; }
pit=$(field "$o" pitot); want=$(python3 "$FIELDS" theory step | sed -n 's/.* pitot \([^ ]*\).*/\1/p')
if python3 -c "import sys; sys.exit(0 if abs(float('${pit:-nan}')/float('$want') - 1) <= 0.02 else 1)" 2> /dev/null
then
    ok "premise: OpenFOAM's fs17 pitot pressure on the step's face $pit, Rayleigh's $want (within 2 %); bow-shock standoff $(field "$o" standoff_bottom)"
else
    fail "premise: OpenFOAM's pitot ${pit:-?} against Rayleigh's ${want:-?}"
fi

# ---- purgeWrite: the directories a run leaves ---------------------------------------------------------------------------
purged()   # purged <name> <brae dir> <OF dir> <purgeWrite>
{
    local tb to n
    tb=$(alltimes "$2"); to=$(alltimes "$3"); n=$(echo $to | wc -w)
    if [ "$tb" = "$to" ] && [ "$n" = $(($4 + 1)) ]
    then
        ok "purgeWrite $4, $1: the same $n time directories as OpenFOAM's: $to"
    else
        fail "purgeWrite $4, $1: brae left '$tb', OpenFOAM '$to'"
    fi
}
purged "fs (CUDA, as shipped)" "$W/b_fs" "$W/of_fs" 10
purged "fs (host reference, as shipped)" "$W/h_fs" "$W/of_fs" 10
purged "sodP (CUDA, timeStep writes)" "$W/b_sodP" "$W/of_sodP" 2
purged "sodPr (CUDA, runTime writes)" "$W/b_sodPr" "$W/of_sodPr" 3

# ---- comparisons (arm, red, balance: tests/rho_central_gate_lib.sh) ---------------------------------------------------
arm "S forwardStep (CUDA, as shipped)" "$W/b_fs" "$W/of_fs" "$(spread fs17)" 1e-6 exact
arm "S forwardStep (host reference, as shipped)" "$W/h_fs" "$W/of_fs" "$(spread fs17)" 1e-6 exact
arm "fs17 (CUDA)" "$W/b_fs17" "$W/of_fs17" "$(spread fs17)" "$BOUND" exact
arm "fs17 (host reference)" "$W/h_fs17" "$W/of_fs17" "$(spread fs17)" "$BOUND" exact
arm "fs17e (CUDA, every step)" "$W/b_fs17e" "$W/of_fs17e" "$(spread fs17e)" "$BOUND" bound
arm "fs17e (host reference, every step)" "$W/h_fs17e" "$W/of_fs17e" "$(spread fs17e)" "$BOUND" bound
arm "fsIn17 (CUDA)" "$W/b_fsIn17" "$W/of_fsIn17" "$(spread fsIn17)" "$BOUND" exact
arm "rev17 (CUDA)" "$W/b_rev17" "$W/of_rev17" "$(spread rev17)" "$BOUND" exact
arm "rev17 (host reference)" "$W/h_rev17" "$W/of_rev17" "$(spread rev17)" "$BOUND" exact
arm "rsRev from OpenFOAM's t = 0.013 (CUDA, every step)" "$W/b_rsRev" "$W/of_rsRev" "$(spread rsRev)" "$BOUND" bound
arm "rsRev from OpenFOAM's t = 0.013 (host reference, every step)" "$W/h_rsRev" "$W/of_rsRev" "$(spread rsRev)" "$BOUND" bound
for c in sodP sodPr; do
    dm=exact; [ "$c" = sodP ] && dm=bound
    for t in $(times "$W/of_$c"); do
        v=$(python3 "$FIELDS" maxrel "$W/b_$c/$t" "$W/of_$c/$t" | awk '{print $2}')
        if le "$v" 1e-6; then ok "$c (CUDA): U p T rho at the kept t = $t within 1e-6, the files' 6 digits ($v)"; else fail "$c: t = $t $v"; fi
    done
done

# ---- the inletOutlet patches' own values --------------------------------------------------------------------------------
patcharm()   # patcharm <name> <brae dir> <OF dir> <patch> "<refs>"
{
    local name=$1 b=$2 o=$3 pa=$4 refs=$5 k t f v r sp bd over="" ratio=0 worst="" i
    k=$(cmpdir "$b")
    for t in $(times "$o")
    do
        python3 "$FIELDS" patchdiff "$b/$t" "$o/$t" "$pa" > "$k/$t.patch" 2>&1
        i=0
        for r in $refs; do i=$((i + 1)); python3 "$FIELDS" patchdiff "$r/$t" "$o/$t" "$pa" > "$k/$t.patch.ref$i" 2>&1; done
        for f in U T rho
        do
            v=$(sed -n "s/^patchdiff $f [^ ]* \([^ ]*\) .*/\1/p" "$k/$t.patch")
            sp=0
            for i in $(seq 1 $(echo $refs | wc -w)); do sp=$(mx "$sp" "$(sed -n "s/^patchdiff $f [^ ]* \([^ ]*\) .*/\1/p" "$k/$t.patch.ref$i")"); done
            bd=$(mx "$BOUND" "$(python3 -c "print(10*float('${sp:-nan}'))")")
            le "$v" "$bd" || over="$over $t:$f($v>$bd)"
            if python3 -c "import sys; sys.exit(0 if float('${v:-nan}')/float('$bd') > float('$ratio') else 1)"; then
                ratio=$(python3 -c "print('%.3f' % (float('$v')/float('$bd')))"); worst="$v at t = $t, $f (OpenFOAM's own $sp)"
            fi
        done
    done
    if [ -z "$over" ] && [ -z "$worst" ]; then ok "$name: $pa's U, T, rho values the same as OpenFOAM's at every write time"
    elif [ -z "$over" ]; then ok "$name: $pa's U, T, rho values at every write time within ten times OpenFOAM's own spread; closest ($ratio of its bound): $worst"
    else fail "$name: $pa values above the bound at$over"; fi
}
patcharm "fs17 (CUDA)" "$W/b_fs17" "$W/of_fs17" outlet "$(spread fs17)"
patcharm "fs17 (host reference)" "$W/h_fs17" "$W/of_fs17" outlet "$(spread fs17)"
patcharm "fsIn17 (CUDA)" "$W/b_fsIn17" "$W/of_fsIn17" inlet "$(spread fsIn17)"
patcharm "rev17 (CUDA)" "$W/b_rev17" "$W/of_rev17" sides "$(spread rev17)"
patcharm "rsRev (CUDA)" "$W/b_rsRev" "$W/of_rsRev" sides "$(spread rsRev)"

# ---- CPU against GPU (reported) ---------------------------------------------------------------------------------------
for c in fs17e rsRev fs17 rev17; do
    t1=$(times "$W/b_$c" | awk '{print $1}'); tl=$(times "$W/b_$c" | awk '{print $NF}')
    a1=$(python3 "$FIELDS" maxrel "$W/h_$c/$t1" "$W/b_$c/$t1"); al=$(python3 "$FIELDS" maxrel "$W/h_$c/$tl" "$W/b_$c/$tl")
    echo "     CPU against GPU, $c: at t = $t1 ${a1#maxrel }, at t = $tl ${al#maxrel }"
done

# ---- shocks: brae's against OpenFOAM's, by the same measure ------------------------------------------------------------
shockline() { python3 "$FIELDS" shock "$1" "$2" "$3" 2> /dev/null; }
for spec in "fs17 standoff_bottom" "fs17 standoff_corner" "fs17 top_x" "fs17 pitot" "fs standoff_bottom" "fs pitot"
do
    read -r c m <<< "$spec"
    for a in b h; do
        gb=$(field "$(shockline "$W/${a}_$c" 4 step)" "$m"); go=$(field "$(shockline "$W/of_$c" 4 step)" "$m")
        sp=0
        for r in $(spread fs17); do
            gr=$(field "$(shockline "$r" 4 step)" "$m")
            sp=$(python3 -c "print('%.3e' % max(float('$sp'), abs(float('${gr:-nan}') - float('${go:-nan}'))))")
        done
        sb=$(python3 -c "print('%.3e' % max(1e-6, 10*float('$sp')))")
        arml=CUDA; [ "$a" = h ] && arml="host reference"
        if python3 -c "import sys; sys.exit(0 if abs(float('${gb:-nan}') - float('${go:-nan}')) <= float('$sb') else 1)" 2> /dev/null
        then
            ok "shock $c $m ($arml): brae $gb, OpenFOAM $go (OpenFOAM's own spread $sp): within $sb"
        else
            fail "shock $c $m ($arml): brae ${gb:-?}, OpenFOAM ${go:-?} (OpenFOAM's own spread $sp), bound $sb"
        fi
    done
done

# ---- conservation and stability ---------------------------------------------------------------------------------------
for d in b_fs b_fs17 h_fs17 b_fsIn17 b_rev17 h_rev17 b_rsRev; do balance "$d"; done
tl=$(times "$W/b_fs17" | awk '{print $NF}'); tp=$(times "$W/b_fs17" | awk '{print $(NF-1)}')
echo "     steadiness, fs17: max change of p from $tp to $tl, brae $(python3 "$FIELDS" stats "$W/b_fs17" "$tl" "$tp" | awk '{print $6}'), OpenFOAM $(python3 "$FIELDS" stats "$W/of_fs17" "$tl" "$tp" | awk '{print $6}')"

# ---- controls --------------------------------------------------------------------------------------------------------
red "inletOutletInverted, fs17" "$W/cI_fs17" "$W/of_fs17" "$(spread fs17)"
red "inletOutletNewT, fs17" "$W/cT_fs17" "$W/of_fs17" "$(spread fs17)"
red "inletOutletNewT, rev17" "$W/cT_rev17" "$W/of_rev17" "$(spread rev17)"
red "laggedPhi, rev17" "$W/cL_rev17" "$W/of_rev17" "$(spread rev17)"
red "laggedPhi, rsRev" "$W/cL_rsRev" "$W/of_rsRev" "$(spread rsRev)"
if [ "$(alltimes "$W/cP_fs")" != "$(alltimes "$W/of_fs")" ]
then
    ok "control noPurge red: brae kept $(alltimes "$W/cP_fs" | wc -w) time directories, OpenFOAM $(alltimes "$W/of_fs" | wc -w)"
else
    fail "control noPurge: the same directories as OpenFOAM's: not red"
fi

# ---- performance (reported) ------------------------------------------------------------------------------------------
for c in fs17 rev17; do
    echo "     runtime $c: OpenFOAM $(grep -h '^ExecutionTime' "$W/of_$c/log" | tail -n 1 | awk '{print $3}') s; brae CUDA $(sed -n 's/^diag steps \([^ ]*\) step-seconds \([^ ]*\) per-step-us \([^ ]*\).*/\2 s (\1 steps, \3 us a step)/p' "$W/b_$c/log"), $(sed -n 's/^diag device-memory/device memory/p' "$W/b_$c/log"); host $(sed -n 's/^diag steps [^ ]* step-seconds \([^ ]*\).*/\1 s/p' "$W/h_$c/log")"
done
opensummary
echo "rho_central_forwardstep_vs_openfoam: $passes passed, $fails failed (work in $W)"
[ "$fails" = 0 ] && [ "$passes" -ge 94 ]
