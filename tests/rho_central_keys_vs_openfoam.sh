#!/usr/bin/env bash
# brae_rhoCentralFoam against OpenFOAM v2412 rhoCentralFoam on three settings brae used to drop or refuse (wave 1, Q4,
# 2026-11-21). A NEW gate, beside the six known-red ones (tools/known_red_check.py holds those to their registered
# FAILs; no arm is added to them).
#
# CASES
#   shock_linear  shockTube (100 cells) with `interpolationSchemes { default linear; }` and no reconstruct entries --
#                 what 19 of the 19 density-based fvSchemes of the other v2412 solvers write. OpenFOAM then
#                 reconstructs rho, rhoU, rPsi, e and c with the plain linear scheme in both directions
#                 (directionInterpolate.H, linear.H); brae REFUSED `linear` (rhoCentralControls.cu) and now runs it as a
#                 limiter of exactly 1 (kLinearReconTwoByk). Fixed deltaT 2e-5, 30 steps, writes every 10: the central
#                 reconstruction of a shock oscillates and OpenFOAM's own adaptive run goes non-finite at its 65th step.
#   jet_lu        jet_sst (tests/rho_central_turb.py: LadenburgJet60psi with kOmegaSST) with div(phi,k) and
#                 div(phi,omega) `Gauss linearUpwind limited` and gradSchemes `limited cellLimited Gauss linear 1`.
#                 OpenFOAM's linearUpwind looks the named gradient up (linearUpwind.H: mesh.gradScheme(name)); brae
#                 accepted any name and ran an unlimited Gauss gradient (compressibleRAS_cpp.cu). To 2e-5 s as shipped.
#   jet_relax     jet_sst with relaxationFactors { equations { "(k|omega)" 0.7; } }: kOmegaSSTBase::correct relaxes both
#                 equations (kOmegaSSTBase.C:590, :616, fvMatrix::relax); brae never read the factors.
# REFERENCE  OpenFOAM serial; its spread in two (simple) and in four (the frozen scotch partitions of
#   tests/data/rho_central_decomposition) -- the envelope of tests/rho_central_gate_lib.sh, unchanged: ten times
#   OpenFOAM's own spread, floor BOUND 8e-14 -- and OpenFOAM serial with p one ulp up, a diagnostic.
# CHECKS, DECLARED BEFORE THE GATE FIRST RAN (2026-11-21), the CUDA arm (the default path; RAS runs there only):
#   per case     steps, every deltaT and name, Courant, directories, U p T rho at every write time (lib `arm`, exact)
#   TURBULENCE   jet_lu, jet_relax: k omega nut alphat at every write time within ten times OpenFOAM's own spread
#   CONTROLS     OpenFOAM WITHOUT each setting -- shipped vanLeer for shock_linear, an unlimited `limited Gauss linear`
#                for jet_lu, no relaxation for jet_relax -- red against OpenFOAM with it, by more than 100 x the bound
#                (U p T rho; and k omega nut alphat for the jet cases, each on its own): what brae computed before, so a
#                brae that dropped the setting again fails
#   SOLVES       jet cases: the solves in order and their iteration counts, REPORTED (not a check)
# NEGATIVE CONTROL: the binary of main 9b7b1e58 refuses shock_linear by name and runs both jet cases outside the bound.
# NOT CLAIMED  the host reference arm (it refuses RAS; shock_linear's host arm is reported), mesh motion, other limiters.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BRAE_RCF_BIN:-$ROOT/build/brae_rhoCentralFoam}"
STEPS="$ROOT/tests/rho_central_steps.py"
FIELDS="$ROOT/tests/rho_central_fields.py"
TURB="$ROOT/tests/rho_central_turb.py"
OFBASHRC=${OF_BASHRC:-${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}}
TUT="${BRAE_OF_TUTORIALS:-/usr/lib/openfoam/openfoam2412/tutorials}"
RCF="$TUT/compressible/rhoCentralFoam"
[ -x "$BIN" ] || { echo "SKIP: no brae_rhoCentralFoam at $BIN"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: no OpenFOAM v2412 at $OFBASHRC"; exit 77; }
[ -d "$RCF/LadenburgJet60psi/0.orig" ] || { echo "SKIP: no LadenburgJet60psi at $RCF"; exit 77; }
[ -d "$RCF/shockTube/0.orig" ] || { echo "SKIP: no shockTube at $RCF"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
W=${BRAE_GATE_OUT:-$(mktemp -d /tmp/brae-rcf-keys.XXXXXX)}
mkdir -p "$W"
export PYTHONDONTWRITEBYTECODE=1
BOUND=${BRAE_RCF_SOD_BOUND:-8e-14}
fails=0
passes=0
. "$ROOT/tests/rho_central_gate_lib.sh"
J=${BRAE_GATE_JOBS:-6}
throttle() { while [ "$(jobs -rp | wc -l)" -ge "$J" ]; do sleep 1; done; }

# edit <file> <python regex> <replacement>: one substitution that must change the file
edit() { python3 - "$@" <<'PY'
import re, sys
p, pat, rep = sys.argv[1], sys.argv[2], sys.argv[3].replace('\\n', '\n')
t = open(p).read()
t2 = re.sub(pat, lambda m: rep, t, count=1, flags=re.S)
assert t2 != t, (p, pat)
open(p, 'w').write(t2)
PY
}

# ---- the cases ------------------------------------------------------------------------------------------------------
stage_shock()   # stage_shock <dir> <linear|shipped>
{
    local d=$1
    rm -rf "$d"; mkdir -p "$d"
    cp -r "$RCF/shockTube/system" "$RCF/shockTube/constant" "$d/"
    cp -r "$RCF/shockTube/0.orig" "$d/0"
    of "cd '$d' && blockMesh > log.blockMesh 2>&1 && setFields > log.setFields 2>&1" < /dev/null || fail "stage $d: mesh"
    python3 - "$d/system/controlDict" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
for key, val in [('adjustTimeStep', 'no'), ('deltaT', '2e-05'), ('endTime', '0.0006'), ('writeControl', 'timeStep'),
                 ('writeInterval', '10'), ('writePrecision', '15'), ('writeFormat', 'ascii')]:
    s, n = re.subn(r'^%s\s.*$' % key, '%s %s;' % (key.ljust(15), val), s, flags=re.M)
    assert n == 1, key
open(p, 'w').write(s)
PY
    [ "$2" = linear ] && edit "$d/system/fvSchemes" 'interpolationSchemes\s*\{[^}]*\}' \
        'interpolationSchemes\n{\n    default         linear;\n}'
    return 0
}
stage_jet()   # stage_jet <dir> <plain|lu|luUnlimited|relax>
{
    local d=$1 k=$2
    rm -rf "$d"; mkdir -p "$d"
    cp -r "$RCF/LadenburgJet60psi/system" "$RCF/LadenburgJet60psi/constant" "$d/"
    cp -r "$RCF/LadenburgJet60psi/0.orig" "$d/0"
    gunzip -f "$d/0/"*.gz || { fail "stage $d: 0.orig"; return 1; }
    python3 "$TURB" setup jet_sst "$d" || { fail "stage $d: setup"; return 1; }
    of "cd '$d' && blockMesh > log.blockMesh 2>&1" < /dev/null || fail "stage $d: blockMesh"
    case $k in
        lu|luUnlimited)
            edit "$d/system/fvSchemes" 'div\(phi,k\)\s+Gauss limitedLinear 1;' 'div(phi,k)      Gauss linearUpwind limited;'
            edit "$d/system/fvSchemes" 'div\(phi,omega\)\s+Gauss limitedLinear 1;' 'div(phi,omega)  Gauss linearUpwind limited;'
            if [ "$k" = lu ]
            then edit "$d/system/fvSchemes" '(gradSchemes\s*\{)' 'gradSchemes\n{\n    limited         cellLimited Gauss linear 1;'
            else edit "$d/system/fvSchemes" '(gradSchemes\s*\{)' 'gradSchemes\n{\n    limited         Gauss linear;'
            fi ;;
        relax)
            printf '\nrelaxationFactors\n{\n    equations\n    {\n        "(k|omega)"     0.7;\n    }\n}\n' >> "$d/system/fvSolution" ;;
    esac
    return 0
}
stage_shock "$W/shock_linear" linear
stage_shock "$W/shock_shipped" shipped
stage_jet "$W/jet_lu" lu
stage_jet "$W/jet_luUnlimited" luUnlimited
stage_jet "$W/jet_relax" relax
stage_jet "$W/jet_plain" plain
CASES="shock_linear jet_lu jet_relax"

# ---- OpenFOAM ---------------------------------------------------------------------------------------------------------
for c in $CASES; do copy "$W/$c" "$W/of_$c"; copy "$W/$c" "$W/of_${c}_np2"; copy "$W/$c" "$W/of_${c}_np4"; done
for c in shock_shipped jet_luUnlimited jet_plain; do copy "$W/$c" "$W/of_$c"; done
for c in $CASES shock_shipped jet_luUnlimited jet_plain; do throttle; ofrun "$W/of_$c" & done
for c in $CASES; do throttle; ofulp "$W/$c" "$W/of_${c}_ulp" 0 & done
wait
for c in $CASES; do throttle; ofnp "$W/of_${c}_np2" 2 & done
wait
for c in $CASES; do throttle; ofnp "$W/of_${c}_np4" 4 & done
wait
for c in $CASES shock_shipped jet_luUnlimited jet_plain; do grep -qx End "$W/of_$c/log" || fail "OpenFOAM did not finish $c"; done
for c in $CASES; do for r in _np2 _np4 _ulp; do grep -qx End "$W/of_$c$r/log" || fail "OpenFOAM $r did not finish $c"; done; done

# ---- brae -------------------------------------------------------------------------------------------------------------
for c in $CASES; do copy "$W/$c" "$W/b_$c"; braerun "$W/b_$c"; done
if grep -q "reconstruct: rho linear, T linear, U linear" "$W/b_shock_linear/log"
then ok "shock_linear: brae reconstructs linearly (its log: $(grep -m1 '^reconstruct:' "$W/b_shock_linear/log"))"
else fail "shock_linear: brae's log does not say it reconstructs rho, T and U linearly"; fi
# the host reference arm on shock_linear (reported: the host arm is not the default path)
copy "$W/shock_linear" "$W/h_shock_linear"
(cd "$W/h_shock_linear" && BRAE_RHOCENTRALFOAM_ARM=cpu "$BIN" -case . > log 2> err < /dev/null; echo $? > rc)

# ---- comparisons ------------------------------------------------------------------------------------------------------
spread() { echo "$W/of_${1}_np2 $W/of_${1}_np4"; }
for c in $CASES; do arm "$c (CUDA)" "$W/b_$c" "$W/of_$c" "$(spread "$c")" "$BOUND" exact; done
if [ "$(cat "$W/h_shock_linear/rc")" = 0 ]
then
    v=$(python3 "$FIELDS" absdiff "$W/h_shock_linear/0.0006" "$W/of_shock_linear/0.0006" U | awk '/^absdiff/ {print $4}')
    echo "     shock_linear (host reference, reported): U at 0.0006 ${v:-?} from OpenFOAM serial"
fi

# the turbulence fields at every write
for c in jet_lu jet_relax; do
    over=""; worst=-1; wl=""
    for t in $(times "$W/of_$c"); do
        for f in k omega nut alphat; do
            vb=$(python3 "$FIELDS" absdiff "$W/b_$c/$t" "$W/of_$c/$t" "$f" | awk '/^absdiff/ {print $4}')
            v2=$(python3 "$FIELDS" absdiff "$W/of_${c}_np2/$t" "$W/of_$c/$t" "$f" | awk '/^absdiff/ {print $4}')
            v4=$(python3 "$FIELDS" absdiff "$W/of_${c}_np4/$t" "$W/of_$c/$t" "$f" | awk '/^absdiff/ {print $4}')
            bd=$(mx "$BOUND" "$(python3 -c "print(10*max(float('${v2:-nan}'), float('${v4:-nan}')))")")
            le "${vb:-nan}" "$bd" || over="$over $t:$f(${vb:-nan}>$bd)"
            r=$(python3 -c "print('%.3g' % (float('${vb:-nan}')/float('$bd')))")
            python3 -c "import sys; sys.exit(0 if float('$r') > float('$worst') else 1)" 2> /dev/null \
                && { worst=$r; wl="$f $vb at t = $t (OpenFOAM in two $v2, in four $v4)"; }
        done
    done
    if [ -z "$over" ]; then ok "$c (CUDA): k omega nut alphat at every write time within ten times OpenFOAM's own spread; closest ($worst of the bound): $wl"
    else fail "$c (CUDA): turbulence fields above the bound at$over"; fi
done

# the solves, reported
for c in jet_lu jet_relax; do
    k=$(cmpdir "$W/b_$c")
    python3 "$STEPS" solves "$W/b_$c/log" "$W/of_$c/log" > "$k/solves.cmp" 2>&1
    read -r nb no same <<< "$(sed -n 's/^solves //p' "$k/solves.cmp")"
    read -r iok ifirst <<< "$(sed -n 's/^iterations //p' "$k/solves.cmp")"
    echo "     $c (CUDA, reported): ${nb:-?} solves against OpenFOAM's ${no:-?}, order the same: ${same:-?}; iteration counts OpenFOAM serial's on ${iok:-?} (first other at ${ifirst:-none})"
done

# ---- controls: OpenFOAM without the setting is what brae computed before ----------------------------------------------
red "OpenFOAM with the shipped vanLeer (brae refused linear)" "$W/of_shock_shipped" "$W/of_shock_linear" "$(spread shock_linear)"
# tred <name> <control dir> <OF dir> <case>: the same on the turbulence fields, which the two jet settings act on first
tred()
{
    local name=$1 cdir=$2 o=$3 c=$4 best=0 at="" t f v v2 v4 bd q
    for t in $(times "$o"); do
        for f in k omega nut alphat; do
            v=$(python3 "$FIELDS" absdiff "$cdir/$t" "$o/$t" "$f" | awk '/^absdiff/ {print $4}')
            v2=$(python3 "$FIELDS" absdiff "$W/of_${c}_np2/$t" "$o/$t" "$f" | awk '/^absdiff/ {print $4}')
            v4=$(python3 "$FIELDS" absdiff "$W/of_${c}_np4/$t" "$o/$t" "$f" | awk '/^absdiff/ {print $4}')
            bd=$(mx "$BOUND" "$(python3 -c "print(10*max(float('${v2:-nan}'), float('${v4:-nan}')))")")
            q=$(python3 -c "print('%.3e' % (float('${v:-0}')/float('$bd')))")
            if python3 -c "import sys; sys.exit(0 if float('$q') > float('$best') else 1)"; then best=$q; at="$f at $t: $v against a bound of $bd"; fi
        done
    done
    if python3 -c "import sys; sys.exit(0 if float('$best') > 100 else 1)"
    then ok "control $name red on the turbulence fields: $at ($best x the bound)"
    else fail "control $name: the turbulence fields at most $best x the bound${at:+ ($at)}: not red"; fi
}
red "OpenFOAM with linearUpwind's gradient unlimited (what brae ran)" "$W/of_jet_luUnlimited" "$W/of_jet_lu" "$(spread jet_lu)"
tred "OpenFOAM with linearUpwind's gradient unlimited (what brae ran)" "$W/of_jet_luUnlimited" "$W/of_jet_lu" jet_lu
red "OpenFOAM without relaxation (what brae ran)" "$W/of_jet_plain" "$W/of_jet_relax" "$(spread jet_relax)"
tred "OpenFOAM without relaxation (what brae ran)" "$W/of_jet_plain" "$W/of_jet_relax" jet_relax
opensummary
echo "rho_central_keys_vs_openfoam: $passes passed, $fails failed (work in $W)"
[ "$fails" = 0 ] && [ "$passes" -ge 1 ]
