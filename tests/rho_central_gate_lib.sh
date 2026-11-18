# Shared by the rhoCentralFoam gates that compare brae with OpenFOAM v2412 and with OpenFOAM's own decomposed runs
# (the wedge, forwardstep, restart, riemann and viscous gates). Sourced, not run.
#
# The caller sets: W (work directory), BIN, STEPS, FIELDS, OFBASHRC and BOUND (the field floor). It gets ok/fail
# counting into passes/fails.
#
# THE ENVELOPE (approved 2026-11-17, milestone C; declared here, never widened by a gate in passing): a field of brae's
# at a write time is within ten times OpenFOAM's own spread for that field at that time, never below BOUND. OpenFOAM's
# own spread is its distance from OpenFOAM serial when decomposed in two (simple, (2 1 1)) and in four (scotch, frozen), the
# larger of the two. BOTH are reported, each on its own, by every arm: against two alone obliqueShock's CUDA arm stood
# at up to 2.9x its bound at t = 5 to 8 (milestone C's first run), because the two-way split's one processor column
# lies in smooth flow and its distance decays there while the four-way split's does not.
# THE FOUR-WAY PARTITION IS FROZEN (milestone E, approved). OpenFOAM's scotch partition is not reproducible on this
# install: scotchDecomp.C calls SCOTCH_randomReset(), yet two runs of the forwardStep gate (2026-11-17) partitioned fs17
# differently (4032 cells on each of the four, then 4066, 4032, 3998, 4032), so OpenFOAM's four-way spread moved between
# runs (forwardStep as shipped against it alone: 0.27 of its bound in one run, 1.40x at t = 3.8, 3.9 in the other) while
# brae, OpenFOAM serial and OpenFOAM in two were bit-identical. One scotch partition per mesh -- that of the runs
# milestones C and D report -- is now kept with its checksums in tests/data/rho_central_decomposition and every
# four-way reference decomposes on it (`manual`); the envelope's rule is unchanged.

# OPEN VALIDATION ITEMS (tests/data/rho_central_open_checks.tsv; milestone G, 2026-11-18). A FAIL stays a FAIL: the
# registry only says, on the line after it, which known open item it is and its category -- round-off amplification,
# timestep sequence divergence, OpenFOAM reference failure, unexplained numerical difference. A FAIL the registry does not
# name is printed as an unexplained numerical difference until it is analysed. No threshold reads the registry.
OPENCHECKS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data/rho_central_open_checks.tsv"
GATE=$(basename "$0" .sh)
declare -A opencount=()
fail()
{
    local msg="$*" cat="" note="" g pat c n
    if [ -f "$OPENCHECKS" ]
    then
        while IFS=$'\t' read -r g pat c n
        do
            case $g in ''|'#'*) continue ;; esac
            [ "$g" = "$GATE" ] || continue
            # shellcheck disable=SC2053
            if [[ $msg == $pat ]]; then cat=$c; note=$n; break; fi
        done < "$OPENCHECKS"
    fi
    echo "FAIL $msg"
    if [ -n "$cat" ]
    then echo "     [open item, $cat: $note]"
    else cat="unexplained numerical difference (not in the registry)"; echo "     [$cat]"
    fi
    fails=$((fails + 1))
    opencount[$cat]=$((${opencount[$cat]:-0} + 1))
}
# opensummary: the FAILs by category, at the end of a gate
opensummary() { local k; for k in "${!opencount[@]}"; do echo "     open: ${opencount[$k]} FAIL(s), $k"; done; }
# ofulp <case> <OF dir> <start time>: OpenFOAM serial started with p one ulp up (rho_central_fields.py nudge), a
# DIAGNOSTIC reference beside serial, in two and in four: arm() reports brae's distance against it, never bounds by it
ofulp() { copy "$1" "$2"; python3 "$FIELDS" nudge "$2/$3/p" > /dev/null && ofrun "$2"; }
ok() { passes=$((passes + 1)); echo "ok   $*"; }
# hist: a historical result, reported and counted apart, neither a pass nor a FAIL. Used for milestone G's five raw
# conservation checks after the decision of 2026-11-18 (acceptance for an implicit solve is the residual-accounted
# balance; the raw imbalance stays reported, with its original allowance, beside it).
historical=0
hist() { historical=$((historical + 1)); echo "hist $*"; }
# OMPI_MCA_mpi_yield_when_idle: an MPI rank waiting on another yields its core (Open MPI's progress loop otherwise
# spins). Under a loaded machine the spinning ranks starved each other: a 2-way biconic25-55Run35 reference ran 391
# steps in 40 minutes (2026-11-18). Scheduling only; no arithmetic changes.
of() { bash -c "set +u; source '$OFBASHRC' > /dev/null 2>&1; export OMPI_MCA_mpi_yield_when_idle=1; set -e; $*"; }
le() { python3 -c "import sys; sys.exit(0 if float('${1:-nan}') <= float('$2') else 1)" 2> /dev/null; }
mx() { python3 -c "print('%.3e' % max(float('$1'), float('$2')))"; }
copy() { rm -rf "$2"; mkdir -p "$2"; cp -r "$1/system" "$1/constant" "$1/0" "$2/"; }
ofrun() { of "cd '$1' && rhoCentralFoam > log 2>&1" < /dev/null || fail "OpenFOAM rhoCentralFoam in $(basename "$1")"; }
# ofnp <dir> <2|4>: in two by simple (2 1 1); in four by the FROZEN scotch partition of the case's mesh (OpenFOAM's
# `manual` method on tests/data/rho_central_decomposition/<mesh>.cellDecomposition, its sha256 and the mesh's checked by
# tests/rho_central_decomposition.py). A mesh with no frozen partition fails the gate: never a fresh scotch run.
DECOMP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/rho_central_decomposition.py"
ofnp()
{
    if [ "$2" = 2 ]
    then
        printf '%s\n' 'FoamFile { version 2.0; format ascii; class dictionary; object decomposeParDict; }' \
            'numberOfSubdomains 2;' 'method simple;' 'coeffs { n (2 1 1); }' > "$1/system/decomposeParDict"
    else
        local n
        n=$(python3 "$DECOMP" install "$1" 2>&1) || { fail "frozen decomposition for $(basename "$1"): $n"; return; }
        [ "$n" = "$2" ] || { fail "frozen decomposition for $(basename "$1") has $n processors, not $2"; return; }
        printf '%s\n' 'FoamFile { version 2.0; format ascii; class dictionary; object decomposeParDict; }' \
            "numberOfSubdomains $2;" 'method manual;' 'manualCoeffs { dataFile "cellDecomposition"; }' \
            > "$1/system/decomposeParDict"
    fi
    of "cd '$1' && decomposePar > log.decomposePar 2>&1 && mpirun -np $2 rhoCentralFoam -parallel > log 2>&1 \
        && reconstructPar > log.reconstructPar 2>&1" < /dev/null || fail "OpenFOAM decomposed in $2, $(basename "$1")"
}
braerun()   # braerun <dir> [env ...]
{
    local d=$1
    shift
    (cd "$d" && env "$@" timeout 900 "$BIN" -case . > log 2> err < /dev/null; echo $? > rc)
    [ "$(cat "$d/rc")" = 0 ] || { fail "$(basename "$d") exited $(cat "$d/rc"): $(head -c 300 "$d/err")"; return 1; }
}
# the write times of a case: its time directories less 0 and the controlDict's startTime (a restart's)
times()
{
    local st
    st=$(sed -n 's/^startTime[[:space:]]\{1,\}\([^;[:space:]]*\);.*/\1/p' "$1/system/controlDict" 2> /dev/null)
    (cd "$1" && ls -d [0-9]* 2> /dev/null | grep -v '0.orig' | grep -vxF -e 0 -e "${st:-0}" | sort -g | tr '\n' ' ')
}
# every time directory a case holds, the start one included (what purgeWrite leaves)
alltimes() { (cd "$1" && ls -d [0-9]* 2> /dev/null | grep -v '0.orig' | sort -g | tr '\n' ' '); }

# cmpdir <brae dir>: where its comparison files go (never inside a case: a stray file there would read as a time)
cmpdir() { local k="$W/cmp/$(basename "$1")"; mkdir -p "$k"; echo "$k"; }
absfile() { python3 "$FIELDS" absdiff "$1" "$2" > "$3" 2>&1; }
relof() { sed -n "s/^absdiff $1 [^ ]* \([^ ]*\) .*/\1/p" "$2"; }
# spreadat <OF serial dir> <t> <field> <cmp dir> <refs...>: OpenFOAM's own spread for the field at t, the larger over
# its decompositions, then each one's: `<largest> <ref 1> <ref 2> ...`
spreadat()
{
    local o=$1 t=$2 f=$3 k=$4 r best=0 v i=0 each=""
    shift 4
    for r in "$@"
    do
        i=$((i + 1))
        [ -f "$k/$t.ref$i.abs" ] || absfile "$r/$t" "$o/$t" "$k/$t.ref$i.abs"
        v=$(relof "$f" "$k/$t.ref$i.abs")
        each="$each ${v:-nan}"
        best=$(python3 -c "print('%.3e' % max(float('$best'), float('${v:-nan}')))")
    done
    echo "$best$each"
}

# arm <name> <brae dir> <OF dir> "<refs>" <field floor> <dt: exact|bound|fixed>
#   OpenFOAM's step count, time names, deltaT (exact: the same digits; bound: within BOUND; fixed), the max Courant
#   number within ten times OpenFOAM's own spread of it, the same write times, U p T rho at every write time within
#   the envelope; then, reported, the same against each decomposition alone, and the absolute errors at the last time.
arm()
{
    local name=$1 b=$2 o=$3 refs=$4 floor=$5 dtmode=$6 k
    k=$(cmpdir "$b")
    python3 "$STEPS" compare "$b/log" "$o/log" > "$k/steps.cmp" 2>&1
    read -r nb no <<< "$(sed -n 's/^steps //p' "$k/steps.cmp")"
    read -r same wdt wdts <<< "$(sed -n 's/^deltaT //p' "$k/steps.cmp")"
    read -r names <<< "$(sed -n 's/^names //p' "$k/steps.cmp")"
    read -r wco wcos <<< "$(sed -n 's/^maxCo //p' "$k/steps.cmp")"
    local rco=0 r v
    for r in $refs
    do
        python3 "$STEPS" compare "$r/log" "$o/log" > "$k/ref.steps.cmp" 2>&1
        read -r v _ <<< "$(sed -n 's/^maxCo //p' "$k/ref.steps.cmp")"
        rco=$(python3 -c "print('%.3e' % max(float('$rco'), float('${v:-nan}')))")
    done
    local cobound
    cobound=$(mx "$BOUND" "$(python3 -c "print(10*float('$rco'))")")
    local dtok=1 dtnote="every deltaT the same"
    case $dtmode in
        exact) [ "$same" = "$nb" ] || dtok=0 ;;
        bound) le "$wdt" "$BOUND" || dtok=0; dtnote="every deltaT within $BOUND (worst $wdt at step $wdts; same digits at $same)" ;;
        fixed) [ "$same" = "$nb" ] || dtok=0; dtnote="deltaT fixed" ;;
    esac
    if [ -n "$nb" ] && [ "$nb" = "$no" ] && [ "$names" = "$nb" ] && [ "$dtok" = 1 ]
    then
        ok "$name: $nb steps as OpenFOAM's; every time name the same; $dtnote"
    else
        fail "$name: ${nb:-?} steps against ${no:-?}; names ${names:-?}; deltaT same ${same:-?} (worst ${wdt:-?} at ${wdts:-?})"
    fi
    if le "$wco" "$cobound"
    then
        ok "$name: max Courant number within $cobound at every step (worst $wco at step $wcos; OpenFOAM's own spread $rco)"
    else
        fail "$name: max Courant number ${wco:-?} at step ${wcos:-?}, above $cobound"
    fi
    local tb to
    tb=$(times "$b"); to=$(times "$o")
    if [ "$tb" = "$to" ] && [ -n "$to" ]
    then
        ok "$name: the same $(echo $to | wc -w) time directories written"
    else
        fail "$name: wrote '$tb', OpenFOAM '$to'"
        return
    fi
    # per decomposition alone (reported): the worst ratio to ten times its spread, and where brae is above it
    local nref
    nref=$(echo $refs | wc -w)
    local -a wr ov
    local j
    for j in $(seq 1 "$nref"); do wr[$j]=0; ov[$j]=""; done
    local over="" worst=0 wt="" wf="" ratio=0 t f sp bd last spl q
    for t in $to
    do
        absfile "$b/$t" "$o/$t" "$k/$t.abs"
        for f in U p T rho
        do
            v=$(relof "$f" "$k/$t.abs")
            spl=($(spreadat "$o" "$t" "$f" "$k" $refs))
            sp=${spl[0]}
            bd=$(mx "$floor" "$(python3 -c "print(10*float('${sp:-nan}'))")")
            le "$v" "$bd" || over="$over $t:$f($v>$bd)"
            if python3 -c "import sys; sys.exit(0 if float('${v:-nan}')/float('$bd') > float('$ratio') else 1)"
            then
                ratio=$(python3 -c "print('%.3f' % (float('$v')/float('$bd')))"); worst=$v; wt=$t; wf="$f (OpenFOAM's own $sp, bound $bd)"
            fi
            for j in $(seq 1 "$nref")
            do
                q=$(python3 -c "print('%.3f' % (float('${v:-nan}')/max(float('$floor'), 10*float('${spl[$j]:-nan}'))))")
                python3 -c "import sys; sys.exit(0 if float('$q') > float('${wr[$j]}') else 1)" && wr[$j]=$q
                python3 -c "import sys; sys.exit(0 if float('$q') > 1 else 1)" && ov[$j]="${ov[$j]} $t:$f"
            done
        done
        last=$t
    done
    if [ -z "$over" ]
    then
        ok "$name: U p T rho at $(echo $to | wc -w) write times within ten times OpenFOAM's own spread (floor $floor); closest to its bound ($ratio of it): $worst at t = $wt, $wf"
    else
        fail "$name: fields above the bound at$over"
    fi
    local label
    for j in $(seq 1 "$nref")
    do
        label="in two (simple)"; [ "$j" = 2 ] && label="in four (scotch, frozen)"
        if [ -z "${ov[$j]}" ]
        then
            echo "     $name: against OpenFOAM $label alone: at most ${wr[$j]} of ten times its spread"
        else
            echo "     $name: against OpenFOAM $label alone: up to ${wr[$j]} x ten times its spread, above it at $(echo ${ov[$j]} | wc -w) field-times (t:field): $(echo ${ov[$j]} | cut -d' ' -f1-12)"
        fi
    done
    # OpenFOAM serial with p one ulp up, when the gate ran it (<OF dir>_ulp): a diagnostic reference, not a bound
    if [ -d "${o}_ulp" ]
    then
        local ul tf n lo hi
        ul=$(python3 "$FIELDS" ulpref "$b" "$o" "${o}_ulp" "$(echo $to | tr ' ' ',')" $(echo "$over" | grep -o '[^ ]*:[A-Za-z]*'))
        read -r _ n lo hi tf <<< "$(echo "$ul" | head -n 1)"
        echo "     $name: diagnostic, OpenFOAM serial with p one ulp up (not a bound): brae's distance from serial is $lo to $hi x the one-ulp run's over $n field-times (largest at $tf)"
        echo "$ul" | sed -n 's/^at \(.*\)/     '"$name"': at \1/p' | head -n 8
    fi
    echo "     $name absolute, t = $last:$(awk '{printf " %s %s (rel %s)", $2, $3, $4}' "$k/$last.abs")"
}

# ctlrun <dir> [env ...]: a negative control's run. It may stop on an unphysical state where OpenFOAM's run goes on:
# that is not a failure of the gate, and red() reads it
ctlrun()
{
    local d=$1
    shift
    (cd "$d" && env "$@" timeout 900 "$BIN" -case . > log 2> err < /dev/null; echo $? > rc)
}
# red <name> <control dir> <OF dir> "<refs>": some field at some write time beyond 100 x its bound, or the control
# stopped on an unphysical state (rho, p or T not positive, or the e -> T inversion failing) where OpenFOAM ran on
red()
{
    local name=$1 c=$2 o=$3 refs=$4 best=0 at="" t f v sp bd q k spl stopped=""
    if [ -f "$c/rc" ] && [ "$(cat "$c/rc")" != 0 ]
    then
        if grep -q "unphysical state" "$c/err"
        then
            stopped=$(grep -o "step to time [^ ]*" "$c/err" | head -n 1)
        else
            fail "control $name exited $(cat "$c/rc") for another reason: $(head -c 300 "$c/err")"
            return
        fi
    fi
    k=$(cmpdir "$c")
    for t in $(times "$o")
    do
        absfile "$c/$t" "$o/$t" "$k/$t.abs" || continue
        for f in U p T rho
        do
            v=$(relof "$f" "$k/$t.abs")
            spl=($(spreadat "$o" "$t" "$f" "$k" $refs))
            sp=${spl[0]}
            bd=$(mx "$BOUND" "$(python3 -c "print(10*float('${sp:-nan}'))")")
            q=$(python3 -c "print('%.3e' % (float('${v:-0}')/float('$bd')))")
            if python3 -c "import sys; sys.exit(0 if float('$q') > float('$best') else 1)"; then best=$q; at="$f at $t: $v against a bound of $bd"; fi
        done
    done
    if python3 -c "import sys; sys.exit(0 if float('$best') > 100 else 1)"
    then
        ok "control $name red: $at ($best x the bound)${stopped:+; it then stopped on an unphysical state ($stopped) where OpenFOAM ran to its end}"
    elif [ -n "$stopped" ]
    then
        ok "control $name red: it stopped on an unphysical state ($stopped) where OpenFOAM ran to its end; before that at most $best x the bound${at:+ ($at)}"
    else
        fail "control $name: at most $best x the bound${at:+ ($at)}: not red"
    fi
}

# balance <brae dir>: `diag balance` within `diag roundoff` (four ulps of every cell's total a step), and positivity
balance()   # balance <dir> [raw]: with "raw", the conservation line is a historical report (hist), not a check
{
    local d=$1 mode=${2:-} bal rnd rel bad t
    bal=$(sed -n 's/^diag balance mass \([^ ]*\) energy \([^ ]*\) momentum \([^ ]*\) \([^ ]*\) \([^ ]*\) .*/\1 \2 \3 \4 \5/p' "$W/$d/log")
    rnd=$(sed -n 's/^diag roundoff mass \([^ ]*\) energy \([^ ]*\) momentum \([^ ]*\) \([^ ]*\) \([^ ]*\) .*/\1 \2 \3 \4 \5/p' "$W/$d/log")
    rel=$(sed -n 's/.*relative to the start totals: mass \([^ ]*\) energy \([^)]*\)).*/mass \1 energy \2/p' "$W/$d/log")
    if python3 -c "
import sys
b = [abs(float(x)) for x in '$bal'.split()]; r = [float(x) for x in '$rnd'.split()]
sys.exit(0 if len(b) == 5 and len(r) == 5 and all(x <= y for x, y in zip(b, r)) else 1)" 2> /dev/null
    then
        if [ "$mode" = raw ]
        then hist "$d raw conservation within its round-off (reported; acceptance is the residual-accounted balance): balance $bal within $rnd"
        else ok "$d conserves mass, momentum and energy to its round-off: balance $bal within $rnd (relative to the start totals: $rel)"
        fi
    elif [ "$mode" = raw ]
    then
        hist "$d raw conservation: balance '${bal:-none}' against round-off '${rnd:-none}' -- milestone G's original check, a historical tolerance-limited result: the implicit U and e solves stop at their declared tolerance and leave deltaT*sum(b - A x); acceptance since 2026-11-18 is the residual-accounted balance (vbalance), with the identity and the tolerance sweep"
    else
        fail "$d conservation: balance '${bal:-none}' against round-off '${rnd:-none}'"
    fi
    bad=""
    for t in $(times "$W/$d")
    do
        read -r _ r0 p0 T0 nan _ <<< "$(python3 "$FIELDS" stats "$W/$d" "$t")"
        python3 -c "import sys; sys.exit(0 if float('$r0') > 0 and float('$p0') > 0 and float('$T0') > 0 and int('$nan') == 0 else 1)" 2> /dev/null || bad="$bad $t"
    done
    if [ -z "$bad" ]; then ok "$d stable: rho, p, T positive and finite at all $(times "$W/$d" | wc -w) write times"; else fail "$d: non-positive or NaN at$bad"; fi
}

# vbalance <brae dir>: a viscous run's momentum and energy balance once the correctors' unconserved remainder,
# deltaT*sum(b - A x), is counted, within `diag roundoff` (milestones G and H: the viscous gates' SOLVE-CONSERVATION)
vbalance()
{
    local d=$1 bal res rnd
    bal=$(sed -n 's/^diag balance mass [^ ]* energy \([^ ]*\) momentum \([^ ]*\) \([^ ]*\) \([^ ]*\) .*/\1 \2 \3 \4/p' "$W/$d/log")
    res=$(sed -n 's/^diag solve-residual momentum \([^ ]*\) \([^ ]*\) \([^ ]*\) energy \([^ ]*\) .*/\4 \1 \2 \3/p' "$W/$d/log")
    rnd=$(sed -n 's/^diag roundoff mass [^ ]* energy \([^ ]*\) momentum \([^ ]*\) \([^ ]*\) \([^ ]*\) .*/\1 \2 \3 \4/p' "$W/$d/log")
    local sums
    sums=$(python3 -c "
b = [float(x) for x in '$bal'.split()]; r = [float(x) for x in '$res'.split()]
print(' '.join('%.3e' % (x + y) for x, y in zip(b, r)) if len(b) == 4 and len(r) == 4 else '')" 2> /dev/null)
    if python3 -c "
import sys
s = [abs(float(x)) for x in '$sums'.split()]; q = [float(x) for x in '$rnd'.split()]
sys.exit(0 if len(s) == 4 and len(q) == 4 and all(x <= y for x, y in zip(s, q)) else 1)" 2> /dev/null
    then ok "$d conserves energy and momentum to round-off once the solves' remainder is counted: balance + deltaT*sum(b - A x) = $sums (energy, momentum x y z) within $rnd; the remainder itself $res"
    else fail "$d solve-conservation: balance '$bal' + remainder '$res' = '$sums' against round-off '$rnd'"; fi
}

# identity <brae dir>: every U and e solve satisfies sum_c rho V (x - x_p) = deltaT*(its patch fluxes [+ Sf & tauMC_b]
# - sum(b - A x)) to within 1e-12 of sum_c rho V |U| (|e|) (`diag solve-identity`); the remainders reported
identity()
{
    local d=$1 id
    id=$(sed -n 's/^diag solve-identity momentum \([^ ]*\) \([^ ]*\) \([^ ]*\) energy \([^ ]*\) .*/\1 \2 \3 \4/p' "$W/$d/log")
    if python3 -c "
import sys
v = [float(x) for x in '$id'.split()]
sys.exit(0 if len(v) == 4 and max(v) <= 1e-12 else 1)" 2> /dev/null
    then ok "$d: every solve satisfies its conservation identity, largest error $(echo $id | tr ' ' '\n' | sort -g | tail -n 1) of sum rho V |U| (|e|) (momentum x y z, energy: $id; bound 1e-12)"
    else fail "$d: solve identity '${id:-none}' above 1e-12"; fi
    echo "     $d remainders: $(sed -n 's/^diag solve-residual momentum \(.*\) (deltaT.*/signed momentum \1/p' "$W/$d/log"); $(sed -n 's/^diag solve-residual-l1 momentum \(.*\) (deltaT.*/unsigned momentum \1/p' "$W/$d/log")"
}
