#!/usr/bin/env bash
# pimpleFoam under `adjustTimeStep yes`, step by step against OpenFOAM v2412 (work package M4, issue #53).
#
# WHAT IT HOLDS. brae's pimpleFoam runs OpenFOAM's clock under adjustTimeStep (run_clock.cuh, gpuPimpleFoam.cu):
# CourantNo.H + setInitialDeltaT.H before the loop, CourantNo.H + setDeltaT.H + ++runTime before EVERY step (the first
# included), Time::adjustDeltaT's trimming to the write times, and <time>/uniform/time written as OpenFOAM writes it.
# Before M4 the first step skipped setDeltaT.H and no step was trimmed, so the sequence -- and the transient fields at a
# given time -- differed from OpenFOAM's from the first step.
#
# THE CASE. incompressible/pimpleFoam/RAS/pitzDaily as shipped (adjustTimeStep yes, maxCo 5, writeControl adjustable,
# kEpsilon), to endTime 0.003 with a write every 0.001 at writePrecision 17, so each logged and written number is the
# double the run held. ~26 steps.
#
# ARMS
#   TRANSCRIPTION  OpenFOAM's own log replayed through pimple_clock_replay.py (OpenFOAM's rules, transcribed): every step's
#                  deltaT to the bit. If it is not, the replay is wrong and the gate says so rather than judge brae by it.
#   CLOCK          brae's log replayed the same way: every step's deltaT is OpenFOAM's rules applied to brae's own Courant
#                  numbers, to the bit. Exact; it holds the loop, not the flow (the Courant numbers are each code's own).
#   SEQUENCE       as many steps as OpenFOAM, the same write times (0.001, 0.002, 0.003); uniform/time beside each.
#   FIELDS         REPORTED, NOT GATED. U p k epsilon nut at each write against OpenFOAM's, beside OpenFOAM's own spread
#                  (its 2-way simple and 4-way scotch runs against its serial run) and beside the same driver's agreement
#                  at a FIXED deltaT on the same case (1.25e-4, landing on the same writes). Under adjustTimeStep each
#                  code's Courant numbers are its own flux's, so a field difference feeds back into the steps; no bound
#                  for that has been derived yet, and none is invented here (measured: p up to ~10x OpenFOAM's spread,
#                  1-3x the fixed-deltaT agreement). This gate holds the CLOCK; the fields' bound is an open decision.
#   RESTART        from the 0.002 write, four ways (brae from brae's and from OpenFOAM's, OpenFOAM from brae's and from
#                  its own): each run's steps replay to the bit from the uniform/time it read (index > 0, so no
#                  setInitialDeltaT); brae restarted equals brae continuous at 0.003 to the bit, as OpenFOAM's does.
#   FO             a forces functionObject at executeControl/writeControl timeStep 3, continuous and restarted from 0.002, each
#                  code on its own steps: a row at every global index divisible by 3 (runTime.timeIndex(), continued from
#                  uniform/time), carrying the time that code logged there. OpenFOAM's own rows are the premise. Under
#                  adjustTimeStep brae labelled them start + n x controlDict's deltaT and counted from the restart (TJunction:
#                  0.003 0.006 0.009 for OpenFOAM's 0.004368 0.011916 0.024959). Control BRAE_CONTROL_FO_STEP_ESTIMATED=1.
#   CONTROLS       BRAE_CONTROL_PIMPLE_NO_FIRST_SETDELTAT=1 (no setDeltaT.H before step 1): the replay must catch step 1.
#                  BRAE_CONTROL_PIMPLE_UNTRIMMED=1 (no adjustDeltaT): the replay must catch it and the writes must land
#                  off OpenFOAM's write times.
# MEASURED (GB10, 2026-11-21): 26 steps, every one to the bit on both logs; writes 0.001 0.002 0.003 in both codes, with
# identical uniform/time; restarts 7 steps each, all to the bit; brae
# restarted = brae continuous exactly; the controls caught at step 1 and on every step (writes 0.00105867 ...).
# FO (Arrhenius 5863200, 2026-11-28): OpenFOAM's rows on its own steps, 8 continuous (indices 3 ... 24) and 2 restarted
# (21, 24); brae's the same, at the same printed times (0.000259615 0.000514123 ...; 0.00228571 0.00271429). Develop 49072d1b
# wrote 0.0003 0.0006 ... and 3 rows after the restart (0.0021 0.0024 0.0027); the control reproduces that.
# Labels: slow. Honours OFBASHRC; exits 77 without OpenFOAM, a GPU or the binary.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BRAE_BIN:-$ROOT/build/brae_pimpleFoam}"
OFBASHRC=${OF_BASHRC:-${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}}
REPLAY="$ROOT/tests/pimple_clock_replay.py"
[ -x "$BIN" ] || { echo "SKIP: $BIN not built"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: no OpenFOAM v2412 at $OFBASHRC"; exit 77; }
command -v nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU"; exit 77; }
set --
set +u   # the bashrc reads unset variables
# shellcheck disable=SC1090
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u
TUT="${BRAE_OF_TUTORIALS:-${FOAM_TUTORIALS:-}}/incompressible/pimpleFoam/RAS/pitzDaily"
[ -d "$TUT/system" ] || { echo "SKIP: no pimpleFoam/RAS/pitzDaily tutorial at $TUT"; exit 77; }
command -v pimpleFoam > /dev/null 2>&1 || { echo "SKIP: pimpleFoam not on PATH"; exit 77; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-86s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1 || true; }
END=0.003; WI=0.001; RS=0.002; MAXCO=5

mkdir -p "$W/base"; cp -r "$TUT/0" "$TUT/constant" "$TUT/system" "$W/base/"
sed -i -e "s/^endTime .*/endTime         $END;/" -e "s/^writeInterval .*/writeInterval   $WI;/" \
       -e 's/^writePrecision .*/writePrecision  17;/' "$W/base/system/controlDict"
for k in "adjustTimeStep +yes" "maxCo +$MAXCO;" "writeControl +adjustable" "deltaT +0.0001;"; do
    grep -qE "^$k" "$W/base/system/controlDict" || { echo "FAIL: the tutorial no longer has \`$k\` -- the gate's premise moved"; exit 1; }
done
( cd "$W/base" && blockMesh > log.blockMesh 2>&1 ) || { echo "FAIL: blockMesh"; exit 1; }
stage() { mkdir -p "$1"; cp -r "$W/base/0" "$W/base/constant" "$W/base/system" "$1/"; }
decomp() { printf 'FoamFile{version 2.0;format ascii;class dictionary;object decomposeParDict;}\nnumberOfSubdomains %d;\nmethod %s;\ncoeffs{n (%d 1 1);}\n' "$2" "$3" "$2" > "$1/system/decomposeParDict"; }
ofrun() { ( cd "$1" && nice -n 10 pimpleFoam > log 2>&1 ); }
ofpar() { ( cd "$1" && decomposePar > log.decompose 2>&1 && nice -n 10 mpirun -np "$2" pimpleFoam -parallel > log 2>&1 \
            && reconstructPar > log.reconstruct 2>&1 ); }
brrun() { local d=$1; shift; ( cd "$d" && env "$@" "$BIN" -case . > log 2>&1 ); }
replay() { python3 "$REPLAY" "$1/log" --maxCo $MAXCO --deltaT0 "$2" --start "$3" --index "$4" --writeControl adjustable \
                   --writeInterval $WI; }
written() { (cd "$1" && ls -d [0-9]* 2>/dev/null | grep -vx 0 | sort -g | tr '\n' ' '); }
utval() { sed -n "s/^$2[[:space:]]\+\"\?\([^;\"]*\)\"\?;/\1/p" "$1/uniform/time"; }

stage "$W/of"; ofrun "$W/of" || { echo "FAIL: OpenFOAM serial"; tail -5 "$W/of/log"; exit 1; }
stage "$W/of2"; decomp "$W/of2" 2 simple; ofpar "$W/of2" 2 || { echo "FAIL: OpenFOAM on 2"; exit 1; }
stage "$W/of4"; decomp "$W/of4" 4 scotch; ofpar "$W/of4" 4 || { echo "FAIL: OpenFOAM on 4"; exit 1; }
stage "$W/br"; brrun "$W/br" || { echo "FAIL: brae"; tail -5 "$W/br/log"; exit 1; }
stage "$W/cf"; brrun "$W/cf" BRAE_CONTROL_PIMPLE_NO_FIRST_SETDELTAT=1
# the same case at a fixed deltaT (1.25e-4: 8 steps a write), both codes: the driver's fixed-step agreement, for reference
fixed() { stage "$1"; sed -i -e 's/^adjustTimeStep .*/adjustTimeStep  no;/' -e 's/^deltaT .*/deltaT          0.000125;/' \
                             -e 's/^writeControl .*/writeControl    runTime;/' "$1/system/controlDict"; }
fixed "$W/fof"; ofrun "$W/fof"; fixed "$W/fbr"; brrun "$W/fbr"
stage "$W/cu"; brrun "$W/cu" BRAE_CONTROL_PIMPLE_UNTRIMMED=1
echo "pimple_adjusttimestep_vs_openfoam: pitzDaily (pimpleFoam/RAS) to $END, adjustTimeStep yes, maxCo $MAXCO, writes every $WI"

r=$(replay "$W/of" 0.0001 0 0 | tail -1)
case "$r" in *" 0 not") say "TRANSCRIPTION OpenFOAM's own log, $r" PASS ;;
    *) say "TRANSCRIPTION OpenFOAM's own log does not replay ($r): the transcription is wrong, nothing below is judged" FAIL
       replay "$W/of" 0.0001 0 0 | head -5; exit 1 ;; esac
r=$(replay "$W/br" 0.0001 0 0 | tail -1)
case "$r" in *" 0 not") say "CLOCK     brae's log, OpenFOAM's rules on brae's Courant numbers: $r" PASS ;;
    *) say "CLOCK     brae's log: $r" FAIL; replay "$W/br" 0.0001 0 0 | head -4 ;; esac

nof=$(grep -c '^deltaT = ' "$W/of/log"); nbr=$(grep -c '^deltaT = ' "$W/br/log")
[ "$nof" -eq "$nbr" ] && say "SEQUENCE  $nbr steps, as OpenFOAM" PASS || say "SEQUENCE  brae $nbr steps, OpenFOAM $nof" FAIL
wo=$(written "$W/of"); wb=$(written "$W/br")
[ "$wo" = "$wb" ] && say "SEQUENCE  the same write times: $wb" PASS || say "SEQUENCE  writes: OpenFOAM $wo, brae $wb" FAIL
same=0; tot=0
for t in $wb; do
    tot=$((tot+1)); [ -f "$W/br/$t/uniform/time" ] || { say "SEQUENCE  no uniform/time in brae's $t" FAIL; continue; }
    [ "$(utval "$W/br/$t" index) $(utval "$W/br/$t" deltaT)" = "$(utval "$W/of/$t" index) $(utval "$W/of/$t" deltaT)" ] && same=$((same+1))
done
echo "            uniform/time index and deltaT identical to OpenFOAM's at $same of $tot writes"

python3 - "$W" $wb <<'PY'
import re, sys
W, times = sys.argv[1], sys.argv[2:]
def rd(p):
    s = open(p).read(); m = re.search(r'internalField\s+nonuniform List<(vector|scalar)>\s*(\d+)\s*\((.*?)\n\)\s*;', s, re.S)
    if m.group(1) == 'vector': return [tuple(float(x) for x in t.split()) for t in re.findall(r'\(([^()]*)\)', m.group(3))]
    return [(float(x),) for x in m.group(3).split()]
def d(a, b): return max(max(abs(x - y) for x, y in zip(u, v)) for u, v in zip(a, b)) / max(max(abs(x) for x in u) for u in a)
print('  FIELDS    reported, not gated (max |difference| / max |OpenFOAM|: brae adaptive | OpenFOAM own spread | brae fixed deltaT)')
for t in times:
    row = []
    for f in ('U', 'p', 'k', 'epsilon', 'nut'):
        s, o2, o4, b = (rd(f'{W}/{c}/{t}/{f}') for c in ('of', 'of2', 'of4', 'br'))
        row.append(f'{f} {d(s, b):.1e}|{max(d(s, o2), d(s, o4)):.1e}|{d(rd(f"{W}/fof/{t}/{f}"), rd(f"{W}/fbr/{t}/{f}")):.1e}')
    print(f'            t={t}: ' + '  '.join(row))
PY

# ---- RESTART: from the 0.002 write, four ways ------------------------------------------------------------------
rs() { mkdir -p "$W/$1"; cp -r "$W/base/constant" "$W/base/system" "$W/$1/"; cp -r "$W/$2/$RS" "$W/$1/"; }
rs rbb br; rs rbo of; rs rob br; rs roo of
brrun "$W/rbb"; brrun "$W/rbo"; ofrun "$W/rob"; ofrun "$W/roo"
for c in rbb:br rbo:of rob:br roo:of; do
    d=${c%%:*}; src=${c#*:}
    r=$(replay "$W/$d" "$(utval "$W/$src/$RS" deltaT)" $RS "$(utval "$W/$src/$RS" index)" | tail -1)
    who=$([ "${d:1:1}" = b ] && echo brae || echo OpenFOAM); from=$([ "$src" = br ] && echo "brae's" || echo "OpenFOAM's")
    case "$r" in *" 0 not") say "RESTART   $who from $from $RS (index $(utval "$W/$src/$RS" index)): $r" PASS ;;
        *) say "RESTART   $who from $from $RS: $r" FAIL ;; esac
done
python3 - "$W" <<'PY' && say "RESTART   brae restarted at $RS equals brae continuous at $END to the bit (OpenFOAM's: likewise)" PASS || say "RESTART   brae restarted differs from brae continuous at $END" FAIL
import re, sys
W = sys.argv[1]
same = all(open(f'{W}/br/0.003/{f}').read().split('internalField', 1)[1] == open(f'{W}/rbb/0.003/{f}').read().split('internalField', 1)[1]
           for f in ('U', 'p', 'k', 'epsilon', 'nut', 'phi'))
sys.exit(0 if same else 1)
PY

# ---- FUNCTION OBJECTS: forces at executeControl/writeControl timeStep 3, continuous and restarted --------------------
# OpenFOAM's Time::run() executes the step just solved before ++runTime (Time.C:797, 825): its row carries that step's time,
# and timeStep 3 counts runTime.timeIndex(), which a restart continues from uniform/time. Each code is judged against its
# OWN steps (the Courant numbers, and so the times, are each code's own): every row at a global index divisible by 3, with
# the time that code logged at that index. OpenFOAM's rows are the premise.
foadd()
{
    cat >> "$1/system/controlDict" << 'FO'

functions
{
    forces1
    {
        type            forces;
        libs            (forces);
        patches         (lowerWall upperWall);
        rho             rhoInf;
        rhoInf          1;
        CofR            (0 0 0);
        executeControl  timeStep;
        executeInterval 3;
        writeControl    timeStep;
        writeInterval   3;
    }
}
FO
}
stage "$W/fo-of"; foadd "$W/fo-of"; ofrun "$W/fo-of" || { echo "FAIL: OpenFOAM with forces"; tail -5 "$W/fo-of/log"; exit 1; }
stage "$W/fo-br"; foadd "$W/fo-br"; brrun "$W/fo-br" || { echo "FAIL: brae with forces"; tail -5 "$W/fo-br/log"; exit 1; }
for c in of br; do
    mkdir -p "$W/fo-$c-r"; cp -r "$W/fo-$c/constant" "$W/fo-$c/system" "$W/fo-$c-r/"; cp -r "$W/fo-$c/$RS" "$W/fo-$c-r/"
done
ofrun "$W/fo-of-r"; brrun "$W/fo-br-r"
stage "$W/fo-cf"; foadd "$W/fo-cf"; brrun "$W/fo-cf" BRAE_CONTROL_FO_STEP_ESTIMATED=1
fojudge()   # <case> <start index>: rows at the global indices divisible by 3, each with the time the log gives that index
{
    python3 - "$1" "$2" << 'PY'
import glob
import re
import sys
case, start = sys.argv[1], int(sys.argv[2])
steps = [float(x) for x in re.findall(r'^Time = ([0-9.eE+-]+)', open(case + '/log').read(), re.M)]
want = [(start + i + 1, t) for i, t in enumerate(steps) if (start + i + 1) % 3 == 0]
got = []
for p in sorted(glob.glob(case + '/postProcessing/forces1/*/force.dat')):
    for line in open(p):
        if line.strip() and not line.startswith('#'):
            got.append(float(line.split()[0]))
ok = len(got) == len(want) and all(abs(g - w) <= 1e-5 * abs(w) for g, (_, w) in zip(got, want))
rows = ' '.join('%.6g' % g for g in got[:4]) + (' ...' if len(got) > 4 else '')
idx = ' '.join(str(i) for i, _ in want[:4]) + (' ...' if len(want) > 4 else '')
print('%s|%d rows at %s; indices %s' % ('ok' if ok else 'no', len(got), rows, idx))
PY
}
for c in "fo-of:OpenFOAM continuous:0:premise" "fo-of-r:OpenFOAM from its $RS:$(utval "$W/fo-of/$RS" index):premise" \
         "fo-br:brae continuous:0:FO" "fo-br-r:brae from its $RS:$(utval "$W/fo-br/$RS" index):FO"; do
    IFS=: read -r d who idx tag <<< "$c"
    r=$(fojudge "$W/$d" "$idx")
    [ "${r%%|*}" = ok ] && say "$tag: $who, forces rows on its own steps (${r#*|})" PASS \
                        || say "$tag: $who, forces rows on its own steps (${r#*|})" FAIL
done
r=$(fojudge "$W/fo-cf" 0)
[ "${r%%|*}" = no ] && say "CONTROL   BRAE_CONTROL_FO_STEP_ESTIMATED=1: the rows leave brae's steps (${r#*|})" PASS \
                    || say "CONTROL   BRAE_CONTROL_FO_STEP_ESTIMATED=1 went unseen" FAIL

# ---- CONTROLS ------------------------------------------------------------------------------------------------------
r=$(replay "$W/cf" 0.0001 0 0); case "$r" in *"step 1:"*) say "CONTROL   no setDeltaT before step 1: the replay catches step 1" PASS ;;
    *) say "CONTROL   no setDeltaT before step 1 went unseen: $(echo "$r" | tail -1)" FAIL ;; esac
r=$(replay "$W/cu" 0.0001 0 0 | tail -1); wc=$(written "$W/cu")
case "$r" in *" 0 not") say "CONTROL   untrimmed steps went unseen by the replay" FAIL ;;
    *) [ "$wc" != "$wo" ] && say "CONTROL   untrimmed: the replay catches it ($r); writes land at $wc" PASS \
                          || say "CONTROL   untrimmed: the writes still land on OpenFOAM's times" FAIL ;; esac

[ "$fail" -eq 0 ] && echo "pimple_adjusttimestep_vs_openfoam: PASS" || echo "pimple_adjusttimestep_vs_openfoam: FAIL"
exit "$fail"
