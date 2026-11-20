#!/usr/bin/env bash
# tools/tutorial_sweep.py on a tutorials tree whose every outcome is known, with a fake brae, fake OpenFOAM utilities
# and fake OpenFOAM solvers that leave a mark if they ever run. Needs neither OpenFOAM nor a GPU.
#   1. a case with no scripts: blockMesh, then `brae check --setup`: set up, with the memory brae reports
#   2. an Allrun that runs potentialFoam, then the solver under mpirun -parallel on a mesh that exists only in
#      processor0/: the stub stands in for potentialFoam, only rank 0 checks, the mesh is reconstructed first, and
#      brae's refusal is recorded with its text
#   3. an Allrun that composes its case from setups.orig/ (one tutorial, its case found where the solver ran):
#      `--setup: not run` is recorded as checked
#   4. an Allrun that never reaches the solver: not reached; one that hangs: timeout; one that names snappyHexMesh
#      under --skip-mesh-if snappyHexMesh: skipped, nothing run; an application not swept: absent
#   5. NO OPENFOAM SOLVER RAN: neither fake solver left its mark, and the stub names potentialFoam
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOOL="${1:-$ROOT/tools/tutorial_sweep.py}"
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
fail=0
say() { printf '  %-90s %s\n' "$1" "$2"; [ "$2" = FAIL ] && fail=1; return 0; }

T="$W/tut/fam"
B="$W/bin"
A="$W/appbin"
mkdir -p "$B" "$A"
dict() { mkdir -p "$1/system"; printf 'FoamFile { object controlDict; }\napplication     %s;\n' "$2" > "$1/system/controlDict"; }
exe() { cat > "$1"; chmod +x "$1"; }

# the fake brae: its answer is chosen by the case's name
exe "$B/brae" <<'EOF'
#!/usr/bin/env bash
c=.; while [ $# -gt 0 ]; do [ "$1" = -case ] && c=$2; shift; done
case $(basename "$(cd "$c" && pwd)") in
blockOnly) echo "verdict: SUPPORTED"
           echo "brae check --setup: simpleFoam set the case up completely (dictionaries, mesh, fields, boundary conditions, device arrays); GPU memory in use 1.25 of 119.70 GiB; no step taken"
           exit 0 ;;
allrunCase) echo "  [refused] boundary    0/U patch inlet: type fooBar is not ported"
            echo "verdict: REFUSED -- brae would stop on this case; nothing was run"
            exit 2 ;;
run1) echo "  [notice ] functions   one function object is not run"
      echo "verdict: SUPPORTED WITH NOTICES -- brae runs the case and says what it runs differently"
      echo "--setup: not run: interFoam on this path does not stop before its first step yet, so setting it up would run it"
      exit 1 ;;
*) echo "unexpected case $c"; exit 9 ;;
esac
EOF
exe "$B/blockMesh" <<'EOF'
#!/usr/bin/env bash
mkdir -p constant/polyMesh && echo mesh > constant/polyMesh/owner
EOF
exe "$B/mpirun" <<'EOF'
#!/usr/bin/env bash
[ "$1" = -np ] && { n=$2; shift 2; }
for r in $(seq 0 $((n - 1))); do OMPI_COMM_WORLD_RANK=$r "$@" || exit; done
EOF
exe "$B/reconstructParMesh" <<'EOF'
#!/usr/bin/env bash
mkdir -p constant/polyMesh && echo reconstructed > constant/polyMesh/owner
EOF
exe "$B/reconstructPar" <<'EOF'
#!/usr/bin/env bash
touch reconstructed.fields
EOF
# OpenFOAM's own solvers, which must never run
for s in simpleFoam interFoam potentialFoam icoFoam
do
    exe "$A/$s" <<EOF
#!/usr/bin/env bash
touch "\$PWD/RAN_$s"
EOF
done

dict "$T/blockOnly" simpleFoam
touch "$T/blockOnly/system/blockMeshDict"

dict "$T/allrunCase" simpleFoam
exe "$T/allrunCase/Allrun" <<'EOF'
#!/bin/sh
cd "${0%/*}" || exit
blockMesh > log.blockMesh 2>&1
mkdir -p processor0/constant/polyMesh processor1/constant/polyMesh
sleep 1
echo decomposed > processor0/constant/polyMesh/owner
potentialFoam > log.potentialFoam 2>&1
mpirun -np 2 simpleFoam -parallel > log.simpleFoam 2>&1
EOF

mkdir -p "$T/setups"
dict "$T/setups/setups.orig/common" interFoam
exe "$T/setups/Allrun" <<'EOF'
#!/bin/sh
cd "${0%/*}" || exit
cp -r setups.orig/common run1
( cd run1 && blockMesh && interFoam > log.interFoam 2>&1 )
EOF

dict "$T/neverRuns" simpleFoam
exe "$T/neverRuns/Allrun" <<'EOF'
#!/bin/sh
cd "${0%/*}" || exit
blockMesh > log.blockMesh 2>&1
EOF

dict "$T/hangs" simpleFoam
exe "$T/hangs/Allrun" <<'EOF'
#!/bin/sh
sleep 60
EOF

dict "$T/snappy" simpleFoam
exe "$T/snappy/Allrun" <<'EOF'
#!/bin/sh
cd "${0%/*}" || exit
snappyHexMesh -overwrite > log.snappyHexMesh 2>&1
simpleFoam > log.simpleFoam 2>&1
EOF

dict "$T/other" icoFoam

( cd "$W" && env PATH="$B:$A:$PATH" FOAM_APPBIN="$A" python3 "$TOOL" --solvers simpleFoam,interFoam \
    --tutorials "$W/tut" --work "$W/work" --brae "$B/brae" --ofbashrc none --skip-mesh-if snappyHexMesh \
    --timeout 3 --jobs 3 --keep > "$W/sweep.log" 2>&1 )
rc=$?
sed 's/^/    /' "$W/sweep.log"
[ $rc = 0 ] && say "the sweep ran to its end" ok || say "the sweep ran to its end (exit $rc)" FAIL
[ -f "$W/work/tutorial_sweep.csv" ] && [ -f "$W/work/tutorial_sweep.json" ] && say "it wrote the CSV and the JSON" ok \
    || say "it wrote the CSV and the JSON" FAIL

python3 - "$W/work/tutorial_sweep.json" "$W/work/tutorial_sweep.csv" <<'PY' || fail=1
import csv, json, sys
j = json.load(open(sys.argv[1]))
rows = {(r['tutorial'], r['case']): r for r in j['rows']}
bad = 0
def check(ok, what):
    global bad
    print('  %-90s %s' % (what, 'ok' if ok else 'FAIL'))
    bad += 0 if ok else 1
r = rows.get(('fam/blockOnly', 'fam/blockOnly'))
check(r and r['script'] == 'blockMesh' and r['mesh'] == 'ok' and r['verdict'] == 'set up'
      and r['memory_gib'] == 1.25 and r['memory_total_gib'] == 119.7, '1. blockMesh, then set up with 1.25 of 119.70 GiB')
r = [v for k, v in rows.items() if k[0] == 'fam/allrunCase']
r = r[0] if len(r) == 1 else None
check(r and r['script'] == 'Allrun' and r['verdict'] == 'refused' and 'type fooBar is not ported' in r['refusal']
      and r['parallel'] is True, '2. Allrun, mpirun -parallel, one check (rank 0), refused with its text')
check(r and 'reconstructed' in r['notes'] and r['mesh'] == 'ok', '2. the processor0-only mesh was reconstructed first')
check(r and r['stubbed'] == 'potentialFoam', '2. potentialFoam was stubbed: %s' % (r and r['stubbed']))
r = [v for k, v in rows.items() if k[0] == 'fam/setups']
check(len(r) == 1 and r[0]['case'] == 'fam/setups/run1' and r[0]['verdict'] == 'checked' and 'setup not run' in r[0]['notes']
      and r[0]['solver'] == 'interFoam', '3. setups.orig composed by the Allrun: case run1, checked')
r = [v for k, v in rows.items() if k[0] == 'fam/neverRuns']
check(len(r) == 1 and r[0]['verdict'] == 'not reached' and r[0]['mesh'] == 'ok', '4. not reached')
r = [v for k, v in rows.items() if k[0] == 'fam/hangs']
check(len(r) == 1 and r[0]['verdict'] == 'timeout', '4. timeout')
r = [v for k, v in rows.items() if k[0] == 'fam/snappy']
check(len(r) == 1 and r[0]['verdict'] == 'skipped' and r[0]['mesh'] == 'skipped', '4. skipped by --skip-mesh-if')
check(not any(k[0] == 'fam/other' for k in rows), '4. icoFoam is not swept')
c = list(csv.DictReader(open(sys.argv[2])))
check(len(c) == len(j['rows']) and set(c[0]) >= {'solver', 'tutorial', 'mesh', 'verdict', 'refusal', 'memory_gib',
      'wall_s'}, 'the CSV holds every row and the columns')
sys.exit(1 if bad else 0)
PY
ran=$(find "$W/work" -name 'RAN_*' | sed "s|$W/||")
[ -z "$ran" ] && say "5. no OpenFOAM solver ran" ok || say "5. no OpenFOAM solver ran: $ran" FAIL
[ ! -e "$W/work/fam/snappy" ] && say "5. nothing was copied or run for the skipped tutorial" ok \
    || say "5. nothing was copied or run for the skipped tutorial" FAIL

[ $fail = 0 ] && echo "PASS: tutorial_sweep.py records what brae says of each tutorial and runs no OpenFOAM solver" || echo "FAIL"
exit $fail
