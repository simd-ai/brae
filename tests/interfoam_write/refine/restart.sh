#!/usr/bin/env bash
# The write gate: a write before the first change and a restart from it (Y4), and from a compressed write (Y5).
. "$(dirname "$0")/../lib.sh"
. "$(dirname "$0")/_common.sh"
wcase damBreakWithObstacle of
if [ -d "$W/w_of_damBreakWithObstacle" ]; then
    for side in of br; do
        d="$W/y4_$side"
        mkdir -p "$d"
        cp -r "$W/w_of_damBreakWithObstacle/0" "$W/w_of_damBreakWithObstacle/constant" "$W/w_of_damBreakWithObstacle/system" "$d/"
        sed -i -E 's/^(refineInterval\s+)[^;]*;/\12;/' "$d/constant/dynamicMeshDict"
        first=$(timedirs "$W/w_of_damBreakWithObstacle" | awk '{print $1}')
        sed -i -E "s/^(endTime\s+)[^;]*;/\1$first;/" "$d/system/controlDict"
    done
    grep -qE "^refineInterval +2;" "$W/y4_of/constant/dynamicMeshDict" || say "ARM Y4 refineInterval was not staged" FAIL
    runof "$W/y4_of"
    runbrae "$W/y4_br" host
    first=$(timedirs "$W/y4_of" | awk '{print $1}')
    [ ! -f "$W/y4_of/$first/polyMesh/faces" ] && grep -q "{0}" "$W/y4_of/$first/polyMesh/cellLevel" \
        && say "fixture witnesses: OpenFOAM's write before any change holds no topology and uniform levels" ok \
        || say "fixture witnesses: OpenFOAM's write before any change holds no topology and uniform levels" FAIL
    python3 "$CMP" "$W/y4_of" "$W/y4_br" $first > "$W/cmp_y4a.txt" 2>&1
    judge "damBreakWithObstacle before a change" "$W/cmp_y4a.txt" 2e-11 "$W/y4_of/log.interFoam" \
        && say "ARM Y4 [host] a write before any change: OpenFOAM's file set, uniform levels and history" ok \
        || say "ARM Y4 [host] a write before any change: OpenFOAM's file set, uniform levels and history" FAIL
    second=$(python3 -c "print('%.10g' % (2*float('$first')))")
    for side in of2 br2; do
        d="$W/y4_$side"
        mkdir -p "$d"
        cp -r "$W/y4_of/constant" "$W/y4_of/system" "$W/y4_of/$first" "$d/"
        sed -i -E "s/^(startFrom\s+)[^;]*;/\1latestTime;/; s/^(endTime\s+)[^;]*;/\1$second;/" "$d/system/controlDict"
    done
    runof "$W/y4_of2"
    runbrae "$W/y4_br2" host
    grep -q "Refined from" "$W/y4_of2/log.interFoam" \
        && say "fixture witnesses: OpenFOAM refines at the restarted run's step 2 (global index 2)" ok \
        || say "fixture witnesses: OpenFOAM refines at the restarted run's step 2 (global index 2)" FAIL
    python3 "$CMP" "$W/y4_of2" "$W/y4_br2" $second > "$W/cmp_y4b.txt" 2>&1
    judge "damBreakWithObstacle restart" "$W/cmp_y4b.txt" 2e-11 "$W/y4_of2/log.interFoam" \
        && say "ARM Y4 [host] restarted from OpenFOAM's write, brae refines at the global index OpenFOAM does" ok \
        || { say "ARM Y4 [host] restarted from OpenFOAM's write, brae refines at the global index OpenFOAM does" FAIL; grep -v RESULT "$W/cmp_y4b.txt" | grep BAD | head -4; }
fi

#   Y5  A RESTART FROM A COMPRESSED REFINED WRITE: damBreakWithObstacle with `writeCompression on` for one
#       step (OpenFOAM writes polyMesh/*.gz, cellLevel.gz, refinementHistory.gz, Uf.gz beside a refined mesh),
#       then both codes restart from OpenFOAM's directory. FAIL-PROOFS (2026-09-30): the refinement state
#       probed on its plain path alone put cellLevel, pointLevel and refinementHistory off OpenFOAM's; Uf and
#       phi probed the same way fell back to the interpolation and put U 3.2e-01 off -- a restart defect
#       shared by every solver that reads its flux through read_surface_field.cuh.
if [ -d "$W/w_of_damBreakWithObstacle" ]; then
    d="$W/y5_of"
    mkdir -p "$d"
    cp -r "$W/w_of_damBreakWithObstacle/0" "$W/w_of_damBreakWithObstacle/constant" "$W/w_of_damBreakWithObstacle/system" "$d/"
    first=$(timedirs "$W/w_of_damBreakWithObstacle" | awk '{print $1}')
    second=$(python3 -c "print('%.10g' % (2*float('$first')))")
    sed -i -E "s/^(writeCompression\s+)[^;]*;/\1on;/; s/^(endTime\s+)[^;]*;/\1$first;/" "$d/system/controlDict"
    runof "$d"
    [ -f "$d/$first/polyMesh/cellLevel.gz" ] && [ -f "$d/$first/Uf.gz" ] \
        && say "fixture witnesses: OpenFOAM's compressed write holds cellLevel.gz and Uf.gz beside a refined mesh" ok \
        || say "fixture witnesses: OpenFOAM's compressed write holds cellLevel.gz and Uf.gz beside a refined mesh" FAIL
    for side in of2 br2; do
        e="$W/y5_$side"
        mkdir -p "$e"
        cp -r "$d/constant" "$d/system" "$d/$first" "$e/"
        sed -i -E "s/^(startFrom\s+)[^;]*;/\1latestTime;/; s/^(endTime\s+)[^;]*;/\1$second;/; s/^(writeCompression\s+)[^;]*;/\1off;/" "$e/system/controlDict"
    done
    runof "$W/y5_of2"
    runbrae "$W/y5_br2" host
    python3 "$CMP" "$W/y5_of2" "$W/y5_br2" $second > "$W/cmp_y5.txt" 2>&1
    judge "damBreakWithObstacle compressed restart" "$W/cmp_y5.txt" 2e-11 "$W/y5_of2/log.interFoam" \
        && say "ARM Y5 [host] restarted from a compressed refined write: OpenFOAM's levels, history and flux" ok \
        || { say "ARM Y5 [host] restarted from a compressed refined write: OpenFOAM's levels, history and flux" FAIL; grep BAD "$W/cmp_y5.txt" | head -4; }
fi
finish "arms Y4, Y5: a refined mesh restarts as OpenFOAM's does"
