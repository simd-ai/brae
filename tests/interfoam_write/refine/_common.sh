# Shared by the files of this folder. Y: a REFINING mesh (dynamicRefineFvMesh) beyond the tutorial rows'
# two steps.
#   Y1  alpha.water_0 is alpha MAPPED with the mesh (MapGeometricFields, then subCycle's restore): at the second
#       write its first nOld cells are the first write's alpha, and each refined parent's seven added cells
#       copy the parent -- asserted of OpenFOAM's own output and of brae's, on every refining case and arm
#       arm W ran. By rule and not by bytes: on oscillatingBox, which also moves, brae's alpha already
#       differs from OpenFOAM's by 3.3e-16 and the mapped level carries that.
#   Y2  CONTROLS on damBreakWithObstacle: BRAE_CONTROL_AMR_NO_WRITE_COMPACT=1 (the history written without the
#       compaction refinementHistory's operator<< does) fails 0.002/polyMesh/refinementHistory;
#       BRAE_CONTROL_AMR_ALPHA0_START=1 (alpha_0 as the start-of-step copy, on the old mesh) fails
#       alpha.water_0.
#   Y3  UNREFINEMENT, which no two-step write witnesses: laminar/oscillatingBox for sixty steps of 1e-3, where
#       OpenFOAM unrefines 9540 -> 9400 at 0.057, every step written and compared -- merged cells, freed history
#       entries compacted at the write, points0 losing points. MEASURED (2026-09-30): structure 0 over all
#       sixty, brae unrefining the same 20 split points at 0.057, worst field 1.6e-12.
#   Y4  A WRITE BEFORE THE FIRST CHANGE, then a restart from it: damBreakWithObstacle with refineInterval 2.
#       Step 1 refines nothing, and OpenFOAM writes hexRef8's files alone, in the uniform forms `32256{0}`,
#       `0()`, `32256{-1}`, and no polyMesh topology. Both codes then restart from OpenFOAM's 0.001 and refine
#       at step 2. FAIL-PROOF (2026-09-30): brae's schedule tested the step count of the run, not the global
#       time index (dynamicRefineFvMesh.C:1320), and refined nothing at step 2 -- U 2.2e-01 off. This restart
#       does NOT read the uniform lists back: with no change the faces instance is constant/, which is where
#       hexRef8 reads (hexRef8.C:1912-1990), so the N{v} readers are matched to ListIO.C in source only.
amrule()   # amrule <case> -- Y1's mapping rule on a case's first two writes
{
    python3 - "$1" <<'EOF_Y1'
import os, re, sys
d = sys.argv[1]
ts = sorted([t for t in os.listdir(d) if re.match(r'^[0-9.e+-]+$', t) and t != '0'], key=float)
def cells(p):
    t = open(p).read()
    b = t[t.find('internalField'):t.find('boundaryField')]
    m = re.search(r'List<scalar>\s*\n?(\d+)\s*\n\(\n(.*?)\n\)', b, re.S)
    return m.group(2).split('\n') if m else None
def level(p):
    t = open(p).read()
    t = t[t.find('// * * *'):]
    m = re.search(r'\n(\d+)\n\((.*?)\n\)', t, re.S)
    return [int(x) for x in m.group(2).split()]
a1, a0 = cells('%s/%s/alpha.water' % (d, ts[0])), cells('%s/%s/alpha.water_0' % (d, ts[1]))
l1, l2 = level('%s/%s/polyMesh/cellLevel' % (d, ts[0])), level('%s/%s/polyMesh/cellLevel' % (d, ts[1]))
n = len(a1)
parents = [c for c in range(n) if l2[c] > l1[c]]
kept = a0[:n] == a1
kids = len(a0) == n + 7 * len(parents) and all(a0[n + 7 * k + j] == a1[p] for k, p in enumerate(parents) for j in range(7))
print('      %s: %d cells kept %s, %d parents with seven children each %s' % (d.rsplit('/', 1)[-1], n, kept, len(parents), kids))
sys.exit(0 if kept and kids and parents else 1)
EOF_Y1
}
