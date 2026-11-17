#!/usr/bin/env python3
"""Decomposition table of a decomposePar'd case: cells, interface (processor-patch) faces and neighbour ranks per rank.
   decomp_stats.py <caseDir>      (markdown on stdout)"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from foamio import Tree


def table(root):
    t = Tree(root)
    rows = []
    for i, r in enumerate(t.rank):
        pp = [p for p in r["patches"] if p["type"] in ("processor", "processorCyclic")]
        rows.append((i, r["n_cells"], sum(p["nFaces"] for p in pp), len(pp),
                     " ".join("%d:%d" % (p["neighbProcNo"], p["nFaces"]) for p in pp)))
    out = ["| rank | cells | interface faces | neighbours | neighbour:faces |", "|---|---|---|---|---|"]
    for row in rows:
        out.append("| %d | %d | %d | %d | %s |" % row)
    cells = [r[1] for r in rows]
    ifs = [r[2] for r in rows]
    nbs = [r[3] for r in rows]
    out.append("")
    out.append("np=%d: cells %d (min %d / max %d, imbalance %.1f%%); cut faces %d; interface faces/rank min %d max %d "
               "(max/cells %.1f%%); neighbours/rank min %d max %d"
               % (len(rows), sum(cells), min(cells), max(cells),
                  100.0 * (max(cells) / (sum(cells) / len(cells)) - 1), sum(ifs) // 2, min(ifs), max(ifs),
                  100.0 * max(i / c for i, c in zip(ifs, cells)), min(nbs), max(nbs)))
    return "\n".join(out)


if __name__ == "__main__":
    print("## decomposition: " + os.path.basename(os.path.abspath(sys.argv[1])))
    print(table(sys.argv[1]))
