#!/usr/bin/env python3
"""Tables for REPORT: decomposition statistics and serial-vs-parallel error floors of every run under a directory.
   summarize.py <runsDir>"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from foamio import Tree
from compare_fields import match_times, diff_rows

FIELDS = ("alpha.water", "p_rgh", "U", "phi")


def floors(a, b):
    ta, tb = Tree(a), Tree(b)
    out = {}
    for na, nb in match_times(ta, tb, "all"):
        row = {}
        for f in FIELDS:
            rows = []
            diff_rows(f, ta.load(na, f), tb.load(nb, f), 1e-3, rows)
            row[f] = max(r[4] for r in rows if r[1] == "internal" or r[1] == "boundary")
        out[na] = row
    return out


def main(root):
    print("| run | cells | cut faces | max iface faces/rank | max iface/cells | neighbours/rank (min-max) | "
          "step | alpha | p_rgh | U | phi |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for run in sorted(os.listdir(root)):
        d = os.path.join(root, run)
        if not os.path.isdir(os.path.join(d, "par")):
            continue
        t = Tree(os.path.join(d, "par"))
        ifs = [sum(p["nFaces"] for p in r["patches"] if p["type"] == "processor") for r in t.rank]
        nbs = [sum(1 for p in r["patches"] if p["type"] == "processor") for r in t.rank]
        cells = [r["n_cells"] for r in t.rank]
        first = "%s | %d | %d | %d | %.1f%% | %d-%d" % (run, t.n_cells, sum(ifs) // 2, max(ifs),
                                                        100 * max(i / c for i, c in zip(ifs, cells)), min(nbs), max(nbs))
        try:
            f1 = floors(os.path.join(d, "serial_d1"), os.path.join(d, "par_d1"))
            f20 = floors(os.path.join(d, "serial"), os.path.join(d, "par"))
        except Exception as e:
            print("| %s | compare failed: %s |" % (first, e))
            continue
        for lab, fl in (("1", f1), ("20", f20)):
            k = sorted(fl, key=float)[-1]
            print("| %s | %s | %s |" % (first if lab == "1" else "", lab,
                                        " | ".join("%.1e" % fl[k][f] for f in FIELDS)))


if __name__ == "__main__":
    main(sys.argv[1])
