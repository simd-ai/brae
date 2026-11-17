#!/usr/bin/env python3
"""Field-by-field comparison of two OpenFOAM result trees, each either a global case (serial, reconstructPar output,
   brae's single-GPU output) or a decomposed case (processorN/..., decomposePar'd np=k, brae's multi-GPU output).
   A decomposed tree is mapped onto the global mesh through cellProcAddressing / faceProcAddressing
   (oriented surface fields are sign-corrected with the face-flip sign, matrix Upper/Lower are swapped on flipped
   faces), so serial vs np=k, reconstructed vs processorN, and np=k vs np=m of the same case all compare in the
   same global numbering.

   compare_fields.py A B [--times all|last|<t1,t2>] [--fields a,b,..] [--skip REGEX] [--dumps] [--tol X]
                         [--tol-field name=X ...] [--ptfloor F] [--csv FILE]
   Times are matched between trees by nearest value (relative 1e-6), so a run whose adjustTimeStep rounded the last
   digit differently still pairs up.  Rows: field, part (cells / internal faces / boundary), n, max|d|, max|d|/max|A|
   (maxrel; scale = max|A| over internal+boundary of that field), L2(d)/max(L2(A), fscale) (l2rel), pointwise relative error with a floor of --ptfloor*max|A| (ptrel).  Fields that
   exist in both trees are compared; NaN (patches without a stored value) is ignored.  With --tol the exit status is 1
   if any maxrel exceeds it.
   Also: --coherence prints, for a decomposed tree, the largest disagreement between the two sides of every
   processor interface for each field."""
import argparse
import os
import re
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from foamio import Tree


def match_times(ta, tb, sel):
    A = [(v, n) for v, n in ta.times if v > 0]
    B = [(v, n) for v, n in tb.times if v > 0]
    pairs = []
    for v, n in A:
        best = min(B, key=lambda x: abs(x[0] - v), default=None)
        if best is not None and abs(best[0] - v) <= 1e-6 * max(abs(v), 1e-30):
            pairs.append((n, best[1]))
    if sel == "last":
        pairs = pairs[-1:]
    elif sel not in ("all", None):
        want = [float(x) for x in sel.split(",")]
        pairs = [p for p in pairs if any(abs(float(p[0]) - w) <= 1e-6 * abs(w) for w in want)]
    return pairs


def swap_ul(tree_fields):
    """matrix Upper/Lower pairs: on flipped faces the global upper is the local lower."""
    for name in list(tree_fields):
        if name.endswith("Upper") and name[:-5] + "Lower" in tree_fields:
            u, l = tree_fields[name], tree_fields[name[:-5] + "Lower"]
            fl = u.get("flipped")
            if fl is not None and fl.any():
                ui = u["internal"].copy()
                li = l["internal"].copy()
                ui[fl], li[fl] = l["internal"][fl], u["internal"][fl]
                u["internal"], l["internal"] = ui, li


def field_scale(a):
    """largest |A| over the internal and boundary parts: the scale every row of this field is measured against
    (a boundary part that is round-off noise next to the internal part, e.g. phi on a wall, is not 'relative 1')"""
    m = 0.0
    for x in (a["internal"], a["boundary"]):
        if x is not None and x.size:
            y = x[np.isfinite(x).all(axis=1)] if x.ndim > 1 else x[np.isfinite(x)]
            if y.size:
                m = max(m, float((np.sqrt((y * y).sum(axis=1)) if y.ndim > 1 else np.abs(y)).max()))
    return m


def _sibling(tree, tname, loaded, name):
    """a field of the same time that the --fields/--skip selection may not have loaded"""
    if name not in loaded:
        try:
            loaded[name] = tree.load(tname, name) if name in tree.field_names(tname) else None
        except Exception:
            loaded[name] = None
    return loaded[name]


def fold_interface_diag(tree, tname, loaded):
    """In a decomposed matrix dump the cut faces' diagonal share sits in the processor patches' internalCoeffs, not in
    diag (lduMatrix adds it back in Amul/solve).  Add it back so Diag compares with the serial matrix's diag."""
    if tree.kind != "decomposed":
        return
    for name in [n for n in loaded if n.endswith("Diag.dump") and loaded[n] is not None]:
        ic = _sibling(tree, tname, loaded, name[:-len("Diag.dump")] + "IntCoeffs.dump")
        if ic is None or not ic.get("proc"):
            continue
        d = loaded[name]["internal"].copy()
        # d is in GLOBAL cell order: scatter through the rank's own cellProcAddressing
        for (ri, pname, nb, gidx, vals) in ic["proc"]:
            cells = tree.patch_cells(ri, pname)
            np.add.at(d, tree.rank[ri]["cellmap"][cells], vals)
        loaded[name]["internal"] = d


def iface_rows(tree, tname, loaded, other, other_tname, other_loaded, rows_out):
    """|processor-patch BouCoeffs/IntCoeffs| of a decomposed matrix dump against |Upper| of the other (global) tree
    at the same cut faces."""
    if tree.kind != "decomposed":
        return
    for dn in [n for n in loaded if n.endswith("Diag.dump")]:
        base = dn[:-len("Diag.dump")]
        up = _sibling(other, other_tname, other_loaded, base + "Upper.dump")
        if up is None:
            continue
        for suf in ("BouCoeffs.dump", "IntCoeffs.dump"):
            f = _sibling(tree, tname, loaded, base + suf)
            if f is None or not f.get("proc"):
                continue
            g = np.concatenate([p[3] for p in f["proc"]])
            v = np.concatenate([np.abs(p[4]) for p in f["proc"]])
            u = np.abs(up["internal"][g])
            ok = np.isfinite(u)
            if ok.any():
                sc = max(float(np.nanmax(np.abs(up["internal"]))), 1e-300)
                dd = np.abs(v[ok] - u[ok])
                rows_out.append((base + suf[:-5] + " |proc| vs |Upper|", "cut faces", int(ok.sum()), float(dd.max()),
                                 float(dd.max()) / sc, float(np.sqrt((dd ** 2).sum()) / max(np.sqrt((u[ok] ** 2).sum()), sc)),
                                 float((dd / np.maximum(np.maximum(u[ok], v[ok]), 1e-3 * sc)).max()), ""))


def diff_rows(name, a, b, ptfloor, rows_out):
    fscale = field_scale(a)
    for part, xa, xb in (("internal", a["internal"], b["internal"]), ("boundary", a["boundary"], b["boundary"])):
        if xa is None or xb is None or xa.shape != xb.shape:
            rows_out.append((name, part, -1, np.nan, np.nan, np.nan, np.nan, "shape %s vs %s" % (
                None if xa is None else xa.shape, None if xb is None else xb.shape)))
            continue
        d = xa - xb
        ok = np.isfinite(xa) & np.isfinite(xb)
        if xa.ndim > 1:
            ok = ok.all(axis=1)
        if not ok.any():
            continue
        xa, xb, d = xa[ok], xb[ok], d[ok]
        mag = (lambda z: np.sqrt((z * z).sum(axis=1))) if xa.ndim > 1 else np.abs
        dm, am = mag(d), mag(xa)
        amax = am.max() if am.size else 0.0
        maxabs = dm.max()
        scale = fscale if fscale > 0 else 1.0
        l2a = np.sqrt((am ** 2).sum())
        pt = (dm / np.maximum(np.maximum(am, mag(xb)), ptfloor * scale)).max()
        rows_out.append((name, part, int(ok.sum()), maxabs, maxabs / scale,
                         np.sqrt((dm ** 2).sum()) / max(l2a, fscale) if max(l2a, fscale) > 0 else 0.0, pt, ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--times", default="all")
    ap.add_argument("--fields", default=None)
    ap.add_argument("--skip", default=None)
    ap.add_argument("--dumps", action="store_true", help="also compare the *.dump / o*.dump operator fields")
    ap.add_argument("--tol", type=float, default=None)
    ap.add_argument("--tol-field", action="append", default=[])
    ap.add_argument("--ptfloor", type=float, default=1e-3)
    ap.add_argument("--coherence", action="store_true")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args(argv)
    ta, tb = Tree(args.a), Tree(args.b)
    pairs = match_times(ta, tb, args.times)
    if not pairs:
        print("no common times between %s %s and %s %s" % (args.a, [n for _, n in ta.times],
                                                          args.b, [n for _, n in tb.times]), file=sys.stderr)
        return 2
    want = None if args.fields is None else set(args.fields.split(","))
    tolf = dict(x.split("=") for x in args.tol_field)
    skip = re.compile(args.skip) if args.skip else None
    bad = []
    csvrows = []
    print("| time A | time B | field | part | n | max abs | maxrel | l2rel | ptrel | note |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for na, nb in pairs:
        fa, fb = set(ta.field_names(na)), set(tb.field_names(nb))
        names = sorted(fa & fb)
        if want is not None:
            names = [n for n in names if n in want]
        else:
            names = [n for n in names if args.dumps or not n.endswith(".dump")]
        if skip:
            names = [n for n in names if not skip.search(n)]
        if want is None and not args.dumps:
            names = [n for n in names if n not in ("uniform", "polyMesh", "cellDist")]
        loaded_a, loaded_b = {}, {}
        for n in names:
            try:
                loaded_a[n] = ta.load(na, n)
                loaded_b[n] = tb.load(nb, n)
            except Exception as e:      # e.g. a dictionary like uniform/ or an unsupported class
                print("| %s | %s | %s | - | - | - | - | - | - | skipped: %s |" % (na, nb, n, type(e).__name__))
                loaded_a.pop(n, None)
                loaded_b.pop(n, None)
        swap_ul(loaded_a)
        swap_ul(loaded_b)
        fold_interface_diag(ta, na, loaded_a)
        fold_interface_diag(tb, nb, loaded_b)
        extra = []
        iface_rows(ta, na, loaded_a, tb, nb, loaded_b, extra)
        iface_rows(tb, nb, loaded_b, ta, na, loaded_a, extra)
        for (nm, part, cnt, mabs, mrel, l2, pt, note) in extra:
            print("| %s | %s | %s | %s | %d | %.3e | %.3e | %.3e | %.3e | %s |" % (na, nb, nm, part, cnt, mabs, mrel, l2, pt, note))
        for n in sorted(k for k in loaded_a if k in names):
            rows = []
            diff_rows(n, loaded_a[n], loaded_b[n], args.ptfloor, rows)
            for (nm, part, cnt, mabs, mrel, l2, pt, note) in rows:
                print("| %s | %s | %s | %s | %d | %.3e | %.3e | %.3e | %.3e | %s |" % (na, nb, nm, part, cnt, mabs, mrel, l2, pt, note))
                csvrows.append((na, nb, nm, part, cnt, mabs, mrel, l2, pt))
                lim = float(tolf.get(nm, args.tol)) if (nm in tolf or args.tol is not None) else None
                if lim is not None and mrel > lim:
                    bad.append((na, nm, part, mrel))
            if args.coherence:
                for side, L in (("A", loaded_a[n]), ("B", loaded_b[n])):
                    pc = L.get("proc_check")
                    if pc and pc[1]:
                        print("|  |  | %s | coherence(%s) | %d | %.3e |  |  |  | two sides of every processor interface |" % (n, side, pc[1], pc[0]))
    if args.csv:
        with open(args.csv, "w") as f:
            f.write("timeA,timeB,field,part,n,maxabs,maxrel,l2rel,ptrel\n")
            for r in csvrows:
                f.write(",".join(str(x) for x in r) + "\n")
    if bad:
        print("\nEXCEEDS TOLERANCE: " + "; ".join("%s %s %s maxrel %.3e" % x for x in bad))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
