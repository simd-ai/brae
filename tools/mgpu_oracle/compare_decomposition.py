#!/usr/bin/env python3
"""File-by-file comparison of two decomposed cases (processor<N>/ directories): OpenFOAM's decomposePar against `brae decompose`,
   or the other way round for the fields of reconstructPar against `brae reconstruct`.

   compare_decomposition.py mesh   A B           every processor's constant/polyMesh: points, faces, owner, neighbour, boundary and the
                                                 four *ProcAddressing files, as numbers (ASCII; label=32)
   compare_decomposition.py fields A B TIME [--vtol X]
                                                 every field file of processor*/TIME: internalField and each patch's type and value,
                                                 processor patches included, to the last bit. --vtol X lets the VALUE of a physical
                                                 patch differ by X of the field's largest value: OpenFOAM re-evaluates a
                                                 fixedFluxPressure patch's value on reading it (1e-14 of the field, measured), where
                                                 brae copies it.
   compare_decomposition.py global A B TIME      every field file of the global TIME directories of two cases

   Exit 1 on the first class of difference, with what differed. A and B are case directories."""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from foamio import read_boundary, read_field, read_label_list, mesh_counts


def procs(root):
    out = []
    while os.path.isdir(os.path.join(root, f"processor{len(out)}")):
        out.append(os.path.join(root, f"processor{len(out)}"))
    return out


def numbers(path):
    """every number of a file's body (after the FoamFile header and the banner line), the list count included"""
    text = open(path, "rb").read().decode("latin-1")
    text = text[text.index("// * * *"):]
    text = re.sub(r"//[^\n]*", "", text)
    return np.array([float(x) for x in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", text)])


def same(a, b, tol=0.0):
    if a.shape != b.shape:
        return False
    if tol == 0.0:
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.all(np.abs(a - b) <= tol * np.maximum(1.0, np.abs(a))))


def compare_mesh(A, B):
    pa, pb = procs(A), procs(B)
    bad = []
    if len(pa) != len(pb):
        print(f"FAIL: {len(pa)} processors against {len(pb)}")
        return 1
    for k, (a, b) in enumerate(zip(pa, pb)):
        da, db = os.path.join(a, "constant/polyMesh"), os.path.join(b, "constant/polyMesh")
        for f in ("faces", "owner", "neighbour", "cellProcAddressing", "faceProcAddressing", "pointProcAddressing",
                  "boundaryProcAddressing"):
            if not same(numbers(os.path.join(da, f)), numbers(os.path.join(db, f))):
                bad.append(f"processor{k}/{f}")
        # points to the last digit OpenFOAM wrote (it prints with writePrecision digits)
        if not same(numbers(os.path.join(da, "points")), numbers(os.path.join(db, "points")), 1e-14):
            bad.append(f"processor{k}/points")
        ba, bb = read_boundary(os.path.join(da, "boundary")), read_boundary(os.path.join(db, "boundary"))
        if ba != bb:
            bad.append(f"processor{k}/boundary: {ba} vs {bb}")
        if mesh_counts(da) != mesh_counts(db):
            bad.append(f"processor{k} counts {mesh_counts(da)} vs {mesh_counts(db)}")
    if bad:
        print("FAIL: the processor meshes differ:\n  " + "\n  ".join(bad[:20]))
        return 1
    print(f"ok: {len(pa)} processor meshes identical")
    return 0


def field(path, polydir):
    nC, nP, nF, nI = mesh_counts(polydir)
    sizes = {p["name"]: p["nFaces"] for p in read_boundary(os.path.join(polydir, "boundary"))}
    cls = re.search(rb"class\s+(\w+);", open(path, "rb").read(4000)).group(1).decode()
    return read_field(path, sizes, nI if cls.startswith("surface") else nC)


def eq(x, y):
    if x is None or y is None:
        return x is None and y is None
    return x.shape == y.shape and np.array_equal(x, y, equal_nan=True)


def close(x, y, vtol, scale):
    if x is None or y is None or x.shape != y.shape:
        return eq(x, y)
    return bool(np.all(np.isnan(x) == np.isnan(y))) and bool(np.nanmax(np.abs(x - y), initial=0.0) <= vtol * scale)


def compare_field_dirs(da, db, polya, polyb, label, vtol=0.0):
    fa = sorted(f for f in os.listdir(da) if os.path.isfile(os.path.join(da, f)) and not f.startswith("."))
    fb = sorted(f for f in os.listdir(db) if os.path.isfile(os.path.join(db, f)) and not f.startswith("."))
    bad = []
    # files that are not volume/surface fields (cellLevel, pointLevel) are not compared
    def is_field(path):
        m = re.search(rb"class\s+(\w+);", open(path, "rb").read(4000))
        return bool(m) and (m.group(1).startswith(b"vol") or m.group(1).startswith(b"surface")) and m.group(1).endswith(b"Field")
    fa = [f for f in fa if is_field(os.path.join(da, f))]
    fb = [f for f in fb if is_field(os.path.join(db, f))]
    if fa != fb:
        bad.append(f"{label}: fields {fa} vs {fb}")
    n = 0
    for f in sorted(set(fa) & set(fb)):
        n += 1
        A, B = field(os.path.join(da, f), polya), field(os.path.join(db, f), polyb)
        if A.cls != B.cls or A.oriented != B.oriented:
            bad.append(f"{label}/{f}: class/oriented {A.cls},{A.oriented} vs {B.cls},{B.oriented}")
            continue
        if not eq(A.internal, B.internal):
            d = "" if A.internal is None or B.internal is None or A.internal.shape != B.internal.shape else \
                f" (max |d| {np.nanmax(np.abs(A.internal - B.internal)):.3e})"
            bad.append(f"{label}/{f}: internalField differs{d}")
        if sorted(A.boundary) != sorted(B.boundary):
            bad.append(f"{label}/{f}: patches {sorted(A.boundary)} vs {sorted(B.boundary)}")
            continue
        for p in A.boundary:
            ta, va, na = A.boundary[p]
            tb, vb, nb = B.boundary[p]
            if ta != tb:
                bad.append(f"{label}/{f}/{p}: type {ta} vs {tb}")
            elif not (eq(va, vb) or (tb != "processor" and close(va, vb, vtol, max(1.0, float(np.nanmax(np.abs(A.internal))))))):
                d = "" if va is None or vb is None or va.shape != vb.shape else f" (max |d| {np.nanmax(np.abs(va - vb)):.3e})"
                bad.append(f"{label}/{f}/{p}: value differs{d}")
    return bad, n


def compare_fields(A, B, time, vtol=0.0):
    bad, n = [], 0
    pa, pb = procs(A), procs(B)
    if len(pa) != len(pb):
        print(f"FAIL: {len(pa)} processors against {len(pb)}")
        return 1
    for k, (a, b) in enumerate(zip(pa, pb)):
        x, c = compare_field_dirs(os.path.join(a, time), os.path.join(b, time),
                                  os.path.join(a, "constant/polyMesh"), os.path.join(b, "constant/polyMesh"), f"processor{k}", vtol)
        bad += x
        n += c
    if bad:
        print("FAIL: the processor fields differ:\n  " + "\n  ".join(bad[:30]))
        return 1
    print(f"ok: {n} field files over {len(pa)} processors identical at {time}")
    return 0


def compare_global(A, B, time):
    bad, n = compare_field_dirs(os.path.join(A, time), os.path.join(B, time), os.path.join(A, "constant/polyMesh"),
                                os.path.join(B, "constant/polyMesh"), "global")
    if bad:
        print("FAIL: the reconstructed fields differ:\n  " + "\n  ".join(bad[:30]))
        return 1
    print(f"ok: {n} global field files identical at {time}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(2)
    mode, A, B = sys.argv[1:4]
    if mode == "mesh":
        sys.exit(compare_mesh(A, B))
    if mode == "fields":
        vtol = float(sys.argv[sys.argv.index("--vtol") + 1]) if "--vtol" in sys.argv else 0.0
        sys.exit(compare_fields(A, B, sys.argv[4], vtol))
    if mode == "global":
        sys.exit(compare_global(A, B, sys.argv[4]))
    print(__doc__)
    sys.exit(2)
