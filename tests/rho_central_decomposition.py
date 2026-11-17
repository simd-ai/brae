#!/usr/bin/env python3
"""The frozen OpenFOAM decompositions the rhoCentralFoam gates use for their four-way reference (no OpenFOAM needed).

OpenFOAM's scotch partition is not reproducible on the GB10's OpenFOAM v2412 (two gate runs split forwardStep 4032 x 4
and then 4066/4032/3998/4032), so the four-way spread moved between runs. One scotch partition per mesh was therefore
frozen -- taken from the runs whose numbers milestones C and D report -- and every four-way reference decomposes with
OpenFOAM's `manual` method on it (manualDecomp.C: a labelList `dataFile` read from constant/).

  rho_central_decomposition.py freeze <decomposed case> <out file>
      the cell -> processor list of a decomposed case (its processor*/constant/polyMesh/cellProcAddressing), written
      as an OpenFOAM labelList `cellDecomposition`
  rho_central_decomposition.py meshsum <case>
      sha256 of the mesh's faces, owner and neighbour (their data, not the headers: the topology a partition belongs
      to), and its cell count
  rho_central_decomposition.py install <case> [mesh name]
      finds the frozen decomposition of the case's mesh (by meshsum, or the one named), checks the frozen file against
      its recorded sha256, and copies it to <case>/constant/cellDecomposition; prints its processor count. Non-zero,
      saying why, when no frozen decomposition is recorded for the mesh or a checksum does not match: a gate then
      fails rather than fall back to a fresh scotch run

The data dir (default tests/data/rho_central_decomposition) holds <mesh name>.cellDecomposition and MANIFEST, one line
per mesh: `<mesh name> <nCells> <nProcs> <meshsum> <sha256 of the frozen file>`.
"""
import hashlib
import os
import re
import shutil
import sys

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "rho_central_decomposition")
HEADER = """FoamFile
{
    version     2.0;
    format      ascii;
    class       labelList;
    location    "constant";
    object      cellDecomposition;
}
"""


def _body(path):
    """the data of an OpenFOAM ascii file: everything after its FoamFile header, comments dropped"""
    s = open(path).read()
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    s = re.sub(r"//[^\n]*", "", s)
    i = s.find("FoamFile")
    if i >= 0:
        s = s[s.index("}", i) + 1:]
    return " ".join(s.split())


def _labels(path):
    b = _body(path)
    m = re.match(r"(\d+)\s*\((.*)\)", b)
    if not m:
        raise SystemExit(f"rho_central_decomposition: {path} is not an ascii labelList")
    v = [int(x) for x in m.group(2).split()]
    if len(v) != int(m.group(1)):
        raise SystemExit(f"rho_central_decomposition: {path}: {len(v)} labels, header says {m.group(1)}")
    return v


def meshsum(case):
    # the TOPOLOGY only: a partition is a cell -> processor list, valid for any mesh with these cells and faces. The
    # points are left out because blockMesh writes them at controlDict's writePrecision (forwardStep as shipped and at 17
    # digits are the same mesh, written 0.05 and 0.050000000000000003)
    pm = f"{case}/constant/polyMesh"
    h = hashlib.sha256()
    for f in ("faces", "owner", "neighbour"):
        h.update(f.encode())
        h.update(_body(f"{pm}/{f}").encode())
    own = _labels(f"{pm}/owner")
    return h.hexdigest(), max(own) + 1


def freeze(case, out):
    procs = sorted((d for d in os.listdir(case) if re.fullmatch(r"processor\d+", d)), key=lambda d: int(d[9:]))
    if not procs:
        raise SystemExit(f"rho_central_decomposition: no processor directories in {case}")
    _, n = meshsum(case)
    dec = [-1] * n
    for d in procs:
        for c in _labels(f"{case}/{d}/constant/polyMesh/cellProcAddressing"):
            dec[c] = int(d[9:])
    if min(dec) < 0:
        raise SystemExit(f"rho_central_decomposition: {case}: a cell is on no processor")
    with open(out, "w") as f:
        f.write(HEADER + "\n%d\n(\n%s\n)\n" % (n, "\n".join(map(str, dec))))
    print("frozen", out, "cells", n, "processors", len(procs),
          "sizes", " ".join(str(dec.count(p)) for p in range(len(procs))))


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def install(case, name=None, data=DATA):
    rows = {r.split()[0]: r.split() for r in open(f"{data}/MANIFEST") if r.strip() and not r.startswith("#")}
    got, gn = meshsum(case)
    if name is None:
        hit = [k for k, r in rows.items() if r[3] == got]
        if not hit:
            raise SystemExit(f"rho_central_decomposition: no frozen decomposition for {case}'s mesh ({gn} cells, "
                             f"meshsum {got[:12]})")
        name = hit[0]
    if name not in rows:
        raise SystemExit(f"rho_central_decomposition: no frozen decomposition named {name}")
    _, n, nproc, msum, fsum = rows[name]
    if gn != int(n) or got != msum:
        raise SystemExit(f"rho_central_decomposition: {case}'s mesh is not {name}'s ({gn} cells, meshsum {got[:12]}; "
                         f"recorded {n}, {msum[:12]})")
    src = f"{data}/{name}.cellDecomposition"
    if sha(src) != fsum:
        raise SystemExit(f"rho_central_decomposition: {src} does not match its recorded sha256")
    shutil.copyfile(src, f"{case}/constant/cellDecomposition")
    print(nproc)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["freeze"] and len(a) == 3:
        freeze(a[1], a[2])
    elif a[:1] == ["meshsum"] and len(a) == 2:
        s, n = meshsum(a[1])
        print(s, n)
    elif a[:1] == ["install"] and len(a) in (2, 3):
        install(*a[1:])
    else:
        raise SystemExit(__doc__)
