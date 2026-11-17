#!/usr/bin/env python3
"""Exports the OpenFOAM -parallel p_rgh matrix of one corrector (oracle dump oPrghEqnc<k>n<j>*) as one flat binary file per
rank, for tests/test_dist_solvers_oracle.cu: the local lduMatrix exactly as OpenFOAM hands it to the solver.

    export_prgh_matrix.py <run>/par_d1 <outdir> [corrector=1] [nonorth=1] [time=first written]
    (a serial run, <run>/serial_d1, exports the monolithic matrix as the single rank0)

Per rank r, <outdir>/rank<r>.bin (little endian):
    int32  magic 0x70524748, nC, nF, nProcPatches, nPatchesOther
    float64[nC]  diag      = Diag + sum over ALL patches of IntCoeffs at the patch's face cells (addBoundaryDiag: what the
                             solver's diagonal is, the processor patches' share included)
    float64[nC]  source    = Source + sum over the NON-coupled patches of BouCoeffs (addBoundarySource; a processor patch's
                             explicit part cancels against the interface update, fvMatrixSolve.C)
    float64[nF]  upper, lower;  int32[nF] owner, neighbour       (internal faces in OpenFOAM's lduAddressing order)
    float64[nC]  psi0      = oPrghPre   (p_rgh before the solve = the solver's initial guess)
    float64[nC]  psiRef    = oPrgh      (p_rgh after the corrector's solve and relax)
    per processor patch (in boundary-file order):
        int32 nbrRank, nFaces; int32[nFaces] faceCells; int32[nFaces] nbrFaceCells; float64[nFaces] bouCoeffs
and <outdir>/solver.txt: the OpenFOAM log's "Solving for p_rgh" line of that corrector (PISO correctors count from 1) (initial residual, final residual,
iterations), the number of ranks and the global cell count.
"""
import os
import re
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import foamio  # noqa: E402


def main():
    run = sys.argv[1]
    out = sys.argv[2]
    corr = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    nono = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    os.makedirs(out, exist_ok=True)
    procs = sorted((d for d in os.listdir(run) if re.fullmatch(r"processor\d+", d)), key=lambda s: int(s[9:]))
    if not procs:                      # a serial run: the case itself is the one "rank", with no processor patches
        procs = ["."]
    nproc = len(procs)
    # the time directory the dumps were written to: the one holding oPrghEqn...Diag.dump on processor0
    base = "oPrghEqnc%dn%d" % (corr, nono)
    tname = sys.argv[5] if len(sys.argv) > 5 else None
    if tname is None:
        for _, t in foamio.time_dirs(os.path.join(run, procs[0])):
            if os.path.exists(os.path.join(run, procs[0], t, base + "Diag.dump")):
                tname = t
                break
    if tname is None:
        sys.exit("no %sDiag.dump under %s/processor0" % (base, run))
    meshes = []
    for p in procs:
        pd = os.path.join(run, p, "constant", "polyMesh")
        nc, _, nfaces, nint = foamio.mesh_counts(pd)
        owner = foamio.read_label_list(os.path.join(pd, "owner"))
        nei = foamio.read_label_list(os.path.join(pd, "neighbour"))
        patches = foamio.read_boundary(os.path.join(pd, "boundary"))
        meshes.append(dict(dir=p, nc=nc, nint=nint, owner=np.asarray(owner), nei=np.asarray(nei), patches=patches))
    ncells = sum(m["nc"] for m in meshes)

    def field(r, name, surf):
        m = meshes[r]
        sizes = {pt["name"]: pt["nFaces"] for pt in m["patches"]}
        return foamio.read_field(os.path.join(run, m["dir"], tname, name + ".dump"), sizes, m["nint"] if surf else m["nc"])

    def patch_cells(r, pt):
        m = meshes[r]
        return m["owner"][pt["startFace"]:pt["startFace"] + pt["nFaces"]]

    for r, m in enumerate(meshes):
        nc, nint = m["nc"], m["nint"]
        diag = np.array(field(r, base + "Diag", False).internal, dtype=np.float64)
        src = np.array(field(r, base + "Source", False).internal, dtype=np.float64)
        upper = np.array(field(r, base + "Upper", True).internal, dtype=np.float64)
        lower = np.array(field(r, base + "Lower", True).internal, dtype=np.float64) if os.path.exists(
            os.path.join(run, m["dir"], tname, base + "Lower.dump")) else upper.copy()
        ic = field(r, base + "IntCoeffs", True)
        bc = field(r, base + "BouCoeffs", True)
        pp = []
        for pt in m["patches"]:
            cells = patch_cells(r, pt)
            kind = pt["type"]
            if pt["nFaces"] == 0:
                continue
            if kind == "empty":
                continue
            icv = ic.boundary[pt["name"]][1]
            bcv = bc.boundary[pt["name"]][1]
            if icv is None or bcv is None:
                raise SystemExit("rank %d patch %s: no coefficient values in the dump" % (r, pt["name"]))
            np.add.at(diag, cells, icv)
            if kind == "processor":
                nb = pt["neighbProcNo"]
                other = [q for q in meshes[nb]["patches"] if q["type"] == "processor" and q["neighbProcNo"] == r]
                assert len(other) == 1, "ranks %d/%d: expected one processor patch pair" % (r, nb)
                pp.append((nb, cells, patch_cells(nb, other[0]), bcv))
            else:
                np.add.at(src, cells, bcv)
        psi0 = np.array(field(r, "oPrghPre" + base[len("oPrghEqn"):], False).internal, dtype=np.float64)
        psir = np.array(field(r, "oPrgh" + "c%d" % corr, False).internal, dtype=np.float64)
        with open(os.path.join(out, "rank%d.bin" % r), "wb") as f:
            f.write(struct.pack("<5i", 0x70524748, nc, nint, len(pp), 0))
            for a in (diag, src, upper, lower):
                f.write(np.ascontiguousarray(a, dtype="<f8").tobytes())
            f.write(np.ascontiguousarray(m["owner"][:nint], dtype="<i4").tobytes())
            f.write(np.ascontiguousarray(m["nei"][:nint], dtype="<i4").tobytes())
            f.write(np.ascontiguousarray(psi0, dtype="<f8").tobytes())
            f.write(np.ascontiguousarray(psir, dtype="<f8").tobytes())
            for nb, cells, ncells_, bcv in pp:
                f.write(struct.pack("<2i", nb, len(cells)))
                f.write(np.ascontiguousarray(cells, dtype="<i4").tobytes())
                f.write(np.ascontiguousarray(ncells_, dtype="<i4").tobytes())
                f.write(np.ascontiguousarray(bcv, dtype="<f8").tobytes())
    # OpenFOAM's report for that solve: the (corr+1)-th "Solving for p_rgh" line of the first time step
    runp = run.rstrip("/")
    log = os.path.join(os.path.dirname(runp), "log." + os.path.basename(runp))
    log = log if os.path.exists(log) else None
    line = ""
    if log:
        hits = [ln for ln in open(log, errors="replace") if "Solving for p_rgh" in ln]
        if len(hits) >= max(corr, 1):
            line = hits[max(corr, 1) - 1].strip()
    with open(os.path.join(out, "solver.txt"), "w") as f:
        f.write("%d %d\n%s\n" % (nproc, ncells, line))
    print("exported %d ranks, %d cells; OpenFOAM: %s" % (nproc, ncells, line))


if __name__ == "__main__":
    main()
