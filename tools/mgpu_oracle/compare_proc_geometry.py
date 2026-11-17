#!/usr/bin/env python3
"""Compare brae's processor-patch geometry with OpenFOAM's, rank by rank.

    compare_proc_geometry.py <decomposed case dir> [--max-ulp N] [--max-ulp-vec N]

Reads <case>/processorN/procGeometry.dump (tools/dumpInterFoam, OpenFOAM v2606) and
<case>/processorN/brae_procGeometry.dump (BRAE_MGPU_GEOMETRY_ONLY=1 brae_interFoam) for every rank and compares, face
by face, in patch order: owner cell and the NEIGHBOUR's owner cell (exact), the weights, deltaCoeffs and
nonOrthDeltaCoeffs (ULP distance, gated by --max-ulp, default 1), magSf, Sf, delta and nonOrthCorrectionVectors
(ULP distance per component, gated by --max-ulp-vec, default 1). It also checks brae's halo slots: the slot of a
face is a function of the neighbour's cell and the slots of one patch are distinct and contiguous.

Prints one row per rank and a summary; exit status 1 when a gate is exceeded or the structure differs.
"""
import argparse
import glob
import os
import struct
import sys

COLS = ["w", "dc", "nodc", "magSf", "Sfx", "Sfy", "Sfz", "dx", "dy", "dz", "cx", "cy", "cz"]
SCALARS = ["w", "dc", "nodc"]


def parse(path):
    rank = None
    ncells = None
    patches = []   # [name, my, nbr, size, start, faces[list of tuples]]
    with open(path) as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            if t[0] == "rank":
                rank = int(t[1])
                ncells = int(t[3])
            elif t[0] == "patch":
                patches.append([t[1], int(t[3]), int(t[5]), int(t[7]), int(t[9]), []])
            elif t[0] == "f":
                patches[-1][5].append((int(t[2]), int(t[3]), int(t[4]), [float(x) for x in t[5:]]))
    return rank, ncells, patches


def ordered(x):
    """Map a double onto a monotone integer line (so that |a-b| is the ULP distance), +0 == -0."""
    (i,) = struct.unpack("<q", struct.pack("<d", x))
    return i if i >= 0 else -(i & 0x7FFFFFFFFFFFFFFF)


def ulp(a, b):
    return abs(ordered(a) - ordered(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--max-ulp", type=int, default=1)
    ap.add_argument("--max-ulp-vec", type=int, default=1)
    ap.add_argument("-q", action="store_true", help="only the summary line")
    a = ap.parse_args()

    dirs = sorted(glob.glob(os.path.join(a.case, "processor*")), key=lambda d: int(d.rsplit("processor", 1)[1]))
    if not dirs:
        print("no processorN in", a.case)
        return 2
    bad = []
    worst = {c: 0 for c in COLS}
    nonbit = {c: 0 for c in COLS}
    tot_faces = 0
    tot_patches = 0
    if not a.q:
        print("%-6s %7s %7s %8s %8s  %s" % ("rank", "cells", "patches", "faces", "nbrCells", "max ULP (w dc nodc | magSf Sf delta corr)"))
    for d in dirs:
        r = int(d.rsplit("processor", 1)[1])
        fo, fb = os.path.join(d, "procGeometry.dump"), os.path.join(d, "brae_procGeometry.dump")
        if not (os.path.exists(fo) and os.path.exists(fb)):
            bad.append("rank %d: missing dump (%s %s)" % (r, os.path.exists(fo), os.path.exists(fb)))
            continue
        ro, no, po = parse(fo)
        rb, nb, pb = parse(fb)
        if ro != r or rb != r or no != nb:
            bad.append("rank %d: header differs (OF rank %s cells %s, brae rank %s cells %s)" % (r, ro, no, rb, nb))
            continue
        if [(p[0], p[1], p[2], p[3], p[4]) for p in po] != [(p[0], p[1], p[2], p[3], p[4]) for p in pb]:
            bad.append("rank %d: patch list differs\n  OF   %s\n  brae %s" % (r, [(p[0], p[3]) for p in po], [(p[0], p[3]) for p in pb]))
            continue
        mx = {c: 0 for c in COLS}
        nfaces = 0
        nbrCells = 0
        for qo, qb in zip(po, pb):
            slot_of = {}
            slots = set()
            # zip() truncates silently: a dump with FEWER faces than the oracle would compare clean. The face
            # lists must have the same length before any per-face comparison (review finding 11).
            if len(qo[5]) != len(qb[5]):
                bad.append("rank %d patch %s: face-list length differs (OF %d, brae %d)"
                           % (r, qo[0], len(qo[5]), len(qb[5])))
                continue
            for i, (fo_, fb_) in enumerate(zip(qo[5], qb[5])):
                nfaces += 1
                if fo_[0] != fb_[0] or fo_[1] != fb_[1]:
                    bad.append("rank %d patch %s face %d: cells OF (%d,%d) brae (%d,%d)" % (r, qo[0], i, fo_[0], fo_[1], fb_[0], fb_[1]))
                    break
                s = fb_[2]
                if slot_of.setdefault(fb_[1], s) != s:
                    bad.append("rank %d patch %s face %d: neighbour cell %d has slots %d and %d" % (r, qo[0], i, fb_[1], slot_of[fb_[1]], s))
                    break
                slots.add(s)
                if len(fo_[3]) != len(fb_[3]) or len(fo_[3]) != len(COLS):
                    bad.append("rank %d patch %s face %d: value-list length differs (OF %d, brae %d, expected %d)"
                               % (r, qo[0], i, len(fo_[3]), len(fb_[3]), len(COLS)))
                    break
                for c, vo, vb in zip(COLS, fo_[3], fb_[3]):
                    u = ulp(vo, vb)
                    mx[c] = max(mx[c], u)
                    if u:
                        nonbit[c] += 1
            nbrCells += len(slots)
            if slots and (max(slots) - min(slots) + 1 != len(slots)):
                bad.append("rank %d patch %s: halo slots %d..%d are not contiguous (%d distinct)" % (r, qo[0], min(slots), max(slots), len(slots)))
            if len(slots) != len(set(slot_of.keys())):
                bad.append("rank %d patch %s: %d slots for %d neighbour cells" % (r, qo[0], len(slots), len(slot_of)))
        for c in COLS:
            worst[c] = max(worst[c], mx[c])
        tot_faces += nfaces
        tot_patches += len(po)
        for c in COLS:
            lim = a.max_ulp if c in SCALARS else a.max_ulp_vec
            if mx[c] > lim:
                bad.append("rank %d: %s differs by %d ULP (limit %d)" % (r, c, mx[c], lim))
        if not a.q:
            print("%-6d %7d %7d %8d %8d  %d %d %d | %d %d %d %d" % (
                r, no, len(po), nfaces, nbrCells, mx["w"], mx["dc"], mx["nodc"], mx["magSf"],
                max(mx["Sfx"], mx["Sfy"], mx["Sfz"]), max(mx["dx"], mx["dy"], mx["dz"]),
                max(mx["cx"], mx["cy"], mx["cz"])))
    print("ranks %d patches %d faces %d; worst ULP w %d dc %d nodc %d magSf %d Sf %d delta %d corr %d; "
          "values not bitwise equal: %s" % (
              len(dirs), tot_patches, tot_faces, worst["w"], worst["dc"], worst["nodc"], worst["magSf"],
              max(worst["Sfx"], worst["Sfy"], worst["Sfz"]), max(worst["dx"], worst["dy"], worst["dz"]),
              max(worst["cx"], worst["cy"], worst["cz"]),
              ", ".join("%s %d" % (c, n) for c, n in nonbit.items() if n) or "none"))
    for b in bad:
        print("FAIL:", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
