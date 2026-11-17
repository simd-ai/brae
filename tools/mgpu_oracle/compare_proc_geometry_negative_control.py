#!/usr/bin/env python3
"""Negative control of compare_proc_geometry.py: a perturbed dump MUST fail the comparison.

    compare_proc_geometry_negative_control.py <decomposed case dir>

Copies the case's brae_procGeometry.dump files to a scratch tree (the case is never written) and runs three
mutations, each of which has to make the comparator exit nonzero:
  weight  the first processor face's weight is changed by 1e-6 relative;
  delta   the first processor face's delta x component is changed by 1e-6 relative;
  slot    the second face's halo slot is replaced by slot+100 (structure, not numbers);
  faces   the LAST face record of the first patch is dropped (the zip truncation guard).
Exit 0 when every mutation is caught, 1 when one passes silently, 77 when the case has no dumps.
"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def run(case):
    return subprocess.run([sys.executable, os.path.join(HERE, "compare_proc_geometry.py"), case, "-q"],
                          capture_output=True, text=True)


def mutate(path, mode):
    lines = open(path).read().split("\n")
    out = []
    seen_f = 0
    for line in lines:
        t = line.split()
        if t and t[0] == "f":
            seen_f += 1
            if seen_f == 1 and mode in ("weight", "delta"):
                idx = 5 if mode == "weight" else 12   # w is the first value column, dx the 8th
                v = float(t[idx])
                t[idx] = repr(v*1.000001 if v != 0 else 1e-6)
                line = " ".join(t)
            if seen_f == 2 and mode == "slot":
                t[4] = str(int(t[4]) + 100)
                line = " ".join(t)
        out.append(line)
    if mode == "faces":
        kept = []
        dropped = False
        for line in reversed(out):
            t = line.split()
            if not dropped and t and t[0] == "f":
                dropped = True
                continue
            kept.append(line)
        out = list(reversed(kept))
    open(path, "w").write("\n".join(out))


def main():
    if len(sys.argv) < 2 or not sys.argv[1]:
        print("SKIP: no decomposed case given (configure with -DBRAE_PROC_GEOMETRY_CASE=<case with both dump sets>)")
        return 77
    case = sys.argv[1]
    dumps = sorted(glob.glob(os.path.join(case, "processor*", "brae_procGeometry.dump")))
    if not dumps:
        print("SKIP: no brae_procGeometry.dump under", case)
        return 77
    bad = 0
    for mode in ("weight", "delta", "slot", "faces"):
        with tempfile.TemporaryDirectory() as tmp:
            work = os.path.join(tmp, "case")
            shutil.copytree(case, work)
            target = sorted(glob.glob(os.path.join(work, "processor*", "brae_procGeometry.dump")))[0]
            mutate(target, mode)
            r = run(work)
            caught = r.returncode != 0
            print("%-7s %s (exit %d)%s" % (mode, "CAUGHT" if caught else "NOT CAUGHT", r.returncode,
                                           "" if caught else "  <-- the comparator accepted a corrupted dump"))
            if not caught:
                bad = 1
            else:
                sys.stdout.write("        " + (r.stdout.strip().splitlines() or [""])[0][:150] + "\n")
    return bad


if __name__ == "__main__":
    sys.exit(main())
