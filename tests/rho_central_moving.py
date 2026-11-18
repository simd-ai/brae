#!/usr/bin/env python3
"""Cases for the rhoCentralFoam gate of milestone K -- mesh motion, OpenFOAM's movingCone -- (no OpenFOAM needed).

  rho_central_moving.py setup <kind> <case>
      rewrites a staged copy of movingCone (0, constant, system), before its blockMesh:
        mc_shipped   nothing: the tutorial as shipped (writeFormat binary, writePrecision 6, purgeWrite 5, endTime 2e-5;
                     U and e symGaussSeidel tolerance 1e-5 relTol 0.1, cellMotionUx PCG/DIC 1e-8)
        mc_tight     writeFormat ascii, writePrecision 17, purgeWrite 0, writes every 2e-6; U and e at tolerance 1e-10
                     relTol 0, cellMotionUx at 1e-14. The shipped relTol 0.1 stops the U and e solves early, and
                     OpenFOAM's own two- and four-way runs then part from its serial by 1e-5 to 3e-4 (2026-11-18); with
                     these solves its spread is 1e-7 in the fields and 1e-13 in the points
  rho_central_moving.py restart <from case> <time> <case>
      a restart directory: <from case>'s system, constant and its <time> directory (points, fields, cellMotionUx,
      pointMotionUx, uniform/time), controlDict startFrom startTime at <time>
"""
import os
import re
import shutil
import sys


def edit(path, pattern, repl):
    s = open(path).read()
    t, n = re.subn(pattern, repl, s, count=1, flags=re.M | re.S)
    if n != 1:
        raise SystemExit(f"rho_central_moving: no match for {pattern!r} in {path}")
    open(path, "w").write(t)


def setup(kind, case):
    if kind == "mc_shipped":
        return
    if kind != "mc_tight":
        raise SystemExit(f"rho_central_moving: no case kind {kind}")
    cd = f"{case}/system/controlDict"
    for k, v in (("writeFormat", "ascii"), ("writePrecision", "17"), ("purgeWrite", "0"), ("writeInterval", "2e-06")):
        edit(cd, r"^(%s\s+)\S+;" % k, r"\g<1>%s;" % v)
    fs = f"{case}/system/fvSolution"
    edit(fs, r'("\(U\|h\|e\|R\|k\|epsilon\|omega\)"\s*\{[^}]*?tolerance\s+)\S+;', r"\g<1>1e-10;")
    edit(fs, r'("\(U\|h\|e\|R\|k\|epsilon\|omega\)"\s*\{[^}]*?relTol\s+)\S+;', r"\g<1>0;")
    edit(fs, r"(cellMotionUx\s*\{[^}]*?tolerance\s+)\S+;", r"\g<1>1e-14;")


def restart(src, time, case):
    shutil.rmtree(case, ignore_errors=True)
    os.makedirs(case)
    for d in ("system", "constant", time):
        shutil.copytree(f"{src}/{d}", f"{case}/{d}")
    cd = f"{case}/system/controlDict"
    edit(cd, r"^(startFrom\s+)\S+;", r"\g<1>startTime;")
    edit(cd, r"^(startTime\s+)\S+;", r"\g<1>%s;" % time)


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 3 and a[0] == "setup":
        setup(a[1], a[2])
    elif len(a) == 4 and a[0] == "restart":
        restart(a[1], a[2], a[3])
    else:
        raise SystemExit(__doc__)
