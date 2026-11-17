#!/usr/bin/env python3
"""compare_rank_dirs.py A B <time> [--ignore=name,name]

Compare processor*/<time>/ of two decomposed cases file by file, BYTE for byte after the 8-line banner (brae's
banner says `Version: 2412`, OpenFOAM v2606's 2606; nothing else of a header may differ). A is OpenFOAM's
(decomposePar, a -parallel run), B is brae's per-rank writer. Prints one line per file kind with the number of ranks
whose file is identical; exit status 1 if any differs."""
import os, sys

def body(path):
    with open(path, 'rb') as f:
        data = f.read()
    # banner = first 8 lines (/*---, 6 lines, \*---)
    lines = data.split(b'\n', 8)
    return lines[8] if len(lines) > 8 else data

def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    ignore = set()
    for a in sys.argv[1:]:
        if a.startswith('--ignore='):
            ignore = set(a[9:].split(','))
    A, B, T = args
    ranks = sorted(d for d in os.listdir(A) if d.startswith('processor'))
    kinds = {}
    for r in ranks:
        ta, tb = os.path.join(A, r, T), os.path.join(B, r, T)
        for root, _, files in os.walk(ta):
            for fn in files:
                rel = os.path.relpath(os.path.join(root, fn), ta)
                if rel in ignore:
                    continue
                pb = os.path.join(tb, rel)
                ok = os.path.exists(pb) and body(os.path.join(root, fn)) == body(pb)
                same, total, why = kinds.get(rel, (0, 0, ''))
                kinds[rel] = (same + ok, total + 1, why if ok else why or ('missing' if not os.path.exists(pb) else 'differs (rank %s)' % r))
        for root, _, files in os.walk(tb):
            for fn in files:
                rel = os.path.relpath(os.path.join(root, fn), tb)
                if rel not in ignore and not os.path.exists(os.path.join(ta, rel)):
                    kinds[rel + ' (extra in B)'] = (0, 1, 'extra')
    bad = 0
    for rel in sorted(kinds):
        same, total, why = kinds[rel]
        print('%-34s %d/%d ranks identical %s' % (rel, same, total, '' if same == total else '  <-- ' + why))
        bad += same != total
    return 1 if bad else 0

if __name__ == '__main__':
    sys.exit(main())
