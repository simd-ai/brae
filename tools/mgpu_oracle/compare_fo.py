#!/usr/bin/env python3
"""compare_fo.py REF OUT [--rtol-sum=1e-12] [--objects=a,b] [--skip=a,b] [--last-row]

Compare the function-object files of two postProcessing/ trees: REF is OpenFOAM's (a `postProcess -parallel` run
or an `interFoam -parallel` run), OUT is brae's (tools/mgpu_oracle/frozen_state.sh, `mgpu_frozen fo`). The two
write the same numbers in different layouts (OpenFOAM v2606's volFieldValue header reads `# Region : cellZone z`,
brae's `# Region type : cellZone z`; the probes pad the columns differently), so the files are PARSED and the
numbers compared:

  volFieldValue.dat    `# Cells`, `# Volume` and every row: sums to --rtol-sum, min/max exactly
  surfaceFieldValue.dat  the same
  fieldMinMax.dat      min/max values, the `processor` columns and the locations: exactly
  probes/<field>       every column of every row: exactly (a probe's value is one cell's value)

"exact" means the two printed numbers (12 significant digits) are the same double. For the sums the report also says
how many are bit-equal to the printed digits and the largest relative difference. Exit status 1 on any mismatch.
"""
import os
import re
import sys

NUM = re.compile(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[-+]?nan|[-+]?inf')


def floats(text):
    return [float(x) for x in NUM.findall(text)]


def rows(path):
    out = []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            out.append(line.rstrip('\n'))
    return out


def header_value(path, key):
    with open(path) as f:
        for line in f:
            m = re.match(r'#\s*%s\s*:\s*(\S+)' % key, line)
            if m:
                return float(m.group(1))
    return None


class Report:
    def __init__(self):
        self.bad = 0
        self.lines = []
        self.bitEqual = 0
        self.total = 0
        self.maxRel = 0.0

    def fail(self, msg):
        self.bad += 1
        self.lines.append('  MISMATCH ' + msg)

    def num(self, what, a, b, rtol, count=True):
        # headers (Cells, Volume, ...) are checked but not counted: OpenFOAM prints one digit more of them
        if count:
            self.total += 1
        if a == b:
            if count:
                self.bitEqual += 1
            return
        if rtol is None:
            self.fail('%s: %r vs %r' % (what, a, b))
            return
        scale = max(abs(a), abs(b), 1e-300)
        rel = abs(a - b) / scale
        if count:
            self.maxRel = max(self.maxRel, rel)
        if rel > rtol:
            self.fail('%s: %r vs %r (relative %.3g > %.3g)' % (what, a, b, rel, rtol))


def clean_probe_row(line):
    """OpenFOAM v2606 fills the column of a probe that is in no cell with -VGREAT (-1e300, `includeOutOfBounds`'s default);
    brae leaves the column out. Drop those numbers from the reference row."""
    return ' '.join(str(x) for x in floats(line) if x > -1e299)


def compare_tables(rep, name, ref, out, rtol, labelled=False, last_row=False, probes=False):
    ra, rb = rows(ref), rows(out)
    if last_row:
        ra = ra[-1:]
    if probes:
        ra = [clean_probe_row(x) for x in ra]
        rb = [' '.join(str(x) for x in floats(y)) for y in rb]
    if len(ra) != len(rb):
        rep.fail('%s: %d rows vs %d' % (name, len(ra), len(rb)))
        return
    for i, (la, lb) in enumerate(zip(ra, rb)):
        if labelled:
            la = la.split(None, 2)
            lb = lb.split(None, 2)
            if la[1] != lb[1]:
                rep.fail('%s row %d: field %s vs %s' % (name, i, la[1], lb[1]))
                continue
            fa, fb = floats(la[2]), floats(lb[2])
            t_a, t_b = float(la[0]), float(lb[0])
        else:
            fa, fb = floats(la), floats(lb)
            t_a, t_b = fa[0], fb[0]
            fa, fb = fa[1:], fb[1:]
        rep.num('%s row %d time' % (name, i), t_a, t_b, None)
        if len(fa) != len(fb):
            rep.fail('%s row %d: %d numbers vs %d' % (name, i, len(fa), len(fb)))
            continue
        for k, (x, y) in enumerate(zip(fa, fb)):
            rep.num('%s row %d col %d' % (name, i, k), x, y, rtol)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    opts = dict(a[2:].split('=', 1) for a in sys.argv[1:] if a.startswith('--') and '=' in a)
    if len(args) != 2:
        print(__doc__)
        return 2
    ref, out = args
    rtol = float(opts.get('rtol-sum', '1e-12'))
    last = '--last-row' in sys.argv
    only = set(opts['objects'].split(',')) if 'objects' in opts else None
    skip = set(opts['skip'].split(',')) if 'skip' in opts else set()
    status = 0
    for obj in sorted(os.listdir(ref)):
        if (only and obj not in only) or obj in skip:
            continue
        rdir = os.path.join(ref, obj)
        if not os.path.isdir(rdir):
            continue
        starts = sorted(os.listdir(rdir))
        if not starts:
            continue
        rs = os.path.join(rdir, starts[0])
        os_ = os.path.join(out, obj, starts[0])
        if not os.path.isdir(os_):
            print('%-18s MISSING in OUT (%s)' % (obj, os_))
            status = 1
            continue
        for fn in sorted(os.listdir(rs)):
            rep = Report()
            a, b = os.path.join(rs, fn), os.path.join(os_, fn)
            if not os.path.exists(b):
                rep.fail('file missing in OUT')
            elif fn in ('volFieldValue.dat', 'surfaceFieldValue.dat'):
                for key in ('Cells', 'Faces'):
                    va, vb = header_value(a, key), header_value(b, key)
                    if va is not None or vb is not None:
                        rep.num('header ' + key, va, vb, None, count=False)
                for key in ('Volume', 'Area'):
                    va, vb = header_value(a, key), header_value(b, key)
                    if va is not None or vb is not None:
                        rep.num('header ' + key, va, vb, 1e-11, count=False)
                compare_tables(rep, fn, a, b, rtol, last_row=last)
            elif fn == 'fieldMinMax.dat':
                compare_tables(rep, fn, a, b, None, labelled=True, last_row=last)
            else:   # probes/<field>
                compare_tables(rep, fn, a, b, None, last_row=last, probes=True)
            print('%-18s %-20s %4d numbers, %4d bit-equal, max relative difference %.2e%s' % (
                obj, fn, rep.total, rep.bitEqual, rep.maxRel, '  OK' if rep.bad == 0 else '  *** %d MISMATCH' % rep.bad))
            for l in rep.lines[:12]:
                print(l)
            if rep.bad:
                status = 1
    return status


if __name__ == '__main__':
    sys.exit(main())
