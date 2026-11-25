#!/usr/bin/env python3
"""Two cases' ASCII fields, entry by entry: the internal field and every patch's written `value`, each on its own scale.

    pimple_entry_compare.py <referenceCase> <otherCase> <time,time,...> <field> [<field> ...]

Prints one line per (time, field, entry), in that order:
    <time> <field> <entry> n=<numbers> max|ref| <s> max|a-b| <d> rel <d/s> worst <index>
rel is the entry's largest |a - b| over the largest |a| of the SAME entry, so a patch whose values are small
next to the internal field's is judged on its own numbers (tools/foam_time_compare.py takes one scale a
file). A `uniform` entry is expanded against the other side's list. An entry the reference writes without
a `value` (zeroGradient, ...) is skipped; one written on one side only is reported and counts as rel inf.
The last line is `MAX <rel> <time> <field> <entry>` over everything printed. Measurement only: the
caller holds the bound.
"""
import os
import re
import sys

WIDTH = {'scalar': 1, 'vector': 3, 'symmTensor': 6, 'tensor': 9}


def strip_comments(s):
    s = re.sub(r'/\*.*?\*/', ' ', s, flags=re.S)
    return re.sub(r'//[^\n]*', ' ', s)


def block_end(s, i):
    """s[i] is '{' or '(': the index one past its matching close."""
    op = s[i]
    cl = '}' if op == '{' else ')'
    depth = 0
    k = i
    while True:
        c = s[k]
        if c == op:
            depth += 1
        elif c == cl:
            depth -= 1
            if depth == 0:
                return k + 1
        k += 1


def value_of(text):
    """The numbers of an entry, read from just after its keyword: ('u', nums) or ('n', nums)."""
    t = text.lstrip()
    if t.startswith('uniform'):
        t = t[len('uniform'):].lstrip()
        if t.startswith('('):
            e = block_end(t, 0)
            return 'u', [float(x) for x in t[1:e - 1].split()]
        return 'u', [float(t.split(';')[0])]
    m = re.match(r'nonuniform\s+List<(\w+)>\s*(\d+)\s*\(', t)
    if not m:
        if re.match(r'nonuniform\s+List<(\w+)>\s*0\s*\(\s*\)', t):
            return 'n', []
        raise ValueError('cannot read: ' + t[:60])
    j = m.end() - 1
    e = block_end(t, j)
    nums = [float(x) for x in t[j + 1:e - 1].replace('(', ' ').replace(')', ' ').split()]
    if len(nums) != int(m.group(2)) * WIDTH[m.group(1)]:
        raise ValueError('count mismatch')
    return 'n', nums


def read_field(path):
    s = strip_comments(open(path).read())
    out = {}
    i = s.index('internalField')
    out['internal'] = value_of(s[i + len('internalField'):])
    b = s.index('boundaryField')
    o = s.index('{', b)
    body = s[o + 1:block_end(s, o) - 1]
    k = 0
    entry = re.compile(r'\s*("?[\w.:<>()|*-]+"?)\s*\{')
    while True:
        m = entry.match(body, k)
        if not m:
            break
        bo = m.end() - 1
        be = block_end(body, bo)
        pb = body[bo + 1:be - 1]
        mv = re.search(r'(?:^|[\s;])value\s', pb)
        out[m.group(1)] = value_of(pb[mv.end():]) if mv else None
        k = be
    return out


def expand(v, n):
    kind, nums = v
    if kind == 'u' and n and nums and n % len(nums) == 0:
        return nums * (n // len(nums))
    return nums


def main(argv):
    ref, oth, times, fields = argv[1], argv[2], argv[3].split(','), argv[4:]
    worst = (0.0, '-', '-', '-')
    for t in times:
        for f in fields:
            pa = os.path.join(ref, t, f)
            pb = os.path.join(oth, t, f)
            if not (os.path.exists(pa) and os.path.exists(pb)):
                print('%-8s %-8s missing: reference %s other %s' % (t, f, os.path.exists(pa), os.path.exists(pb)))
                worst = max(worst, (float('inf'), t, f, 'file'))
                continue
            A = read_field(pa)
            B = read_field(pb)
            for name in A:
                if A[name] is None:
                    continue
                if name not in B or B[name] is None:
                    print('%-8s %-8s %-14s written by the reference only' % (t, f, name))
                    worst = max(worst, (float('inf'), t, f, name))
                    continue
                n = max(len(A[name][1]), len(B[name][1]))
                x = expand(A[name], n)
                y = expand(B[name], n)
                if len(x) != len(y):
                    print('%-8s %-8s %-14s sizes differ %d %d' % (t, f, name, len(x), len(y)))
                    worst = max(worst, (float('inf'), t, f, name))
                    continue
                scale = max([abs(v) for v in x] + [0.0])
                diffs = [abs(p - q) for p, q in zip(x, y)]
                d = max(diffs + [0.0])
                at = diffs.index(d) if diffs else -1
                rel = d / scale if scale > 0 else (0.0 if d == 0 else float('inf'))
                print('%-8s %-8s %-14s n=%-6d max|ref| %-12.6g max|a-b| %-12.6g rel %.3e worst %d'
                      % (t, f, name, len(x), scale, d, rel, at))
                worst = max(worst, (rel, t, f, name))
    print('MAX %.3e %s %s %s' % worst)


if __name__ == '__main__':
    main(sys.argv)
