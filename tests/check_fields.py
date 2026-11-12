#!/usr/bin/env python3
"""Two OpenFOAM time directories laid beside each other, for tests/simple_ahmed_limited_vs_openfoam.sh.

  check_fields.py fields <time dir> <reference time dir> <bound> [field ...]
      every field's internal values and every patch's written values (a `uniform` value against a list is the value on
      every face) against the reference's: relative L2 (a vector by its largest component; a patch's reference norm at
      least the internal field's RMS times the root of its size); FAIL above <bound>, a number or field=number,... (the
      rest take default=). The last line is `worst <figure> <field> <where>`
  check_fields.py lag <time dir> <reference time dir> <reference's previous time dir> <bound> <field>
      <field>'s internal values against the reference's at the same time, and each patch's against the reference's at the
      previous time: a writer whose patch values are one correction behind. Prints `ok ...` or `FAIL ...`
"""
import gzip
import os
import re
import sys

import numpy


def text(f):
    """the file as text, one character a byte (a binary list's bytes survive the round trip through latin-1)"""
    for c in (f, f + '.gz'):
        if os.path.exists(c):
            with (gzip.open(c, 'rb') if c.endswith('.gz') else open(c, 'rb')) as fh:
                return fh.read().decode('latin-1')
    return None


WIDTH = {'scalar': 1, 'vector': 3, 'symmTensor': 6, 'tensor': 9, 'sphericalTensor': 1}


def values(body, start, binary):
    """(array, end) of the value at `start`: uniform x, or nonuniform List<T> n ( ... ), ascii or binary; an array of
    shape (n,) or (n, width); (None, start) when there is none"""
    m = re.compile(r'(uniform|nonuniform)\s+').match(body, start)
    if not m:
        return None, start
    k = m.end()
    if m.group(1) == 'uniform':
        if body[k] == '(':
            e = body.index(')', k)
            return numpy.array([[float(x) for x in body[k + 1:e].split()]]), e + 1
        e = body.index(';', k)
        return numpy.array([float(body[k:e])]), e
    # an empty list first: `List<scalar> 0()` is how OpenFOAM writes a patch with no face on a processor, and the general
    # pattern below would take it and hand numpy's parser a lone `)`
    m0 = re.compile(r'List<\w+>\s*0\s*\(\s*\)').match(body, k)
    if m0:
        return numpy.zeros(0), m0.end()
    m = re.compile(r'List<(\w+)>\s*(\d+)\s*\(').match(body, k)
    if not m:
        return None, k
    typ, n = m.group(1), int(m.group(2))
    w = WIDTH[typ]
    k = m.end()
    if binary:
        nbytes = 8 * w * n
        a = numpy.frombuffer(body[k:k + nbytes].encode('latin-1'), dtype='<f8').copy()
        e = k + nbytes
        if body[e] != ')':
            raise ValueError('a binary List<%s> of %d does not end where its size says' % (typ, n))
        return (a if w == 1 else a.reshape(n, w)), e + 1
    e = body.index(';', k)
    # numpy's own parser: a 14.1 M-cell vector field is 42 M numbers
    nums = numpy.fromstring(body[k:e].replace('(', ' ').replace(')', ' '), sep=' ')
    if nums.size != n * w:
        raise ValueError('an ascii List<%s> of %d holds %d numbers' % (typ, n, nums.size))
    return (nums if w == 1 else nums.reshape(n, w)), e


def read(f):
    """{'internal': array, <patch>: array} of a field file, the patches by their `value` entry; None when there is no
    file (or a <field>.npz that tests/gather_fields.py put together from a run split by rank)"""
    s = text(f)
    if s is None:
        if os.path.exists(f + '.npz'):
            z = numpy.load(f + '.npz')
            return {k: z[k] for k in z.files}
        return None
    binary = re.search(r'format\s+binary\s*;', s[:2000]) is not None
    r = {}
    m = re.search(r'internalField\s+', s)
    r['internal'], k = values(s, m.end(), binary)
    k = s.index('{', s.index('boundaryField', k)) + 1
    word = re.compile(r'\s*([^\s{};]+)\s*')
    while True:
        m = word.match(s, k)
        if not m:
            break   # the closing brace of boundaryField
        patch = m.group(1)
        k = s.index('{', m.end()) + 1
        while True:
            m = word.match(s, k)
            if not m:
                k = s.index('}', k) + 1
                break
            key = m.group(1)
            v, e = values(s, m.end(), binary)
            if v is not None:
                if key == 'value':
                    r[patch] = v
                k = s.index(';', e) + 1
            else:
                k = s.index(';', m.end()) + 1
    return r


def relative(a, b, scale=0.0):
    """relative L2 of a against b (a vector by its largest component); a one-value `uniform` broadcast to the other's
    size; None when the sizes differ otherwise. scale: the reference's norm is at least scale sqrt(n) -- a patch is held
    to the internal field's RMS, so that a patch whose values are about zero (phi on a slip patch: 1.45 on motorBike after
    three iterations, a ratio of round-off to round-off) reads its difference against the field's own size"""
    if a is None or b is None:
        return None
    if len(a) != len(b):
        if len(a) == 1:
            a = numpy.repeat(a, len(b), axis=0)
        elif len(b) == 1:
            b = numpy.repeat(b, len(a), axis=0)
        else:
            return None
    if len(b) == 0:
        return 0.0
    if b.ndim == 2:
        d = numpy.abs(a - b).max(axis=1)
        g = numpy.abs(b).max(axis=1)
    else:
        d = numpy.abs(a - b)
        g = numpy.abs(b)
    num = float(numpy.dot(d, d))
    den = max(float(numpy.dot(g, g)), scale * scale * len(b))
    if den == 0.0:
        return 0.0 if num == 0.0 else float('inf')
    return (num / den) ** 0.5


def bounds(spec):
    if '=' not in spec:
        return {'default': float(spec)}
    out = {}
    for item in spec.split(','):
        k, v = item.split('=')
        out[k] = float(v)
    return out


def fields(run, ref, spec, names):
    limit = bounds(spec)
    names = names or ['U', 'p', 'phi', 'k', 'epsilon', 'nut']
    fails = 0
    worst = (0.0, '')
    for f in names:
        a, b = read(run + '/' + f), read(ref + '/' + f)
        bound = limit.get(f, limit.get('default'))
        if b is None:
            # a field of the case's 0/ that the reference run does not write (pitzDaily's omega under kEpsilon)
            print('     %-8s not written by the reference' % f)
            continue
        if a is None:
            print('FAIL %s: %s missing' % (f, run))
            fails += 1
            continue
        parts = []
        # the internal field's RMS (a vector's largest component), the floor of every patch's norm
        bi = b.get('internal')
        g = numpy.abs(bi).max(axis=1) if bi is not None and bi.ndim == 2 else numpy.abs(bi) if bi is not None else numpy.zeros(0)
        rms = float(numpy.sqrt(numpy.dot(g, g) / len(g))) if len(g) > 1 else 0.0
        for where in b:
            if where != 'internal' and where not in a:
                # a patch whose value one writer leaves out (zeroGradient writes none) is not compared
                continue
            r = relative(a.get(where), b.get(where), 0.0 if where == 'internal' else rms)
            if r is None:
                print('FAIL %s %s: the two runs write different sizes' % (f, where))
                fails += 1
                continue
            parts.append((r, where))
        if not parts:
            continue
        r, where = max(parts)
        internal = [x for x, w in parts if w == 'internal'][0]
        flag = 'ok  ' if r <= bound else 'FAIL'
        print('%s %-8s internal %.3e  worst %-12s %.3e  (bound %g)' % (flag, f, internal, where, r, bound))
        if r > bound:
            fails += 1
        if r > worst[0]:
            worst = (r, f + ' ' + where)
    print('worst %.3e %s' % worst)
    return fails


def lag(run, ref, prev, bound, name):
    a, b, c = read(run + '/' + name), read(ref + '/' + name), read(prev + '/' + name)
    if a is None or b is None or c is None:
        print('FAIL %s: a file is missing' % name)
        return 1
    g = numpy.abs(b['internal'])
    rms = float(numpy.sqrt(numpy.dot(g, g) / len(g)))
    inside = relative(a['internal'], b['internal'])
    behind, same = [], []
    for where in b:
        if where == 'internal' or where not in a:
            continue
        r = relative(a.get(where), c.get(where), rms)
        behind.append((r if r is not None else float('inf'), where))
        same.append((relative(a.get(where), b.get(where), rms), where))
    w = max(behind)
    good = inside is not None and inside <= bound and w[0] <= bound
    print('%s %s internal %.3e, patches one correction behind %.3e (%s), against the same time %s' % (
        'ok' if good else 'FAIL', name, inside if inside is not None else float('inf'), w[0], w[1],
        ' '.join('%s %.1e' % (p, r) for r, p in sorted(same, key=lambda x: x[1]))))
    return 0 if good else 1


if __name__ == '__main__':
    what = sys.argv[1]
    if what == 'fields':
        sys.exit(1 if fields(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]) else 0)
    if what == 'lag':
        sys.exit(lag(sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]), sys.argv[6]))
    sys.exit('check_fields.py fields|lag ...')
