#!/usr/bin/env python3
"""A device buffer that is ACCUMULATED into without being zeroed first.

DeviceBuffer::resize does NOT zero (device_buffer.cuh:134-141). It hands the old block back to the
pool and takes one out, and what the pool hands back is whatever the last owner of that block left in
it. Worse, `resize(n)` with n already n_ RETURNS IMMEDIATELY -- so the second call of a function that
resizes a scratch buffer keeps the FIRST call's values, byte for byte.

A kernel that ASSIGNS (`out[i] = ...`) does not care. A kernel that ACCUMULATES (`out[i] += ...`, or
atomicAdd into it) inherits that rubbish and adds to it. That is not hypothetical: interFoam's vector
limitedLinear built its magSqr limiter in a resized-but-unzeroed buffer and read U 5.1e-01 of
OpenFOAM's (2026-09-22, #28) -- green on the first step of a fresh process, wrong from the second.

WHAT IT CHECKS. For every `__global__` kernel, which POINTER PARAMETERS are accumulated into. Then
for every launch of such a kernel, the argument in that position, and the last thing the enclosing
function did to that buffer before the launch:

    zeroed        cudaMemset/deviceFill(0)/deviceZero/zeroBuffer, or a copyFrom/copyTo/assignment
                  that overwrites every element                              -- safe
    resized only  `buf.resize(n)` with no zero after it                      -- REPORTED
    untouched     the buffer is a parameter or a member this function only
                  passes on; whoever owns it decides                         -- reported as `carried`

`carried` is the common and usually correct case (a matrix's diag and source are assembled by an
earlier ASSIGNING kernel, then added to), so it is listed rather than failed, and a justified one goes
in the ledger. A RESIZED-THEN-ACCUMULATED site fails: that is the shape with the measurement behind it.

usage: resize_audit.py [--allow FILE] [--carried] <dir>...   exit 1 on an unzeroed accumulation
"""
import os
import re
import sys


def strip_comments(text):
    """Comments out, LENGTHS AND LINES KEPT (blanks in their place), so a reported line number is the
    file's own. This is not cosmetic: the audit's first run reported exactly one hit and it was the
    words `src[own] += ffc` inside a comment explaining the host's loop. The dangerous direction is
    the other one -- a `cudaMemset(buf...)` named only in a comment would have marked a real
    accumulation safe, and the audit would have reported nothing at all, which is what a suppression
    file looks like from the outside. tools/default_audit.py lost 24 sites to the same class."""
    out = []
    i, n = 0, len(text)
    while i < n:
        if text.startswith('//', i):
            j = text.find('\n', i)
            j = n if j < 0 else j
            out.append(' ' * (j - i))
            i = j
        elif text.startswith('/*', i):
            j = text.find('*/', i + 2)
            j = n if j < 0 else j + 2
            out.append(''.join(c if c == '\n' else ' ' for c in text[i:j]))
            i = j
        elif text[i] in '"\'':
            q = text[i]
            j = i + 1
            while j < n and text[j] != q:
                j += 2 if text[j] == '\\' else 1
            j = min(j + 1, n)
            out.append(' ' * (j - i))
            i = j
        else:
            out.append(text[i])
            i += 1
    return ''.join(out)

# what counts as putting a known value into every element of `buf`
# NO TRAILING \b: a buffer's name is often an ELEMENT -- `c.r[k]`, `ffc[0]` -- and there is no word
# boundary between `]` and the `,` that follows it, so `{b}\b` silently matched nothing and every
# deviceCopy into such a buffer read as "never initialised". That false positive was the audit's
# second finding about itself, after the comment one. The terminator is spelled out instead.
ZEROERS = (
    r'cudaMemset\w*\s*\(\s*{b}\s*[.,)]',
    r'\bdeviceFill\s*\(\s*{b}\s*[.,)]',
    r'\bdeviceZero\s*\(\s*{b}\s*[.,)]',
    r'\bzeroBuffer\s*\(\s*{b}\s*[.,)]',
    r'{b}\s*\.\s*copyFrom\s*\(',
    r'\bdeviceCopy\s*\(\s*{b}\s*[.,)]',
    r'\bdeviceScale\s*\(\s*{b}\s*[.,)]',
    # ...and a launch in the same function that ASSIGNS into it, which is recognised from the kernel
    # table rather than from a name (see the `assign2` walk below).
)


def kernels_of(text):
    """{kernel name: set(parameter names accumulated into)} for one translation unit."""
    out = {}
    for m in re.finditer(r'__global__\s+(?:static\s+)?\w[\w:<>, ]*\s+(\w+)\s*\(', text):
        name = m.group(1)
        i = m.end() - 1
        depth = 0
        j = i
        while j < len(text):
            if text[j] == '(':
                depth += 1
            elif text[j] == ')':
                depth -= 1
                if depth == 0:
                    break
            j += 1
        params = text[m.end():j]
        k = text.find('{', j)
        if k < 0:
            continue
        depth = 0
        e = k
        while e < len(text):
            if text[e] == '{':
                depth += 1
            elif text[e] == '}':
                depth -= 1
                if depth == 0:
                    break
            e += 1
        body = text[k:e]
        names = []
        for p in params.split(','):
            p = p.strip()
            mm = re.search(r'(\w+)\s*$', p)
            if mm and '*' in p:
                names.append(mm.group(1))
        acc, assign = set(), set()
        for pn in names:
            if (re.search(r'\b' + pn + r'\s*\[[^\]]*\]\s*\+=', body)
                    or re.search(r'atomicAdd\s*\(\s*&?\s*' + pn + r'\b', body)):
                acc.add(pn)
            if re.search(r'\b' + pn + r'\s*\[[^\]]*\]\s*=[^=]', body):
                assign.add(pn)
        out[name] = (names, acc, assign)
    return out


def function_at(text, pos):
    """(name, [parameter names]) of the function whose body `pos` sits in, or (None, [])."""
    open_brace = enclosing_function(text, pos)
    head = text[max(0, open_brace - 4000):open_brace]
    m = None
    for m in re.finditer(r'(\w+)\s*\(([^;{}]*)\)\s*$', head.rstrip()):
        pass
    if not m:
        return None, []
    names = []
    for prm in m.group(2).split(','):
        mm = re.search(r'(\w+)\s*$', prm.strip())
        if mm:
            names.append(mm.group(1))
    return m.group(1), names


def enclosing_function(text, pos):
    """Index of the enclosing FUNCTION body's opening brace -- the nearest one at column zero.

    Walking out one block at a time returns the nearest `{`, which for a launch inside an `if` is the
    IF's brace: the window then holds three lines and the `buf.resize(...)` two lines above the `if`
    is invisible, so the site is filed as `carried` and the defect is missed. That is exactly what
    happened to the injected P.source defect -- the audit named the second accumulation and not the
    first. This tree writes Allman braces, so a function body opens at column zero and nothing else
    does except a namespace, which is further out."""
    i = text.rfind('\n{', 0, pos)
    return i + 1 if i >= 0 else 0


def main(argv):
    allow_path = None
    show_carried = '--carried' in argv
    argv = [a for a in argv if a != '--carried']
    if '--allow' in argv:
        i = argv.index('--allow')
        allow_path = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    roots = argv or ['src']

    allow = {}
    if allow_path and os.path.exists(allow_path):
        for line in open(allow_path):
            line = line.split('#')[0].strip()
            if line:
                allow[line] = True

    files = []
    for root in roots:
        for r, _, ns in os.walk(root):
            for n in ns:
                if n.endswith(('.cu', '.cuh')):
                    files.append(os.path.join(r, n))
    files.sort()

    # the kernel table is per translation unit; a launch only ever names a kernel in its own file or
    # one it includes, and every accumulating kernel in this tree is launched in the file that defines
    # it -- checked by reporting a launch whose kernel is unknown.
    table = {}
    texts = {}
    for f in files:
        t = strip_comments(open(f, errors='ignore').read())
        texts[f] = t
        for k, v in kernels_of(t).items():
            table.setdefault(k, v)

    hits, carried, unknown = [], [], []
    # (function, parameter index) whose buffer is accumulated into and never zeroed inside it: the
    # caller owns the zeroing, so every call site is a question this audit has to ask
    risky = {}
    for f in files:
        t = texts[f]
        for m in re.finditer(r'\b(\w+)\s*<<<([^>]*)>>>\s*\(', t):
            kname = m.group(1)
            if kname not in table:
                unknown.append((f, kname))
                continue
            names, acc, assign = table[kname]
            if not acc:
                continue
            i = m.end() - 1
            depth = 0
            j = i
            while j < len(t):
                if t[j] == '(':
                    depth += 1
                elif t[j] == ')':
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            args, depth, cur = [], 0, ''
            for ch in t[m.end():j]:
                if ch in '(<[':
                    depth += 1
                elif ch in ')>]':
                    depth -= 1
                if ch == ',' and depth == 0:
                    args.append(cur.strip())
                    cur = ''
                else:
                    cur += ch
            args.append(cur.strip())
            start = enclosing_function(t, m.start())
            before = t[start:m.start()]
            line = t[:m.start()].count('\n') + 1
            for pn in sorted(acc):
                if pn not in names:
                    continue
                k = names.index(pn)
                # the pointer parameters are a subset of the parameter list, in order; map by counting
                ptr_args = [a for a in args if '.data()' in a or a.endswith('data()')]
                if k >= len(ptr_args):
                    continue
                arg = ptr_args[k]
                bm = re.search(r'([\w.\->\[\]]+)\s*\.\s*data\s*\(\)\s*$', arg)
                if not bm:
                    continue
                buf = bm.group(1)
                base = re.escape(buf)
                zeroed = any(re.search(z.format(b=base), before) for z in ZEROERS)
                resized = re.search(base + r'\s*\.\s*resize\s*\(', before)
                # ...or an earlier launch in this function that ASSIGNS into the same buffer
                for m2 in re.finditer(r'\b(\w+)\s*<<<[^>]*>>>\s*\(', before):
                    k2 = m2.group(1)
                    if k2 in table and buf in before[m2.end():before.find(';', m2.end())]:
                        _, _, assign2 = table[k2]
                        if assign2:
                            zeroed = True
                rel = os.path.relpath(f)
                key = '%s %s %s' % (os.path.basename(f), kname, buf)
                if zeroed:
                    continue
                if resized:
                    hits.append((rel, line, kname, pn, buf, key))
                else:
                    fname, fparams = function_at(t, m.start())
                    carried.append((rel, line, kname, pn, buf, key))
                    if fname and buf in fparams:
                        risky.setdefault((fname, fparams.index(buf)), []).append((rel, line, kname, buf))

    print('resize_audit: %d files, %d accumulating kernels' % (len(files), sum(1 for v in table.values() if v[1])))
    unlisted = [h for h in hits if h[5] not in allow]
    print('accumulated into a buffer this function RESIZED and did not zero: %d (%d unlisted)'
          % (len(hits), len(unlisted)))
    for rel, line, kname, pn, buf, key in hits:
        mark = ' ' if key in allow else '*'
        print('  %s %s:%d  %s accumulates into `%s` (%s)' % (mark, rel, line, kname, pn, buf))
    # ...and the CROSS-FUNCTION half: a caller that resizes the buffer it hands to one of those
    # functions and does not zero it. This is the shape the tree actually had -- the accumulation is
    # in a shared helper, the resize is in the driver, and neither file shows both.
    crossed = []
    for (fname, k), sites in sorted(risky.items()):
        for f in files:
            t = texts[f]
            for c in re.finditer(r'(?<![\w:>.])' + re.escape(fname) + r'\s*\(', t):
                head = t[max(0, c.start() - 40):c.start()]
                if '__global__' in head or '<<<' in t[c.end():c.end() + 40]:
                    continue
                i = c.end() - 1
                depth, j = 0, i
                while j < len(t):
                    if t[j] == '(':
                        depth += 1
                    elif t[j] == ')':
                        depth -= 1
                        if depth == 0:
                            break
                    j += 1
                args, depth, cur = [], 0, ''
                for ch in t[c.end():j]:
                    if ch in '(<[':
                        depth += 1
                    elif ch in ')>]':
                        depth -= 1
                    if ch == ',' and depth == 0:
                        args.append(cur.strip())
                        cur = ''
                    else:
                        cur += ch
                args.append(cur.strip())
                if k >= len(args):
                    continue
                bm = re.match(r'^[*&\s]*([\w.\->\[\]]+)$', args[k])
                if not bm:
                    continue
                buf = bm.group(1)
                base = re.escape(buf)
                start = enclosing_function(t, c.start())
                before = t[start:c.start()]
                if not re.search(base + r'\s*\.\s*resize\s*\(', before):
                    continue
                if any(re.search(z.format(b=base), before) for z in ZEROERS):
                    continue
                # ...or an earlier LAUNCH in this function that assigns into it, which is how an
                # explicit divergence fills its output before a deferred correction is added to it
                # (rhoEEqn.cu: explicitUpwindDivKernel writes `d[c] = s`, then deviceAxpy adds the
                # linearUpwind correction). Without this the audit called that a defect; it is the
                # same rule the within-function pass already applies.
                assigned = False
                for m2 in re.finditer(r'\b(\w+)\s*<<<[^>]*>>>\s*\(', before):
                    k2 = m2.group(1)
                    end = before.find(';', m2.end())
                    if k2 in table and table[k2][2] and buf in before[m2.end():end if end > 0 else len(before)]:
                        assigned = True
                        break
                if assigned:
                    continue
                rel = os.path.relpath(f)
                line = t[:c.start()].count('\n') + 1
                crossed.append((rel, line, fname, buf, sites[0][0], sites[0][1]))
    unlisted_cross = [c for c in crossed
                      if '%s %s %s' % (os.path.basename(c[0]), c[2], c[3]) not in allow]
    print('accumulated into a buffer the function only carries (owner decides): %d' % len(carried))
    print('...of those, a CALLER that resized the buffer and did not zero it: %d (%d unlisted)'
          % (len(crossed), len(unlisted_cross)))
    for rel, line, fname, buf, site, sline in crossed:
        mark = ' ' if '%s %s %s' % (os.path.basename(rel), fname, buf) in allow else '*'
        print('  %s %s:%d  %s(%s) -- accumulated at %s:%d' % (mark, rel, line, fname, buf, site, sline))
    if show_carried:
        for rel, line, kname, pn, buf, _ in carried:
            print('    %s:%d  %s += %s (%s)' % (rel, line, kname, pn, buf))
    keys = set(h[5] for h in hits)
    keys |= set('%s %s %s' % (os.path.basename(c[0]), c[2], c[3]) for c in crossed)
    stale = [a for a in allow if a not in keys]
    for a in sorted(stale):
        print('STALE ledger entry, no such site: %s' % a)
    return 1 if (unlisted or unlisted_cross or stale) else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
