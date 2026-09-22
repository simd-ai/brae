#!/usr/bin/env python3
"""Defaults left over from porting: a controls struct whose fields carry default member initialisers,
built BY HAND at more than one call site, with a field set at some sites and not at others.

That shape is a substitution waiting for a case: the site that forgot the field runs the default where
the case said otherwise, and nothing refuses. It is how the device driver's start-up CorrectPhi ran
grad(pcorr) as Gauss linear whatever fvSchemes named (2026-09-22, #28), and how the device loop would
have run a RAS closure on a moving mesh with ddt(k) on the current volumes and the absolute flux.

A field missing at one site is not always wrong -- a static-mesh site has no meshPhi -- so each
justified omission is listed in tools/default_audit_allow.txt with its reason, one per line:
    <Struct> <variable> <field>    # why the default IS the case's value at that site
The list is a ledger of defaults knowingly kept, not a suppression file: an entry leaves when the
field is wired or the sites are folded into one builder (correctPhiControlsOf), and an entry whose
struct, variable or field no longer exists is reported as stale so the ledger cannot rot.

usage: default_audit.py [--allow FILE] <dir>...      exit 1 on an unlisted omission or a stale entry
"""
import os
import re
import sys
from collections import defaultdict


def parse_structs(text):
    """{name: (file, [(field, defaulted)])} for structs with >= 3 data members and a default."""
    struct_re = re.compile(r'\bstruct\s+(\w+)\s*(?::\s*[^{]+)?\{(.*?)\n\};', re.S)
    field_re = re.compile(r'(?:mutable\s+|static\s+|const\s+)*[\w:<>\*&]+(?:\s*[\*&])?\s+'
                          r'(\w+(?:\s*,\s*\w+)*)\s*(=|\{|;)')
    out = {}
    for f, s in text.items():
        for m in struct_re.finditer(s):
            name, body = m.group(1), re.sub(r'//.*', '', m.group(2))
            fields = []
            for line in body.split('\n'):
                line = line.strip()
                if not line or line.startswith(('struct', 'enum', 'class', 'using', 'typedef', '#')):
                    continue
                if '(' in line and ')' in line and '=' not in line.split('(')[0]:
                    continue                                  # a method
                fm = re.match(field_re, line)
                if fm:
                    for n in fm.group(1).split(','):
                        fields.append((n.strip(), fm.group(2) in '={'))
            if len(fields) >= 3 and any(d for _, d in fields):
                out[name] = (f, fields)
    return out


def enclosing_scope(s, start):
    """The text from `start` to the close of the block the declaration sits in: what a hand-built
    struct can still be assigned in. A fixed window missed assignments 6000 characters down a long
    site and reported them as omissions."""
    depth = 0
    i = start
    n = len(s)
    while i < n:
        c = s[i]
        if c == '/' and s.startswith('//', i):
            i = s.find('\n', i)
            if i < 0:
                break
            continue
        if c == '/' and s.startswith('/*', i):
            j = s.find('*/', i + 2)
            i = n if j < 0 else j + 2
            continue
        if c == '"':
            j = i + 1
            while j < n and s[j] != '"':
                j += 2 if s[j] == '\\' else 1
            i = j + 1
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth < 0:
                return s[start:i]
        i += 1
    return s[start:]


def hand_built_sites(name, text):
    """[(file, line, var, assigned fields)] where `Type var;` is followed by var.field = ... ."""
    inst_re = re.compile(r'\b%s\s+(\w+)\s*(?:\{\s*\})?;' % re.escape(name))
    sites = []
    for f, s in text.items():
        for m in inst_re.finditer(s):
            var = m.group(1)
            window = enclosing_scope(s, m.end())
            assigned = set(re.findall(r'\b%s\.(\w+)\s*=' % re.escape(var), window))
            assigned |= set(re.findall(r'\b%s\.(\w+)\s*\.(?:push_back|assign|resize)' % re.escape(var), window))
            if assigned:
                sites.append((f, s[:m.start()].count('\n') + 1, var, assigned))
    return sites


def main(argv):
    allow_path = None
    dirs = []
    i = 0
    while i < len(argv):
        if argv[i] == '--allow':
            allow_path = argv[i + 1]
            i += 2
        else:
            dirs.append(argv[i])
            i += 1
    files = []
    for root in dirs:
        for dp, _, fs in os.walk(root):
            files.extend(os.path.join(dp, f) for f in fs if f.endswith(('.cu', '.cuh')))
    text = {f: open(f, errors='replace').read() for f in files}

    allow = {}
    if allow_path and os.path.exists(allow_path):
        for line in open(allow_path):
            body = line.split('#', 1)[0].strip()
            if body:
                parts = body.split()
                if len(parts) == 3:
                    allow[tuple(parts)] = line.strip()
    used = set()

    structs = parse_structs(text)
    unlisted = []
    evaluated = set()          # structs with >= 2 sites INSIDE this scan: the only ones whose ledger lines can be stale
    for name, (_, fields) in sorted(structs.items()):
        fieldnames = {n for n, _ in fields}
        sites = hand_built_sites(name, text)
        if len(sites) < 2:
            continue
        evaluated.add(name)
        union = set().union(*[a for _, _, _, a in sites])
        for f, line, var, assigned in sites:
            for missing in sorted((union - assigned) & fieldnames):
                key = (name, var, missing)
                if key in allow:
                    used.add(key)
                else:
                    unlisted.append((name, os.path.relpath(f), line, var, missing, len(sites)))

    for name, f, line, var, missing, n in unlisted:
        print("UNLISTED  %-26s %s:%d  %-10s field `%s` set at another of the %d sites, not here"
              % (name, f, line, var, missing, n))
    # a ledger line for a struct this scan did not evaluate (its other sites live outside the directories
    # given) is out of scope, not stale; the tree-wide run is the one that retires entries
    stale = sorted(k for k in set(allow) - used if k[0] in evaluated)
    for key in stale:
        print("STALE     %s -- no such omission any more; remove the entry" % allow[key])
    print("default_audit: %d structs with defaults, %d unlisted omissions, %d stale entries"
          % (len(structs), len(unlisted), len(stale)))
    return 1 if unlisted or stale else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
