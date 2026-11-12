#!/usr/bin/env python3
"""brae_patch_types_vs_factory.py <source root>: patchFieldTypeNames() in src/finiteVolume/fields/fv_patch_field.cuh is the list of
the boundary-condition factory's branches (makePatchFieldImpl), the list `brae capabilities` prints and `brae check` consults. It
must be that list exactly:
  every `d.type == "<name>"` branch of makePatchFieldImpl, and every coupled type isCoupledInterfaceType (foam_dict.cuh) takes,
  is in the table (a branch missing from it would be refused by the factory's first line and listed by nothing), and
  every name in the table has a branch (a name with none would be listed as implemented and refused at the chain's end);
  and the factory's first statement refuses a type outside the table.
CONTROL (the gate's own fail-proof, run every time): the same comparison on two mutated copies of the source, one with a table
name removed and one with a name added that has no branch, must each FAIL."""
import os
import re
import sys

ROOT = sys.argv[1]
FACTORY = os.path.join(ROOT, 'src/finiteVolume/fields/fv_patch_field.cuh')
COUPLED = os.path.join(ROOT, 'src/OpenFOAM/db/foam_dict.cuh')


def table(src):
    a = src.index('inline const std::vector<std::string>& patchFieldTypeNames()')
    b = src.index('return names;', a)
    return re.findall(r'"([^"]+)"', src[a:b])


def branches(src, coupled_src):
    a = src.index('std::unique_ptr<fvPatchField<T>> makePatchFieldImpl(const FvPatch& p, const PatchFieldData<T>& d)\n{')
    b = src.rindex('throw std::runtime_error("brae: unsupported BC type')
    body = src[a:b]
    names = set(re.findall(r'd\.type == "([^"]+)"', body))
    c = coupled_src.index('inline bool isCoupledInterfaceType')
    names |= set(re.findall(r't == "([^"]+)"', coupled_src[c:coupled_src.index('}', c)]))
    return names, body


def compare(src, coupled_src):
    t = table(src)
    br, body = branches(src, coupled_src)
    problems = []
    for n in sorted(br - set(t)):
        problems.append('branch %s is not in patchFieldTypeNames' % n)
    for n in sorted(set(t) - br):
        problems.append('patchFieldTypeNames lists %s, which no branch takes' % n)
    if len(t) != len(set(t)):
        problems.append('patchFieldTypeNames lists a name twice')
    first = body[body.index('{') + 1:].lstrip()
    while first.startswith('//'):
        first = first[first.index('\n') + 1:].lstrip()
    if not first.startswith('if (!isPatchFieldTypeName(d.type)) throw'):
        problems.append('makePatchFieldImpl does not begin by refusing a type outside the table')
    return t, br, problems


src = open(FACTORY).read()
coupled_src = open(COUPLED).read()
t, br, problems = compare(src, coupled_src)
fail = 0
if problems:
    for p in problems:
        print('FAIL: %s' % p)
    fail = 1
else:
    print('ok:   patchFieldTypeNames is the factory\'s %d branches (%d coupled types among them), and the factory refuses '
          'any other type first' % (len(t), len(set(t) & {'cyclic', 'cyclicAMI', 'cyclicACMI', 'cyclicPeriodicAMI'})))

# CONTROL: a name removed from the table, and a name added without a branch, must each be caught
victim = t[len(t) // 2]
removed = src.replace('"%s", ' % victim, '', 1)
added = src.replace('"zeroGradient",', '"zeroGradient", "notABranchOfTheFactory",', 1)
for label, mutated in (('a table name removed (%s)' % victim, removed), ('a name added with no branch', added)):
    if mutated == src:
        print('FAIL: CONTROL %s: the mutation did not change the source' % label)
        fail = 1
        continue
    _, _, p = compare(mutated, coupled_src)
    if p:
        print('ok:   CONTROL %s caught: %s' % (label, p[0]))
    else:
        print('FAIL: CONTROL %s was not caught' % label)
        fail = 1
print('PASS' if fail == 0 else 'FAIL')
sys.exit(fail)
