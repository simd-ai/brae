#!/usr/bin/env python3
"""Hold brae's simpleFoam stage dump against dumpSimpleFoam's, one SIMPLE iteration at a time.

  usage: compare_stages.py <OF time dir> <brae dump dir> <constant/polyMesh> [base time dir]

The OpenFOAM side is `dumpSimpleFoam` run with BRAE_DUMP_STAGE_ITER=n; the brae side is the V2 CUDA
driver run with BRAE_STAGE_DUMP_DIR=<dir> BRAE_STAGE_DUMP_ITER=n. Both read the same polyMesh, so cell
order and internal-face order are OpenFOAM's by construction; the flattened boundary array is brae's own
and is deliberately not compared here.

TWO MEASUREMENTS, and the second is the one that matters near a fixed point. The STATE comparison
(|OF - brae| / |OF|) says whether the two codes hold the same field. The INCREMENT comparison, given a
base time, compares what the ITERATION DID -- (stage - base) on each side. At a converged state the
increment is ~1e-5 of the state, so a state agreement of 1e-6 can hide an increment that is 100% wrong,
which is exactly what the iteration's stability depends on. Reading only the state number here once said
the momentum step agreed when its per-step increment was 2.7e-02 out.
"""
import gzip, os, re, sys
import numpy as np


def _read(p):
    if os.path.exists(p):       return open(p, 'rb').read()
    if os.path.exists(p+'.gz'): return gzip.open(p+'.gz', 'rb').read()
    raise IOError(p)


def of_internal(path):
    """internalField of an ascii OpenFOAM field -> (N,) or (N,3)."""
    s = _read(path).decode('utf-8', 'replace')
    tail = s[s.index('internalField'):]
    m = re.match(r'internalField\s+uniform\s+([^;]+);', tail)
    if m:
        tok = m.group(1).strip()
        if tok.startswith('('):
            return np.array([[float(x) for x in tok.strip('()').split()]])
        return np.array([float(tok)])
    m = re.match(r'internalField\s+nonuniform\s+List<(\w+)>\s*\n?\s*(\d+)\s*\(', tail)
    if not m:
        raise ValueError('cannot parse internalField of ' + path)
    kind, n = m.group(1), int(m.group(2))
    body = tail[m.end():]
    body = body[:body.index('\n)')]
    if kind == 'scalar':
        a = np.fromstring(body.replace('\n', ' '), sep=' ')
        assert a.size == n, (path, a.size, n)
        return a
    a = np.fromstring(body.replace('(', ' ').replace(')', ' ').replace('\n', ' '), sep=' ')
    assert a.size == 3*n, (path, a.size, n)
    return a.reshape(n, 3)


def of_list(path):
    """A bare `N ( ... )` list (polyMesh owner/neighbour)."""
    s = _read(path).decode('utf-8', 'replace')
    m = re.search(r'\n(\d+)\s*\n\(', s)
    body = s[m.end():]
    body = body[:body.index('\n)')]
    return np.fromstring(body.replace('\n', ' '), sep=' ').astype(np.int64)


def patches_of(meshdir):
    s = _read(os.path.join(meshdir, 'boundary')).decode('utf-8', 'replace')
    s = s[s.index('// * * *'):]
    out = []
    for m in re.finditer(r'(\w+)\s*\{([^}]*)\}', s):
        d = m.group(2)
        t  = re.search(r'\btype\s+(\w+)\s*;', d)
        nf = re.search(r'nFaces\s+(\d+)\s*;', d)
        sf = re.search(r'startFace\s+(\d+)\s*;', d)
        if nf and sf:
            out.append((m.group(1), t.group(1) if t else '?', int(nf.group(1)), int(sf.group(1))))
    return out


CELLS = {'stage_Uass': 'v', 'stage_V': 's', 'stage_nuEff': 's', 'stage_UDiag0': 's', 'stage_UDiag': 's',
         'stage_USrc': 'v', 'stage_rAU': 's', 'stage_rAtU': 's', 'stage_HbyA': 'v', 'stage_Upred': 'v',
         'stage_pOut': 's', 'stage_Uout': 'v'}
FACES = {'stage_phiU': 's', 'stage_UUpper': 's', 'stage_ULower': 's', 'stage_phiHbyA': 's',
         'stage_phiOut': 's'}
# What each increment is measured against, when a base time is given.
BASE = {'stage_Uass': 'U', 'stage_HbyA': 'U', 'stage_Upred': 'U', 'stage_Uout': 'U', 'stage_pOut': 'p',
        'stage_phiU': 'phi', 'stage_phiHbyA': 'phi', 'stage_phiOut': 'phi'}


def main():
    ofdir, brdir, meshdir = sys.argv[1], sys.argv[2], sys.argv[3]
    basedir = sys.argv[4] if len(sys.argv) > 4 else None

    own = of_list(os.path.join(meshdir, 'owner'))
    nei = of_list(os.path.join(meshdir, 'neighbour'))
    nCells = int(max(own.max(), nei.max())) + 1
    pats = patches_of(meshdir)

    touched = np.zeros(nCells, bool)
    wall = np.zeros(nCells, bool)
    # `empty` is not a boundary in the finite-volume sense: on a 2D mesh every cell touches one, so
    # counting it leaves the interior set empty and hides the split this comparison exists for.
    for nm, ty, nf, sf in pats:
        if ty == 'empty': continue
        touched[own[sf:sf+nf]] = True
        if ty == 'wall': wall[own[sf:sf+nf]] = True
    sets = [('all', None), ('interior', ~touched), ('patch-adj', touched), ('wall-adj', wall)]

    print(f'mesh: {nCells} cells, {nei.size} internal faces; {(~touched).sum()} interior, '
          f'{touched.sum()} patch-adjacent, {wall.sum()} wall-adjacent')
    for nm, ty, nf, sf in pats:
        print(f'   patch {nm:16s} {ty:10s} {nf} faces')

    def brae(name, kind):
        a = np.loadtxt(os.path.join(brdir, name))
        return a if kind == 'v' else (a[:, 0] if a.ndim == 2 else a)

    def rel(a, b):
        return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-300)

    def line(name, o, b, base, subsets):
        if o.ndim == b.ndim and o.shape[0] == 1 and b.shape[0] > 1:
            o = np.repeat(o, b.shape[0], axis=0)          # OF writes a uniform field as one row
        if o.shape != b.shape:
            print(f'  {name:22s} SHAPE {o.shape} vs {b.shape}'); return
        parts = []
        for label, idx in subsets:
            oo, bb = (o, b) if idx is None else (o[idx], b[idx])
            if oo.size: parts.append(f'{label} {rel(oo, bb):.3e}')
        out = f'  {name:22s} state ' + '  '.join(parts)
        if base is not None:
            do, db = o - base, b - base
            out += f'   | INCREMENT {rel(do, db):.3e} (|d_OF| {np.linalg.norm(do):.3e})'
        print(out)

    baseflds = {}
    if basedir:
        for f in ('U', 'p', 'phi'):
            try:    baseflds[f] = of_internal(os.path.join(basedir, f))
            except Exception: pass

    print('\nCELL FIELDS')
    for nm, kind in CELLS.items():
        try:    o = of_internal(os.path.join(ofdir, nm))
        except Exception: print(f'  {nm:22s} MISSING (OpenFOAM side)'); continue
        if not os.path.exists(os.path.join(brdir, nm)):
            print(f'  {nm:22s} MISSING (brae side)'); continue
        line(nm, o, brae(nm, kind), baseflds.get(BASE.get(nm)), sets)

    print('\nINTERNAL-FACE FIELDS')
    for nm, kind in FACES.items():
        try:    o = of_internal(os.path.join(ofdir, nm))
        except Exception: print(f'  {nm:22s} MISSING (OpenFOAM side)'); continue
        if not os.path.exists(os.path.join(brdir, nm)):
            print(f'  {nm:22s} MISSING (brae side)'); continue
        b = brae(nm, kind)
        n = min(o.shape[0], b.shape[0])
        bs = baseflds.get(BASE.get(nm))
        line(nm, o[:n], b[:n], None if bs is None else bs[:n], [('all', None)])

    # SIMPLEC. stage_rowSum is the literal 1/rAU - H1 that pEqn.H inverts, and rAtU is its reciprocal,
    # so this compares H1 without either side reconstructing it from two fields.
    try:
        rowOF = of_internal(os.path.join(ofdir, 'stage_rowSum'))
        rAtUb = brae('stage_rAtU', 's')
        rAtUo = of_internal(os.path.join(ofdir, 'stage_rAtU'))
        rAUo  = of_internal(os.path.join(ofdir, 'stage_rAU'))
        print('\nSIMPLEC')
        line('rowSum == 1/rAtU', rowOF, 1.0/rAtUb, None, sets)
        print(f'  rAtU/rAU   OF   [{(rAtUo/rAUo).min():.4f}, {(rAtUo/rAUo).max():.4f}]'
              f'   brae [{(rAtUb/brae("stage_rAU","s")).min():.4f}, '
              f'{(rAtUb/brae("stage_rAU","s")).max():.4f}]')
        print(f'  rAtU       OF   [{rAtUo.min():.6e}, {rAtUo.max():.6e}]'
              f'   brae [{rAtUb.min():.6e}, {rAtUb.max():.6e}]   negatives {(rAtUb < 0).sum()}')
        d = np.abs(rAtUo - rAtUb)/np.maximum(np.abs(rAtUo), 1e-300)
        print('  worst cells by relative rAtU error, with the patches each touches:')
        for c in np.argsort(-d)[:8]:
            pn = [nm for nm, ty, nf, sf in pats if c in own[sf:sf+nf]]
            print(f'    cell {c:7d}  OF {rAtUo[c]:.6e}  brae {rAtUb[c]:.6e}  rel {d[c]:.3e}  {pn}')
    except Exception as e:
        print('\nSIMPLEC: not comparable --', e)


main()
