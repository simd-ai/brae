#!/usr/bin/env python3
"""Which PARALLEL preconditioner puts a substituted PBiCGStab where DILU puts it, on the epsilon system
brae actually solves (item 78).

DILU is the preconditioner that keeps epsilon positive, and it is the one a GPU cannot afford: its apply
is a level-scheduled sequential walk, one kernel launch per dependency level (376 of them at 307k cells).
The question this answers is whether a FULLY PARALLEL preconditioner reaches the same stopping point.
Candidates, all of them O(1) kernel launches per apply:

    Jacobi                the current substitute; the thing that fails
    red-black DILU        DILU on a two-colour reordering: the red block is diagonal (red cells share no
                          face), so eliminating it is exact, and the black Schur complement is
                          approximated by its diagonal exactly as DILU does. nColours launches per
                          half-sweep, not nLevels.
    red-black SGS         symmetric Gauss-Seidel by colour
    Neumann polynomial    M^-1 = sum_{j<k} (I - D^-1 A)^j D^-1, k SpMVs, no dependencies at all
    DILU (natural order)  the reference: what OpenFOAM computes and what brae runs today

WHAT IS REPORTED is not the iteration count. It is where the solve STOPS under OpenFOAM's own rule
(relTol on the normFactor-scaled residual) and what the returned field looks like there -- specifically
min(x), because the failure being chased is epsilon going non-positive and being floored at 1e-15 by
bound(), after which nut = Cmu k^2/epsilon explodes.

Usage: eps_precond_experiment.py <dump dir> <polyMesh dir> [relTol] [field prefix, default eps]
Files: <prefix>D, <prefix>Src, <prefix>Upper, <prefix>Lower, <prefix>SolveIn (the folded system and the
initial guess). The prefix is what kEpsilon.cu's dump() writes: `eps` and `k`.
"""
import re, sys, time
import numpy as np
import scipy.sparse as sp

dump, mesh = sys.argv[1], sys.argv[2]
relTol = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
FLD = sys.argv[4] if len(sys.argv) > 4 else 'eps'
MAXIT = 400

def readlist(fn):
    b = open(fn, 'rb').read()
    m = re.search(rb'\n(\d+)\s*\n\(', b); n = int(m.group(1)); start = m.end()
    fmt = re.search(rb'format\s+(\w+)', b[:800]).group(1)
    if fmt == b'binary': return np.frombuffer(b[start:start + n * 4], dtype='<i4').astype(np.int64)
    return np.array([int(x) for x in b[start:].split(b')')[0].split()], dtype=np.int64)

own = readlist(mesh + '/owner'); nei = readlist(mesh + '/neighbour'); nF = len(nei); own = own[:nF]
nC = int(max(own.max(), nei.max())) + 1
diag = np.loadtxt(dump + '/' + FLD + 'D'); b = np.loadtxt(dump + '/' + FLD + 'Src')
upper = np.loadtxt(dump + '/' + FLD + 'Upper'); lower = np.loadtxt(dump + '/' + FLD + 'Lower')
x0 = np.loadtxt(dump + '/' + FLD + 'SolveIn')
assert upper.size == nF and lower.size == nF and diag.size == nC, (upper.size, lower.size, nF, diag.size)

# OpenFOAM lduMatrix::Amul: Apsi[nei] += lower[f]*psi[own], Apsi[own] += upper[f]*psi[nei]
rows = np.concatenate([np.arange(nC), nei, own]); cols = np.concatenate([np.arange(nC), own, nei])
A = sp.csr_matrix((np.concatenate([diag, lower, upper]), (rows, cols)), shape=(nC, nC))

# two-colouring of the face graph (a hex mesh is bipartite; the script says so if it is not)
adj = sp.csr_matrix((np.ones(2 * nF), (np.concatenate([own, nei]), np.concatenate([nei, own]))), shape=(nC, nC))
colour = -np.ones(nC, dtype=np.int64); colour[0] = 0; frontier = np.array([0])
while frontier.size:
    nb = np.unique(adj[frontier].nonzero()[1]); nxt = nb[colour[nb] < 0]
    colour[nxt] = 1 - colour[frontier[0]]; frontier = nxt
bip = not np.any(colour[own] == colour[nei]) and not np.any(colour < 0)
red = np.where(colour == 0)[0]; black = np.where(colour == 1)[0]
print('cells %d, faces %d, bipartite %s (red %d, black %d)' % (nC, nF, bip, red.size, black.size))
print('initial %s: min %.4e  max %.4e' % (FLD, x0.min(), x0.max()))

def normFactor(x):
    xRef = x.mean(); A1 = A @ np.ones(nC)
    return np.sum(np.abs(A @ x - xRef * A1) + np.abs(b - xRef * A1)) + 1e-20

# ---- preconditioners -------------------------------------------------------------------------------
def prec_jacobi():
    rD = 1.0 / diag
    return lambda r: rD * r, 'O(1) launches'

def prec_poly(k=3):
    """Truncated Neumann series, M^-1 = sum_{j<k} (I - D^-1 A)^j D^-1. Converges iff the iteration matrix
    I - D^-1 A has spectral radius below 1, i.e. iff the matrix is diagonally dominant -- which the
    epsilon equation is, strongly, because of its Sp reaction term. k-1 SpMVs per apply and nothing
    else: no factorisation, no ordering, no dependency between cells."""
    rD = 1.0 / diag
    def apply(r):
        w = rD * r; t = w.copy()
        for _ in range(k - 1):
            t = t - rD * (A @ t); w = w + t
        return w
    return apply, '%d SpMV' % (k - 1)

def prec_cheb(deg, lo_frac=1.0/30.0):
    """Chebyshev polynomial in D^-1 A over [lambdaMax/30, lambdaMax], the same interval brae's AMG
    smoother uses (CHEB_EIGRATIO). Optimal in the min-max sense over that interval, so it should beat a
    truncated Neumann series of the same degree -- at the same cost, deg SpMVs."""
    rD = 1.0 / diag
    # lambdaMax(D^-1 A) by a short power iteration, over-covered as the AMG does (CHEB_UPPER_SAFETY)
    v = np.random.default_rng(0).standard_normal(nC)
    for _ in range(30):
        v = rD * (A @ v); nv = np.linalg.norm(v)
        if nv == 0: break
        v /= nv
    lmax = 1.2 * float(v @ (rD * (A @ v)) / (v @ v))
    lmin = lo_frac * lmax
    d0 = (lmax + lmin) / 2.0; c = (lmax - lmin) / 2.0
    def apply(r):
        x = np.zeros(nC); rr = r.copy(); p = np.zeros(nC); alpha = beta = 0.0
        for i in range(deg):
            z = rD * rr
            if i == 0:   p = z; alpha = 1.0 / d0
            elif i == 1: beta = 0.5 * (c * alpha) ** 2; alpha = 1.0 / (d0 - beta / alpha); p = z + beta * p
            else:        beta = (c * alpha / 2.0) ** 2; alpha = 1.0 / (d0 - beta / alpha); p = z + beta * p
            x = x + alpha * p
            rr = rr - alpha * (A @ p)
        return x
    return apply, 'Chebyshev deg %d (%d SpMV)' % (deg, deg)

def prec_dilu_natural():
    """OpenFOAM DILUPreconditioner: calcReciprocalD then the two face sweeps, in FACE ORDER."""
    rD = diag.copy()
    ownl, neil, ul, ll = own.tolist(), nei.tolist(), upper.tolist(), lower.tolist()
    for f in range(nF): rD[neil[f]] -= ul[f] * ll[f] / rD[ownl[f]]
    rD = 1.0 / rD; rDl = rD.tolist()
    def apply(r):
        w = (rD * r).tolist()
        for f in range(nF):            w[neil[f]] -= rDl[neil[f]] * ll[f] * w[ownl[f]]
        for f in range(nF - 1, -1, -1): w[ownl[f]] -= rDl[ownl[f]] * ul[f] * w[neil[f]]
        return np.array(w)
    return apply, 'nLevels launches (sequential)'

def prec_dilu_rb():
    """DILU on the RED-BLACK ordering. Red cells share no face, so eliminating them is exact and can be
    done in one parallel step; the black diagonal then takes DILU's rank-1 correction from every red
    neighbour. Apply is two steps forward and two back -- nColours, not nLevels."""
    isRedOwn = colour[own] == 0                      # face f: red -> black when true
    rD = diag.copy()
    # black diagonal correction: for every face joining a red owner to a black neighbour and vice versa,
    # the same product upper*lower/rD[red] that natural-order DILU subtracts, gathered onto the black cell
    contrib = upper * lower
    np.add.at(rD, np.where(isRedOwn, nei, own), -contrib / diag[np.where(isRedOwn, own, nei)])
    rDi = 1.0 / rD
    # the four scatter lists, precomputed
    fRB = np.where(isRedOwn)[0]; fBR = np.where(~isRedOwn)[0]
    def apply(r):
        w = rDi * r
        # forward: black -= rD[black] * (lower or upper) * w[red]
        upd = np.zeros(nC)
        np.add.at(upd, nei[fRB], lower[fRB] * w[own[fRB]])
        np.add.at(upd, own[fBR], upper[fBR] * w[nei[fBR]])
        w = w - rDi * upd
        # backward: red -= rD[red] * (upper or lower) * w[black]
        upd = np.zeros(nC)
        np.add.at(upd, own[fRB], upper[fRB] * w[nei[fRB]])
        np.add.at(upd, nei[fBR], lower[fBR] * w[own[fBR]])
        w = w - rDi * upd
        return w
    return apply, '2 colours (4 launches)'

def prec_rbsgs(sweeps=1):
    rD = 1.0 / diag; Ared = A[red]; Ablack = A[black]
    def apply(r):
        w = np.zeros(nC)
        for _ in range(sweeps):
            w[red]   = rD[red]   * (r[red]   - (Ared   @ w) + diag[red]   * w[red])
            w[black] = rD[black] * (r[black] - (Ablack @ w) + diag[black] * w[black])
            w[black] = rD[black] * (r[black] - (Ablack @ w) + diag[black] * w[black])
            w[red]   = rD[red]   * (r[red]   - (Ared   @ w) + diag[red]   * w[red])
        return w
    return apply, '2 colours x %d sweeps' % sweeps

# THE REFERENCE SOLUTION. What separates these preconditioners is not whether they converge -- they all
# do -- but WHERE THE ITERATE IS when OpenFOAM's relTol stops them. That error is what the next outer
# iteration inherits, and it is what compounds into the collapse over eight of them, so it is the
# quantity to rank on. Solved here to 1e-14, far past anything the SIMPLE loop asks for.
xExact = None

# ---- OpenFOAM's PBiCGStab, stopped by OpenFOAM's rule ----------------------------------------------
def bicgstab(prec, nf, label, cost):
    x = x0.copy(); r = b - A @ x; r0 = r.copy()
    res0 = np.sum(np.abs(r)) / nf; res = res0
    if res0 == 0: print('  %-24s initial residual 0' % label); return
    rho_old = alpha = omega = 1.0; p = np.zeros(nC); v = np.zeros(nC); n = 0
    t0 = time.time()
    while n < MAXIT and res >= relTol * res0:
        rho = r0 @ r
        p = r.copy() if n == 0 else r + (rho / rho_old) * (alpha / omega) * (p - omega * v)
        y = prec(p); v = A @ y; alpha = rho / (r0 @ v)
        s = r - alpha * v; x = x + alpha * y
        res = np.sum(np.abs(s)) / nf; n += 1
        if res < relTol * res0: break
        z = prec(s); t = A @ z; omega = (t @ s) / (t @ t)
        x = x + omega * z; r = s - omega * t
        res = np.sum(np.abs(r)) / nf; rho_old = rho
    err = np.linalg.norm(x - xExact) / np.linalg.norm(xExact) if xExact is not None else float('nan')
    print('  %-24s %3d it  res %.3e -> %.3e   |x-x*|/|x*| %.3e   min(%s) %10.4e   [%s, %.1fs]'
          % (label, n, res0, res, err, FLD, x.min(), cost, time.time() - t0))
    return x

nf = normFactor(x0)
print('normFactor %.4e, relTol %g -- OpenFOAM\'s own stopping rule' % (nf, relTol))
import scipy.sparse.linalg as spla
t0 = time.time()
xExact = spla.spsolve(A.tocsc(), b)
print('reference solution by direct sparse LU (%.1fs): min %.4e  max %.4e\n'
      % (time.time() - t0, xExact.min(), xExact.max()))
print('  %-24s %s' % ('preconditioner', 'where the solve stops, and how far that is from the answer'))
for maker, name in ((prec_jacobi, 'Jacobi (today)'),
                    (lambda: prec_poly(3), 'Neumann 3'),
                    (lambda: prec_poly(6), 'Neumann 6'),
                    (lambda: prec_poly(10), 'Neumann 10'),
                    (lambda: prec_poly(16), 'Neumann 16'),
                    (lambda: prec_cheb(3), 'Chebyshev 3'),
                    (lambda: prec_cheb(6), 'Chebyshev 6'),
                    (lambda: prec_cheb(10), 'Chebyshev 10'),
                    (prec_rbsgs, 'red-black SGS'),
                    (lambda: prec_rbsgs(2), 'red-black SGS x2'),
                    (prec_dilu_rb, 'red-black DILU'),
                    (prec_dilu_natural, 'DILU natural (OF)')):
    if not bip and name.startswith('red-black'): continue
    f, cost = maker()
    bicgstab(f, nf, name, cost)
