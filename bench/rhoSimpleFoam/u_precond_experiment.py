#!/usr/bin/env python3
"""Momentum-solver experiment on a DUMPED brae momentum system (BRAE_STAGE_DUMP_DIR at one iteration):
how many iterations each candidate needs to reach OpenFOAM's stopping rule (relTol 0.1 on the
normFactor-scaled residual, lduMatrix::solver::normFactor) on the matrix brae actually solves.

Usage: u_precond_experiment.py <dump dir> <polyMesh dir> [relTol]
Files: UsolveDiag{X,Y,Z}, UsolveB{X,Y,Z} (the folded, relaxed system per component), UUpper, ULower,
       Uass (the initial guess, three columns).
Candidates: BiCGStab with Jacobi / DILU (OpenFOAM's) / red-black symmetric Gauss-Seidel / 3-step Jacobi
polynomial; and plain red-black Gauss-Seidel and plain Jacobi sweeps with no Krylov at all. Iteration
counts only -- what to build is decided from them, not from taste.
"""
import re, sys, time
import numpy as np
import scipy.sparse as sp

dump, mesh = sys.argv[1], sys.argv[2]
relTol = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
MAXIT = 300

def readlist(fn):
    b = open(fn, 'rb').read()
    m = re.search(rb'\n(\d+)\s*\n\(', b); n = int(m.group(1)); start = m.end()
    fmt = re.search(rb'format\s+(\w+)', b[:800]).group(1)
    if fmt == b'binary': return np.frombuffer(b[start:start + n * 4], dtype='<i4').astype(np.int64)
    return np.array([int(x) for x in b[start:].split(b')')[0].split()], dtype=np.int64)

own = readlist(mesh + '/owner'); nei = readlist(mesh + '/neighbour'); nF = len(nei); own = own[:nF]
nC = int(max(own.max(), nei.max())) + 1
upper = np.loadtxt(dump + '/UUpper'); lower = np.loadtxt(dump + '/ULower')
assert upper.size == nF and lower.size == nF, (upper.size, lower.size, nF)
# OpenFOAM lduMatrix::Amul: Apsi[u] += lower[f]*psi[l]; Apsi[l] += upper[f]*psi[u]  (l = owner, u = neighbour)
def matrix(diag):
    rows = np.concatenate([np.arange(nC), nei, own]); cols = np.concatenate([np.arange(nC), own, nei])
    vals = np.concatenate([diag, lower, upper])
    return sp.csr_matrix((vals, (rows, cols)), shape=(nC, nC))

# red-black colouring (the mesh is bipartite on this case; the script says so if it is not)
adj_rows = np.concatenate([own, nei]); adj_cols = np.concatenate([nei, own])
G = sp.csr_matrix((np.ones(2 * nF), (adj_rows, adj_cols)), shape=(nC, nC))
colour = -np.ones(nC, dtype=np.int64); colour[0] = 0; frontier = np.array([0])
while frontier.size:
    nb = G[frontier].nonzero()[1]; nb = np.unique(nb); nxt = nb[colour[nb] < 0]
    colour[nxt] = 1 - colour[frontier[0]]
    frontier = nxt
bip = not np.any(colour[adj_rows] == colour[adj_cols]) and not np.any(colour < 0)
red = np.where(colour == 0)[0]; black = np.where(colour == 1)[0]
print('cells %d, faces %d, bipartite %s (red %d, black %d)' % (nC, nF, bip, red.size, black.size))

def normFactor(A, x, b):
    xRef = x.mean(); A1 = A @ np.ones(nC)
    return np.sum(np.abs(A @ x - xRef * A1) + np.abs(b - xRef * A1)) + 1e-20

def resid(A, x, b, nf): return np.sum(np.abs(b - A @ x)) / nf

# ---- preconditioners: each returns a function w = M^-1 r ---------------------------------------------
def prec_jacobi(A, diag):
    rD = 1.0 / diag
    return lambda r: rD * r

def prec_poly(A, diag, k=3):   # M^-1 = sum_{j<k} (I - D^-1 A)^j D^-1 : k Jacobi-like sweeps, no sequential part
    rD = 1.0 / diag
    def apply(r):
        w = rD * r; t = w.copy()
        for _ in range(k - 1):
            t = t - rD * (A @ t)
            w = w + t
        return w
    return apply

def prec_dilu(A, diag):   # OpenFOAM DILUPreconditioner: calcReciprocalD + the two face sweeps, sequential
    rD = diag.copy()
    ownl, neil = own.tolist(), nei.tolist(); ul, ll = upper.tolist(), lower.tolist()
    for f in range(nF):
        rD[neil[f]] -= ul[f] * ll[f] / rD[ownl[f]]
    rD = 1.0 / rD
    rDl = rD.tolist()
    def apply(r):
        w = (rD * r).tolist()
        for f in range(nF):                       # forward:  wA[u] -= rD[u]*lower[f]*wA[l]
            w[neil[f]] -= rDl[neil[f]] * ll[f] * w[ownl[f]]
        for f in range(nF - 1, -1, -1):           # backward: wA[l] -= rD[l]*upper[f]*wA[u]
            w[ownl[f]] -= rDl[ownl[f]] * ul[f] * w[neil[f]]
        return np.array(w)
    return apply

def prec_rbsgs(A, diag, sweeps=1):   # symmetric red-black Gauss-Seidel: red, black, black, red per sweep
    rD = 1.0 / diag
    Ared = A[red]; Ablack = A[black]          # rows
    def apply(r):
        w = np.zeros(nC)
        for _ in range(sweeps):
            w[red]   = rD[red]   * (r[red]   - (Ared   @ w) + diag[red]   * w[red])
            w[black] = rD[black] * (r[black] - (Ablack @ w) + diag[black] * w[black])
            w[black] = rD[black] * (r[black] - (Ablack @ w) + diag[black] * w[black])
            w[red]   = rD[red]   * (r[red]   - (Ared   @ w) + diag[red]   * w[red])
        return w
    return apply

# ---- solvers -----------------------------------------------------------------------------------------
def bicgstab(A, b, x0, prec, nf, label):
    x = x0.copy(); r = b - A @ x; r0 = r.copy()
    res0 = np.sum(np.abs(r)) / nf; res = res0
    if res0 == 0: print('  %-28s initial residual 0' % label); return 0
    rho_old = alpha = omega = 1.0; p = np.zeros(nC); v = np.zeros(nC)
    hist = [res0]; n = 0
    while n < MAXIT and res >= relTol * res0:
        rho = r0 @ r
        if n == 0: p = r.copy()
        else:
            beta = (rho / rho_old) * (alpha / omega); p = r + beta * (p - omega * v)
        y = prec(p); v = A @ y; alpha = rho / (r0 @ v)
        s = r - alpha * v; x = x + alpha * y
        res = np.sum(np.abs(s)) / nf; n += 1
        if res < relTol * res0: hist.append(res); break
        z = prec(s); t = A @ z; omega = (t @ s) / (t @ t)
        x = x + omega * z; r = s - omega * t
        res = np.sum(np.abs(r)) / nf; rho_old = rho; hist.append(res)
    print('  %-28s %3d iterations   residual %.3e -> %.3e (%.3f of initial)' % (label, n, res0, res, res / res0))
    return n

def sweeps(A, b, x0, diag, kind, nf, label):
    x = x0.copy(); rD = 1.0 / diag
    res0 = np.sum(np.abs(b - A @ x)) / nf; res = res0; n = 0
    Ared = A[red]; Ablack = A[black]
    while n < MAXIT and res >= relTol * res0:
        if kind == 'jacobi':
            x = x + rD * (b - A @ x)
        else:   # one red-black Gauss-Seidel sweep (forward)
            x[red]   = rD[red]   * (b[red]   - (Ared   @ x) + diag[red]   * x[red])
            x[black] = rD[black] * (b[black] - (Ablack @ x) + diag[black] * x[black])
        res = np.sum(np.abs(b - A @ x)) / nf; n += 1
    print('  %-28s %3d sweeps       residual %.3e -> %.3e (%.3f of initial)' % (label, n, res0, res, res / res0))
    return n

U0 = np.loadtxt(dump + '/Uass')
for ci, comp in enumerate('XYZ'):
    try:
        diag = np.loadtxt(dump + '/UsolveDiag' + comp); b = np.loadtxt(dump + '/UsolveB' + comp)
    except OSError:
        print('component %s: no UsolveDiag/UsolveB dump, skipped' % comp); continue
    A = matrix(diag); x0 = U0[:, ci] if U0.ndim == 2 else U0
    nf = normFactor(A, x0, b)
    dd = np.abs(diag) / (np.abs(A) @ np.ones(nC) - np.abs(diag) + 1e-300)
    print('component %s: diagonal dominance |a_ii|/sum|a_ij|  min %.2f  median %.2f' % (comp, dd.min(), np.median(dd)))
    t = time.time()
    bicgstab(A, b, x0, prec_jacobi(A, diag), nf, 'BiCGStab + Jacobi')
    bicgstab(A, b, x0, prec_poly(A, diag, 3), nf, 'BiCGStab + 3-step polynomial')
    if bip:
        bicgstab(A, b, x0, prec_rbsgs(A, diag, 1), nf, 'BiCGStab + red-black SGS')
        sweeps(A, b, x0, diag, 'rbgs', nf, 'red-black GS sweeps alone')
    sweeps(A, b, x0, diag, 'jacobi', nf, 'Jacobi sweeps alone')
    bicgstab(A, b, x0, prec_dilu(A, diag), nf, 'BiCGStab + DILU (OpenFOAM)')
    print('  (%.0f s)' % (time.time() - t))
