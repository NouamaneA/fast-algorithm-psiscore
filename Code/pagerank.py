import numpy as np
import scipy.sparse as sps
from progressbar import progressbar
from time import time

def to_stochastic(A):
    
    np.seterr(divide='ignore', invalid='ignore')
    N = A.shape[0]
    D_inv = np.array(1/A.sum(axis=1)).reshape(N,)
    D_inv = sps.csr_matrix((D_inv, (list(range(N)), list(range(N)))), shape=(N, N))
    P = D_inv @ A

    return P

def power_iter(P, pi, Deg, alpha=0.85, n_max=500, eps=1e-9):
    N = P.shape[0]
    Pt = P.T
    n_mult = 0
    n_msg = 0
    for _ in progressbar(range(n_max)):
        pi_old = pi.copy()
        nnz_idx = np.where(pi != 0)[0]
        n_msg += Deg[nnz_idx].sum()
        pi = alpha * Pt.dot(pi) + [(1-alpha)/N] * N
        n_mult += 1
        gap = np.sum(abs(pi - pi_old))
        if N*gap < eps:
            return pi, n_msg, n_mult
    raise RuntimeError(f"PageRank: power iteration failed to converge in {n_max} iterations.")

def pagerank(A, alpha=0.85, max_iter=500, eps=1e-9):
    N = A.shape[0]
    P = to_stochastic(A)
    Deg = np.array(A.sum(axis=1)).reshape(N,)
    pi = [1/N] * N
    t = time()
    pi, n_msg, n_mult = power_iter(P, pi, Deg, alpha=alpha, n_max=max_iter, eps=eps)
    duration = time() - t
    return pi, duration, n_msg, n_mult
