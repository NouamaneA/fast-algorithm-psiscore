import scipy.sparse as sps
import numpy as np
from scipy.sparse.linalg import norm
from time import time
from utils import l_plus_m, dict_to_sparse_matrix
from progressbar import progressbar

def propagation_matrix(L_dict, ls, ms, lpms, deg=False):
    N = len(L_dict)
    A = dict()
    B = {i: {} for i in range(N)}
    Deg = N*[0]
    c = []
    d = []
    for j in L_dict:
        A[j] = dict()
        for k in L_dict[j]:
            A[j][k] = ms[k] / lpms[j]
            B[j][k] = ls[k] / lpms[j]
            Deg[j] += 1
        if ls[j]+ms[j] == 0:
            c.append(0)
            d.append(0)
        else:
            c.append(ms[j]/(ls[j]+ms[j]))
            d.append(ls[j]/(ls[j]+ms[j]))

    A = dict_to_sparse_matrix(A, shape=(N, N))
    B = dict_to_sparse_matrix(B, shape=(N, N))
    if deg:
        return A, B, c, d, np.array(Deg)
    else:
        return A, B, c, d

def power_psi(L_dict, ls, ms, n_max=1000, eps=1e-9):
    N = len(L_dict)
    lpms = l_plus_m(L_dict, ls, ms)
    A, B, c, d, Deg = propagation_matrix(L_dict, ls, ms, lpms, deg=True)
    t = time()
    At = A.T 
    S = c
    B_norm = norm(B, ord=1)
    n_msg = 0
    n_mult = 0
    for i in progressbar(range(n_max)):
        S_old = S.copy()
        nnz_idx = np.where(S != 0)[0]
        n_msg += Deg[nnz_idx].sum()
        S = At.dot(S) + c
        n_mult += 1
        gap = sum(abs(S - S_old))
        gap = gap * B_norm
        if gap < eps:
            Psi = 1/N * (B.T.dot(S) + d)
            n_mult += 1
            duration = time() - t
        return Psi, duration, n_msg, n_mult
    raise RuntimeError(f"Psi-score: power method failed to converge at {n_max} iterations.")

def psi_solve(L_dict, ls, ms):
    N = len(L_dict)
    lpms = l_plus_m(L_dict, ls, ms)
    A, B, c, d = propagation_matrix(L_dict, ls, ms, lpms)
    t = time()
    P = sps.linalg.spsolve((sps.identity(N)-A).tocsc(), B.tocsc())
    Psi = 1/N * (P.T.dot(c) + d)
    duration = time() - t
    return Psi, duration

# Below, the default way: Power-NF

def X(u, N):
    x = np.zeros(N)
    x[u] = 1
    return x

def power_newsfeed(A, b_i, Deg, n_max=500, eps=1e-9):
    N = A.shape[0]
    p_i = b_i.copy()
    n_mult = 0
    n_msg = 0
    for _ in range(n_max):
        p_i_old = p_i.copy()
        nnz_idx = np.where(p_i != 0)[0]
        n_msg += Deg[nnz_idx].sum()     
        p_i = A.dot(p_i) + b_i
        n_mult += 1
        gap = np.sum(abs(p_i - p_i_old))
        # gap = norm(p_i - p_i_old, ord=1)
        if gap < eps:
            return p_i, n_msg, n_mult
    raise RuntimeError(f"Psi-Score: steady-state newsfeed power-method failed to converge in {n_max} iterations.")

def wall_steady_state(i, c, p_i, d):
    N = len(d)
    d_i = d[i]*X(i, N)
    q_i = c * p_i + d_i
    return q_i

def psi_score_old(L_dict, ls, ms, n_max=1000, eps=1e-6):
    N = len(L_dict)
    lpms = l_plus_m(L_dict, ls, ms)
    A, B, c, d, Deg = propagation_matrix(L_dict, ls, ms, lpms, deg=True)
    c = np.array(c)
    d = np.array(d)
    t = time()
    psi = []
    n_msg = 0
    n_mult = 0
    for i in progressbar(range(N)):
        b_i = B.dot(X(i, N))
        p_i, n_msg_i, n_mult_i = power_newsfeed(A, b_i, Deg, n_max=n_max, eps=eps)
        n_mult += n_mult_i
        n_msg += n_msg_i
        q_i = wall_steady_state(i, c, p_i, d)
        psi.append(1/N * np.sum(q_i))
    duration = time() - t
    return psi, duration, n_msg, n_mult