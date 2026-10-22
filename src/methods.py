"""The recommenders. Each takes a Split and returns a dense float32 score matrix, users x candidates.

Interactions are the n80 counts (airings a user watched >= 80 % of in one view) over the history
weeks: the probe in experiments/EDA.md §8b showed the target behaves like n80. Already-watched
programmes are allowed in every list. A user with no usable history gets an all-zero row, which
split.top_k turns into the popularity list, so the cold-start fallback is identical for every method.

  popularity   (required baseline)  number of users with an n80 interaction; the same list for all
  replay       (reference)          the user's own n80 count, ties by seconds watched
  mf_als       matrix factorization for implicit feedback (alternating least squares, Hu et al. 2008)
  ppr          Personalized PageRank on the programme co-view graph, restarting at the user's history
  assoc_rules  temporal association rules "watched x in the earlier half -> watches y in the later half"
  cbf          cosine between the user's attribute profile and each programme's attribute vector
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from threadpoolctl import threadpool_limits

from split import Split


def _dense(m) -> np.ndarray:
    return np.asarray(m.todense() if sp.issparse(m) else m, dtype=np.float32)


def popularity(split: Split) -> np.ndarray:
    row = (len(split.candidates) - split.pop_rank).astype(np.float32)
    return np.broadcast_to(row, (split.n_users, len(row)))


def replay(split: Split) -> np.ndarray:
    n80 = _dense(split.matrix("n80")[:, split.cand_pos])
    secs = _dense(split.matrix("secs")[:, split.cand_pos])
    return n80 + secs / (secs.max() + 1)  # seconds only break ties between equal counts


def mf_als(split: Split, factors: int = 64, regularization: float = 0.05, alpha: float = 1.0,
           iterations: int = 20, log_conf: bool = False, seed: int = 42) -> np.ndarray:
    from implicit.cpu.als import AlternatingLeastSquares

    x = split.matrix("n80").astype(np.float32)
    if log_conf:
        x.data = np.log1p(x.data)
    model = AlternatingLeastSquares(factors=factors, regularization=regularization, alpha=alpha,
                                    iterations=iterations, random_state=seed, calculate_training_loss=False)
    with threadpool_limits(1, "blas"):  # implicit parallelises itself; a threaded BLAS inside it only slows it down
        model.fit(x, show_progress=False)
    return (model.user_factors @ model.item_factors[split.cand_pos].T).astype(np.float32)


def mf_wals(split: Split, factors: int = 64, regularization: float = 1.0, alpha: float = 10.0,
            iterations: int = 15, seed: int = 42) -> np.ndarray:
    """Weighted ALS on graded counts: minimise sum_ui c_ui (r_ui - x_u.y_i)^2 + reg (|x|^2 + |y|^2) with
    r_ui = log(1 + n80_ui) (0 where the user never watched) and confidence c_ui = 1 + alpha * r_ui.
    The Hu et al. objective with the graded value instead of a 0/1 preference, so the factors keep how
    often a user watches a programme, not only whether they did."""
    r = split.matrix("n80").astype(np.float64)
    r.data = np.log1p(r.data)
    rng = np.random.default_rng(seed)
    v = rng.normal(0, 0.01, (r.shape[1], factors))
    r_t = r.T.tocsr()
    u = np.zeros((r.shape[0], factors))
    with threadpool_limits(1, "blas"):  # thousands of tiny solves: threading overhead dominates
        for _ in range(iterations):
            u = _wals_half(r, v, regularization, alpha)
            v = _wals_half(r_t, u, regularization, alpha)
    return (u @ v[split.cand_pos].T).astype(np.float32)


def _wals_half(r: sp.csr_matrix, y: np.ndarray, reg: float, alpha: float) -> np.ndarray:
    """Solve every row of one side with the other side fixed (unobserved cells: target 0, confidence 1)."""
    k = y.shape[1]
    base = y.T @ y + reg * np.eye(k)
    out = np.zeros((r.shape[0], k))
    for row in np.flatnonzero(np.diff(r.indptr)):
        s, e = r.indptr[row], r.indptr[row + 1]
        yu, val = y[r.indices[s:e]], r.data[s:e]
        c = 1.0 + alpha * val
        out[row] = np.linalg.solve(base + (yu.T * (c - 1.0)) @ yu, yu.T @ (c * val))
    return out


def ppr(split: Split, restart: float = 0.3, cosine: bool = True) -> np.ndarray:
    """Programme co-view graph: edge weight = number of users with an n80 interaction on both
    programmes (cosine-normalised by default), no self-loops; P = row-normalised weights. The walk
    restarts with probability `restart` at r_u, the user's n80 counts L1-normalised, so its stationary
    vector is pi_u = restart * r_u (I - (1 - restart) P)^-1. That inverse is computed once for all
    users (exact, no power iterations). Programmes without edges keep only their restart mass."""
    xb = split.matrix("n80", binary=True)
    deg = np.asarray(xb.sum(axis=0)).ravel()
    active = np.flatnonzero(deg > 0)  # programmes someone interacted with; the rest score 0
    xa = xb[:, active]
    if cosine:
        xa = (xa @ sp.diags(1.0 / np.sqrt(deg[active]))).tocsr()
    g = (xa.T @ xa).toarray().astype(np.float64)
    np.fill_diagonal(g, 0.0)
    row = g.sum(axis=1)
    p = g / np.where(row > 0, row, 1.0)[:, None]
    m = np.linalg.inv(np.eye(len(active)) - (1 - restart) * p)
    r = split.matrix("n80")[:, active]
    r = sp.diags(1.0 / np.maximum(np.asarray(r.sum(axis=1)).ravel(), 1e-12)) @ r
    in_active = np.searchsorted(active, split.cand_pos)
    hit = (in_active < len(active)) & (active[np.minimum(in_active, len(active) - 1)] == split.cand_pos)
    out = np.zeros((split.n_users, len(split.candidates)), dtype=np.float32)
    out[:, hit] = restart * (r @ m[:, in_active[hit]])
    return out


def assoc_rules(split: Split, min_support: int = 5, basket: str = "recent") -> np.ndarray:
    """Rules x -> y mined across time: support = users with x in the first half of the history and y
    in the second half; confidence = support / users with x in the first half. x -> x is a rule too
    (how often a programme's viewers come back). A user's score for y is the highest confidence over
    the rules whose antecedent is in their basket (the later half, or all history); a small term in
    the sum of confidences breaks ties."""
    lo, hi = split.hist_weeks
    mid = (lo + hi) // 2
    b1 = split.matrix("n80", weeks=(lo, mid), binary=True)
    b2 = split.matrix("n80", weeks=(mid + 1, hi), binary=True)
    support = (b1.T @ b2).tocsr()
    support.data[support.data < min_support] = 0
    support.eliminate_zeros()
    n_x = np.asarray(b1.sum(axis=0)).ravel()
    conf = (sp.diags(np.divide(1.0, n_x, out=np.zeros_like(n_x), where=n_x > 0)) @ support).tocsr()
    conf = conf[:, split.cand_pos].tocsr()
    bask = b2 if basket == "recent" else split.matrix("n80", binary=True)
    total = _dense(bask @ conf)
    best = np.zeros_like(total)
    for u in np.flatnonzero(np.diff(bask.indptr)):
        rows = bask.indices[bask.indptr[u]:bask.indptr[u + 1]]
        best[u] = conf[rows].max(axis=0).toarray().ravel()
    return best + 1e-6 * total / (total.max() + 1)


def cbf(split: Split, item_vectors: sp.csr_matrix, profile_weight: str = "n80") -> np.ndarray:
    """item_vectors: content.item_matrix(tokens, split.items), rows aligned with split.items."""
    x = split.matrix("n80")
    if profile_weight == "log":
        x.data = np.log1p(x.data)
    elif profile_weight == "binary":
        x.data = np.ones_like(x.data)
    prof = (x @ item_vectors).tocsr()
    norms = np.sqrt(np.asarray(prof.multiply(prof).sum(axis=1)).ravel())
    prof = sp.diags(np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)) @ prof
    return _dense(prof @ item_vectors[split.cand_pos].T)
