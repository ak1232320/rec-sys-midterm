"""Rank fusion of several score matrices.

Scores from different methods are not on one scale (PPR mass, factor dot products, rule confidences,
cosines), so each method contributes only through its ranks. A candidate at rank r (0 = best) in a
method's top `depth` for a user gets w / (c + r) from that method, and nothing below that depth. Ranks
are taken with the same popularity tie-break as split.top_k.
"""
from __future__ import annotations

import numpy as np

from split import Split, best_positions


def ranked_top(scores: np.ndarray, split: Split, depth: int = 100) -> np.ndarray:
    """Per user, the positions of the `depth` best candidates, best first (users x depth)."""
    depth = min(depth, scores.shape[1])
    return np.stack([best_positions(scores[u], split.pop_rank, depth) for u in range(scores.shape[0])])


def fuse(top_idx: dict, weights: dict, c: float, n_users: int, n_cand: int, depth: int = 100) -> np.ndarray:
    """Sum over methods of w_m / (c + rank_m); a user's top list holds each candidate once, so plain
    fancy-index addition is safe within one method."""
    out = np.zeros((n_users, n_cand), dtype=np.float32)
    rr = (1.0 / (c + np.arange(depth, dtype=np.float32)))[None, :]
    rows = np.arange(n_users)[:, None]
    for name, w in weights.items():
        if w:
            out[rows, top_idx[name][:, :depth]] += w * rr
    return out
