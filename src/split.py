"""A history window, a target window, and everything a method may see.

A Split holds:
  * the history interactions, per user x programme (secs, n60, n80) and per week;
  * the candidate pool: labelled programmes of the target window's schedule (repeats allowed);
  * the item index: history programmes + candidates, sorted;
  * for validation only, the truth: each user's top-5 programmes by n80 in the target weeks.

Methods return a dense score matrix, users x candidates. `top_k` turns any score matrix into exactly
k distinct ids per user. Ties, including a row of zeros (no usable history), are broken by the
popularity order. That makes the cold-start fallback identical for every method.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from interactions import SIGNALS, eligible, top_lists, window

K = 5


@dataclass
class Split:
    name: str
    hist_weeks: tuple[int, int]
    target_window: str
    n_users: int
    hist: pd.DataFrame  # user, tv_show_id, secs, n60, n80 over the history weeks
    upw: pd.DataFrame  # the same per week (history weeks only)
    candidates: np.ndarray  # sorted tv_show_ids airing in the target window
    items: np.ndarray  # sorted: history programmes + candidates
    truth: dict | None = None  # user -> set of ids (validation only)
    pop_rank: np.ndarray = field(default=None)  # rank of each candidate in the popularity order (0 = most popular)

    @property
    def cand_pos(self) -> np.ndarray:
        """Position of each candidate in `items`."""
        return np.searchsorted(self.items, self.candidates)

    def matrix(self, signal: str = "n80", weeks: tuple[int, int] | None = None, binary: bool = False) -> sp.csr_matrix:
        """users x items matrix of one signal (optionally over a sub-range of the history weeks)."""
        df = self.hist if weeks is None else window(self.upw, *weeks)
        df = eligible(df, signal)
        values = np.ones(len(df), dtype=np.float32) if binary else df[signal].to_numpy(dtype=np.float32)
        cols = np.searchsorted(self.items, df["tv_show_id"].to_numpy())
        return sp.csr_matrix((values, (df["user"].to_numpy(), cols)), shape=(self.n_users, len(self.items)))


def make_split(name: str, upw: pd.DataFrame, schedule: pd.DataFrame, n_users: int, hist_weeks: tuple[int, int],
               target_window: str, target_weeks: tuple[int, int] | None = None) -> Split:
    upw_h = upw[(upw["week"] >= hist_weeks[0]) & (upw["week"] <= hist_weeks[1])]
    hist = window(upw, *hist_weeks)
    sched = schedule[(schedule["window"] == target_window) & (schedule["tv_show_id"] != 0)]
    candidates = np.sort(sched["tv_show_id"].unique())
    items = np.union1d(hist["tv_show_id"].unique(), candidates)
    truth = None
    if target_weeks is not None:
        truth = {u: set(v) for u, v in top_lists(window(upw, *target_weeks), "n80").items()}
    split = Split(name, hist_weeks, target_window, n_users, hist, upw_h, candidates, items, truth)
    # popularity order: users with an n80 interaction in the history, then seconds; unseen candidates last
    pop = eligible(hist, "n80").groupby("tv_show_id").agg(users=("user", "nunique"), secs=("secs", "sum"))
    pop = pop.reindex(candidates).fillna(0)
    order = np.lexsort((candidates, -pop["secs"].to_numpy(), -pop["users"].to_numpy()))
    split.pop_rank = np.empty(len(candidates), dtype=np.int64)
    split.pop_rank[order] = np.arange(len(candidates))
    return split


def best_positions(row: np.ndarray, pop_rank: np.ndarray, k: int) -> np.ndarray:
    """Positions of the k best candidates in one score row: highest score first, ties by popularity.
    Every candidate tied with the k-th score is considered, so the cut never depends on argpartition's
    arbitrary choice among equal scores."""
    part = np.argpartition(-row, k - 1)[:k]
    kth = row[part].min()
    idx = np.union1d(part[row[part] > kth], np.flatnonzero(row == kth))
    return idx[np.lexsort((pop_rank[idx], -row[idx]))][:k]


def top_k(scores: np.ndarray, split: Split, users, k: int = K) -> dict:
    """Exactly k distinct candidate ids per user."""
    return {int(u): split.candidates[best_positions(scores[u], split.pop_rank, k)].tolist() for u in users}
