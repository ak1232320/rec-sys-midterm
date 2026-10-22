"""Checks for the recommenders on small hand-made data (no competition files needed).

Writes tests/results_methods.json.  Run:  python tests/check_methods.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import methods  # noqa: E402
from fusion import fuse, ranked_top  # noqa: E402
from split import Split, top_k  # noqa: E402

RESULTS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append({"check": name, "passed": bool(ok), "detail": detail})
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def toy_split(seed: int = 0, n_users: int = 30, n_items: int = 12, weeks=(1, 4)) -> Split:
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users - 2):  # the last two users have no history
        for p in rng.choice(n_items, rng.integers(1, 6), replace=False):
            for w in range(weeks[0], weeks[1] + 1):
                n80 = int(rng.integers(0, 4))
                if n80:
                    rows.append((u, 100 + p, w, 600.0 * n80 + rng.integers(0, 60), n80, n80))
    upw = pd.DataFrame(rows, columns=["user", "tv_show_id", "week", "secs", "n60", "n80"])
    hist = upw.groupby(["user", "tv_show_id"])[["secs", "n60", "n80"]].sum().reset_index()
    items = np.arange(100, 100 + n_items)
    candidates = items[2:]  # two programmes do not air in the target window
    s = Split("toy", weeks, "toy", n_users, hist, upw, candidates, items)
    s.pop_rank = np.arange(len(candidates))[::-1].copy()  # popularity order: last candidate first
    return s


def main() -> None:
    s = toy_split()
    users = list(range(s.n_users))

    # top_k
    recs = top_k(methods.replay(s), s, users)
    ok = all(len(r) == 5 and len(set(r)) == 5 and set(r) <= set(s.candidates.tolist()) for r in recs.values())
    check("top_k: exactly 5 distinct candidates for every user", ok)
    pop_top = s.candidates[np.argsort(s.pop_rank)][:5].tolist()
    check("top_k: a user with no history gets the popularity list", recs[s.n_users - 1] == pop_top, str(recs[s.n_users - 1]))
    u0 = s.hist[s.hist["user"] == 0]
    u0 = u0[u0["tv_show_id"].isin(s.candidates)].sort_values(["n80", "secs"], ascending=False)
    check("replay: own programmes first, by n80 then seconds", recs[0][: len(u0)] == u0["tv_show_id"].tolist()[:5])

    # PPR exact inverse vs power iteration on the same graph
    restart = 0.3
    exact = methods.ppr(s, restart=restart, cosine=True)
    xb = s.matrix("n80", binary=True).toarray()
    deg = xb.sum(axis=0)
    act = deg > 0
    y = xb[:, act] / np.sqrt(deg[act])
    g = y.T @ y
    np.fill_diagonal(g, 0)
    p = g / np.where(g.sum(axis=1) > 0, g.sum(axis=1), 1)[:, None]
    r = s.matrix("n80").toarray()[:, act]
    r = r / np.maximum(r.sum(axis=1, keepdims=True), 1e-12)
    pi = r.copy()
    for _ in range(500):
        pi = restart * r + (1 - restart) * pi @ p
    full = np.zeros((s.n_users, len(s.items)))
    full[:, act] = pi
    err = np.abs(full[:, s.cand_pos] - exact).max()
    check("ppr: closed form equals 500 power iterations", err < 1e-5, f"max abs diff {err:.2e}")

    # weighted ALS half-step solves its normal equations
    rr = s.matrix("n80").astype(np.float64)
    rr.data = np.log1p(rr.data)
    v = np.random.default_rng(1).normal(size=(rr.shape[1], 4))
    u = methods._wals_half(rr, v, reg=0.5, alpha=3.0)
    dense = rr.toarray()
    c = 1 + 3.0 * dense
    grad = ((c * (u @ v.T - dense)) @ v) + 0.5 * u
    rows_with_data = np.diff(rr.indptr) > 0
    check("mf_wals: user step satisfies its normal equations",
          np.abs(grad[rows_with_data]).max() < 1e-8, f"max |grad| {np.abs(grad[rows_with_data]).max():.1e}")
    check("mf_wals: users without history get a zero vector", np.allclose(u[~rows_with_data], 0))

    # association rules by hand: x in weeks 1-2 -> y in weeks 3-4
    b1 = s.matrix("n80", weeks=(1, 2), binary=True).toarray()
    b2 = s.matrix("n80", weeks=(3, 4), binary=True).toarray()
    support = b1.T @ b2
    conf = np.where(b1.sum(axis=0)[:, None] > 0, support / np.maximum(b1.sum(axis=0)[:, None], 1), 0)
    conf[support < 2] = 0
    ar = methods.assoc_rules(s, min_support=2, basket="recent")
    want = np.array([conf[b2[u] > 0][:, s.cand_pos].max(axis=0) if b2[u].any() else np.zeros(len(s.candidates))
                     for u in range(s.n_users)])
    check("assoc_rules: max confidence over the basket matches a hand count", np.abs(ar - want).max() < 1e-5)

    # fusion of one method with weight 1 keeps that method's ranking
    sc = methods.ppr(s, restart=0.5)
    fused = fuse({"ppr": ranked_top(sc, s, depth=8)}, {"ppr": 1.0}, 5.0, s.n_users, len(s.candidates), depth=8)
    check("fusion: one method alone reproduces its own top-5", top_k(fused, s, users) == top_k(sc, s, users))

    # CBF: a programme's cosine with a profile made of itself is 1
    vec = sp.csr_matrix(np.eye(len(s.items), dtype=np.float32))
    one = Split("one", (1, 1), "x", 1, pd.DataFrame({"user": [0], "tv_show_id": [s.items[3]], "secs": [600.0],
                                                     "n60": [1], "n80": [1]}), s.upw.iloc[:0], s.candidates, s.items)
    one.pop_rank = s.pop_rank
    cb = methods.cbf(one, vec)
    check("cbf: profile of a single programme has cosine 1 with it, 0 with orthogonal ones",
          np.isclose(cb[0, np.searchsorted(s.candidates, s.items[3])], 1) and np.isclose(cb.sum(), 1))

    passed = sum(r["passed"] for r in RESULTS)
    print(f"\n{passed} passed / {len(RESULTS) - passed} failed")
    (Path(__file__).resolve().parent / "results_methods.json").write_text(
        json.dumps({"passed": passed, "failed": len(RESULTS) - passed, "checks": RESULTS}, indent=2))
    sys.exit(0 if passed == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
