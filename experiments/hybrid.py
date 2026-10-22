"""Hybrid: rank fusion of the implemented methods (src/fusion.py), on the validation split (11-20 -> 21-30).

Weights and the rank offset c are chosen by coordinate ascent on the MAP@5 of one half of the users and
scored on the other half (2-fold, users shuffled with seed 42), so the hybrid's number is a held-out
number. Each single method is scored on the same halves. The weights used for the final submission are
then chosen the same way on all users.

Writes experiments/hybrid.json.  Run:  python experiments/hybrid.py   (after experiments/validate.py)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import content  # noqa: E402
import methods  # noqa: E402
import submission  # noqa: E402
from fusion import fuse, ranked_top  # noqa: E402
from metrics import apk  # noqa: E402
from split import make_split, top_k  # noqa: E402

WORK = ROOT / "work"
HERE = Path(__file__).resolve().parent
OUT = HERE / "hybrid.json"
COMPONENTS = ("ppr", "mf_wals", "assoc_rules", "cbf", "mf_als")
SINGLE = ("popularity", "replay") + COMPONENTS
WEIGHT_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
C_GRID = (1.0, 2.0, 5.0, 10.0, 20.0, 60.0)
SEED = 42


def method_scores(name: str, split, params: dict, tokens) -> np.ndarray:
    if name == "cbf":
        return methods.cbf(split, content.item_matrix(tokens, split.items, params["weights"]),
                           profile_weight=params["profile_weight"])
    return getattr(methods, name)(split, **params)


def search(top_idx, split, users, n_cand) -> tuple[dict, float, float]:
    """Coordinate ascent over the component weights and c, starting from PPR alone."""
    def score(w, c):
        recs = top_k(fuse(top_idx, w, c, split.n_users, n_cand), split, users)
        return float(np.mean([apk(recs[u], split.truth[u]) for u in users]))

    weights, c = {m: (1.0 if m == "ppr" else 0.0) for m in COMPONENTS}, 5.0
    best = score(weights, c)
    for _ in range(2):
        improved = False
        for m in COMPONENTS:
            for v in WEIGHT_GRID:
                if v == weights[m]:
                    continue
                trial = {**weights, m: v}
                s = score(trial, c)
                if s > best + 1e-5:
                    best, weights, improved = s, trial, True
        for v in C_GRID:
            if v != c:
                s = score(weights, v)
                if s > best + 1e-5:
                    best, c, improved = s, v, True
        if not improved:
            break
    return weights, c, best


def list_profile(recs: dict, users, truth: dict, seen: dict, pop_rank: dict, is_test: np.ndarray) -> dict:
    """What the lists consist of (test users): familiar vs new programmes, and where the hits come from."""
    tu = [u for u, t in zip(users, is_test) if t]
    items = [(u, p) for u in tu for p in recs[u]]
    familiar = [p in seen.get(u, ()) for u, p in items]
    hit = [p in truth[u] for u, p in items]
    new_hits = sum(h and not f for h, f in zip(hit, familiar))
    return {
        "familiar_share": round(float(np.mean(familiar)), 4),
        "precision_familiar": round(sum(h and f for h, f in zip(hit, familiar)) / max(sum(familiar), 1), 4),
        "precision_new": round(new_hits / max(len(items) - sum(familiar), 1), 4),
        "new_hits_per_user": round(new_hits / len(tu), 4),
        "distinct_programmes": len({p for _, p in items}),
        "median_popularity_rank": float(np.median([pop_rank[p] for _, p in items])),
    }


def main() -> None:
    t0 = time.perf_counter()
    upw = pd.read_parquet(WORK / "upw.parquet")
    schedule = pd.read_parquet(WORK / "schedule.parquet")
    n_users = len(pd.read_parquet(WORK / "users.parquet"))
    split = make_split("A->B", upw, schedule, n_users, (11, 20), "w21_30", (21, 30))
    tokens = content.programme_tokens(schedule)
    selected = {m: r["selected"]["params"] for m, r in json.loads((HERE / "validation.json").read_text()).items()}
    test = set(int(c) for c in submission.test_user_codes())
    users = np.array(sorted(split.truth))
    n_cand = len(split.candidates)

    scores = {m: method_scores(m, split, selected.get(m, {}), tokens) for m in SINGLE}
    single_ap, lists = {}, {}
    for m in SINGLE:
        lists[m] = top_k(scores[m], split, users)
        single_ap[m] = np.array([apk(lists[m][u], split.truth[u]) for u in users])
    top_idx = {m: ranked_top(scores[m], split) for m in COMPONENTS}
    del scores
    seen = split.hist[split.hist["secs"] >= 60].groupby("user")["tv_show_id"].agg(set).to_dict()
    pop_rank = dict(zip(split.candidates.tolist(), split.pop_rank.tolist()))
    print(f"scores ready [{time.perf_counter() - t0:.0f}s]", flush=True)

    perm = np.random.default_rng(SEED).permutation(len(users))
    halves = [np.sort(perm[: len(users) // 2]), np.sort(perm[len(users) // 2:])]
    folds, held_ap = [], np.zeros(len(users))
    for f in (0, 1):
        tune, held = users[halves[f]], users[halves[1 - f]]
        w, c, tune_map = search(top_idx, split, list(tune), n_cand)
        recs = top_k(fuse(top_idx, w, c, split.n_users, n_cand), split, held)
        held_ap[halves[1 - f]] = [apk(recs[u], split.truth[u]) for u in held]
        folds.append({"weights": w, "c": c, "tune_map5": round(tune_map, 4),
                      "held_out_map5": round(float(held_ap[halves[1 - f]].mean()), 4)})
        print(f"fold {f}: {folds[-1]}  [{time.perf_counter() - t0:.0f}s]", flush=True)

    is_test = np.array([u in test for u in users])
    table = {m: {"map5_all": round(float(ap.mean()), 4), "map5_test": round(float(ap[is_test].mean()), 4)}
             for m, ap in single_ap.items()}
    table["hybrid (2-fold held-out)"] = {"map5_all": round(float(held_ap.mean()), 4),
                                         "map5_test": round(float(held_ap[is_test].mean()), 4)}
    # paired comparison with the best single method on the same users
    best_single = max(COMPONENTS, key=lambda m: single_ap[m].mean())
    diff = held_ap - single_ap[best_single]
    boot = np.random.default_rng(SEED).choice(len(users), (2000, len(users)))
    ci = np.percentile(diff[boot].mean(axis=1), [2.5, 97.5])

    # by activity: quartiles of the user's n80 airings in the history window
    activity = split.hist.groupby("user")["n80"].sum().reindex(users).fillna(0).to_numpy()
    edges = np.quantile(activity, [0.25, 0.5, 0.75])
    bucket = np.digitize(activity, edges)
    by_activity = {"quartile_edges_n80": edges.tolist(), "users": np.bincount(bucket).tolist(),
                   "map5": {m: [round(float(ap[bucket == k].mean()), 4) for k in range(4)]
                            for m, ap in {**single_ap, "hybrid (2-fold held-out)": held_ap}.items()}}

    # what the rules know: x -> x confidence (users of x in weeks 11-15 who come back in 16-20), by category
    lab = schedule[schedule["tv_show_id"] != 0]
    category = lab.groupby("tv_show_id")["tv_show_category"].first().reindex(split.items)
    b1 = split.matrix("n80", weeks=(11, 15), binary=True)
    b2 = split.matrix("n80", weeks=(16, 20), binary=True)
    n1 = np.asarray(b1.sum(axis=0)).ravel()
    back = np.asarray(b1.multiply(b2).sum(axis=0)).ravel()
    persist = pd.DataFrame({"category": category.to_numpy(), "n1": n1, "conf": back / np.maximum(n1, 1)})
    persist = persist[persist["n1"] >= 20].groupby("category")["conf"].agg(["count", "median"]).round(3)
    ids = {w: set(g["tv_show_id"]) for w, g in lab.groupby("window")}
    new_c = ids["w31_42"] - ids["w11_20"] - ids["w21_30"]
    new_by_cat = lab[lab["tv_show_id"].isin(new_c)].drop_duplicates("tv_show_id")["tv_show_category"].value_counts()

    w_all, c_all, map_all = search(top_idx, split, list(users), n_cand)
    lists["hybrid"] = top_k(fuse(top_idx, w_all, c_all, split.n_users, n_cand), split, users)
    profile = {m: list_profile(lists[m], users, split.truth, seen, pop_rank, is_test) for m in lists}
    result = {"selected_params": selected, "components": list(COMPONENTS), "folds": folds, "table": table,
              "list_profile": profile, "by_activity": by_activity,
              "self_rule_confidence_by_category": {c: {"programmes": int(r["count"]), "median": float(r["median"])}
                                                   for c, r in persist.iterrows()},
              "new_test_window_ids_by_category": {str(k): int(v) for k, v in new_by_cat.items()},
              "hybrid_minus_best_single": {"best_single": best_single, "mean": round(float(diff.mean()), 4),
                                           "bootstrap_95ci": [round(float(x), 4) for x in ci]},
              "final_weights": {"weights": w_all, "c": c_all, "in_sample_map5": round(map_all, 4)},
              "runtime_s": round(time.perf_counter() - t0, 1)}
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
