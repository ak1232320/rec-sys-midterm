"""Validation: fit on weeks 11-20, evaluate on weeks 21-30.

Truth: each user's top-5 programmes by n80 in weeks 21-30 (EDA §8b). Candidates: the labelled programmes
of the 21-30 schedule; already-watched programmes are allowed. Metric: MAP@5 over every user with a
non-empty truth (4,623), also reported for the test users among them (1,236). Hyperparameters are
selected on the all-user MAP@5; the test-user column is reported, never used for selection.

Writes experiments/validation.json (every configuration) and experiments/validation_ap.npz (per-user AP
of the selected configuration of each method, for paired comparisons).

Run:  python experiments/validate.py [method ...]   (default: all)
"""
from __future__ import annotations

import itertools
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
from metrics import apk  # noqa: E402
from split import make_split, top_k  # noqa: E402

WORK = ROOT / "work"
OUT = Path(__file__).resolve().parent / "validation.json"
OUT_AP = Path(__file__).resolve().parent / "validation_ap.npz"

GRIDS = {
    "popularity": [{}],
    "replay": [{}],
    "mf_als": [dict(factors=f, regularization=r, alpha=a, log_conf=lc)
               for f, r, a, lc in itertools.product((32, 64, 128, 256), (0.5, 2.0, 5.0, 20.0, 50.0),
                                                    (0.1, 0.3, 1.0, 10.0), (False, True))],
    "mf_wals": [dict(factors=f, regularization=r, alpha=a)
                for f, r, a in itertools.product((32, 64), (10.0, 30.0, 100.0, 300.0), (3.0, 10.0, 30.0))]
               + [dict(factors=128, regularization=r, alpha=10.0) for r in (30.0, 100.0)],  # 128 checked near the optimum only
    "ppr": [dict(restart=r, cosine=c) for r, c in itertools.product((0.1, 0.3, 0.5, 0.7, 0.9), (True, False))],
    "assoc_rules": [dict(min_support=s, basket=b)
                    for s, b in itertools.product((2, 5, 10, 20, 50, 100, 200), ("recent", "all"))],
    "cbf": [dict(weights=w, profile_weight=p) for w, p in itertools.product(
        ({}, {"act": 0.5, "dir": 0.25}, {"cat": 2.0, "genre": 2.0}, {"act": 2.0, "dir": 1.0},
         {"act": 4.0, "dir": 2.0}, {"act": 4.0, "dir": 2.0, "decade": 0.0}, {"act": 8.0, "dir": 4.0, "decade": 0.0},
         {"act": 16.0, "dir": 8.0, "decade": 0.0}), ("n80", "log"))],
}


def main() -> None:
    wanted = sys.argv[1:] or list(GRIDS)
    t0 = time.perf_counter()
    upw = pd.read_parquet(WORK / "upw.parquet")
    schedule = pd.read_parquet(WORK / "schedule.parquet")
    n_users = len(pd.read_parquet(WORK / "users.parquet"))
    split = make_split("A->B", upw, schedule, n_users, (11, 20), "w21_30", (21, 30))
    tokens = content.programme_tokens(schedule)
    test = set(int(c) for c in submission.test_user_codes())
    users = sorted(split.truth)
    test_users = [u for u in users if u in test]
    print(f"split ready: {len(users)} users with truth, {len(test_users)} test users, "
          f"{len(split.candidates)} candidates, {len(split.items)} items  [{time.perf_counter() - t0:.0f}s]", flush=True)

    for name in wanted:
        best_ap = None
        rows = []
        for hp in GRIDS[name]:
            t1 = time.perf_counter()
            if name == "cbf":
                vectors = content.item_matrix(tokens, split.items, hp["weights"])
                scores = methods.cbf(split, vectors, profile_weight=hp["profile_weight"])
            else:
                scores = getattr(methods, name)(split, **hp)
            fit_s = time.perf_counter() - t1
            recs = top_k(scores, split, users)
            ap = np.array([apk(recs[u], split.truth[u]) for u in users])
            is_test = np.array([u in test for u in users])
            rows.append({"params": hp, "map5_all": round(float(ap.mean()), 4),
                         "map5_test": round(float(ap[is_test].mean()), 4), "seconds": round(fit_s, 1)})
            print(f"  {name:12s} {json.dumps(hp, ensure_ascii=False):70s} all {ap.mean():.4f}  "
                  f"test {ap[is_test].mean():.4f}  {fit_s:6.1f}s", flush=True)
            if not rows[:-1] or rows[-1]["map5_all"] > max(r["map5_all"] for r in rows[:-1]):
                best_ap = ap
        save(name, {"selected": max(rows, key=lambda r: r["map5_all"]), "grid": rows}, best_ap, users)
    print(f"done in {time.perf_counter() - t0:.0f}s")


def save(name: str, entry: dict, ap: np.ndarray, users: list) -> None:
    """Re-read and update only this method's entry, so several runs can share the files."""
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    results[name] = entry
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    aps = dict(np.load(OUT_AP)) if OUT_AP.exists() else {}
    aps.update({name: ap, "users": np.array(users)})
    np.savez(OUT_AP, **aps)


if __name__ == "__main__":
    main()
