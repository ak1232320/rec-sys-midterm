"""Design and build the Kaggle probe for the target definition.

The competition does not say how "most preferred" was computed. Three candidate signals are compared
(src/interactions.py): secs, n60 and n80. One instrument, history replay (src/replay.py), is run with
each signal as its ranking signal, and nothing else changes.

1. Proxy (no Kaggle): history = weeks 11-20, truth = each user's top-5 in weeks 21-30 under each
   definition. This gives a 3 x 3 table of MAP@5, ranking signal x truth definition. The probe pair is
   the two ranking signals whose ORDER flips between truth definitions, so the returned public scores
   can tell the hypotheses apart.
2. Submissions: history = weeks 11-30, candidates = labelled programmes of the 31-42 schedule, written
   to submissions/probe_<signal>.csv and validated.

Writes experiments/target_probe.json.  Run:  python experiments/target_probe.py
"""
from __future__ import annotations

import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from interactions import SIGNALS, top_lists, window  # noqa: E402
from metrics import mapk  # noqa: E402
from replay import recommend  # noqa: E402
import submission  # noqa: E402

WORK = ROOT / "work"
OUT = Path(__file__).resolve().parent / "target_probe.json"
SUBMISSIONS = ROOT / "submissions"


def main() -> None:
    t0 = time.perf_counter()
    upw = pd.read_parquet(WORK / "upw.parquet")
    schedule = pd.read_parquet(WORK / "schedule.parquet", columns=["window", "tv_show_id"])
    ids = {w: set(g.loc[g["tv_show_id"] != 0, "tv_show_id"].unique()) for w, g in schedule.groupby("window")}
    test_codes = submission.test_user_codes()
    is_test = set(int(c) for c in test_codes)

    # 1. proxy: weeks 11-20 -> 21-30
    hist, future = window(upw, 11, 20), window(upw, 21, 30)
    truths = {t: {u: set(v) for u, v in top_lists(future, t).items()} for t in SIGNALS}
    users = sorted(set().union(*truths.values()))
    recs = {h: recommend(hist, h, users, ids["w21_30"]) for h in SIGNALS}
    table = {}
    for h in SIGNALS:
        for t in SIGNALS:
            all_users = sorted(truths[t])
            test_users = [u for u in all_users if u in is_test]
            table[f"{h}->{t}"] = {"all": round(mapk(recs[h], truths[t], all_users), 4),
                                  "test": round(mapk(recs[h], truths[t], test_users), 4)}
    flips = {}
    for h1, h2 in combinations(SIGNALS, 2):
        diff = {t: round(table[f"{h1}->{t}"]["test"] - table[f"{h2}->{t}"]["test"], 4) for t in SIGNALS}
        flips[f"{h1}-{h2}"] = {"test_map_difference_by_truth": diff,
                               "order_flips": bool(min(diff.values()) < 0 < max(diff.values()))}

    # 2. submissions: weeks 11-30 -> 31-42
    hist_all = window(upw, 11, 30)
    users_c = [int(c) for c in test_codes]
    subs = {h: recommend(hist_all, h, users_c, ids["w31_42"]) for h in SIGNALS}
    files = {}
    for h in SIGNALS:
        path = SUBMISSIONS / f"probe_{h}.csv"
        submission.write(subs[h], path)
        files[h] = {"path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "validation": submission.validate(path, ids["w31_42"])}
    personal = {h: top_lists(hist_all, h, allowed=ids["w31_42"]) for h in SIGNALS}
    for h in SIGNALS:
        files[h]["users_padded_with_popularity"] = int(sum(len(personal[h].get(u, [])) < 5 for u in users_c))
        files[h]["users_with_no_history"] = int(sum(u not in personal[h] for u in users_c))
    overlap = {}
    for h1, h2 in combinations(SIGNALS, 2):
        same_set = [len(set(subs[h1][u]) & set(subs[h2][u])) for u in users_c]
        same_slot = [sum(a == b for a, b in zip(subs[h1][u], subs[h2][u])) for u in users_c]
        overlap[f"{h1}-{h2}"] = {"mean_shared_ids_of_5": round(float(np.mean(same_set)), 3),
                                 "mean_same_position_of_5": round(float(np.mean(same_slot)), 3),
                                 "identical_lists": int(sum(s == 5 for s in same_slot))}

    result = {"proxy_map5_signal_to_truth": table, "pairs": flips, "files": files,
              "submission_overlap": overlap, "runtime_s": round(time.perf_counter() - t0, 1)}
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
