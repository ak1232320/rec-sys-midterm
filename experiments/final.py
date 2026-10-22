"""Final run: fit every method on weeks 11-30 and rank the labelled programmes of the 31-42 schedule.

Hyperparameters come from experiments/validation.json, the hybrid's weights from experiments/hybrid.json.
Nothing here looks at viewing after week 30 (there is none in the data); the 31-42 schedule is used only
for the candidate pool (which programmes air) and their attributes.

Writes submissions/<method>.csv for every method and the hybrid, and experiments/final.json (runtimes,
validation of every file, md5).  Run:  python experiments/final.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import content  # noqa: E402
import submission  # noqa: E402
from fusion import fuse, ranked_top  # noqa: E402
from split import make_split, top_k  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hybrid import COMPONENTS, SINGLE, method_scores  # noqa: E402

WORK = ROOT / "work"
HERE = Path(__file__).resolve().parent
SUBMISSIONS = ROOT / "submissions"


def main() -> None:
    t0 = time.perf_counter()
    upw = pd.read_parquet(WORK / "upw.parquet")
    schedule = pd.read_parquet(WORK / "schedule.parquet")
    n_users = len(pd.read_parquet(WORK / "users.parquet"))
    split = make_split("11-30 -> 31-42", upw, schedule, n_users, (11, 30), "w31_42")
    tokens = content.programme_tokens(schedule)
    selected = {m: r["selected"]["params"] for m, r in json.loads((HERE / "validation.json").read_text()).items()}
    hyb = json.loads((HERE / "hybrid.json").read_text())["final_weights"]
    users = [int(c) for c in submission.test_user_codes()]
    candidates = set(split.candidates.tolist())

    report, top_idx = {"split": {"history_weeks": [11, 30], "candidates": len(split.candidates),
                                 "items": len(split.items)}, "files": {}}, {}
    for m in SINGLE:
        t1 = time.perf_counter()
        scores = method_scores(m, split, selected.get(m, {}), tokens)
        fit_s = time.perf_counter() - t1
        if m in COMPONENTS:
            top_idx[m] = ranked_top(scores, split)
        write(m, top_k(scores, split, users), candidates, report, fit_s)
        del scores
    t1 = time.perf_counter()
    fused = fuse(top_idx, hyb["weights"], hyb["c"], split.n_users, len(split.candidates))
    write("hybrid", top_k(fused, split, users), candidates, report, time.perf_counter() - t1)
    report["runtime_s"] = round(time.perf_counter() - t0, 1)
    (HERE / "final.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def write(name: str, recs: dict, candidates: set, report: dict, seconds: float) -> None:
    path = SUBMISSIONS / f"{name}.csv"
    submission.write(recs, path)
    report["files"][name] = {"path": f"submissions/{name}.csv", "fit_seconds": round(seconds, 1),
                             "md5": hashlib.md5(path.read_bytes()).hexdigest(),
                             "validation": submission.validate(path, candidates)}
    print(f"{name:12s} {seconds:6.1f}s  valid={report['files'][name]['validation']['valid']}", flush=True)


if __name__ == "__main__":
    main()
