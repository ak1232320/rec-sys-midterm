"""Checks for the ingest and the view-schedule join.

  1. interval_join agrees with a brute-force join on random synthetic data;
  2. the parquet caches reproduce the counts printed in the exam statement;
  3. invariants of the real join output (work/pairs.parquet);
  4. every view whose duraton differs from stop - start spans the DST switch.

Writes tests/results.json.  Run:  python tests/check_data.py   (after src/ingest.py and src/join.py)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from join import clean_schedule, interval_join  # noqa: E402

WORK = ROOT / "work"
RESULTS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append({"check": name, "passed": bool(ok), "detail": detail})
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def brute_force(v_ch, v_start, v_stop, s_ch, s_start, s_end):
    out = set()
    for i in range(len(v_ch)):
        for j in range(len(s_ch)):
            if v_ch[i] == s_ch[j]:
                ov = min(v_stop[i], s_end[j]) - max(v_start[i], s_start[j])
                if ov > 0:
                    out.add((i, j, int(ov)))
    return out


def synthetic(seed: int):
    rng = np.random.default_rng(seed)
    rows = []
    for ch in (3, 7, 8, 40):
        t = int(rng.integers(0, 500))
        for _ in range(int(rng.integers(5, 15))):
            length = int(rng.integers(1, 400))
            rows.append((ch, t, t + length, length))
            t += length + int(rng.choice([0, 0, 0, rng.integers(1, 100)]))  # mostly contiguous, some gaps
    sch = pd.DataFrame(rows, columns=["channel_id", "start", "end", "duration"])
    sch = sch.sample(frac=1, random_state=seed)  # clean_schedule must sort
    sch, _ = clean_schedule(sch)
    n = 300
    v_ch = rng.choice([3, 7, 8, 40, 99], n)  # channel 99 has no schedule
    v_start = rng.integers(-200, 6000, n)
    v_stop = v_start + rng.integers(61, 2000, n)
    return v_ch, v_start, v_stop, sch


def main() -> None:
    # 1. synthetic agreement
    for seed in range(5):
        v_ch, v_start, v_stop, sch = synthetic(seed)
        s = (sch["channel_id"].to_numpy(), sch["start"].to_numpy(), sch["end"].to_numpy())
        vi, ai, ov = interval_join(v_ch, v_start, v_stop, *s)
        fast = set(zip(vi.tolist(), ai.tolist(), ov.tolist()))
        slow = brute_force(v_ch, v_start, v_stop, *s)
        check(f"interval_join == brute force (seed {seed})", fast == slow, f"{len(fast)} pairs")

    # 1b. clean_schedule cuts an airing that runs into the next one
    over = pd.DataFrame({"channel_id": [1, 1, 1], "start": [0, 50, 60], "end": [100, 60, 200], "duration": [100, 10, 140]})
    cleaned, st = clean_schedule(over)
    check("clean_schedule cuts overlapping airings", cleaned["end"].tolist() == [50, 60, 200] and st["airings_cut_at_next_start"] == 1)

    # 2. counts from the exam statement
    views = pd.read_parquet(WORK / "views.parquet")
    schedule = pd.read_parquet(WORK / "schedule.parquet", columns=["window", "tv_show_id", "channel_id"])
    check("view log rows = 5,821,987", len(views) == 5_821_987, f"{len(views):,}")
    rows = schedule["window"].value_counts().to_dict()
    check("schedule rows = 1,074,235 / 1,074,394 / 1,295,378",
          rows == {"w11_20": 1_074_235, "w21_30": 1_074_394, "w31_42": 1_295_378}, str(rows))
    lab_11_20 = int((schedule.loc[schedule["window"] == "w11_20", "tv_show_id"] != 0).sum())
    check("labelled rows in weeks 11-20 = 115,613", lab_11_20 == 115_613, f"{lab_11_20:,}")
    distinct = schedule.groupby("window")["tv_show_id"].nunique().to_dict()  # the statement counts id 0 as a value
    check("distinct tv_show_id incl. 0 = 5,848 / 6,364 / 6,637",
          distinct == {"w11_20": 5_848, "w21_30": 6_364, "w31_42": 6_637}, str(distinct))
    channels = schedule.groupby("window")["channel_id"].nunique().to_dict()
    check("channels = 400 / 403 / 404", channels == {"w11_20": 400, "w21_30": 403, "w31_42": 404}, str(channels))
    test_users = pd.read_parquet(WORK / "test_users.parquet")
    check("submission template has 1,260 distinct users", test_users["user_id"].nunique() == 1_260)
    check("median view duration = 396 s, max = 342,145 s",
          views["duration"].median() == 396 and views["duration"].max() == 342_145)

    # 3. invariants of the real join
    pairs = pd.read_parquet(WORK / "pairs.parquet")
    airings = pd.read_parquet(WORK / "airings_train.parquet", columns=["channel_id", "start", "end"])
    v, a = pairs["view"].to_numpy(), pairs["airing"].to_numpy()
    check("pair channel == view channel == airing channel",
          bool(np.all(views["channel_id"].to_numpy()[v] == pairs["channel_id"].to_numpy())
               and np.all(airings["channel_id"].to_numpy()[a] == pairs["channel_id"].to_numpy())))
    ov = pairs["overlap"].to_numpy()
    span = (views["stop"] - views["start"]).to_numpy()
    check("0 < overlap <= view span and <= airing length",
          bool(np.all(ov > 0) & np.all(ov <= span[v]) & np.all(ov <= (airings["end"] - airings["start"]).to_numpy()[a])))
    per_view = np.bincount(v, weights=ov, minlength=len(views))
    check("per view, credited seconds <= view span (no second counted twice)", bool(np.all(per_view <= span)))

    # 4. the DST signature
    diff = (views["stop"] - views["start"] - views["duration"]).to_numpy()
    dst = int(np.datetime64("2020-03-29T03:00:00", "s").astype(np.int64))
    spans = (views["start"].to_numpy() < dst) & (views["stop"].to_numpy() >= dst)
    check("duraton != stop - start only across the 2020-03-29 DST switch, always by 3600 s",
          bool(np.all(diff[diff != 0] == 3600) and np.array_equal(diff != 0, spans)), f"{int((diff != 0).sum())} views")

    passed = sum(r["passed"] for r in RESULTS)
    print(f"\n{passed} passed / {len(RESULTS) - passed} failed")
    (Path(__file__).resolve().parent / "results.json").write_text(
        json.dumps({"passed": passed, "failed": len(RESULTS) - passed, "checks": RESULTS}, indent=2))
    sys.exit(0 if passed == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
