"""Join the view log to the training schedule: same channel, overlapping intervals.

Every (view, airing) pair whose intervals overlap by at least one second becomes a
row of work/pairs.parquet, labelled (tv_show_id != 0) or not. No "watched"
threshold is applied here: thresholds are chosen downstream from the overlap
distribution (experiments/eda.py), so the join itself stays assumption-free.

Schedule cleaning, applied to the two training files (weeks 11-20 and 21-30):
  * rows with duration <= 0 are dropped;
  * an airing listed in both files (same channel and start) is kept once;
  * on a channel, an airing that runs past the next airing's start is cut at
    that start, so airings on one channel never overlap and every second of a
    view is credited to at most one airing.

Outputs:
  * work/airings_train.parquet  cleaned training schedule, row i = airing i
  * work/users.parquet          dense user code -> original uint64 user_id
  * work/pairs.parquet          view, user, airing, channel_id, tv_show_id, overlap (s)
  * work/join_stats.json        counts quoted by the EDA

Run:  python src/join.py   (after src/ingest.py)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work"
TRAIN_WINDOWS = ("w11_20", "w21_30")

# Channel ids are small and naive epoch seconds of 2020 are < 2**31, so channel * 2**32 + t is a
# single sortable int64 key that keeps channels apart.
CHANNEL_SHIFT = np.int64(1) << 32


def clean_schedule(schedule: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Sort by (channel, start), drop duplicates and cut overlapping airings."""
    stats = {"rows_in": int(len(schedule))}
    schedule = schedule[schedule["duration"] > 0]
    stats["dropped_nonpositive_duration"] = stats["rows_in"] - int(len(schedule))
    # Stable sort: for a duplicated (channel, start) the earlier file wins.
    schedule = schedule.sort_values(["channel_id", "start"], kind="stable")
    dup = schedule.duplicated(["channel_id", "start"], keep="first")
    stats["dropped_duplicate_channel_start"] = int(dup.sum())
    schedule = schedule[~dup].reset_index(drop=True)

    ch = schedule["channel_id"].to_numpy()
    start = schedule["start"].to_numpy()
    end = schedule["end"].to_numpy().copy()
    same_next = np.r_[ch[1:] == ch[:-1], False]
    next_start = np.r_[start[1:], np.iinfo(np.int64).max]
    cut = same_next & (end > next_start)
    stats["airings_cut_at_next_start"] = int(cut.sum())
    stats["seconds_cut"] = int((end[cut] - next_start[cut]).sum())
    end[cut] = next_start[cut]
    schedule["end"] = end
    stats["rows_out"] = int(len(schedule))
    return schedule, stats


def interval_join(v_ch, v_start, v_stop, s_ch, s_start, s_end):
    """All (view, airing) pairs on the same channel with a positive overlap.

    The schedule must be sorted by (channel, start) with non-overlapping airings per
    channel (see clean_schedule). Returns (view_idx, airing_idx, overlap_seconds).
    """
    v_ch = np.asarray(v_ch, dtype=np.int64)
    s_ch = np.asarray(s_ch, dtype=np.int64)
    key_start = s_ch * CHANNEL_SHIFT + s_start
    key_end = s_ch * CHANNEL_SHIFT + s_end
    if not (np.all(np.diff(key_start) > 0) and np.all(np.diff(key_end) >= 0)):
        raise ValueError("schedule must be sorted by (channel, start) with non-overlapping airings")
    # First airing ending after the view starts, and the end of the airings starting before it stops.
    lo = np.searchsorted(key_end, v_ch * CHANNEL_SHIFT + v_start, side="right")
    hi = np.searchsorted(key_start, v_ch * CHANNEL_SHIFT + v_stop, side="left")
    n = np.maximum(hi - lo, 0)
    view_idx = np.repeat(np.arange(len(v_ch), dtype=np.int64), n)
    within = np.arange(int(n.sum()), dtype=np.int64) - np.repeat(np.cumsum(n) - n, n)
    airing_idx = np.repeat(lo, n) + within
    overlap = np.minimum(v_stop[view_idx], s_end[airing_idx]) - np.maximum(v_start[view_idx], s_start[airing_idx])
    keep = overlap > 0
    return view_idx[keep], airing_idx[keep], overlap[keep]


def main() -> None:
    t0 = time.perf_counter()
    views = pd.read_parquet(WORK / "views.parquet")
    schedule = pd.read_parquet(WORK / "schedule.parquet")
    schedule = schedule[schedule["window"].isin(TRAIN_WINDOWS)]
    airings, stats = clean_schedule(schedule)
    airings.to_parquet(WORK / "airings_train.parquet", index=False)

    user_code, user_ids = pd.factorize(views["user_id"].to_numpy(), sort=True)
    pd.DataFrame({"user": np.arange(len(user_ids), dtype=np.int32), "user_id": user_ids}).to_parquet(
        WORK / "users.parquet", index=False
    )

    v_idx, a_idx, overlap = interval_join(
        views["channel_id"].to_numpy(), views["start"].to_numpy(), views["stop"].to_numpy(),
        airings["channel_id"].to_numpy(), airings["start"].to_numpy(), airings["end"].to_numpy(),
    )
    pairs = pd.DataFrame(
        {
            "view": v_idx.astype(np.int32),
            "user": user_code[v_idx].astype(np.int32),
            "airing": a_idx.astype(np.int32),
            "channel_id": airings["channel_id"].to_numpy()[a_idx].astype(np.int32),
            "tv_show_id": airings["tv_show_id"].to_numpy()[a_idx],
            "overlap": overlap.astype(np.int32),
        }
    )
    pairs.to_parquet(WORK / "pairs.parquet", index=False)

    matched_views = np.zeros(len(views), dtype=bool)
    matched_views[v_idx] = True
    labelled = pairs["tv_show_id"].to_numpy() != 0
    stats.update(
        {
            "views": int(len(views)),
            "views_with_any_airing": int(matched_views.sum()),
            "view_seconds": int(views["duration"].sum()),
            "view_seconds_matched": int(overlap.sum()),
            "view_seconds_on_labelled_airings": int(overlap[labelled].sum()),
            "pairs": int(len(pairs)),
            "labelled_pairs": int(labelled.sum()),
            "users": int(len(user_ids)),
            "seconds": round(time.perf_counter() - t0, 1),
        }
    )
    (WORK / "join_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
