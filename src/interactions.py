"""User x programme interaction signals, built from the view x airing join.

Per user x labelled airing (`user_airing`):
  * union              seconds of the airing the user watched: the union of all their views of it, so
                       simultaneous views (several devices, 12% of views) are not counted twice;
  * best_single_cover  the largest share of the airing covered by ONE view (no channel change).

Per user x programme x week (`user_programme_week`), three signals:
  * secs  total union seconds;
  * n60   number of airings with >= 60 s watched (the view log has no view under 61 s, so anything
          shorter is the edge of a view straddling a programme boundary);
  * n80   number of airings that one view covered by >= 80 %.

Run:  python src/interactions.py   (after src/join.py) -> work/upw.parquet
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ingest import WEEK, WEEK11_MONDAY

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work"

MIN_SECONDS = 60
COVER = 0.8
SIGNALS = ("secs", "n60", "n80")
WEEK21_MONDAY = WEEK11_MONDAY + 10 * WEEK


def user_airing(pairs: pd.DataFrame, views: pd.DataFrame, airings: pd.DataFrame) -> pd.DataFrame:
    """One row per user x labelled airing they overlapped."""
    lp = pairs[pairs["tv_show_id"].to_numpy() != 0]
    v_i, a_i = lp["view"].to_numpy(), lp["airing"].to_numpy()
    a_start, a_end = airings["start"].to_numpy()[a_i], airings["end"].to_numpy()[a_i]
    src = pd.DataFrame({
        "user": lp["user"].to_numpy(), "airing": a_i,
        "cs": np.maximum(views["start"].to_numpy()[v_i], a_start),
        "ce": np.minimum(views["stop"].to_numpy()[v_i], a_end),
        "cover": lp["overlap"].to_numpy() / (a_end - a_start),
    }).sort_values(["user", "airing", "cs"], kind="stable")
    user, airing = src["user"].to_numpy(), src["airing"].to_numpy()
    new_group = np.r_[True, (user[1:] != user[:-1]) | (airing[1:] != airing[:-1])]
    gid = np.cumsum(new_group) - 1
    cs, ce = src["cs"].to_numpy(), src["ce"].to_numpy()
    run_max = pd.Series(ce).groupby(gid).cummax().to_numpy()
    prev = np.r_[np.iinfo(np.int64).min, run_max[:-1]]
    prev[new_group] = np.iinfo(np.int64).min
    union_part = np.maximum(0, ce - np.maximum(cs, prev))  # the part of this interval not covered yet
    first = np.flatnonzero(new_group)
    ua = pd.DataFrame({
        "user": user[first], "airing": airing[first],
        "union": np.add.reduceat(union_part, first),
        "summed": np.add.reduceat(ce - cs, first),
        "best_single_cover": np.maximum.reduceat(src["cover"].to_numpy(), first),
    })
    a = ua["airing"].to_numpy()
    ua["union_cover"] = ua["union"] / (airings["end"] - airings["start"]).to_numpy()[a]
    ua["tv_show_id"] = airings["tv_show_id"].to_numpy()[a]
    ua["week"] = np.clip(airings["week"].to_numpy()[a], 11, 30)
    return ua


def user_programme_week(ua: pd.DataFrame) -> pd.DataFrame:
    return (ua.assign(a60=ua["union"] >= MIN_SECONDS, a80=ua["best_single_cover"] >= COVER)
            .groupby(["user", "tv_show_id", "week"], sort=False)
            .agg(secs=("union", "sum"), n60=("a60", "sum"), n80=("a80", "sum")).reset_index())


def window(upw: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    """Signals summed over ISO weeks lo..hi, one row per user x programme."""
    w = upw[(upw["week"] >= lo) & (upw["week"] <= hi)]
    return w.groupby(["user", "tv_show_id"], sort=False)[list(SIGNALS)].sum().reset_index()


def eligible(df: pd.DataFrame, signal: str) -> pd.DataFrame:
    """Rows that count as an interaction under `signal`."""
    if signal == "secs":
        return df[df["secs"] >= MIN_SECONDS]
    return df[df[signal] > 0]


def top_lists(df: pd.DataFrame, signal: str, k: int = 5, allowed: set | None = None) -> dict:
    """Each user's programmes ranked by `signal` (ties: more seconds, then smaller id), first k."""
    d = eligible(df, signal)
    if allowed is not None:
        d = d[d["tv_show_id"].isin(allowed)]
    d = d.sort_values(["user", signal, "secs", "tv_show_id"], ascending=[True, False, False, True])
    head = d.groupby("user", sort=False).head(k)
    return {u: g.tolist() for u, g in head.groupby("user", sort=False)["tv_show_id"]}


def popularity(df: pd.DataFrame, signal: str, allowed: set | None = None, k: int = 5) -> list:
    """Programmes ranked by how many users interacted with them under `signal` (ties: total seconds)."""
    d = eligible(df, signal)
    if allowed is not None:
        d = d[d["tv_show_id"].isin(allowed)]
    pop = d.groupby("tv_show_id").agg(users=("user", "nunique"), secs=("secs", "sum"))
    return pop.sort_values(["users", "secs"], ascending=False).index[:k].tolist()


def main() -> None:
    t0 = time.perf_counter()
    views = pd.read_parquet(WORK / "views.parquet", columns=["start", "stop"])
    airings = pd.read_parquet(WORK / "airings_train.parquet", columns=["start", "end", "week", "tv_show_id"])
    pairs = pd.read_parquet(WORK / "pairs.parquet", columns=["view", "user", "airing", "tv_show_id", "overlap"])
    upw = user_programme_week(user_airing(pairs, views, airings))
    upw.to_parquet(WORK / "upw.parquet", index=False)
    print(f"user x programme x week: {len(upw):,} rows  {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
