"""Stage 1: exploratory analysis of the sweet.tv data.

Reads the caches written by src/ingest.py and src/join.py and writes
  * experiments/eda.json        every number quoted in experiments/EDA.md
  * experiments/figures/*.png   the figures

Windows: A = weeks 11-20, B = weeks 21-30, C = weeks 31-42 (test: schedule only,
no views). A -> B is used as a stand-in for B -> C wherever the question is "how
much does the next window repeat the previous one". The test users' ids come from
submission.csv; no test-window viewing exists in the data, so none is used.

Run:  python experiments/eda.py   (after src/ingest.py and src/join.py; ~1-2 min)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from ingest import WEEK11_MONDAY  # noqa: E402
from interactions import (  # noqa: E402
    COVER, MIN_SECONDS, SIGNALS, WEEK21_MONDAY, eligible, popularity, top_lists, user_airing, user_programme_week,
)
from interactions import window as signal_window  # noqa: E402
from metrics import mapk  # noqa: E402

WORK = ROOT / "work"
OUT_JSON = Path(__file__).resolve().parent / "eda.json"
FIG = Path(__file__).resolve().parent / "figures"

DST_SWITCH = int(np.datetime64("2020-03-29T03:00:00", "s").astype(np.int64))  # Kyiv: 03:00 -> 04:00
LONG_VIEW = 4 * 3600
DEFINITIONS = SIGNALS

T0 = time.perf_counter()
R: dict = {}

# Reference palette (dataviz skill, light mode): slots 1-3, surface and ink.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update(
    {
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
        "legend.frameon": False, "lines.linewidth": 1.6,
    }
)


def lap(msg: str) -> None:
    print(f"[{time.perf_counter() - T0:6.1f}s] {msg}", flush=True)


def q(x, ps=(1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9)) -> dict:
    return {str(p): float(v) for p, v in zip(ps, np.percentile(x, ps))}


def r4(x) -> float:
    return round(float(x), 4)


def auc(score: np.ndarray, positive: np.ndarray) -> float:
    ranks = rankdata(score)
    n_pos, n_neg = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# ----------------------------------------------------------------------------- load
views = pd.read_parquet(WORK / "views.parquet")
user_ids = pd.read_parquet(WORK / "users.parquet")["user_id"].to_numpy()
views["user"] = np.searchsorted(user_ids, views["user_id"].to_numpy()).astype(np.int32)
test_ids = pd.read_parquet(WORK / "test_users.parquet")["user_id"].to_numpy()
test_code = np.searchsorted(user_ids, test_ids)
assert np.all(user_ids[np.minimum(test_code, len(user_ids) - 1)] == test_ids), "a test user is missing from the log"
is_test = np.zeros(len(user_ids), dtype=bool)
is_test[test_code] = True
airings = pd.read_parquet(WORK / "airings_train.parquet",
                          columns=["window", "channel_id", "start", "end", "duration", "week", "tv_show_id"])
pairs = pd.read_parquet(WORK / "pairs.parquet")
schedule = pd.read_parquet(WORK / "schedule.parquet",
                           columns=["window", "channel_id", "start", "duration", "week", "tv_show_id", "tv_show_category"])
join_stats = json.loads((WORK / "join_stats.json").read_text())
lap("loaded")

# ----------------------------------------------------------------------------- 1. inventory
R["inventory"] = {
    "views": int(len(views)), "users": int(len(user_ids)), "view_channels": int(views["channel_id"].nunique()),
    "view_weeks": [int(views["week"].min()), int(views["week"].max())],
    "test_users": int(len(test_ids)), "test_users_in_log": int(is_test.sum()),
    "schedule_rows": {w: int(n) for w, n in schedule["window"].value_counts().sort_index().items()},
    "view_channels_missing_from_schedule": sorted(
        int(c) for c in set(views["channel_id"].unique()) - set(schedule["channel_id"].unique())),
    "join": join_stats,
}
missing = R["inventory"]["view_channels_missing_from_schedule"]
R["inventory"]["views_on_missing_channels"] = int(views["channel_id"].isin(missing).sum())

# ----------------------------------------------------------------------------- 2. clock alignment
d_span = (views["stop"] - views["start"] - views["duration"]).to_numpy()
mism = d_span != 0
spans_dst = (views["start"].to_numpy() < DST_SWITCH) & (views["stop"].to_numpy() >= DST_SWITCH)
key = lambda c, t: np.asarray(c, dtype=np.int64) * (np.int64(1) << 32) + t  # noqa: E731
lab_air = airings["tv_show_id"].to_numpy() != 0
bounds = np.unique(np.r_[key(airings["channel_id"], airings["start"].to_numpy()),
                         key(airings["channel_id"], airings["end"].to_numpy())])
lab_starts = np.unique(key(airings["channel_id"].to_numpy()[lab_air], airings["start"].to_numpy()[lab_air]))


def nearest_offset(keys: np.ndarray, sorted_bounds: np.ndarray) -> np.ndarray:
    i = np.searchsorted(sorted_bounds, keys)
    left = sorted_bounds[np.clip(i - 1, 0, len(sorted_bounds) - 1)]
    right = sorted_bounds[np.clip(i, 0, len(sorted_bounds) - 1)]
    return np.where(np.abs(keys - left) <= np.abs(right - keys), keys - left, keys - right)


vch = views["channel_id"].to_numpy()
shift_rows = []
for sh in (-3 * 3600, -2 * 3600, -3600, -300, -60, 0, 60, 300, 3600, 2 * 3600, 3 * 3600):
    stops = np.abs(nearest_offset(key(vch, views["stop"].to_numpy() + sh), bounds)) <= 60
    starts = np.abs(nearest_offset(key(vch, views["start"].to_numpy() + sh), lab_starts)) <= 60
    shift_rows.append({"shift_s": sh, "stops_within_60s_of_boundary": r4(stops.mean()),
                       "starts_within_60s_of_labelled_start": r4(starts.mean())})
start_offset = nearest_offset(key(vch, views["start"].to_numpy()), lab_starts)
hour = (views["start"].to_numpy() % 86400) // 3600
by_hour = np.bincount(hour, weights=views["duration"].to_numpy(), minlength=24)
R["clock"] = {
    "duration_ne_stop_minus_start": int(mism.sum()),
    "those_differences_s": {str(k): int(v) for k, v in zip(*np.unique(d_span[mism], return_counts=True))},
    "those_spanning_dst_switch_2020_03_29_03h": int((mism & spans_dst).sum()),
    "all_views_spanning_dst_switch": int(spans_dst.sum()),
    "shift_test": shift_rows,
    "view_seconds_share_by_start_hour": [r4(x) for x in by_hour / by_hour.sum()],
}
lap("clock")

# ----------------------------------------------------------------------------- 3. view durations
dur = views["duration"].to_numpy()
cuts = (61, 120, 300, 600, 1800, 3600, LONG_VIEW, 86400)
vs = views.sort_values(["user", "start"], kind="stable")
u_sorted = vs["user"].to_numpy()
prev_max_stop = vs.groupby("user")["stop"].cummax().shift(1).to_numpy()
simultaneous = np.r_[False, u_sorted[1:] == u_sorted[:-1]] & (vs["start"].to_numpy() < prev_max_stop)
R["view_durations"] = {
    "min_s": int(dur.min()), "max_s": int(dur.max()), "quantiles_s": q(dur),
    "share_of_views_below": {str(c): r4((dur < c).mean()) for c in cuts},
    "share_of_view_seconds_below": {str(c): r4(dur[dur < c].sum() / dur.sum()) for c in cuts},
    "views_longer_than_24h": int((dur > 86400).sum()),
    "views_starting_before_an_earlier_view_of_same_user_ended": r4(simultaneous.mean()),
    "users_with_any_simultaneous_view": r4(pd.Series(simultaneous).groupby(u_sorted).any().mean()),
    "exact_duplicate_views": int(views.duplicated(["user", "channel_id", "start", "stop"]).sum()),
}
del vs, prev_max_stop
lap("view durations")

# ----------------------------------------------------------------------------- 4. airing durations and labelling
sch_lab = schedule["tv_show_id"].to_numpy() != 0
R["airings"] = {}
for w, g in schedule.groupby("window"):
    lab = g["tv_show_id"].to_numpy() != 0
    gd = g["duration"].to_numpy()
    R["airings"][w] = {
        "rows": int(len(g)), "labelled_rows": int(lab.sum()), "labelled_row_share": r4(lab.mean()),
        "labelled_airtime_share": r4(gd[lab].sum() / gd.sum()),
        "distinct_programmes": int(g.loc[lab, "tv_show_id"].nunique()),
        "channels": int(g["channel_id"].nunique()),
        "channels_with_any_labelled_airing": int(g.loc[lab, "channel_id"].nunique()),
        "duration_median_labelled_s": float(np.median(gd[lab])),
        "duration_median_unlabelled_s": float(np.median(gd[~lab])),
        "duration_quantiles_labelled_s": q(gd[lab], (10, 50, 90)),
    }
cat = (schedule.assign(lab=sch_lab, air=schedule["duration"].where(sch_lab, 0))
       .groupby("tv_show_category", dropna=False)
       .agg(rows=("lab", "size"), labelled=("lab", "sum"), airtime=("duration", "sum"), lab_airtime=("air", "sum")))
R["labelling_by_category"] = {
    str(c): {"rows": int(r.rows), "labelled_row_share": r4(r.labelled / r.rows),
             "labelled_airtime_share": r4(r.lab_airtime / r.airtime)}
    for c, r in cat.sort_values("rows", ascending=False).iterrows()
}

pair_lab = pairs["tv_show_id"].to_numpy() != 0
ov = pairs["overlap"].to_numpy().astype(np.int64)
pair_week = np.clip(airings["week"].to_numpy()[pairs["airing"].to_numpy()], 11, 30)
wk_all = np.bincount(pair_week - 11, weights=ov, minlength=20)
wk_lab = np.bincount(pair_week - 11, weights=ov * pair_lab, minlength=20)
user_all = np.bincount(pairs["user"].to_numpy(), weights=ov, minlength=len(user_ids))
user_lab = np.bincount(pairs["user"].to_numpy(), weights=ov * pair_lab, minlength=len(user_ids))
user_share = np.divide(user_lab, user_all, out=np.zeros_like(user_lab), where=user_all > 0)
R["viewing_on_labelled_airings"] = {
    "share_of_matched_view_seconds": r4(ov[pair_lab].sum() / ov.sum()),
    "share_by_week": {str(w + 11): r4(wk_lab[w] / wk_all[w]) for w in range(20)},
    "per_user_share_quantiles": q(user_share, (10, 25, 50, 75, 90)),
    "users_with_zero_labelled_seconds": int((user_lab == 0).sum()),
    "test_users_with_zero_labelled_seconds": int((user_lab[is_test] == 0).sum()),
}
lap("labelling")

# ----------------------------------------------------------------------------- 5. user x airing interactions
UA = user_airing(pairs, views, airings)
air_len = (airings["end"] - airings["start"]).to_numpy()[UA["airing"].to_numpy()]
best = UA["best_single_cover"].to_numpy()
lp = pairs[pair_lab]
pair_cover = lp["overlap"].to_numpy() / (airings["end"] - airings["start"]).to_numpy()[lp["airing"].to_numpy()]
pair_long = dur[lp["view"].to_numpy()] > LONG_VIEW
n80_from_long = (pair_long & (pair_cover >= COVER)).sum() / (pair_cover >= COVER).sum()
del lp, pair_cover, pair_long
lap("user x airing")

un = UA["union"].to_numpy()
uc = UA["union_cover"].to_numpy()
R["watched_threshold"] = {
    "user_airing_pairs": int(len(UA)),
    "union_seconds_quantiles": q(un),
    "fragments_below_60s": {"pairs": r4((un < MIN_SECONDS).mean()), "seconds": r4(un[un < MIN_SECONDS].sum() / un.sum())},
    "pairs_where_summed_views_exceed_airing": int((UA["summed"].to_numpy() > air_len).sum()),
    "pairs_where_summed_exceeds_union": int((UA["summed"].to_numpy() > un).sum()),
    "absolute_cut": {str(t): {"pairs_kept": r4((un >= t).mean()), "seconds_kept": r4(un[un >= t].sum() / un.sum())}
                     for t in (60, 120, 300, 600, 900, 1800)},
    "union_cover_cut": {str(t): {"pairs_kept": r4((uc >= t).mean()), "seconds_kept": r4(un[uc >= t].sum() / un.sum())}
                        for t in (0.05, 0.1, 0.2, 0.5, 0.8)},
    "single_view_cover_0.8": {"pairs_kept": r4((best >= COVER).mean()), "seconds_kept": r4(un[best >= COVER].sum() / un.sum())},
    "single_view_cover_0.8_from_views_longer_than_4h": r4(n80_from_long),
    "union_cover_hist_20": [r4(x) for x in np.histogram(np.clip(uc, 0, 1), bins=20, range=(0, 1))[0] / len(uc)],
    "single_cover_hist_20": [r4(x) for x in np.histogram(np.clip(best, 0, 1), bins=20, range=(0, 1))[0] / len(best)],
}

# per (user, programme, week): seconds, airings with >= 60 s, airings watched >= 80 % in one view
UPW = user_programme_week(UA)


def window(lo: int, hi: int) -> pd.DataFrame:
    return signal_window(UPW, lo, hi)


A, B = window(11, 20), window(21, 30)
lap("user x programme")

# ----------------------------------------------------------------------------- 6. what "most preferred" can mean
truth_B = {d: {u: set(v) for u, v in top_lists(B, d).items()} for d in DEFINITIONS}
order_B = {d: top_lists(B, d) for d in DEFINITIONS}
agreement = {}
for pred_def in DEFINITIONS:
    for true_def in DEFINITIONS:
        users = list(truth_B[true_def])
        agreement[f"{pred_def}->{true_def}"] = r4(mapk(order_B[pred_def], truth_B[true_def], users))
items_per_user = {d: eligible(B, d).groupby("user").size().reindex(range(len(user_ids)), fill_value=0).to_numpy()
                  for d in DEFINITIONS}
hours_total = np.bincount(views["user"], weights=dur, minlength=len(user_ids)) / 3600
in_B = views["start"].to_numpy() >= WEEK21_MONDAY
hours_B = np.bincount(views["user"].to_numpy()[in_B], weights=dur[in_B], minlength=len(user_ids)) / 3600
R["preference_definitions"] = {
    "definitions": {
        "secs": "total seconds of the programme watched (union of the user's views per airing), >= 60 s",
        "n60": "number of airings with >= 60 s watched",
        "n80": "number of airings of which a single view covered >= 80 % (no channel change)",
    },
    "map5_oracle_of_one_definition_scored_on_another_window_B": agreement,
    "users_with_items_in_B": {d: {"any": int((items_per_user[d] > 0).sum()), "ge5": int((items_per_user[d] >= 5).sum()),
                                  "test_any": int((items_per_user[d][is_test] > 0).sum()),
                                  "test_ge5": int((items_per_user[d][is_test] >= 5).sum())} for d in DEFINITIONS},
    "test_membership_auc": {
        "hours_total": r4(auc(hours_total, is_test)), "hours_B": r4(auc(hours_B, is_test)),
        "labelled_hours_total": r4(auc(user_lab, is_test)),
        **{f"programmes_B_{d}": r4(auc(items_per_user[d], is_test)) for d in DEFINITIONS},
    },
}
lap("definitions")

# ----------------------------------------------------------------------------- 7. does B repeat A?
ids = {w: set(g.loc[g["tv_show_id"] != 0, "tv_show_id"].unique()) for w, g in schedule.groupby("window")}
A_any = eligible(A, "secs").groupby("user")["tv_show_id"].agg(set).to_dict()
A_recent = window(16, 20)
rep = {}
for d in DEFINITIONS:
    truth = truth_B[d]
    users = sorted(truth)
    test_users = [u for u in users if is_test[u]]
    hist_top = top_lists(A, d)
    personal = top_lists(A, d, allowed=ids["w21_30"])
    pop = popularity(A, d, allowed=ids["w21_30"])
    items = [(u, p) for u in users for p in truth[u]]
    row = {
        "users_with_truth": len(users), "test_users_with_truth": len(test_users),
        "mean_truth_size": r4(np.mean([len(truth[u]) for u in users])),
        "truth_items_seen_in_A_history": r4(np.mean([p in A_any.get(u, ()) for u, p in items])),
        "truth_items_in_A_top5": r4(np.mean([p in hist_top.get(u, ()) for u, p in items])),
        "truth_items_not_aired_in_A": r4(np.mean([p not in ids["w11_20"] for _, p in items])),
        "truth_items_still_aired_in_C": r4(np.mean([p in ids["w31_42"] for _, p in items])),
        "users_without_A_history": int(sum(u not in A_any for u in users)),
        "map5_personal_A_top5": r4(mapk(personal, truth, users)),
        "map5_popularity_A_top5": r4(mapk({u: pop for u in users}, truth, users)),
        "map5_personal_A_top5_test_users": r4(mapk(personal, truth, test_users)),
        "map5_popularity_A_top5_test_users": r4(mapk({u: pop for u in users}, truth, test_users)),
        "popularity_list": [int(x) for x in pop],
    }
    if d == "secs":
        row["map5_personal_weeks16_20_only"] = r4(mapk(top_lists(A_recent, d, allowed=ids["w21_30"]), truth, users))
    rep[d] = row
R["repeatability_A_to_B"] = rep
lap("repeatability")

# ----------------------------------------------------------------------------- 8. programme overlap between windows
def jacc(x: set, y: set) -> float:
    return r4(len(x & y) / len(x | y))


sch_C = schedule[(schedule["window"] == "w31_42") & sch_lab]
seen_AB = ids["w11_20"] | ids["w21_30"]
watched_AB = set(UA["tv_show_id"].unique())
airings_per_id = {w: g.loc[g["tv_show_id"] != 0].groupby("tv_show_id").size() for w, g in schedule.groupby("window")}
B_secs = B.groupby("tv_show_id")["secs"].sum()
A_secs = A.groupby("tv_show_id")["secs"].sum()
regular_B = set(airings_per_id["w21_30"][airings_per_id["w21_30"] >= 20].index)
R["programme_overlap"] = {
    "distinct": {w: len(s) for w, s in ids.items()},
    "intersections": {
        "11_20&21_30": len(ids["w11_20"] & ids["w21_30"]), "21_30&31_42": len(ids["w21_30"] & ids["w31_42"]),
        "11_20&31_42": len(ids["w11_20"] & ids["w31_42"]), "all_three": len(ids["w11_20"] & ids["w21_30"] & ids["w31_42"]),
    },
    "jaccard": {"11_20~21_30": jacc(ids["w11_20"], ids["w21_30"]), "21_30~31_42": jacc(ids["w21_30"], ids["w31_42"]),
                "11_20~31_42": jacc(ids["w11_20"], ids["w31_42"])},
    "C_ids_new_vs_A_or_B": len(ids["w31_42"] - seen_AB),
    "C_labelled_airings_of_ids_aired_in_A_or_B": r4(sch_C["tv_show_id"].isin(seen_AB).mean()),
    "C_labelled_airtime_of_ids_aired_in_A_or_B": r4(sch_C.loc[sch_C["tv_show_id"].isin(seen_AB), "duration"].sum()
                                                    / sch_C["duration"].sum()),
    "C_ids_watched_by_anyone_in_A_or_B": len(ids["w31_42"] & watched_AB),
    "C_labelled_airings_of_ids_watched_in_A_or_B": r4(sch_C["tv_show_id"].isin(watched_AB).mean()),
    "B_viewing_seconds_on_ids_that_air_in_C": r4(B_secs[B_secs.index.isin(ids["w31_42"])].sum() / B_secs.sum()),
    "A_viewing_seconds_on_ids_that_air_in_B": r4(A_secs[A_secs.index.isin(ids["w21_30"])].sum() / A_secs.sum()),
    "B_ids_aired_once": r4((airings_per_id["w21_30"] == 1).mean()),
    "B_ids_aired_20_plus_times": r4((airings_per_id["w21_30"] >= 20).mean()),
    "B_viewing_seconds_on_ids_aired_20_plus_times": r4(B_secs[B_secs.index.isin(regular_B)].sum() / B_secs.sum()),
}
lap("programme overlap")

# ----------------------------------------------------------------------------- 9. weekly activity and test users
wk = views.groupby("week").agg(users=("user", "nunique"), views=("user", "size"), hours=("duration", "sum"))
wk["hours"] /= 3600
active = views.groupby(["week", "user"]).size().reset_index()[["week", "user"]]
active["test"] = is_test[active["user"].to_numpy()]
act_test = active[active["test"]].groupby("week").size() / is_test.sum()
act_rest = active[~active["test"]].groupby("week").size() / (~is_test).sum()
weeks_active = active.groupby("user").size().reindex(range(len(user_ids)), fill_value=0).to_numpy()
last_week = views.groupby("user")["week"].max().reindex(range(len(user_ids))).to_numpy()
first_week = views.groupby("user")["week"].min().reindex(range(len(user_ids))).to_numpy()
programmes_AB = eligible(window(11, 30), "secs").groupby("user").size().reindex(range(len(user_ids)), fill_value=0).to_numpy()
day = (views["start"].to_numpy() - WEEK11_MONDAY) // 86400
all_days = np.arange(140)  # weeks 11-30
daily = pd.DataFrame({"day": day, "user": views["user"].to_numpy(), "dur": dur, "test": is_test[views["user"].to_numpy()]})
daily_users = daily.groupby("day")["user"].nunique().reindex(all_days, fill_value=0)
daily_test = daily[daily["test"]].groupby("day")["user"].nunique().reindex(all_days, fill_value=0) / is_test.sum()
daily_rest = daily[~daily["test"]].groupby("day")["user"].nunique().reindex(all_days, fill_value=0) / (~is_test).sum()
daily_hours = daily.groupby("day")["dur"].sum().reindex(all_days, fill_value=0) / 3600
daily_long = daily.assign(long=daily["dur"] > LONG_VIEW).groupby("day")["long"].mean().reindex(all_days, fill_value=0)
day_name = lambda i: str(np.datetime64(int(WEEK11_MONDAY + i * 86400), "s").astype("datetime64[D]"))  # noqa: E731
low = daily_long[(daily_long < 0.005) & (daily_users > 0)].index
R["daily"] = {
    "days_without_any_view": [day_name(i) for i in all_days if daily_users[i] == 0],
    "days_below_half_median_users": [day_name(i) for i in all_days if 0 < daily_users[i] < daily_users.median() / 2],
    "median_daily_users": float(daily_users.median()),
    "median_daily_hours": round(float(daily_hours.median()), 1),
    "days_with_long_view_share_below_0.5pct": [day_name(i) for i in low],
    "daily_hours_mean_2020_06_17_to_07_01": round(float(daily_hours.loc[100:114].mean()), 1),
    "daily_hours_mean_weeks_11_19": round(float(daily_hours.loc[0:62].mean()), 1),
    "daily_hours_mean_weeks_28_30": round(float(daily_hours.loc[119:139].mean()), 1),
    "daily_users_mean_2020_06_17_to_07_01": round(float(daily_users.loc[100:114].mean()), 1),
    "daily_users_mean_weeks_11_19": round(float(daily_users.loc[0:62].mean()), 1),
    "daily_users_mean_weeks_28_30": round(float(daily_users.loc[119:139].mean()), 1),
}
R["weekly"] = {
    "per_week": {str(w): {"active_users": int(r.users), "views": int(r.views), "hours": round(float(r.hours), 1),
                          "test_active_share": r4(act_test.get(w, 0)), "other_active_share": r4(act_rest.get(w, 0))}
                 for w, r in wk.iterrows()},
    "weeks_active_median": {"test": float(np.median(weeks_active[is_test])), "other": float(np.median(weeks_active[~is_test]))},
    "first_week_median": {"test": float(np.median(first_week[is_test])), "other": float(np.median(first_week[~is_test]))},
    "last_week_median": {"test": float(np.median(last_week[is_test])), "other": float(np.median(last_week[~is_test]))},
    "test_users_active_in_week_30": int(((last_week == 30) & is_test).sum()),
    "test_users_last_active_before_week_26": int(((last_week < 26) & is_test).sum()),
    "users_last_active_before_week_26": int((last_week < 26).sum()),
}
R["test_users"] = {
    "hours_total_median": {"test": round(float(np.median(hours_total[is_test])), 1),
                           "other": round(float(np.median(hours_total[~is_test])), 1)},
    "distinct_programmes_11_30_median": {"test": float(np.median(programmes_AB[is_test])),
                                         "other": float(np.median(programmes_AB[~is_test]))},
    "test_users_with_no_programme_11_30": int((programmes_AB[is_test] == 0).sum()),
    "test_users_with_fewer_than_5_programmes_11_30": int((programmes_AB[is_test] < 5).sum()),
    "test_users_with_no_programme_21_30": int((items_per_user["secs"][is_test] == 0).sum()),
    "test_users_with_no_n80_airing_21_30": int((items_per_user["n80"][is_test] == 0).sum()),
}
lap("weekly")

# ----------------------------------------------------------------------------- figures
FIG.mkdir(exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
bins = np.logspace(0, 5.6, 57)
axes[0].hist(dur, bins=bins, color=BLUE, histtype="stepfilled", alpha=0.85)
axes[0].axvline(61, color=INK2, lw=0.8, ls="--")
axes[0].text(55, axes[0].get_ylim()[1] * 0.5, "shortest\nview: 61 s", color=INK2, fontsize=8, ha="right")
for ax in axes:
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y / 1000:.0f}k"))
axes[0].set(xscale="log", xlabel="view length, s (log)", ylabel="views", title="(a) View log: one row per view")
axes[1].hist(un, bins=bins, color=BLUE, histtype="stepfilled", alpha=0.85)
axes[1].axvspan(1, MIN_SECONDS, color=ORANGE, alpha=0.15, lw=0)
axes[1].text(1.5, axes[1].get_ylim()[1] * 0.9, f"< 60 s: {R['watched_threshold']['fragments_below_60s']['pairs']:.1%}\nof pairs",
             color=INK2, fontsize=8, va="top")
axes[1].set(xscale="log", xlabel="seconds of one airing watched, s (log)", ylabel="user x airing pairs",
            title="(b) Labelled airings: time watched")
fig.tight_layout()
fig.savefig(FIG / "durations.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(4.6, 2.6))
edges = np.linspace(0, 1, 21)
ax.stairs(np.array(R["watched_threshold"]["union_cover_hist_20"]), edges, color=BLUE, lw=1.6, label="all views of the airing")
ax.stairs(np.array(R["watched_threshold"]["single_cover_hist_20"]), edges, color=ORANGE, lw=1.6, label="best single view")
ax.axvline(COVER, color=INK2, lw=0.8, ls="--")
ax.text(COVER - 0.01, 0.08, "80 %", color=INK2, fontsize=8, ha="right")
ax.set(xlabel="share of the airing watched", ylabel="share of user x airing pairs", title="Coverage of labelled airings")
ax.legend(loc="upper center", bbox_to_anchor=(0.45, 0.85))
fig.tight_layout()
fig.savefig(FIG / "coverage.png", dpi=200)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
weeks = np.arange(11, 31)
x_days = 11 + all_days / 7
axes[0].plot(x_days, daily_test.to_numpy(), color=BLUE, lw=1.2, label="test users (1,260)")
axes[0].plot(x_days, daily_rest.to_numpy(), color=ORANGE, lw=1.2, label=f"other users ({(~is_test).sum():,})")
axes[0].axvspan(11 + 100 / 7, 11 + 117 / 7, color=INK2, alpha=0.08, lw=0)
axes[0].text(11 + 99.5 / 7, 0.04, "17 Jun - 3 Jul:\nlog thins out", color=INK2, fontsize=7, ha="right")
axes[0].set(xlabel="ISO week of 2020", ylabel="share of users active that day", ylim=(0, 1),
            xticks=range(11, 32, 2), title="(a) Daily activity")
axes[0].legend(loc="lower left", fontsize=7)
axes[1].plot(weeks, [wk_lab[w - 11] / wk_all[w - 11] for w in weeks], color=BLUE, marker="o", ms=3)
axes[1].axhline(R["airings"]["w11_20"]["labelled_airtime_share"], color=INK2, lw=0.8, ls="--")
axes[1].text(11, R["airings"]["w11_20"]["labelled_airtime_share"] + 0.02, "labelled share of airtime (11-20)", color=INK2, fontsize=8)
axes[1].set(xlabel="ISO week of 2020", ylabel="share of watched seconds", ylim=(0, 0.8),
            title="(b) Viewing that lands on labelled airings")
fig.tight_layout()
fig.savefig(FIG / "weekly.png", dpi=200)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
m = np.abs(start_offset) <= 900
axes[0].hist(start_offset[m] / 60, bins=np.arange(-15, 15.01, 0.5), color=BLUE, histtype="stepfilled", alpha=0.85)
axes[0].set(xlabel="view start - nearest labelled programme start, min", ylabel="views",
            title="(a) Viewers arrive as programmes start")
sh = [r["shift_s"] / 3600 for r in shift_rows]
axes[1].plot(sh, [r["stops_within_60s_of_boundary"] for r in shift_rows], color=BLUE, marker="o", ms=3,
             label="view stops within 60 s of a boundary")
axes[1].plot(sh, [r["starts_within_60s_of_labelled_start"] for r in shift_rows], color=ORANGE, marker="o", ms=3,
             label="view starts within 60 s of a labelled start")
axes[1].set(xlabel="shift applied to the view log, h", ylabel="share of views", ylim=(0, None),
            title="(b) No timezone shift improves the match")
axes[1].legend(loc="lower center", fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "clock.png", dpi=200)
plt.close(fig)
lap("figures")

R["runtime_s"] = round(time.perf_counter() - T0, 1)
OUT_JSON.write_text(json.dumps(R, indent=2, ensure_ascii=False))
print(json.dumps(R, indent=1, ensure_ascii=False))
