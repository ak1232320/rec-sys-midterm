"""Convert the raw competition CSVs into typed parquet caches under work/.

The view log and the three schedule files are read once with pyarrow and
written as:

  * work/views.parquet     user_id (uint64), channel_id, start, stop, duration
  * work/schedule.parquet  one row per airing of all three schedule files, with
                           start/end, the week window it belongs to and every
                           metadata column kept as text
  * work/test_users.parquet the 1,260 user ids of submission.csv, in file order

Times stay naive local timestamps, stored as int64 seconds since 1970-01-01:
both sources use the same clock (checked in experiments/eda.py, section
"clock alignment"), so no timezone conversion is applied.

Run:  python src/ingest.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data_raw"
WORK = ROOT / "work"

VIEW_FILE = "dataset11-30.csv"
SCHEDULE_FILES = {
    "w11_20": "export_arh_11-20-final.csv",
    "w21_30": "export_arh_21-30-final.csv",
    "w31_42": "export_arh_31-42-final.csv",
}
SUBMISSION_FILE = "submission.csv"

# ISO week 11 of 2020 starts on Monday 2020-03-09; the view log's first row is 2020-03-09 00:00:02.
WEEK11_MONDAY = np.datetime64("2020-03-09T00:00:00", "s").astype(np.int64)
WEEK = 7 * 24 * 3600


def iso_week(seconds: np.ndarray) -> np.ndarray:
    """ISO week number of 2020 for naive epoch seconds (valid for weeks 2-52 of 2020)."""
    return 11 + np.floor_divide(seconds - WEEK11_MONDAY, WEEK)


def to_seconds(col: pa.Array) -> np.ndarray:
    return pc.cast(col, pa.timestamp("s")).to_numpy(zero_copy_only=False).astype("datetime64[s]").astype(np.int64)


def read_views() -> pa.Table:
    table = pv.read_csv(
        RAW / VIEW_FILE,
        convert_options=pv.ConvertOptions(
            column_types={
                "user_id": pa.uint64(),
                "vsetv_id": pa.int32(),
                "start_time": pa.timestamp("s"),
                "stop_time": pa.timestamp("s"),
                "duraton": pa.int64(),
            }
        ),
    )
    start = to_seconds(table["start_time"])
    stop = to_seconds(table["stop_time"])
    return pa.table(
        {
            "user_id": table["user_id"],
            "channel_id": table["vsetv_id"],
            "start": start,
            "stop": stop,
            "duration": table["duraton"],
            "week": iso_week(start).astype(np.int16),
        }
    )


def read_schedule(window: str, name: str) -> pa.Table:
    text_cols = [
        "channel_title", "tv_show_title", "tv_show_category", "tv_show_genre_1", "tv_show_genre_2",
        "tv_show_genre_3", "year_of_production", "director", "actors",
    ]
    table = pv.read_csv(
        RAW / name,
        convert_options=pv.ConvertOptions(
            column_types={
                "channel_id": pa.int32(),
                "start_time": pa.timestamp("s"),
                "duration": pa.int64(),
                "tv_show_id": pa.int64(),
                **{c: pa.string() for c in text_cols},
            },
            timestamp_parsers=["%d.%m.%Y %H:%M:%S"],
            strings_can_be_null=True,
        ),
    )
    start = to_seconds(table["start_time"])
    duration = table["duration"].to_numpy()
    n = table.num_rows
    cols = {
        "window": pa.array([window] * n, pa.string()),
        "row": np.arange(n, dtype=np.int64),
        "channel_id": table["channel_id"],
        "start": start,
        "end": start + duration,
        "duration": table["duration"],
        "week": iso_week(start).astype(np.int16),
        "tv_show_id": table["tv_show_id"],
    }
    cols.update({c: table[c] for c in text_cols})
    return pa.table(cols)


def read_submission_users() -> pa.Table:
    table = pv.read_csv(
        RAW / SUBMISSION_FILE,
        convert_options=pv.ConvertOptions(column_types={"user_id": pa.uint64(), "tv_show_id": pa.string()}),
    )
    return pa.table({"user_id": table["user_id"], "order": np.arange(table.num_rows, dtype=np.int32)})


def main() -> None:
    WORK.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    views = read_views()
    pq.write_table(views, WORK / "views.parquet")
    print(f"views     {views.num_rows:>10,} rows  {time.perf_counter() - t0:6.1f} s")

    t1 = time.perf_counter()
    schedule = pa.concat_tables([read_schedule(w, f) for w, f in SCHEDULE_FILES.items()])
    pq.write_table(schedule, WORK / "schedule.parquet")
    print(f"schedule  {schedule.num_rows:>10,} rows  {time.perf_counter() - t1:6.1f} s")

    users = read_submission_users()
    pq.write_table(users, WORK / "test_users.parquet")
    print(f"test users {users.num_rows:>9,} rows  total {time.perf_counter() - t0:6.1f} s")


if __name__ == "__main__":
    main()
