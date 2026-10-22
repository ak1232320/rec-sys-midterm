"""Write and validate Kaggle submission files.

The file copies the template's user_id text and row order exactly (the ids are uint64 and must not
pass through a float), and writes five space-separated tv_show_ids per row.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "data_raw" / "submission.csv"
WORK = ROOT / "work"
K = 5


def template_user_ids() -> list[str]:
    lines = TEMPLATE.read_text(encoding="ascii").splitlines()
    assert lines[0] == "user_id,tv_show_id"
    return [line.split(",", 1)[0] for line in lines[1:]]


def test_user_codes() -> np.ndarray:
    """Dense user codes (work/users.parquet) of the template's users, in template order."""
    user_ids = pd.read_parquet(WORK / "users.parquet")["user_id"].to_numpy()
    raw = np.array([int(u) for u in template_user_ids()], dtype=np.uint64)
    codes = np.searchsorted(user_ids, raw)
    assert np.all(user_ids[np.minimum(codes, len(user_ids) - 1)] == raw), "a template user is missing from the log"
    return codes


def write(recs: dict, path: Path) -> None:
    rows = ["user_id,tv_show_id"]
    for uid, code in zip(template_user_ids(), test_user_codes()):
        rows.append(f"{uid},{' '.join(str(int(p)) for p in recs[int(code)])}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("\n".join(rows) + "\n").encode("ascii"))


def validate(path: Path, candidates: set) -> dict:
    lines = path.read_text(encoding="ascii").splitlines()
    ids = [line.split(",", 1) for line in lines[1:]]
    lists = [[int(x) for x in row[1].split(" ")] for row in ids]
    report = {
        "header_ok": lines[0] == "user_id,tv_show_id",
        "rows": len(ids),
        "same_users_and_order_as_template": [row[0] for row in ids] == template_user_ids(),
        "all_rows_have_5_distinct_ids": all(len(lst) == K and len(set(lst)) == K for lst in lists),
        "all_ids_are_candidates": all(p in candidates for lst in lists for p in lst),
        "no_zero_ids": all(p != 0 for lst in lists for p in lst),
    }
    report["valid"] = all(v for k, v in report.items() if k != "rows") and report["rows"] == 1260
    return report
