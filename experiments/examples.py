"""Two example users, every method's Top-5 side by side, on the validation split (11-20 -> 21-30),
where what they actually watched next is known.

The users are picked by a fixed rule, before looking at any list: among the test users with a
non-empty truth in weeks 21-30, the one at the 90th percentile of n80 airings in weeks 11-20 (heavy)
and the one at the 10th percentile (light).

Writes experiments/examples.json and experiments/examples.md.  Run:  python experiments/examples.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
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
EPISODE = re.compile(r"[,.]?\s*(\d+[-–]?\d*\s*(с\.|с|эп\.|эп|серия|серии|сезон)\b.*)$", re.IGNORECASE)


def programme_names(schedule: pd.DataFrame) -> pd.DataFrame:
    lab = schedule[schedule["tv_show_id"] != 0]
    title = lab.groupby("tv_show_id")["tv_show_title"].agg(lambda t: t.value_counts().index[0])
    title = title.str.replace("'", "", regex=False).map(lambda t: EPISODE.sub("", t).strip(" .,"))
    cat = lab.groupby("tv_show_id")["tv_show_category"].first()
    return pd.DataFrame({"title": title, "category": cat})


def main() -> None:
    upw = pd.read_parquet(WORK / "upw.parquet")
    schedule = pd.read_parquet(WORK / "schedule.parquet")
    n_users = len(pd.read_parquet(WORK / "users.parquet"))
    split = make_split("A->B", upw, schedule, n_users, (11, 20), "w21_30", (21, 30))
    tokens = content.programme_tokens(schedule)
    names = programme_names(schedule)
    selected = {m: r["selected"]["params"] for m, r in json.loads((HERE / "validation.json").read_text()).items()}
    hyb = json.loads((HERE / "hybrid.json").read_text())["final_weights"]

    test = set(int(c) for c in submission.test_user_codes())
    n80_total = split.hist.groupby("user")["n80"].sum()
    pool = sorted((u for u in split.truth if u in test and n80_total.get(u, 0) > 0), key=lambda u: (n80_total[u], u))
    chosen = {"heavy": pool[int(0.9 * (len(pool) - 1))], "light": pool[int(0.1 * (len(pool) - 1))]}

    lists, top_idx = {}, {}
    for m in SINGLE:
        scores = method_scores(m, split, selected.get(m, {}), tokens)
        lists[m] = top_k(scores, split, chosen.values())
        if m in COMPONENTS:
            top_idx[m] = ranked_top(scores, split)
    fused = fuse(top_idx, hyb["weights"], hyb["c"], split.n_users, len(split.candidates))
    lists["hybrid"] = top_k(fused, split, chosen.values())

    user_ids = pd.read_parquet(WORK / "users.parquet")["user_id"].to_numpy()
    out, md = {}, []
    for label, u in chosen.items():
        h = split.hist[(split.hist["user"] == u) & (split.hist["n80"] > 0)].sort_values(["n80", "secs"], ascending=False)
        seen = set(split.hist.loc[(split.hist["user"] == u) & (split.hist["secs"] >= 60), "tv_show_id"])
        truth = split.truth[u]

        def row(p):
            return {"id": int(p), "title": names.at[p, "title"], "category": names.at[p, "category"],
                    "in_history": p in seen, "in_actual_next_top5": p in truth}

        out[label] = {
            "user_id": str(user_ids[u]), "n80_airings_11_20": int(n80_total[u]),
            "programmes_with_n80_11_20": int(len(h)),
            "history_top5": [{**row(p), "n80": int(n)} for p, n in zip(h["tv_show_id"].head(5), h["n80"].head(5))],
            "actual_top5_21_30": [row(p) for p in sorted(truth)],
            "lists": {m: [row(p) for p in lists[m][u]] for m in lists},
            "hits": {m: sum(p in truth for p in lists[m][u]) for m in lists},
        }
        md.append(f"### {label} viewer (user {user_ids[u]}): {int(n80_total[u])} n80 airings of "
                  f"{len(h)} programmes in weeks 11-20\n")
        md.append("| method | " + " | ".join(f"#{i}" for i in range(1, 6)) + " | hits |")
        md.append("|---" * 7 + "|")
        for m, items in out[label]["lists"].items():
            cells = [f"{r['id']} {r['title'][:28]} ({r['category']}){' ✓' if r['in_actual_next_top5'] else ''}"
                     f"{'' if r['in_history'] else ' *new*'}" for r in items]
            md.append(f"| {m} | " + " | ".join(cells) + f" | {out[label]['hits'][m]} |")
        md.append("\nActual top-5 in weeks 21-30: " + "; ".join(
            f"{r['id']} {r['title'][:40]} ({r['category']}){'' if r['in_history'] else ' *new*'}"
            for r in out[label]["actual_top5_21_30"]) + "\n")
    (HERE / "examples.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    (HERE / "examples.md").write_text("\n".join(md), encoding="utf-8")
    (HERE / "examples_compact.md").write_text(compact(out), encoding="utf-8")
    print("\n".join(md))


SHORT_CAT = {"Инфо": "Инфо", "Развлечения": "Разв", "Детям": "Дет", "Фильмы": "Фильм", "Сериалы": "Сер",
             "Познавательное": "Позн", "Другое": "Др", "Спорт": "Спорт", "Музыка": "Муз"}
NAMES = {"popularity": "Popularity", "replay": "Replay", "mf_als": "MF (a)", "mf_wals": "MF (b)",
         "ppr": "PageRank", "assoc_rules": "Rules", "cbf": "CBF", "hybrid": "Hybrid"}


def compact(out: dict) -> str:
    """The report's version: id, short title, short category, ✓ = in the actual next top-5, * = new."""
    lines = ['<table class="ex"><tr><th>user</th><th>method</th>'
             + "".join(f"<th>#{i}</th>" for i in range(1, 6)) + "<th>hits</th></tr>"]
    for label, u in out.items():
        methods = list(NAMES)
        for j, m in enumerate(methods):
            cells = "".join(f"<td>{r['id']} {r['title'][:18]} [{SHORT_CAT.get(r['category'], r['category'])}]"
                            f"{' ✓' if r['in_actual_next_top5'] else ''}{'' if r['in_history'] else ' *'}</td>"
                            for r in u["lists"][m])
            head = (f'<td rowspan="{len(methods) + 1}">{label}: {u["n80_airings_11_20"]} n80 airings, '
                    f'{u["programmes_with_n80_11_20"]} programmes</td>') if j == 0 else ""
            lines.append(f"<tr>{head}<td>{NAMES[m]}</td>{cells}<td>{u['hits'][m]}</td></tr>")
        actual = "; ".join(f"{r['id']} {r['title'][:22]} [{SHORT_CAT.get(r['category'], r['category'])}]"
                           f"{'' if r['in_history'] else ' *'}" for r in u["actual_top5_21_30"])
        lines.append(f'<tr><td>actual 21-30</td><td colspan="6">{actual}</td></tr>')
    return "\n".join(lines) + "\n</table>"


if __name__ == "__main__":
    main()
