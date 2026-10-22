"""Submit a file to the sweet.tv competition and keep a record of it.

  python experiments/kaggle_submit.py submit <file> "<label>"   submit; the description is "<NAME> | <label>"
  python experiments/kaggle_submit.py scores [--wait 600]         status and scores of the logged submissions
                                                                  -> experiments/kaggle_scores.json

The description is assembled here, in UTF-8, so the Cyrillic name never passes through a shell's
code page. Every submission is appended to submissions/log.jsonl with the file's md5. `scores` only
reports submissions listed in that log.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "submissions" / "log.jsonl"
SCORES = ROOT / "experiments" / "kaggle_scores.json"
COMPETITION = "sweettv-tv-program-recommender"
NAME = "Косычев Алексей"


def field(obj, *names):
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return None


def read_log() -> list[dict]:
    return [json.loads(line) for line in LOG.read_text(encoding="utf-8").splitlines() if line.strip()] if LOG.exists() else []


def submit(api: KaggleApi, file: Path, label: str) -> None:
    description = f"{NAME} | {label}"
    md5 = hashlib.md5(file.read_bytes()).hexdigest()
    response = api.competition_submit(str(file), description, COMPETITION, quiet=True)
    entry = {"submitted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "file": str(file.resolve().relative_to(ROOT)).replace("\\", "/"), "md5": md5,
             "description": description, "ref": field(response, "ref"), "message": field(response, "message")}
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(json.dumps(entry, ensure_ascii=False, indent=2))


def scores(api: KaggleApi, wait: int) -> None:
    logged = read_log()
    descriptions = {e["description"] for e in logged}
    refs = {str(e["ref"]) for e in logged if e.get("ref")}
    deadline = time.time() + wait
    while True:
        mine = [s for s in api.competition_submissions(COMPETITION)
                if str(field(s, "ref")) in refs or field(s, "description") in descriptions]
        pending = [s for s in mine if "complete" not in str(field(s, "status")).lower()
                   and "error" not in str(field(s, "status")).lower()]
        if not pending or time.time() > deadline:
            break
        time.sleep(15)
    rows = [{"ref": field(s, "ref"), "date": str(field(s, "date")), "description": field(s, "description"),
             "status": str(field(s, "status")), "public": field(s, "public_score", "publicScore"),
             "private": field(s, "private_score", "privateScore"), "file": field(s, "file_name", "fileName")}
            for s in mine]
    md5 = {e["description"]: e["md5"] for e in logged}
    for r in rows:
        r["md5"] = md5.get(r["description"])
    SCORES.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("file", type=Path)
    s.add_argument("label")
    c = sub.add_parser("scores")
    c.add_argument("--wait", type=int, default=0)
    args = p.parse_args()
    api = KaggleApi()
    api.authenticate()
    if args.cmd == "submit":
        submit(api, args.file, args.label)
    else:
        scores(api, args.wait)


if __name__ == "__main__":
    main()
