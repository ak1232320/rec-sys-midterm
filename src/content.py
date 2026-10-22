"""Programme attribute vectors for content-based filtering.

Tokens per programme, collected over every schedule row that carries its id (the attributes are static
for almost all ids; the rare conflicting values are all kept):
  cat:<category>   genre:<genre_1..3>   decade:<year of production // 10>   dir:<director>   act:<actor>

Each field is TF-IDF weighted, L2-normalised within the field and multiplied by its field weight. The
fields are then concatenated and the whole row is L2-normalised. Actors and directors that appear on a
single programme carry no similarity and are dropped.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import scipy.sparse as sp

FIELDS = ("cat", "genre", "decade", "dir", "act")
DEFAULT_WEIGHTS = {"cat": 1.0, "genre": 1.0, "decade": 0.5, "dir": 0.5, "act": 1.0}


def programme_tokens(schedule: pd.DataFrame) -> pd.DataFrame:
    """One row per (tv_show_id, field, token)."""
    s = schedule[schedule["tv_show_id"] != 0].drop_duplicates(
        ["tv_show_id", "tv_show_category", "tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3",
         "year_of_production", "director", "actors"])
    parts = [pd.DataFrame({"tv_show_id": s["tv_show_id"], "field": "cat", "token": s["tv_show_category"]})]
    for g in ("tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3"):
        parts.append(pd.DataFrame({"tv_show_id": s["tv_show_id"], "field": "genre", "token": s[g]}))
    year = s["year_of_production"].str.extract(r"(\d{4})", expand=False)
    parts.append(pd.DataFrame({"tv_show_id": s["tv_show_id"], "field": "decade",
                               "token": (year.astype(float) // 10 * 10).astype("Int64").astype(str).where(year.notna())}))
    for col, field in (("director", "dir"), ("actors", "act")):
        names = s[["tv_show_id", col]].dropna()
        names = names.assign(token=names[col].map(lambda v: [n.strip() for n in re.split(r"[,;]", v) if n.strip()]))
        names = names.explode("token")[["tv_show_id", "token"]]
        parts.append(names.assign(field=field))
    tokens = pd.concat(parts, ignore_index=True).dropna(subset=["token"])
    tokens["token"] = tokens["token"].astype(str).str.strip().str.lower()
    tokens = tokens[tokens["token"] != ""].drop_duplicates()
    # names on a single programme cannot link two programmes
    df = tokens.groupby(["field", "token"])["tv_show_id"].transform("nunique")
    return tokens[~(tokens["field"].isin(["dir", "act"]) & (df < 2))]


def item_matrix(tokens: pd.DataFrame, items: np.ndarray, weights: dict | None = None) -> sp.csr_matrix:
    """L2-normalised attribute vectors, row i = programme items[i] (zero row if it has no attributes)."""
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    tok = tokens[tokens["tv_show_id"].isin(items)]
    blocks = []
    for field in FIELDS:
        t = tok[tok["field"] == field]
        vocab, col = np.unique(t["token"].to_numpy(), return_inverse=True)
        rows = np.searchsorted(items, t["tv_show_id"].to_numpy())
        m = sp.csr_matrix((np.ones(len(t), dtype=np.float32), (rows, col)), shape=(len(items), len(vocab)))
        df = np.asarray((m > 0).sum(axis=0)).ravel()
        m = m @ sp.diags(np.log((1 + len(items)) / (1 + df)).astype(np.float32) + 1)
        norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
        m = sp.diags(np.divide(weights[field], norms, out=np.zeros_like(norms), where=norms > 0)) @ m
        blocks.append(m)
    v = sp.hstack(blocks).tocsr()
    norms = np.sqrt(np.asarray(v.multiply(v).sum(axis=1)).ravel())
    return (sp.diags(np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)) @ v).tocsr().astype(np.float32)
