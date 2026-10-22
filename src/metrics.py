"""MAP@K as Kaggle's MAP@{K} computes it: AP@K = sum of precision@i at each hit, over min(|truth|, K)."""
from __future__ import annotations

import numpy as np

K = 5


def apk(pred: list, truth: set, k: int = K) -> float:
    hits, total = 0, 0.0
    for i, p in enumerate(pred[:k]):
        if p in truth and p not in pred[:i]:
            hits += 1
            total += hits / (i + 1)
    return total / min(len(truth), k) if truth else 0.0


def mapk(preds: dict, truths: dict, users, k: int = K) -> float:
    """Mean AP@K over `users`; a user with no prediction scores 0."""
    users = list(users)
    return float(np.mean([apk(preds.get(u, []), truths[u], k) for u in users])) if users else float("nan")
