"""History replay: a user's own programmes ranked by one signal, padded with the most popular ones.

Used as the instrument of the target-definition probe (experiments/target_probe.py), not as one of
the two required methods. Every list has exactly k distinct ids from `candidates`: first the user's
own programmes that air in the target window, ranked by the signal, then the most popular candidates
under the same signal. A user with no history gets the popularity list alone.
"""
from __future__ import annotations

from interactions import popularity, top_lists


def recommend(hist, signal: str, users, candidates: set, k: int = 5) -> dict:
    personal = top_lists(hist, signal, k, allowed=candidates)
    pop = popularity(hist, signal, allowed=candidates, k=2 * k)
    out = {}
    for u in users:
        items = list(personal.get(u, []))
        for p in pop:
            if len(items) == k:
                break
            if p not in items:
                items.append(p)
        out[u] = items
    return out
