# LLM4Rec midterm — sweet.tv TV programme recommender

Aleksei Kosychev (amkosychev@edu.hse.ru), HSE LLM4Rec, midterm quiz (GSB).
Competition: [sweet.tv — TV program recommender](https://www.kaggle.com/competitions/sweettv-tv-program-recommender)
(MAP@5, late submissions).

**Task.** Join the view log to the broadcast schedule, then predict for each of 1,260 users the five
programmes they will prefer most in weeks 31-42.

**Final submission:** `submissions/hybrid.csv` (md5 `9f8079869fed5d078f134109af58f49d`), Kaggle
**public 0.42645**, private 0.42206 (ref 57527334). Report: `report/kosychev_midterm_report.pdf`.

**What was found.**

* **The target.** It behaves like *the number of airings a user watched ≥ 80% of in one view* (n80).
  Two probe submissions that differ only in the ranking signal scored 0.396 (n80) and 0.279 (seconds),
  which is the pattern the n80 hypothesis predicts. Details: `experiments/EDA.md` §8b.
* **Habits dominate.** 67% of a user's next-window top-5 were already in their history. Every
  history-based method fills ~95% of its slots with familiar programmes, so repeats are allowed.
* **Labels are scarce.** Only 10.8% of schedule rows carry a programme id, yet 56% of viewing seconds
  land on labelled airings.

**Validation** (fit on weeks 11-20, truth = n80 top-5 in weeks 21-30; 4,623 users, of whom 1,236 are test
users) **and Kaggle** (every method refit on weeks 11-30):

| method | MAP@5 all | test users | Kaggle public / private |
|---|---|---|---|
| Popularity (baseline) | 0.2234 | 0.2477 | — |
| History replay (reference) | 0.3079 | 0.3518 | 0.3960 / 0.3877 (probe) |
| MF (a) implicit ALS | 0.3139 | 0.3502 | — |
| MF (b) weighted ALS on log n80 | 0.3241 | 0.3609 | 0.4191 / 0.4133 |
| Personalized PageRank | 0.3192 | 0.3571 | 0.4084 / 0.4017 |
| Association rules (temporal) | 0.2767 | 0.3062 | — |
| CBF | 0.1851 | 0.2067 | — |
| **Hybrid, rank fusion** (held-out) | **0.3273** | **0.3642** | **0.4265 / 0.4221** |

The hybrid beats the best single method by +0.0031 on held-out users (bootstrap 95% CI [0.0014, 0.0049]).
It is the best of all methods in every activity quartile.

**Decisions every method shares.**

* Join: same channel and overlapping intervals; a user's time on an airing is the union of their views.
* Interaction floor: 60 s.
* `tv_show_id = 0` is never an item.
* Candidate pool: the labelled programmes of the target window's schedule, repeats allowed.
* Users with no usable history get the popularity list.

Full reasoning in the report.

## Layout

| Path | What |
|---|---|
| `src/ingest.py` | raw CSV → typed parquet caches in `work/` |
| `src/join.py` | view × airing interval join (same channel, overlapping intervals) |
| `src/interactions.py` | user × programme × week signals: seconds, airings ≥ 60 s, airings ≥ 80 % in one view (n80) |
| `src/split.py` | history / target windows, candidate pool, truth, the shared Top-k with the popularity tie-break |
| `src/methods.py` | popularity, history replay, MF (implicit ALS and weighted ALS on counts), Personalized PageRank, association rules, CBF |
| `src/content.py` | programme attribute vectors for CBF |
| `src/fusion.py` | rank fusion used by the hybrid |
| `src/replay.py`, `src/submission.py` | probe instrument; writing and validating submission files |
| `experiments/eda.py`, `EDA.md` | exploratory analysis and its findings |
| `experiments/target_probe.py` | the Kaggle probe that identified the target definition |
| `experiments/validate.py` | hyperparameter grids on the 11-20 → 21-30 split → `validation.json` |
| `experiments/hybrid.py` | rank-fusion weights by 2-fold cross-validation over users → `hybrid.json` |
| `experiments/final.py` | every method fit on weeks 11-30 → `submissions/*.csv`, `final.json` |
| `experiments/examples.py` | two users, every method's Top-5 → `examples.md` |
| `experiments/kaggle_submit.py` | submissions to Kaggle, logged in `submissions/log.jsonl`; scores in `kaggle_scores.json` |
| `tests/` | `check_data.py` (ingest and join), `check_methods.py` (methods on hand-made data) |
| `report/` | report source and PDF |

## Reproduce

Download the competition data into `data_raw/` (accept the competition rules first; the files are not
redistributed here):

```
kaggle competitions download -c sweettv-tv-program-recommender -p data_raw && unzip data_raw/*.zip -d data_raw
```

Then, from the repository root:

```
python src/ingest.py                 # 11-23 s
python src/join.py                   # 16-45 s
python src/interactions.py           # 4-12 s
python experiments/eda.py            # 2.5-5 min
python experiments/target_probe.py   # ~10 s
python experiments/validate.py       # 10-40 min (all grids; run it in the foreground)
python experiments/hybrid.py         # 1-6 min
python experiments/final.py          # ~1 min
python experiments/examples.py       # ~0.5 min
python tests/check_data.py
python tests/check_methods.py
```

Environment actually used: Windows 11, Intel i5-13420H (12 threads), 16 GB RAM, Python 3.12.10,
numpy 2.3.5, pandas 2.3.3, pyarrow 26.0.0, scipy 1.17.0, implicit 0.7.3, threadpoolctl (via
scikit-learn 1.8.0), matplotlib 3.10.8, kaggle 2.2.4. Random seed 42 wherever randomness enters
(ALS initialisation, the cross-validation split). The ranges are the fastest and slowest runs on this laptop,
whose clock varied with temperature; a full rerun from an empty `work/` reproduced every JSON result and
every submission file byte for byte (md5 in `experiments/final.json` and `submissions/log.jsonl`).
