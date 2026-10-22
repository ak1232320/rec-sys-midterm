# Stage 1: exploratory analysis

Every number below is in `experiments/eda.json`, written by `experiments/eda.py`. The figures are in
`experiments/figures/`. Reproduce it with:

```
python src/ingest.py        # raw CSV -> work/*.parquet                         (~11 s)
python src/join.py          # view x airing interval join -> work/pairs.parquet (~16 s)
python experiments/eda.py   # this analysis -> eda.json + figures               (~2.5 min)
python tests/check_data.py  # 17 checks on the ingest and the join              (~10 s)
python src/interactions.py          # user x programme x week signals -> work/upw.parquet (~4 s)
python experiments/target_probe.py  # §8b proxy table + submissions/probe_*.csv           (~10 s)
```

Environment: Windows 11, Intel i5-13420H, 16 GB RAM, Python 3.12.10, pandas 2.3.3, pyarrow 26.0.0,
numpy 2.3.5, scipy 1.17.0, matplotlib 3.10.8.

Windows used throughout: **A** = weeks 11-20, **B** = weeks 21-30, **C** = weeks 31-42 (the test window,
which has a schedule but no views). Wherever the question is "how well does the past predict the next
window", **A → B** stands in for **B → C**. The test users' ids come from `submission.csv`. The data
contains no viewing from the test window, so none is used.

## 1. What is in the data

| | |
|---|---|
| View log | 5,821,987 views, 4,914 users, 202 channels, ISO weeks 11-30 of 2020 (9 Mar - 26 Jul) |
| Schedules | 1,074,235 / 1,074,394 / 1,295,378 rows; 400 / 403 / 404 channels |
| Test users | 1,260, **all of them present in the view log** |
| Join | 92.9% of views overlap at least one airing; 94.1% of viewing seconds are matched |
| Unmatched | 5 channels of the log (113,579 views) never appear in any schedule |

`tests/check_data.py` reproduces every count printed in the exam statement: rows per file, 115,613 labelled
rows in weeks 11-20, 5,848 / 6,364 / 6,637 distinct ids, median view 396 s, max 342,145 s. One detail: the
statement's distinct-id counts include id 0. The real programme counts are 5,847 / 6,363 / 6,636.

**Join rule** (`src/join.py`): an airing and a view match when they are on the same channel
(`vsetv_id = channel_id`) and their intervals overlap by at least one second. Every such pair is stored,
labelled or not, with no threshold yet. Before the join, the two training schedules are merged. One row of
zero duration is dropped. Five airings that run past the next airing on the same channel are cut at that
start (42,600 s in total). After this, no second of a view can be credited to two airings: this is checked
on all 7,752,424 pairs. On five random synthetic schedules, the vectorised join gives exactly the same
pairs as a brute-force join.

## 2. The log and the schedule share one clock

* **DST signature.** For 117 views, `duraton ≠ stop − start`. All 117 differ by exactly **3,600 s**, and
  they are exactly the 117 views that span 2020-03-29 03:00, when Ukraine switched to summer time.
  So `start_time` / `stop_time` are local wall-clock times, and `duraton` is the real elapsed time.
* **Boundary test** (`figures/clock.png`). 4.72% of views stop within 60 s of a programme boundary on
  their channel. Shifting the log by ±1, ±2 or ±3 h drops this to 3.2-3.5%. View starts peak in the first
  two minutes after a labelled programme starts.
* Viewing peaks at 19:00-22:00 (8.5-8.7% of all viewing seconds per hour), which is local prime time.

→ No timezone conversion is applied.

## 3. Durations

**Views** (`figures/durations.png` a): minimum **61 s**, median 396 s, p90 3,911 s, p99 14,079 s, max
342,145 s (95 h). **Not one view is shorter than 61 s**, so the provider has already removed channel
zapping. The distribution is heavily skewed. Views under 10 min are 58.4% of rows but only 8.3% of
viewing seconds. Views over 4 h are 0.96% of rows but 14.8% of seconds, and 52 views exceed 24 h. These
are very likely screens left on.

**Simultaneous views.** 11.9% of views start before the same user's previous view has ended. 66% of
users do this at least once, which suggests several devices on one account. Summing the overlaps would
count such seconds twice: on 42,098 user × airing pairs the sum exceeds the actual union, and on 24,006
it exceeds the airing's own length. So time watched is computed as the **union** of the user's
intervals on that airing.

**Airings.** Labelled airings are longer than unlabelled ones: median 2,700 s against 1,500 s, and
p10/p90 of 600 s / 7,200 s. The distribution is the same in all three windows.

## 4. Labelling: 11% of rows, 17% of airtime, 56% of viewing

| | 11-20 | 21-30 | 31-42 |
|---|---|---|---|
| labelled rows | 10.76% | 11.31% | 11.18% |
| labelled airtime | 15.97% | 16.90% | 16.67% |
| distinct programmes | 5,847 | 6,363 | 6,636 |
| channels with ≥ 1 labelled airing | 177 / 400 | 188 / 403 | 191 / 404 |

Labels follow the **category**, not chance. The share of rows labelled is Развлечения 85.5%,
Фильмы 56.1%, Сериалы 45.5%, Детям 44.0%, Инфо 8.8%, Познавательное 1.6%, Другое 0.1%, Спорт 0.01% and
Музыка 0%. Sport and music are therefore never items.

Viewing is concentrated on labelled airtime. **56.4% of matched viewing seconds land on labelled
airings**, against 16-17% of airtime. The weekly share stays between 51% and 62% (`figures/weekly.png`
b). Per user, the median labelled share is 58% (p10 34%, p90 73%). Only 19 users, 2 of them test users,
never touch labelled airtime.

## 5. Activity over time: the log is not stationary

`figures/weekly.png` (a) shows daily activity. The weekly counts are in `eda.json → weekly`.

* Weeks 11-19: about 4,090 active users and 19,700 viewing hours per day.
* From early May the share of active users declines steadily, from about 4,000 to 2,500 per day by
  mid-June. This could be seasonal (lockdown easing) or a recording effect. The data cannot tell.
* **17 Jun - 2 Jul: the log thins out.** All 16 days are below half the median daily user count. From
  17 Jun to 1 Jul the averages are 1,755 users and 5,881 h per day. **3 Jul is missing completely.** On 5 Jul the count returns to about 3,950
  users per day within three days. A real audience does not come back that abruptly, which points to a
  recording gap.
* **The recording of long sessions changes over time.** In weeks 11-18, 0.03% of views exceed 4 h and
  p99.9 is about 3 h. In weeks 19-24 it is 1.4-2.9%, with p99.9 at 9-10 h. In weeks 25-26 it falls to
  0.1-0.7%, and in weeks 27-30 it is back at 1.8-3.3%. All of 9 Mar - 3 May and 17 Jun - 2 Jul have
  under 0.5% of views longer than 4 h.

Consequences: window B contains three under-recorded weeks (25-27). Seconds-based signals are not
comparable across regimes, because the long "left-on" sessions exist only in some weeks. Weeks 28-30 look
clean.

Test users are heavier and more regular viewers than the others. They have a median of 536 viewing hours
against 340, a median of 20 active weeks out of 20 against 18, and 330 distinct programmes against 216.
1,216 of them are active in week 30. Only 16 were last active before week 26.

## 6. Programme overlap between windows

| | 11-20 & 21-30 | 21-30 & 31-42 | 11-20 & 31-42 | all three |
|---|---|---|---|---|
| shared programme ids | 3,976 | 4,389 | 4,070 | 3,260 |
| Jaccard | 0.483 | 0.510 | 0.484 | — |

By id count, consecutive windows share only about half of their programmes. By volume, though, the
schedule repeats:

* Of the 6,636 ids in C, 1,437 (21.7%) never aired in A or B. Yet **95.6% of C's labelled airings and
  92.7% of its labelled airtime** belong to programmes that did air before. 70.9% of C's labelled airings
  belong to programmes that someone in the log has watched.
* **88.7% of window-B viewing seconds** are on programmes that also air in C. The same figure for A → B
  is 87.0%.
* In B, 18.3% of programme ids air only once (mostly films), and 9.2% air 20 or more times (news,
  series, children's shows). That 9.2% receives **51.3% of all viewing seconds**.

## 7. What counts as "watched"

Unit of analysis: user × labelled airing, using the union of the user's views of that airing. There
are 3,014,129 such pairs. Figures: `durations.png` (b) and `coverage.png`.

* **Absolute seconds.** On a log scale the distribution has a small shoulder below 60 s, followed by
  a broad plateau from 60 s to about 2 h. The pairs below 60 s are **2.5% of pairs but only 0.05% of
  seconds**. They can only be edge fragments of a view that straddles a programme boundary, because no
  view is shorter than 61 s. Above 60 s there is **no valley**, so a higher absolute cut would be
  arbitrary:

  | cut | 60 s | 120 s | 300 s | 600 s | 900 s | 1,800 s |
  |---|---|---|---|---|---|---|
  | pairs kept | 97.5% | 84.9% | 69.0% | 55.3% | 45.1% | 27.7% |
  | seconds kept | 99.95% | 99.2% | 97.1% | 93.2% | 88.4% | 73.3% |

* **Share of the airing.** The coverage distribution is U-shaped. 27.1% of pairs cover less than 5% of
  the airing (sampling), 15.8% cover at least 95% (watched through), and the bins in between are thin.
  This U-shape is a natural place for a relative cut. "≥ 80% within one view" keeps 18.7% of pairs and
  37.1% of seconds. However, **24.0% of the airings that qualify this way come from views longer than
  4 h**, which are probably screens left on (see §5).

**Proposal:**
1. Use ≥ 60 s per user × airing as the floor that defines an interaction. It matches the log's own
   61-s floor and removes only boundary fragments.
2. Keep two strengths per user × programme: watched seconds, and the number of airings watched ≥ 80%
   in one view.
3. The choice between them is the subject of §8.

## 8. What "most preferred" means

The competition text reads: *"predict the 5 most preferred tv_show_ids … from the most preferred (#1) to
the least preferred (#5). The preferences were calculated for the users using the subsequent 12 weeks
(weeks 31-42) of historical views."* The exact aggregation is not stated.

One external source, a search-engine summary of the second-place write-up ("Two-layered recommender
system methodology: a prize-winning solution", Towards Data Science), states the organisers' rule:
*"a TV-program was considered watched if the user watched over 80% of it and did not change the channel"*.
The same summary describes a baseline that ranks programmes by how often they were watched. **I could not
open the article itself** (HTTP 403), so this claim is **not verified**.

Three candidate definitions, each computed on window B:

* **secs**: total seconds watched, union per airing, ≥ 60 s;
* **n60**: number of airings watched for ≥ 60 s;
* **n80**: number of airings that a single view covered by ≥ 80%. This is the external claim.

The definitions disagree. A perfect oracle of one definition, scored by MAP@5 against another
definition's top-5 sets in B:

| oracle ↓ / truth → | secs | n60 | n80 |
|---|---|---|---|
| secs | 1 | 0.552 | 0.497 |
| n60 | 0.536 | 1 | 0.573 |
| n80 | 0.455 | 0.542 | 1 |

So the definition is worth roughly half the metric, and it cannot be settled from the training data.
Further points:

* Under **n80**, 4,623 users have at least one item in B and 3,923 have five or more. Among test users
  the counts are 1,236 and 1,124. Under secs and n60, 1,244 of 1,260 test users have five or more.
* Test users are not recognisably those with many n80 programmes. As a predictor of being a test user,
  the n80 count in B has an AUC of 0.593, against 0.667 for total viewing hours. They look like a sample
  of heavier viewers, not a group selected by an n80 count.

### 8b. Kaggle probe: the target behaves like n80

`experiments/target_probe.py` runs one instrument, history replay (`src/replay.py`). It ranks each
user's own programmes from weeks 11-30 by a signal and pads the list with the most popular programmes
under the same signal. Candidates are the labelled programmes of the 31-42 schedule. Only the ranking
signal changes. On the A → B proxy, for test users, each truth definition is best predicted by its own
signal:

| ranking signal ↓ / truth → | secs | n60 | n80 |
|---|---|---|---|
| secs | **0.302** | 0.338 | 0.258 |
| n60 | 0.291 | **0.437** | 0.340 |
| n80 | 0.238 | 0.360 | **0.353** |

The predicted difference secs − n80 is +0.064 if the truth is secs, −0.022 if it is n60, and −0.095 if
it is n80. The three hypotheses are therefore at least 0.04 apart, so two submissions were enough. Both
were submitted on 2026-10-22 and are logged in `submissions/log.jsonl` and `experiments/kaggle_scores.json`:

| file | md5 | public | private |
|---|---|---|---|
| `submissions/probe_secs.csv` | f287038c… | 0.27937 | 0.24944 |
| `submissions/probe_n80.csv` | 5ae14057… | **0.39596** | **0.38767** |

The observed difference is **−0.117**. This is the n80 pattern, and slightly stronger than predicted.
The levels agree as well. If the truth were secs, the n80 file would be expected near 0.24, not 0.40. If
it were n60, the secs file would be expected near 0.34, not 0.28.

**Conclusion:** the evaluation behaves like "top-5 programmes by the number of airings a user watched
≥ 80% of in one view". This is consistent with the external claim. The probe compared only three
candidates, so a definition that is close to n80 but not identical (other cut-off, another tie-break)
is not excluded. From here on, validation truth = n80 top-5.

## 9. How much does the next window repeat the previous one? (A → B)

Here, truth = a user's top-5 in B under each definition, and history = window A. These are diagnostics,
not models. "Personal" means the user's own A top-5 under the same definition, restricted to programmes
that air in B. "Popularity" means the five programmes in A with the most users, restricted the same way.

| | secs | n60 | n80 |
|---|---|---|---|
| users with a non-empty truth (test users) | 4,806 (1,246) | 4,806 (1,246) | 4,623 (1,236) |
| truth items the user watched in A (≥ 60 s) | 60.5% | 74.5% | 66.6% |
| truth items that were in the user's A top-5 | 29.7% | 43.0% | 36.4% |
| truth items that never aired in A | 10.0% | 4.3% | 6.4% |
| truth items that still air in C | 94.2% | 97.9% | 96.8% |
| MAP@5, personal A top-5: all / test users | 0.243 / 0.301 | 0.370 / 0.435 | 0.305 / 0.350 |
| MAP@5, popularity A top-5: all / test users | 0.154 / 0.179 | 0.266 / 0.306 | 0.223 / 0.248 |

The popular programmes are news. In A the top five by user count are ТСН (2400480), Факты (240081),
Сегодня (2400508), Мир наизнанку (6500479) and Вікна-Новини (2400467), and under n80, Чистоnews
(200344) moves into the list. The same `tv_show_id` airs on several channels: ТСН has 3,888 airings
across all three files.

Reading:
* **The future largely repeats the past.** 60-75% of a user's next-window favourites are already in
  their history, so the candidate pool must allow already-watched programmes. Forbidding repeats would
  throw away most of the hits.
* **The user's own history beats popularity under every definition.** The gap is 0.08-0.10 MAP@5. Test
  users are more predictable than average, by +0.04 to +0.07 MAP@5 for personal history.
* **Unrecoverable from history:** about 25-40% of truth items. These are 4-10% that never aired in A,
  plus programmes that aired but the user did not watch. This is where a model has to beat replay.
* Using all ten weeks of history works better than using only the most recent five. Under secs, weeks
  16-20 alone give MAP@5 0.221 against 0.243. This is confounded by the drop in recording in week 20,
  see §5.

## 10. Test users and cold start

* All 1,260 test users appear in the log. **2 have no labelled interaction ≥ 60 s in weeks 11-30**,
  which is true cold start. One more has fewer than five programmes.
* 14 have nothing in B, and 24 have no n80 airing in B. These users depend on A or on a fallback.

## 11. What this suggests for stage 2 (to agree before modelling)

1. **Interactions:** same channel plus overlap. Time per user × airing is the union of their views.
   Keep pairs ≥ 60 s. `tv_show_id = 0` is never an item. It is 44% of viewing, so it can serve as
   context (for example channel affinity), not as a target.
2. **Candidate pool:** labelled ids in the 31-42 schedule (6,636). **Repeats allowed**, identical across
   methods (§9).
3. **Validation:** fit on A, evaluate on B. This is the 10 → 10-week analogue of B → C. It should be
   checked for robustness on clean weeks only, for example history 11-24 → truth 28-30, because weeks
   25-27 are under-recorded.
4. **Target definition:** n80, supported by the Kaggle probe (§8b). Validation truth is the n80 top-5,
   and secs and n60 remain available as features.
5. **Cold start (2 test users):** popularity fallback, identical across methods.
6. **Long sessions:** consider capping or discounting views over 4 h. They supply 24% of n80 airings and
   are recorded only in some weeks.

## 12. What stage 2 did with this

Adopted: proposals 1, 2, 4 and 5, and the main split of proposal 3 (fit on 11-20, evaluate on 21-30).
Not done: the robustness check on clean weeks only (proposal 3) and capping or discounting long sessions
(proposal 6). The methods, the validation results and the final submission are in `README.md` and
`report/kosychev_midterm_report.pdf`. The numbers behind them are in
`experiments/validation.json`, `hybrid.json`, `final.json` and `kaggle_scores.json`.
