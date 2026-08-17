# Experiment Log

Running record of investigations, decisions and dead ends. Kept as work happens
rather than reconstructed afterwards, so the final report can be assembled from
evidence instead of memory. Rough and chronological by design — it will be
revised and restructured before submission.

Every number here was measured on this project's own data. Failed hypotheses are
kept deliberately: they are the reason the surviving conclusions are trustworthy.

---

## Phase 1 — Audit (2026-08-12)

Reviewed the repository against the project brief as a strict evaluator would.
Verified claims by execution and live queries rather than reading code.

**Verified healthy:** no secrets in git history, `.gitignore` correct, all modules
import, `pyflakes` clean, EDA notebook has no errored cells, EPA breakpoints match
the 2024-revised PM2.5 table, truncation rules and µg/m³→ppb conversion correct.

### Finding 1 (BLOCKER) — the automation was inert

The live pipeline stamped rows with OpenWeather's `dt` (observation calculation
time, e.g. `11:19:16`). The backfill stamped hour-aligned timestamps from the
hourly pollution endpoint. Labels are built by matching a row to the row at
exactly `t + 24/48/72h`, so an unaligned timestamp can never match anything.

```
total rows      : 2100
hour-aligned    : 2065   <- all from backfill
NOT hour-aligned:   35   <- ALL live rows

horizon 24h: labeled rows=1945 | from LIVE data=0
horizon 48h: labeled rows=1921 | from LIVE data=0
horizon 72h: labeled rows=1897 | from LIVE data=0
```

Two weeks of hourly automation had contributed **zero** training examples. Every
daily retrain registered byte-identical metrics while reporting success.

**Fixed** (`ee736c3`) by flooring the timestamp to the hour. This also lets the
`(city_name, timestamp)` primary key deduplicate two runs in the same hour —
there were already rows 42 seconds apart from a double run.

### Finding 2 (CRITICAL) — train/test leakage across the horizon

`time_based_split` splits chronologically but does not drop training rows whose
*label* falls inside the test window. A row at `T_split − 1h` carries a label from
`T_split + 71h`.

| Split (72h horizon) | R² |
|---|---|
| As implemented | 0.326 |
| With 72h purge/embargo | **0.237** |

~27% of the reported skill was leakage. All experiments after this point use a
purged split. *(Fix to the repo still pending.)*

### Finding 3 (CRITICAL) — skill was calendar memorization

With leakage purged and `day`/`month` removed, R² at 72h fell to **−0.081** —
worse than predicting the mean. With only 3.4 months of data spanning months 4–7,
the model was memorizing "late July looks like this" rather than learning
pollution dynamics. It was also being served in month 8, a value it had never
seen, which a Random Forest cannot extrapolate to.

### Other findings

- **HIGH** — `aqi_change_rate` skew: a 1-hour delta in training vs "since the last
  successful run" in production (std 8.12 → 13.10). Fixed in `d628ca6`.
- **HIGH** — blanket `except RestAPIError` silently converted auth failures into
  "no previous reading", writing a wrong change rate with no error. Fixed in `d628ca6`.
- **HIGH** — EPA breakpoints are defined over averaging windows (PM 24h, O3/CO 8h,
  SO2/NO2 1h) but are applied to instantaneous hourly readings. *Open: fix or disclose.*
- **NOT IMPLEMENTED** — zero tests. Addressed in `afbf11a` (30 tests).
- Workflows had no `timeout-minutes` and no pip caching. Fixed in `ec1fe5f`.

**Audit score: 50/100, BORDERLINE.**

---

## Phase 2 — Data starvation (2026-08-12)

### Diagnosis

A learning curve on the existing 2,065 rows was still climbing steeply at 100%
of the data, which is the signature of a model limited by sample count rather
than by algorithm choice:

| Train rows | R² (24h) |
|---|---|
| 383 | −0.328 |
| 766 | −0.105 |
| 1,149 | +0.010 |
| 1,532 | **+0.340** |

The original backfill fetched 90 days in a single request. Probing the API
directly showed OpenWeather's pollution archive serves data back to
**2020-11-27** — roughly 24× more history was available and unused.

### Action

Rewrote the backfill to chunk by calendar month, insert in batches with retry,
and cast explicitly to the feature group's locked schema (`b2d33ba`, `ebc887d`).

### Two failures found along the way

1. **`compute_aqi` crashed on the wider data.** OpenWeather's pollution figures
   come from a chemical transport model that occasionally emits small *negative*
   concentrations. EPA breakpoints start at 0, so a negative reading matched no
   range and `_sub_index` returned `None`, poisoning `max()`. Ninety days
   contained none of these; 5.7 years hit them immediately. Fixed in `afbf11a`
   by clamping to the bottom breakpoint, plus 30 regression tests.

2. **The bulk insert died on the local connection** — `KafkaError _MSG_TIMED_OUT`
   during `producer.flush()` on batch 3 of 10. The retry never engaged because it
   only caught `requests.RequestException`. Fixed by broadening the retry and
   moving bulk loads to a GitHub Actions runner (`ec1fe5f`).

3. **Materialization had been frozen since 2026-08-09.** Hopsworks writes land in
   Kafka and a background job files them into the queryable offline store. That
   job was stuck `INITIALIZING` for three days, so the 49,080 inserted rows were
   invisible to `fg.read()`. Killed the stuck execution and re-ran it; completed
   in about a minute.

   *Both this and Finding 1 share a trait worth calling out in the report:
   everything reported success. The pipeline said "inserted", GitHub Actions went
   green, the daily training ran — and no data was arriving.*

### Result

```
ROWS: 49,115   span 2020-11-27 -> 2026-08-12
years: 2020, 2021, 2022, 2023, 2024, 2025, 2026
duplicate timestamps: 0     nulls: 0
```

2,065 → 49,115 rows (24×). 24h-ahead R² rose from **0.34 → 0.66**.

Two earlier conclusions reversed once there was enough data to test them properly:

- **Lag/rolling features now help** (0.645 → 0.660 at 24h; 0.414 → 0.468 at 72h).
  They *hurt* at 2k rows — too many features for too few samples.
- **`day`/`month` now hurt** (0.628 vs 0.645). The memorization crutch is gone.
- **72h went from broken to working**: −0.081 → +0.468.

---

## Phase 3 — The persistence problem (2026-08-17)

Adding a persistence baseline ("AQI in H hours = AQI now") to the comparison
changed the picture completely.

| Horizon | Best model | Persistence |
|---|---|---|
| 24h | 0.660 | **0.805** |
| 48h | 0.487 | **0.705** |
| 72h | 0.468 | **0.631** |

The model loses to the naive baseline at every horizon. Note this reversed the
earlier result (persistence scored −0.742 on the old data) purely because the
test set changed: it used to be three weeks of stable summer and is now 1.1 years
including winter smog season, where AQI swings from 50 to 500. Old MAE (~19) and
new MAE (~48) are not comparable — it is a harder problem now, measured honestly.

### Root cause — regime shift

```
TRAIN 2020-11-27 -> 2025-06-19   mean AQI 261.9   std 142.4
TEST  2025-06-19 -> 2026-08-12   mean AQI 169.6   std 108.8

Yearly mean AQI: 2020:387  2021:270  2022:270  2023:269
                 2024:245  2025:198  2026:162
```

**Lahore's air quality has improved ~58% since 2020.** The model trains on a
dirtier era and systematically over-predicts. Persistence adapts for free because
it starts from today's actual reading. This also explains why Ridge beats tree
models here — linear models extrapolate a trend; trees only reproduce values seen
in training.

It also explains an earlier result flagged as "bizarre": in the learning curve,
the most recent 10% of rows (0.546) *beat* all 35k rows (0.382). Old data from
the high-pollution era actively poisons the model.

### Hypotheses tested against persistence

| # | Hypothesis | Result |
|---|---|---|
| 1 | More data | ✅ 0.34 → 0.66, but still below persistence |
| 2 | Delta target (predict change, add to current) | ❌ No effect: 0.703 vs 0.697 at 24h |
| 3 | Per-horizon model choice | ✅ **Ridge > RF at every horizon** — see below |
| 4 | Trailing training window | ~ Marginal: best 0.736 (24mo+Ridge) vs 0.720 all-history |
| 5 | Forecast weather as features | ❌ **Worse** at every horizon (24h 0.718→0.695; 72h 0.519→0.368) |

### Model bake-off, per horizon (full features, purged split)

| Horizon | Ridge | RandomForest | HistGradientBoosting | Persistence |
|---|---|---|---|---|
| 24h | **0.700** | 0.438 | 0.641 | 0.807 |
| 48h | **0.564** | 0.258 | 0.417 | 0.689 |
| 72h | **0.500** | 0.184 | 0.380 | 0.604 |

**Random Forest — the model currently in production — is the worst of the three
at every horizon.** It was the right choice for 90 days of summer data and became
the wrong choice at 49k rows spanning a trend. The original bake-off was run only
at 72h and its verdict was applied to 24h and 48h by assumption; re-running it
per horizon overturned that.

### Trailing-window results (delta target)

| Window | 24h Ridge | 24h HistGB | 72h Ridge | 72h HistGB |
|---|---|---|---|---|
| 3 months | 0.036 | 0.326 | −0.128 | −0.381 |
| 6 months | 0.217 | 0.479 | −0.301 | 0.471 |
| 12 months | 0.676 | 0.421 | 0.430 | 0.484 |
| 24 months | **0.736** | 0.624 | 0.526 | 0.459 |
| 48 months | 0.721 | 0.659 | **0.562** | 0.498 |
| all history | 0.720 | 0.703 | 0.556 | 0.477 |

Short windows are disastrous — years of data are needed to learn the dynamics;
the goal is to weight recent years more, not to discard old ones entirely.

### The anomaly driving the next experiment

Under the delta framing, a model predicting "no change" would score *exactly*
persistence. Every model scores below it. A flexible model with 40 features
cannot learn to output zero — because the deltas it learned come from an era with
larger swings (std 142 vs 108), so it confidently predicts change into a calmer
regime.

**Currently running:** shrink the predicted delta by a factor `w` chosen on a
validation slice (`prediction = AQI(t) + w · predicted_delta`; `w=0` is exactly
persistence). This measures directly whether the model adds *any* information
over persistence, rather than guessing at another fix.

---

## Open items

- Apply the purge/embargo fix to `time_based_split` in the repo
- Replace Random Forest with the per-horizon winner; retrain and re-register
- Add the persistence baseline to `train.py` and persist comparison results
- Pin `requirements.txt`; seed the TensorFlow model
- Fix or explicitly disclose the EPA averaging-window approximation
- Rewrite README (still says "Project scaffolding in progress")
- Assemble and structure the final report from this log
