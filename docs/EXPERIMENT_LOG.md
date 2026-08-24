# Experiment Log

Running record of investigations, decisions and dead ends. Kept as work happens
rather than reconstructed afterwards, so the final report can be assembled from
evidence instead of memory. Rough and chronological by design — it will be
revised and restructured before submission.

Every number here was measured on this project's own data. Failed hypotheses are
kept deliberately: they are the reason the surviving conclusions are trustworthy.

---

## Phase 0 — Setting up and building the system (2026-07-21 → 2026-08-09)

### 0.1 Accounts, keys and local environment

Two external services, both on free tiers:

- **OpenWeather** — current weather, current air pollution, and *historical* air
  pollution. Note the asymmetry that shaped the backfill design: OpenWeather's
  **air-pollution history is free**, but its **weather history requires a paid
  plan**. Historical weather therefore comes from **Open-Meteo's archive API**,
  which is free and needs no key.
- **Hopsworks** (project `AQI_Predictor4478`) — feature store *and* model
  registry in one service, which is why it was chosen over stitching two tools
  together.

Local setup:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Secrets live in `.env` (git-ignored), with `.env.example` committed as the
template so the required variables are documented without leaking values:

```
OPENWEATHER_API_KEY=      HOPSWORKS_API_KEY=       CITY_NAME=Lahore
HOPSWORKS_PROJECT_NAME=   CITY_LAT=31.5497         CITY_LON=74.3436
```

The same six variables were added as **GitHub Actions Secrets** via the `gh`
CLI, so the scheduled workflows could authenticate without any secret ever
entering the repository. Verified afterwards: a full history scan found no
committed keys, and `.gitignore` correctly excludes `.env`, `*.pem` and `*.pkl`.

**City:** Lahore, Pakistan — chosen for genuinely severe and variable air
quality, which makes the forecasting problem meaningful rather than a flat line.

### 0.2 Windows-specific obstacles (worth recording — they cost real time)

| Problem | Resolution |
|---|---|
| pip 22.3 silently corrupted large wheel downloads | Upgrade pip first, before anything else |
| `hopsworks` → `pyjks` → `twofish` has no Windows wheel and needs a 64-bit C compiler | Install `hopsworks`/`pyjks` with `--no-deps`; `twofish` proved unnecessary for the JKS paths actually used |
| Hopsworks client hardcodes `/tmp` in places | Pass `cert_folder=` for the main client; create `C:\tmp` for the Kafka cert path, which has no override |
| Large uploads from this machine fail | Confirmed by testing: a 30-packet ping showed 0% loss, but a raw 10 MB upload to Cloudflare failed. Small requests fine, sustained transfers not. **Not a code bug** — route bulk operations through GitHub Actions |

That last one recurs throughout the project and drove several later design
decisions, including running the multi-year backfill on a GitHub runner.

### 0.3 Feature pipeline and the EPA AQI calculation (`cd20179`, `f899b19`)

OpenWeather returns raw pollutant **concentrations**, not an AQI, so the US EPA
AQI is computed from scratch in `feature_pipeline/aqi.py`:

1. Convert µg/m³ → ppb/ppm for the gases (`ppb = µg/m³ × 24.45 / molecular weight`
   at 25 °C, 1 atm; CO additionally ppb → ppm).
2. **Truncate** each concentration to the EPA's specified precision before
   matching. This is easy to miss and it matters: the breakpoint tables have
   gaps between buckets (PM10 runs 0–54 then 55–154), so an untruncated 54.5
   matches *nothing* and returns no sub-index.
3. Look up the piecewise-linear sub-index per pollutant.
4. **Overall AQI = the worst sub-index**, and the pollutant producing it is
   recorded as `dominant_pollutant`.

The tables use the **2024-revised** PM2.5 breakpoints (Good ends at 9.0, not the
obsolete 12.0).

*Known approximation, carried forward deliberately:* EPA breakpoints are defined
over averaging windows — 24h for PM, 8h for O3/CO, 1h for SO2/NO2 — but are
applied here to instantaneous hourly readings. This makes the value an hourly
proxy rather than official EPA AQI. Listed in open items to fix or disclose.

### 0.4 Feature group v1 → v2 (`5f341d3`)

The first feature group stored raw weather and pollutant values plus the computed
AQI. Re-reading the brief showed it explicitly requires **time-based features
(hour, day, month)** and **derived features such as AQI change rate**, so a v2 was
created adding `hour`, `day`, `month` and `aqi_change_rate`.

**v1 was left untouched rather than edited.** Feature groups are versioned
precisely so history stays reproducible; rewriting one in place would invalidate
anything trained against it.

Feature group design: `primary_key=["city_name", "timestamp"]`, `event_time="timestamp"`,
HUDI format. The composite key makes inserts idempotent — re-running a backfill
upserts rather than duplicating, which later made recovery from a failed bulk load
trivial.

### 0.5 Historical backfill, first attempt (`6f18c13`)

Merged two sources on the hour: OpenWeather air-pollution history + Open-Meteo
weather archive. **90 days, ~2,065 rows.** This number is the origin of the data
starvation diagnosed much later in Phase 2 — a single request was issued for the
whole range, and nobody checked whether more history was available. It was: 5.7
years.

A dtype bug surfaced here and set a pattern: `pressure` arrived as a float, was
stored as an int, and pandas inferred the column type from whatever the batch
happened to contain. The same class of bug reappeared twice more (Phase 0.11 and
Phase 2), each time because pandas infers dtypes per-batch while the feature
group schema is fixed.

### 0.6 Exploratory data analysis (`b87a2a7`, `notebooks/01_eda.ipynb`)

Findings that shaped later modelling decisions:

- **No nulls, no duplicate timestamps** — but ~5% of hours missing (107 of 2,172).
  This directly motivated matching labels by *timestamp* rather than row position.
- **Mean AQI ≈ 125** across the window — Lahore's air is never "clean" here.
- **PM2.5 correlates 0.90 with AQI** — it is almost always the dominant pollutant.
- **Weather features correlate weakly with AQI (≤0.14)** — an early warning that
  the physical features might carry less signal than hoped, which Phase 3
  eventually confirmed.
- Clear daily cycle in AQI by hour.

### 0.7 Target construction and the split (`3a9951a`)

The single most important modelling decision. To forecast 72h ahead, each row
needs the AQI value from exactly 72 hours later:

```python
future_timestamps = df["timestamp"] + pd.Timedelta(hours=horizon_hours)
df[TARGET_COLUMN] = aqi_by_time.reindex(future_timestamps).values
```

Matched **by timestamp, not by shifting 72 row positions**. With ~5% of hours
missing, a positional shift would silently pair a row with the wrong future value
— a bug that produces no error and quietly corrupts every label.

`time_based_split()` splits **chronologically**, never randomly. Shuffling
time-series data lets the model evaluate on rows whose near-identical neighbours
it trained on, which inflates scores meaninglessly.

*(A subtler leak in this same function — training rows whose labels fall inside
the test window — went unnoticed until the Phase 1 audit.)*

### 0.8 Four models compared (`9e28cf3`, `2b5bce1`, `efc4702`, `f76bc6e`)

The brief asks for a variety of approaches, from statistical to deep learning.
All four were wired through one shared interface — `(X_train, y_train, X_test) → y_pred`
— so they plug into a single comparison loop despite being entirely different
libraries.

| Model | Library | R² @72h | Note |
|---|---|---|---|
| Ridge | scikit-learn | −0.09 | Linear; needs feature scaling |
| **Random Forest** | scikit-learn | **0.33** | **Winner at the time** |
| Neural Net (MLP 32/16) | TensorFlow/Keras | −1.16 | Worst — far too little data |
| ARIMA (2,1,2) | statsmodels | −0.27 | Ignores all features; uses only AQI's own past |

**Interpretation at the time:** Random Forest's bounded predictions handled the
train/test distribution shift more gracefully than Ridge or the neural net, both
of which extrapolated badly. Three of four models scored *below zero* — worse than
predicting the mean.

This "fancier is not automatically better" result was treated as a genuine finding
rather than something to hide. *(Phase 3 later reversed the conclusion entirely
once there was 24× more data — see below.)*

ARIMA needed a different data shape from the others: a single evenly-spaced
series rather than independent rows, so `load_raw_aqi_series()` reindexes to a
regular hourly grid and interpolates small gaps.

### 0.9 Explainability, and a methodological catch (`806fe63`)

SHAP (`TreeExplainer`) on the Random Forest produced
`reports/shap_feature_importance.png`. It showed **`day` (day-of-month) dominating
feature importance**, which looked spurious.

Rather than assume, this was tested by ablation: removing `day` collapsed R² from
0.33 to −0.03. At the time this was read as "`day` is genuinely important."

**Phase 3 revised that reading.** The feature was load-bearing precisely *because*
the model had learned little else — with only 3.4 months spanning months 4–7, it
was memorising "late July looks like this" rather than learning pollution
dynamics. Same measurement, correct interpretation only visible with more data.

### 0.10 Diagnostics that justified daily retraining

- **Persistence baseline: R² = −0.74** vs Random Forest 0.33 — at the time, strong
  evidence the model added real value. *(This flipped completely in Phase 3 when
  the test window widened from three summer weeks to a full year.)*
- **Systematic over-prediction bias of ~8.6 AQI points.**
- **Error grows with distance from the training cutoff**: RMSE 20.5 over the first
  half of the test period vs 28.3 over the second.

That last one is concrete evidence for *why* daily retraining matters — not just a
checkbox from the brief. It also foreshadowed the regime-shift finding in Phase 3.

### 0.11 Multi-horizon models and the Model Registry (`806fe63`)

A single 72h number is a poor "3-day forecast", so `add_target()` and `register.py`
were generalised to take `horizon_hours`, and `register_forecast_models.py` loops
over `[24, 48, 72]`, registering three independent models.

| Model | R² (at the time) |
|---|---|
| `aqi_random_forest_24h` | 0.47 |
| `aqi_random_forest_48h` | 0.27 |
| `aqi_random_forest_72h` | 0.33 |

Near-term is easier than far-term, as expected.

Registration trains **twice** on purpose: an "honest" model on the training split
only, whose held-out metrics are what get registered, and a "final" model refit on
all available data, which is the artifact actually deployed.

**Registry gotcha, found by reading the `hsml` source:** `mr.get_model(name)`
defaults to **version 1**, not the latest. Left unfixed, the dashboard would have
pinned itself to the first model forever and silently ignored every daily retrain.
`load_latest_model()` therefore fetches all versions and takes the maximum.

### 0.12 Streamlit dashboard (`a22c317`, `b66c99f`)

`app/app.py` loads the latest registered model per horizon plus the newest feature
row, and renders: a hero card with current AQI, a day-by-day forecast chart
(24/48/72h), a hazard alert banner, a SHAP expander and a model-version expander.

Design decision worth noting: the dashboard uses the **official EPA/AirNow AQI
colours** rather than a neutral palette. AQI colour-coding is a globally recognised
convention like traffic lights; inventing a different scheme would be actively
worse for users.

Alerts key off `aqi_category()` in `feature_pipeline/aqi.py`, so the alert
thresholds and the prediction scale are guaranteed to agree — they read from one
table.

### 0.13 Automation (`.github/workflows/`)

- `feature_pipeline.yml` — hourly (`0 * * * *`)
- `training_pipeline.yml` — daily (`0 2 * * *`)

Both confirmed genuinely running, not merely committed: a scheduled hourly run
fired unattended and added real rows, and manual `workflow_dispatch` runs of both
succeeded. The three multi-horizon models were in fact registered *from* a GitHub
runner, precisely because this machine's uploads were unreliable.

### 0.14 Bugs fixed during the build

| Bug | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'app.load_model'` (`8f91aa5`) | `streamlit run app/app.py` puts the script's own folder on `sys.path`, not the repo root | Explicit `sys.path.insert(0, repo_root)` — the identical fix the EDA notebook needed for the same reason under nbconvert |
| App hung 30+ minutes on load (`8c406a2`, `b66c99f`) | `hopsworks.login()` has no timeout and relies on the OS socket default | Run it in a background thread with a client-side timeout and retry |
| …and the first version of that fix silently didn't work | `with ThreadPoolExecutor() as ...` calls `shutdown(wait=True)` on exit, blocking until the hung thread finishes — defeating the timeout that had just fired | `shutdown(wait=False)`; stop waiting and retry |
| ~40% of hourly runs failing (`e8a5477`) | Single-row inserts: when OpenWeather returned a whole number (`"no": 0`), pandas typed the column `int64`, clashing with the `double` schema locked in by the multi-row backfill | Explicit `float()` on every double column — third instance of the pandas-dtype-inference class of bug |
| Transient `RemoteDisconnected` in a workflow (`933dbc1`, PR #1) | GitHub runner ↔ Hopsworks network blip, not a config error | Retry with backoff around `fg.insert()`. Fixed using **GitHub Copilot's cloud agent** — this repo has been touched by two different AI tools |

**State at the end of Phase 0:** every numbered requirement in the brief was
implemented and the system ran end to end. That appearance of completeness is
exactly what the Phase 1 audit set out to test.

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

## Phase 4 — Matching the evaluation to the deployment (2026-08-17 → 18)

### The evaluation was measuring a system that does not exist

Every experiment to this point trained once on 2020–2025 and then predicted up
to **14 months** into a frozen future. Production does nothing of the sort: the
training pipeline retrains **daily** and the model only ever forecasts 24–72h
ahead. It always has last week's data.

Against a pollution level that has fallen ~58% since 2020, that setup was mostly
scoring the model on its inability to predict a multi-year trend — a task it
never faces. The tell came from the shrinkage experiment, where the blend weight
chosen on validation (`w* = 0.90–1.00`, "trust the model") was contradicted by
the test slice. Validation and test disagreeing is the signature of
non-stationarity, not of a bad model.

### Walk-forward evaluation

Retrain at successive monthly origins, score only the following month, roll
forward. 12 retrains, ~7,578 predictions per horizon, persistence scored on
identical rows.

| Horizon | Persistence | HistGB | RandomForest | Ridge |
|---|---|---|---|---|
| 24h | **0.814** | 0.780 | 0.759 | 0.735 |
| 48h | **0.709** | 0.643 | 0.598 | 0.605 |
| 72h | **0.628** | 0.613 | 0.579 | 0.574 |

Most of the gap closed; at 72h the model came within 0.015. Persistence still won
everywhere.

**The model ranking reversed.** Under the single frozen split, Ridge beat the
tree models at every horizon. Under walk-forward, Ridge is the *worst* and
gradient boosting wins everywhere. Ridge only looked good because a frozen model
cannot track a falling trend and linear models extrapolate; once retraining is
frequent, that advantage disappears and flexible models win on their merits.

*This is the single most useful methodological finding in the project: changing
the evaluation changed which model wins. An evaluation that does not match
deployment selects the wrong model.*

### Blending with persistence — what finally worked

Persistence and the model are both decent and make partly different errors, so a
weighted average should beat either. Weight chosen from the **previous month
only**, never the month being scored.

| Horizon | Persistence | Model alone | Blend (adaptive w) | Blend (fixed 50/50) |
|---|---|---|---|---|
| 24h | 0.814 | 0.780 | **0.831** | 0.829 |
| 48h | 0.709 | 0.643 | **0.748** | 0.745 |
| 72h | 0.628 | 0.613 | 0.706 | **0.710** |

**Beats persistence at all three horizons.** A fixed 50/50 split performs
essentially identically to the adaptive weight, which is reassuring — the gain is
structural, not the product of weight tuning.

The blend also simplifies algebraically. With `model = now + delta`:

```
blend = (1-w)·now + w·(now + delta) = now + w·delta
```

So deployment needs only a delta-predicting model plus a scalar `w` — no separate
persistence branch at inference.

### Hypotheses tested, in order

| # | Hypothesis | Outcome |
|---|---|---|
| 1 | More data | ✅ 0.34 → 0.66, exposed the real problem |
| 2 | Delta target alone | ❌ No effect under the frozen split |
| 3 | Per-horizon model choice | ✅ Real, but the winner depends on the evaluation |
| 4 | Trailing training window | ~ Marginal (0.736 vs 0.720) |
| 5 | Forecast weather features | ❌ Worse at every horizon |
| 6 | Walk-forward evaluation | ✅ Closed most of the gap |
| 7 | Blend with persistence | ✅ **Beat persistence everywhere** |

Five of seven failed or were marginal. Recording them matters: without the
failures, the two that worked look like lucky guesses rather than the result of
elimination.

### A self-inflicted production incident (2026-08-18)

Hourly runs began failing roughly a quarter of the time with
`FlightUnavailableError: Socket closed`.

**Cause:** the hourly job computed `aqi_change_rate` by calling
`read_features_df()` and filtering in pandas — downloading the whole feature
group to read one value. At 2k rows that was wasteful; after backfilling to 49k
it dropped the Arrow Flight transfer mid-stream. The audit had explicitly flagged
this ("full-table scan per hourly run, O(n) growth, will degrade") and it was not
fixed before making n 24× larger.

**First fix (`af6cc34`) did not work.** Pushing the filter down to Hopsworks
still failed all three retries on the runner, ~2.5 minutes each — filter pushdown
on this HUDI group does not avoid the scan.

**Second fix (`8e2627b`) worked.** The hourly job should not need the feature
store to be *queryable* in order to *write* to it. The previous hour's AQI is now
recomputed from OpenWeather's pollution archive, which the project already
depends on; AQI is deterministic given the concentrations, so the value is
identical. Verified on a runner: success, 1 second, no Hopsworks read.

---

## Phase 5 — The serving path had silently diverged (2026-08-24)

Phases 3–4 rewrote the training side: delta target, lag/rolling features,
HistGradientBoosting, the anchored blend, a new registered model name
(`aqi_forecast_{h}h`). Nothing in `app/` was touched. Checking the dashboard
against the models now being registered found **three separate training/serving
skews, all introduced by one commit that only edited the training side**:

| Skew | Consequence |
|---|---|
| `app.py` loaded `aqi_random_forest_{h}h` | Served a model the daily job no longer retrains — the app would have quietly drifted further out of date every day while looking healthy |
| `load_latest_row()` applied `add_time_features` but not `add_lag_features` | `latest_row[FEATURE_COLUMNS]` cannot resolve `aqi_lag_*` / `aqi_rmean_*` — hard failure on load, the *cheapest* of the three to discover |
| The prediction was rendered as an absolute AQI | The models now output a **change**. A predicted delta of `−4` would have been displayed as "AQI 4 — Good" for a city sitting at 170 |

The middle one crashes and gets noticed. The other two produce plausible
numbers, which is worse. Both belong to the same family as Phase 1's Finding 1
and Phase 2's frozen materialization: **the failure mode of this project is
consistently a green pipeline that is quietly serving nothing, or serving
garbage that looks like a number.**

### Fixes

- `app/load_model.py` returns the blend weight alongside the model, read from
  `blend.json` next to the artifact (registry metrics as fallback). It returns
  `None` rather than defaulting to `w=1.0`: a silent default would deploy the
  unshrunk model, which loses to persistence at every horizon, and it would
  look exactly like a working forecast. The dashboard withholds that horizon
  and says why instead.
- `app/app.py` builds the live row through the *same* `add_time_features` +
  `add_lag_features` functions the training pipeline uses, rather than
  recomputing features by hand — the only way the two stay in step. Forecasts
  now go through `current_aqi + w × predicted_delta`, clamped to 0–500.
- `training_pipeline/explain.py` was explaining a Random Forest predicting
  absolute AQI — a model that no longer runs. Now explains the deployed
  gradient-boosting model on the delta target. It had also been broken since
  Phase 3 by an unnoticed signature change (`load_training_data` returns four
  values, not three) — nothing imports it, so nothing caught it.

### Housekeeping done at the same time

- `requirements.txt` pinned to the versions every number in this log was
  measured against. Unpinned, a minor upgrade on a GitHub runner can change
  model behaviour with no commit to point at.
- TensorFlow model seeded via `tf.keras.utils.set_random_seed` (Python, NumPy
  and TF in one call — seeding `tf.random` alone leaves initialisers free).
  Without it, two runs of the comparison disagreed by more than the gaps it is
  meant to measure.
- The EPA averaging-window approximation is now disclosed in `aqi.py`'s module
  docstring, with its actual consequence stated: applying 24h PM breakpoints to
  hourly readings makes this proxy *more volatile* than official AQI — higher
  during a spike, lower after it. Not fixed, deliberately: every model result
  above was measured against this definition.
- README rewritten. It had claimed "Project scaffolding in progress" since day
  one and described Random Forest as the model.

### Re-reading the brief against the repo

Checked every line of `Project Guidelines & Requirements.pdf` against what
exists, rather than against what the log says was built. Two genuine gaps:

1. **"Use Streamlit/Gradio *and* Flask/FastApi for the web app."** Only Streamlit
   existed. `fastapi` and `uvicorn` had been sitting in `requirements.txt` since
   the first commit with nothing importing them — the requirement had been read,
   provisioned for, and then forgotten.
2. **"A detailed report documenting everything you managed to achieve"** — one of
   four graded Final Submissions, still unwritten. This log is raw material for
   it, not a substitute.

Also worth recording as a process note: the requirement to include time features
`(hour, day, month)` is satisfied by *storing* them in the feature group, while
`day` and `month` are excluded from the model's feature set because they
measurably hurt (Phase 2). That is a defensible reading, but only if the report
states it explicitly — otherwise it looks like a missed requirement.

### FastAPI service, and one serving path (`serving/forecast.py`)

The obvious way to satisfy requirement 11 would be a second copy of the feature
assembly and blend logic behind an HTTP handler. That is precisely the mistake
this phase started by fixing, so the forecast moved into `serving/forecast.py`
instead, and both front ends became clients of it:

- `api/main.py` — `/health`, `/current`, `/forecast`, `/models`, with OpenAPI
  docs generated from typed response models.
- `app/app.py` — presentation only. Reads the API when `AQI_API_URL` is set,
  otherwise calls the same serving functions in-process. One process when
  running locally, two when deployed, identical numbers either way.

Decisions in the API worth keeping:

- **Models load lazily, not at startup.** A startup hook means a transient
  Hopsworks outage stops the service from booting at all, rather than degrading
  one retryable request.
- **`/health` never touches Hopsworks.** It reports cached state. A health check
  that can trigger a registry login becomes the load it is meant to detect.
- **Missing data is 503, not 500.** The service is fine; the data is not there
  yet, and the client should retry.
- **The blend weight is in the response.** A consumer cannot interpret the
  number without knowing how much of it is model and how much is persistence.

### Tests grew from 30 to 57, and now run in CI

The 30 existing tests all covered `aqi.py`. Nothing covered the arithmetic that
turns a predicted delta into a displayed AQI — the exact thing that had been
silently wrong. Added `tests/test_forecast.py` (blend arithmetic, clamping,
withheld horizons, alert selection, schema-drift detection) and
`tests/test_api.py` (routing, status codes, response contract). Neither needs
credentials; the store and registry are substituted.

`.github/workflows/tests.yml` runs pyflakes and pytest on every push. Until now
the only CI was two scheduled jobs whose failures surface hours later.

Two bugs the new tests found immediately:

- Schema drift produced a bare `KeyError: ['pm10', 'humidity']` from inside
  pandas. It now raises a `ForecastUnavailable` naming the missing columns.
- `add_lag_features` called `DataFrame.interpolate` on a frame containing text
  columns. pandas already skips them, but it is deprecated and scheduled to
  **start raising** — a pandas upgrade would have broken training with no code
  change. Restricted to numeric columns; behaviour identical today.

`.github/workflows/explainability.yml` regenerates the SHAP plot from the
deployed model weekly (and on demand) and commits it back. Weekly rather than
daily because it is a binary file and daily retrains would add a blob to the
history for a picture that barely moves.

### Pinning immediately failed CI, and that was the point

The first push with a pinned `requirements.txt` failed the new Tests workflow in
20 seconds:

```
hopsworks 5.0.3   depends on protobuf<5.0.0 and >=4.25.4
tensorflow 2.21.0 depends on protobuf<8.0.0 and >=6.31.1
```

Unsatisfiable. The local venv only holds both because Phase 0.2 installed
`hopsworks --no-deps` to get around the Windows compiler problem, so pip never
enforced that bound here.

The interesting part is what pip had been doing on the runner *before* the pins.
Resolving `hopsworks==5.0.3` with an unpinned `tensorflow` selects
**tensorflow 2.19.1 and numpy 2.1.3** — while local development has been on
2.21.0 and 2.4.6 throughout. The scheduled jobs were never running the stack any
result in this log was measured on. Nothing failed, nothing warned; the versions
simply differed. Pinning did not create this problem, it made the pre-existing
one impossible to ignore, which is the entire argument for pinning.

**Fix:** TensorFlow moved to `requirements-deep.txt`, installed as a second step.
Nothing deployed needs it — not the hourly feature job, the daily registration
job, the API, the dashboard, or the tests. Only the neural-net row of the model
comparison does, and `train.py` now imports it optionally and prints that the row
was skipped rather than omitting it silently.

Two things worth stating plainly in the report:

- hopsworks' `protobuf<5` bound is over-tight; this project has run against
  protobuf 7.x throughout. Declaring the conflict and working around it
  deliberately is different from not knowing it exists.
- A dependency conflict that only appears once you pin is a conflict you already
  had.

---

## Open items

- Assemble and structure the final report from this log
- Run the explainability workflow once to replace the committed Random Forest
  SHAP plot
- Consider rolling concentrations to the EPA's proper averaging windows as a
  second AQI column, so the approximation can be measured rather than only
  disclosed
- Refit the blend weight on a schedule rather than only at registration

### Resolved

- ~~Apply the purge/embargo fix to `time_based_split`~~ — Phase 3
- ~~Replace Random Forest with the per-horizon winner~~ — Phase 4
- ~~Add the persistence baseline to `train.py` and persist comparison results~~
  — Phase 4, written to `reports/model_comparison.json`
- ~~Pin `requirements.txt`; seed the TensorFlow model~~ — Phase 5
- ~~Fix or explicitly disclose the EPA averaging-window approximation~~ —
  disclosed, Phase 5
- ~~Rewrite README~~ — Phase 5
