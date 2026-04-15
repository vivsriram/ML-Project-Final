# ML Assignment 2 – Project Context & Requirements

## Project Overview
**Airline Passenger Flow Model** — predicting hourly TSA checkpoint throughput across U.S. airports.

**Team:** Arend Colle, Vivek Sriram, Tiffany T. Nguyen

---

## Coding Standards (Enforce Throughout)
- **camelCase** for all variable and function names (e.g., `trainDf`, `hourlyPassengers`, `loadFactor`)
- **Concise, readable code** — avoid redundant logic; prefer vectorized operations over loops
- **Clear inline comments** for every non-obvious step
- **Docstrings** on every function: describe inputs, outputs, and purpose
- **Runbook / README** must be included so others can reproduce the full pipeline

---

## ML Objective
**Regression task:** Given airport, checkpoint, date, hour, and flight/calendar features — predict `totalPassengers` at a TSA checkpoint.

---

## Models to Train
Based on the referenced literature, implement at least one of the following:

| Model | Source |
|---|---|
| **Random Forest Regressor** | Monmousseau et al. (2020) — CDG Airport study |
| **LSTM (Long Short-Term Memory)** | Monmousseau et al. (2020) — CDG Airport study |
| **Regression Tree + Simulation** | Guo, Grushka-Cockayne & De Reyck (2021) — Heathrow study |

Start with **Random Forest** as the baseline. LSTM is the stretch goal.

---

## Airport Scope
Filter all datasets to the following 20 airports (by IATA code) before any modeling:

```python
targetAirports = [
    "ATL", "LAX", "DFW", "DEN", "ORD",
    "JFK", "MCO", "LAS", "CLT", "MIA",
    "PHX", "IAH", "BOS", "FLL", "MSP",
    "LGA", "DTW", "SEA", "SFO", "EWR"
]
```

These are the 20 busiest U.S. airports by passenger enplanements (FAA data). All 20 are covered by the TSA checkpoint throughput dataset.

---

## Datasets

### 1. TSA Checkpoint Throughput (Prediction Target)
Source: `mikelor/TsaThroughput` GitHub repo (pre-processed from TSA FOIA PDFs)

**Raw format:** Wide — each row is a `Date` + `Hour`, with one column per checkpoint named `"AAA CheckpointName"` (e.g., `ATL Main Checkpoint`, `JFK Terminal 8`).

**Reshape to long format before use:**
```python
# After melting:
# date | hour | airportCode | checkpointName | totalPassengers
```

**Key notes:**
- Multiple checkpoints per airport (e.g., ATL has 8, DFW has 14, LAX has 12)
- Hours are in `HH:MM:SS` format (e.g., `03:00:00`) — parse and extract integer hour
- Values are floats; nulls mean no data for that checkpoint/hour
- Filter columns to only those whose prefix matches `targetAirports`

---

### 2. BTS On-Time Performance (Departure Features)
Source: BTS Reporting Carrier On-Time Performance database

**Key fields from raw data:**

| Field | Description |
|---|---|
| `FL_DATE` | Flight date |
| `ORIGIN` | Origin airport IATA code |
| `DEP_TIME` | Actual departure time (HHMM) |
| `CRS_DEP_TIME` | Scheduled departure time (HHMM) |
| `OP_UNIQUE_CARRIER` | Airline code |
| `CANCELLED` | 1 if cancelled |
| `DEP_DELAY` | Departure delay in minutes |

**Engineer from this:**
- `hourlyDepartureCount` — count of scheduled departures per `ORIGIN` per hour (use `CRS_DEP_TIME` floored to hour)
- Filter to rows where `ORIGIN` is in `targetAirports` — `DEST` can be any airport (domestic or outside the 20)
- Exclude cancelled flights (`CANCELLED == 1`)

---

### 3. BTS T-100 Domestic Segment (Load Factor)
Source: BTS T-100 Domestic Segment table

**Key fields:** `ORIGIN`, `DEST`, `PASSENGERS`, `SEATS`, `DEP_PERFORMED`

**Engineer from this:**
- `avgLoadFactor` = `PASSENGERS / SEATS` per route per month
- Join to on-time data by `ORIGIN` + `DEST` + month to estimate `expectedPassengerVolume`

---

### 4. Calendar Features
Source: Python `holidays` library + date math

| Feature | Description |
|---|---|
| `isHoliday` | Binary — U.S. federal holiday |
| `daysToNearestHoliday` | Integer distance to nearest holiday |
| `dayOfWeek` | 0=Monday ... 6=Sunday |
| `month` | 1–12 |
| `isWeekend` | Binary |

---

## Feature Engineering
Construct these before training:

- `hourlyDepartureCount` — scheduled departures per airport per hour (from BTS On-Time)
- `avgLoadFactor` — `passengers / seats` per route (from T-100)
- `expectedPassengerVolume` — `hourlyDepartureCount * avgLoadFactor`
- `dayOfWeek`, `month`, `isHoliday`, `daysToHoliday` — calendar indicators
- `checkpointId`, `airportCode` — categorical encodings

---

## Evaluation Metrics
Match the metrics used in the source studies:

| Metric | Justification |
|---|---|
| **R²** | Measures proportion of variance explained — used by Monmousseau et al. |
| **MAE (Mean Absolute Error)** | Interpretable in passenger-count units — used by Monmousseau et al. |
| **Pearson Correlation (daily)** | Captures alignment of predicted vs. actual daily patterns — used by Monmousseau et al. |
| **RMSE** | Penalizes large errors; useful for staffing decisions where spikes matter |

---

## Best Practices Required
- **Train/validation/test split** — respect temporal order (no future leakage); split chronologically
- **Cross-validation** — use time-series CV (e.g., `TimeSeriesSplit`) not random K-fold
- **Hyperparameter tuning** — document which params were tuned and how (e.g., grid search, manual)
- **Baseline comparison** — compare model against a naive baseline (e.g., same hour, same weekday, prior week)
- **No data leakage** — features must only use information available before the prediction timestamp

---

## Code Organization
- One notebook per major step is acceptable; suggested structure:
  - `01_dataIngestion.ipynb` — load and merge all datasets
  - `02_featureEngineering.ipynb` — construct all model features
  - `03_modelTraining.ipynb` — train, tune, and evaluate models
- Save trained model artifacts (e.g., `randomForestModel.pkl`) to a `/models` directory
- Version control everything on GitHub
- Include a `README.md` runbook with: environment setup, data download steps, and notebook execution order

---

## Deliverable Checklist
- [ ] At least 1 ML model trained (Random Forest minimum)
- [ ] Metrics defined and justified (R², MAE, Pearson, RMSE)
- [ ] Model evaluated on held-out test set
- [ ] Best practices documented (temporal split, CV strategy)
- [ ] Code organized, commented, and version-controlled
- [ ] Runbook written so teammates can reproduce results
- [ ] Project update write-up: max 3 pages (figures/screenshots excluded)
