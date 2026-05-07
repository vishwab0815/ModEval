# 📈 QuickHyre — State-Level Sales Forecasting System

> **Production-ready** time series forecasting pipeline that trains four competing algorithms per US state, auto-selects the best model using Optuna-tuned hyperparameters, and serves 8-week ahead sales predictions through a FastAPI REST interface.

---

## 🏆 Results at a Glance

| Metric | Value |
|---|---|
| States trained | **43** |
| Average accuracy (100 − MAPE) | **84.56 %** |
| Median accuracy | **84.91 %** |
| Best state | Pennsylvania — **89.08 %** (Prophet) |
| Worst state | Missouri — **79.28 %** (LSTM) |
| Target accuracy | ≥ 80 % ✅ |

**Model Win Distribution across 43 states:**

| Model | States Won | Share |
|---|---|---|
| 🥇 Prophet | 21 | 48.8 % |
| 🥈 XGBoost | 16 | 37.2 % |
| 🥉 LSTM | 6 | 14.0 % |
| SARIMA | 0 | 0 % |

---

## 📂 Repository Layout

```
QuickHyre-Assignment/
├── data/
│   └── Forecasting-Case-Study.csv     # Raw input: 8,084 rows, 43 US states, mixed date formats
├── models/
│   └── state_best_models.pkl          # Saved artifacts for all 43 states (auto-generated)
├── src/
│   ├── config.py                      # Centralised paths & column-name constants
│   ├── preprocessing.py               # Date parsing, weekly resampling, gap interpolation
│   ├── feature_engineering.py         # Lag, rolling, cyclical, and holiday features
│   ├── evaluation.py                  # MAE / RMSE / MAPE metric calculations
│   ├── model_training.py              # SARIMA · Prophet · XGBoost · LSTM + Optuna tuning
│   ├── prediction.py                  # Recursive 8-week forecasting per model type
│   ├── visualization.py               # Matplotlib/Plotly chart generation
│   └── api.py                         # FastAPI REST service
├── train_all_models.py                # End-to-end training pipeline script
├── compare_models.py                  # Side-by-side model comparison for a single state
├── main.py                            # Shortcut entry point for the API server
└── requirements.txt                   # Python dependencies
```

---

## ⚙️ Quick Start

### 1 — Clone & set up environment

```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install all dependencies
pip install -r requirements.txt
```

### 2 — Verify pre-trained models exist

```powershell
python -c "import os; print(os.path.exists('models/state_best_models.pkl'))"
# → True  (models already bundled)
```

### 3 — Start the API server

```powershell
.\venv\Scripts\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

### 4 — Explore the interactive docs

Open **http://127.0.0.1:8000/docs** in your browser to see the full Swagger UI.

---

## 🔁 Full Retraining (optional)

> ⚠️ This is long-running (~30–60 minutes depending on hardware). GPU strongly recommended.

```powershell
python train_all_models.py
```

This runs the complete pipeline:
1. Loads and cleans raw CSV  
2. Engineers 13 features  
3. Trains 4 models × 43 states (with Optuna hyperparameter search)  
4. Selects best per-state model  
5. Saves `models/state_best_models.pkl`

---

## 🌐 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Landing page with endpoint map |
| `GET` | `/health` | Service health check |
| `GET` | `/states` | Lists all 43 states with their best model + accuracy |
| `POST` | `/train` | Triggers end-to-end retraining (long-running) |
| `GET` | `/forecast/{state}` | Returns 8-week JSON forecast for the named state |
| `GET` | `/plot/{state}` | Returns a PNG chart: historical + 8-week forecast |

### Example — fetch a forecast

```bash
curl http://127.0.0.1:8000/forecast/California
```

**Response:**
```json
{
  "state": "California",
  "best_model": "XGBoost",
  "forecast_dates": ["2026-05-11", "2026-05-18", "..."],
  "forecast_values": [123456789.0, 124000000.0, "..."],
  "forecast": {
    "dates": ["2026-05-11", "..."],
    "forecast": [123456789.0, "..."]
  }
}
```

---

## 🏗️ Pipeline Architecture

```
Raw CSV (8,084 rows)
  │
  ▼ src/preprocessing.py
  ├─ Parse mixed date formats (MM/DD/YYYY and DD-MM-YYYY)
  ├─ Resample each state to weekly Monday cadence
  ├─ Linear interpolation for missing weeks
  └─ Output: Clean DataFrame (11,008 rows × 43 states)
  │
  ▼ src/feature_engineering.py
  ├─ Lag features:        lag_1, lag_7, lag_30
  ├─ Rolling statistics:  rolling_mean_7, rolling_std_7
  ├─ Temporal:            day_of_week, month, week_of_year
  ├─ Cyclical encoding:   month_sin, month_cos, week_sin, week_cos
  └─ Holiday flag:        is_holiday (US calendar)
  │
  ▼ src/model_training.py  (per state: 70% train / 15% val / 15% test)
  ├─ SARIMA   —  9 Optuna trials
  ├─ Prophet  — 15 Optuna trials
  ├─ XGBoost  — 25 Optuna trials  (GPU if available)
  └─ LSTM     — 10 Optuna trials  (GPU if available)
  │
  ▼ Best model selected by lowest validation RMSE
  │
  ▼ src/prediction.py
  └─ Recursive 8-week ahead forecast per state
  │
  ▼ src/api.py
  └─ FastAPI REST service  →  /forecast/{state}  /plot/{state}
```

---

## 🤖 Model Comparison: Why Four Models?

Each forecasting algorithm makes different assumptions about time series structure. Training all four and choosing the winner per state is the core strategy that achieves **84.56 % average accuracy**.

---

### 1 — SARIMA *(Seasonal AutoRegressive Integrated Moving Average)*

**What it is:** Classical statistical model. Decomposes the series into autoregressive, differencing, and moving-average components with an explicit seasonal period.

**How it works here:**
- Order `(p, d, q)` tuned by Optuna (9 trials; `p,d,q ∈ {0,1,2}`)
- Seasonal order fixed at `(0,1,1,13)` — quarterly seasonality on weekly data

**Strengths:**
- Mathematically interpretable; built on decades of statistical theory
- Works well on stationary series with clear seasonal structure
- No feature engineering needed — models the series itself

**Weaknesses:**
- Assumes linearity — cannot capture non-linear demand spikes
- Slow to fit on long series (O(n²) in some implementations)
- Sensitive to unit-root and stationarity assumptions
- **Won 0 of 43 states** — real-world beverage sales exhibit too much non-linearity for SARIMA

---

### 2 — Prophet *(Facebook/Meta Additive Decomposition)*

**What it is:** Additive regression model that decomposes a time series into `trend + seasonality + holidays + error`.

**How it works here:**
- Hyperparameters tuned by Optuna (15 trials):
  - `changepoint_prior_scale` (0.001 → 0.5): controls how flexible the trend is
  - `seasonality_prior_scale` (0.001 → 10.0): controls seasonal amplitude
  - `seasonality_mode`: `additive` vs `multiplicative`
- Yearly and weekly seasonality auto-detected
- Holiday effects built-in via the US holiday calendar

**Strengths:**
- Handles missing data and irregular intervals naturally
- Explicit changepoint detection (e.g., promotional events, COVID)
- Interpretable components — stakeholders can see trend vs seasonality
- Robust to outliers and scale changes
- **Won 21 of 43 states (48.8 %)** — best overall model

**Weaknesses:**
- Less precise when the series has complex, multi-lag autoregressive patterns
- Can over-smooth volatile states (e.g., Mississippi, Nebraska)
- Not GPU-acceleratable (CPU-only with Stan backend)

**Why it wins the most states:** Beverage sales follow structured yearly seasonality (summer peaks, holiday spikes) with gradual trend changes — exactly Prophet's sweet spot.

---

### 3 — XGBoost *(eXtreme Gradient Boosted Trees)*

**What it is:** Ensemble of decision trees, each correcting the residual errors of the previous. Transformed into a time series model by feeding engineered lag and rolling features as inputs.

**How it works here:**
- 13 hand-crafted features (lags, rolling stats, cyclical encodings, holiday flag)
- Hyperparameters tuned by Optuna (25 trials):
  - `n_estimators` (200–500), `max_depth` (2–8), `learning_rate` (0.01–0.15)
  - `subsample`, `colsample_bytree`, `colsample_bylevel` (row/feature sampling)
  - `reg_alpha` (L1), `reg_lambda` (L2), `gamma`, `min_child_weight`
- Auto-detects CUDA GPU for `hist` tree method (2–3× speedup)

**Strengths:**
- Handles non-linear interactions between features without assumptions
- Excellent at capturing complex lag relationships (e.g., lag_30 × month)
- Built-in regularisation prevents overfitting on smaller state datasets
- Fast inference; easy to interpret via feature importance
- **Won 16 of 43 states (37.2 %)**

**Weaknesses:**
- Requires future lag features at inference time → recursive forecasting accumulates error over 8 weeks
- Cannot extrapolate beyond the training range — relies on feature coverage
- Black-box compared to SARIMA or Prophet

**Why it wins many states:** States with high volatility or non-seasonal demand patterns benefit from XGBoost's ability to model complex, non-linear feature interactions.

---

### 4 — LSTM *(Long Short-Term Memory — Deep Learning)*

**What it is:** Recurrent neural network with memory gates that can selectively retain or forget past information over long sequences.

**How it works here:**
- Architecture: `Input(1) → LSTM(hidden_dim) → FC(1) → Output`
- Sequence length: 30 weeks of historical context
- Hyperparameters tuned by Optuna (10 trials): `hidden_dim ∈ {32, 64, 128}`, `learning_rate` (1e-4 → 1e-2), `dropout` (0 → 0.3)
- Final training: 200 epochs with early stopping (patience = 15)
- Full GPU acceleration via PyTorch + CUDA 12.8 on RTX 5050 (5–10× speedup)
- MinMax scaling applied per state before training

**Strengths:**
- Captures long-range temporal dependencies (30-week memory window)
- GPU-parallelised matrix operations — fastest absolute training on long series
- No hand-crafted features needed — learns representations from raw sequence
- Theoretically strongest for irregular, complex sequential patterns

**Weaknesses:**
- Needs more data to generalise well — smaller states suffer
- Slowest to converge in hyperparameter search (10 trials × 200 epochs)
- Recursive 8-week forecasting compounds prediction error
- **Won only 6 states** — mostly mid-size states with irregular patterns

**Why it wins some but not most:** States like Mississippi and Nebraska have volatile, irregular demand where LSTM's memory architecture helps, but the limited dataset size (~250 weekly observations per state) constrains its full potential.

---

### Model Selection Logic

```python
# From src/model_training.py — train_and_evaluate_all()
for name, (model, metrics) in results.items():
    if metrics and metrics['RMSE'] < best_score:
        best_score = metrics['RMSE']
        best_model = model
        best_model_name = name
```

Selection criterion: **lowest validation-set RMSE**. MAPE is then used for human-readable accuracy reporting (`accuracy = 100 - MAPE`).

---

### Summary Comparison Table

| Criterion | SARIMA | Prophet | XGBoost | LSTM |
|---|:---:|:---:|:---:|:---:|
| **States won** | 0 | **21** | 16 | 6 |
| **Handles non-linearity** | ❌ | ⚠️ Partial | ✅ Strong | ✅ Strong |
| **Seasonal patterns** | ✅ | ✅ Best | ✅ via features | ✅ via memory |
| **Holiday awareness** | ❌ | ✅ Built-in | ✅ via feature | ❌ |
| **GPU acceleration** | ❌ | ❌ | ✅ | ✅ |
| **Interpretability** | ✅ High | ✅ High | ⚠️ Medium | ❌ Low |
| **Data requirement** | Low | Low | Medium | High |
| **Optuna trials** | 9 | 15 | **25** | 10 |
| **Typical fit time/state** | Slow | Medium | Fast (GPU) | Slow (GPU) |

---

## 🧪 Feature Engineering Details

| Feature | Type | Purpose |
|---|---|---|
| `lag_1`, `lag_7`, `lag_30` | Autoregressive | Last 1, 7, 30 weeks of sales |
| `rolling_mean_7` | Trend | 7-week moving average |
| `rolling_std_7` | Volatility | 7-week moving standard deviation |
| `month`, `week_of_year`, `day_of_week` | Temporal | Calendar position |
| `month_sin`, `month_cos` | Cyclical | Preserves Dec→Jan continuity |
| `week_sin`, `week_cos` | Cyclical | Preserves week 52→1 continuity |
| `is_holiday` | Event | US federal holiday binary flag |

> **Why cyclical encoding?** A raw `month` feature treats December (12) as distant from January (1). Sine/cosine encoding places them adjacent on a circle — critical for capturing year-end patterns.

---

## 📊 Evaluation Metrics

| Metric | Formula | What it tells you |
|---|---|---|
| **MAE** | mean(\|actual − pred\|) | Average absolute error in dollar terms |
| **RMSE** | √mean((actual − pred)²) | Penalises large errors more heavily; used for model selection |
| **MAPE** | mean(\|actual − pred\| / actual) × 100 | Scale-independent; used to compute accuracy % |
| **Accuracy** | 100 − MAPE | Human-readable score; target ≥ 80 % |

---

## 🖥️ GPU Acceleration

| Component | Hardware | Speedup |
|---|---|---|
| LSTM training | NVIDIA RTX 5050 (CUDA 12.8) | ~5–10× vs CPU |
| XGBoost (`hist` method) | NVIDIA RTX 5050 (CUDA 12.8) | ~2–3× vs CPU |
| Overall pipeline | GPU-enabled | ~2–3× faster total |

> CPU-only training is fully supported. Remove or omit the `device` parameter in `src/model_training.py` if no CUDA GPU is available.

---

## 🛠️ Troubleshooting

| Symptom | Fix |
|---|---|
| API fails to start | `pip install -r requirements.txt` in active venv |
| `state_best_models.pkl` not found | Run `python train_all_models.py` to generate it |
| XGBoost GPU warning | Non-fatal. Force CPU: remove `'device': 'cuda'` in `model_training.py` |
| Prophet Stan warnings | Non-fatal. Prophet falls back to robust optimisers automatically |
| LSTM skipped | `torch` not installed. Run `pip install torch` |

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `pandas` | Data manipulation and resampling |
| `numpy` | Numerical operations |
| `scikit-learn` | MinMaxScaler, train/test utilities |
| `statsmodels` | SARIMAX implementation |
| `prophet` | Facebook Prophet (via cmdstanpy) |
| `xgboost` | XGBRegressor with GPU support |
| `torch` | LSTM via PyTorch (CUDA optional) |
| `optuna` | Bayesian hyperparameter optimisation |
| `fastapi` | REST API framework |
| `uvicorn` | ASGI server |
| `matplotlib` | Static plot generation |
| `joblib` | Model serialisation (`.pkl`) |
| `holidays` | US holiday calendar |
| `openpyxl` | Excel support (optional data export) |

---

## 🗺️ Training Data Split

Each state's time series is split **chronologically** (no data leakage):

```
|←──────────── 70% Train ────────────→|←─ 15% Val ─→|←─ 15% Test ─→|
  Used to fit models                    Optuna tunes    Final RMSE
                                        hyperparams     reported
```

With ~256 weekly observations per state:
- **Train**: ~179 weeks (~3.4 years)
- **Validation**: ~38 weeks (~9 months)
- **Test**: ~38 weeks (~9 months)

---

## 🔎 Complete API Reference (detailed)

Below are every endpoint implemented by `src.api` with request/response examples, common status codes, and notes to help integrate the service.

### GET /
- Description: API landing page with version and endpoint map
- Method: `GET`
- URL: `http://127.0.0.1:8000/`
- Success: `200 OK`

Example:
```bash
curl http://127.0.0.1:8000/
```

Response (JSON):
```json
{
  "message": "QuickHyre Forecasting API is running",
  "documentation": "Open /docs for the interactive Swagger UI or /redoc for the alternative documentation.",
  "version": "1.0.0",
  "available_endpoints": {"health":"GET /health","states":"GET /states","train":"POST /train","forecast":"GET /forecast/{state}","plot":"GET /plot/{state}"}
}
```

### GET /health
- Description: Lightweight service health check for orchestration/monitoring
- Method: `GET`
- URL: `http://127.0.0.1:8000/health`
- Success: `200 OK`

Example & response (PowerShell):
```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
# -> { status: 'healthy', service: 'QuickHyre Forecasting API', version: '1.0.0' }
```

### GET /states
- Description: Lists trained states, their selected model, and accuracy (100 - MAPE)
- Method: `GET`
- URL: `http://127.0.0.1:8000/states`
- Success: `200 OK`

Sample response:
```json
{
  "total_states": 43,
  "states": [
    {"state":"Alabama","best_model":"XGBoost","accuracy":"83.12%"},
    {"state":"California","best_model":"XGBoost","accuracy":"85.23%"}
  ]
}
```

Notes:
- If `models/state_best_models.pkl` is missing the endpoint returns `400` with message: "Models not trained yet. Call POST /train first.".

### POST /train
- Description: Trigger full end-to-end retraining of all models and save artifacts to `models/state_best_models.pkl` (long-running).
- Method: `POST`
- URL: `http://127.0.0.1:8000/train`
- Success: `200 OK` (on completion)
- Failure: `500 Internal Server Error` (if training crashes)

Behavior & caution:
- This endpoint runs `load_and_preprocess`, feature engineering, and `TimeSeriesTrainer.train_and_evaluate_all()`; it performs Optuna searches per model and per-state and can take from several minutes to hours depending on hardware and number of trials.
- Use this on a development machine or a worker instance — do not call repeatedly from a synchronous request in production without asynchronous/queueing wrappers.

Sample (curl):
```bash
curl -X POST http://127.0.0.1:8000/train
```

Sample successful response (abbreviated):
```json
{
  "message": "Training completed successfully",
  "total_states": 43,
  "summary": {"California":{"model":"XGBoost","rmse":12345.67},"Alabama":{"model":"Prophet","rmse":23456.78}}
}
```

### GET /forecast/{state}
- Description: Return an 8-week forecast for the requested `state` (title-cased), using the stored best model for that state.
- Method: `GET`
- URL: `http://127.0.0.1:8000/forecast/California`
- Success: `200 OK`
- Client error: `404 Not Found` (if state not present), `400 Bad Request` (models not trained)

Example:
```bash
curl http://127.0.0.1:8000/forecast/California
```

Response (example):
```json
{
  "state": "California",
  "best_model": "XGBoost",
  "forecast_dates": ["2026-05-11","2026-05-18","2026-05-25","2026-06-01","2026-06-08","2026-06-15","2026-06-22","2026-06-29"],
  "forecast_values": [945699027.41,898772004.28,860994249.91, ...],
  "forecast": {
    "dates": ["2026-05-11", ...],
    "forecast": [945699027.41, ...]
  }
}
```

Notes:
- The response includes both `forecast_dates` / `forecast_values` for immediate use and a nested `forecast` object for compatibility with internal code.
- For reproducibility, the model metadata (scalers, seq_length for LSTM, etc.) is loaded from the saved artifact.

### GET /plot/{state}
- Description: Render and return a PNG chart containing historical data, the holdout test series, and the 8-week forecast.
- Method: `GET`
- URL: `http://127.0.0.1:8000/plot/California`
- Success: `200 OK` (Content-Type: `image/png`)

Example (save PNG):
```bash
curl http://127.0.0.1:8000/plot/California --output California_forecast.png
```

Notes:
- The plotting code uses Matplotlib (and Plotly when available). If running the server without a display, the endpoint returns a static PNG built in memory.

### Docs & OpenAPI
- Swagger UI: `http://127.0.0.1:8000/docs` — interactive testing and generated OpenAPI spec.
- ReDoc: `http://127.0.0.1:8000/redoc`

---

If you'd like, I can add sample Postman/Insomnia collection JSON or a `demo_script.ps1` that automates `curl` calls for the demo—tell me which format you prefer. 
## 🚀 Deployment Checklist

- [ ] `python train_all_models.py` — train and save all models
- [ ] Confirm `models/state_best_models.pkl` exists
- [ ] `uvicorn src.api:app --host 0.0.0.0 --port 8000` — start the service
- [ ] `curl http://localhost:8000/health` — verify service is live
- [ ] `curl http://localhost:8000/forecast/California` — test a forecast
- [ ] Open `http://localhost:8000/docs` — explore Swagger UI
- [ ] Schedule `POST /train` weekly or monthly to retrain with fresh data

---

## 📁 Key Files Reference

| File | Role |
|---|---|
| `src/config.py` | Single source of truth: paths, column names |
| `src/preprocessing.py` | `load_and_preprocess()` — cleans raw CSV |
| `src/feature_engineering.py` | `create_features()` — builds 13 ML features |
| `src/model_training.py` | `TimeSeriesTrainer` — all 4 models + Optuna |
| `src/prediction.py` | `generate_forecasts()` — recursive 8-week prediction |
| `src/evaluation.py` | `evaluate_metrics()` — MAE, RMSE, MAPE |
| `src/api.py` | FastAPI app with 5 endpoints |
| `src/visualization.py` | PNG chart builder |
| `train_all_models.py` | Full CLI training pipeline with summary table |
| `compare_models.py` | Ad-hoc side-by-side model comparison for one state |

---

