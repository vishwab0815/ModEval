# QuickHyre Time Series Forecasting - Project Explanation Guide

## Project Overview
**Objective**: Build a production-ready time series forecasting system that trains multiple forecasting algorithms, compares and selects the best model per state, and exposes predictions via REST API.

**Dataset**: US beverage sales by state (weekly aggregated, 8,084 raw rows → 11,008 cleaned rows across 43 states)

**Target Accuracy**: 80%+ (currently achieving 84.50% average)

---

## Project Structure

```
QuickHyre-Assignment/
├── data/
│   └── Forecasting-Case-Study.csv     # Raw data: 8,084 rows, mixed date formats
├── models/
│   └── state_best_models.pkl          # Saved trained models for all 43 states
├── src/
│   ├── api.py                         # FastAPI REST endpoints
│   ├── config.py                      # Configuration constants
│   ├── evaluation.py                  # Metric calculations (MAPE, RMSE, MAE)
│   ├── feature_engineering.py         # Feature creation (lags, rolling stats, temporal)
│   ├── model_training.py              # Model implementations & training logic
│   ├── prediction.py                  # 8-week recursive forecasting
│   ├── preprocessing.py               # Data cleaning & normalization
│   └── visualization.py               # Plotting utilities
├── main.py                            # Entry point for full pipeline
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
```

---

## Module-by-Module Explanation

### 1. **src/config.py** - Configuration Hub
**Purpose**: Centralized configuration constants

**Key Constants**:
- `DATA_PATH`: Path to input CSV
- `TARGET_COL`: 'Total' (sales column to forecast)
- `DATE_COL`: 'Date' (temporal column)
- `STATE_COL`: 'State' (grouping column)
- `MODELS_DIR`: Directory to save trained models

**Why it matters**: Single source of truth for all paths and column names. Easy to modify without touching multiple files.

---

### 2. **src/preprocessing.py** - Data Cleaning & Preparation
**Purpose**: Transform raw messy data into clean weekly time series

**Key Functions**:

#### `load_and_preprocess(data_path)`
- **Input**: CSV file path
- **Process**:
  1. Load CSV with pandas
  2. Parse mixed date formats (MM/DD/YYYY and DD-MM-YYYY)
  3. Normalize to weekly Monday cadence (resampling)
  4. Handle missing weeks with linear interpolation
  5. Remove rows with NaN values
- **Output**: Clean DataFrame (11,008 rows, 43 states)
- **Key Issue Fixed**: Mixed date formats were causing 40% data loss initially

**Data Flow**:
```
Raw CSV (8,084 rows) 
  → Parse dates (both formats) 
  → Resample to weekly Monday 
  → Interpolate gaps 
  → Clean DataFrame (11,008 rows)
```

---

### 3. **src/feature_engineering.py** - Feature Creation
**Purpose**: Create intelligent features to improve model accuracy

**Key Functions**:

#### `create_features(df)`
- **Input**: Clean DataFrame from preprocessing
- **Features Created**:
  1. **Lag Features**: 
     - `lag_1`, `lag_7`, `lag_30` (previous week, 7 weeks, 30 weeks)
     - Captures temporal dependencies
  
  2. **Rolling Statistics**:
     - `rolling_mean_7`: 7-week moving average (trend)
     - `rolling_std_7`: 7-week moving std deviation (volatility)
  
  3. **Temporal Features**:
     - `day_of_week`: Day of week (0-6)
     - `month`: Month number (1-12)
     - `week_of_year`: Week number (1-52)
  
  4. **Cyclical Encoding** (preserve circularity):
     - `month_sin`, `month_cos`: Sine/cosine of month (preserves year-end wrap)
     - `week_sin`, `week_cos`: Sine/cosine of week (preserves year-end wrap)
  
  5. **Holiday Flag**:
     - `is_holiday`: Binary flag for US holidays

- **Output**: DataFrame with 13 engineered features
- **Why cyclical encoding**: Month 12 should be close to Month 1, not far away

**Feature Importance Ranking** (typical):
1. Lag features (strongest predictor)
2. Rolling statistics (trend capture)
3. Month cyclical encoding (seasonality)
4. Holiday flag (event detection)

---

### 4. **src/evaluation.py** - Performance Metrics
**Purpose**: Measure model accuracy using multiple metrics

**Key Functions**:

#### `evaluate_metrics(actual, predicted)`
- **Returns Dictionary**:
  ```python
  {
    'MAE': Mean Absolute Error (average $ error),
    'RMSE': Root Mean Squared Error (penalizes large errors),
    'MAPE': Mean Absolute Percentage Error (% error, normalized by scale)
  }
  ```

**How Accuracy is Calculated**:
```
Accuracy (%) = 100 - MAPE (%)
Example: MAPE=15.50% → Accuracy=84.50%
```

**Why 3 metrics**?
- **MAE**: Easy to interpret (in dollar terms)
- **RMSE**: Penalizes outliers (important for forecasting)
- **MAPE**: Scale-independent (compare across states)

---

### 5. **src/model_training.py** - The Core Engine
**Purpose**: Train 4 different forecasting algorithms + hyperparameter optimization

**Key Classes & Functions**:

#### Class: `TimeSeriesTrainer`
Constructor: `__init__(self, df)`
- **Input**: DataFrame with all 43 states
- **Stores**: Full dataset for training access

#### Method: `train_val_test_split_ts(group)`
- **Purpose**: Time-series safe splitting (no leakage)
- **Split**: 70% train / 15% validation / 15% test
- **Why this split**: Temporal order preserved (no mixing past with future)
- **Output**: (train_df, val_df, test_df)

#### Method: `train_sarima(train, val)`
- **Model**: SARIMA (Seasonal ARIMA)
- **Hyperparameters Tuned**: (p, d, q) order
- **Search Space**: p∈[0-2], d∈[0-2], q∈[0-2] → 9 trials
- **Seasonal Order**: Fixed (0,1,1,52) for weekly data (52 weeks/year)
- **Output**: (model, metrics_dict)
- **Pros**: Interpretable, theory-based
- **Cons**: Assumes stationarity, slow

#### Method: `train_prophet(train, val)`
- **Model**: Facebook Prophet (Additive decomposition)
- **Hyperparameters Tuned** (15 trials):
  - `changepoint_prior_scale`: Controls flexibility (0.001-0.5)
  - `seasonality_prior_scale`: Controls seasonal strength (0.001-10.0)
  - `seasonality_mode`: Additive vs Multiplicative
- **Output**: (model, metrics_dict)
- **Pros**: Handles holidays, robust, fast
- **Cons**: Less accurate for complex patterns

#### Method: `train_xgboost(train, val)`
- **Model**: XGBoost (Gradient Boosted Trees)
- **Hyperparameters Tuned** (25 trials):
  - `n_estimators`: Number of trees (200-500)
  - `max_depth`: Tree depth (2-8)
  - `learning_rate`: Step size (0.01-0.15)
  - `subsample`: Row sampling (0.5-0.95)
  - `colsample_bytree`: Feature sampling (0.5-0.95)
  - `reg_alpha`, `reg_lambda`: Regularization (prevents overfitting)
  - `gamma`: Minimum loss reduction
- **GPU Support**: Auto-detects CUDA, runs on GPU if available
- **Output**: (model, metrics_dict)
- **Pros**: Accurate, handles non-linearity, prevents overfitting
- **Cons**: Black-box, needs tuning

#### Method: `train_lstm(train, val)`
- **Model**: Long Short-Term Memory (Deep Learning)
- **Architecture**: Input → LSTM (hidden units) → FC layer → Output
- **Hyperparameters Tuned** (10 trials):
  - `hidden_dim`: LSTM units (32, 64, 128)
  - `lr`: Learning rate (1e-4 to 1e-2)
- **Early Stopping**: Monitors validation loss, stops if no improvement (patience=15)
- **GPU Support**: Full GPU training with PyTorch
- **Sequence Length**: 30 weeks of history
- **Output**: (model, metrics_dict)
- **Pros**: Captures long-term dependencies, GPU accelerated
- **Cons**: Needs more data, slower training

#### Method: `train_and_evaluate_all()`
- **Main Training Loop**:
  ```python
  For each state:
    1. Split data (train/val/test)
    2. Train SARIMA, Prophet, XGBoost, LSTM
    3. Compare RMSE scores
    4. Select best model (lowest RMSE)
    5. Save state result
  ```
- **Output**: Dictionary with best model for each state
- **Best Model Distribution** (typical): XGBoost (50%), Prophet (30%), LSTM (15%), SARIMA (5%)

---

### 6. **src/prediction.py** - Forecasting (8 weeks ahead)
**Purpose**: Generate future predictions using trained models

**Key Functions**:

#### `generate_forecasts(state, best_model_dict)`
- **Input**: State name + trained model
- **Process**: Recursive multi-step forecasting
  - Predict week 1 → Use for week 2 prediction
  - Predict week 2 → Use for week 3 prediction
  - ... repeat for 8 weeks
- **Output**: Array of 8 predicted sales values

**Recursion Example**:
```
Week 1 prediction: use actual history (lags)
Week 2 prediction: use week 1 pred + history
Week 3 prediction: use week 1, 2 preds + history
...
Week 8 prediction: use weeks 1-7 preds + history
```

---

### 7. **src/api.py** - REST API Service
**Purpose**: Expose models via HTTP endpoints

**Framework**: FastAPI

**Presentation Improvements**:
- API landing page now shows version and endpoint map
- Swagger/OpenAPI docs are grouped with tags: System, Training, Forecasting
- `/states` endpoint lists all trained states and their best model
- `/forecast/{state}` returns both structured forecast arrays and metadata

**Key Endpoints**:

#### `GET /`
- **Purpose**: API landing page
- **Response**: Service version, documentation links, and endpoint map

#### `GET /health`
- **Purpose**: Check service health
- **Response**: `status`, `service`, and `version`

#### `GET /states`
- **Purpose**: List trained states and best models
- **Response**: Total state count plus model names per state

#### `POST /train`
- **Purpose**: Retrain all models for all states
- **Response**: Training summary (accuracy, model counts)
- **Use**: Update models with new data

#### `GET /forecast/{state}`
- **Purpose**: Get 8-week forecast for a state
- **Parameters**: `state` (e.g., "California")
- **Response**: 
  ```json
  {
    "state": "California",
    "best_model": "XGBoost",
    "forecast_dates": ["2026-05-11", "2026-05-18"],
    "forecast_values": [123456789.0, 124000000.0],
    "forecast": {
      "dates": ["2026-05-11", "2026-05-18"],
      "forecast": [123456789.0, 124000000.0]
    }
  }
  ```
- **Use**: Production predictions

#### `GET /plot/{state}`
- **Purpose**: Plot historical + forecasted data
- **Response**: Interactive Plotly chart
- **Use**: Visualization

**How to Run**:
```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

---

### 8. **main.py** - Pipeline Orchestration
**Purpose**: Full end-to-end pipeline

**Execution Flow**:
```
1. Load & preprocess data
2. Create features
3. Train all 4 models on 43 states
4. Select best model per state
5. Save models to disk
6. Generate summary report
```

**Run**:
```bash
python main.py
```

---

## Data Flow Diagram

```
Raw CSV
  ↓
Preprocessing (src/preprocessing.py)
  • Parse mixed dates
  • Weekly resampling
  • Interpolation
  ↓
Clean DataFrame
  ↓
Feature Engineering (src/feature_engineering.py)
  • Lag features (lag_1, lag_7, lag_30)
  • Rolling stats (mean_7, std_7)
  • Temporal (month, week, day)
  • Cyclical (sin/cos)
  • Holiday flag
  ↓
Feature-Rich DataFrame (13 columns)
  ↓
Model Training (src/model_training.py)
  ├─ SARIMA (9 trials)
  ├─ Prophet (15 trials)
  ├─ XGBoost (25 trials on GPU)
  └─ LSTM (10 trials on GPU)
  ↓
Best Model Selection (per state)
  ↓
Predictions (src/prediction.py)
  • 8-week recursive forecast
  ↓
REST API (src/api.py)
  ├─ /train
  ├─ /forecast/{state}
  └─ /plot/{state}
```

---

## GPU Acceleration Setup

**Hardware**: NVIDIA GeForce RTX 5050 Laptop GPU

**Software**:
- PyTorch 2.11.0+cu128 (CUDA 12.8)
- XGBoost with GPU support

**Speedup**:
- LSTM: ~5-10x faster
- XGBoost: ~2-3x faster
- Overall: ~2-3x faster than CPU-only

---

## Key Performance Metrics

**Current Results** (with hyperparameter tuning):
- Average Accuracy: 84.50%
- Median Accuracy: 84.73%
- Average MAPE: 15.50%
- Best Model Distribution:
  - XGBoost: 23 states (53%)
  - Prophet: 13 states (30%)
  - LSTM: 7 states (16%)
  - SARIMA: 0 states (0%)

**Top 3 States**:
1. West Virginia: 89.21% (XGBoost)
2. Maryland: 88.82% (XGBoost)
3. Wisconsin: 88.45% (XGBoost)

**Lowest 3 States**:
1. Mississippi: 76.40% (LSTM)
2. Nebraska: 77.62% (LSTM)
3. Michigan: 78.78% (LSTM)

---

## Study Order (Presentation Flow)

### For Beginners:
1. **config.py** - Understand constants
2. **preprocessing.py** - See data cleaning
3. **feature_engineering.py** - Learn feature creation
4. **evaluation.py** - Understand metrics
5. **model_training.py** - Study one model first (Prophet)
6. **main.py** - Full pipeline
7. **api.py** - Deployment

### For Data Scientists:
1. **model_training.py** - All 4 models in detail
2. **feature_engineering.py** - Feature strategies
3. **preprocessing.py** - Data quality issues
4. **evaluation.py** - Cross-validation strategy
5. **prediction.py** - Multi-step forecasting

### For ML Engineers:
1. **src/model_training.py** - GPU setup, Optuna tuning
2. **src/api.py** - FastAPI deployment
3. **requirements.txt** - Dependencies
4. **main.py** - Production pipeline

---

## Hyperparameter Tuning Details

**Optuna Framework** - Bayesian Optimization

### XGBoost Tuning:
```python
25 trials searching:
- n_estimators: 200-500 (more trees)
- max_depth: 2-8 (deeper trees)
- learning_rate: 0.01-0.15 (smaller steps)
- reg_alpha: 1e-5 to 1.0 (L1 penalty)
- reg_lambda: 1e-5 to 1.0 (L2 penalty)
- subsample: 0.5-0.95 (row sampling)
- colsample_bytree: 0.5-0.95 (feature sampling)

Result: Best params per state saved + applied
```

### LSTM Tuning:
```python
10 trials searching:
- hidden_dim: [32, 64, 128]
- learning_rate: 1e-4 to 1e-2

Early Stopping: Patience=15 (stops if no improvement)
Final Training: 200 epochs with early stopping
GPU: Full training on CUDA 12.8
```

---

## Common Questions & Answers

**Q: Why multiple models?**
A: Different models capture different patterns. XGBoost excels at non-linearity, Prophet at seasonality, LSTM at temporal dependencies.

**Q: Why weekly aggregation?**
A: Reduces noise, captures natural business cycles (weekly patterns), reduces overfitting.

**Q: Why cyclical encoding?**
A: Month 12 is closer to Month 1 than to Month 6. Sin/cos encoding preserves this circularity.

**Q: Why 70/15/15 split?**
A: Maximize training data (70%), validate hyperparameters (15%), final test (15%). Time-order preserved.

**Q: Why Optuna for tuning?**
A: Bayesian optimization smarter than grid search. Focuses on promising regions of parameter space.

**Q: How does GPU accelerate?**
A: Matrix operations parallelized across thousands of GPU cores. LSTM matrices computed simultaneously.

---

## Deployment Checklist

- [ ] Train all models: `python main.py`
- [ ] Save models: `models/state_best_models.pkl`
- [ ] Start API: `uvicorn src.api:app --host 0.0.0.0 --port 8000`
- [ ] Test endpoints: `curl http://localhost:8000/forecast/California`
- [ ] Monitor accuracy via `/train` endpoint
- [ ] Schedule retraining weekly/monthly

---

## Dependencies Overview

| Package | Purpose |
|---------|---------|
| pandas | Data manipulation |
| numpy | Numerical computing |
| scikit-learn | Preprocessing (scalers, splits) |
| statsmodels | SARIMA implementation |
| prophet | Prophet implementation |
| xgboost | XGBoost implementation |
| torch | LSTM (PyTorch) |
| optuna | Hyperparameter tuning |
| fastapi | REST API framework |
| uvicorn | API server |
| plotly | Interactive plots |
| joblib | Model serialization |

---

## Summary

This project demonstrates:
✅ **Data Engineering**: Multi-format parsing, time-series resampling
✅ **Feature Engineering**: Lag, rolling stats, temporal, cyclical features
✅ **Model Development**: 4 different algorithms with automatic selection
✅ **Hyperparameter Tuning**: Optuna-based Bayesian optimization
✅ **GPU Acceleration**: CUDA 12.8 for 2-3x speedup
✅ **REST API**: Production-ready FastAPI deployment
✅ **Performance**: 84.50% average accuracy (above 80% target)


