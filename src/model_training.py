import pandas as pd
import numpy as np
import joblib
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from src.config import TARGET_COL, DATE_COL, STATE_COL, MODELS_DIR
from src.evaluation import evaluate_metrics
import optuna
import logging
from typing import Any

# Set optuna logging level
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Stable module-scope defaults keep Pylance from treating torch symbols as possibly unbound.
torch: Any = None
nn: Any = None
TORCH_DEVICE: Any = None
HAS_TORCH = False
LSTMModelClass: Any = None

# Try to import torch for LSTM
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
    TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    class LSTMModelImpl(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(LSTMModelImpl, self).__init__()
            self.hidden_dim = hidden_dim
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out
    LSTMModelClass = LSTMModelImpl
except ImportError:
    HAS_TORCH = False
    TORCH_DEVICE = None
    print("Torch not found. LSTM will be skipped.")

class TimeSeriesTrainer:
    def __init__(self, df):
        self.df = df
        self.best_model = None
        self.best_score = float('inf')
        self.best_model_name = ""

    def train_val_test_split_ts(self, group):
        n = len(group)
        train_end = max(int(n * 0.70), 1)
        val_end = max(int(n * 0.85), train_end + 1)
        val_end = min(val_end, n - 1)

        train = group.iloc[:train_end]
        val = group.iloc[train_end:val_end]
        test = group.iloc[val_end:]
        return train, val, test

    def train_sarima(self, train, val):
        try:
            # Hyperparameter tuning for SARIMA
            def objective_sarima(trial):
                p = trial.suggest_int('p', 0, 2)
                d = trial.suggest_int('d', 0, 2)
                q = trial.suggest_int('q', 0, 2)
                
                try:
                    model = SARIMAX(train[TARGET_COL], 
                                    order=(p, d, q), 
                                    seasonal_order=(0, 1, 1, 13),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
                    results: Any = model.fit(disp=False)
                    preds: Any = results.forecast(len(val))  # type: ignore
                    # Validate predictions
                    if np.isnan(preds).any() or np.isinf(preds).any():
                        return float('inf')
                    metrics = evaluate_metrics(val[TARGET_COL], preds)
                    return metrics['RMSE']
                except:
                    return float('inf')
            
            # Optimize SARIMA parameters
            study = optuna.create_study(direction='minimize')
            study.optimize(objective_sarima, n_trials=9, show_progress_bar=False)
            
            best_params = study.best_params
            model = SARIMAX(train[TARGET_COL], 
                            order=(best_params['p'], best_params['d'], best_params['q']), 
                            seasonal_order=(0, 1, 1, 13),
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            results: Any = model.fit(disp=False)
            preds: Any = results.forecast(len(val))  # type: ignore
            # Validate predictions
            if np.isnan(preds).any() or np.isinf(preds).any():
                print(f"SARIMA produced NaN/inf predictions - skipping")
                return None, None
            metrics = evaluate_metrics(val[TARGET_COL].values, preds)
            return results, metrics
        except Exception as e:
            print(f"SARIMA failed: {e}")
            return None, None

    def train_prophet(self, train, val):
        try:
            # Hyperparameter tuning for Prophet
            def objective_prophet(trial):
                changepoint_prior = trial.suggest_float('changepoint_prior_scale', 0.001, 0.5, log=True)
                seasonality_prior = trial.suggest_float('seasonality_prior_scale', 0.001, 10.0, log=True)
                seasonality_mode = trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative'])
                
                m = Prophet(
                    changepoint_prior_scale=changepoint_prior, 
                    seasonality_prior_scale=seasonality_prior,
                    seasonality_mode=seasonality_mode,
                    yearly_seasonality='auto',  # type: ignore
                    weekly_seasonality='auto',  # type: ignore
                    daily_seasonality=False,  # type: ignore
                    interval_width=0.95
                )
                df_prophet = train[[DATE_COL, TARGET_COL]].rename(columns={DATE_COL: 'ds', TARGET_COL: 'y'})
                m.fit(df_prophet)
                future = m.make_future_dataframe(periods=len(val), freq='W-MON')
                forecast = m.predict(future)
                preds = forecast.iloc[-len(val):]['yhat'].values
                # Validate predictions
                if np.isnan(preds).any() or np.isinf(preds).any():
                    return float('inf')
                metrics = evaluate_metrics(val[TARGET_COL].values, preds)
                return metrics['RMSE']
            
            # Optimize Prophet hyperparameters
            study = optuna.create_study(direction='minimize')
            study.optimize(objective_prophet, n_trials=15, show_progress_bar=False)
            
            # Train final model with best parameters
            best_params = study.best_params
            m = Prophet(
                changepoint_prior_scale=best_params['changepoint_prior_scale'],
                seasonality_prior_scale=best_params['seasonality_prior_scale'],
                seasonality_mode=best_params['seasonality_mode'],
                yearly_seasonality='auto',  # type: ignore
                weekly_seasonality='auto',  # type: ignore
                daily_seasonality=False,  # type: ignore
                interval_width=0.95
            )
            df_prophet = train[[DATE_COL, TARGET_COL]].rename(columns={DATE_COL: 'ds', TARGET_COL: 'y'})
            m.fit(df_prophet)
            future = m.make_future_dataframe(periods=len(val), freq='W-MON')
            forecast = m.predict(future)
            preds = forecast.iloc[-len(val):]['yhat'].values
            # Validate predictions
            if np.isnan(preds).any() or np.isinf(preds).any():
                print(f"Prophet produced NaN/inf predictions - skipping")
                return None, None
            metrics = evaluate_metrics(val[TARGET_COL].values, preds)
            return m, metrics
        except Exception as e:
            print(f"Prophet failed: {e}")
            return None, None

    def train_xgboost(self, train, val):
        try:
            features = ['day_of_week', 'month', 'week_of_year', 'month_sin', 'month_cos', 'week_sin', 'week_cos', 'is_holiday', 'lag_1', 'lag_7', 'lag_30', 'rolling_mean_7', 'rolling_std_7']
            train_clean = train.dropna(subset=features)
            val_clean = val.dropna(subset=features)
            if len(train_clean) == 0 or len(val_clean) == 0: return None, None

            xgb_device = 'cuda' if (HAS_TORCH and torch.cuda.is_available()) else 'cpu'

            def objective(trial):
                param = {
                    'n_estimators': trial.suggest_int('n_estimators', 200, 500),
                    'max_depth': trial.suggest_int('max_depth', 2, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                    'subsample': trial.suggest_float('subsample', 0.5, 0.95),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
                    'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 0.95),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1.0, log=True),
                    'min_child_weight': trial.suggest_float('min_child_weight', 0.5, 8.0),
                    'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                    'tree_method': 'hist',
                    'device': xgb_device,
                    'random_state': 42
                }
                model = XGBRegressor(**param)
                model.fit(train_clean[features], train_clean[TARGET_COL], verbose=False)
                preds = model.predict(val_clean[features])
                # Validate predictions
                if np.isnan(preds).any() or np.isinf(preds).any() or len(preds) != len(val_clean):
                    return float('inf')
                metrics = evaluate_metrics(val_clean[TARGET_COL].values, preds)
                return metrics['RMSE']

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=25, show_progress_bar=False)  # Increased from 10 to 25 trials
            
            best_params = dict(study.best_params)
            best_params.update({
                'tree_method': 'hist',
                'device': xgb_device,
                'random_state': 42,
            })
            best_model = XGBRegressor(**best_params)
            best_model.fit(train_clean[features], train_clean[TARGET_COL], verbose=False)
            preds = best_model.predict(val_clean[features])
            # Validate predictions
            if np.isnan(preds).any() or np.isinf(preds).any():
                print(f"XGBoost produced NaN/inf predictions - skipping")
                return None, None
            metrics = evaluate_metrics(val_clean[TARGET_COL].values, preds)
            return best_model, metrics
        except Exception as e:
            print(f"XGBoost failed: {e}")
            return None, None

    def train_lstm(self, train, val):
        if not HAS_TORCH or LSTMModelClass is None:
            return None, None
        try:
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            seq_length = 30
            
            scaler.fit(train[TARGET_COL].values.reshape(-1, 1))
            scaled_train = scaler.transform(train[TARGET_COL].values.reshape(-1, 1)).flatten()
            scaled_val = scaler.transform(val[TARGET_COL].values.reshape(-1, 1)).flatten()
            if len(scaled_train) <= seq_length: return None, None
            X_train, y_train = create_sequences(scaled_train, seq_length)
            X_train = torch.FloatTensor(X_train).unsqueeze(-1).to(TORCH_DEVICE)
            y_train = torch.FloatTensor(y_train).unsqueeze(-1).to(TORCH_DEVICE)

            def objective(trial):
                hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128])
                lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
                dropout_rate = trial.suggest_float('dropout', 0.0, 0.3)
                
                model = LSTMModelClass(1, hidden_dim, 1).to(TORCH_DEVICE)
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                
                model.train()
                best_val_loss = float('inf')
                patience = 10
                no_improve = 0
                
                for epoch in range(100):
                    optimizer.zero_grad()
                    output = model(X_train)
                    loss = criterion(output, y_train)
                    loss.backward()
                    optimizer.step()
                    
                    # Early stopping on validation
                    if epoch % 10 == 0:
                        model.eval()
                        inputs = scaled_train[-seq_length:]
                        X_val = torch.FloatTensor(inputs).view(1, seq_length, 1).to(TORCH_DEVICE)
                        with torch.no_grad():
                            val_pred = model(X_val).item()
                        actual = scaled_val[0]
                        val_loss = (val_pred - actual)**2
                        
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            no_improve = 0
                        else:
                            no_improve += 1
                        
                        if no_improve >= patience:
                            break
                        model.train()
                
                return best_val_loss

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=10, show_progress_bar=False)
            
            best_h = study.best_params['hidden_dim']
            best_lr = study.best_params['lr']
            
            model = LSTMModelClass(1, best_h, 1).to(TORCH_DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
            criterion = nn.MSELoss()
            
            model.train()
            best_val_loss = float('inf')
            patience = 15
            no_improve = 0
            
            for epoch in range(200):
                optimizer.zero_grad()
                output = model(X_train)
                loss = criterion(output, y_train)
                loss.backward()
                optimizer.step()
                
                if epoch % 10 == 0:
                    model.eval()
                    inputs = scaled_train[-seq_length:]
                    X_val = torch.FloatTensor(inputs).view(1, seq_length, 1).to(TORCH_DEVICE)
                    with torch.no_grad():
                        val_pred = model(X_val).item()
                    actual = scaled_val[0]
                    val_loss = (val_pred - actual)**2
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        no_improve = 0
                    else:
                        no_improve += 1
                    
                    if no_improve >= patience:
                        break
                    model.train()

            model.eval()
            inputs = np.concatenate((scaled_train[-seq_length:], scaled_val))
            def create_sequences_local(data, slen):
                xs = []
                for i in range(len(data) - slen):
                    xs.append(data[i:(i + slen)])
                return np.array(xs)
            
            X_test = torch.FloatTensor(create_sequences_local(inputs, seq_length)).unsqueeze(-1).to(TORCH_DEVICE)
            with torch.no_grad():
                scaled_preds = model(X_test).squeeze().detach().cpu().numpy()
            
            if scaled_preds.ndim == 0: scaled_preds = np.array([scaled_preds.item()])
            preds = scaler.inverse_transform(scaled_preds.reshape(-1, 1)).flatten()
            # Validate predictions
            if np.isnan(preds).any() or np.isinf(preds).any() or len(preds) < len(val):
                print(f"LSTM produced NaN/inf or wrong-sized predictions - skipping")
                return None, None
            preds = preds[-len(val):]  # Ensure exact length match
            metrics = evaluate_metrics(val[TARGET_COL].values, preds)
            model.__dict__["scaler"] = scaler
            model.__dict__["seq_length"] = seq_length
            return model, metrics
        except Exception as e:
            print(f"LSTM failed: {e}")
            return None, None

    def train_and_evaluate_all(self):
        state_best_models = {}
        for state, group in self.df.groupby(STATE_COL):
            print(f"Training for State: {state}")
            train, val, test = self.train_val_test_split_ts(group)
            results = {
                'SARIMA': self.train_sarima(train, val),
                'Prophet': self.train_prophet(train, val),
                'XGBoost': self.train_xgboost(train, val),
                'LSTM': self.train_lstm(train, val)
            }
            best_m, best_metrics, best_name, best_score = None, None, "", float('inf')
            for name, (model, metrics) in results.items():
                if metrics and metrics['RMSE'] < best_score:
                    best_score, best_m, best_name, best_metrics = metrics['RMSE'], model, name, metrics
            state_best_models[state] = {'model': best_m, 'name': best_name, 'metrics': best_metrics, 'score': best_score, 'train_rows': len(train), 'val_rows': len(val), 'test_rows': len(test)}
            print(f"Best model for {state}: {best_name} with RMSE: {best_score}")
        return state_best_models

def create_sequences(data, seq_length=30):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:(i + seq_length)])
        ys.append(data[i + seq_length])
    return np.array(xs), np.array(ys)

def save_models(state_best_models):
    if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)
    joblib.dump(state_best_models, os.path.join(MODELS_DIR, 'state_best_models.pkl'))

def load_best_models():
    return joblib.load(os.path.join(MODELS_DIR, 'state_best_models.pkl'))
