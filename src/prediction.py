import pandas as pd
import numpy as np
import holidays
from src.config import FORECAST_WEEKS, DATE_COL, TARGET_COL, STATE_COL

def _future_weekly_dates(last_date, periods):
    return pd.date_range(start=last_date + pd.Timedelta(days=7), periods=periods, freq='W-MON')


def _recursive_xgboost_forecast(model, history, future_dates):
    feature_names = ['day_of_week', 'month', 'week_of_year', 'month_sin', 'month_cos', 'week_sin', 'week_cos', 'is_holiday', 'lag_1', 'lag_7', 'lag_30', 'rolling_mean_7', 'rolling_std_7']
    years = range(future_dates[0].year, future_dates[-1].year + 1)
    us_holidays = holidays.US(years=years)
    values = list(history)
    predictions = []

    for forecast_date in future_dates:
        lag_1 = values[-1]
        lag_7 = values[-7] if len(values) >= 7 else values[-1]
        lag_30 = values[-30] if len(values) >= 30 else values[-1]
        rolling_window = values[-7:] if len(values) >= 7 else values
        rolling_mean_7 = float(np.mean(rolling_window))
        rolling_std_7 = float(np.std(rolling_window, ddof=1)) if len(rolling_window) > 1 else 0.0

        feature_row = pd.DataFrame([{
            'day_of_week': forecast_date.dayofweek,
            'month': forecast_date.month,
            'week_of_year': int(forecast_date.isocalendar().week),
            'month_sin': float(np.sin(2 * np.pi * forecast_date.month / 12)),
            'month_cos': float(np.cos(2 * np.pi * forecast_date.month / 12)),
            'week_sin': float(np.sin(2 * np.pi * int(forecast_date.isocalendar().week) / 52)),
            'week_cos': float(np.cos(2 * np.pi * int(forecast_date.isocalendar().week) / 52)),
            'is_holiday': int(forecast_date.normalize() in us_holidays),
            'lag_1': lag_1,
            'lag_7': lag_7,
            'lag_30': lag_30,
            'rolling_mean_7': rolling_mean_7,
            'rolling_std_7': rolling_std_7,
        }])

        prediction = float(model.predict(feature_row[feature_names])[0])
        predictions.append(prediction)
        values.append(prediction)

    return predictions


def _recursive_lstm_forecast(model, history, future_dates):
    import torch

    scaler = getattr(model, 'scaler', None)
    seq_length = getattr(model, 'seq_length', 30)
    if scaler is None:
        raise ValueError('LSTM model is missing its fitted scaler.')

    device = next(model.parameters()).device
    scaled_history = scaler.transform(np.array(history).reshape(-1, 1)).flatten().tolist()
    predictions = []

    for _ in future_dates:
        sequence = np.array(scaled_history[-seq_length:], dtype=np.float32).reshape(1, seq_length, 1)
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32, device=device)
        with torch.no_grad():
            scaled_prediction = float(model(sequence_tensor).item())
        prediction = float(scaler.inverse_transform(np.array([[scaled_prediction]])).ravel()[0])
        predictions.append(prediction)
        scaled_history.append(scaled_prediction)

    return predictions


def generate_forecasts(state_best_models, df):
    """
    Generate 8-week forecasts for each state using their best model.
    """
    all_forecasts = {}
    
    for state, info in state_best_models.items():
        model = info['model']
        model_name = info['name']
        
        state_data = df[df[STATE_COL] == state].sort_values(DATE_COL)
        last_date = state_data[DATE_COL].max()
        future_dates = _future_weekly_dates(last_date, FORECAST_WEEKS)
        
        if model_name == 'SARIMA':
            preds = model.forecast(FORECAST_WEEKS)
        elif model_name == 'Prophet':
            future = model.make_future_dataframe(periods=FORECAST_WEEKS, freq='W-MON')
            forecast = model.predict(future)
            preds = forecast.iloc[-FORECAST_WEEKS:]['yhat'].values
        elif model_name == 'XGBoost':
            preds = _recursive_xgboost_forecast(model, state_data[TARGET_COL].tolist(), future_dates)
        elif model_name == 'LSTM':
            preds = _recursive_lstm_forecast(model, state_data[TARGET_COL].tolist(), future_dates)
        else:
            preds = [state_data[TARGET_COL].iloc[-1]] * FORECAST_WEEKS
            
        all_forecasts[state] = {
            'dates': [d.strftime('%Y-%m-%d') for d in future_dates],
            'forecast': [float(p) for p in preds]
        }
        
    return all_forecasts
