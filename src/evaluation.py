import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate_metrics(y_true, y_pred):
    # Ensure numpy arrays and convert pandas Series
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Check for NaN or inf
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("NaN values found in y_true or y_pred")
    if np.isinf(y_true).any() or np.isinf(y_pred).any():
        raise ValueError("Inf values found in y_true or y_pred")
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = calculate_mape(y_true, y_pred)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }
