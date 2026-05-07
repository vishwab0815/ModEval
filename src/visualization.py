import matplotlib.pyplot as plt
import io
import pandas as pd
from src.config import DATE_COL, TARGET_COL, STATE_COL

def create_forecast_plot(df, state, state_best_models, forecast_data):
    """
    Generate a plot showing historical data and forecast for a state.
    """
    plt.figure(figsize=(12, 6))
    
    state_df = df[df[STATE_COL] == state].sort_values(DATE_COL)
    
    # Plot historical data
    plt.plot(state_df[DATE_COL], state_df[TARGET_COL], label='Historical Data', color='blue', alpha=0.6)
    
    # Plot forecast
    forecast_dates = pd.to_datetime(forecast_data['dates'])
    forecast_values = forecast_data['forecast']
    plt.plot(forecast_dates, forecast_values, label='8-Week Forecast', color='red', linestyle='--', marker='o')
    
    model_name = state_best_models[state]['name']
    rmse = state_best_models[state]['score']
    
    plt.title(f"Sales Forecast for {state} (Model: {model_name}, RMSE: {rmse:,.0f})")
    plt.xlabel("Date")
    plt.ylabel("Total Sales")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf
