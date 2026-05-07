from src.preprocessing import load_and_preprocess
from src.feature_engineering import create_features
from src.model_training import TimeSeriesTrainer
from src.config import DATA_PATH
import pandas as pd

def model_comparison():
    df = load_and_preprocess(DATA_PATH)
    df = create_features(df)
    state = 'Alabama'
    group = df[df['State'] == state]
    trainer = TimeSeriesTrainer(df)
    train, val, test = trainer.train_val_test_split_ts(group)
    
    results = {
        'SARIMA': trainer.train_sarima(train, val),
        'Prophet': trainer.train_prophet(train, val),
        'XGBoost': trainer.train_xgboost(train, val),
        'LSTM': trainer.train_lstm(train, val)
    }
    
    print("\nMODEL COMPARISON REPORT (State: Alabama)")
    print("-" * 78)
    print(f"{'Model':<10} | {'RMSE':>14} | {'MAE':>14} | {'MAPE %':>10} | {'Approx. Fit %':>13}")
    print("-" * 78)
    best_name = None
    best_metrics = None
    for name, (model, metrics) in results.items():
        if metrics:
            accuracy = max(0.0, 100.0 - float(metrics['MAPE']))
            print(
                f"{name:<10} | {float(metrics['RMSE']):>14,.2f} | {float(metrics['MAE']):>14,.2f} | "
                f"{float(metrics['MAPE']):>10.2f} | {accuracy:>13.2f}"
            )
            if best_metrics is None or metrics['RMSE'] < best_metrics['RMSE']:
                best_name = name
                best_metrics = metrics
        else:
            print(f"{name:<10} | {'FAILED':>14} | {'FAILED':>14} | {'FAILED':>10} | {'FAILED':>13}")

    if best_name is not None:
        print("-" * 78)
        print(f"Best validation fit: {best_name} with RMSE {float(best_metrics['RMSE']):,.2f} and MAPE {float(best_metrics['MAPE']):.2f}%")
        
        # Fitting sanity check: show actual vs fitted values for the validation slice.
        if results[best_name][0] is not None:
            print("\nValidation fit sample:")
            print("Actual vs Prediction")
            if best_name == 'SARIMA':
                preds = results[best_name][0].forecast(len(val))
            elif best_name == 'Prophet':
                future = results[best_name][0].make_future_dataframe(periods=len(val), freq='W-MON')
                forecast = results[best_name][0].predict(future)
                preds = forecast.iloc[-len(val):]['yhat'].values
            elif best_name == 'XGBoost':
                features = ['day_of_week', 'month', 'week_of_year', 'is_holiday', 'lag_1', 'lag_7', 'lag_30', 'rolling_mean_7', 'rolling_std_7']
                val_clean = val.dropna(subset=features)
                preds = results[best_name][0].predict(val_clean[features]) if len(val_clean) else []
                print(val_clean[["Date", "Total"]].head(5).to_string(index=False))
            elif best_name == 'LSTM':
                print(val[["Date", "Total"]].head(5).to_string(index=False))
                preds = []
            else:
                preds = []

            if len(preds):
                sample_actual = val['Total'].iloc[:min(5, len(preds))].values
                sample_pred = pd.Series(preds).iloc[:min(5, len(preds))].values
                fit_df = pd.DataFrame({'Actual': sample_actual, 'Predicted': sample_pred})
                print(fit_df.to_string(index=False))

if __name__ == "__main__":
    model_comparison()
