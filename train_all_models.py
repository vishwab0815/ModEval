"""
Comprehensive training script for all 43 US states with GPU acceleration.
Includes detailed logging and results summary.
"""
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from src.preprocessing import load_and_preprocess
from src.feature_engineering import create_features
from src.model_training import TimeSeriesTrainer, save_models
from src.config import DATA_PATH

print("=" * 80)
print("COMPREHENSIVE FORECASTING MODEL TRAINING - ALL 43 STATES")
print("=" * 80)

start_time = datetime.now()
print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

# Load and prepare data
print("\n[1/4] Loading and preprocessing data...")
df = load_and_preprocess(DATA_PATH)
df = create_features(df)
print(f"✓ Loaded {len(df):,} rows across {df['State'].nunique()} states")

# Train models
print("\n[2/4] Training all models for 43 states...")
print("-" * 80)
trainer = TimeSeriesTrainer(df)
state_best_models = trainer.train_and_evaluate_all()

# Save models
print("\n[3/4] Saving trained models...")
save_models(state_best_models)
print("✓ Models saved to models/state_best_models.pkl")

# Generate summary
print("\n[4/4] Generating results summary...")
print("=" * 80)
print("FINAL RESULTS - ALL 43 STATES")
print("=" * 80)

results_list = []
for state, info in sorted(state_best_models.items()):
    if info['model'] is not None:
        rmse = info['score']
        accuracy = 100 - info['metrics']['MAPE']
        mae = info['metrics']['MAE']
        mape = info['metrics']['MAPE']
        results_list.append({
            'State': state,
            'Model': info['name'],
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'Accuracy': accuracy,
            'TrainRows': info['train_rows'],
            'ValRows': info['val_rows'],
        })
    else:
        print(f"⚠ {state:15} - FAILED (all models returned None)")

results_df = pd.DataFrame(results_list).sort_values('Accuracy', ascending=False)

# Print table
print("\n{:<15} {:<12} {:>12} {:>10} {:>10} {:>8}".format(
    "State", "Model", "Accuracy%", "MAPE%", "RMSE", "MAE"))
print("-" * 80)
for _, row in results_df.iterrows():
    print("{:<15} {:<12} {:>11.2f}% {:>9.2f}% {:>12.0f} {:>8.0f}".format(
        row['State'], row['Model'], row['Accuracy'], row['MAPE'], row['RMSE'], row['MAE']))

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print(f"Total states: {len(results_df)}")
print(f"Average accuracy: {results_df['Accuracy'].mean():.2f}%")
print(f"Median accuracy: {results_df['Accuracy'].median():.2f}%")
print(f"Std Dev accuracy: {results_df['Accuracy'].std():.2f}%")
print(f"Min accuracy: {results_df['Accuracy'].min():.2f}% ({results_df.loc[results_df['Accuracy'].idxmin(), 'State']})")
print(f"Max accuracy: {results_df['Accuracy'].max():.2f}% ({results_df.loc[results_df['Accuracy'].idxmax(), 'State']})")

model_counts = results_df['Model'].value_counts()
print(f"\nBest model distribution:")
for model, count in model_counts.items():
    pct = (count / len(results_df)) * 100
    print(f"  {model:<12} {count:>3} states ({pct:>5.1f}%)")

# Timing
elapsed = (datetime.now() - start_time).total_seconds()
print("\n" + "=" * 80)
print(f"Total training time: {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
print(f"Time per state: {elapsed/len(results_df):.1f} seconds")
print("=" * 80)

print("\n✅ TRAINING COMPLETE!")
print("\nNext steps:")
print("  1. Review results above for consistency")
print("  2. Run: uvicorn src.api:app --reload")
print("  3. Test API at: http://127.0.0.1:8000/docs")
