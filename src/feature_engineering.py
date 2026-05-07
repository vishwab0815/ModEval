import pandas as pd
import holidays
import numpy as np
from src.config import DATE_COL, TARGET_COL, STATE_COL

def create_features(df):
    """
    Create lag features, rolling features, temporal features and holiday flags.
    """
    df = df.copy()
    
    # Sort just in case
    df = df.sort_values(by=[STATE_COL, DATE_COL])
    
    # 1. Temporal Features
    df['day_of_week'] = df[DATE_COL].dt.dayofweek
    df['month'] = df[DATE_COL].dt.month
    df['week_of_year'] = df[DATE_COL].dt.isocalendar().week.astype(int)
    
    # Cyclical encodings help tree-based models and preserve seasonality smoothly.
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    
    # 2. Holiday Flag
    # US state sales data should use a US holiday calendar.
    years = range(df[DATE_COL].dt.year.min(), df[DATE_COL].dt.year.max() + 1)
    us_holidays = holidays.US(years=years)
    df['is_holiday'] = df[DATE_COL].dt.normalize().isin(us_holidays).astype(int)
    
    # 3. Lag Features (t-1, t-7, t-30)
    # Note: These are based on the row order. If data is daily, these are days.
    for lag in [1, 7, 30]:
        df[f'lag_{lag}'] = df.groupby(STATE_COL)[TARGET_COL].shift(lag)
    
    # 4. Rolling Features
    # 7-day rolling mean and std
    df['rolling_mean_7'] = df.groupby(STATE_COL)[TARGET_COL].transform(lambda x: x.shift(1).rolling(window=7).mean())
    df['rolling_std_7'] = df.groupby(STATE_COL)[TARGET_COL].transform(lambda x: x.shift(1).rolling(window=7).std())
    
    # Drop rows with NaN resulting from lags/rolling (optional, usually done before training)
    # df = df.dropna().reset_index(drop=True)
    
    return df
