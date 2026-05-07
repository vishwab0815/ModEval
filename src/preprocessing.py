import pandas as pd
import numpy as np
from src.config import DATE_COL, TARGET_COL, STATE_COL, CATEGORY_COL

def _parse_mixed_dates(date_series):
    """Parse slash-separated MM/DD/YYYY and hyphen-separated DD-MM-YYYY values."""
    raw = date_series.astype(str).str.strip()
    parsed = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")

    slash_mask = raw.str.contains("/", na=False)
    hyphen_mask = raw.str.contains("-", na=False)

    if slash_mask.any():
        parsed.loc[slash_mask] = pd.to_datetime(raw.loc[slash_mask], format="%m/%d/%Y", errors="coerce")
    if hyphen_mask.any():
        parsed.loc[hyphen_mask] = pd.to_datetime(raw.loc[hyphen_mask], format="%d-%m-%Y", errors="coerce")

    remaining_mask = parsed.isna()
    if remaining_mask.any():
        parsed.loc[remaining_mask] = pd.to_datetime(raw.loc[remaining_mask], errors="coerce", dayfirst=False)

    return parsed


def _print_data_audit(df, raw_rows, parsed_date_valid, target_valid):
    """Print a compact data audit for manual inspection."""
    print("\n=== DATA AUDIT ===")
    print(f"Raw rows: {raw_rows:,}")
    print(f"Clean rows: {len(df):,}")
    print(f"States: {df[STATE_COL].nunique()}")
    print(f"Parsed date success: {parsed_date_valid:,} / {raw_rows:,}")
    print(f"Valid target values after cleaning: {target_valid:,} / {raw_rows:,}")
    print(f"Date range: {df[DATE_COL].min().date()} -> {df[DATE_COL].max().date()}")
    print(f"Target missing values: {df[TARGET_COL].isna().sum()}")
    print("Per-state observations (min/median/max): "
          f"{int(df.groupby(STATE_COL).size().min())} / "
          f"{int(df.groupby(STATE_COL).size().median())} / "
          f"{int(df.groupby(STATE_COL).size().max())}")
    print("Sample cleaned rows:")
    print(df.head(5).to_string(index=False))
    print("=== END DATA AUDIT ===\n")


def load_and_preprocess(file_path, verbose=False):
    """
    Read CSV, clean data, normalize dates, and resample to weekly frequency.
    """
    try:
        df = pd.read_csv(file_path)
        raw_rows = len(df)
        # Clean column names for any hidden spaces
        df.columns = [c.strip() for c in df.columns]
        
        # Clean the Total column (remove commas and convert to float)
        df[TARGET_COL] = df[TARGET_COL].astype(str).str.replace(',', '').str.strip()
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
        
        # Convert Date column to datetime using explicit parsing for mixed formats.
        df[DATE_COL] = _parse_mixed_dates(df[DATE_COL])
        parsed_date_valid = df[DATE_COL].notna().sum()
        
        # Drop rows with critical missing data
        df = df.dropna(subset=[DATE_COL, TARGET_COL])
        target_valid = len(df)
        
        # Sort chronologically
        df = df.sort_values(by=[STATE_COL, DATE_COL])
        
        processed_dfs = []
        for state, group in df.groupby(STATE_COL):
            category_value = group[CATEGORY_COL].mode().iloc[0] if CATEGORY_COL in group.columns and not group[CATEGORY_COL].mode().empty else None
            # Normalize each state to a weekly Monday cadence.
            group_resampled = (
                group.set_index(DATE_COL)
                .resample('W-MON')
                .sum(min_count=1)
                .reset_index()
            )
            
            group_resampled[STATE_COL] = state
            if category_value is not None:
                group_resampled[CATEGORY_COL] = category_value
            
            # Fill only missing weeks created by resampling; keep genuine zeros intact.
            group_resampled[TARGET_COL] = (
                group_resampled[TARGET_COL]
                .interpolate(method='linear')
                .ffill()
                .bfill()
            )
            
            processed_dfs.append(group_resampled)
        
        df = pd.concat(processed_dfs).reset_index(drop=True)
        if verbose:
            _print_data_audit(df, raw_rows, parsed_date_valid, target_valid)
        return df
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        raise e

if __name__ == "__main__":
    from src.config import DATA_PATH
    df = load_and_preprocess(DATA_PATH, verbose=True)
    print(f"Dataset shape: {df.shape}")
    print(df.head())
    print(f"Non-zero values count: {(df[TARGET_COL] > 0).sum()}")
