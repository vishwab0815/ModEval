import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Forecasting-Case- Study.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Time series parameters
TARGET_COL = 'Total'
DATE_COL = 'Date'
STATE_COL = 'State'
CATEGORY_COL = 'Category'

FORECAST_WEEKS = 8

# Model hyperparams (simplified for now)
LAG_WEEKS = [1, 7, 30] # The prompt says t-1, t-7, t-30. If these are days, but forecasting is weekly...
# Actually, the user says "forecast the next 8 weeks". 
# Usually lag features are based on the frequency. 
# If data is daily, t-1 is yesterday, t-7 is last week, t-30 is last month.
# If data is weekly, t-1 is last week.
# I will assume daily data and aggregate to weekly or use daily lags if appropriate.
# Let's see the data first if possible.
