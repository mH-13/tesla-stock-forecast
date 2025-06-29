"""Visualization, Feature Engineering, and Train/Test Split
- Seasonality & rolling-statistics plots
- Creation of technical features:
    • Moving averages
    • Volatility
    • Monthly returns
- Train/test split for modeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Reload cleaned data from previous step
df = pd.read_csv('data/processed/tesla_cleaned.csv',
                parse_dates=['Date'], index_col='Date') 

# 2. Seasonality & Rolling Statistics
def plot_rolling_stats(data, window=30):
    
    rol_mean = data['Close'].rolling(window=window).mean()
    rol_std  = data['Close'].rolling(window=window).std()
    
    
    plt.figure(figsize=(12, 4))
    plt.plot(data.index, data['Close'], label='Close')
    plt.plot(data.index, rol_mean,  label=f'{window}-day MA')
    plt.plot(data.index, rol_std,   label=f'{window}-day STD')
    plt.title(f'Rolling Mean & Std (window={window})')
    plt.legend()
    plt.show()
    

plot_rolling_stats(df, window=30)
plot_rolling_stats(df, window=90)




# 3. Feature Engineering
def add_technical_indicators(data):
    
    """Adding  technical features:
    - MA5, MA10, MA20
    - Volatility (std over 5/10/20 days)
    - Daily % returns
    - Monthly returns (shifted)
    """
    df_feat = data.copy()
    
    # Moving Averages
    for w in [5, 10, 20]:
        df_feat[f'MA_{w}'] = df_feat['Close'].rolling(window=w).mean()
    
    # Rolling Volatility
    for w in [5, 10, 20]:
        df_feat[f'Volatility_{w}'] = df_feat['Close'].rolling(window=w).std()
    
    # Daily Returns
    df_feat['Ret_1D'] = df_feat['Close'].pct_change()
    
    # Monthly Return: (Close_t / Close_{t-30d}) - 1
    df_feat['Ret_30D'] = df_feat['Close'].pct_change(periods=30)
    
    # Drop initial NaNs
    df_feat.dropna(inplace=True)
    return df_feat

df_feats = add_technical_indicators(df)
print("Features added. New shape:", df_feats.shape)



# 4. Train/Test Split
from sklearn.model_selection import TimeSeriesSplit

# We’ll reserve last 20% of data as “test”, rest as “train”
split_idx = int(len(df_feats) * 0.8)
train_df = df_feats.iloc[:split_idx]
test_df  = df_feats.iloc[split_idx:]

X_cols = [c for c in df_feats.columns if c not in ['Close', 'Open', 'High', 'Low', 'Volume']]
y_col  = 'Close'

X_train, y_train = train_df[X_cols], train_df[y_col]
X_test,  y_test  = test_df[X_cols],  test_df[y_col]

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
