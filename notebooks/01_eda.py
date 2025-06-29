
"""Exploratory Data Analysis for Tesla Stock Forecasting
- Loading the historical Tesla stock data
- Inspecting for nulls, basic statistics
- Preliminary EDA: time trends, summary, correlations
"""

# 1. Imports
import pandas as pd
import numpy as np

# Visualization libraries (we’ll use these more in Notebook 02)
import matplotlib.pyplot as plt
import seaborn as sns

# For time series date handling
from datetime import datetime

# 2. Load Dataset
def load_data(filepath: str) -> pd.DataFrame:
    
    df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
    # Sort by date in ascending order
    df.sort_index(inplace=True)
    return df


raw_df = load_data('data/raw/Tasla_Stock_Updated_V2.csv')



# 3. Basic Inspection
def inspect_data(df: pd.DataFrame):
    """
    Prints basic info about the DataFrame:
    - head, tail
    - shape
    - missing values
    - summary statistics
    """
    print("First 5 rows:\n", df.head(), "\n")
    print("Last 5 rows:\n", df.tail(), "\n")
    print("Shape:", df.shape, "\n")
    print("Missing values per column:\n", df.isnull().sum(), "\n")
    print("Summary statistics:\n", df.describe(), "\n")

inspect_data(raw_df)





# 4. Preliminary EDA
def eda_overview(df: pd.DataFrame):
    """
    Generates simple EDA plots:
    - Time series of Close price
    - Histogram of daily returns
    - Heatmap of feature correlations
    """
    # 4.1 Time series plot of Close Price
    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df['Close'], label='Close Price')
    plt.title('Tesla Closing Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()

    # 4.2 Daily returns histogram
    df['Daily_Return'] = df['Close'].pct_change()
    plt.figure(figsize=(8, 4))
    sns.histplot(df['Daily_Return'].dropna(), bins=50, kde=True)
    plt.title('Distribution of Daily Returns')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.show()

    # 4.3 Correlation heatmap
    corr = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return']].corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.show()

# Run the EDA overview
eda_overview(raw_df)
