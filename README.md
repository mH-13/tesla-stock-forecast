# Tesla Stock Price Forecasting

A professional time series forecasting pipeline to predict Tesla’s next-month closing stock price using classical, machine learning, and deep learning models.

 

## 📁 Folder Structure

```text
├── data/               # raw & processed CSVs
├── notebooks/          # EDA & prototyping
├── src/                # core modules
│   ├── data/           # loading utilities
│   ├── features/       # feature engineering
│   ├── models/         # training & prediction
│   └── visualization/  # plotting helpers
├── tests/              # pytest unit tests
├── .github/            # workflows & PR templates
├── .vscode/            # editor settings
├── requirements.txt
└── Makefile

```


## Updated Project Structure So Far: 

```text
Tesla-S/
├─ data/
│  ├─ raw/
│  │  └─ Tasla_Stock_Updated_V2.csv
│  └─ processed/
│     ├─ tesla_cleaned.csv
│     └─ tesla_features.csv
├─ models/
│  ├─ best_params.json
│  ├─ xgb_pipeline.joblib
│  └─ arima_model.joblib
├─ notebooks/
│  ├─ 01_eda.py
│  ├─ 02_feature_engineering_and_split.py
│  ├─ 03_modeling_and_evaluation.py
│  ├─ 04_hyperparameter_tuning.py
│  ├─ 05_walk_forward_validation.py
│  └─ 06_model_packaging.py
└─ reports/
   └─ final_report.md
```

## Current Findings: Model Comparison Results

### 📊 Performance Metrics Summary

| Model          | MAE (USD) | RMSE (USD) | R² Score   | Rank |
|----------------|-----------|------------|------------|------|
| MA5_Baseline   | 9.84      | 12.38      | 0.938      | 🥇   |
| XGBoost        | 11.28     | 14.71      | 0.912      | 🥈   |
| DummyLast      | 157.14    | 164.78     | -10.045    | 🏳️   |
| ARIMA(5,1,0)   | 130.18    | 139.19     | -6.882     | ❌   |
| LSTM           | 121.62    | 130.88     | -5.969     | ❌   |

## 🔍 Key Observations

### 🎯 Top Performers
1. **MA5_Baseline (5-Day Moving Average)**
   - Best in all metrics (Lowest MAE/RMSE, Highest R²)
   - Suggests strong short-term price momentum
   - Implementation: `y_pred = data.rolling(5).mean().shift(1)`

2. **XGBoost**  
   - Close second with 14.5% higher RMSE than MA5
   - Shows ML can approximate technical indicators

### ⚠️ Underperformers
- **ARIMA & LSTM**:
  - Negative R² indicates worse than mean prediction
  - Possible issues: 
    - Insufficient differencing (ARIMA)
    - Need for hyperparameter tuning (LSTM)
    - Lookback window mismatch

- **DummyLast (Naive Baseline)**:
  - Predicts last observed value
  - Serves as absolute minimum benchmark

## 📈 Interpretation Guide

### Metric Definitions
- **MAE (Mean Absolute Error)**: Average $ prediction error
- **RMSE (Root Mean Squared Error)**: Penalizes large errors more
- **R² (R-Squared)**: 
  - 1 = Perfect prediction 
  - 0 = Same as predicting mean
  - <0 = Worse than simple mean

