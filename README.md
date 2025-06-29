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