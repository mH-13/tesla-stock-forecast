"""Notebook 03: Model Definitions, Training & Evaluation

1. Loading engineered features and train/test split
2. Defining baseline & forecasting models:
    • Naïve (last-value) & simple moving-average baseline
    • ARIMA
    • XGBoost
    • LSTM
3. Training each model
4. Evaluating with MAE, RMSE, R²
5. Summarizing & comparing performance
"""

# 1. Imports
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor

import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

# 2. Load data & split
df = pd.read_csv('data/processed/tesla_features.csv',
                parse_dates=['Date'], index_col='Date')
split_idx = int(len(df) * 0.8)
train, test = df.iloc[:split_idx], df.iloc[split_idx:]

X_cols = [c for c in df.columns if c not in ['Close','Open','High','Low','Volume']]
y_col = 'Close'

X_train, y_train = train[X_cols], train[y_col]
X_test,  y_test  = test[X_cols],  test[y_col]


# 3. Evaluation helper
def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"{name} → MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
    return dict(model=name, MAE=mae, RMSE=rmse, R2=r2)


results = []


# 4. Baseline models
# 4.1 DummyRegressor (last observed value)
#dummy_last = DummyRegressor(strategy='mean') 

# mean strategy for time series
# Note: 'mean' is used here as a proxy for the last observed value in time
# series context, as DummyRegressor does not have a 'last' strategy.
# If you want to use the last value directly, you can use 'constant' with the
# last value from the training set. then it would be:
# dummy_last = DummyRegressor(strategy='constant', constant=y_train.iloc[-1])
#best apporach is to use 'constant' with the last value from the training set.
dummy_last = DummyRegressor(strategy='constant', constant=y_train.iloc[-1])


dummy_last.fit(X_train, y_train)
y_pred = dummy_last.predict(X_test)
results.append(evaluate("DummyLast", y_test, y_pred))

# 4.2 Simple moving-average baseline (5-day MA)
y_pred_ma5 = test['Close'].shift(1).rolling(window=5).mean().fillna(method='bfill')
results.append(evaluate("MA5_Baseline", y_test, y_pred_ma5))


# 5. ARIMA
# model on the 'Close' series only
arima_order = (5,1,0) 

# tune as needed like: auto_arima(train['Close'])
# Note: ARIMA order (p,d,q) can be tuned using auto_arima from pmdarima
# but here we use a fixed order for simplicity. best practice is to use
# auto_arima to find the best order based on AIC/BIC. that will be like: 
# from pmdarima import auto_arima
# arima_order = auto_arima(train['Close'], seasonal=False, stepwise=True).
# arima_order = auto_arima(train['Close'], seasonal=False, stepwise=True).
# arima_order = arima_model.order  # get the best order from auto_arima
# arima_order = (5,1,0)  # for example, you can use (5,1,0) as a starting point
# Note: ARIMA requires the series to be stationary, so ensure you have differenced it
# appropriately. Here we assume the series is already stationary or has been differenced. best approach is to use
# auto_arima to find the best order based on AIC/BIC.



arima_model = ARIMA(train['Close'], order=arima_order).fit()
y_pred_arima = arima_model.forecast(steps=len(test))
results.append(evaluate("ARIMA(5,1,0)", y_test, y_pred_arima))


# 6. XGBoost
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False)
y_pred_xgb = xgb_model.predict(X_test)
results.append(evaluate("XGBoost", y_test, y_pred_xgb))


# 7. LSTM
# Prepare 3D input: [samples, timesteps=1, features]
X_train_l = X_train.values.reshape(-1, 1, X_train.shape[1])
X_test_l  = X_test.values.reshape(-1, 1, X_test.shape[1])

lstm = Sequential([
    LSTM(50, input_shape=(1, X_train.shape[1]), return_sequences=False),
    Dense(1)
])
lstm.compile(optimizer='adam', loss='mse')
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = lstm.fit(
    X_train_l, y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=16,
    callbacks=[es],
    verbose=1
)
y_pred_lstm = lstm.predict(X_test_l).flatten()
results.append(evaluate("LSTM", y_test, y_pred_lstm))


# 8. Summarize results
results_df = pd.DataFrame(results).set_index('model')
print("\n=== Model Comparison ===")
print(results_df)

# 9. Plot actual vs predicted for best two
best_two = results_df.sort_values('RMSE').head(2).index.tolist()
plt.figure(figsize=(12,5))
plt.plot(test.index, y_test, label='Actual', color='black')
for model in best_two:
    yhat = {
        "DummyLast": y_pred,
        "MA5_Baseline": y_pred_ma5,
        "ARIMA(5,1,0)": y_pred_arima,
        "XGBoost": y_pred_xgb,
        "LSTM": y_pred_lstm
    }[model]
    
    plt.plot(test.index, yhat, label=model)
plt.legend(); plt.title('Actual vs Predicted (Top 2 Models)')
plt.show()
