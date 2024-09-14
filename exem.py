# import okx.Account as Account
# import okx.Trade as trade
# api_key = '43f5df59-5e61-4d24-875e-f32c003e0430'
# secret_key = '5B1063B322635A27CF01BACE3772E0E0'
# passphrase = 'Parkwood270298)'
# flag = "1"
#
# accountAPI = Account.AccountAPI(api_key, secret_key, passphrase, False, flag)
# tradeAPI = trade.TradeAPI(api_key, secret_key, passphrase, False, flag)
# print(dir(tradeAPI))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import logging


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
df = pd.read_csv('LTC-USDT-SWAP.csv')

# Extract features from historical data
def extract_features(df):
    df['return'] = df['close'].pct_change()
    df['volatility'] = df['close'].rolling(window=10).std()
    df['momentum'] = df['close'].rolling(window=5).mean() - df['close'].rolling(window=15).mean()

    # Add features for swing highs and lows
    df['swing'] = df.apply(lambda x: is_swing(x.name, 10, df), axis=1)
    df['pointpos'] = df.apply(pointpos, axis=1)

    df.dropna(inplace=True)
    return df


# Determine swing high/low
def is_swing(candle, window, df):
    if candle - window < 0 or candle + window >= len(df):
        return 0
    swing_high = all(df.iloc[candle].high >= df.iloc[i].high for i in range(candle - window, candle + window + 1))
    swing_low = all(df.iloc[candle].low <= df.iloc[i].low for i in range(candle - window, candle + window + 1))
    return 1 if swing_high else (2 if swing_low else 0)


# Return high/low price based on swing
def pointpos(row):
    return row['low'] if row['swing'] == 2 else (row['high'] if row['swing'] == 1 else np.nan)
df['swing'] = df.apply(lambda x: is_swing(x.name, 10, df), axis=1)
def detect_structure(candle, df, backcandles=60, window=10):
    localdf = df.iloc[candle - backcandles - window:candle - window]
    highs = localdf[localdf['swing'] == 1].high.tail(2).values
    lows = localdf[localdf['swing'] == 2].low.tail(2).values
    zone_width = 0.001
    if len(highs) == 2 and df.loc[candle].close - highs.mean() > zone_width * 2:
        return 1
    if len(lows) == 2 and lows.mean() - df.loc[candle].close > zone_width * 2:
        return 2
    return 0
df['pattern_detected'] = df.apply(lambda x: detect_structure(x.name, df), axis=1)
df.to_csv('train.csv', index=False)
# Prepare data for training
# def prepare_data(df):
#     df = extract_features(df)
#
#     if len(df) < 2:
#         logging.error(f"Insufficient data for training. Available samples: {len(df)}")
#         return None, None
#
#     X = df[['return', 'volatility', 'momentum', 'swing']]
#     y = df['swing']  # Target variable: 1 for buy, 2 for sell, 0 for no pattern
#     X = X.iloc[:-1]  # Align X with y by removing the last row
#     y = y.iloc[1:]  # Align y with X by shifting
#
#     return X.dropna(), y.dropna()
#
#
# # Train or update the model
# def train_or_update_model(X, y, model, scaler):
#     if X is None or y is None or len(X) < 2:
#         logging.error("Not enough data to train the model.")
#         return model, scaler
#
#     try:
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
#
#         if len(X_train) == 0 or len(y_train) == 0:
#             logging.error("Training set is empty after the split.")
#             return model, scaler
#
#         scaler = StandardScaler()
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_test_scaled = scaler.transform(X_test)
#
#         if model is None:
#             model = LogisticRegression()
#
#         model.fit(X_train_scaled, y_train)
#         y_pred = model.predict(X_test_scaled)
#         accuracy = accuracy_score(y_test, y_pred)
#         logging.info(f"Model accuracy: {accuracy:.2f}")
#
#         return model, scaler
#     except ValueError as e:
#         logging.error(f"Error during model training: {e}")
#         return model, scaler
#
#
# # Make prediction and decide on the trading action
# def make_prediction(model, scaler, df):
#     df = extract_features(df)
#     if len(df) < 1:
#         logging.error("Insufficient data for prediction.")
#         return None
#
#     X_new = df[['return', 'volatility', 'momentum', 'swing']].iloc[-1:].copy()
#     X_new_scaled = scaler.transform(X_new)
#     prediction = model.predict(X_new_scaled)
#     return prediction[0]
#
#
# # Example usage
# def main():
#     model = None
#     scaler = None
#
#     # Load historical data
#     df = pd.read_csv('LTC-USDT-SWAP.csv')
#
#     # Prepare data and train/update model
#     X, y = prepare_data(df)
#     model, scaler = train_or_update_model(X, y, model, scaler)
#
#     # Make a trading decision
#     decision = make_prediction(model, scaler, df)
#     if decision == 1:
#         print("Buy")
#     elif decision == 2:
#         print("Sell")
#     else:
#         print("No pattern detected")
#
#     # Continue to update model with new data
#     # For continuous learning, integrate this into your data pipeline
#     # new_data = pd.read_csv('new_data.csv')
#     # X_new, y_new = prepare_data(new_data)
#     # model, scaler = train_or_update_model(X_new, y_new, model, scaler)
#
#
# if __name__ == "__main__":
#     main()
