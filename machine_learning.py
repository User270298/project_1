import okx.Account as Account
import okx.Trade as trade
import pprint
import requests
import datetime
import logging
from time import sleep
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Telegram
TOKEN = '6959314930:AAHnekjhCc2d_CHFLxE9hFWAZuIgQMD8wzY'
chat_id = '947159905'


def send_message(text):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={text}"
    try:
        requests.get(url)
    except requests.RequestException as e:
        logging.error(f"Failed to send message: {e}")


# Функции для определения паттернов
def is_swing(candle, window, df):
    if candle - window < 0 or candle + window >= len(df):
        return 0
    swing_high = all(df.iloc[candle].high >= df.iloc[i].high for i in range(candle - window, candle + window + 1))
    swing_low = all(df.iloc[candle].low <= df.iloc[i].low for i in range(candle - window, candle + window + 1))
    return 1 if swing_high else (2 if swing_low else 0)


def pointpos(row):
    return row['low'] if row['isSwing'] == 2 else (row['high'] if row['isSwing'] == 1 else np.nan)


def detect_structure(candle, df, backcandles=60, window=10):
    localdf = df.iloc[candle - backcandles - window:candle - window]
    highs = localdf[localdf['isSwing'] == 1].high.tail(2).values
    lows = localdf[localdf['isSwing'] == 2].low.tail(2).values
    zone_width = 0.001
    if len(highs) == 2 and df.loc[candle].close - highs.mean() > zone_width * 2:
        return 1
    if len(lows) == 2 and lows.mean() - df.loc[candle].close > zone_width * 2:
        return 2
    return 0


# Машинное обучение
def extract_features(df):
    df['return'] = df['close'].pct_change()
    df['volatility'] = df['close'].rolling(window=10).std()
    df['momentum'] = df['close'].rolling(window=5).mean() - df['close'].rolling(window=15).mean()
    df.dropna(inplace=True)
    return df


def prepare_data(df):
    df = extract_features(df)

    if len(df) < 2:  # Проверка, что есть хотя бы 2 записи для обучения
        logging.error(f"Insufficient data for training. Available samples: {len(df)}")
        return None, None

    X = df[['return', 'volatility', 'momentum']]
    y = (df['close'].shift(-1) > df['close']).astype(int)

    return X.dropna(), y.dropna()  # Убедитесь, что нет NaN значений


def train_model(X, y):
    if X is None or y is None or len(X) < 2:  # Проверка, что есть достаточно данных для обучения
        logging.error("Not enough data to train the model.")
        return None, None

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        if len(X_train) == 0 or len(y_train) == 0:  # Проверка на пустые обучающие выборки
            logging.error("Training set is empty after the split.")
            return None, None

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Model accuracy: {accuracy:.2f}")
        return model, scaler
    except ValueError as e:
        logging.error(f"Error during model training: {e}")
        return None, None


def make_prediction(model, scaler, df):
    df = extract_features(df)
    if len(df) < 1:
        logging.error("Insufficient data for prediction.")
        return None

    # Сохраняем имена признаков и делаем предсказание с правильными именами столбцов
    X_new = df[['return', 'volatility', 'momentum']].iloc[-1:].copy()
    X_new_scaled = scaler.transform(X_new)
    prediction = model.predict(X_new_scaled)
    return prediction[0]

# Обработка сигналов торговли
def handle_trade_signal(coin, df, pattern, accountAPI, tradeAPI, risk, deliver):
    rslt_df_high = df[df['isSwing'] == 1]
    rslt_df_low = df[df['isSwing'] == 2]
    high = rslt_df_high['pointpos'].iloc[-1]
    low = rslt_df_low['pointpos'].iloc[-1]
    close = df['close'].iloc[-1]


    # Define stop loss and take profit based on trade direction
    if pattern == 1:  # Buy
        stop = low * 0.99  # Stop below the low
        take = ((close - stop) * 3) + close  # Take profit above current pric

    elif pattern == 2:  # Sell
        stop = high * 1.01  # Stop above the high
        take = close - ((stop - close) * 3)  # Take profit below current price
    print(low)
    print(high)
    print(close)
    print(stop)
    print(take)
    percent_sz = round(((risk / ((close - stop) / stop)) * deliver) / close, 1) if pattern == 1 else round(
        ((risk / ((high - close) / high)) * deliver) / close, 1)

    side = "buy" if pattern == 1 else "sell"
    pos_side = "long" if pattern == 1 else "short"
    order_id = f'{pos_side[:1].upper()}{coin[:2].upper()}'

    logging.info(f'Placing {pos_side} order for {coin}: size {percent_sz}, stop {stop}, take {take}')
    result = tradeAPI.place_order(
        instId=coin,
        tdMode="isolated",
        side=side,
        posSide=pos_side,
        ordType="market",
        sz=percent_sz,
        tpTriggerPx=float(take),
        tpOrdPx="-1",
        tpTriggerPxType="last",
        slTriggerPx=float(stop),
        slOrdPx="-1",
        slTriggerPxType="last",
        clOrdId=order_id
    )

    send_message(f'------{pos_side.upper()}------- \n'
                 f'coin: {coin}\n'
                 f'Percent size {percent_sz}\n'
                 f'Take profit {take}\n'
                 f'Coin {close}\n'
                 f'Stop loss {stop}\n'
                 f'{result}\n')

def close_position_if_needed(coin, df, ml_signal, accountAPI, tradeAPI, model, scaler, risk, deliver):
    # Check current open positions
    positions = accountAPI.get_positions()

    for pos in positions['data']:
        if pos['instId'] == coin:
            current_side = pos['posSide'].lower()
            if current_side == "long":
                # Check if signal is for selling
                latest_signal = make_prediction(model, scaler, df)
                if ml_signal==1:
                    logging.info(f"Closing long position for {coin} due to signal change")
                    tradeAPI.cancel_order(instId=coin, ordType="market", clOrdId=order_id)
                    send_message(f"Closed long position for {coin} due to signal change")
            elif ml_signal==0:
                # Check if signal is for buying
                latest_signal = make_prediction(model, scaler, df)
                if latest_signal == 1:
                    logging.info(f"Closing short position for {coin} due to signal change")
                    tradeAPI.cancel_order(instId=coin,  ordType="market")
                    send_message(f"Closed short position for {coin} due to signal change")

def process_coin_with_ml(coin, accountAPI, tradeAPI, list_coins, deliver, risk=5, model=None, scaler=None):
    df = pd.read_csv(coin)
    if len(df) < 2:
        logging.error(f"Insufficient data for processing coin {coin}")
        return

    window = 10
    df['isSwing'] = df.apply(lambda x: is_swing(x.name, window, df), axis=1)
    df['pointpos'] = df.apply(pointpos, axis=1)
    df['pattern_detected'] = df.apply(lambda x: detect_structure(x.name, df), axis=1)
    latest_pattern = df["pattern_detected"].iloc[-1]
    ml_signal = make_prediction(model, scaler, df)

    if ml_signal is None:
        logging.error(f"No prediction available for coin {coin}")
        return
    print(ml_signal)
    print(latest_pattern)
    # print(list_coins)
    coin = coin[:-4]

    if coin in list_coins:
        # Check if position needs to be closed based on new signal
        close_position_if_needed(coin, df, ml_signal, accountAPI, tradeAPI, model, scaler, risk, deliver)

    # Условие для роста (покупка)
    if ml_signal == 1 and latest_pattern in [1, 2] and coin not in list_coins:
        handle_trade_signal(coin, df, latest_pattern, accountAPI, tradeAPI, risk, deliver)

    # Условие для падения (продажа)
    elif ml_signal == 0 and latest_pattern in [1, 2] and coin not in list_coins:
        handle_trade_signal(coin, df, latest_pattern, accountAPI, tradeAPI, risk, deliver)


# Обучение модели для каждой криптовалюты
def train_models_for_all_coins(coins):
    models = {}
    scalers = {}
    for coin in coins:
        try:
            historical_data = pd.read_csv(coin)
            logging.info(
                f"Training model for {coin}. Total data points: {len(historical_data)}")  # Логирование перед обучением
            X, y = prepare_data(historical_data)
            if X is not None and y is not None:
                logging.info(f"Prepared data for {coin}. Data points after preparation: {len(X)}")
                model, scaler = train_model(X, y)
                if model and scaler:
                    models[coin] = model
                    scalers[coin] = scaler
                    logging.info(f'Model trained for {coin}')
                else:
                    logging.error(f'Failed to train model for {coin}')
            else:
                logging.error(f'Insufficient data for {coin}')
        except FileNotFoundError:
            logging.error(f"File not found: {coin}")
        except Exception as e:
            logging.error(f"Error while training model for {coin}: {e}")
    return models, scalers


def main():
    # Список криптовалют, для которых нужно обучить модели
    coins = ['LTC-USDT-SWAP.csv', 'OP-USDT-SWAP.csv']

    # Обучаем модели для всех криптовалют
    models, scalers = train_models_for_all_coins(coins)

    # Параметры API
    api_key = '43f5df59-5e61-4d24-875e-f32c003e0430'
    secret_key = '5B1063B322635A27CF01BACE3772E0E0'
    passphrase = 'Parkwood270298)'
    flag = "1"

    accountAPI = Account.AccountAPI(api_key, secret_key, passphrase, False, flag)
    tradeAPI = trade.TradeAPI(api_key, secret_key, passphrase, False, flag)

    while True:
        try:
            result = accountAPI.get_positions()
            list_coins = [pos['instId'] for pos in result['data']]
            logging.info(f'Active positions: {list_coins}')

            for coin in coins:
                deliver = 100 if coin == 'LTC-USDT-SWAP.csv' else 0.1
                model = models.get(coin)  # Используем соответствующую модель
                scaler = scalers.get(coin)  # Используем соответствующий нормализатор
                if model and scaler:
                    process_coin_with_ml(coin, accountAPI, tradeAPI, list_coins, deliver, model=model, scaler=scaler)
                else:
                    logging.error(f"No model or scaler available for {coin}")

            sleep(60)
        except Exception as e:
            logging.error(f'Error in main loop: {e}')
            send_message(f'Error: {e}')
            sleep(60)


if __name__ == "__main__":
    main()



