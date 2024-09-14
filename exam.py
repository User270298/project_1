# import okx.Account as Account
# import okx.Trade as trade
# import pprint
# import requests
# import datetime
# from time import sleep
# import pandas as pd
# import numpy as np
#
# # telegram
# TOKEN = '6959314930:AAHnekjhCc2d_CHFLxE9hFWAZuIgQMD8wzY'
# chat_id = '947159905'
#
#
# def message(x):
#     message = (f'{x}')
#     url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
#     requests.get(url).json()
#
#
# # live trading: 0, demo trading: 1
# # OKX demo trading
# # api_key='43f5df59-5e61-4d24-875e-f32c003e0430'
# # secret_key='5B1063B322635A27CF01BACE3772E0E0'
# # passphrase='Parkwood270298)'
# # flag = "1"
#
# # OKX live trading
# api_key = 'f8bcadcc-bed3-4fca-96e7-4f314f43136b'
# secret_key = 'F56CF3942B876FDEDEF547C90B04F206'
# passphrase = 'Parkwood270298)'
# flag = "0"
#
# accountAPI = Account.AccountAPI(api_key, secret_key, passphrase, False, flag)
# tradeAPI = trade.TradeAPI(api_key, secret_key, passphrase, False, flag)
# # -------------------------------------------------
# count_long = 0
# count_short = 0
# ordId = 0
# risk = 20
# foulder = 20
#
# pattern_by_long_op=0
# pattern_by_long_ltc=0
# pattern_by_short_op = 0
# pattern_by_short_ltc= 0
# clord_long=''
# clord_short=''
# while True:
#     try:
#         # в 1 час 12 раз по 5 минут, 4 раза по 15 минут, 2 раза по 30 минут
#         # Нужно каждые 61 бар делать статистику
#         for i in [ 'LTC-USDT-SWAP.csv', 'OP-USDT-SWAP.csv']:  # , 'SOL-USDT-SWAP.csv'
#             coin = i
#             df = pd.read_csv(i)
#             pd.options.display.max_rows = 2000
#             pd.set_option('display.max_rows', None)
#
#
#             def isSwing(candle, window):
#                 if candle - window < 0 or candle + window >= len(df):
#                     return 0
#                 # print(candle, window)
#                 swingHigh = 1
#                 swingLow = 2
#                 for i in range(candle - window, candle + window + 1):
#                     if df.iloc[candle].low > df.iloc[i].low:
#                         swingLow = 0
#                     if df.iloc[candle].high < df.iloc[i].high:
#                         swingHigh = 0
#                 if (swingHigh and swingLow):
#                     return 3
#                 elif swingHigh:
#                     return swingHigh
#                 elif swingLow:
#                     return swingLow
#                 else:
#                     return 0
#
#
#             window = 10
#             df['isSwing'] = df.apply(lambda x: isSwing(x.name, window), axis=1)
#
#
#             def pointpos(x):
#                 if x['isSwing'] == 2:
#                     return x['low']
#                 elif x['isSwing'] == 1:
#                     return x['high']
#                 else:
#                     return np.nan
#
#
#             df['pointpos'] = df.apply(lambda row: pointpos(row), axis=1)
#
#
#             def detect_structure(candle, backcandles, window):
#                 localdf = df.iloc[
#                           candle - backcandles - window:candle - window]  # window must be greater than pivot window to avoid look ahead bias
#                 highs = localdf[localdf['isSwing'] == 1].high.tail(2).values
#                 lows = localdf[localdf['isSwing'] == 2].low.tail(2).values
#                 levelbreak = 0
#                 zone_width = 0.001
#                 if len(highs) == 2:  # long
#                     resistance_condition = True
#                     mean_high = highs.mean()
#                     if resistance_condition and (df.loc[candle].close - mean_high) > zone_width * 2:
#                         levelbreak = 1
#                 if len(lows) == 2:  # short
#                     support_condition = True
#                     mean_low = lows.mean()
#                     if support_condition and (mean_low - df.loc[candle].close) > zone_width * 2:
#                         levelbreak = 2
#                 return levelbreak
#
#
#             df['pattern_detected'] = df.apply(lambda row: detect_structure(row.name, backcandles=60, window=9), axis=1)
#             # print(df.tail(2))
#             print(df["pattern_detected"].iloc[-1])
#             # print(df['close'].iloc[-1])
#             coin = coin[:-4]
#
#
#             if coin == 'OP-USDT-SWAP':
#                 deliver = 1
#             elif coin == 'LTC-USDT-SWAP':
#                 deliver = 1
#
#             result = accountAPI.get_positions()
#             list_coins = []
#             for i in range(len(result['data'])):
#                 res = result['data'][i]['instId']
#                 list_coins.append(res)
#             print(f'List coins: {list_coins}')
#             if df["pattern_detected"].iloc[-1] == 1 and (coin not in list_coins):
#                 # Long
#                 rslt_df_high = df[df['isSwing'] == 1]
#                 rslt_df_low = (df[df['isSwing'] == 2])
#                 high = rslt_df_high['pointpos'].iloc[-1]
#                 low = rslt_df_low['pointpos'].iloc[-1]
#                 close = df['close'].iloc[-1]
#                 # middle = (high + low) / 2
#                 stop = low * 0.9996
#                 take = ((close - stop) * 3) + close
#                 percent_sz = round(((risk / ((close - stop) / stop)) * deliver) / close, 1)
#                 if coin == 'OP-USDT-SWAP':
#                     pattern_by_long_op=((close - stop)) + close
#                     op_close_long=close
#                     clord_long='11'
#                 elif coin == 'LTC-USDT-SWAP':
#                     pattern_by_long_ltc=((close - stop)) + close
#                     ltc_close_long=close
#                     clord_long = '12'
#                 print('------------LONG-------------')
#                 print(f'Take {take}')
#                 print(f'Coin {close}')
#                 print(f'Stop {stop}')
#                 result = tradeAPI.place_order(
#                     instId=coin,
#                     tdMode="isolated",
#                     side="buy",
#                     posSide="long",
#                     ordType="market",
#                     sz=percent_sz,
#                     # px=middle,
#                     tpTriggerPx=float(take),  # take profit trigger price
#                     tpOrdPx="-1",
#                     # taker profit order price。When it is set to -1，the order will be placed as an market order
#                     tpTriggerPxType="last",
#                     slTriggerPx=float(stop),  # take profit trigger price
#                     slOrdPx="-1",
#                     # taker profit order price。When it is set to -1，the order will be placed as an market order
#                     slTriggerPxType="last",
#                     clOrdId=clord_long
#                 )
#                 message(f'------LONG------- \n'
#                         f'coin: {coin}\n'
#                         f'Percent size {percent_sz}\n'
#                         f'Take profit {take}\n'
#                         f'Coin {close}\n'
#                         f'Stop loss {stop}\n'
#                         f'{result}\n'
#                         f'{list_coins}\n'
#                         )
#
#             elif df["pattern_detected"].iloc[-1] == 2 and (coin not in list_coins):
#                 # Short
#                 rslt_df_high = df[df['isSwing'] == 1]
#                 rslt_df_low = (df[df['isSwing'] == 2])
#                 high = rslt_df_high['pointpos'].iloc[-1]
#                 low = rslt_df_low['pointpos'].iloc[-1]
#                 close = df['close'].iloc[-1]
#                 # middle = (high + low) / 2
#                 stop = high * 1.0004
#                 take = close - ((stop - close) * 3)
#                 percent_sz = round(((risk / ((stop - close) / close)) * deliver) / close, 1)
#
#                 if coin == 'OP-USDT-SWAP':
#                     pattern_by_short_op = close - ((stop - close) )
#                     op_close_short=close
#                     clord_short = '21'
#                 elif coin == 'LTC-USDT-SWAP':
#                     pattern_by_short_ltc= close - ((stop - close) )
#                     ltc_close_short = close
#                     clord_short = '22'
#                 print('------------SHORT-------------')
#                 print(f'Stop {stop}')
#                 print(f'Coin {close}')
#                 print(f'Take {take}')
#                 result = tradeAPI.place_order(
#                     instId=coin,
#                     tdMode="isolated",
#                     side="sell",
#                     posSide="short",
#                     ordType="market",
#                     sz=percent_sz,
#                     # px=middle,
#                     tpTriggerPx=float(take),  # take profit trigger price
#                     tpOrdPx="-1",
#                     # taker profit order price。When it is set to -1，the order will be placed as an market order
#                     tpTriggerPxType="last",
#                     slTriggerPx=float(stop),  # take profit trigger price
#                     slOrdPx="-1",
#                     # taker profit order price。When it is set to -1，the order will be placed as an market order
#                     slTriggerPxType="last",
#                     clOrdId=clord_short
#                 )
#                 message(f'------SHORT------- \n'
#                         f'Coin: {coin}\n'
#                         f'Percent size {percent_sz}\n'
#                         f'Take profit {take}\n'
#                         f'Coin {close}\n'
#                         f'Stop loss {stop}\n'
#                         f'{result}\n'
#                         f'{list_coins}\n')
#             close = df['close'].iloc[-1]
#             if close==pattern_by_long_op:
#                 result = tradeAPI.amend_order(
#                 instId=coin,
#                 newTpTriggerPxType=op_close_long,
#                     clOrdId=clord_long
#                 )
#             elif close==pattern_by_long_ltc:
#                 result = tradeAPI.amend_order(
#                 instId=coin,
#                 newTpTriggerPxType=ltc_close_long,
#                     clOrdId=clord_long
#                 )
#             elif close==pattern_by_short_op:
#                 result = tradeAPI.amend_order(
#                     instId=coin,
#                     newTpTriggerPxType=op_close_short,
#                     clOrdId=clord_short
#                 )
#             elif close==pattern_by_short_ltc:
#                 result = tradeAPI.amend_order(
#                     instId=coin,
#                     newTpTriggerPxType=ltc_close_short,
#                     clOrdId=clord_short
#                 )
#
#         sleep(60)
#     except Exception as e:
#         message(f'Exam: {e}')




import okx.Account as Account
import okx.Trade as trade
import pprint
import requests
import datetime
import logging
from time import sleep
import pandas as pd
import numpy as np
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging
from time import sleep
import pickle

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

#Слом структуры________________________________________________________________
def is_swing(candle, window, df):
    if candle - window < 0 or candle + window >= len(df):
        return 0
    swing_high = all(df.iloc[candle].high >= df.iloc[i].high for i in range(candle - window, candle + window + 1))
    swing_low = all(df.iloc[candle].low <= df.iloc[i].low for i in range(candle - window, candle + window + 1))
    return 1 if swing_high else (2 if swing_low else 0)


def pointpos(row):
    return row['low'] if row['isSwing'] == 2 else (row['high'] if row['isSwing'] == 1 else np.nan)


def detect_structure(candle, df, backcandles=60, window=10):
    # Проверка, что индекс в пределах допустимого диапазона
    if candle - backcandles - window < 0 or candle >= len(df):
        return 0

    # Выбор локального диапазона данных
    localdf = df.iloc[candle - backcandles - window:candle - window]
    if localdf.empty:
        return 0

    # Проверка на боковик
    high_range = localdf['high'].max() - localdf['high'].min()
    low_range = localdf['low'].max() - localdf['low'].min()
    zone_width = 0.001

    if high_range < zone_width and low_range < zone_width:
        # Если цена находится в боковике, возвращаем 0
        return 0

    # Проверка на максимумы и минимумы
    highs = localdf[localdf['isSwing'] == 1].high.tail(2).values
    lows = localdf[localdf['isSwing'] == 2].low.tail(2).values

    if len(highs) < 2 or len(lows) < 2:
        # Если недостаточно данных для определения структуры
        return 0

    # Проверка на слом структуры
    if len(highs) == 2 and df.loc[candle].close - highs.mean() > zone_width * 2:
        return 1  # Сигнал на покупку (смотрим, что текущая цена выше средних максимумов)

    if len(lows) == 2 and lows.mean() - df.loc[candle].close > zone_width * 2:
        return 2  # Сигнал на продажу (смотрим, что текущая цена ниже средних минимумов)

    return 0  # Нет сигнала
#___________________________________________________________________________

#Машинное обучение
global_model = None
global_scaler = None


# Функции для обучения и использования модели
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
    X_new = df[['return', 'volatility', 'momentum']].iloc[-1:].copy()
    X_new_scaled = scaler.transform(X_new)
    prediction = model.predict(X_new_scaled)
    return prediction[0]

def train_model_for_coin(coin):
    global global_model, global_scaler
    if global_model is not None and global_scaler is not None:
        return global_model, global_scaler

    historical_data = pd.read_csv(coin)
    logging.info(f"Training model for {coin}. Total data points: {len(historical_data)}")
    X, y = prepare_data(historical_data)
    if X is not None and y is not None:
        logging.info(f"Prepared data for {coin}. Data points after preparation: {len(X)}")
        global_model, global_scaler = train_model(X, y)
        if global_model and global_scaler:
            logging.info(f'Model trained for {coin}')
        else:
            logging.error(f'Failed to train model for {coin}')
    return global_model, global_scaler

#_________________________________________________________________________

def process_coin(coin, accountAPI, tradeAPI, list_coins, deliver, model, scaler, risk=4):
    df = pd.read_csv(coin)
    window = 10
    df['isSwing'] = df.apply(lambda x: is_swing(x.name, window, df), axis=1)
    df['pointpos'] = df.apply(pointpos, axis=1)
    df['pattern_detected'] = df.apply(lambda x: detect_structure(x.name, df), axis=1)

    ml_signal = make_prediction(model, scaler, df)
    print(f'Ml_signal: {ml_signal}')
    latest_pattern = df["pattern_detected"].iloc[-1]
    print(f'Latest_pattern: {latest_pattern}')
    coin = coin[:-4]
    if latest_pattern==1 and ml_signal==1 and coin not in list_coins:
        handle_trade_signal(coin, df, latest_pattern, accountAPI, tradeAPI, risk, deliver)
    if latest_pattern==2 and ml_signal==2 and coin not in list_coins:
        handle_trade_signal(coin, df, latest_pattern, accountAPI, tradeAPI, risk, deliver)


def handle_trade_signal(coin, df, pattern, accountAPI, tradeAPI, risk, deliver):
    rslt_df_high = df[df['isSwing'] == 1]
    rslt_df_low = df[df['isSwing'] == 2]
    high = rslt_df_high['pointpos'].iloc[-1]
    low = rslt_df_low['pointpos'].iloc[-1]
    close = df['close'].iloc[-1]
    stop = low * 0.9996 if pattern == 1 else high * 1.0004
    take = ((close - stop) * 3) + close if pattern == 1 else close - ((stop - close) * 3)
    percent_sz = round(((risk / ((close - stop) / stop)) * deliver) / close, 1) if pattern == 1 else round(((risk / ((high - close) / high)) * deliver) / close, 1)

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
                 f'{result}\n'
                 )


def main():
    # Demo
    api_key = '43f5df59-5e61-4d24-875e-f32c003e0430'
    secret_key = '5B1063B322635A27CF01BACE3772E0E0'
    passphrase = 'Parkwood270298)'
    flag = "1"
    # REAL
    # api_key = 'f8bcadcc-bed3-4fca-96e7-4f314f43136b'
    # secret_key = 'F56CF3942B876FDEDEF547C90B04F206'
    # passphrase = 'Parkwood270298)'
    # flag = "0"
    accountAPI = Account.AccountAPI(api_key, secret_key, passphrase, False, flag)
    tradeAPI = trade.TradeAPI(api_key, secret_key, passphrase, False, flag)
    train_model_for_coin('LTC-USDT-SWAP.csv')
    # train_model_for_coin('BTC-USDT-SWAP.csv')

    while True:

        try:
            result = accountAPI.get_positions()
            list_coins = [pos['instId'] for pos in result['data']]
            logging.info(f'Active positions: {list_coins}')
            for coin in ['LTC-USDT-SWAP.csv']: #, 'BTC-USDT-SWAP.csv'
                model, scaler = train_model_for_coin(coin)
                if coin=='LTC-USDT-SWAP.csv':
                    deliver=1000
                # elif coin=='BTC-USDT-SWAP.csv':
                #     deliver=1000
                process_coin(coin, accountAPI, tradeAPI, list_coins, deliver, model, scaler)

            sleep(60)
        except Exception as e:
            logging.error(f'Error in main loop: {e}')
            send_message(f'Error: {e}')
            sleep(60)


if __name__ == "__main__":
    main()









#####
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# import logging
#
#
# # Example: Adding a machine learning model to predict trades
#
# def create_features(df):
#     df['ma5'] = df['close'].rolling(window=5).mean()
#     df['ma10'] = df['close'].rolling(window=10).mean()
#     df['rsi'] = calculate_rsi(df['close'])
#     df['return'] = df['close'].pct_change()
#     df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)  # 1 for up, 0 for down
#     df = df.dropna()
#     return df
#
#
# def calculate_rsi(series, period=14):
#     delta = series.diff(1)
#     gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
#     rs = gain / loss
#     return 100 - (100 / (1 + rs))
#
#
# def train_model(df):
#     features = ['ma5', 'ma10', 'rsi', 'return']
#     X = df[features]
#     y = df['target']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
#
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     logging.info(f'Model Accuracy: {accuracy}')
#
#     return model
#
#
# def make_prediction(model, df):
#     features = ['ma5', 'ma10', 'rsi', 'return']
#     X = df[features].iloc[-1].values.reshape(1, -1)
#     prediction = model.predict(X)
#     return prediction[0]
#
#
# def process_coin(coin, accountAPI, tradeAPI, list_coins, model, risk=20, deliver=1):
#     df = pd.read_csv(coin)
#     df = create_features(df)
#
#     prediction = make_prediction(model, df)
#
#     if prediction == 1 and coin not in list_coins:
#         handle_trade_signal(coin, df, 1, accountAPI, tradeAPI, risk, deliver)
#     elif prediction == 0 and coin not in list_coins:
#         handle_trade_signal(coin, df, 2, accountAPI, tradeAPI, risk, deliver)
#
#
# def main():
#     api_key = 'Your API key'
#     secret_key = 'Your secret key'
#     passphrase = 'Your passphrase'
#     flag = "0"
#
#     accountAPI = Account.AccountAPI(api_key, secret_key, passphrase, False, flag)
#     tradeAPI = trade.TradeAPI(api_key, secret_key, passphrase, False, flag)
#
#     # Training model with historical data
#     df = pd.read_csv('historical_data.csv')  # Replace with your historical data file
#     df = create_features(df)
#     model = train_model(df)
#
#     while True:
#         try:
#             result = accountAPI.get_positions()
#             list_coins = [pos['instId'] for pos in result['data']]
#             logging.info(f'Active positions: {list_coins}')
#
#             for coin in ['LTC-USDT-SWAP.csv', 'OP-USDT-SWAP.csv']:
#                 process_coin(coin, accountAPI, tradeAPI, list_coins, model)
#
#             sleep(60)
#         except Exception as e:
#             logging.error(f'Error in main loop: {e}')
#             send_message(f'Error: {e}')
#             sleep(60)
#
#
# if __name__ == "__main__":
#     main()
