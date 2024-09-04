#Trailing stop
import okx.Account as Account
import okx.Trade as trade
import logging
from time import sleep
import pandas as pd
import numpy as np
import requests

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


def is_swing(candle, window, df):
    if candle - window < 0 or candle + window >= len(df):
        return 0
    swing_high = all(df.iloc[candle].high >= df.iloc[i].high for i in range(candle - window, candle + window + 1))
    swing_low = all(df.iloc[candle].low <= df.iloc[i].low for i in range(candle - window, candle + window + 1))
    return 1 if swing_high else (2 if swing_low else 0)


def pointpos(row):
    return row['low'] if row['isSwing'] == 2 else (row['high'] if row['isSwing'] == 1 else np.nan)


def detect_structure(candle, df, backcandles=60, window=9):
    localdf = df.iloc[candle - backcandles - window:candle - window]
    highs = localdf[localdf['isSwing'] == 1].high.tail(2).values
    lows = localdf[localdf['isSwing'] == 2].low.tail(2).values
    zone_width = 0.001
    if len(highs) == 2 and df.loc[candle].close - highs.mean() > zone_width * 2:
        return 1
    if len(lows) == 2 and lows.mean() - df.loc[candle].close > zone_width * 2:
        return 2
    return 0


def update_trailing_stop(coin, df, position_side, trailing_percentage, tradeAPI, order_id):
    """
    Обновление трейлинг-стопа на основе текущей цены.
    """
    last_price = df['close'].iloc[-1]
    current_stop = last_price * (1 - trailing_percentage) if position_side == "long" else last_price * (1 + trailing_percentage)

    logging.info(f'Updating trailing stop for {coin}. New stop: {current_stop}')
    coin = coin[:-4]
    # Здесь мы обновляем существующий ордер, меняя цену стоп-лосса
    result = tradeAPI.amend_order(
        instId=coin,
        newSlTriggerPx=float(current_stop),
        clOrdId=order_id
    )

    logging.info(f'Trailing stop updated: {result}')


def process_coin(coin, accountAPI, tradeAPI, list_coins, deliver, trailing_percentage=0.01, risk=4):
    df = pd.read_csv(coin)
    window = 10
    df['isSwing'] = df.apply(lambda x: is_swing(x.name, window, df), axis=1)
    df['pointpos'] = df.apply(pointpos, axis=1)
    df['pattern_detected'] = df.apply(lambda x: detect_structure(x.name, df), axis=1)

    latest_pattern = df["pattern_detected"].iloc[-1]
    if latest_pattern in [1, 2] and coin not in list_coins:
        handle_trade_signal(coin, df, latest_pattern, accountAPI, tradeAPI, trailing_percentage, risk, deliver)
    elif coin in list_coins:
        position_side = "long" if latest_pattern == 1 else "short"
        order_id = f'{position_side[:1].upper()}{coin[:2].upper()}'
        update_trailing_stop(coin, df, position_side, trailing_percentage, tradeAPI, order_id)


def handle_trade_signal(coin, df, pattern, accountAPI, tradeAPI, trailing_percentage, risk, deliver):
    rslt_df_high = df[df['isSwing'] == 1]
    rslt_df_low = df[df['isSwing'] == 2]
    high = rslt_df_high['pointpos'].iloc[-1]
    low = rslt_df_low['pointpos'].iloc[-1]
    close = df['close'].iloc[-1]

    stop = low * 0.9996 if pattern == 1 else high * 1.0004
    take = ((close - stop) * 3) + close if pattern == 1 else close - ((stop - close) * 3)
    percent_sz = round(((risk / ((close - stop) / stop)) * deliver) / close, 1) if pattern == 1 else round(((risk / ((stop - close) / close)) * deliver) / close, 1)

    side = "buy" if pattern == 1 else "sell"
    pos_side = "long" if pattern == 1 else "short"
    order_id = f'{pos_side[:1].upper()}{coin[:2].upper()}'

    logging.info(f'Placing {pos_side} order for {coin}: size {percent_sz}, stop {stop}, take {take}')
    coin=coin[:-4]
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
    #Demo
    api_key='43f5df59-5e61-4d24-875e-f32c003e0430'
    secret_key='5B1063B322635A27CF01BACE3772E0E0'
    passphrase='Parkwood270298)'
    flag = "1"
    #REAL
    # api_key = 'f8bcadcc-bed3-4fca-96e7-4f314f43136b'
    # secret_key = 'F56CF3942B876FDEDEF547C90B04F206'
    # passphrase = 'Parkwood270298)'
    # flag = "0"

    accountAPI = Account.AccountAPI(api_key, secret_key, passphrase, False, flag)
    tradeAPI = trade.TradeAPI(api_key, secret_key, passphrase, False, flag)

    while True:
        try:
            result = accountAPI.get_positions()
            list_coins = [pos['instId'] for pos in result['data']]
            logging.info(f'Active positions: {list_coins}')

            for coin in ['BTC-USDT-SWAP.csv', 'ETH-USDT-SWAP.csv']:
                if 'BTC-USDT-SWAP.csv':
                    deliver=1000
                elif 'ETH-USDT-SWAP.csv':
                    deliver=100
                process_coin(coin, accountAPI, tradeAPI, list_coins, deliver)

            sleep(60)
        except Exception as e:
            logging.error(f'Error in main loop: {e}')
            send_message(f'Error: {e}')
            sleep(60)


if __name__ == "__main__":
    main()
