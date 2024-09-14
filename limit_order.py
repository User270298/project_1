#limit order

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

def calculate_stop_loss_take_profit(high, low, pattern):
    if pattern == 1:  # Для long позиций
        stop_loss = low - (high - low) * 0.1  # Стоп-лосс чуть ниже последнего low
        take_profit = high + 3 * (high - stop_loss)  # Тейк-профит в 3 раза больше расстояния до стоп-лосса
    else:  # Для short позиций
        stop_loss = high + (high - low) * 0.1  # Стоп-лосс чуть выше последнего high
        take_profit = low - 3 * (stop_loss - low)  # Тейк-профит в 3 раза больше расстояния до стоп-лосса
    return stop_loss, take_profit


def place_limit_order(coin, tradeAPI, high, low, percent_sz, order_id, position_side, stop_loss, take_profit):
    limit_price = (high + low) / 2
    side = "buy" if position_side == "long" else "sell"
    logging.info(f'Placing limit {position_side} order for {coin} at price {limit_price}')
    coin = coin[:-4]
    result = tradeAPI.place_order(
        instId=coin,
        tdMode="isolated",
        side=side,
        posSide=position_side,
        ordType="limit",
        sz=percent_sz,
        px=limit_price,
        tpTriggerPx=float(take_profit),
        tpOrdPx="-1",
        tpTriggerPxType="last",
        slTriggerPx=float(stop_loss),
        slOrdPx="-1",
        slTriggerPxType="last",
        clOrdId=order_id
    )

    return result, limit_price


def cancel_order_if_far(coin, tradeAPI, current_price, limit_price, tolerance, order_id):
    """
    Функция для отмены ордера, если цена ушла слишком далеко.
    """
    coin = coin[:-4]
    if abs(current_price - limit_price) > tolerance:
        logging.info(f'Cancelling order for {coin} due to price moving too far from limit price.')
        result = tradeAPI.cancel_order(instId=coin, clOrdId=order_id)
        logging.info(f'Order cancelled: {result}')
        return True
    return False


def process_coin(coin, accountAPI, tradeAPI, list_coins, risk=10, deliver=1, tolerance=0.002):
    df = pd.read_csv(coin)
    window = 10
    df['isSwing'] = df.apply(lambda x: is_swing(x.name, window, df), axis=1)
    df['pointpos'] = df.apply(pointpos, axis=1)
    df['pattern_detected'] = df.apply(lambda x: detect_structure(x.name, df), axis=1)

    latest_pattern = df["pattern_detected"].iloc[-1]
    if latest_pattern in [1, 2] and coin not in list_coins:
        handle_trade_signal(coin, df, latest_pattern, accountAPI, tradeAPI, risk, deliver, tolerance)
    elif coin in list_coins:
        position_side = "long" if latest_pattern == 1 else "short"
        order_id = f'{position_side[:1].upper()}{coin[:2].upper()}'
        current_price = df['close'].iloc[-1]
        limit_price = df['limit_price'].iloc[-1]  # Добавьте сохранение лимитной цены после размещения ордера
        cancel_order_if_far(coin, tradeAPI, current_price, limit_price, tolerance, order_id)


def handle_trade_signal(coin, df, pattern, accountAPI, tradeAPI, risk, deliver, tolerance):
    rslt_df_high = df[df['isSwing'] == 1]
    rslt_df_low = df[df['isSwing'] == 2]
    high = rslt_df_high['pointpos'].iloc[-1]
    low = rslt_df_low['pointpos'].iloc[-1]
    stop_loss, take_profit = calculate_stop_loss_take_profit(high, low, pattern)

    close = df['close'].iloc[-1]
    percent_sz = round(((risk / ((close - low) / low)) * deliver) / close, 1) if pattern == 1 else round(((risk / ((high - close) / high)) * deliver) / close, 1)

    pos_side = "long" if pattern == 1 else "short"
    order_id = f'{pos_side[:1].upper()}{coin[:2].upper()}'
    coin = coin[:-4]

    result, limit_price = place_limit_order(coin, tradeAPI, high, low, percent_sz, order_id, pos_side, stop_loss, take_profit)

    df['limit_price'] = limit_price  # Сохраняем цену лимитного ордера для возможного отзыва

    send_message(f'------LIMIT {pos_side.upper()} ORDER------- \n'
                 f'coin: {coin}\n'
                 f'Limit price {limit_price}\n'
                 f'Stop loss {stop_loss}, Take profit {take_profit}\n'
                 f'Percent size {percent_sz}\n'
                 f'High: {high}, Low: {low}\n'
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
                process_coin(coin, accountAPI, tradeAPI, list_coins)

            sleep(60)
        except Exception as e:
            logging.error(f'Error in main loop: {e}')
            send_message(f'Error: {e}')
            sleep(60)


if __name__ == "__main__":
    main()