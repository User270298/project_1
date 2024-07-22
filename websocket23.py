from pybit.unified_trading import WebSocket
from time import sleep
import json
from datetime import datetime
import pandas as pd
import csv
import threading
import requests

#telegram
TOKEN = '6959314930:AAHnekjhCc2d_CHFLxE9hFWAZuIgQMD8wzY'
chat_id='947159905'


df = pd.DataFrame()
ws = WebSocket(
    testnet=True,
    channel_type="linear",
)
print('Start')
# with open("BTC-USDT-SWAP.csv", mode="a", encoding='utf-8') as w_file:
#     file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
#     file_writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
# with open("ETH-USDT-SWAP.csv", mode="a", encoding='utf-8') as w_file:
#     file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
#     file_writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
# with open("SOL-USDT-SWAP.csv", mode="a", encoding='utf-8') as w_file:
#     file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
#     file_writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
# with open("ADA-USDT-SWAP.csv", mode="a", encoding='utf-8') as w_file:
#     file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
#     file_writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])

my_timeframe=1

def message(e):
    message = (f'{e}')
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    requests.get(url).json()

def handle_message_1(message):
    try:
        info=message['data'][0]
        opened=info['open']
        close=info['close']
        high=info['high']
        low=info['low']
        volume=info['volume']
        timeframe=str(datetime.fromtimestamp(int(message['ts'])/1000))
        timeframe=timeframe[:19]


        if int(timeframe[14:16])%my_timeframe==0 and str(timeframe[17:19])=='00': #int(timeframe[14:16])%5==0 and int(timeframe[14:16])%15==0 and
            # print('Save to BTC-USDT-SWAP.csv')
            with open("BTC-USDT-SWAP.csv", mode="a", encoding='utf-8') as w_file:
                file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
                file_writer.writerow([timeframe, opened, high, low, close, volume])

    except Exception as e:
        print(e)
        message(e)
def handle_message_2(message):
    try:
        info=message['data'][0]
        opened=info['open']
        close=info['close']
        high=info['high']
        low=info['low']
        volume=info['volume']
        timeframe=str(datetime.fromtimestamp(int(message['ts'])/1000))
        timeframe=timeframe[:19]
        if int(timeframe[14:16])%my_timeframe==0 and str(timeframe[17:19])=='00': #int(timeframe[14:16])%5==0 and int(timeframe[14:16])%15==0 and
            # print('Save to ETH-USDT-SWAP.csv')
            with open("ETH-USDT-SWAP.csv", mode="a", encoding='utf-8') as w_file:
                file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
                file_writer.writerow([timeframe, opened, high, low, close, volume])
    except Exception as e:
        print(e)
        message(e)
def handle_message_3(message):
    try:
        info=message['data'][0]
        opened=info['open']
        close=info['close']
        high=info['high']
        low=info['low']
        volume=info['volume']
        timeframe=str(datetime.fromtimestamp(int(message['ts'])/1000))
        timeframe=timeframe[:19]
        if int(timeframe[14:16])%my_timeframe==0 and str(timeframe[17:19])=='00': #int(timeframe[14:16])%5==0 and int(timeframe[14:16])%15==0 and
            # print('Save to SOL-USDT-SWAP.csv')
            with open("SOL-USDT-SWAP.csv", mode="a", encoding='utf-8') as w_file:
                file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
                file_writer.writerow([timeframe, opened, high, low, close, volume])
    except Exception as e:
        print(e)
        message(e)
# def handle_message_4(message):
#     try:
#         info=message['data'][0]
#         opened=info['open']
#         close=info['close']
#         high=info['high']
#         low=info['low']
#         volume=info['volume']
#         timeframe=str(datetime.fromtimestamp(int(message['ts'])/1000))
#         timeframe=timeframe[:19]
#         if int(timeframe[14:16])%my_timeframe==0 and str(timeframe[17:19])=='00': #int(timeframe[14:16])%5==0 and int(timeframe[14:16])%15==0 and
#             print('Save to ADA-USDT-SWAP.csv')
#             with open("ADA-USDT-SWAP.csv", mode="a", encoding='utf-8') as w_file:
#                 file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
#                 file_writer.writerow([timeframe, opened, high, low, close, volume])
#     except Exception as e:
#         print(e)
#         message(e)
try:
    threading.Thread(target=ws.kline_stream, args=(my_timeframe, "BTCUSDT", handle_message_1, )).start()
except Exception as e:
    print(e, '1')
    message(e)
try:
    threading.Thread(target=ws.kline_stream, args=(my_timeframe, "ETHUSDT", handle_message_2, )).start()
except Exception as e:
    print(e, '2')
    message(e)
try:
    threading.Thread(target=ws.kline_stream, args=(my_timeframe, "SOLUSDT", handle_message_3, )).start()
except Exception as e:
    print(e, '3')
    message(e)
# try:
#     threading.Thread(target=ws.kline_stream, args=(my_timeframe, "ADAUSDT", handle_message_4, )).start()
# except Exception as e:
#     print(e, '4')
#     message(e)

while True:
    sleep(1)

# #{'topic': 'kline.1.ETHUSDT', 'data': [{'start': 1714474440000, 'end': 1714474499999, 'interval': '1', 'open': '303
# 8.14', 'close': '3037.13', 'high': '3038.14', 'low': '3036.35', 'volume': '220.56', 'turnover': '669809.5108', 'confirm': False, 'timestamp': 1714474460602}], 'ts': 1714474460602, 'type': 'snapshot'}


