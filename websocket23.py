from pybit.unified_trading import WebSocket
from time import sleep
import json
from datetime import datetime
import pandas as pd
import csv
import threading
import requests
import okx.MarketData as MarketData
import okx.Account as Account
import okx.Trade as trade

#telegram
TOKEN = '6959314930:AAHnekjhCc2d_CHFLxE9hFWAZuIgQMD8wzY'
chat_id='947159905'

api_key='43f5df59-5e61-4d24-875e-f32c003e0430'
secret_key='5B1063B322635A27CF01BACE3772E0E0'
passphrase='Parkwood270298)'
flag = "1"  # live trading: 0, demo trading: 1

# api_key='f8bcadcc-bed3-4fca-96e7-4f314f43136b'
# secret_key='F56CF3942B876FDEDEF547C90B04F206'
# passphrase='Parkwood270298)'
# flag = "0"

accountAPI = Account.AccountAPI(api_key, secret_key, passphrase, False, flag)
tradeAPI = trade.TradeAPI(api_key, secret_key, passphrase, False, flag)

# df = pd.DataFrame()
# ws = WebSocket(
#     testnet=True,
#     channel_type="linear",
# )
print('Start')
# with open("BTC-USDT-SWAP.csv", mode="a", encoding='utf-8') as w_file:
#     file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
#     file_writer.writerow(["timestamp", "open", "high", "low", "close"])
# with open("ETH-USDT-SWAP.csv", mode="a", encoding='utf-8') as w_file:
#     file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
#     file_writer.writerow(["timestamp", "open", "high", "low", "close"])
# with open("SOL-USDT-SWAP.csv", mode="a", encoding='utf-8') as w_file:
#     file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
#     file_writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
# with open("ADA-USDT-SWAP.csv", mode="a", encoding='utf-8') as w_file:
#     file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
#     file_writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])

def message(e):
    message = (f'{e}')
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    requests.get(url).json()

# def handle_message_1(message):
#     try:
#         info=message['data'][0]
#         opened=info['open']
#         close=info['close']
#         high=info['high']
#         low=info['low']
#         volume=info['volume']
#         timeframe=str(datetime.fromtimestamp(int(message['ts'])/1000))
#         timeframe=timeframe[:19]
#         # print(info)
#         print(type(timeframe))
#         print(type(opened))
#         print(type(close))
#         # print(type())
#         # print(type())
#
#         if int(timeframe[14:16])%my_timeframe==0 and str(timeframe[17:19])=='00': #int(timeframe[14:16])%5==0 and int(timeframe[14:16])%15==0 and
#             # print('Save to BTC-USDT-SWAP.csv')
#             with open("BTC-USDT-SWAP.csv", mode="a", encoding='utf-8') as w_file:
#                 file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
#                 file_writer.writerow([timeframe, opened, high, low, close, volume])
#
#     except Exception as e:
#         print(e)
#         message(e)
# # def handle_message_2(message):
# #     try:
# #         info=message['data'][0]
# #         opened=info['open']
# #         close=info['close']
# #         high=info['high']
# #         low=info['low']
# #         volume=info['volume']
# #         timeframe=str(datetime.fromtimestamp(int(message['ts'])/1000))
# #         timeframe=timeframe[:19]
# #         if int(timeframe[14:16])%my_timeframe==0 and str(timeframe[17:19])=='00': #int(timeframe[14:16])%5==0 and int(timeframe[14:16])%15==0 and
# #             # print('Save to ETH-USDT-SWAP.csv')
# #             with open("ETH-USDT-SWAP.csv", mode="a", encoding='utf-8') as w_file:
# #                 file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
# #                 file_writer.writerow([timeframe, opened, high, low, close, volume])
# #     except Exception as e:
# #         print(e)
# #         message(e)
# # def handle_message_3(message):
# #     try:
# #         info=message['data'][0]
# #         opened=info['open']
# #         close=info['close']
# #         high=info['high']
# #         low=info['low']
# #         volume=info['volume']
# #         timeframe=str(datetime.fromtimestamp(int(message['ts'])/1000))
# #         timeframe=timeframe[:19]
# #         if int(timeframe[14:16])%my_timeframe==0 and str(timeframe[17:19])=='00': #int(timeframe[14:16])%5==0 and int(timeframe[14:16])%15==0 and
# #             # print('Save to SOL-USDT-SWAP.csv')
# #             with open("SOL-USDT-SWAP.csv", mode="a", encoding='utf-8') as w_file:
# #                 file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
# #                 file_writer.writerow([timeframe, opened, high, low, close, volume])
# #     except Exception as e:
# #         print(e)
# #         message(e)
# # def handle_message_4(message):
# #     try:
# #         info=message['data'][0]
# #         opened=info['open']
# #         close=info['close']
# #         high=info['high']
# #         low=info['low']
# #         volume=info['volume']
# #         timeframe=str(datetime.fromtimestamp(int(message['ts'])/1000))
# #         timeframe=timeframe[:19]
# #         if int(timeframe[14:16])%my_timeframe==0 and str(timeframe[17:19])=='00': #int(timeframe[14:16])%5==0 and int(timeframe[14:16])%15==0 and
# #             print('Save to ADA-USDT-SWAP.csv')
# #             with open("ADA-USDT-SWAP.csv", mode="a", encoding='utf-8') as w_file:
# #                 file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
# #                 file_writer.writerow([timeframe, opened, high, low, close, volume])
# #     except Exception as e:
# #         print(e)
# #         message(e)
# try:
#     threading.Thread(target=ws.kline_stream, args=(my_timeframe, "BTCUSDT", handle_message_1, )).start()
# except Exception as e:
#     print(e, '1')
#     message(e)
# # try:
# #     threading.Thread(target=ws.kline_stream, args=(my_timeframe, "ETHUSDT", handle_message_2, )).start()
# # except Exception as e:
# #     print(e, '2')
# #     message(e)
# # try:
# #     threading.Thread(target=ws.kline_stream, args=(my_timeframe, "SOLUSDT", handle_message_3, )).start()
# # except Exception as e:
# #     print(e, '3')
# #     message(e)
# # try:
# #     threading.Thread(target=ws.kline_stream, args=(my_timeframe, "ADAUSDT", handle_message_4, )).start()
# # except Exception as e:
# #     print(e, '4')
# #     message(e)
#
# while True:
#     sleep(60)
#
# # #{'topic': 'kline.1.ETHUSDT', 'data': [{'start': 1714474440000, 'end': 1714474499999, 'interval': '1', 'open': '303
# # 8.14', 'close': '3037.13', 'high': '3038.14', 'low': '3036.35', 'volume': '220.56', 'turnover': '669809.5108', 'confirm': False, 'timestamp': 1714474460602}], 'ts': 1714474460602, 'type': 'snapshot'}

my_timeframe=5

def websocket(coin:str,my_timeframe:int,num:int ):
    while True:
        if int(str(datetime.now())[14:16]) % my_timeframe == 0:
            x=int(str(datetime.now())[14:16])+my_timeframe
            list_close = []
            while True:
                try:
                    marketDataAPI = MarketData.MarketAPI(flag=flag)
                    result = marketDataAPI.get_tickers(instType="SWAP")
                    close = float(result['data'][num]['askPx'])
                    timeframe = datetime.fromtimestamp(float(result['data'][12]['ts']) / 1000).strftime('%Y-%m-%d %H:%M:%S')
                    list_close.append(close)
                    # print(f'{x + time} == {int(str(timeframe)[14:16])}')
                    # print(f'X 1: {x}')
                    # print(f'X+my_timeframe: {x+my_timeframe}')
                    times=int(str(timeframe)[14:16])
                    if str(times) == '05':
                        times=5
                    elif str(times)=='00':
                        times=0
                    # print(list_close)
                    if times%my_timeframe==0:
                        x += 5
                        # print(f'X 2: {x}')
                        timeframe = str(timeframe)
                        high = str(max(list_close))
                        low = str(min(list_close))
                        opened = str(list_close[0])
                        close = str(list_close[-1])
                        print(f'Save to {coin}')
                        with open(coin, mode="a", encoding="utf-8") as w_file:
                            file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
                            file_writer.writerow([timeframe, opened, high, low, close])
                        list_close.clear()
                        sleep(60)
                    sleep(5)
                except Exception as e:
                    print(e)
                    message(e)
threading.Thread(target=websocket, args=("BTC-USDT-SWAP.csv", my_timeframe , 12,)).start()
threading.Thread(target=websocket, args=("ETH-USDT-SWAP.csv", my_timeframe , 37,)).start()
threading.Thread(target=websocket, args=("SOL-USDT-SWAP.csv", my_timeframe , 30, )).start()
# threading.Thread(target=websocket, args=("SOL-USDT-SWAP.csv", my_timeframe , 30, )).start()

# while True:
#     x = int(str(datetime.now())[14:16])
#     # print(x)
#     if x%time==0 : #and (str(x)[17:19]=='00' or str(x)[17:19]=='01')
#         while True:
#             try:
#                 marketDataAPI = MarketData.MarketAPI(flag=flag)
#                 result = marketDataAPI.get_tickers(instType="SWAP")
#                 close=float(result['data'][12]['askPx'])
#                 timeframe=datetime.fromtimestamp(float(result['data'][12]['ts'])/1000).strftime('%Y-%m-%d %H:%M:%S')
#                 list_close.append(close)
#                 if x+time==int(str(timeframe)[14:16]) or x+time==int(str(timeframe)[15:16]):
#                     timeframe=(timeframe)
#                     high=str(max(list_close))
#                     low=str(min(list_close))
#                     opened=str(list_close[0])
#                     close=str(list_close[-1])
#                     print('Save to BTC-USDT-SWAP.csv')
#                     with open("BTC-USDT-SWAP.csv", mode="a", encoding="utf-8") as w_file:
#                         file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
#                         file_writer.writerow([timeframe, opened, high, low, close])
#                     x=x+time
#                     list_close.clear()
#                 sleep(10)
#             except Exception as e:
#                 print(e)
#                 message(e)
