import okx.Account as Account
import okx.Trade as trade
import pprint
import datetime
import requests
from time import sleep

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
# result = tradeAPI.get_orders_history(
#     instType="SWAP"
# )
risk=5
foulder=20

# percent_sz = ((risk / (foulder * ((close - stop) / close))) * 100) / close
result = tradeAPI.place_order(
                instId="ETH-USDT-SWAP",
                tdMode="isolated",
                side="buy",
                posSide="long",
                ordType="market",
                sz=5.6,
                # ccy='50',
                tpTriggerPx='4000',  # take profit trigger price
                tpOrdPx="-1",  # taker profit order price。When it is set to -1，the order will be placed as an market order
                tpTriggerPxType="last",
                slTriggerPx='3000',      # take profit trigger price
                slOrdPx="-1",           # taker profit order price。When it is set to -1，the order will be placed as an market order
                slTriggerPxType="last",
                clOrdId='1',
                # tgtCcy = "base_ccy"
            )
print(result)
# result=accountAPI.get_positions()
# list_coins=[]
# if len(result['data'])!=0:
#     for i in range(len(result['data'])):
#         res=result['data'][i]['instId']
#         list_coins.append(res)
#         # print(i)
# cd='BTC-USDT-SWAP'
# print(cd not in list_coins)


#Закрыть позицию
# res=tradeAPI.close_positions(instId="BTC-USDT-SWAP", mgnMode='isolated', posSide="long")
# # ordId=result['data'][0]['ordId']
# print(res)

# #{'accFillSz': '10', 'algoClOrdId': '', 'algoId': '', 'attachAlgoClOrdId': '', 'attachAlgoOrds': [], 'avgPx': '60626.1', 'cTime': '1715366309473', 'cancelSource': '', 'cancelSourceReason': '', 'category': 'normal', 'ccy': '', 'clOrdId': '', 'fee': '-0.3031305', 'feeCcy': 'USDT', 'fillPx': '60626.1', 'fillSz': '10', 'fillTime': '1715366309474', 'instId': 'BTC-USDT-SWAP', 'instType': 'SWAP', 'isTpLimit': 'false', 'lever': '20.0', 'linkedAlgoOrd': {'algoId': ''}, 'ordId': '1438274135692328960', 'ordType': 'market', 'pnl': '18.916', 'posSide': 'short', 'px': '', 'pxType': '', 'pxUsd': '', 'pxVol': '', 'quickMgnType': '', 'rebate': '0', 'rebateCcy': 'USDT', 'reduceOnly': 'true', 'side': 'buy', 'slOrdPx': '', 'slTriggerPx': '', 'slTriggerPxType': '', 'source': '', 'state': 'filled', 'stpId': '', 'stpMode': 'cancel_maker', 'sz': '10', 'tag': '', 'tdMode': 'isolated', 'tgtCcy': '', 'tpOrdPx': '', 'tpTriggerPx': '', 'tpTriggerPxType': '', 'tradeId': '940364590', 'uTime': '1715366309475'}

# while True:
#     try:
#         sleep(604800)  #неделя
#         result = tradeAPI.get_orders_history(
#             instType="SWAP"
#         )
#         # pprint.pprint(result)
#         date_extreme=datetime.datetime.fromtimestamp(int(result['data'][0]['fillTime'])/1000)
#         # print(date_extreme)
#         date_extreme=str(date_extreme)[:10]
#         count_plus=0
#         count_minus=0
#         res_pnl=0
#         # pprint.pprint(result['data'])
#         for i in  result['data']:
#             if i['pnl']!='0':
#                 # print(i)
#                 res_pnl+=float(i['pnl'])
#                 if float(i['pnl'])>0:
#                     count_plus+=1
#                 if float(i['pnl'])<0:
#                     count_minus+=1
#         print(f'Суммарный pnl за неделю: {round(res_pnl, 2)} USDT\n')
#         print(f'Win rate за неделю: {round((count_plus /(count_plus+count_minus))*100, 0)} %\n')
#         print(f'Plus: {count_plus}\n'
#               f'Minus: {count_minus}')
#         message = (f'Суммарный pnl за неделю: {round(res_pnl, 2)} USDT\n'
#                    f'Win rate за неделю: {round((count_plus /(count_plus+count_minus))*100, 0)} %\n'
#                    f'Plus: {count_plus}\n'
#                     f'Minus: {count_minus}')
#         url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
#         requests.get(url).json()
#     except Exception as e:
#         print(e)
#         message = (f'{e}')
#         url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
#         requests.get(url).json()