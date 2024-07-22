import okx.Account as Account
import okx.Trade as trade
import pprint
import requests
import datetime
from time import sleep
import pandas as pd
import numpy as np

#OKX demo trading
api_key='43f5df59-5e61-4d24-875e-f32c003e0430'
secret_key='5B1063B322635A27CF01BACE3772E0E0'
passphrase='Parkwood270298)'
flag = "1"

#OKX live trading
# api_key='f8bcadcc-bed3-4fca-96e7-4f314f43136b'
# secret_key='F56CF3942B876FDEDEF547C90B04F206'
# passphrase='Parkwood270298)'
# flag = "0"

accountAPI = Account.AccountAPI(api_key, secret_key, passphrase, False, flag)
tradeAPI = trade.TradeAPI(api_key, secret_key, passphrase, False, flag)

result=tradeAPI.get_order_list()
pprint.pprint(result['data'])

# result = accountAPI.get_positions()
# print(len(result['data']))
# result = tradeAPI.place_order(
#                         instId='ETH-USDT-SWAP',
#                         tdMode="isolated",
#                         side="buy",
#                         posSide="long",
#                         ordType="limit",
#                         sz='1',
#                         px=3380,
#                         tpTriggerPx=4000,  # take profit trigger price
#                         tpOrdPx="-1",  # taker profit order price。When it is set to -1，the order will be placed as an market order
#                         tpTriggerPxType="last",
#                         slTriggerPx=3000,      # take profit trigger price
#                         slOrdPx="-1",           # taker profit order price。When it is set to -1，the order will be placed as an market order
#                         slTriggerPxType="last"
#                     )
# print(result)

# df = pd.read_csv("BTC-USDT-SWAP.csv")
# print(len(df))
# df = df.iloc[1:, :]
#
# # df.drop(index=2)
# print(len(df))