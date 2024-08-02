import okx.Account as Account
import okx.Trade as trade
import requests
from time import sleep
import pandas as pd
import numpy as np
import threading
import datetime
#telegram
TOKEN = '6959314930:AAHnekjhCc2d_CHFLxE9hFWAZuIgQMD8wzY'
chat_id='947159905'
def message(x):
    message = (f'{x}')
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    requests.get(url).json()

# live trading: 0, demo trading: 1
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
#-------------------------------------------------
while True:
    try:
         #неделя
        result = tradeAPI.get_orders_history(
            instType="SWAP"
        )
        # pprint.pprint(result)
        date_extreme=datetime.datetime.fromtimestamp(int(result['data'][0]['fillTime'])/1000)
        # print(date_extreme)
        date_extreme=str(date_extreme)[:10]
        count_plus=0
        count_minus=0
        res_pnl=0
        # pprint.pprint(result['data'])
        for i in  result['data']:
            if i['pnl']!='0':
                # print(i)
                res_pnl+=float(i['pnl'])
                if float(i['pnl'])>0:
                    count_plus+=1
                if float(i['pnl'])<0:
                    count_minus+=1
        print(f'Суммарный pnl за неделю: {round(res_pnl, 2)} USDT\n')
        print(f'Win rate за неделю: {round((count_plus /(count_plus+count_minus))*100, 0)} %\n')
        print(f'Plus: {count_plus}\n'
              f'Minus: {count_minus}')
        message = (f'Суммарный pnl за неделю: {round(res_pnl, 2)} USDT\n'
                   f'Win rate за неделю: {round((count_plus /(count_plus+count_minus))*100, 0)} %\n'
                   f'Plus: {count_plus}\n'
                    f'Minus: {count_minus}')
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
        requests.get(url).json()
        sleep(604800)
    except Exception as e:
        print(e)
        message = (f'{e}')
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
        requests.get(url).json()
