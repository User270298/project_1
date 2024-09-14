import csv
import time
import logging
import threading
import requests
from datetime import datetime
import okx.MarketData as MarketData
import okx.Account as Account
import okx.Trade as Trade

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
print('Start')
# Загрузка переменных окружения
TOKEN = '6959314930:AAHnekjhCc2d_CHFLxE9hFWAZuIgQMD8wzY'
CHAT_ID='947159905'

API_KEY='43f5df59-5e61-4d24-875e-f32c003e0430'
SECRET_KEY='5B1063B322635A27CF01BACE3772E0E0'
PASSPHRASE='Parkwood270298)'
FLAG = "1"  # live trading: 0, demo trading: 1

# Инициализация API
accountAPI = Account.AccountAPI(API_KEY, SECRET_KEY, PASSPHRASE, False, FLAG)
tradeAPI = Trade.TradeAPI(API_KEY, SECRET_KEY, PASSPHRASE, False, FLAG)

def send_telegram_message(message):
    """ Отправка сообщений в Telegram """
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={CHAT_ID}&text={message}"
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        logging.error(f"Ошибка при отправке сообщения в Telegram: {e}")

def websocket(coin, my_timeframe, num):
    """ Получение данных через WebSocket и запись в CSV """
    list_close = []
    marketDataAPI = MarketData.MarketAPI(flag=FLAG)
    csv_file_path = f"{coin}"

    # with open(csv_file_path, mode="a", newline='', encoding="utf-8") as file:
    #     file_writer = csv.writer(file, delimiter=",")
    #     file_writer.writerow(["timestamp", "open", "high", "low", "close"])

    while True:
        try:
            result = marketDataAPI.get_tickers(instType="SWAP")
            close = float(result['data'][num]['askPx'])
            list_close.append(close)
            current_time = datetime.fromtimestamp(float(result['data'][num]['ts']) / 1000)

            if current_time.minute % my_timeframe == 0:
                if list_close:  # Проверяем, есть ли данные
                    high = max(list_close)
                    low = min(list_close)
                    opened = list_close[0]
                    closed = list_close[-1]
                    logging.info(f'Save to {coin}')
                    with open(csv_file_path, mode="a", newline='', encoding="utf-8") as file:
                        file_writer = csv.writer(file, delimiter=",")
                        file_writer.writerow([str(current_time)[:-7], opened, high, low, closed])
                    list_close.clear()
                time.sleep(65)  # Пауза, чтобы избежать повторения записи
            time.sleep(5)  # Частота проверки
        except Exception as e:
            logging.error(e)
            send_telegram_message(f'Error in websocket connection for {coin}: {e}')

# Запуск потоков для разных криптовалют
coins = [("BTC-USDT-SWAP.csv", 5, 12), ("ETH-USDT-SWAP.csv", 5, 37), ("LTC-USDT-SWAP.csv", 5 , 10), ("OP-USDT-SWAP.csv", 5 , 48)]
for coin, timeframe, num in coins:
    threading.Thread(target=websocket, args=(coin, timeframe, num)).start()
