from datetime import datetime
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import ccxt
from backtesting import Strategy, Backtest

exchange = ccxt.okx()

# Название торговой пары (например, BTC/USDT)
# symbol = 'BTC/USDT'
# # Получение данных котировок за 10000 свечей
# candles = exchange.fetch_ohlcv(symbol, timeframe='5m',limit=2000 )
# pd.options.display.max_rows = 2000
# pd.set_option('display.max_rows', None)
# # Преобразование данных в pandas DataFrame
# df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
# Преобразование timestamp в удобочитаемый формат
# df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
# df.to_csv('BTC-USDT-SWAP.csv', index=False)

df = pd.read_csv("proverka\\ETH-USDT-SWAP.csv")
df=df[df['volume']!=0]
df=df[0:1000]
# print(len(df))
def isSwing(candle, window):
    if candle - window < 0 or candle + window >= len(df):
        return 0
    # print(candle, window)
    swingHigh = 1
    swingLow = 2
    for i in range(candle - window, candle + window + 1):
        if df.iloc[candle].low > df.iloc[i].low:
            swingLow = 0
        if df.iloc[candle].high < df.iloc[i].high:
            swingHigh = 0
    if (swingHigh and swingLow):
        return 3
    elif swingHigh:
        return swingHigh
    elif swingLow:
        return swingLow
    else:
        return 0
window=10
df['isSwing'] = df.apply(lambda x: isSwing(x.name,window), axis=1)

def pointpos(x):
    if x['isSwing']==2:
        return x['low']
    elif x['isSwing']==1:
        return x['high']
    else:
        return np.nan

df['pointpos'] = df.apply(lambda row: pointpos(row), axis=1)
# df.to_csv('break_structure.csv', index=False)

dfpl = df
fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                open=dfpl['open'],
                high=dfpl['high'],
                low=dfpl['low'],
                close=dfpl['close'])])

fig.add_scatter(x=dfpl.index, y=dfpl['pointpos'], mode="markers",
                marker=dict(size=6, color="MediumPurple"),
                name="Break")
fig.update_layout(xaxis_rangeslider_visible=True)
fig.show()
#
#
def detect_structure(candle, backcandles, window):

    localdf = df.iloc[
              candle - backcandles - window:candle - window]  # window must be greater than pivot window to avoid look ahead bias
    highs = localdf[localdf['isSwing'] == 1].high.tail(2).values
    lows = localdf[localdf['isSwing'] == 2].low.tail(2).values
    levelbreak = 0
    zone_width = 0.001
    if len(highs) == 2: #long
        resistance_condition = True
        mean_high = highs.mean()
        if resistance_condition and (df.loc[candle].close - mean_high) > zone_width * 2:
            levelbreak = 1

    if len(lows) == 2: #short
        support_condition = True
        mean_low = lows.mean()
        if support_condition and (mean_low - df.loc[candle].close) > zone_width * 2:
            levelbreak = 2
    return levelbreak
df['pattern_detected'] = df.apply(lambda row: detect_structure(row.name, backcandles=60, window=9), axis=1)
# print(df)
rslt_df_high = df[df['isSwing'] == 1]
rslt_df_low = (df[df['isSwing'] == 2])
high = rslt_df_high['pointpos'].iloc[-1]
low = rslt_df_low['pointpos'].iloc[-1]
close = df['close'].iloc[-1]
middle=(high+low)/2
print(rslt_df_high)
print(rslt_df_low)
print(high)
print(low)
print(close)
print(middle)
# #BACKTEST
# df = df.rename({'timestamp':'Local time','open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, axis=1)
# # df.index = pd.DatetimeIndex(df['Local time'])
# # print(df)
#
# def SIGNAL():
#     return df.pattern_detected
#
# class MyStrat(Strategy):
#     mysize = 100
#     slcoef = 0.99
#     TPSLRatio =3
#     def init(self):
#         super().init()
#         self.signal1 = self.I(SIGNAL)
#     def next(self):
#         super().next()
#         slcoef = self.slcoef
#         TPSLRatio= self.TPSLRatio
#         # Short
#         if self.signal1 == 1 and self.position.size==0:
#             print('-----------Long-------------')
#             print(self.data.index.values[-1]+1)
#             stop_loss = self.data.Close[-1] * slcoef
#             take_profit = ((self.data.Close[-1] - stop_loss) * TPSLRatio) + self.data.Close[-1]
#             print('TP ', take_profit)
#             print('Coin ',  self.data.Close[-1])
#             print('SL ', stop_loss)
#             self.buy(sl=round(float(stop_loss), 1),  tp=round(float(take_profit), 1))
#         # Long
#         if self.signal1 == 2  and self.position.size==0 :
#             print('-----------Short-------------')
#             print(self.data.index.values[-1] + 1)
#             stop_loss = self.data.Close[-1] * (slcoef+0.02)
#             print('SL ', stop_loss)
#             print('Coin ', self.data.Close[-1])
#             take_profit = self.data.Close[-1]-((stop_loss-self.data.Close[-1])*TPSLRatio)
#             print('TP ', take_profit)
#             self.sell(sl=round(float(stop_loss), 1),  tp=round(float(take_profit), 1))

# class MyStrat(Strategy):
#     mysize = 100
#     slcoef = 0.975
#     TPSLRatio =1.5
#     def init(self):
#         super().init()
#         self.signal1 = self.I(SIGNAL)
#     def next(self):
#         super().next()
#         slcoef = self.slcoef
#         TPSLRatio= self.TPSLRatio
#         # Short
#         if self.signal1 == 1 and self.position.size==0:
#             print('-----------Short-------------')
#             print(self.data.index.values[-1] + 1)
#             stop_loss = self.data.Close[-1] * (slcoef+0.05)
#             print('SL ', stop_loss)
#             print('Coin ', self.data.Close[-1])
#             take_profit = self.data.Close[-1]-((stop_loss-self.data.Close[-1])*TPSLRatio)
#             print('TP ', take_profit)
#             self.sell(sl=round(float(stop_loss), 1),  tp=round(float(take_profit), 1))
#             # print('-----------Short-------------')
#             # print(self.data.index.values[-1]+1)
#             # stop_loss = self.data.Close[-1] * slcoef
#             # take_profit = ((self.data.Close[-1] - stop_loss) * TPSLRatio) + self.data.Close[-1]
#             # print('SL ', stop_loss)
#             # print('Coin ',  self.data.Close[-1])
#             # print('TP ', take_profit)
#             # self.sell(sl=round(float(take_profit), 1),  tp=round(float(stop_loss), 1))
#         # Long
#         if self.signal1 == 2  and self.position.size==0 :
#             print('-----------Long-------------')
#             print(self.data.index.values[-1]+1)
#             stop_loss = self.data.Close[-1] * slcoef
#             take_profit = ((self.data.Close[-1] - stop_loss) * TPSLRatio) + self.data.Close[-1]
#             print('TP ', stop_loss)
#             print('Coin ',  self.data.Close[-1])
#             print('SL ', take_profit)
#             self.buy(sl=round(float(stop_loss), 1),  tp=round(float( take_profit), 1))
#             # print('-----------Long-------------')
#             # print(self.data.index.values[-1] + 1)
#             # stop_loss = self.data.Close[-1] * (slcoef+0.05)
#             # print('TP ', stop_loss)
#             # print('Coin ', self.data.Close[-1])
#             # take_profit = self.data.Close[-1]-((stop_loss-self.data.Close[-1])*TPSLRatio)
#             # print('SL ', take_profit)
#             # self.buy(sl=round(float(take_profit), 1),  tp=round(float(stop_loss), 1))
#
# bt = Backtest(df, MyStrat, cash=70000, commission=0.001)
#
# stats=bt.run()
# print(stats)
# bt.plot()

