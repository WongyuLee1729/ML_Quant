#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
RSI는 가장 대표적인 모멘텀지표이며, MACD는 대표적인 추세지표임 
RSI는 추세의 강도와 방향을 알려주며, 이를 통해 현재 주가가 과매도/과매수인지 판단할 수 있음 
MACD는 두 이동평균선의 차이를 활용하여 주가의 추세 전환 신호를 얻을 수 있음 

RSI 
특징 : 1) 추세장에서는 신뢰도가 다소 떨어지고 박스권에서 잘 먹힘
      2) 매매 타이밍을 너무 빨리 잡는 경향이 있음 

MACD
특징 : 1) 추세가 강하게 형성된 구간일수록 더 신뢰도가 높아짐 
      2) 매매 타이밍을 한 발짝 늦게 잡는 경우가 종종 발생함
'''

'''
RSI + MACD 조합 

1. 이중 근거 
- MACD가 시그널 선을 상향 돌파 (골든 크로스) 하기 직전에 RSI도 시그널 선을 상향돌파(골든 크로스)하면 신뢰도가 높은 매수 신호 
- MACD가 시그널선을 하향돌파(데드 크로스) 하기 직전에 RSI도 시그널선을 하향돌파(데드 크로스) 하면 신뢰도 높은 매도 신호 

2. 포지션 철회 
- MACD 신호가 나온 후 가까운 시일 내에 RSI 신호가 반대로 나오면 포지션 정리 

3. 추세 강화 
- MACD 히스토그램이 양의 리테스팅을 보이고 RSI가 시그널선을 상향 돌파하면 높은 확률로 급등 
- MACD 히스토그램이 음의 리테스팅을 보이고 RSI가 시그널선을 상향 돌파하면 높은 확률로 급등 

'''
import pandas as pd
import numpy as np 
import copy
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


def SMA(data, period=30, column='Close'):
    '''
    Simple Moving Average, SMA
    '''
    return data[column].rolling(window= period).mean()


def EMA(data, period=20, column='Close'):
    '''
    Exponential Moving Average, EMA
    '''
    return data[column].ewm(span=period, adjust=False).mean()

def MACD(data, period_long=26, period_short=12, period_signal= 9, column='Close'):
    '''
    Moving Average Convergence & Divergence
    
    input:
        data :  OHLC dataframe
        period_long
        period_short
        period_signal
    output:
        data with MACD
        
    '''
    
    
    shortEMA = EMA(data, period_short, column=column)
    longEMA = EMA(data, period_long, column=column)
    data["MACD"] = shortEMA-longEMA
    data['signal_line'] = EMA(data, period_signal, column='MACD')
    return data

def RSI(data, period= 14, column='Close'):
    '''
    Relative Strength Index 
    input:
        data : OHLC dataframe
        period
    
    output:
        data with RSI
    '''
    
    delta = data[column].diff(1)
    delta = delta.dropna() 
    
    up = delta.copy()
    down = delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    data['up'] = up
    data['down'] = down
    
    avg_gain = SMA(data, period, column='up')
    avg_loss = abs(SMA(data, period, column = 'down'))
    RS = avg_gain / avg_loss
    data['RSI'] = 100.0 -(100.0/ 1.0+ RS) 
    return data


# if __name__ == "__main__":
    # import yfinance as yf 
    # df = yf.download('TSLA', start = '2019-12-01', end = '2020-11-30')
    # df['Date'] = df.index
    # df = MACD(df, period_long = 26, period_short = 12, period_signal = 9)
    # df = RSI(df, period=14)
    # df['SMA'] = SMA(df, period=30)
    # df['EMA'] = EMA(df, period=20)
    
    # # 이동 평균 수렴/발산과 신호선 시각화
    # column_list = ['MACD', 'signal_line', 'Close']
    # df[column_list].plot(figsize=(12,6))
    # plt.title("MACD")
    # plt.ylabel('Price')

    # # 단순 이동 평균선과 가격 데이터 시각화 
    # column_list = ['SMA','Close']
    # df[column_list].plot(figsize=(12,6))
    # plt.title("SMA")
    # plt.ylabel('Price')
    
    # # 지수 이동 평균선과 가격 데이터 시각화 
    # column_list = ['EMA', 'Close']
    # df[column_list].plot(figsize=(12,6))
    # plt.title('EMA')
    # plt.ylabel('Price')
    
    # # RSI 시각화 
    # column_list = ['RSI']
    # df[column_list].plot(figsize=(12,6))
    # plt.title('RSI')
    # plt.ylabel('Price')
    #%%
    # import pandas as pd
    # import numpy as np
    # import plotly.graph_objects as go
    # import yfinance as yf 
    # import plotly.io as pio
    # pio.renderers.default = "svg"

    # data = yf.download('TSLA', start = '2019-12-01', end = '2020-11-30')
    # data['Date'] = data.index
    
    # # path_dir = './data'
    # # data = np.loadtxt(path_dir + '/' + '000660_from_2010.csv', delimiter = ',')
    # # data = pd.DataFrame(data)
    # # data.columns = ['Open', 'High', "Low", "Close", "Volumn", "Adj"]
    # data = data[['Open', 'High', "Low", "Close", "Volume", "Adj Close"]]
    
    # def cal_ema_macd(data, n_fast=12, n_slow=26, n_signal=9) : 
    #     data["EMAFast"] = data["Close"].ewm(span=n_fast).mean()
    #     data["EMASlow"] = data["Close"].ewm(span=n_slow).mean()
    #     data["MACD"] = data["EMAFast"] - data["EMASlow"]
    #     data["MACDSignal"] = data["MACD"].ewm(span=n_signal).mean()
    #     data["MACDDiff"] = data["MACD"] - data["MACDSignal"]
        
    #     return data
        
    # data = cal_ema_macd(data)
    
    # fig = go.Figure()
    
    # fig.add_trace(go.Scatter(x=data.index, y=data["Close"],
    #                         mode='lines',
    #                         name="Close"))
    # fig.update_layout(title='Close',
    #                    xaxis_title='days',
    #                    yaxis_title='StockValue')
    
    # fig.show()
    
    # fig = go.Figure()
    
    # fig.add_trace(go.Scatter(x=data.index, y=data["MACD"],
    #                         mode='lines',
    #                         name="MACD"))
    # fig.add_trace(go.Scatter(x=data.index, y=data["MACDSignal"],
    #                         mode='lines',
    #                         name="MACDSignal"))
    # fig.add_trace(go.Bar(x=data.index, y=data["MACDDiff"],
    #                     name="MACD DIff",
    #                      width=2.5,
    #                     marker_color='Black'))
    # fig.add_trace(go.Scatter(x=data.index, y=np.zeros(len(data.index)),name='0',
    #                         line = dict(color='gray', width=2, dash='dot')))
    
    # fig.update_layout(title='MACD 16 26 9',
    #                    xaxis_title='days',
    #                    yaxis_title='MACD')
    
    # fig.show()
#%%


# -*- coding: utf-8 -*-
"""
Moving Average Convergence/Divergence (MACD) of Stock Prices
 
https://en.wikipedia.org/wiki/MACD
 
Author:
    Aleksandar Haber
 
Date: January 27, 2021
Some parts of this code are inspired by the following sources:
    
-- https://en.wikipedia.org/wiki/MACD
-- https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
-- "Learn Algorithmic Trading: Build and deploy algorithmic trading systems and strategies
using Python and advanced data analysis", by Sebastien Donadio and Sourav Ghosh
"""
 
"""
Definitions
     
    -- MACD: 
         
            MACD is the difference between a fast (short period) exponential moving average and 
            a slow (long-period) exponential moving average of time series.
     
    -- Signal or average:
     
            Signal or average is an exponential moving average of the computed MACD.
     
    -- Divergence (histograms or bars):
         
            Difference between the MACD and signal.
             
Usual notation:
     
    MACD(a,b,c)
     
    -- a - short period 
     
    -- b - long period 
     
    -- c - signal period
     
    commonly used values:
         
    MACD(12,26,9), that is a=12, b=26, and c=9
         
"""
 
# standard imports
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
# used to download the stock prices
import pandas_datareader as pdr
 
 
# MACD parameters
short_period = 12
long_period  = 26
signal_period = 9
 
# # define the dates for downloading the data
# startDate= '2020-01-01'
# endDate= '2021-01-28'
 
# # Read about Python pickle file format:  https://www.datacamp.com/community/tutorials/pickle-python-tutorial
# fileName = 'downloadedData.pkl'
 
# stock_symbol='SPY'
 
# # this piece of code either reads the data from the saved file or if the saved file does not exist 
# # it downloads the data 
 
# try:
#     data=pd.read_pickle(fileName)
#     print('Loading the data from the local disk.')
# except FileNotFoundError:
#     print('The data is not found on the local disk.')
#     print('Downloading the data from Yahoo.')
#     data = pdr.get_data_yahoo(stock_symbol,start=startDate,end=endDate)
#     # save the data to file
#     print('Saving the data to the local disk.')
#     data.to_pickle(fileName)


import yfinance as yf 
# import plotly.io as pio
# pio.renderers.default = "svg"

data = yf.download('TSLA', start = '2019-12-01', end = '2020-12-30')
data['Date'] = data.index

# inspect the data
 
data.head()
 
# data['Adj Close'].plot()

# isolate the closing price
closingPrice = data['Adj Close']


ewm_short=data['Adj Close'].ewm(span=short_period, adjust=False).mean()
ewm_long=data['Adj Close'].ewm(span=long_period, adjust=False).mean()
MACD=ewm_short-ewm_long
signal_MACD=MACD.ewm(span=signal_period, adjust=False).mean()
bars=MACD-signal_MACD
# bar_values=bars.values
# bar_index_number=np.arange(0,len(bar_values))

# with plt.style.context('ggplot'):
#     import matplotlib
#     font = { 'weight' : 'bold', 'size'   : 10}
#     matplotlib.rc('font', **font)
#     fig = plt.figure(figsize=(10,30))
    
    
#     ax1=fig.add_subplot(311, ylabel='Stock price $')
#     data['Adj Close'].plot(ax=ax1,color='r',lw=2,label='Close price',legend=True)
#     ewm_short.plot(ax=ax1,color='b',lw=2,label=str(short_period)+'-exp moving average',legend=True)
#     ewm_long.plot(ax=ax1,color='m',lw=2,label=str(long_period)+'-exp moving average',legend=True)
    
#     ax2=fig.add_subplot(312, ylabel='MACD')
#     MACD.plot(ax=ax2,color='k',lw=2,label='MACD',legend=True)
#     signal_MACD.plot(ax=ax2,color='r',lw=2,label='Signal',legend=True)
    
#     ax3=fig.add_subplot(313, ylabel='MACD bars')
#     x_axis = ax3.axes.get_xaxis()
#     x_axis.set_visible(False)
#     bars.plot(ax=ax3,color='r', kind='bar',label='MACD minus signal', legend=True,use_index=False)
#     # plt.savefig('MACD_spy.png')
    
#     plt.subplots_adjust(left=0, bottom=0.1, right=0, top=0, wspace=0.4, hspace=0.4)
#     plt.show()






# =============================================================================
# Working on it ...
# =============================================================================
with plt.style.context('ggplot'):
    import matplotlib
    font = { 'weight' : 'bold', 'size'   : 10}
    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(10,30))
    
    fig, axs = plt.subplots(3,1, sharex=True)
    
    
    
    
    
    ax1=fig.add_subplot(311, ylabel='Stock price $')
    data['Adj Close'].plot(ax=ax1,color='r',lw=2,label='Close price',legend=True)
    ewm_short.plot(ax=ax1,color='b',lw=2,label=str(short_period)+'-exp moving average',legend=True)
    ewm_long.plot(ax=ax1,color='m',lw=2,label=str(long_period)+'-exp moving average',legend=True)
    
    ax2=fig.add_subplot(312, ylabel='MACD')
    MACD.plot(ax=ax2,color='k',lw=2,label='MACD',legend=True)
    signal_MACD.plot(ax=ax2,color='r',lw=2,label='Signal',legend=True)
    
    ax3=fig.add_subplot(313, ylabel='MACD bars')
    x_axis = ax3.axes.get_xaxis()
    x_axis.set_visible(False)
    bars.plot(ax=ax3,color='r', kind='bar',label='MACD minus signal', legend=True,use_index=False)
    # plt.savefig('MACD_spy.png')
    
    plt.subplots_adjust(left=0, bottom=0.1, right=0, top=0, wspace=0.4, hspace=0.4)
    plt.show()

















