
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import copy
plt.style.use('fivethirtyeight')

import yfinance as yf
df = yf.download('AZN', start="2018-12-01", end="2020-11-30")
df['Date'] = df.index

# =============================================================================
#  stock data visualizatoin
# =============================================================================
# plt.figure(figsize=(12,4))
# plt.plot(df['Close'], label='Close') 
# plt.xticks(rotation= 45)
# plt.title('Close Price History')
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Price USD ($)', fontsize=18)
# plt.show()

# calculate OBV
OBV = [] 
OBV.append(0)
for i in range(1, len(df.Close)):
    # If the closing price is above the prior close price 
    if df.Close[i] > df.Close[i-1]:
        # Then: Current OBV = Previous OBV + Current Volume 
        OBV.append(OBV[-1] + df.Volume[i])
    elif df.Close[i] < df.Close[i-1]:
        OBV.append(OBV[-1] - df.Volume[i])
    else:
        OBV.append(OBV[-1])

# OBV + Exponential Moving Average (EMA) 
df['OBV'] = OBV
df['OBV_EMA'] = df['OBV'].ewm(com=10).mean()


# =============================================================================
# OBV & OBV EMA visualization 
# =============================================================================
# plt.figure(figsize=(12,4)) 
# plt.plot(df['OBV'], label = 'OBV', color= 'orange')
# plt.plot(df['OBV_EMA'], label = 'OBV_EMA', color='purple')
# plt.xticks(rotation=45)
# plt.title('OBV/OBV_EMA')
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Price USD ($)', fontsize=18)
# plt.show()


# function for finding buy/sell timing signal 
# Buy signal OBV > OBV_EMA
# Sell signal OBV < OBV_EMA

def buy_sell(signal, col1, col2):
    flag = -1 # flag for the trend upward/downward 
    signal['sigPriceBuy'] = np.nan
    signal['sigPriceSell'] = np.nan
    for i in signal.index:
        # if OBV > OBV_EMA and flag != 1 then buy else sell 
        if signal[col1][i] > signal[col2][i] and flag != 1:
            signal.loc[signal.index == i, 'sigPriceBuy'] = signal['Close'][i]
            # signal.loc[signal.index == i, 'sigPriceSell'] = np.nan
            
            flag = 1
        # else if OBV < OBV_EMA and flag !=0 then sell else buy
        elif signal[col1][i] < signal[col2][i] and flag != 0:
            
            # signal.loc[signal.index == i,'sigPriceBuy'] = np.nan
            signal.loc[signal.index == i,'sigPriceSell'] = signal['Close'][i] #.copy()             
            flag = 0

# add buy/sell signal

buy_sell(df, 'OBV', 'OBV_EMA')

#%%
# buy/sell signal visualization
plt.figure(figsize=(12,4))
plt.scatter(df.index, df['sigPriceBuy'], color = 'green',
                            label='Buy_Signal', marker ='^', alpha=1)
plt.scatter(df.index, df['sigPriceSell'], color ='red',
                            label = 'Sell Signal', marker ='v', alpha=1)
plt.plot(df['Close'], label='Close Price', alpha = 0.35)
plt.xticks(rotation =0)
plt.title('The Stock Buy / Sell Signals')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.legend(loc='upper left')
plt.show()












