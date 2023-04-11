# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd 
# from pandas_datareader import data as web
from datetime import datetime 
import pandas_datareader as pdr
import matplotlib.pyplot as plt

tickers = ['AAPL', 'F', 'AMZN', 'GE', 'TSLA']

pxclose = pd.DataFrame()

start_date = datetime(2019,1,1)
end_date =datetime(2021,12,31)

for t in tickers:
    pxclose[t] = pdr.DataReader(t, data_source='yahoo', start=start_date, end=end_date)['Adj Close']

ret_daily = pxclose.pct_change()
ret_annual = ret_daily.mean() * 250
cov_daily = ret_daily.cov()
cov_annual = cov_daily*250
p_returns = []
p_volatility = []
p_weights = []

n_assets = len(tickers)
n_ports = 30000 
for s in range(n_ports):
    wgt = np.random.random(n_assets)
    wgt /=np.sum(wgt)
    ret = np.dot(wgt, ret_annual)
    vol = np.sqrt(np.dot(wgt.T, np.dot(cov_annual, wgt)))
    p_returns.append(ret)
    p_volatility.append(vol)
    p_weights.append(wgt)
    
p_volatility = np.array(p_volatility)
p_returns = np.array(p_returns)

colors = np.random.randint(0, n_ports, n_ports)

plt.style.use('seaborn')
plt.scatter(p_volatility, p_returns, c= colors, marker='o')
plt.xlabel("Volatility (Std. Deviation)")
plt.ylabel('Expected Returns')
plt.title('Efficient Frontier')
plt.show()