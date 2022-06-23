# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
from preprocessing import stock_preprop
import torch
from sklearn.model_selection import train_test_split
train = True

M_start = '09:29:00' # 10:00/09:30
M_end = '16:00:00'
ticker_lst = ['AAPL'] # , 'MSFT', 'AMZN']

# ['AAPL','AMGN', 'AXP', 'BA' , 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW', 'GS',  
# 'HD', 'HON','IBM' ,'INTC' ,'JNJ',
# 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT'  ]


# /Users/wongyui/Desktop/1dcnn/processed_price/AAPL

path = 'C:/Users/wongyu Lee/Desktop/1d_cnn_predict/processed_prices/'
file_list = os.listdir(path)
file_list_py = [file for file in file_list if file.endswith('.csv')]
# tr_df = pd.DataFrame()
df = pd.DataFrame()
for i in file_list_py:
    # if i == 'AAPL.csv':
    #     tr_df = pd.read_csv(path+i)
    # else:
    data = pd.read_csv(path + i)
    df = pd.concat([df,data])

# tr_df = df.reset_index(drop = True)
df = df.reset_index(drop = True)

# tr_df.rename(columns = {df.columns[0]:'Date'}, inplace= True)
# tr_df.set_index('Date',inplace=True)
df.rename(columns={df.columns[0]:'Date'},inplace=True)
df.set_index('Date', inplace = True)
df = df[df.index >= 20180102]
df = df.pct_change(axis=1).dropna(axis=1)
df_train = df.loc[:,:M_start]
df_test = df.loc[:,M_start:M_end]


x_train, x_test, y_train, y_test = train_test_split(df_train.values, df_test.values, test_size=0.33, random_state=42)
# # X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.33, random_state=42)


# (Batch, feature dimension, time_step)

x_train = torch.FloatTensor(x_train)#.to(DEVICE).unsqueeze(1)*100
y_train = torch.FloatTensor(y_train)#.to(DEVICE)*100 #.unsqueeze(1)
x_test = torch.FloatTensor(x_test)#.to(DEVICE).unsqueeze(1)*100
y_test = torch.FloatTensor(y_test)#.to(DEVICE)*100 #.unsqueeze(1)




