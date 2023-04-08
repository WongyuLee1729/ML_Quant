import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import warnings
import os
# import FinanceDataReader as fdr
# from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import glob
import datetime
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Lambda
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error
# matplotlib inline
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'NanumGothic'


'''

Re-balancing Model using Multi-Armed Bandit + LSTM

'''


def data_preprocessing(sample, ticker, base_date):   
    sample['CODE'] = ticker 
    sample = sample[sample['Date'] >= base_date][['Date','CODE','Close']].copy() 
    sample.reset_index(inplace= True, drop= True)
    sample['STD_YM'] = sample['Date'].map(lambda x : datetime.datetime.strptime(x,'%Y-%m-%d').strftime('%Y-%m')) 
    sample['1M_RET'] = 0.0 
    ym_keys = list(sample['STD_YM'].unique()) 
    return sample, ym_keys


def create_trade_book(sample, sample_codes, rebal_dates):
    book = pd.DataFrame()
    book = sample[sample_codes].copy()

    rebal_dates = list(month_thompson.index)
    new_idx = [] 
    cnt = 0

    for i in book.index:
        new_idx.append(rebal_dates[cnt])
        
        if (cnt+1) < (len(rebal_dates)):
            if rebal_dates[cnt+1] == i:
                cnt += 1

    book['STD_YM'] = new_idx
    for c in sample_codes:
        book['p '+c] = ''
        book['r '+c] = ''
    return book

# Relative Strength Momentum
def tradings(book, s_codes):
    std_ym = ''
    buy_phase = False
    for s in s_codes : 
        for i in book.index:
            if book.loc[i,'p '+s] == '' and book.shift(1).loc[i,'p '+s] == 'ready ' + s:
                std_ym = book.loc[i,'STD_YM']
                buy_phase = True 
            if book.loc[i,'p '+s] == '' and book.loc[i,'STD_YM'] == std_ym and buy_phase == True : 
                book.loc[i,'p '+s] = 'buy ' + s
            
            if book.loc[i,'p '+ s] == '' :
                std_ym = None
                buy_phase = False
    return book


def multi_returns(book, s_codes):
    rtn = 1.0
    buy_dict = {}
    num = len(s_codes)
    sell_dict = {}
    
    for i in book.index:
        for s in s_codes:
            if book.loc[i, 'p ' + s] == 'buy '+ s and book.shift(1).loc[i,'p '+s] == 'ready '+s and book.shift(2).loc[i, 'p '+s] == '' :  
                buy_dict[s] = book.loc[i, s]
            elif book.loc[i, 'p '+ s] == '' and book.shift(1).loc[i, 'p '+s] == 'buy '+ s: # 
                sell_dict[s] = book.loc[i, s]
                rtn = (sell_dict[s] / buy_dict[s]) -1
                book.loc[i, 'r '+s] = rtn
                # print('sell date :',i,' stock code :', s , 'long buy price :', buy_dict[s],'|long sell price : ',sell_dict[s],' | return:', round(rtn * 100, 2),'%') 
            if book.loc[i, 'p '+ s] == '': 
                buy_dict[s] = 0.0
                sell_dict[s] = 0.0
    acc_rtn = 1.0        
    for i in book.index:
        rtn  = 0.0
        count = 0
        for s in s_codes:
            if book.loc[i, 'p '+ s] == '' and book.shift(1).loc[i,'p '+ s] == 'buy '+ s: 
                count += 1
                rtn += book.loc[i, 'r '+s]
        if (rtn != 0.0) & (count != 0) :
            acc_rtn *= (rtn /count )  + 1
            # print('cum_sell date : ',i,'sell stock # : ',count,'sell_rtn : ',round((rtn /count),4),'cum_rtn : ' ,round(acc_rtn, 4)) 
        book.loc[i,'acc_rtn'] = acc_rtn
    # print ('cum_rtn :', round(acc_rtn, 4))


def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def windowed_dataset(series, window_size, batch_size, shuffle):
    
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))

    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.map(lambda w: (w[:-1], w[-1])) 
    return ds.batch(batch_size).prefetch(1) 


def lstm_model(WINDOW_SIZE):
    model = Sequential([
        # 1차원 feature map 생성
        Conv1D(filters=32, kernel_size=5,
               padding="causal",
               activation="relu",
               input_shape=[WINDOW_SIZE, 1]),
        # LSTM
        LSTM(16, activation='tanh'),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    return model

def bollinger_band(price_df, rolln, sigma,name):
    bb = pd.DataFrame(columns=[name+'_center', name+'_ub', name+'_lb'])
    bb[name+'_vol'] = price_df['Volume']
    bb[name+'_center'] = price_df['Volume'].rolling(rolln).mean()
    bb[name+'_ub'] = bb[name+'_center']+ sigma*price_df['Volume'].rolling(rolln).std()
    bb[name+'_lb'] = bb[name+'_center']-sigma*price_df['Volume'].rolling(rolln).std() 
    bb.set_index(price_df['Date'], inplace=True)
    return bb 


#%%
set_date = '2011-02-01' # '2012-01-03'
ratio = 9
dates = '2017-01-03'
WINDOW_SIZE = 20
BATCH_SIZE = 64
tkrs = {}
train = False
rolln = 20
sigma = 2
vb = pd.DataFrame()

files = glob.glob('data_link')
month_last_df = pd.DataFrame(columns=['Date','CODE','1M_RET'])
stock_df = pd.DataFrame(columns =['Date','CODE','Close'])

for file in files:
    if os.path.isdir(file):
        print('%s <DIR> '%file)
    else:
        folder, name = os.path.split(file)
        head, tail = os.path.splitext(name)
        print(file)
        read_df = pd.read_csv(file) 
        tmp_vb = bollinger_band(read_df, rolln, sigma,head)
        vb[tmp_vb.columns] = tmp_vb
        price_df, ym_keys = data_preprocessing(read_df,head,base_date='2010-01-02')
        stock_df = stock_df.append(price_df.loc[:,['Date','CODE','Close']],sort=False)

        for ym in ym_keys:
            m_ret = price_df.loc[price_df[price_df['STD_YM'] == ym].index[-1],'Close']/price_df.loc[price_df[price_df['STD_YM'] == ym].index[0],'Close'] 
            price_df.loc[price_df['STD_YM'] == ym, ['1M_RET']] = m_ret
            month_last_df = month_last_df.append(price_df.loc[price_df[price_df['STD_YM'] == ym].index[-1],['Date','CODE','1M_RET']])    


# Stage 2. filtering based on Relative Strength Momentum
month_ret_df = month_last_df.pivot('Date','CODE','1M_RET').copy() 
month_ret_df = month_ret_df.rank(axis=1, ascending=False, method="max", pct=True) 
month_ret_df = month_ret_df.where( month_ret_df < ratio/10 , np.nan) 
month_ret_df.fillna(0,inplace=True)
month_ret_df[month_ret_df != 0] = 1
stock_codes = list(stock_df['CODE'].unique())

univ = stock_df.pivot('Date','CODE','Close').copy()
univ['STD_YM'] = univ.index.map(lambda x : datetime.datetime.strptime(x,'%Y-%m-%d').strftime('%Y'))
univ = univ.loc[set_date:]
vb.fillna(0, inplace=True)
vb = vb.loc[set_date:]
# univ.fillna(0, inplace=True)

daily_reward = univ[univ.columns[:-1]].pct_change()*100
daily_reward.drop([daily_reward.index[0]],inplace=True)
daily_reward.fillna(-100,inplace=True)
daily_reward = daily_reward.rank(axis=1, ascending=False, method="max", pct=True) 

no_daily_reward = daily_reward.copy() 
no_daily_reward = no_daily_reward.where( no_daily_reward >= ratio/10 , np.nan) 
no_daily_reward.fillna(0,inplace=True)

daily_reward = daily_reward.where( daily_reward < ratio/10 , np.nan)
daily_reward.fillna(0,inplace=True)
daily_reward = 1-daily_reward
daily_reward[daily_reward == 1] = 0


reward = pd.DataFrame(columns = univ.columns[:-1],index = ['T'] )
reward.fillna(0, inplace=True)

answer = reward.copy()
no_answer = reward.copy()

no_reward = reward.copy()

gamma = 0.9


month_thompson = pd.DataFrame(columns= reward.columns)
old_slot = 0

tmp_cnt = 0 

#%%

model = lstm_model(WINDOW_SIZE)
loss = Huber()
optimizer = Adam(0.0005)
model.compile(loss=Huber(), optimizer=optimizer, metrics=['mse'])
earlystopping = EarlyStopping(monitor='val_loss', patience=10)


if train:
    print('Training Phase')
    for tkr in daily_reward.columns:
        x_train, x_test, y_train, y_test = train_test_split( vb[tkr+'_center'], univ[tkr], test_size=0.2, random_state=0, shuffle=False)
        y_train = y_train/y_train.max(axis=0)
        y_test = y_test/y_test.max(axis=0)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size =0.1, shuffle=False)
        train_data = windowed_dataset(y_train, WINDOW_SIZE, BATCH_SIZE, True)
        val_data = windowed_dataset(y_val, WINDOW_SIZE, BATCH_SIZE, True)
        test_data = windowed_dataset(y_test, WINDOW_SIZE, BATCH_SIZE, False)
        filename = os.path.join('tmp', 'ckeckpointer'+tkr+'.ckpt')
        checkpoint = ModelCheckpoint(filename, save_weights_only=True, save_best_only=True, monitor='val_loss', verbose=1)
        history = model.fit(train_data, validation_data=(val_data),epochs=50, callbacks=[checkpoint, earlystopping])
        model.load_weights('tmp\ckeckpointer'+tkr+'.ckpt')
else:
    print('Testing Phase')
    for tk in daily_reward.columns:
        x_train, x_test, y_train, y_test = train_test_split( vb[tk+'_center'], univ[tk], test_size=0.2, random_state=0, shuffle=False)
        y_train = y_train/y_train.max(axis=0)
        y_test = y_test/y_test.max(axis=0)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size =0.1, shuffle=False)
        train_data = windowed_dataset(y_train, WINDOW_SIZE, BATCH_SIZE, True)
        val_data = windowed_dataset(y_val, WINDOW_SIZE, BATCH_SIZE, True)
        test_data = windowed_dataset(y_test, WINDOW_SIZE, BATCH_SIZE, False)
        model.load_weights('tmp\ckeckpointer'+tk+'.ckpt')


pred = model.predict(test_data)
tkrs[tk] = pd.DataFrame(pred, index= y_test.index[WINDOW_SIZE:]) 

test_start = y_test.index[WINDOW_SIZE] 


# =============================================================================
# Multi-Armed Bandit
# =============================================================================

for d in univ.index[1:]: 
    max_val = 0
    ticker = 0 
    tmp_dates = d
    old_ticker = set()
    
    answer.loc['T'] = 0.9*answer.loc['T']+daily_reward.loc[d] 
    no_answer.loc['T'] = 0.9*no_answer.loc['T']+no_daily_reward.loc[d]
    
    if d<= test_start: # training
        reward.loc['T'] = 0.9*reward.loc['T']+daily_reward.loc[d]
        no_reward.loc['T'] = 0.9*no_reward.loc['T']+ no_daily_reward.loc[d]
        # rebal_dist = reward.loc['T'].values
    else:    
        for r in range(ratio):
            for n in univ.columns[:-1]: # stock list loop 

                for x in range(4):
                    slot = random.betavariate(reward.loc['T',n] + 1, no_reward.loc['T',n] + 1) 
                    old_slot = old_slot +slot
                slot = old_slot/4
                
                if slot > max_val:
                    max_val = slot
                    ticker = n 
                    old_ticker.add(ticker)
            
            if ticker in old_ticker:
                
                if float(tkrs[n].loc[d].values)>0.6  and vb.loc[d, n+'_vol'] > vb.loc[d, n+'_ub']:              
                    reward.loc['T',ticker] = reward.loc['T',ticker] + 2.0*float(tkrs[n].loc[d].values) 
                    answer.loc['T'] = answer.loc['T'] + 2.0*float(tkrs[n].loc[d].values)
                    
                elif float(tkrs[n].loc[d].values)<0.4  and vb.loc[d, n+'_vol'] > vb.loc[d, n+'_ub']:               
                    reward.loc['T',ticker] = reward.loc['T',ticker] - 1.0*float(tkrs[n].loc[d].values)
                    answer.loc['T'] = answer.loc['T'] - 1*float(tkrs[n].loc[d].values)
                    
                else:     
                    reward.loc['T',ticker] = reward.loc['T',ticker] + float(tkrs[n].loc[d].values) #+ volf 
                
            elif ticker not in old_ticker:
                no_reward.loc['T',ticker] = no_reward.loc['T',ticker]+ 0.2*no_daily_reward.loc[d,ticker] #+ volf
            
        reward = reward*0.9
        no_reward = no_reward*0.9

        if kl_divergence(answer.values/answer.values.sum()+0.01, reward.values/reward.values.sum()+0.01)>0.4 or d == dates: 
            # tmp_cnt = 0 
            print('rebalancing date :',d)
            month_thompson.loc[d] = reward.loc['T'].copy()
            # rebal_dist.loc['T'] = reward.loc['T'].values.copy()  
            reward.loc['T'] = answer.loc['T'].copy()
            no_reward.loc['T'] = no_answer.loc['T'].copy()

month_thompson = month_thompson.rank(axis = 1, ascending=False, method="max", pct=True)  
month_thompson = month_thompson.where( month_thompson < ratio/10 , np.nan)
month_thompson.fillna(0, inplace=True)
month_thompson[month_thompson != 0] = 1


# Stage 3. Using signal list -> trading + positioning
sig_dict = dict()
for date in month_thompson.index[1:]: # 1st one is not for rebal
    ticker_list = list(month_thompson.loc[date,month_thompson.loc[date,:] >= 1.0].index)
    sig_dict[date] = ticker_list
stock_c_matrix = stock_df.pivot('Date','CODE','Close').copy()
stock_c_matrix = stock_c_matrix[dates:]
book = create_trade_book(stock_c_matrix, list(stock_df['CODE'].unique()), list(month_thompson.index))

for date,values in sig_dict.items():
    for stock in values:
        book.loc[date,'p '+ stock] = 'ready ' + stock
        
book = tradings(book, stock_codes)

# Stage 4. Calculate revenue
multi_returns(book, stock_codes)

book.tail()




