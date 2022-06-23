import os 
import glob
import pandas as pd
import numpy as np
import datetime
import random
# import matplotlib.pyplot as plt


def data_preprocessing(sample, ticker, base_date):   
    sample['CODE'] = ticker # 종목코드 추가
    sample = sample[sample['Date'] >= base_date][['Date','CODE','Adj Close']].copy() # 기준일자 이후 데이터 사용
    sample.reset_index(inplace= True, drop= True)
    # 기준년월 
    sample['STD_YM'] = sample['Date'].map(lambda x : datetime.datetime.strptime(x,'%Y-%m-%d').strftime('%Y-%m')) 
    # strptime: str 타입으로 된 날짜 데이터를 datetime 형태로 변환, strftime(): datetime 형태의 데이터를 str 타입으로 변환
    sample['1M_RET'] = 0.0 # 수익률 컬럼
    ym_keys = list(sample['STD_YM'].unique()) # 중복 제거한 기준년월 리스트
    return sample, ym_keys


def create_trade_book(sample, sample_codes):
    book = pd.DataFrame()
    book = sample[sample_codes].copy()
    book['STD_YM'] = book.index.map(lambda x : datetime.datetime.strptime(x,'%Y-%m-%d').strftime('%Y-%m'))
    for c in sample_codes:
        book['p '+c] = ''
        book['r '+c] = ''
    return book


# 상대모멘텀 tradings
def tradings(book, s_codes):
    std_ym = ''
    buy_phase = False
    # 종목코드별 순회
    for s in s_codes : 
        print(s) # s는 ticker 임
        # 종목코드 인덱스(=날짜) 순회 
        for i in book.index:
            # 해당 종목코드 포지션을 잡아준다. # 이전에 buy가 아니지만 바로 다음에 buy 신호(ready+s)가 있다면 
            if book.loc[i,'p '+s] == '' and book.shift(1).loc[i,'p '+s] == 'ready ' + s:
                std_ym = book.loc[i,'STD_YM'] # 년 월까지만 표시 됨 , 반면에 i는 년월일까지 표시 됨
                buy_phase = True
            # 해당 종목코드에서 신호가 잡혀있으면 매수상태를 유지한다. # 왜 두개의 if 문으로 나누어 놓는지모르겠음 .. 
            if book.loc[i,'p '+s] == '' and book.loc[i,'STD_YM'] == std_ym and buy_phase == True : 
                book.loc[i,'p '+s] = 'buy ' + s
            
            if book.loc[i,'p '+ s] == '' :
                std_ym = None
                buy_phase = False
    return book


def multi_returns(book, s_codes):
    # 손익 계산
    rtn = 1.0
    buy_dict = {}
    num = len(s_codes)
    sell_dict = {}
    
    for i in book.index:
        for s in s_codes:
            if book.loc[i, 'p ' + s] == 'buy '+ s and book.shift(1).loc[i,'p '+s] == 'ready '+s and book.shift(2).loc[i, 'p '+s] == '' :     # long 진입
                buy_dict[s] = book.loc[i, s]
#                 print('진입일 : ',i, '종목코드 : ',s ,' long 진입가격 : ', buy_dict[s])
            elif book.loc[i, 'p '+ s] == '' and book.shift(1).loc[i, 'p '+s] == 'buy '+ s: # long 청산
                sell_dict[s] = book.loc[i, s]
                # 손익 계산
                rtn = (sell_dict[s] / buy_dict[s]) -1
                book.loc[i, 'r '+s] = rtn
                print('개별 청산일 :',i,' 종목코드 :', s , 'long 진입가격 :', buy_dict[s],'|long 청산가격 : ',sell_dict[s],' | return:', round(rtn * 100, 2),'%') # 수익률 계산.
            if book.loc[i, 'p '+ s] == '':     # zero position || long 청산.
                buy_dict[s] = 0.0
                sell_dict[s] = 0.0
    acc_rtn = 1.0        
    for i in book.index:
        rtn  = 0.0
        count = 0
        for s in s_codes:
            if book.loc[i, 'p '+ s] == '' and book.shift(1).loc[i,'p '+ s] == 'buy '+ s: 
                # 청산 수익률계산.
                count += 1
                rtn += book.loc[i, 'r '+s]
        if (rtn != 0.0) & (count != 0) :
            acc_rtn *= (rtn /count )  + 1
            print('누적 청산일 : ',i,'청산 종목수 : ',count,'청산 수익률 : ',round((rtn /count),4),'누적 수익률 : ' ,round(acc_rtn, 4)) # 수익률 계산.
        book.loc[i,'acc_rtn'] = acc_rtn
    print ('누적 수익률 :', round(acc_rtn, 4))

#종목 데이터 읽어오기.
files = glob.glob('C:/Users/wongyu Lee/Desktop/algoTrade-master/data/us_etf_data/*.csv')

# 필요한 데이터 프레임 생성
# Monthly 데이터를 저장하기 위함이다.
month_last_df = pd.DataFrame(columns=['Date','CODE','1M_RET'])
# 종목 데이터 프레임 생성
stock_df = pd.DataFrame(columns =['Date','CODE','Adj Close'])

for file in files:
    """
    데이터 저장 경로에 있는 개별 종목들을 읽어온다.
    """
    if os.path.isdir(file):
        print('%s <DIR> '%file)
    else:
        folder, name = os.path.split(file)
        head, tail = os.path.splitext(name)
        print(file)
        read_df = pd.read_csv(file) # 경로를 읽은 데이터를 하나씩 읽어들인다.        
        # 1단계. 데이터 가공
        price_df, ym_keys = data_preprocessing(read_df,head,base_date='2010-01-02')
        # 가공한 데이터 붙이기.
        stock_df = stock_df.append(price_df.loc[:,['Date','CODE','Adj Close']],sort=False)
        # 월별 상대모멘텀 계산을 위한 1개월간 수익률 계산
        for ym in ym_keys:
            m_ret = price_df.loc[price_df[price_df['STD_YM'] == ym].index[-1],'Adj Close']/price_df.loc[price_df[price_df['STD_YM'] == ym].index[0],'Adj Close'] 
            # 주어진 기간 (ym) 중 가장 처음값 index[0]과 가장 마지막 값 index[-1] 의 비율 [-1]/[0] 
            price_df.loc[price_df['STD_YM'] == ym, ['1M_RET']] = m_ret
            month_last_df = month_last_df.append(price_df.loc[price_df[price_df['STD_YM'] == ym].index[-1],['Date','CODE','1M_RET']])    

# 2단계. 상대모멘텀 수익률로 filtering 하기.
month_ret_df = month_last_df.pivot('Date','CODE','1M_RET').copy() # df.pivot(index='Date', columns='CODE', values='1M_RET')
month_ret_df = month_ret_df.rank(axis=1, ascending=False, method="max", pct=True) # 투자종목 선택할 rank 각 행별로 가장 낮은 값을 1부터 순차적으로 내려감
# 상위 40%에 드는 종목들만 Signal list.
month_ret_df = month_ret_df.where( month_ret_df < 0.4 , np.nan) # 0.4 보다 큰 값들은 전부 np.nan으로 바꾸어줌
month_ret_df.fillna(0,inplace=True)
month_ret_df[month_ret_df != 0] = 1
stock_codes = list(stock_df['CODE'].unique()) # ticker 정보 수집 

# 3단계. signal list로 trading + positioning
sig_dict = dict()
for date in month_ret_df.index:
    ticker_list = list(month_ret_df.loc[date,month_ret_df.loc[date,:] >= 1.0].index)
    sig_dict[date] = ticker_list
stock_c_matrix = stock_df.pivot('Date','CODE','Adj Close').copy()
book = create_trade_book(stock_c_matrix, list(stock_df['CODE'].unique()))

for date,values in sig_dict.items():
    for stock in values:
        book.loc[date,'p '+ stock] = 'ready ' + stock
        
# 3-2  tradings
book = tradings(book, stock_codes)

# 4 단게. 수익률 계산하기.
multi_returns(book, stock_codes)


book.tail()





