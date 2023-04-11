# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 10:12:12 2022

@author: wongyu Lee
"""

import numpy as np
import cvxpy as cp
import pandas as pd
from pandas_datareader import data as web
import warnings
from typing import Optional
from matplotlib import pyplot as plt
from datetime import datetime
# from Model.PortfolioModel import portfolio_calc
'''

risk_type : 0 (초저위험), 1 (저위험), 2(중위험), 3(고위험), 4(초고위험) 
start_date : YYYY-MM-DD  로드할 date range의 시작 날짜
end_date : YYYY-MM-DD 로드할 date range의 종료 날짜
target_date : YYY-MM-DD 포트폴리오를 산출하고자 하는 날짜, 일반적으로 end_date와 동일 


def get_daily_mp(risk_type: int, start_date: str, end_date: str, taget_date: str) -> pd.DataFrame:
      # ...
    return pd.DataFrame(data=data, colums=["Date", tickers])


def get_strategy_info() -> Tuple:
      # ...
    return (abbreviation, fullname)

'''


def mean_variance(mean_vec: pd.DataFrame, cov_mat: pd.DataFrame, v_lambda, no_short=True,
                     opt: Optional[dict] = None):
    """
    Solve the following formulation with tradeoff as the objective (lambda required)
        min  E(r) - lambda * 0.5 * Var(r)
        s.t. [constraints]
    (Code uses cvxpy)
    """
    n = mean_vec.shape[1]
    w = cp.Variable((n, 1))
    objective = cp.Maximize(cp.matmul(mean_vec, w) - 0.5 * v_lambda * cp.quad_form(w, cov_mat))
    constraints = [sum(w) == 1]
    if no_short:
        constraints.append(w >= 0)
    if opt is not None:
        if 'lowerbound' in opt:
            constraints.append(w >= opt['lowerbound'])
        if 'upperbound' in opt:
            constraints.append(w <= opt['upperbound'])

    problem = cp.Problem(objective, constraints)
    problem.solve(verbose=False, solver=cp.CVXOPT)

    if problem.status == "optimal":
        w_opt = np.array(w.value).flatten()
    else:
        warnings.warn('mean_variance not optimized, running optimize_gmv')
        # Find GMV if solution not found
        w_opt = optimize_gmv(cov_mat, no_short=no_short, opt=opt)

    return w_opt


def optimize_gmv(cov_mat: pd.DataFrame, no_short=True, opt: Optional[dict] = None):
    """
    GMV portfolio optimization
    (Code uses cvxpy)
    """
    n = cov_mat.shape[1]
    w = cp.Variable((n, 1))
    objective = cp.Minimize(cp.quad_form(w, cov_mat))
    constraints = [sum(w) == 1]
    if no_short:
        constraints.append(w >= 0)
    if opt is not None:
        if 'lowerbound' in opt:
            constraints.append(w >= opt['lowerbound'])
        if 'upperbound' in opt:
            constraints.append(w <= opt['upperbound'])

    problem = cp.Problem(objective, constraints)
    problem.solve(verbose=False, solver=cp.CVXOPT)

    if problem.status == "optimal":
        w_opt = np.array(w.value).flatten()
    elif opt is not None:
        warnings.warn('optimize_gmv not optimized, running without options')
        # Find GMV without options
        w_opt = optimize_gmv(cov_mat, no_short=no_short)
    else:
        warnings.warn('optimize_gmv not optimized, returning 1/n')
        # Otherwise, return equal weights
        w_opt = np.repeat(1 / n, n)

    return w_opt



# daily_rtn = returns(sectclose.loc[sectclose.index.year==i], shyclose.loc[shyclose.index.year==i],risk_list, 'DR',mvo,risk_Lv)

def returns(df,safe_list:str, mid_list:str, risk_list:str, return_type:str, weights,risk_level=None):
    '''
    DR : Daily returns 
    DCR : Daily compound returns (TWRR)
    MR : Monthly returns
    W : weights
    '''
    estimated_price = df.dot(weights.values[0])
    daily_rtn = estimated_price.pct_change().dropna()
    
    if return_type == 'EP':
        return estimated_price
    
    if return_type == 'DR':
        return daily_rtn
        
    if return_type == 'DCR':
        daily_compound_rtn = (1+daily_rtn).cumprod()-1
        return daily_compound_rtn
    
    if return_type == 'MR':
        month_end = pd.date_range(estimated_price.index[0],estimated_price.index[-1], freq='BM')
        monthly_rtn = estimated_price[estimated_price.index.isin(month_end)] #.reset_index(drop=True)
        monthly_rtn = monthly_rtn.pct_change().dropna()
        return monthly_rtn
    
    if return_type == 'W':
        
        if risk_level == 'high':
            weights[safe_list] = 0
            weights[risk_list] = weights[risk_list]+ weights[risk_list]*0.3
            weights = weights.div(weights.sum(axis=1),axis=0)
            
        elif risk_level == 'mid':
            weights[safe_list] = weights[safe_list]+weights[safe_list]*0.3        
            weights = weights.div(weights.sum(axis=1),axis=0)
            
        elif risk_level == 'low':
            weights[risk_list] = 0
            weights[safe_list] = weights[safe_list]+weights[safe_list]*0.5
            weights = weights.div(weights.sum(axis=1),axis=0)    
    
        else:
            print('Risk Level Error!')
        
        return weights 
        
    if return_type == 'W_level':
        w_level = pd.concat(
            [weights[risk_list[-2:]].sum(axis=1),
            weights[risk_list[:-2]].sum(axis=1),
            weights[mid_list].sum(axis=1),
            weights[safe_list] ],axis=1)
        w_level.rename({0:'4', 1:'3', 2:'2' ,'SPTS':'1','BSV':'0'},axis=1,inplace=True)
        return w_level



def get_daily_mp(risk_type: int, start_date: str, end_date: str) -> pd.DataFrame:
    
    tickers = ['XLC' ,'XLY' ,'XLP', 'XLE', 'XLF' ,'XLV' ,'XLI' ,'XLB', 'XLRE' ,'XLK' ,'XLU', 'EEM', 'IEMG','VWO','SCHE','FNDE','QAI','MNA','SPTS','BSV']   
    mid_list = ['XLC' ,'XLY' ,'XLP', 'XLE', 'XLF' ,'XLV' ,'XLI' ,'XLB', 'XLRE' ,'XLK' ,'XLU']
    risk_list = ['EEM', 'IEMG','VWO','SCHE','FNDE','QAI','MNA'] # high_risk = ['QAI','MNA']
    safe_list = ['SPTS','BSV'] # high_safe = ['BSV']
    
    if risk_type == 0:
        return print('risk_type 0 is not exist!')
    
    if risk_type == 4:
        return print('risk_type 4 is not exist!')
    
    if risk_type==1:
        risk_Lv= 'low'
    elif risk_type==2:
        risk_Lv= 'mid'
    elif risk_type == 3:
        risk_Lv = 'high'
        
    # wht = 0
    close_price = pd.DataFrame()
    tmpclose = pd.DataFrame()
    tmp_weight = pd.DataFrame(columns = tickers, index = ['0'])
    # tmp_weight.columns = tickers
    
    for t in tickers:
        close_price[t] = web.DataReader(t, data_source='yahoo', start= start_date, end= end_date)['Adj Close']
    
    opt = {}
    opt['lowerbound'] = 0.02 #/len(tickers)
    opt['upperbound'] = 0.8 # np.round(len(tickers)/2)/len(tickers)
    # quarter = 21*6 = 126
    # idx_list = []
    w = [1/3,1/3, 1/3]
    
    weightdf = close_price*np.nan
    start = 0
    end =21
    if end >len(close_price):
        return print('Minimum days required for this model is 21')
    while True:
        tmpclose = close_price.iloc[start:end,:].pct_change().dropna()     
        expect_df = pd.DataFrame(index= ['high','mid','low'], columns= tmpclose.columns)      
        idx = tmpclose.index[-1]
    
        for k in tmpclose.columns:
            tmp_df = tmpclose[k]
            expect_df[k].loc['high'] = tmp_df.max() 
            expect_df[k].loc['mid'] = tmp_df.median() 
            expect_df[k].loc['low'] = tmp_df.min()
    
        expected_rtn = pd.DataFrame(np.dot(w,expect_df.values).reshape(1,len(tickers)) ,columns= close_price.columns)    
        cov_mat = tmpclose.cov()
        tmp_weight[tmp_weight.index == '0'] = mean_variance(expected_rtn, cov_mat, 0.6, opt=opt) 
        port_weight = returns(close_price,safe_list, mid_list, risk_list, 'W', tmp_weight,risk_Lv)
        
        diff = np.round(1.00- np.array(port_weight.values[0], dtype=np.float64).round(2).sum(),3)
        port_weight[port_weight.index == '0'] = np.array(port_weight.values[0], dtype=np.float64).round(2)
        port_weight.iloc[0,port_weight.values.argmax()] = port_weight.iloc[0,port_weight.values.argmax()]+diff
        # print(port_weight.sum(axis=1))
        weightdf[weightdf.index == idx] = port_weight.values
        
        start += 1
        end += 1
        
        if end > len(close_price):
            break; # tmp.size = (13,30) -> 390
    
    weightdf.index = weightdf.index.strftime('%Y-%m-%d')
    result = weightdf[weightdf.index == end_date]
    
    return result[result != 0].dropna(axis=1)


def get_strategy_info() -> tuple:
    '''
    Mean-Variance model 
    
    '''
    abbreviation = 'M 1.0' 
    fullname = 'Mean-Variance model 1.0'
    return (abbreviation, fullname)


#%%
'''
Sample trial
'''

risk_type = 3 # 1~3
start_date= '2020-01-01'
end_date = '2020-01-31'
# target_date = '2020-04-23'


sample = get_daily_mp(risk_type, start_date, end_date)

info = get_strategy_info()









