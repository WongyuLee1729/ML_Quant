#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


def equal_weight_portfolio(f1: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
  ew_portfolio = f1.copy().fillna(0)*0
  
  for date, row in f1.iterrows():
      row.dropna(inplace =True)
      row = row.sort_values(ascending=False)[0:top_n]
      ew_portfolio.loc[date,row.index]= 1/top_n
  return ew_portfolio


def market_weight_portfolio(f1: pd.DataFrame, market_val: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    mw_portfolio = f1.copy().fillna(0)*0
    market_weight = market_val.fillna(0)
    market_weight = market_weight.div(market_weight.sum(axis=1), axis=0)
    
    for date, row in f1.iterrows():
      row.dropna(inplace =True)
      row = row.sort_values(ascending=False)[0:top_n]
      mw_portfolio.loc[date,row.index]= market_weight.loc[date,row.index]
      mw_portfolio.loc[date,row.index] = mw_portfolio.loc[date,row.index].div(market_weight.loc[date,row.index].sum(), axis=0)
    return mw_portfolio


def get_long_short_portfolio(f1: pd.DataFrame, percentile: float = 0.25, long_ratio: float = 1.5, short_ratio: float = 0.5) -> pd.DataFrame:
    long_ratio = 1.5
    short_ratio = -0.5
    ls_portfolio = f1.copy().fillna(0)*0
    
    for date, row in f1.iterrows():
        row.dropna(inplace =True)
        row = row.sort_values(ascending=False)
        ratio_val = int(np.round(len(row)*0.25))
        long = row.sort_values(ascending=False)[0:ratio_val]
        short = row.sort_values(ascending=True)[0:ratio_val]        
        ls_portfolio.loc[date,long.index]= 1/ratio_val*long_ratio
        ls_portfolio.loc[date,short.index]= 1/ratio_val*short_ratio
        ls_portfolio.loc[date,:] = ls_portfolio.loc[date,:].div(ls_portfolio.loc[date,:].sum(), axis=0)
    return ls_portfolio




def calculate_metrics(portfolio: pd.DataFrame, total_return: pd.DataFrame, benchmark_return: pd.DataFrame) -> dict:
    portfolio_rtn = total_return.loc[portfolio.index,portfolio.columns]*portfolio
    portfolio_rtn = portfolio_rtn.sum(axis=1)
    portfolio_cum_rtn = (1+portfolio_rtn).cumprod()
    bm_rtn = benchmark_return.squeeze() # change type to Series
    bm_cum_rtn = (1+bm_rtn).cumprod()
    
    port_cagr = portfolio_cum_rtn.iloc[-1]**(12./len(mkt_weight_cum_rtn.index)) -1
    port_sharpe = np.mean(portfolio_rtn)/np.std(portfolio_rtn)*np.sqrt(12.)
    port_mdd = np.min(portfolio_cum_rtn/portfolio_cum_rtn.expanding().max()) -1
    
    bm_cagr = bm_cum_rtn.iloc[-1]**(12./len(bm_cum_rtn.index)) -1
    bm_sharpe = np.mean(bm_rtn)/np.std(bm_rtn)*np.sqrt(12.)
    bm_mdd = np.min(bm_cum_rtn/bm_cum_rtn.expanding().max()) -1
    
    return {'port_CAGR': port_cagr, 'port_Sharpe': port_sharpe, 'port_MDD': port_mdd, 'bm_CAGR': bm_cagr, 'bm_Sharpe':bm_sharpe , 'bm_MDD':bm_mdd}
