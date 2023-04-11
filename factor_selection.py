#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd 
import os
import numpy as np 


factor_data = pd.read_csv('data.csv', index_col = 0)

total_rtn = pd.read_csv('data.csv', index_col = 0)

def get_weight_portfolio(factor_data: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
  ew_portfolio = factor_data.copy().fillna(0)*0
  
  for date, row in factor_data.iterrows():
      row.dropna(inplace =True)
      row = row.sort_values(ascending=False)[0:top_n]
      ew_portfolio.loc[date,row.index]= 1/top_n
  return ew_portfolio



def get_selected_factors() -> list: # Get out the important factors using PCA
    path = 'drive/data/'
    file_list = os.listdir(path)
    file_list_py = [file for file in file_list if file.endswith('.csv')] 
    factor_dict = {}
    factor_return_list =  pd.DataFrame(index = factor_data.index)
    for i in file_list_py:
        factor_dict[i.split('.')[0]] = pd.read_csv(path + i, index_col = 0)
        factor_portfolio = get_weight_portfolio(factor_dict[i.split('.')[0]] , 50)
        factor_return = total_rtn.loc[factor_portfolio.index, factor_portfolio.columns]*factor_portfolio
        factor_return = factor_return.sum(axis=1) 
        factor_return_list[i.split('.')[0]] = factor_return
 
    cov_matrix = np.cov(factor_return_list.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
    # We reduce dimension to 1, since 1 eigenvector has 85% (enough) variances
    # print('1st PC eigenvalue percentage :', round(eig_vals[0]/sum(eig_vals)*100,2),'%')
    eig_vec_list = pd.DataFrame(data = eig_vecs, index= factor_return_list.columns)
    selected_factors = eig_vec_list.iloc[:,0].sort_values(ascending=False)[0:20]
    return selected_factors.index

