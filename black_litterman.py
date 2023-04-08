import numpy as np
import pandas as pd 
from pandas_datareader import data as web
import matplotlib.pyplot as plt
import matplotlib as mpl 
from numpy.linalg import inv

def Black_Litterman(df, r_bm, r_f, w_mkt, tau, q,p):
    '''
    df: data 
    r_bm: return for bench mark
    r_f: Risk-free rate of return
    w_mkt: market weight (market cap ratio)
    tau: measure of uncertainty
    q: view vector
    p: the assets involved in the views
    '''
    sigma = df.cov()
    bm_cov = np.dot(np.dot(w_mkt.T,sigma),w_mkt)
    lamda = (np.dot(w_mkt,r_bm)-0.00098)/bm_cov
    pi = lamda*np.dot(sigma,w_mkt)
    omega = np.dot(np.dot(p,tau*sigma),p.T)
    inv_tausig = inv(tau*sigma)
    p_invomg = np.dot(p.T, inv(omega))
    er = np.dot(inv(inv_tausig + np.dot(p_invomg,p)),(np.dot(inv_tausig,pi)+np.dot(p_invomg,q)))
    w_bl = np.dot(inv(lamda*sigma),er-r_f)
    w_bl = w_bl+abs(np.min(w_bl))+1
    w_bl = w_bl/np.sum(w_bl)
    return w_bl

# Optimization process is needed 
