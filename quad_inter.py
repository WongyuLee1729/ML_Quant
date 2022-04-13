
import numpy as np 
import pandas as pd
# import sympy
from sympy import Derivative, symbols, expand
# import matplotlib.pyplot as plt


def quad_interpol(ls_val, x_inpt):
    fcn = 0
    for x0 in val.keys():
        y0 = ls_val[x0]
        btm = 1
        top = 1
        for x in val.keys():
            if x != x0:
                top = top*(x_inpt-x)
                btm = btm*(x0-x)
        tmp = y0*top/btm 
        fcn = fcn + tmp
    return fcn


# Data x:y
val ={ 11: 0.3079, 12: 0.2978, 13: 0.2872, 14: 0.2762,15: 0.2651,16:0.2540,17: 0.2429,18: 0.2319,19: 0.2211, \
           20: 0.2105,21: 0.2003,22: 0.1903,23: 0.1806,24: 0.1713,25: 0.1624, 26: 0.1538, 27: 0.1455, 28: 0.1376, \
           29: 0.1301,30: 0.1229,31: 0.1160,32: 0.1095,33: 0.1033,34: 0.0974,35: 0.0918,36: 0.0864,37: 0.0814,38: 0.0766, 39: 0.0721, \
           40: 0.0679,41: 0.0638,42: 0.0600,43: 0.0564,44: 0.0530,45: 0.0498,46: 0.0468,47: 0.0439, 48: 0.0413}

# check the function     

X = symbols('X')
fn = quad_interpol(val, X)
ans = expand(fn)