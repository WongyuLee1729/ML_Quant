import numpy as np
import pandas as pd 

class stock_preprop:
    ''' 
    input data df output data numpy array
    makes 7:3 ratio train & test set
    time-series testing dataset
      '''
    def __init__(self, df:pd.DataFrame, row_w, col_w, stride):
        self.df = df
        self.row_w = row_w
        self.col_w = col_w
        self.stride = stride
        self.ratio = np.ceil(len(self.df)/(len(self.df)*3/10))
        # self.tickers = df.columns

    def RowCol(self):
        x_train = np.empty((0, self.row_w))
        y_train = np.empty((0, 12)) # (0,1)
        x_test = np.empty((0, self.row_w))
        y_test = np.empty((0, 12)) # (0,1)
        cnt = 0
        
        for _, value in self.df.iterrows():
            cnt += 1
            data = value.values
            x_start = 0
            x_end = self.row_w
            tmp = np.empty((0, self.row_w))
            while True:
                if x_end >= len(data):
                    break; # tmp.size = (13,30) -> 390
                tmp = np.vstack((tmp, data[x_start:x_end]))
                x_start += self.stride
                x_end += self.stride       
    
            
            if cnt%self.ratio == 0:
                x_test = np.vstack((x_test, tmp[:-1,:]))
                cum_rtn = ((1+tmp[1:,:self.col_w]).cumprod(axis=1)-1)[:,-1]
                cum_rtn =cum_rtn.reshape(1,len(cum_rtn)) # tmp[1:,:self.col_w].std(1)
                # std_rtn = tmp[1:,:self.col_w].std(1)
                # y_test = np.vstack((y_test, np.vstack((cum_rtn, std_rtn)).T))
                rtn = np.triu(np.ones((cum_rtn.shape[1],cum_rtn.shape[1])),0)*np.vstack([cum_rtn]*cum_rtn.shape[1])
                y_test = np.vstack((y_test, rtn))
                # y_test = np.vstack((y_test, tmp[1:,:]))
                
            else:
                x_train = np.vstack((x_train, tmp[:-1,:]))
                cum_rtn = ((1+tmp[1:,:self.y_w]).cumprod(axis=1)-1)[:,-1]
                cum_rtn =cum_rtn.reshape(1,len(cum_rtn))
                # std_rtn = tmp[1:,:self.y_w].std(1)
                # y_train = np.vstack((y_train, np.vstack((cum_rtn, std_rtn)).T))
                rtn = np.triu(np.ones((cum_rtn.shape[1],cum_rtn.shape[1])),0)*np.vstack([cum_rtn]*cum_rtn.shape[1])
                
                y_train = np.vstack((y_train, rtn))
            
        return [x_train,y_train, x_test,y_test]
