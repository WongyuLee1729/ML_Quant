
import numpy as np
import pandas as pd 

'''
Price data preprocessing code

X: dates (x-axis in dataframe) 
Y: ticker_name (y-axis in dataframe)

'''

class stock_preprop:
    ''' 
    input data: df 
    output data: numpy array
    input shape: n x m matrix for each stock where .. 
    n: date 
    m: price change 
    self.ratio: train/test ratio = 7:3 default
      '''
    def __init__(self, df_tr:pd.DataFrame, df_te:pd.DataFrame, x_w, y_w, stride):
        self.df_tr = df_tr
        self.df_te = df_te
        self.x_w = x_w
        self.y_w = y_w
        self.stride = stride
        self.ratio = np.ceil(len(self.df_tr)/(len(self.df_te)*3/10)) # train/test ratio = 7:3
        # self.tickers = df.columns

    def XY_TS(self,):
        '''
        predict 
        '''
        if len(self.df_tr) != len(self.df_te):
            return print('Train Test data size mismatch!')
        else:
            x_train = np.empty((0, self.x_w))
            y_train = np.empty((0, self.y_w)) # (0,1)
            x_test = np.empty((0, self.x_w))
            y_test = np.empty((0, self.y_w)) # (0,1)
            cnt = 0
            for (_,tr),(_,te) in zip(self.df_tr.iterrows(),self.df_te.iterrows()):
            # for _, value in self.df.iterrows():
                cnt += 1
                x_data = tr.values
                x_start = 0
                x_end = self.x_w
                
                y_data = te.values
                y_start = 0 
                y_end = self.y_w
                
                x_tmp = np.empty((0, self.x_w))
                y_tmp = np.empty((0, self.y_w))
                
                while True:
                    if x_end > len(x_data):  # since len(x) == len(y)
                        break; # tmp.size = (13,30) -> 390
                    x_tmp = np.vstack((x_tmp, x_data[x_start:x_end]))
                    y_tmp = np.vstack((y_tmp, y_data[y_start:y_end]))
                    
                    x_start += self.stride
                    x_end += self.stride       
        
                    y_start += self.stride
                    y_end += self.stride       
        
                if cnt%self.ratio == 0:
                    x_test = np.vstack((x_test, x_tmp))
                    y_test = np.vstack((y_test, y_tmp))
                    
                    
                else:
                    x_train = np.vstack((x_train, x_tmp))
                    y_train = np.vstack((y_train, y_tmp))
            
            return [x_train,y_train, x_test,y_test]


    def XY_basic(self):
        '''
        predict the basic price change
        '''
        x_train = np.empty((0, self.x_w))
        y_train = np.empty((0, self.y_w)) # (0,1)
        x_test = np.empty((0, self.x_w))
        y_test = np.empty((0, self.y_w)) # (0,1)
        cnt = 0
        
        for _, value in self.df.iterrows():
            cnt += 1
            data = value.values
            x_start = 0
            x_end = self.x_w
            tmp = np.empty((0, self.x_w))
            while True:
                if x_end >= len(data):
                    break; # tmp.size = (13,30) -> 390
                tmp = np.vstack((tmp, data[x_start:x_end]))
                x_start += self.stride
                x_end += self.stride       
            
            if cnt%self.ratio == 0:
                x_test = np.vstack((x_test, tmp[:-1,:]))      
                y_test = np.vstack((y_test, tmp[1:,:]))
                
            else:
                x_train = np.vstack((x_train, tmp[:-1,:]))
                y_train = np.vstack((y_train, tmp[1:,:]))
            
        return [x_train,y_train, x_test,y_test]

    def XY_CUM(self):
        '''
        predict cumulative return
        '''
        x_train = np.empty((0, self.x_w))
        y_train = np.empty((0, 1)) 
        x_test = np.empty((0, self.x_w))
        y_test = np.empty((0, 1)) 
        cnt = 0

        for _, value in self.df.iterrows():
            cnt += 1
            data = value.values
            x_start = 0
            x_end = self.x_w
            tmp = np.empty((0, self.x_w))
            while True:
                if x_end >= len(data):
                    break; 
                tmp = np.vstack((tmp, data[x_start:x_end]))
                x_start += self.stride
                x_end += self.stride       
    
            if cnt%self.ratio == 0:
                x_test = np.vstack((x_test, tmp[:-1,:]))
                cum_rtn = ((1+tmp[1:,:self.y_w]).cumprod(axis=1)-1)[:,-1]
                y_test = np.vstack((y_test, cum_rtn ))
                
            else:
                x_train = np.vstack((x_train, tmp[:-1,:]))
                cum_rtn = ((1+tmp[1:,:self.y_w]).cumprod(axis=1)-1)[:,-1]
                y_train = np.vstack((y_train, cum_rtn)) 
        return [x_train,y_train, x_test,y_test]


    def XY_CUM_STD(self):
        '''
        predict cumulative return and std
        '''
        x_train = np.empty((0, self.x_w))
        y_train = np.empty((0, 2)) # (0,1)
        x_test = np.empty((0, self.x_w))
        y_test = np.empty((0, 2)) # (0,1)
        cnt = 0
        
        for _, value in self.df.iterrows():
            cnt += 1
            data = value.values
            x_start = 0
            x_end = self.x_w
            tmp = np.empty((0, self.x_w))
            while True:
                if x_end >= len(data):
                    break; # tmp.size = (13,30) -> 390
                tmp = np.vstack((tmp, data[x_start:x_end]))
                x_start += self.stride
                x_end += self.stride       
    
            if cnt%self.ratio == 0:
                x_test = np.vstack((x_test, tmp[:-1,:]))
                cum_rtn = ((1+tmp[1:,:self.y_w]).cumprod(axis=1)-1)[:,-1]
                std_rtn = tmp[1:,:self.y_w].std(1)
                y_test = np.vstack((y_test, np.vstack((cum_rtn, std_rtn)).T))
                
            else:
                x_train = np.vstack((x_train, tmp[:-1,:]))
                cum_rtn = ((1+tmp[1:,:self.y_w]).cumprod(axis=1)-1)[:,-1]
                std_rtn = tmp[1:,:self.y_w].std(1)
                y_train = np.vstack((y_train, np.vstack((cum_rtn, std_rtn)).T)) 
        return [x_train,y_train, x_test,y_test]










