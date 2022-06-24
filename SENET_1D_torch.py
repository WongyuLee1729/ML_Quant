import numpy as np
import pandas as pd 
import torch.nn as nn
import torch.nn.functional as F


class stock_preprop:
    ''' 
    input data df output data numpy array
    makes 7:3 ratio train & test set
    time-series testing dataset
      '''
    def __init__(self, df:pd.DataFrame, x_w, y_w, stride):
        self.df = df
        self.x_w = x_w
        self.y_w = y_w
        self.stride = stride
        self.ratio = np.ceil(len(self.df)/(len(self.df)*3/10))
        # self.tickers = df.columns

    def XY(self):
        x_train = np.empty((0, self.x_w))
        y_train = np.empty((0, 13)) # (0,1)
        x_test = np.empty((0, self.x_w))
        y_test = np.empty((0, 13)) # (0,1)
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
                cum_rtn =cum_rtn.reshape(1,len(cum_rtn)) # tmp[1:,:self.y_w].std(1)
                rtn = np.triu(np.ones((cum_rtn.shape[1],cum_rtn.shape[1])),0)*np.vstack([cum_rtn]*cum_rtn.shape[1])
                y_test = np.vstack((y_test, rtn))
            else:
                x_train = np.vstack((x_train, tmp[:-1,:]))
                cum_rtn = ((1+tmp[1:,:self.y_w]).cumprod(axis=1)-1)[:,-1]
                cum_rtn =cum_rtn.reshape(1,len(cum_rtn))
                rtn = np.triu(np.ones((cum_rtn.shape[1],cum_rtn.shape[1])),0)*np.vstack([cum_rtn]*cum_rtn.shape[1])
                y_train = np.vstack((y_train, rtn))
        return [x_train,y_train, x_test,y_test]


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16): # r = reduction ratio
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid() # F_ex(w_2, ReLU(w_1, x))
        )

    def forward(self, input_filter): # row: 분단위 , col: 날짜 21
        batch, channel, _ = input_filter.size() # input_filter [16,10,49]
        se = self.squeeze(input_filter).view(batch, channel) # [16, 10]
        se = self.excitation(se).view(batch, channel, 1) # [16,10,1]
        return input_filter * se.expand_as(input_filter)


class SENET_1D(nn.Module):
    def __init__(self):
        super(SENET_1D, self).__init__()

        self.conv1 = nn.Conv1d(1, 128, kernel_size=3)
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(128, 256, kernel_size=3)
        self.drop2 = nn.Dropout(0.3) # nn.Dropout(p=0.5)

        self.conv3 = nn.Conv1d(256, 512, kernel_size=3)

        self.seblock1 = SEBlock(512)
        self.drop3 = nn.Dropout(0.3)
        self.conv4 = nn.Conv1d(512, 512, kernel_size=3)
        self.seblock2 = SEBlock(512)          
        
        self.fc1 = nn.Linear(43520, 512)
        self.fc2 = nn.Linear(512, 362) 


    def forward(self, x):
        x = F.relu((self.drop1(self.conv1(x)))) # conv1
        x = F.relu(F.max_pool1d(self.drop2(self.conv2(x)), 2)) # conv2
        # x = self.seblock0(x)
        x = F.relu(F.max_pool1d(self.drop3(self.conv3(x)), 2)) # conv3
        x = self.seblock1(x)
        x = F.relu(self.conv4(x)) # conv2
        x = self.seblock2(x)
        x = x.view(-1, 43520) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x