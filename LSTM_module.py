import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd 


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) # input:(batch_size, seq_len, input_size)
        self.fc1 = nn.Linear(self.hidden_size, 512)
        self.fc2 = nn.Linear(512, output_size) 


    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # h_0, c_0 is the initial values for  hidden/ cell state. Hence, we set zero tensors for them where requires_gard= False
        out, _ = self.lstm(x, (h_0,c_0)) 
        out = out[:,-1,:]
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out



# model = lstm.LSTM(input_size, hidden_size, num_layers, output_size).to(device)

    






















