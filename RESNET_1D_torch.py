import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets, models


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False) 

        self.shortcut = nn.Sequential() 
        if stride != 1 or in_planes != planes: #
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.conv1(x)
        out += self.shortcut(x) 
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ):
        super(ResNet, self).__init__()
        self.in_planes = 128 # initial value
        self.conv1 = nn.Conv1d(1, 128, kernel_size=3 , bias=False)
        self.drop1 = nn.Dropout(0.3)
        
        self.layer1 = self._make_layer(256, 2, stride=1) 
        self.drop2 = nn.Dropout(0.3)
        self.layer2 = self._make_layer(512, 2, stride=1)
        self.drop3 = nn.Dropout(0.3)
        self.conv4 = nn.Conv1d(512, 512, kernel_size=3)
        self.fc1 = nn.Linear(44544, 512)
        self.fc2 = nn.Linear(512, 352) 

    def _make_layer(self, planes, num_blocks, stride): 
        layers = []
        # for stride in strides:
        layers.append(BasicBlock(self.in_planes, planes, stride))
        self.in_planes = planes # in_place re-define             
        return nn.Sequential(*layers)                              

    def forward(self, x):
        out = F.relu(self.drop1(self.conv1(x))) 
        # out = F.max_pool1d(out,2)
        out = self.drop2(self.layer1(out)) 
        out = F.max_pool1d(out,2) 
        out = self.drop3(self.layer2(out)) 
        out = F.max_pool1d(out,2) 
        out = self.conv4(out)
        out = out.view(-1, 44544) 
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out




