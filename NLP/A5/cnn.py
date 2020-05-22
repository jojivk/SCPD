#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch import nn
import torch

### YOUR CODE HERE for part 1e
class CNN(nn.Module):
    def __init__(self, eword, ksize):
        super(CNN, self).__init__()
        self.eword = eword
        self.ksize = ksize
        self.conv1d = nn.Conv1d(eword, eword, kernel_size=ksize, bias=True)
        nn.init.xavier_uniform_(self.conv1d.weight) 
        self.relu = nn.ReLU()

    def forward(self, xreshaped):
        batch_size = xreshaped.size(0)
        echar = xreshaped.size(-1)
        xconv =[]
        mpool = nn.MaxPool1d(echar-self.ksize+1)
        for i in range(batch_size):
          single_xconv = self.relu(self.conv1d(xreshaped[i])) #shape= single_xconv.size(-1)
          single_xconv = mpool(single_xconv).squeeze(-1)
          xconv.append(single_xconv)

        xconv = torch.stack(xconv) 
        return xconv

def run_conv():
    x = torch.randn(50, 50, 30, dtype=torch.float32)
    cnn = CNN(50, 100, 5)
    y = cnn(x)
    print(y.size())

#run_conv()
### END YOUR CODE

