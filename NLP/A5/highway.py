#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1d

import torch.nn as nn
import torch
import torch.nn.functional as F

class Highway(nn.Module):
    def __init__(self, eword):
        super(Highway, self).__init__() 
        self.proj = nn.Linear(eword, eword, bias=True)
        self.gate = nn.Linear(eword, eword, bias=True)

    def forward(self, xconv):
        xproj = F.relu(self.proj(xconv))
        xgate = torch.sigmoid(self.gate(xconv))
        xhighway = torch.mul(xgate, xproj) + torch.mul((1.0-xgate), xconv)
        return xhighway


def run_hw():
  hw = Highway(21)
  x = torch.randn(21, 21, dtype=torch.float32)
  x = hw(x)
  print(x.size())

#run_hw()
### END YOUR CODE 

