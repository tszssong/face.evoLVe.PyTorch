# import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import  PReLU
# Support: ['FocalLoss', 'TripletLoss']

class FocalLoss(nn.Module):
    def __init__(self, gamma = 2, eps = 1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, input, target, device="cpu",  m=0, use_hard = True):
        _batchsize = input.size(0)
        assert _batchsize==target.size(0) 

        input = F.normalize(input)
        if use_hard:
            ind_list = [idx*3 for idx in range( int(_batchsize/3) )]
            indices = torch.LongTensor(ind_list).to(device)
            anchor   = torch.index_select(input, 0, indices)
            indices = indices + 1
            positive = torch.index_select(input, 0, indices)
            indices = indices + 1
            negative = torch.index_select(input, 0, indices)
        else:
            anchor   = input[0:_batchsize//3,:]
            positive = input[_batchsize//3:2*_batchsize//3,:]
            negative = input[2*_batchsize//3:_batchsize,:]
        
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        if(m==0):
            distances = distance_positive - distance_negative + self.margin
        else:
            distances = distance_positive - distance_negative + m
        losses = F.relu(distances)
        # m=PReLU
        # losses = PReLU(distances)
        # losses = F.prelu(distances)
        return losses.mean(), losses 

