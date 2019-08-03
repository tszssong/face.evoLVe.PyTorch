import torch
import torch.nn as nn
import torch.nn.functional as F

# Support: ['FocalLoss', 'TripletLoss']
class FocalLoss(nn.Module):
    def __init__(self, gamma = 2, eps = 1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        # print("size:", input.size(), target.size(), input.size(0), input.size()[0], input.size(1), input.size()[1])
        # for batch in range(input.size()[0]):
        #     print(batch,":")
        #     print(input[batch])
        #     print(target[batch])
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

class TripletLoss(nn.Module):
    def __init__(self, margin=0.4):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, input, target):
        _batchsize = input.size(0)
        #TODO: target is null 
        assert _batchsize==target.size(0) 
        anchor   = input[0:_batchsize//3,:]
        positive = input[_batchsize//3:2*_batchsize//3,:]
        negative = input[2*_batchsize//3:_batchsize,:]
        # print("total:",input.size(), "anchor:", anchor.size())
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() 
    # def forward(self, anchor, positive, negative, size_average=True):
    #     distance_positive = (anchor - positive).pow(2).sum(1)
    #     distance_negative = (anchor - negative).pow(2).sum(1)
    #     losses = F.relu(distance_positive - distance_negative + self.margin)
    #     return losses.mean() if size_average else losses.sum() 
        