import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

import sys
sys.path.append("../")
from .utils import to_np
import numpy as np

class seglosses(nn.Module):
    def __init__(self,type_loss = 'L2', reduction = None):
        super(seglosses, self).__init__()
        self.reduction=reduction
        self.type_loss = type_loss
    def forward(self, inputs, targets):

        if self.type_loss == 'L1':
            ret = torch.abs(inputs - targets)

        elif self.type_loss == 'CE':
            ret = -1*(torch.log(torch.max(inputs,1e-20*torch.ones_like(inputs)))*targets)

        else:
            ret = (inputs - targets) ** 2

        if self.reduction is not None:
            ret = torch.mean(ret,-1) if self.reduction == 'mean' else torch.sum(ret,-1)

        return ret


class reclosses(nn.Module):
    def __init__(self,type_loss = 'gaussian', reduction = None ):
        super(reclosses, self).__init__()
        self.reduction=reduction
        self.type_loss = type_loss
    def forward(self, targets, mean = 0, logstd = 0):

        if self.type_loss == 'laplacian':
            K = np.log(2)
            ret = torch.abs(targets - mean)/torch.exp(logstd) + K + logstd

        else:
            K = np.log(np.sqrt(2*np.pi))
            ret = 1/2*(((targets - mean)/torch.exp(logstd)) ** 2)  + K + logstd

        if self.reduction is not None:
            ret = torch.mean(ret,-1) if self.reduction == 'mean' else torch.sum(ret,-1)

        return ret

### This is from: https://github.com/jongcye/CNN_MumfordShah_Loss
class gradientLoss2d(nn.Module):
    def __init__(self, penalty='L1',reduction = 'mean'):
        super(gradientLoss2d, self).__init__()
        self.penalty = penalty
        self.reduction = reduction

    def forward(self, input):
        dH = torch.abs(input[:, :, 1:, :] - input[:, :, :-1, :])
        dW = torch.abs(input[:, :, :, 1:] - input[:, :, :, :-1])
        if(self.penalty == "L2"):
            dH = dH * dH
            dW = dW * dW

        if self.reduction == 'mean':
            loss = (torch.mean(dH) + torch.mean(dW))/2
        else:
            loss = torch.sum(dH) + torch.sum(dW)

        return loss


class metrics(nn.Module):
    def __init__(self,type_loss = 'acc'):
        super(metrics, self).__init__()
        self.type_loss = type_loss

    def forward(self, inputs, targets):

        if self.type_loss == 'acc':
            inputs = torch.nn.Softmax(dim=-1)(inputs)  # last dim are the classes
            ret = np.sum(to_np(inputs)*to_np(targets),-1)

        if self.type_loss == 'auc':
            inputs = torch.nn.Softmax(dim=-1)(inputs)  # last dim are the classes
            y_true = to_np(targets)
            y_pred = to_np(inputs)
            # print(roc_auc_score(y_true, y_pred))
            # return torch.from_numpy(np.array([roc_auc_score(y_true, y_pred)]))
            return np.array([roc_auc_score(y_true, y_pred)])

        return ret
