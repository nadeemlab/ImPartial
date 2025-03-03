import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score
from utils import to_np


class LossConfig():
    def __init__(self, device, classification_tasks, weight_tasks, weight_objectives):
        self.device = device
        self.rec_loss = 'gaussian'
        self.reg_loss = 'L1'

        self.classification_tasks = classification_tasks
        self.weight_tasks = weight_tasks
        self.weight_objectives = weight_objectives
        self.mean = True
        self.std = False


class ImpartialLoss():
    def __init__(self, loss_config):
        self.config = loss_config

        self.criterio_seg = None
        self.criterio_rec = None 
        self.criterio_reg = None

        self.criterio_seg = seglosses(type_loss='CE', reduction=None)

        if 'reg' in self.config.weight_objectives.keys():
            if self.config.weight_objectives['reg'] > 0:
                self.criterio_reg = gradientLoss2d(penalty=self.config.reg_loss, reduction='mean')

        if 'rec' in loss_config.weight_objectives.keys():
            if self.config.weight_objectives['rec'] > 0:
                self.criterio_rec = reclosses(type_loss=self.config.rec_loss, reduction=None)
                

    def compute_loss(self, out, input, scribble, mask):

        rec_loss_dic = {}
        seg_fore_loss_dic = {}
        seg_back_loss_dic = {}
        reg_loss_dic = {}

        total_loss = {}
        total_loss['rec'] = 0
        total_loss['seg_fore'] = 0
        total_loss['seg_back'] = 0
        total_loss['reg'] = 0

        ix = 0 #out channel index
        ix_scribbles = 0 #scribbles index
        for class_tasks_key in self.config.classification_tasks.keys():
            classification_tasks = self.config.classification_tasks[class_tasks_key]
            seg_fore_loss_dic[class_tasks_key] = 0
            seg_back_loss_dic[class_tasks_key] = 0

            nclasses = int(classification_tasks['classes']) #number of classes
            rec_channels = classification_tasks['rec_channels'] #list with channels to reconstruct
            nrec_channels = len(rec_channels)

            ncomponents = np.array(classification_tasks['ncomponents'])
            if len(out.shape)<= 4:
                out_seg = torch.nn.Softmax(dim=1)(out[:, ix:ix + np.sum(ncomponents),...]) # batch_size x channels x h x w
            else:
                out_seg = torch.nn.Softmax(dim=2)(out[:, :, ix:ix + np.sum(ncomponents), ...])  #predictions x batch_size x channels x h x w
                out_seg = torch.mean(out_seg, 0)
            ix += np.sum(ncomponents)

            ## foreground scribbles loss for each class ##
            ix_seg = 0
            for ix_class in range(nclasses):
                out_seg_class = torch.sum(out_seg[:, ix_seg:ix_seg + ncomponents[ix_class], ...], 1) #batch_size x h x w
                scribbles_class = scribble[:, ix_scribbles, ...] #batch_size x h x w
                num_scribbles = torch.sum(scribbles_class, [1, 2]) #batch_size
                ix_scribbles += 1
                ix_seg += ncomponents[ix_class]

                if torch.sum(num_scribbles) > 0:
                    seg_class_loss = self.criterio_seg(out_seg_class, scribbles_class) * scribbles_class #batch_size x h x w
                    seg_class_loss = torch.sum(seg_class_loss, [1, 2]) / torch.max(num_scribbles, torch.ones_like(num_scribbles)) #batch_size
                    seg_class_loss = torch.sum(seg_class_loss)/torch.sum(torch.min(num_scribbles, torch.ones_like(num_scribbles)))
                    seg_fore_loss_dic[class_tasks_key] += classification_tasks['weight_classes'][ix_class] * seg_class_loss #mean of nonzero nscribbles across batch samples

            total_loss['seg_fore'] += seg_fore_loss_dic[class_tasks_key] * self.config.weight_tasks[class_tasks_key]


            ## background ##
            out_seg_back = torch.sum(out_seg[:, ix_seg:ix_seg + ncomponents[ix_class+1], ...],1)  # batch_size x h x w
            scribbles_back = scribble[:, ix_scribbles, ...]  # batch_size x h x w
            num_scribbles = torch.sum(scribbles_back, [1, 2])  # batch_size
            ix_scribbles += 1
            ix_seg += ncomponents[ix_class+1]

            if torch.sum(num_scribbles) > 0:
                seg_back_loss = self.criterio_seg(out_seg_back, scribbles_back) * scribbles_back  # batch_size x h x w
                seg_back_loss = torch.sum(seg_back_loss, [1, 2]) / torch.max(num_scribbles,torch.ones_like(num_scribbles))  # batch_size
                seg_back_loss_dic[class_tasks_key] += torch.sum(seg_back_loss) / torch.sum(torch.min(num_scribbles,torch.ones_like(num_scribbles)))  # mean of nonzero nscribbles across batch samples

            total_loss['seg_back'] += seg_back_loss_dic[class_tasks_key] * self.config.weight_tasks[class_tasks_key]


            ## Regularization loss (MS penalty)
            if self.criterio_reg is not None:
                ## Regularization
                reg_loss_dic[class_tasks_key] = self.criterio_reg(out_seg)
                total_loss['reg'] += reg_loss_dic[class_tasks_key] * self.config.weight_tasks[class_tasks_key]


            ## Reconstruction ##
            rec_loss_dic[class_tasks_key] = 0

            if (nrec_channels > 0) & (self.criterio_rec is not None):
                for ix_ch in range(nrec_channels):
                    ch = rec_channels[ix_ch]
                    # channel to reconstruct for this class object
                    num_mask = torch.sum(1 - mask[:, ch, :, :], [1, 2]) #size batch

                    if self.config.mean:  # mean values per fore+back
                        # mean_values = torch.mean(out[:, ix:ix + np.sum(ncomponents), ...], [2, 3]) #batch x (nfore*nclasses + nback)
                        if len(out.shape) <= 4:
                            mean_values = torch.sum(out[:, ix: ix + np.sum(ncomponents), ...]*out_seg, [2, 3])  # batch x (nfore*nclasses + nback)
                        else:
                            mean_values = torch.sum(torch.mean(out[:, :, ix: ix + np.sum(ncomponents), ...],0)*out_seg, [2, 3])  # prediction x batch x (nfore*nclasses + nback)
                        mean_values = mean_values/torch.sum(out_seg,[2, 3])
                        ix += np.sum(ncomponents)
                    else:
                        mean_values = torch.zeros([out_seg.shape[0], np.sum(ncomponents)])
                        mean_values = mean_values.to(self.config.device)
                    mean_values = torch.unsqueeze(torch.unsqueeze(mean_values, -1), -1)

                    if self.config.std:  # logstd values per fore+back
                        # std_values = torch.mean(out[:, ix:ix + np.sum(ncomponents), ...],[2, 3])  # assume this is log(std)
                        if len(out.shape) <= 4:
                            std_values = torch.sum(out[:, ix: ix + np.sum(ncomponents), ...]*out_seg, [2, 3])
                        else:
                            std_values = torch.sum(torch.mean(out[:, :, ix: ix + np.sum(ncomponents), ...],0)*out_seg, [2, 3])
                        std_values = std_values/torch.sum(out_seg,[2, 3])
                        ix += np.sum(ncomponents)
                    else:
                        std_values = torch.zeros([out_seg.shape[0], np.sum(ncomponents)])  # assume this is log(std)
                        std_values = std_values.to(self.config.device)
                    std_values = torch.unsqueeze(torch.unsqueeze(std_values, -1), -1)

                    mean_x = torch.sum(mean_values * out_seg, 1)
                    std_x = torch.sum(std_values * out_seg, 1)
                    rec_x = self.criterio_rec(input[:, ch, ...], mean=mean_x, logstd=std_x) * (1 - mask[:, ch, :, :])
                    rec_loss_dic[class_tasks_key] += torch.mean(torch.sum(rec_x, [1, 2]) / num_mask) * classification_tasks['weight_rec_channels'][ix_ch] #average over al channels

                total_loss['rec'] += rec_loss_dic[class_tasks_key] * self.config.weight_tasks[class_tasks_key]

        ## Additional losses for reference
        total_loss['seg_fore_classes'] = seg_fore_loss_dic
        total_loss['seg_back_classes'] = seg_back_loss_dic
        total_loss['rec_channels'] = rec_loss_dic
        total_loss['reg_classes'] = reg_loss_dic

        return total_loss



class seglosses(nn.Module):

    def __init__(self, type_loss = 'L2', reduction = None):
        super(seglosses, self).__init__()
        self.reduction=reduction
        self.type_loss = type_loss

    def forward(self, inputs, targets):

        if self.type_loss == 'L1':
            ret = torch.abs(inputs - targets)

        elif self.type_loss == 'CE':
            ret = -1*(torch.log(torch.max(inputs, 1e-20*torch.ones_like(inputs)))*targets)

        else:
            ret = (inputs - targets) ** 2

        if self.reduction is not None:
            ret = torch.mean(ret, -1) if self.reduction == 'mean' else torch.sum(ret, -1)

        return ret


class reclosses(nn.Module):

    def __init__(self, type_loss = 'gaussian', reduction = None ):
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
