import torch
import sys
sys.path.append("../")
from general.utils import to_np
import numpy as np



def compute_segloss(out,scribble,config,criterio_seg):
    seg_fore_loss_dic = {}
    seg_back_loss_dic = {}

    total_loss = {}
    total_loss['seg_fore'] = 0
    total_loss['seg_back'] = 0

    ix = 0 #out channel index
    ix_scribbles = 0 #scribbles index
    for class_tasks_key in config.classification_tasks.keys():
        classification_tasks = config.classification_tasks[class_tasks_key]
        seg_fore_loss_dic[class_tasks_key] = 0
        seg_back_loss_dic[class_tasks_key] = 0

        nclasses = int(classification_tasks['classes']) #number of classes
        ncomponents = np.array(classification_tasks['ncomponents'])
        out_seg = torch.nn.Softmax(dim=1)(out[:, ix:ix + np.sum(ncomponents),...])
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
                seg_class_loss = criterio_seg(out_seg_class, scribbles_class) * scribbles_class #batch_size x h x w
                seg_class_loss = torch.sum(seg_class_loss, [1, 2]) / torch.max(num_scribbles,torch.ones_like(num_scribbles)) #batch_size
                seg_class_loss = torch.sum(seg_class_loss)/torch.sum(torch.min(num_scribbles,torch.ones_like(num_scribbles)))
                seg_fore_loss_dic[class_tasks_key] += classification_tasks['weight_classes'][ix_class]*seg_class_loss #mean of nonzero nscribbles across batch samples

        total_loss['seg_fore'] += seg_fore_loss_dic[class_tasks_key] * config.weight_tasks[class_tasks_key]

        ## background ##
        out_seg_back = torch.sum(out_seg[:, ix_seg:ix_seg + ncomponents[ix_class+1], ...],1)  # batch_size x h x w
        scribbles_back = scribble[:, ix_scribbles, ...]  # batch_size x h x w
        num_scribbles = torch.sum(scribbles_back, [1, 2])  # batch_size
        ix_scribbles += 1
        ix_seg += ncomponents[ix_class+1]

        if torch.sum(num_scribbles) > 0:
            seg_back_loss = criterio_seg(out_seg_back, scribbles_back) * scribbles_back  # batch_size x h x w
            seg_back_loss = torch.sum(seg_back_loss, [1, 2]) / torch.max(num_scribbles,torch.ones_like(num_scribbles))  # batch_size
            seg_back_loss_dic[class_tasks_key] += torch.sum(seg_back_loss) / torch.sum(torch.min(num_scribbles,torch.ones_like(num_scribbles)))  # mean of nonzero nscribbles across batch samples

        total_loss['seg_back'] += seg_back_loss_dic[class_tasks_key] * config.weight_tasks[class_tasks_key]

    ## Additional losses
    total_loss['seg_fore_classes'] = seg_fore_loss_dic
    total_loss['seg_back_classes'] = seg_back_loss_dic

    return total_loss

def get_outputs(out, config):
    output = {}

    ix = 0
    for class_tasks_key in config.classification_tasks.keys():
        output_task = {}

        classification_tasks = config.classification_tasks[class_tasks_key]
        nclasses = int(classification_tasks['classes'])  # number of classes
        ncomponents = np.array(classification_tasks['ncomponents'])
        out_seg = torch.nn.Softmax(dim=1)(out[:, ix:ix + np.sum(ncomponents), ...])
        ix += np.sum(ncomponents)
        out_seg = to_np(out_seg)

        ## class segmentations
        out_classification = np.zeros([out_seg.shape[0],nclasses,out_seg.shape[2],out_seg.shape[3]])

        ix_seg = 0
        for ix_class in range(nclasses):
            out_classification[:,ix_class,...] = np.sum(out_seg[:, ix_seg:ix_seg + ncomponents[ix_class], ...], 1)
            ix_seg += ncomponents[ix_class]
        output_task['class_segmentation'] = out_classification

        ### Factors & Reconstruction Loss ###
        output_factors = {}
        output_factors['components'] = out_seg
        output_task['factors'] = output_factors

        #task
        output[class_tasks_key] = output_task

    return output