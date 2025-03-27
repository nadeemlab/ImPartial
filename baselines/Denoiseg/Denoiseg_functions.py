import torch
import sys
import numpy as np

sys.path.append("../")
from general.utils import to_np
from scipy.special import softmax


def compute_denoiseg_losses(out, input, scribble, mask, config, criterio_seg, criterio_rec):
    rec_loss_dic = {}
    seg_fore_loss_dic = {}
    seg_back_loss_dic = {}

    total_loss = {}
    total_loss['rec'] = 0
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
        # out_seg = torch.nn.Softmax(dim=1)(out[:, ix:ix + np.sum(ncomponents),...])
        if len(out.shape)<= 4:
            out_seg = torch.nn.Softmax(dim=1)(out[:, ix:ix + np.sum(ncomponents),...]) # batch_size x channels x h x w
        else:
            out_seg = torch.nn.Softmax(dim=2)(out[:, :, ix:ix + np.sum(ncomponents), ...])  #predictions x batch_size x channels x h x w
            out_seg = torch.mean(out_seg,0)
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

    ### Reconstruction Loss ###
    for ch in range(input.shape[1]): #channels

        # channel to reconstruct for this class object
        num_mask = torch.sum(1 - mask[:, ch, :, :], [1, 2]) #size batch
        if len(out.shape) <= 4:
            rec_ch = criterio_rec(out[:, ix:1 + ix, ...].squeeze(),input[:, ch, ...]) * (1 - mask[:, ch, :, :])
        else:
            rec_ch = criterio_rec(torch.mean(out[:, :, ix:1 + ix, ...],0).squeeze(), input[:, ch, ...]) * (1 - mask[:, ch, :, :])
        # print(rec_ch.shape)
        ix += 1
        rec_loss_dic[ch] = torch.mean(torch.sum(rec_ch, [1, 2]) / num_mask)
        total_loss['rec'] += rec_loss_dic[ch]*config.weight_rec_channels[ch]

    ## Additional losses
    total_loss['seg_fore_classes'] = seg_fore_loss_dic
    total_loss['seg_back_classes'] = seg_back_loss_dic
    total_loss['rec_channels'] = rec_loss_dic

    return total_loss


def get_denoiseg_outputs(out, config):

    output = {}
    if len(out.shape)<=4:
        ix = 0
        for class_tasks_key in config.classification_tasks.keys():
            output_task = {}

            classification_tasks = config.classification_tasks[class_tasks_key]
            nclasses = int(classification_tasks['classes'])  # number of classes
            ncomponents = np.array(classification_tasks['ncomponents'])

            # out_seg = torch.nn.Softmax(dim=1)(out[:, ix:ix + np.sum(ncomponents), ...])
            out_seg = softmax(out[:, ix:ix + np.sum(ncomponents), ...],axis = 1)
            ix += np.sum(ncomponents)
            # out_seg = to_np(out_seg)

            ## class segmentations
            mean_classification = np.zeros([out_seg.shape[0], nclasses,out_seg.shape[2], out_seg.shape[3]])

            ix_seg = 0
            for ix_class in range(nclasses):
                mean_classification[:,ix_class,...] = np.sum(out_seg[:, ix_seg:ix_seg + ncomponents[ix_class], ...], 1)
                ix_seg += ncomponents[ix_class]
            output_task['class_segmentation'] = mean_classification

            ### Factors & Reconstruction Loss ###
            output_factors = {}
            output_factors['components'] = out_seg
            output_task['factors'] = output_factors

            #task
            output[class_tasks_key] = output_task

        output_rec = {}
        for ch in range(out.shape[1] - ix):  # remaining channels
            output_rec[ch] = out[:, ix:1 + ix, ...]
            ix += 1
        output['rec'] = output_rec

    else:
        ix = 0
        # epsilon = sys.float_info.min
        for class_tasks_key in config.classification_tasks.keys():
            output_task = {}

            classification_tasks = config.classification_tasks[class_tasks_key]
            nclasses = int(classification_tasks['classes'])  # number of classes
            ncomponents = np.array(classification_tasks['ncomponents'])

            # out_seg = torch.nn.Softmax(dim=1)(out[:, ix:ix + np.sum(ncomponents), ...])
            out_seg = softmax(out[:, :, ix:ix + np.sum(ncomponents), ...], axis=2)  #size : predictions, batch, channels , h, w
            ix += np.sum(ncomponents)
            # out_seg = to_np(out_seg)

            ## class segmentations
            mean_classification = np.zeros([out_seg.shape[1],nclasses,out_seg.shape[-2],out_seg.shape[-1]])
            # entropy_classification = np.zeros([out_seg.shape[1], nclasses, out_seg.shape[-2], out_seg.shape[-1]])
            variance_classification = np.zeros([out_seg.shape[1], nclasses, out_seg.shape[-2], out_seg.shape[-1]])

            ix_seg = 0
            for ix_class in range(nclasses):
                aux = np.sum(out_seg[:,:, ix_seg:ix_seg + ncomponents[ix_class], ...], 2) #size : predictions, batch, h, w
                mean_classification[:,ix_class,...] = np.mean(aux,axis=0) #batch, h, w
                variance_classification[:,ix_class,...] = np.var(aux,axis=0)
                # entropy_classification[:,ix_class,...] = -mean_classification[:,ix_class,...]*np.log(np.maximum(mean_classification[:,ix_class,...],epsilon))
                ix_seg += ncomponents[ix_class]
            output_task['class_segmentation'] = mean_classification
            output_task['class_segmentation_variance'] = variance_classification
            # output_task['class_segmentation_entropy'] = entropy_classification

            ### Factors & Reconstruction Loss ###
            output_factors = {}
            output_factors['components'] = np.mean(out_seg,axis=0)
            output_factors['components_variance'] = np.var(out_seg,axis=0)
            output_task['factors'] = output_factors

            #task
            output[class_tasks_key] = output_task

        output_rec = {}
        output_rec_variance = {}
        for ch in range(out.shape[1] - ix):  # remaining channels
            output_rec[ch] = np.mean(out[:, :, ix:1 + ix, ...],axis=0)
            output_rec_variance[ch] = np.var(out[:, :, ix:1 + ix, ...],axis=0)
            ix += 1
        output['rec'] = output_rec
        output['rec_variance'] = output_rec_variance

    return output


