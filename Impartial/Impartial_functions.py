
import sys
sys.path.append("../")
import torch
from general.utils import to_np
import numpy as np

from scipy.special import softmax

def compute_impartial_losses(out,input,scribble,mask,config,criterio_seg,criterio_rec,criterio_reg=None):
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
    for class_tasks_key in config.classification_tasks.keys():
        classification_tasks = config.classification_tasks[class_tasks_key]
        seg_fore_loss_dic[class_tasks_key] = 0
        seg_back_loss_dic[class_tasks_key] = 0

        nclasses = int(classification_tasks['classes']) #number of classes
        rec_channels = classification_tasks['rec_channels'] #list with channels to reconstruct
        nrec_channels = len(rec_channels)

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

        ## Regularization loss (MS penalty)
        if criterio_reg is not None:
            ## Regularization
            reg_loss_dic[class_tasks_key] = criterio_reg(out_seg)
            total_loss['reg'] += reg_loss_dic[class_tasks_key] * config.weight_tasks[class_tasks_key]

        ## Reconstruction ##
        rec_loss_dic[class_tasks_key] = 0

        if nrec_channels > 0:
            for ix_ch in range(nrec_channels):
                ch = rec_channels[ix_ch]
                # channel to reconstruct for this class object
                num_mask = torch.sum(1 - mask[:, ch, :, :], [1, 2]) #size batch

                if config.mean:  # mean values per fore+back
                    # mean_values = torch.mean(out[:, ix:ix + np.sum(ncomponents), ...], [2, 3]) #batch x (nfore*nclasses + nback)
                    mean_values = torch.sum(out[:, ix: ix + np.sum(ncomponents), ...]*out_seg,[2, 3])  # batch x (nfore*nclasses + nback)
                    mean_values = mean_values/torch.sum(out_seg,[2, 3])
                    ix += np.sum(ncomponents)
                else:
                    mean_values = torch.zeros([out.shape[0], np.sum(ncomponents)])
                    mean_values = mean_values.to(config.DEVICE)
                mean_values = torch.unsqueeze(torch.unsqueeze(mean_values, -1), -1)

                if config.std:  # logstd values per fore+back
                    # std_values = torch.mean(out[:, ix:ix + np.sum(ncomponents), ...],[2, 3])  # assume this is log(std)
                    std_values = torch.sum(out[:, ix:ix + np.sum(ncomponents), ...]*out_seg, [2, 3])
                    std_values = std_values/torch.sum(out_seg,[2, 3])
                    ix += np.sum(ncomponents)
                else:
                    std_values = torch.zeros([out.shape[0], np.sum(ncomponents)])  # assume this is log(std)
                    std_values = std_values.to(config.DEVICE)
                std_values = torch.unsqueeze(torch.unsqueeze(std_values, -1), -1)

                mean_x = torch.sum(mean_values * out_seg, 1)
                std_x = torch.sum(std_values * out_seg, 1)
                rec_x = criterio_rec(input[:, ch, ...], mean=mean_x, logstd=std_x) * (1 - mask[:, ch, :, :])
                rec_loss_dic[class_tasks_key] += torch.mean(torch.sum(rec_x, [1, 2]) / num_mask) * classification_tasks['weight_rec_channels'][ix_ch] #average over al channels

            total_loss['rec'] += rec_loss_dic[class_tasks_key] * config.weight_tasks[class_tasks_key]

    ## Additional losses for reference
    total_loss['seg_fore_classes'] = seg_fore_loss_dic
    total_loss['seg_back_classes'] = seg_back_loss_dic
    total_loss['rec_channels'] = rec_loss_dic
    total_loss['reg_classes'] = reg_loss_dic

    return total_loss

def get_impartial_outputs(out, config):
    output = {}

    if len(out.shape) <= 4: #there are no multiple predictions as in MCdropout or Ensemble: dims are batchxchannelsxwxh
        ix = 0
        for class_tasks_key in config.classification_tasks.keys():
            output_task = {}

            classification_tasks = config.classification_tasks[class_tasks_key]
            nclasses = int(classification_tasks['classes'])  # number of classes
            rec_channels = classification_tasks['rec_channels']  # list with channels to reconstruct
            nrec_channels = len(rec_channels)

            ncomponents = np.array(classification_tasks['ncomponents'])
            out_seg = softmax(out[:, ix:ix + np.sum(ncomponents), ...],axis = 1)
            ix += np.sum(ncomponents)

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
            if nrec_channels > 0:
                for ch in rec_channels:
                    if config.mean:
                        output_factors['mean_ch' + str(ch)] = out[:, ix:ix + np.sum(ncomponents), ...]
                        ix += np.sum(ncomponents)

                    if config.std:
                        output_factors['std_ch' + str(ch)] = out[:, ix:ix + np.sum(ncomponents), ...]
                        ix += np.sum(ncomponents)
            output_task['factors'] = output_factors

            #task
            output[class_tasks_key] = output_task
    else:
        ix = 0
        # epsilon = sys.float_info.min
        for class_tasks_key in config.classification_tasks.keys():
            output_task = {}

            classification_tasks = config.classification_tasks[class_tasks_key]
            nclasses = int(classification_tasks['classes'])  # number of classes
            rec_channels = classification_tasks['rec_channels']  # list with channels to reconstruct
            nrec_channels = len(rec_channels)
            ncomponents = np.array(classification_tasks['ncomponents'])

            out_seg = softmax(out[:, :, ix:ix + np.sum(ncomponents), ...], axis=2)  #size : predictions, batch, channels , h, w
            ix += np.sum(ncomponents)

            ## class segmentations
            mean_classification = np.zeros([out_seg.shape[1],nclasses,out_seg.shape[-2],out_seg.shape[-1]])
            variance_classification = np.zeros([out_seg.shape[1], nclasses, out_seg.shape[-2], out_seg.shape[-1]])

            ix_seg = 0
            for ix_class in range(nclasses):
                aux = np.sum(out_seg[:,:, ix_seg:ix_seg + ncomponents[ix_class], ...], 2) #size : predictions, batch, h, w
                mean_classification[:,ix_class,...] = np.mean(aux,axis=0) #batch, h, w
                variance_classification[:,ix_class,...] = np.var(aux,axis=0)
                ix_seg += ncomponents[ix_class]
            output_task['class_segmentation'] = mean_classification
            output_task['class_segmentation_variance'] = variance_classification

            ### Factors & Reconstruction Loss ###
            output_factors = {}
            output_factors['components'] = np.mean(out_seg, axis=0)
            output_factors['components_variance'] = np.var(out_seg, axis=0)

            if nrec_channels > 0:
                for ch in rec_channels:
                    if config.mean:
                        output_factors['mean_ch' + str(ch)] = np.mean(out[:, :, ix:ix + np.sum(ncomponents), ...], axis=0)
                        output_factors['mean_variance_ch' + str(ch)] = np.var(out[:, :, ix:ix + np.sum(ncomponents), ...], axis=0)
                        ix += np.sum(ncomponents)

                    if config.std:
                        output_factors['logstd_ch' + str(ch)] = np.mean(out[:, :, ix:ix + np.sum(ncomponents), ...], axis=0)
                        output_factors['logstd_variance_ch' + str(ch)] = np.var(out[:, :, ix:ix + np.sum(ncomponents), ...], axis=0)
                        ix += np.sum(ncomponents)
            output_task['factors'] = output_factors

            #task
            output[class_tasks_key] = output_task

    return output

# ##### OLDER ######
# def compute_impartial_losses(out,input,scribble,mask,config,criterio_seg,criterio_rec):
#     rec_loss_dic = {}
#     seg_fore_loss_dic = {}
#     seg_back_loss_dic = {}
#
#     total_loss = {}
#     total_loss['rec'] = 0
#     total_loss['seg_fore'] = 0
#     total_loss['seg_back'] = 0
#
#     ix = 0 #out channel index
#     ix_scribbles = 0 #scribbles index
#     for class_tasks_key in config.classification_tasks.keys():
#         classification_tasks = config.classification_tasks[class_tasks_key]
#         seg_fore_loss_dic[class_tasks_key] = 0
#         seg_back_loss_dic[class_tasks_key] = 0
#         nclasses = int(classification_tasks['classes']) #number of classes
#         rec_channels = classification_tasks['rec_channels'] #list with channels to reconstruct
#         nrec_channels = len(rec_channels)
#         out_seg = torch.nn.Softmax(dim=1)(out[:, ix:config.nfore*nclasses + config.nback + ix, ...])
#
#         ## foreground scribbles loss for each class ##
#         for ix_class in range(nclasses):
#             out_seg_class = torch.sum(out_seg[:, ix_class*config.nfore:ix_class*config.nfore + config.nfore, ...], 1) #batch_size x h x w
#             scribbles_class = scribble[:, ix_scribbles, ...] #batch_size x h x w
#             num_scribbles = torch.sum(scribbles_class, [1, 2]) #batch_size
#             ix_scribbles += 1
#
#             if torch.sum(num_scribbles) > 0:
#                 seg_class_loss = criterio_seg(out_seg_class, scribbles_class) * scribbles_class #batch_size x h x w
#                 seg_class_loss = torch.sum(seg_class_loss, [1, 2]) / torch.max(num_scribbles,torch.ones_like(num_scribbles)) #batch_size
#                 seg_fore_loss_dic[class_tasks_key] += (1/nclasses)*torch.sum(seg_class_loss)/torch.sum(torch.min(num_scribbles,torch.ones_like(num_scribbles))) #mean of nonzero nscribbles across batch samples
#
#         total_loss['seg_fore'] += seg_fore_loss_dic[class_tasks_key] * config.weight_classes[class_tasks_key]
#
#         ## background ##
#         out_seg_back = torch.sum(out_seg[:, nclasses * config.nfore:nclasses * config.nfore + config.nback, ...],1)  # batch_size x h x w
#         scribbles_back = scribble[:, ix_scribbles, ...]  # batch_size x h x w
#         num_scribbles = torch.sum(scribbles_back, [1, 2])  # batch_size
#         ix_scribbles += 1
#
#         if torch.sum(num_scribbles) > 0:
#             seg_back_loss = criterio_seg(out_seg_back, scribbles_back) * scribbles_back  # batch_size x h x w
#             seg_back_loss = torch.sum(seg_back_loss, [1, 2]) / torch.max(num_scribbles,torch.ones_like(num_scribbles))  # batch_size
#             seg_back_loss_dic[class_tasks_key] += torch.sum(seg_back_loss) / torch.sum(torch.min(num_scribbles,torch.ones_like(num_scribbles)))  # mean of nonzero nscribbles across batch samples
#
#         total_loss['seg_back'] += seg_back_loss_dic[class_tasks_key] * config.weight_classes[class_tasks_key]
#
#
#         ## Reconstruction ##
#         ix += config.nfore*nclasses + config.nback
#         rec_loss_dic[class_tasks_key] = 0
#         if nrec_channels > 0:
#             for ch in rec_channels:
#
#                 # channel to reconstruct for this class object
#                 num_mask = torch.sum(1 - mask[:, ch, :, :], [1, 2]) #size batch
#
#                 if config.mean:  # mean values per fore+back
#                     # mean_values = torch.mean(out[:, ix:config.nfore*nclasses + config.nback + ix, ...], [2, 3]) #batch x (nfore*nclasses + nback)
#                     mean_values = torch.sum(out[:, ix:config.nfore * nclasses + config.nback + ix, ...]*out_seg,[2, 3])  # batch x (nfore*nclasses + nback)
#                     mean_values = mean_values/torch.sum(out_seg,[2, 3])
#                     ix += config.nfore*nclasses + config.nback
#                 else:
#                     mean_values = torch.zeros([out.shape[0], config.nfore*nclasses + config.nback])
#                     mean_values = mean_values.to(config.DEVICE)
#                 mean_values = torch.unsqueeze(torch.unsqueeze(mean_values, -1), -1)
#
#                 if config.std:  # logstd values per fore+back
#                     # std_values = torch.mean(out[:, ix:config.nfore*nclasses + config.nback + ix, ...],[2, 3])  # assume this is log(std)
#                     std_values = torch.sum(out[:, ix:config.nfore * nclasses + config.nback + ix, ...]*out_seg, [2, 3])
#                     std_values = std_values/torch.sum(out_seg,[2, 3])
#                     ix += config.nfore*nclasses + config.nback
#                 else:
#                     std_values = torch.zeros([out.shape[0], config.nfore*nclasses + config.nback])  # assume this is log(std)
#                     std_values = std_values.to(config.DEVICE)
#                 std_values = torch.unsqueeze(torch.unsqueeze(std_values, -1), -1)
#
#                 mean_x = torch.sum(mean_values * out_seg, 1)
#                 std_x = torch.sum(std_values * out_seg, 1)
#                 rec_x = criterio_rec(input[:, ch, ...], mean=mean_x, logstd=std_x) * (1 - mask[:, ch, :, :])
#                 rec_loss_dic[class_tasks_key] += torch.mean(torch.sum(rec_x, [1, 2]) / num_mask) / nrec_channels #average over al channels
#
#             total_loss['rec'] += rec_loss_dic[class_tasks_key] * config.weight_classes[class_tasks_key]
#
#     ## Additional losses for reference
#     total_loss['seg_fore_classes'] = seg_fore_loss_dic
#     total_loss['seg_back_classes'] = seg_back_loss_dic
#     total_loss['rec_channels'] = rec_loss_dic
#
#     return total_loss
#
#
# def get_impartial_outputs(out, config):
#     output = {}
#
#     ix = 0
#     for class_tasks_key in config.classification_tasks.keys():
#         classification_tasks = config.classification_tasks[class_tasks_key]
#         nclasses = int(classification_tasks['classes'])  # number of classes
#         rec_channels = classification_tasks['rec_channels']  # list with channels to reconstruct
#
#         out_seg = torch.nn.Softmax(dim=1)(out[:, ix:config.nfore * nclasses + config.nback + ix, ...])
#         out_seg = to_np(out_seg)
#
#         output_class = {}
#         for ix_class in range(nclasses):
#             output_class['fore_class'+str(ix_class)] = out_seg[:, ix_class*config.nfore : ix_class*config.nfore + config.nfore, ...]
#         output_class['back'] = out_seg[:, config.nfore*nclasses : config.nfore*nclasses + config.nback, ...]
#         ix += config.nfore*nclasses + config.nback
#
#         ### Reconstruction Loss ###
#         if len(rec_channels) > 0:
#
#             for ch in rec_channels:
#                 output_class_ch = {}
#                 if config.mean:  # mean values per fore+back
#                     for ix_class in range(nclasses):
#                         output_class_ch['mean_fore_class'+str(ix_class)] = to_np(out[:, ix:config.nfore + ix, ...])
#                         ix += config.nfore
#                     output_class_ch['mean_back'] = to_np(out[:, ix:ix + config.nback, ...])
#                     ix += config.nback
#
#                 if config.std:  # logstd values per fore+back
#                     for ix_class in range(nclasses):
#                         output_class_ch['logstd_fore_class'+str(ix_class)] = to_np(out[:, ix:config.nfore + ix, ...])
#                         ix += config.nfore
#                     output_class_ch['logstd_back'] = to_np(out[:, ix: ix + config.nback, ...])
#                     ix += config.nback
#                 output_class['rec_channel_' + str(ch)] = output_class_ch
#
#         output[class_tasks_key] = output_class
#     return output

# def get_impartial_outputs(out,config):
#     output = {}
#
#     ix = 0
#     for segclasses in config.segclasses_channels.keys():
#         print(segclasses)
#         output_class = {}
#
#         out_seg = torch.nn.Softmax(dim=1)(out[:, ix:config.nfore + config.nback + ix, ...])
#         out_seg = to_np(out_seg)
#
#         output_class['fore'] = out_seg[:,0:config.nfore,...] #todo:! swap axes??
#         output_class['back'] = out_seg[:,config.nfore :config.nfore + config.nback,...]
#         ix += config.nfore + config.nback
#
#         ### Reconstruction Loss ###
#         channel_rec_list = config.segclasses_channels[segclasses]
#         if len(channel_rec_list) > 0:
#
#             for ch in channel_rec_list:
#                 output_class_ch = {}
#                 if config.mean:  # mean values per fore+back
#                     output_class_ch['mean_fore'] = to_np(out[:, ix:config.nfore + ix, ...])
#                     output_class_ch['mean_back'] = to_np(out[:, config.nfore + ix:config.nfore + ix + config.nback, ...])
#                     ix += config.nfore + config.nback
#
#                 if config.std:  # logstd values per fore+back
#                     output_class_ch['logstd_fore'] = to_np(out[:, ix:config.nfore + ix, ...])
#                     output_class_ch['logstd_back'] = to_np(out[:, config.nfore + ix:config.nfore + ix + config.nback, ...])
#                     ix += config.nfore + config.nback
#                 output_class['rec_channel_'+str(ch)] = output_class_ch
#
#         output['segclass_' + segclasses] =  output_class
#     return output











