import torch
import sys
sys.path.append("../")
from general.utils import to_np
import numpy as np

def compute_denoiseg_losses(out,input,scribble,mask,config,criterio_seg,criterio_rec):
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

    ### Reconstruction Loss ###
    for ch in range(input.shape[1]): #channels

        # channel to reconstruct for this class object
        num_mask = torch.sum(1 - mask[:, ch, :, :], [1, 2]) #size batch
        rec_ch = criterio_rec(out[:, ix:1 + ix, ...].squeeze(),input[:, ch, ...]) * (1 - mask[:, ch, :, :])
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

    output_rec = {}
    for ch in range(out.shape[1]-ix): #remaining channels
        output_rec[ch] = to_np(out[:, ix:1 + ix, ...])
        ix += 1
    output['rec'] = output_rec

    return output









# def compute_denoiseg_losses(out,input,scribble,mask,config,criterio_seg,criterio_rec):
#     rec_loss_dic = {}
#     seg_fore_loss_dic = {}
#     seg_back_loss_dic = {}
#
#     total_loss = {}
#     total_loss['rec'] = 0
#     total_loss['seg_fore'] = 0
#     total_loss['seg_back'] = 0
#
#     #out is going to have channels = segclasses x 2 + input.channels
#     #first (nfore+nback)*seg_classes are segmentation masks, then input channels are reconstructions
#
#     ix = 0 #out channel index
#     ix_scribbles = 0 #scribbles index
#     for class_tasks_key in config.classification_tasks.keys():
#         classification_tasks = config.classification_tasks[class_tasks_key]
#         seg_fore_loss_dic[class_tasks_key] = 0
#         seg_back_loss_dic[class_tasks_key] = 0
#         nclasses = classification_tasks['classes'] #number of classes
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
#         ix += config.nfore * nclasses + config.nback
#
#     ### Reconstruction Loss ###
#     for ch in range(input.shape[1]): #channels
#
#         # channel to reconstruct for this class object
#         num_mask = torch.sum(1 - mask[:, ch, :, :], [1, 2]) #size batch
#         rec_ch = criterio_rec(out[:, ix:1 + ix, ...].squeeze(),input[:, ch, ...]) * (1 - mask[:, ch, :, :])
#         # print(rec_ch.shape)
#         ix += 1
#         rec_loss_dic[ch] = torch.mean(torch.sum(rec_ch, [1, 2]) / num_mask)
#         total_loss['rec'] += rec_loss_dic[ch]*config.weight_rec_channels[ch]
#
#     ## Additional losses
#     total_loss['seg_fore_classes'] = seg_fore_loss_dic
#     total_loss['seg_back_classes'] = seg_back_loss_dic
#     total_loss['rec_channels'] = rec_loss_dic
#
#     return total_loss
#
# # def compute_denoiseg_losses(out,input,scribble,mask,config,criterio_seg,criterio_rec):
# #     rec_loss_dic = {}
# #     seg_fore_loss_dic = {}
# #     seg_back_loss_dic = {}
# #
# #     total_loss = {}
# #     total_loss['rec'] = 0
# #     total_loss['seg_fore'] = 0
# #     total_loss['seg_back'] = 0
# #
# #     #out is going to have channels = segclasses x 2 + input.channels
# #     #first (nfore+nback)*seg_classes are segmentation masks, then input channels are reconstructions
# #
# #
# #     ix = 0 #out channel index
# #     ix_class = 0 #scribbles index
# #     for segclasses in config.segclasses:
# #         out_seg = torch.nn.Softmax(dim=1)(out[:, ix:config.nfore + config.nback + ix, ...])
# #
# #         ### Segmentation Loss ###
# #         out_seg_fore = torch.sum(out_seg[:, 0:config.nfore, ...], 1)
# #         out_seg_back = torch.sum(out_seg[:, config.nfore:, ...], 1)
# #
# #         num_scribbles_fore = torch.sum(scribble[:, ix_class, ...], [1, 2])
# #         num_scribbles_back = torch.sum(scribble[:, ix_class + 1, ...], [1, 2])
# #
# #         # foreground scribbles loss
# #         if torch.sum(num_scribbles_fore) > 0:
# #             seg_fore_loss = criterio_seg(out_seg_fore, scribble[:, ix_class, ...]) * scribble[:, ix_class, ...]
# #             # loss conditioned to foreground
# #             seg_fore_loss = torch.sum(seg_fore_loss, [1, 2]) / torch.max(num_scribbles_fore,
# #                                                                          torch.ones_like(num_scribbles_fore))
# #             # average over batch that have non zero scribbles:
# #             seg_fore_loss_dic[segclasses] = torch.sum(seg_fore_loss) / torch.sum(torch.min(num_scribbles_fore,
# #                                                                                            torch.ones_like(
# #                                                                                                num_scribbles_fore)))
# #             total_loss['seg_fore'] += seg_fore_loss_dic[segclasses] * config.weight_classes[segclasses]
# #
# #         # background scribbles loss
# #         if torch.sum(num_scribbles_back) > 0:
# #             seg_back_loss = criterio_seg(out_seg_back, scribble[:, ix_class + 1, ...]) * scribble[:, ix_class + 1, ...]
# #             # loss conditioned to background
# #             seg_back_loss = torch.sum(seg_back_loss, [1, 2]) / torch.max(num_scribbles_back,
# #                                                                          torch.ones_like(num_scribbles_back))
# #             # average over batch that have non zero scribbles:
# #             seg_back_loss_dic[segclasses] = torch.sum(seg_back_loss) / torch.sum(torch.min(num_scribbles_back,
# #                                                                                            torch.ones_like(
# #                                                                                                num_scribbles_back)))
# #             total_loss['seg_back'] += seg_back_loss_dic[segclasses] * config.weight_classes[segclasses]
# #         ix_class += 2
# #         ix += config.nfore + config.nback
# #
# #     ### Reconstruction Loss ###
# #     for ch in range(input.shape[1]): #channels
# #
# #         # channel to reconstruct for this class object
# #         num_mask = torch.sum(1 - mask[:, ch, :, :], [1, 2]) #size batch
# #         rec_ch = criterio_rec(out[:, ix:1 + ix, ...].squeeze(),input[:, ch, ...]) * (1 - mask[:, ch, :, :])
# #         # print(rec_ch.shape)
# #         ix += 1
# #         rec_loss_dic[ch] = torch.mean(torch.sum(rec_ch, [1, 2]) / num_mask)
# #         total_loss['rec'] += rec_loss_dic[ch]*config.weight_rec_channels[ch]
# #
# #     total_loss['seg_fore_classes'] = seg_fore_loss_dic
# #     total_loss['seg_back_classes'] = seg_back_loss_dic
# #     total_loss['rec_channels'] = rec_loss_dic
# #
# #     return total_loss
# #


# def get_denoiseg_outputs(out,config):
#     output = {}
#
#     ix = 0
#     for class_tasks_key in config.classification_tasks.keys():
#         classification_tasks = config.classification_tasks[class_tasks_key]
#         nclasses = classification_tasks['classes']  # number of classes
#         out_seg = torch.nn.Softmax(dim=1)(out[:, ix:config.nfore * nclasses + config.nback + ix, ...])
#         out_seg = to_np(out_seg)
#
#         output_class = {}
#         for ix_class in range(nclasses):
#             output_class['fore_class'+str(ix_class)] = out_seg[:, ix_class*config.nfore : ix_class*config.nfore + config.nfore, ...]
#         output_class['back'] = out_seg[:, config.nfore*nclasses : config.nfore*nclasses + config.nback, ...]
#         ix += config.nfore*nclasses + config.nback
#         output[class_tasks_key] = output_class
#
#     output_rec = {}
#     for ch in range(out.shape[1]-ix): #remaining channels
#         output_rec[ch] = to_np(out[:, ix:1 + ix, ...])
#         ix += 1
#     output['rec'] = output_rec
#     return output




