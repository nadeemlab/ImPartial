import torch
from general.utils import to_np
import numpy as np
# import numpy as np

def compute_imf_losses(out,target,scribble,mask,config,criterio_seg,criterio_rec):
    rec_loss_dic = {}
    seg_fore_loss_dic = {}
    seg_back_loss_dic = {}

    total_loss = {}
    total_loss['rec'] = 0
    total_loss['seg_fore'] = 0
    total_loss['seg_back'] = 0

    out_classes = out[0]
    out_factors = out[1]
    out_meanstd = out[2]
    nclasses = config.nclasses

    ix_scribbles = 0
    # print(out_classes.shape)
    # out_seg_class = torch.nn.Sigmoid()(out_classes)

    for ix_class in range(nclasses):
        seg_fore_loss_dic[ix_class] = 0
        seg_back_loss_dic[ix_class] = 0

        out_seg_class = torch.nn.Softmax(dim=1)(out_classes[:,ix_class*2:ix_class*2+2])



        ## foreground scribbles loss ##
        scribbles_class = scribble[:, ix_scribbles, ...]
        num_scribbles = torch.sum(scribbles_class, [1, 2])  # batch_size

        if torch.sum(num_scribbles) > 0:
            # seg_class_loss = criterio_seg(out_seg_class[:, ix_class, ...], scribbles_class) * scribbles_class  # batch_size x h x w
            seg_class_loss = criterio_seg(out_seg_class[:, 0, ...],scribbles_class) * scribbles_class  # batch_size x h x w
            seg_class_loss = torch.sum(seg_class_loss, [1, 2]) / torch.max(num_scribbles,torch.ones_like(num_scribbles))  # batch_size
            seg_fore_loss_dic[ix_class] += (1 / nclasses) * torch.sum(seg_class_loss) / torch.sum(torch.min(num_scribbles, torch.ones_like(num_scribbles)))  # mean of nonzero nscribbles across batch samples

        ## background scribbles loss ##
        scribbles_back = scribble[:, ix_scribbles + 1, ...]  # batch_size x h x w
        num_scribbles = torch.sum(scribbles_back, [1, 2])  # batch_size
        ix_scribbles += 2

        if torch.sum(num_scribbles) > 0:
            # seg_back_loss = criterio_seg(1-out_seg_class[:, ix_class, ...], scribbles_back) * scribbles_back  # batch_size x h x w
            seg_back_loss = criterio_seg(out_seg_class[:, 1, ...],scribbles_back) * scribbles_back  # batch_size x h x w
            seg_back_loss = torch.sum(seg_back_loss, [1, 2]) / torch.max(num_scribbles,torch.ones_like(num_scribbles))  # batch_size
            seg_back_loss_dic[ix_class] += torch.sum(seg_back_loss) / torch.sum(torch.min(num_scribbles,torch.ones_like(num_scribbles)))  # mean of nonzero nscribbles across batch samples

        total_loss['seg_fore'] += seg_fore_loss_dic[ix_class] * config.weight_classes[ix_class]
        total_loss['seg_back'] += seg_back_loss_dic[ix_class] * config.weight_classes[ix_class]

    ## Reconstruction ##
    ix = 0
    for ch in range(config.n_channels):

        # channel to reconstruct for this class object
        num_mask = torch.sum(1 - mask[:, ch, :, :], [1, 2])  # size batch
        rec_loss_dic[ch] = 0

        if config.mean:  # mean values per fore+back
            mean_values = torch.mean(out_meanstd[:, ix:config.nfactors + ix, ...], [2, 3]) #batch x (nfore*nclasses + nback)
            # mean_values = torch.sum(out_meanstd[:, ix:config.nfactors + ix, ...] * out_factors,[2, 3])  # batch x (nfore*nclasses + nback)
            mean_values = mean_values / torch.sum(out_factors, [2, 3])
            ix += config.nfactors
        else:
            mean_values = torch.zeros([out_factors.shape[0], config.nfactors])
            mean_values = mean_values.to(config.DEVICE)
        mean_values = torch.unsqueeze(torch.unsqueeze(mean_values, -1), -1)

        if config.std:  # logstd values per fore+back
            std_values = torch.mean(out_meanstd[:, ix:config.nfactors + ix, ...],[2, 3])  # assume this is log(std)
            # std_values = torch.sum(out_meanstd[:, ix:config.nfactors + ix, ...] * out_factors, [2, 3])
            std_values = std_values / torch.sum(out_factors, [2, 3])
            ix += config.nfactors
        else:
            std_values = torch.zeros([out_factors.shape[0], config.nfactors])  # assume this is log(std)
            std_values = std_values.to(config.DEVICE)
        std_values = torch.unsqueeze(torch.unsqueeze(std_values, -1), -1)

        mean_x = torch.sum(mean_values * out_factors, 1)
        std_x = torch.sum(std_values * out_factors, 1)
        rec_x = criterio_rec(target[:, ch, ...], mean=mean_x, logstd=std_x) * (1 - mask[:, ch, :, :])
        rec_loss_dic[ch] += torch.mean(torch.sum(rec_x, [1, 2]) / num_mask)  # average over al channels
        total_loss['rec'] += rec_loss_dic[ch] * config.weight_rec_channels[ch]

    ## Additional losses for reference
    total_loss['seg_fore_classes'] = seg_fore_loss_dic
    total_loss['seg_back_classes'] = seg_back_loss_dic
    total_loss['rec_channels'] = rec_loss_dic

    return total_loss


def get_imf_outputs(out, config):
    output = {}

    out_classes = out[0]
    out_factors = out[1]
    out_meanstd = out[2]

    # out_seg_class = torch.nn.Sigmoid()(out_classes)
    output['class_segmentation'] = np.zeros([out_classes.shape[0],config.nclasses,out_classes.shape[2],out_classes.shape[3]])
    for ix_class in range(config.nclasses):
        out_seg_class = torch.nn.Softmax(dim=1)(out_classes[:, ix_class * 2:ix_class * 2 + 2])
        output['class_segmentation'][:,ix_class,...] = to_np(out_seg_class[:,0,...])

    # output['class_segmentation'] = to_np(out_seg_class)

    # output['factors'] = to_np(out_factors)
    output_factors = {}
    output_factors['components'] = to_np(out_factors)
    ix = 0
    for ch in range(config.n_channels):

        if config.mean:
            output_factors['mean_ch'+str(ch)] = to_np(out_meanstd[:, ix:config.nfactors + ix, ...])
            ix += config.nfactors

        if config.std:
            output_factors['std_ch' + str(ch)] = to_np(out_meanstd[:, ix:config.nfactors + ix, ...])
            ix += config.nfactors

    output['factors'] = output_factors
    return output

