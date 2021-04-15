import torch
import sys
sys.path.append("../")
from general.utils import to_np

def compute_ds_losses(out,input,scribble,mask,config,criterio_seg,criterio_rec):
    rec_loss_dic = {}
    seg_fore_loss_dic = {}
    seg_back_loss_dic = {}

    total_loss = {}
    total_loss['rec'] = 0
    total_loss['seg_fore'] = 0
    total_loss['seg_back'] = 0

    out_classes = out[0]
    out_rec_ch = out[2]
    nclasses = config.nclasses

    ix_scribbles = 0
    # print(out_classes.shape)
    out_seg_class = torch.nn.Sigmoid()(out_classes)

    for ix_class in range(nclasses):
        seg_fore_loss_dic[ix_class] = 0
        seg_back_loss_dic[ix_class] = 0

        ## foreground scribbles loss ##
        scribbles_class = scribble[:, ix_scribbles, ...]
        num_scribbles = torch.sum(scribbles_class, [1, 2])  # batch_size

        if torch.sum(num_scribbles) > 0:
            seg_class_loss = criterio_seg(out_seg_class[:, ix_class, ...], scribbles_class) * scribbles_class  # batch_size x h x w
            seg_class_loss = torch.sum(seg_class_loss, [1, 2]) / torch.max(num_scribbles,torch.ones_like(num_scribbles))  # batch_size
            seg_fore_loss_dic[ix_class] += (1 / nclasses) * torch.sum(seg_class_loss) / torch.sum(torch.min(num_scribbles, torch.ones_like(num_scribbles)))  # mean of nonzero nscribbles across batch samples

        ## background scribbles loss ##
        scribbles_back = scribble[:, ix_scribbles + 1, ...]  # batch_size x h x w
        num_scribbles = torch.sum(scribbles_back, [1, 2])  # batch_size
        ix_scribbles += 2

        if torch.sum(num_scribbles) > 0:
            seg_back_loss = criterio_seg(1-out_seg_class[:, ix_class, ...], scribbles_back) * scribbles_back  # batch_size x h x w
            seg_back_loss = torch.sum(seg_back_loss, [1, 2]) / torch.max(num_scribbles,torch.ones_like(num_scribbles))  # batch_size
            seg_back_loss_dic[ix_class] += torch.sum(seg_back_loss) / torch.sum(torch.min(num_scribbles,torch.ones_like(num_scribbles)))  # mean of nonzero nscribbles across batch samples

        total_loss['seg_fore'] += seg_fore_loss_dic[ix_class] * config.weight_classes[ix_class]
        total_loss['seg_back'] += seg_back_loss_dic[ix_class] * config.weight_classes[ix_class]

    ### Reconstruction Loss ###
    for ch in range(input.shape[1]): #channels

        # channel to reconstruct for this class object
        num_mask = torch.sum(1 - mask[:, ch, :, :], [1, 2]) #size batch
        rec_ch = criterio_rec(out_rec_ch[:, ch , ...].squeeze(),input[:, ch, ...]) * (1 - mask[:, ch, :, :])

        rec_loss_dic[ch] = torch.mean(torch.sum(rec_ch, [1, 2]) / num_mask)
        total_loss['rec'] += rec_loss_dic[ch]*config.weight_rec_channels[ch]

    ## Additional losses
    total_loss['seg_fore_classes'] = seg_fore_loss_dic
    total_loss['seg_back_classes'] = seg_back_loss_dic
    total_loss['rec_channels'] = rec_loss_dic

    return total_loss


def get_ds_outputs(out):
    output = {}

    out_classes = out[0]
    out_factors = out[1]
    out_rec_ch = out[2]
    out_seg_class = torch.nn.Sigmoid()(out_classes)
    output['class_segmentation'] = to_np(out_seg_class)

    # output['factors'] = to_np(out_factors)
    output['factors'] = to_np(out_factors)
    output['rec_channels'] = to_np(out_rec_ch)
    return output
