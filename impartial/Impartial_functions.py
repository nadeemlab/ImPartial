import collections
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt

from impartial.Impartial_classes import ImPartialConfig


def compute_impartial_losses(
        out: torch.Tensor,
        input: torch.Tensor,
        scribble: torch.Tensor,
        mask: torch.Tensor,
        config: ImPartialConfig,
        criterio_seg: nn.Module,
        criterio_rec: Optional[nn.Module] = None,
        criterio_reg: Optional[nn.Module] = None
):
    """
    Compute mixed losses.

    Args:
        out: Output of model evaluation, post-processed using get_impartial_outputs().
            (batch_size, sum((task.n_classes + 1) * task.n_components + task.n_components * task.n_rec_channels), h, w)
        input: Input image used for the reconstruction loss. (batch_size, config.n_channels , h, w)
        scribble: Scribbles for all tasks. (batch_size, sum(tasks.n_classes + 1), h, w)
        mask: Mask used for the reconstruction loss. (batch_size, config.n_channels , h, w)
        config: Impartial configuration used to get classification_tasks and weights.
        criterio_seg: Scribble loss function.
        criterio_rec: Reconstruction loss function.
        criterio_reg: Regularization loss function.
    """
    total_loss = collections.defaultdict(int)

    seg_fore_loss = collections.defaultdict(int)
    seg_back_loss = collections.defaultdict(int)
    rec_loss = collections.defaultdict(int)
    reg_loss = collections.defaultdict(int)

    outputs = outputs_by_task(config.classification_tasks, out)

    from_npz = False
    if from_npz:
        scribbles = scribbles_by_task(config.classification_tasks, scribble)
    else:
        # assumes ImPartial was configured with
        # one task only "0"
        scribbles = {"0": scribble}

    for key, task in config.classification_tasks.items():
        # Foreground scribbles loss for each class
        seg_fore_loss[key] = weighted_scribble_loss(
            outputs=outputs[key]['segmentation']['classes'],
            scribbles=scribbles[key]['classes'],
            weights=task['weight_classes'],
            criterion=criterio_seg
        )
        total_loss['seg_fore'] += seg_fore_loss[key] * config.weight_tasks[key]

        # Background
        seg_back_loss[key] = scribble_loss(
            output=outputs[key]['segmentation']['background'],  # batch_size x h x w
            scribble=scribbles[key]['background'],  # batch_size x h x w
            criterion=criterio_seg
        )
        total_loss['seg_back'] += seg_back_loss[key] * config.weight_tasks[key]

        # Reconstruction
        if criterio_rec:
            rec_loss[key] = reconstruction_loss(
                input=input,
                output=outputs[key]["reconstruction"],
                out_seg=outputs[key]["segmented"],
                mask=mask,
                task=task,
                criterion=criterio_rec,
                config=config
            )
            total_loss['rec'] += rec_loss[key] * config.weight_tasks[key]

        # Regularization loss (MS penalty)
        if criterio_reg:
            reg_loss[key] = criterio_reg(outputs[key]["segmented"])
            total_loss['reg'] += reg_loss[key] * config.weight_tasks[key]

    # Additional losses for reference
    total_loss['seg_fore_classes'] = seg_fore_loss
    total_loss['seg_back_classes'] = seg_back_loss
    total_loss['rec_channels'] = rec_loss
    total_loss['reg_classes'] = reg_loss

    return total_loss


def save_outputs(out):
    for it, task in out.items():
        for ic, c in enumerate(task["segmentation"]["classes"]):
            batch = c.detach().numpy()
            plt.imsave(f"/tmp/task_{it}_class_{ic}.png", np.hstack(np.hstack(batch)))
        batch = task["segmentation"]["background"].detach().numpy()
        plt.imsave(f"/tmp/task_{it}_background.png", np.hstack(np.hstack(batch)))


def save_patches():
    for i, (im, s, m) in enumerate(zip(images, scribbles, validation_masks)):
        plt.imsave(f"/tmp/{i}_full_image.png", im)
        plt.imsave(f"/tmp/{i}_full_scribble.png", np.sum(s, 2))
        plt.imsave(f"/tmp/{i}_full_mask.png", m)

    for i, patch in enumerate(training_list):
        plt.imsave(f"/tmp/{i}_image.png", patch[0][..., 0])
        plt.imsave(f"/tmp/{i}_scribble.png", np.sum(patch[1], 2))

    for i, patch in enumerate(validation_list):
        plt.imsave(f"/tmp/{i}_val_image.png", patch[0][..., 0])
        plt.imsave(f"/tmp/{i}_val_scribble.png", np.sum(patch[1], 2))


def outputs_by_task(tasks, outputs):
    """
    Returns model outputs corresponding to each task
    """
    res = {}

    out_idx = 0
    for k, t in tasks.items():
        step = np.sum(t['ncomponents'])

        out = outputs[:, out_idx:out_idx + step, ...]
        out_seg = torch.nn.Softmax(dim=1)(out)

        idx = 0
        seg_classes = []
        for n in t['ncomponents'][:-1]:
            seg_classes.append(out_seg[:, idx:idx + n, ...])
            idx += n
        seg = {
            "classes": seg_classes,
            "background": out_seg[:, idx:idx + t['ncomponents'][-1], ...]
        }
        out_idx += step

        res[k] = {
            "segmentation": seg,
            "reconstruction": outputs[:, out_idx:out_idx + step, ...],
            "segmented": out_seg
        }
        out_idx += step

    return res


def scribbles_by_task(tasks, scribbles):
    """
    Return scribbles for each task.
    """
    res = {}

    scr_idx = 0
    for k, t in tasks.items():
        class_scribbles = []
        for i in range(t['classes']):
            class_scribbles.append(scribbles[:, scr_idx, ...])
            scr_idx += 1

        res[k] = {
            "classes": class_scribbles,
            "background": scribbles[:, scr_idx, ...]
        }
        scr_idx += 1

    return res


def weighted_scribble_loss(outputs, scribbles, weights, criterion):
    """
    Compute a weighted average of scribble losses.
    """
    return sum(weights[i] * scribble_loss(
            output=outputs[i],  # batch_size x h x w
            scribble=scr,  # batch_size x h x w
            criterion=criterion
        ) for i, scr in enumerate(scribbles)
    )


def scribble_loss(output, scribble, criterion):
    """
    Compute scribble loss.

    Args:
        output: (batch_size, components, h, w)
        scribble: (batch_size, h, w)
    """
    if len(output.shape) != 4:
        raise RuntimeError(f"'output' should have 4 dims (batch_size, components, h, w), "
                           f"got {len(output.shape)} {tuple(output.shape)}")
    if len(scribble.shape) != 3:
        raise RuntimeError(f"'scribble' should have 3 dims (batch_size, h, w), "
                           f"got {len(scribble.shape)} {tuple(scribble.shape)}")
    if output.shape[0] != scribble.shape[0] or output.shape[-2:] != scribble.shape[-2:]:
        raise RuntimeError(f"'scribble' ({scribble.shape}) and 'output' ({output.shape}) dims mismatch")

    batch_size = torch.sum(scribble, [1, 2])  # batch_size
    if torch.sum(batch_size) > 0:
        loss = criterion(torch.sum(output, 1), scribble) * scribble  # batch_size x h x w
        loss = torch.sum(loss, [1, 2]) / torch.max(batch_size, torch.ones_like(batch_size))
        # mean of nonzero nscribbles across batch samples
        return torch.sum(loss) / torch.sum(torch.min(batch_size, torch.ones_like(batch_size)))

    return 0


def reconstruction_loss(input, output, out_seg, mask, task, criterion, config):
    """
    Compute reconstruction loss.
    """
    # channel to reconstruct for this class object
    def get_mean(zeros=False):
        if zeros:
            ts = torch.zeros([out_seg.shape[0], out_seg.shape[1]]).to(config.DEVICE)
        else:
            ts = torch.sum(output * out_seg, [2, 3]) / torch.sum(out_seg, [2, 3]) # batch x (nfore*nclasses + nback)
        return torch.sum(torch.unsqueeze(torch.unsqueeze(ts, -1), -1) * out_seg, 1)

    mean_x = get_mean(not config.mean)
    std_x = get_mean(not config.std)

    loss = 0
    rec_channels = task['rec_channels']  # list with channels to reconstruct
    for i, ch in enumerate(rec_channels):
        mask_inv = 1 - mask[:, ch, :, :]
        rec_x = criterion(input[:, ch, ...], mean=mean_x, logstd=std_x) * mask_inv
        # average over al channels
        num_mask = torch.sum(mask_inv, [1, 2])  # size batch
        loss += torch.mean(torch.sum(rec_x, [1, 2]) / num_mask) * task['weight_rec_channels'][i]

    return loss


def get_impartial_outputs(out, config):
    output = {}

    if len(out.shape) <= 4:  # there are no multiple predictions as in MCdropout or Ensemble: dims are batchxchannelsxwxh
        ix = 0
        for class_tasks_key in config.classification_tasks.keys():
            output_task = {}

            classification_tasks = config.classification_tasks[class_tasks_key]
            nclasses = int(classification_tasks['classes'])  # number of classes
            rec_channels = classification_tasks['rec_channels']  # list with channels to reconstruct
            nrec_channels = len(rec_channels)

            ncomponents = np.array(classification_tasks['ncomponents'])
            out_seg = softmax(out[:, ix:ix + np.sum(ncomponents), ...], axis=1)
            ix += np.sum(ncomponents)

            ## class segmentations
            out_classification = np.zeros([out_seg.shape[0], nclasses, out_seg.shape[2], out_seg.shape[3]])

            ix_seg = 0
            for ix_class in range(nclasses):
                out_classification[:, ix_class, ...] = np.sum(out_seg[:, ix_seg:ix_seg + ncomponents[ix_class], ...], 1)
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

            # task
            output[class_tasks_key] = output_task
    else:
        ix = 0
        # epsilon = sys.float_info.min
        for class_tasks_key in config.classification_tasks.keys():
            output_task = {}

            classification_tasks = config.classification_tasks[class_tasks_key]
            nclasses = int(classification_tasks['classes'])  # number of classes
            rec_channels = classification_tasks['rec_channels']  # list with channels to reconstruct
            ncomponents = np.array(classification_tasks['ncomponents'])

            out_seg = softmax(out[:, :, ix:ix + np.sum(ncomponents), ...],
                              axis=2)  # size : predictions, batch, channels , h, w
            ix += np.sum(ncomponents)

            ## class segmentations
            mean_classification = np.zeros([out_seg.shape[1], nclasses, out_seg.shape[-2], out_seg.shape[-1]])
            variance_classification = np.zeros([out_seg.shape[1], nclasses, out_seg.shape[-2], out_seg.shape[-1]])

            ix_seg = 0
            for ix_class in range(nclasses):
                aux = np.sum(out_seg[:, :, ix_seg:ix_seg + ncomponents[ix_class], ...],
                             2)  # size : predictions, batch, h, w
                mean_classification[:, ix_class, ...] = np.mean(aux, axis=0)  # batch, h, w
                variance_classification[:, ix_class, ...] = np.var(aux, axis=0)
                ix_seg += ncomponents[ix_class]
            output_task['class_segmentation'] = mean_classification
            output_task['class_segmentation_variance'] = variance_classification

            ### Factors & Reconstruction Loss ###
            output_factors = {}
            output_factors['components'] = np.mean(out_seg, axis=0)
            output_factors['components_variance'] = np.var(out_seg, axis=0)

            for ch in rec_channels:
                if config.mean:
                    output_factors['mean_ch' + str(ch)] = np.mean(out[:, :, ix:ix + np.sum(ncomponents), ...], axis=0)
                    output_factors['mean_variance_ch' + str(ch)] = np.var(out[:, :, ix:ix + np.sum(ncomponents), ...],
                                                                          axis=0)
                    ix += np.sum(ncomponents)

                if config.std:
                    output_factors['logstd_ch' + str(ch)] = np.mean(out[:, :, ix:ix + np.sum(ncomponents), ...], axis=0)
                    output_factors['logstd_variance_ch' + str(ch)] = np.var(out[:, :, ix:ix + np.sum(ncomponents), ...],
                                                                            axis=0)
                    ix += np.sum(ncomponents)

            output_task['factors'] = output_factors

            # task
            output[class_tasks_key] = output_task

    return output
