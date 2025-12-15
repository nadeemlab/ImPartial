import numpy as np
import pandas as pd
import skimage
from scipy import ndimage
from sklearn.metrics import roc_auc_score, jaccard_score

from impartial.general.outlines import dilate_masks
from impartial.general.metrics_stardist import matching_dataset
from impartial.general.metrics_cellpose import aggregated_jaccard_index, average_precision

def safe_division(x, y, eps=1e-10):
    """computes a safe divide which returns 0 if y is zero"""
    if np.isscalar(x) and np.isscalar(y):
        return x/y if np.abs(y)>eps else 0.0
    else:
        out = np.zeros(np.broadcast(x,y).shape, np.float32)
        np.divide(x,y, out=out, where=np.abs(y)>eps)
        return out


def pixel_sharing_bipartite(lab1, lab2):
    assert lab1.shape == lab2.shape
    psg = np.zeros((lab1.max() + 1, lab2.max() + 1), dtype=np.int32)
    for i in range(lab1.size):
        psg[lab1.flat[i], lab2.flat[i]] += 1
    return psg


def post_process_predictions(y_pred, th=0.8, n_iter=2):
    mask_pred = (y_pred > th).astype(int)
    label_pred_before, _ = ndimage.label(mask_pred)
    label_pred = skimage.morphology.remove_small_objects(label_pred_before, min_size=5)
    label_pred = dilate_masks(label_pred, n_iter=n_iter)
    return label_pred


def get_performance(label_gt, y_pred, threshold=0.75, iou_threshold=0.5, dilate=True):
    
    if dilate:
        label_pred = post_process_predictions(y_pred, th=threshold)
    else:
        label_pred = y_pred > threshold
        label_pred, _ = ndimage.label(label_pred)
        
    return get_performance_labels(label_gt, label_pred, iou_threshold)


def get_performance_labels(label_gt, label_pred, iou_threshold=0.5):

    # print("impartial_evaluation_debug_label_gt", label_gt.shape)
    # print("impartial_evaluation_debug_y_pred", y_pred.shape)

    # auc
    auc = 0.0
    # auc = roc_auc_score((label_gt > 0).flatten(), y_pred.flatten())
    # jaccard or iou semantic level
    jacc = jaccard_score((label_gt > 0).flatten(), (label_pred > 0).flatten())

    # instance: IoU & Dice
    psg = pixel_sharing_bipartite(label_gt, label_pred)
    fn = np.sum(psg, 1, keepdims=True) - psg
    fp = np.sum(psg, 0, keepdims=True) - psg
    # IoU = psg / (fp + fn + psg)
    IoU = safe_division(psg, (fp + fn + psg))
    # Dice = 2 * psg / (fp + fn + 2 * psg)
    Dice = safe_division(2 * psg, (fp + fn + 2 * psg))

    # Precision (AP)
    matching = IoU > iou_threshold
    matching[:, 0] = False
    matching[0, :] = False
    assert matching.sum(0).max() < 2
    assert matching.sum(1).max() < 2
    n_gt = len(set(np.unique(label_gt)) - {0})
    n_hyp = len(set(np.unique(label_pred)) - {0})
    n_matched = matching.sum()

    # precision = n_matched / (n_gt + n_hyp - n_matched)
    precision = safe_division(n_matched, (n_gt + n_hyp - n_matched))

    def calculate_AP(label_gt, label_pred, iou_threshold=0.5):

        # instance: IoU & Dice
        psg = pixel_sharing_bipartite(label_gt, label_pred)
        fn = np.sum(psg, 1, keepdims=True) - psg
        fp = np.sum(psg, 0, keepdims=True) - psg
        # IoU = psg / (fp + fn + psg)
        IoU = safe_division(psg, (fp + fn + psg))

        # Precision (AP)
        matching = IoU > iou_threshold
        matching[:, 0] = False
        matching[0, :] = False
        assert matching.sum(0).max() < 2
        assert matching.sum(1).max() < 2
        n_gt = len(set(np.unique(label_gt)) - {0})
        n_hyp = len(set(np.unique(label_pred)) - {0})
        n_matched = matching.sum()

        # precision = n_matched / (n_gt + n_hyp - n_matched)
        precision = safe_division(n_matched, (n_gt + n_hyp - n_matched))
        return precision

    iouThrs = np.asarray([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    mAP_dict = {}
    mAP_05_95 = []
    for th in iouThrs:
        apt = calculate_AP(label_gt, label_pred, th)
        mAP_05_95.append(apt)
        mAP_dict["mAP_" + str(th)] = apt

    mAP_05_95_dict = {}
    mAP_05_95_dict["mAP_05_95"] = np.mean(mAP_05_95)

    metrics = {
               'im_auc': auc,
               'im_Jacc': jacc,
               'im_mIoU': np.mean(np.max(IoU, axis=1)),
               'im_mDice': np.mean(np.max(Dice, axis=1)),
               'im_AP': precision
               }
    
    # stardist based metrics
    metrics_sd_label = matching_dataset([label_gt], [label_pred], thresh=iou_threshold, show_progress=False)

    metrics_sd = { 
                'sd_precision': metrics_sd_label.precision, 
                'sd_recall' : metrics_sd_label.recall, 
                'sd_accuracy': metrics_sd_label.accuracy, 
                'sd_f1': metrics_sd_label.f1,
                'sd_panoptic_quality': metrics_sd_label.panoptic_quality 
                }

    metrics.update(metrics_sd)
    
    
    # cellpose based metrics
    aji_cp = aggregated_jaccard_index(label_gt, label_pred)
    ap, tp, fp, fn = average_precision(label_gt, label_pred, threshold=[0.5])

    metrics_cp = { 
                'cp_aji': aji_cp, 
                # 'cp_AP': [ap[0], tp[0], fp[0], fn[0]],
                'cp_AP': ap[0],
                }
    metrics.update(metrics_cp)
    metrics.update(mAP_05_95_dict)
    metrics.update(mAP_dict)
    return metrics
