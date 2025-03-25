import numpy as np
import pandas as pd
import skimage
from scipy import ndimage
from sklearn.metrics import roc_auc_score, jaccard_score
from impartial.general.outlines import dilate_masks
from impartial.general.metrics_stardist import matching_dataset


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


def get_performance(label_gt, y_pred, threshold=0.5, iou_threshold=0.5, dilate=False):
    pred_thresholded = y_pred > threshold
    label_pred, _ = ndimage.label(pred_thresholded)
    if dilate:
        # min_size = 2 
        # dilation iter = 1
        label_pred = skimage.morphology.remove_small_objects(label_pred, min_size=5)
        label_pred = dilate_masks(label_pred, n_iter=1)

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
        mAP_dict[str(th)] = apt

    mAP_dict["mAP_05_95"] = np.mean(mAP_05_95)

    metrics = {
               'auc': auc,
               'Jacc': jacc,
               'mIoU': np.mean(np.max(IoU, axis=1)),
               'mDice': np.mean(np.max(Dice, axis=1)),
               'AP': precision
               }
    
    # stardist based metrics
    metrics_sd_label = matching_dataset(label_gt, label_pred, iou_threshold, show_progress=False)
    metrics_sd = { 
                'precision_sd': metrics_sd_label.precision, 
                'recall_sd' : metrics_sd_label.recall, 
                'accuracy_sd': metrics_sd_label.accuracy, 
                'f1_sd': metrics_sd_label.f1,
                'panoptic_quality': metrics_sd_label.panoptic_quality 
                }

    metrics.update(metrics_sd)
    
    metrics.update(mAP_dict)
    return metrics




# def get_performance_labels(label_gt, label_pred, iou_threshold=0.5):

#     # print("impartial_evaluation_debug_label_gt", label_gt.shape)
#     # print("impartial_evaluation_debug_y_pred", y_pred.shape)

#     # auc
#     auc = 0.0
#     # auc = roc_auc_score((label_gt > 0).flatten(), y_pred.flatten())
#     # jaccard or iou semantic level
#     jacc = jaccard_score((label_gt > 0).flatten(), (label_pred > 0).flatten())

#     # instance: IoU & Dice
#     psg = pixel_sharing_bipartite(label_gt, label_pred)
#     fn = np.sum(psg, 1, keepdims=True) - psg
#     fp = np.sum(psg, 0, keepdims=True) - psg
#     # IoU = psg / (fp + fn + psg)
#     IoU = safe_division(psg, (fp + fn + psg))
#     # Dice = 2 * psg / (fp + fn + 2 * psg)
#     Dice = safe_division(2 * psg, (fp + fn + 2 * psg))

#     # Precision (AP)
#     matching = IoU > iou_threshold
#     matching[:, 0] = False
#     matching[0, :] = False
#     assert matching.sum(0).max() < 2
#     assert matching.sum(1).max() < 2
#     n_gt = len(set(np.unique(label_gt)) - {0})
#     n_hyp = len(set(np.unique(label_pred)) - {0})
#     n_matched = matching.sum()

#     # precision = n_matched / (n_gt + n_hyp - n_matched)
#     precision = safe_division(n_matched, (n_gt + n_hyp - n_matched))

#     metrics = {
#                'auc': auc,
#                'Jacc': jacc,
#                'mIoU': np.mean(np.max(IoU, axis=1)),
#                'mDice': np.mean(np.max(Dice, axis=1)),
#                'AP': precision
#                }
    
#     # stardist based metrics
#     metrics_sd_label = matching_dataset(label_gt, label_pred, iou_threshold, show_progress=False)
#     metrics_sd = { 
#                 'precision_sd': metrics_sd_label.precision, 
#                 'recall_sd' : metrics_sd_label.recall, 
#                 'accuracy_sd': metrics_sd_label.accuracy, 
#                 'f1_sd': metrics_sd_label.f1,
#                 'panoptic_quality': metrics_sd_label.panoptic_quality 
#                 }

#     metrics.update(metrics_sd)
    
#     return metrics




def summary_performance(pd_summary, best_all=False, metric='AP', group='train'):
    if best_all: #pick best performance per sample per metric ('best possible achievable')
        row_list = []
        for task in pd_summary.task.unique():
            for segclass in pd_summary.loc[(pd_summary.task == task)].segclass.unique():
                for prefix in pd_summary.prefix.unique():
                    aux = pd_summary.loc[(pd_summary.prefix == prefix)&(pd_summary.task == task)&(pd_summary.segclass == segclass)].max()
                    row = []
                    for col in pd_summary.columns:
                        row.append(aux[col])
                    row_list.append(row)
                # print(row_list)
        pd_best = pd.DataFrame(data=row_list,columns=pd_summary.columns)

    else: #chose best threshold (model) according to metric and group
        ix = 0
        for task in pd_summary.task.unique():
            for segclass in pd_summary.loc[(pd_summary.task == task)].segclass.unique():
                aux = pd_summary.loc[((pd_summary.group == group)&(pd_summary.task == task)&(pd_summary.segclass == segclass))]
                th_list = aux.th.unique()
                value_list = []

                for th in th_list:
                    value_list.append(np.mean(aux.loc[aux.th == th][metric].values))

                ix_best = np.argmax(np.array(value_list))
                th_best = th_list[ix_best]
                if ix == 0:
                    pd_best = pd_summary.loc[((pd_summary.th == th_best)&(pd_summary.task == task)&(pd_summary.segclass == segclass))]
                else:
                    pd_out_aux = pd_summary.loc[((pd_summary.th == th_best)&(pd_summary.task == task)&(pd_summary.segclass == segclass))]
                    pd_best = pd.concat([pd_best,pd_out_aux])
                ix += 1
    return pd_best


def summary_performance_best(pd_summary, best_all=False, metric='AP', group='train'):
    if best_all: #pick best performance per sample per metric ('best possible achievable')
        row_list = []
        for segclass in pd_summary.segclass.unique():
            for prefix in pd_summary.prefix.unique():
                aux = pd_summary.loc[(pd_summary.prefix == prefix)&(pd_summary.segclass == segclass)].max()
                row = []
                for col in pd_summary.columns:
                    row.append(aux[col])
                row_list.append(row)
                # print(row_list)
        pd_best = pd.DataFrame(data=row_list,columns=pd_summary.columns)
    else: #chose best threshold (model) according to metric and group
        ix = 0
        for segclass in pd_summary.segclass.unique():
            aux = pd_summary.loc[((pd_summary.group == group)&(pd_summary.segclass == segclass))]
            th_list = aux.th.unique()
            value_list = []

            for th in th_list:
                value_list.append(np.mean(aux.loc[aux.th == th][metric].values))

            ix_best = np.argmax(np.array(value_list))
            th_best = th_list[ix_best]
            if ix == 0:
                pd_best = pd_summary.loc[((pd_summary.th == th_best)&(pd_summary.segclass == segclass))]
            else:
                pd_out_aux = pd_summary.loc[((pd_summary.th == th_best)&(pd_summary.segclass == segclass))]
                pd_best = pd.concat([pd_best,pd_out_aux])
            ix += 1
    return pd_best
