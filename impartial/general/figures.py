from matplotlib import cm
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from skimage.segmentation import find_boundaries
import numpy as np
import random 
import shutil
import os
import sys
import matplotlib.pyplot as plt
import glob
import skimage
from scipy import ndimage

from general.outlines import dilate_masks
from general.metrics_stardist import matching_dataset

from general.training import plot_impartial_outputs 
from dataprocessing.reader import read_image, read_label
from general.inference import get_entropy
from general.evaluation import get_performance_labels

def post_process(out, th=0.8, iter=2):
    mask_pred = (out > th).astype(int)
    label_pred_before, _ = ndimage.label(mask_pred)
    label_pred = skimage.morphology.remove_small_objects(label_pred_before, min_size=5)
    label_pred = dilate_masks(label_pred, n_iter=iter)
    return label_pred_before, label_pred



def apply_colormap_to_img(label_img, cmap):
    # coolwarm = cm.get_cmap('twilight_shifted', 256)
    # coolwarm = cm.get_cmap('PiYG', 256)
    # coolwarm = cm.get_cmap('PRGn', 256)
    # coolwarm = sns.diverging_palette(220, 20, as_cmap=True, s=80, l=75)
    # coolwarm = sns.diverging_palette(220, 20, as_cmap=True)  # ------------ liked
    # coolwarm = sns.diverging_palette(145, 300, s=60, as_cmap=True)


    newcolors = cmap(np.linspace(0, 1, 256))
    black = np.array([0, 0, 0, 1])
    newcolors[1:2, :] = black
    newcmp = ListedColormap(newcolors)

    transformed = np.copy(label_img)
    transformed -= np.min(transformed)
    transformed /= np.max(transformed)

    transformed = newcmp(transformed)

    return transformed


def get_matching_true_ids(true_label, pred_label):

    true_ids, pred_ids = [], []

    for pred_cell in np.unique(pred_label[pred_label > 0]):
        pred_mask = pred_label == pred_cell
        overlap_ids, overlap_counts = np.unique(true_label[pred_mask], return_counts=True)

        # get ID of the true cell that overlaps with pred cell most
        true_id = overlap_ids[np.argmax(overlap_counts)]

        true_ids.append(true_id)
        pred_ids.append(pred_cell)

    return true_ids, pred_ids


def get_cell_size(label_list, label_map):
    size_list = []
    for label in label_list:
        size = np.sum(label_map == label)
        size_list.append(size)

    return size_list


def label_image_by_ratio(true_label, pred_label, threshold=2):

    true_ids, pred_ids = get_matching_true_ids(true_label, pred_label)

    true_sizes = get_cell_size(true_ids, true_label)
    pred_sizes = get_cell_size(pred_ids, pred_label)
    fill_val = -threshold + 0.02
    disp_img = np.full_like(pred_label.astype('float32'), fill_val)
    for i in range(len(pred_ids)):
        current_id = pred_ids[i]
        true_id = true_ids[i]
        if true_id == 0:
            ratio = threshold
        else:
            ratio = np.log2(pred_sizes[i] / true_sizes[i])
        mask = pred_label == current_id
        boundaries = find_boundaries(mask, mode='inner')
        mask[boundaries > 0] = 0
        if ratio > threshold:
            ratio = threshold
        if ratio < -threshold:
            ratio = -threshold
        disp_img[mask] = ratio

    disp_img[-1, -1] = -threshold
    disp_img[-1, -2] = threshold

    return disp_img



def generate_figure_save(image_fname, iname, gt_fname, npz_out_fname, output_dir, th=0.75, iter=2, save=False):

    plt.ioff()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    image = read_image(image_fname)
    label_gt = read_label(gt_fname, image.shape)
    outputs = np.load(npz_out_fname, allow_pickle=False)

    out = outputs['out']
    label_pred_before, label_pred = post_process(out, th=th, iter=iter)
    
    print(label_gt.dtype, label_pred.dtype)
    metrics = get_performance_labels(label_gt.astype('int'), label_pred)
    # plot_impartial_outputs(out, 'test.png')

    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    """
    disp_img_true = label_image_by_ratio(label_gt, label_gt)
    disp_img_true_final = apply_colormap_to_img(disp_img_true, cmap)
    lb_path = os.path.join(output_dir, '{}_label_gt.png'.format(iname))

    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.tight_layout()
    plt.imshow(disp_img_true_final)
    plt.savefig(lb_path, bbox_inches='tight')
    plt.close()
    """
    
    pred_labels = label_pred
    disp_img_mesmer = label_image_by_ratio(label_gt, pred_labels)
    disp_img_mesmer_final = apply_colormap_to_img(disp_img_mesmer, cmap)
    res_path = os.path.join(output_dir, '{}_{}_out.png'.format(iname, th))

    # io.imsave(res_path, disp_img_true_final.astype('float32'))
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.tight_layout()
    plt.imshow(disp_img_mesmer_final)
    plt.savefig(res_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    """
    entropy = get_entropy(out)
    cmap = sns.color_palette("icefire", as_cmap=True)
    cmap = sns.diverging_palette(220, 20, as_cmap=True, l=40, s=60)

    display_entropy = apply_colormap_to_img(entropy, cmap)
    # io.imsave(data_dir + 'Figure_2f_mesmer.png', disp_img_mesmer_final.astype('float32'))
    entropy_path = os.path.join(output_dir, '{}_{}_entropy.png'.format(iname, th))

    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.tight_layout()
    plt.imshow(display_entropy)
    plt.savefig(entropy_path, bbox_inches='tight')
    plt.close()
    """
    return metrics