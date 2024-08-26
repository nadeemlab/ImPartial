import os
import sys
import glob
import roifile
import cv2 as cv
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology

import copy
from scipy import ndimage
from csbdeep.utils import normalize

from roifile import ImagejRoi
from PIL import Image

import torch


def percentile_normalization(image, pmin=1, pmax=98, clip = False):
    # Normalize the image using percentiles
    lo, hi = np.percentile(image, (pmin, pmax))
    image_norm_percentile = (image - lo) / (hi - lo)
    
    if clip:
        image_norm_percentile[image_norm_percentile>1] = 1
        image_norm_percentile[image_norm_percentile<0] = 0
        
    return image_norm_percentile


def plot(image):

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.imshow(image[:,:,0])

    plt.subplot(1,2,2)
    plt.imshow(image[:,:,1])

    plt.show()
    pass


def plotLabel(label, title):
    plt.title(title)
    plt.imshow(label)

    plt.show()
    pass


def read_image(path):
    extension = os.path.splitext(path)[-1][1:].lower()

    if extension == "png":                              ####check and add conditions -->conveting 3 channel png to 2ch
        image_org = np.array(Image.open(path))
        image = np.zeros((image_org.shape[0], image_org.shape[1], 2))
        image[:,:,0] = image_org[:,:,0]
        image[:,:,1] = image_org[:,:,2]
    elif extension in ("tiff", "tif"):
        image = tiff.imread(path)
    else:
        raise RuntimeError(f"File type '{extension}' not supported from path '{path}'")

    if len(image.shape) == 2:
        image = image[np.newaxis, ...]

    idx = np.argmin(image.shape)
    image = np.moveaxis(image, idx, -1)

    return image

def get_contour(roi):
    new_coord = []
    if roi.integer_coordinates is not None:
        coord = roi.integer_coordinates
        for i in range(len(coord)):
            new_coord.append([coord[i, 0] + roi.left, coord[i, 1] + roi.top])
        # coord[:, 0] += roi.left
        # coord[:, 1] += roi.top
        return np.asarray(new_coord).astype(np.int32)

    elif roi.multi_coordinates is not None:
        coord = ImagejRoi.path2coords(roi.multi_coordinates)[0]
    else:
        raise RuntimeError("ROI type not supported")

    return np.asarray(coord).astype(np.int32)



def readRoiFile(image, roi_path):
    rois = roifile.roiread(roi_path)

    label = np.zeros((image.shape[0], image.shape[1]))
    # print("label shape", label.shape)
    # print("label dtype: ", label.dtype)
    # .astype(np.int32)
    # print("len(rois): ", len(rois))
    # convert_roi_to_label(label, roi)
    for i in range(0, len(rois)):
        
        contour = get_contour(rois[i])

        cv.fillPoly(label, pts=[contour], color=i)

    label = (label).astype(np.float32)

    return label


def erosion_labels(label, radius_pointer=1):

    selem = morphology.disk(radius_pointer)

    mask = np.zeros_like(label)

    for vlabel in np.unique(label):
        if vlabel != 0:  
            mask_label = np.zeros_like(label)
            mask_label[label == vlabel] = 1
            aux = morphology.dilation(mask_label, footprint=selem)
            if np.unique(aux*label).shape[0] > 2: #overlaps with other label
                erode_label = morphology.erosion(mask_label, footprint=selem)
                mask[erode_label>0] = vlabel
            else:
                mask[mask_label>0] = vlabel
                
    return mask



def get_scribbles_mask(label_image, fov_box=(32, 32), max_labels=4,
                       radius_pointer=0, disk_scribble=False, sample_back = False):
    mask_image = np.zeros_like(label_image)
    mask_image[label_image > 0] = 1.0

    nlabels = np.unique(label_image[label_image > 0]).shape[0]
    max_labels = np.minimum(max_labels, nlabels)
    print("get_scribbles_mask: ", nlabels, max_labels)

    ### Set the instance scribble mask mask_sk
    mask_sk = morphology.skeletonize(mask_image) * mask_image
    if radius_pointer > 0:
        selem = morphology.disk(radius_pointer)
        mask_sk = morphology.dilation(mask_sk, footprint=selem) * mask_image
    if disk_scribble:
        selem = morphology.disk(3)
        outline_arc = morphology.erosion(mask_image, footprint=selem) - \
                      morphology.erosion(morphology.erosion(mask_image, footprint=selem), footprint=morphology.disk(1))
        if radius_pointer > 0:
            selem = morphology.disk(radius_pointer)
            outline_arc = morphology.dilation(outline_arc, footprint=selem)

        mask_sk += outline_arc
        mask_sk[mask_sk > 0] = 1
        mask_sk = mask_sk * mask_image

    nbudget = max_labels + 0
    labels_image_res = np.array(label_image)

    back_aux = np.array(mask_image)
    back_aux = morphology.dilation(back_aux, footprint=morphology.disk(4))

    fov_image_res = np.array(1-mask_image)*back_aux #background entre foreground
    fov_image_res = morphology.skeletonize(fov_image_res)

    # plt.figure(figsize=(10,5))
    # plt.subplot(1,2,1)
    # plt.imshow(fov_image_res)
    # plt.subplot(1, 2, 2)

    back_aux = morphology.skeletonize(1-back_aux)
    fov_image_res += morphology.dilation(back_aux,footprint=morphology.disk(4))

    # plt.imshow(fov_image_res)
    # plt.show()

    foreground = np.zeros_like(labels_image_res)
    background = np.zeros_like(labels_image_res)

    while nbudget > 0:

        ## Pick a point at random
        if sample_back:
            aux = np.array(fov_image_res)
        else:
            aux = np.array(labels_image_res)

        ix0_back, ix1_back = np.nonzero(aux)
        ix = np.random.randint(0, ix0_back.shape[0])

        # bounding box
        xmin = np.maximum(0, ix0_back[ix] - int(fov_box[0] / 2))
        xmax = np.minimum(xmin + int(fov_box[0]), labels_image_res.shape[0])

        ymin = np.maximum(0, ix1_back[ix] - int(fov_box[1] / 2))
        ymax = np.minimum(ymin + int(fov_box[1]), labels_image_res.shape[0])

        bb_image = np.zeros_like(labels_image_res)
        bb_image[xmin:xmax, ymin:ymax] = 1

        active_labels = labels_image_res * bb_image

        if np.sum(active_labels>0) > 0:
            active_labels = np.unique(active_labels[active_labels > 0])

            for label in active_labels:
                label_aux = np.zeros_like(labels_image_res)
                label_aux[labels_image_res == label] = 1

                ## foreground
                foreground += label_aux * mask_sk

                back_aux = morphology.dilation(label_aux, footprint=morphology.disk(4))
                back_aux += bb_image
                back_aux = back_aux * (1 - mask_image)
                back_aux[back_aux > 0] = 1
                back_aux = morphology.skeletonize(back_aux)

                if radius_pointer > 0:
                    selem = morphology.disk(radius_pointer)
                    back_aux = morphology.dilation(back_aux, footprint=selem) * (1 - mask_image)

                background += back_aux * (1 - mask_image)  # double secure

                nbudget -= 1
                labels_image_res = labels_image_res * (1 - label_aux)
                fov_image_res = fov_image_res * (1 - label_aux)
        else:
            #print('NO labels')
            if np.sum(fov_image_res*bb_image) > 0:
                back_aux = morphology.skeletonize(bb_image*(1-mask_image)*fov_image_res) ## box with background
                if radius_pointer > 0:
                    selem = morphology.disk(radius_pointer)
                    back_aux = morphology.dilation(back_aux, footprint=selem) * (1 - mask_image)
                background += back_aux * (1 - mask_image)  # double secure
                # ix0_nonz, ix1_nonz = np.nonzero(back_aux)
                # # if radius_pointer > 0:
                # w=15
                # back_aux = np.zeros_like(back_aux)
                # back_aux[ix0_nonz[0],ix1_nonz[0]-w:ix1_nonz[0]+w] = 1
                # # selem[:,int(fov_box[1] / 2)] = 1
                #
                # # back_aux = morphology.dilation(back_aux, selem=selem) * (1 - mask_image)

        fov_image_res = fov_image_res * (1 - bb_image)

        foreground[foreground>0] = 1
        background[background>0] = 1
        mask_scribbles = np.zeros([mask_image.shape[0], mask_image.shape[1], 2])
        mask_scribbles[..., 0] = np.array(foreground)  # fore
        mask_scribbles[..., 1] = np.array(background)   # back

    return mask_scribbles, max_labels - nbudget, nlabels


def get_scribble(label, scribble_rate=1.0):

    unique_labels = np.unique(label).shape[0]
    nlabels_budget = int(scribble_rate * unique_labels)
    print("unique_labels: ", unique_labels)
    print("nlabels_budget: ", nlabels_budget)

    label_mask = erosion_labels(label, radius_pointer=1)

    Y_gt_train_ch0_list = []
    Y_gt_train_ch0_list.append(label_mask)
    

    Y_out_ch0_list, nscribbles_ch0_list, nlabels_ch0_list = get_scribbles_mask(
                                                                     Y_gt_train_ch0_list[0],
                                                                     fov_box=(32,32),
                                                                     max_labels=nlabels_budget,
                                                                     radius_pointer=0,
                                                                     disk_scribble = True,
                                                                     sample_back = False) # True GS check

    Y_out_ch0_list = [Y_out_ch0_list]

    
    ## concaticating scribbles :
    label_gt = Y_gt_train_ch0_list[0]
    iscribble = Y_out_ch0_list[0] 

    label_ch = np.array(label_gt)        
    background = np.array(iscribble[...,1])*(1-label_ch) + 0 #make sure no foreground is set as background
    scribble_task = np.array(iscribble[...,0])[...,np.newaxis] + 0
            
    scribble_task = np.concatenate([scribble_task,background[...,np.newaxis]],axis = -1)
    scribble_task[scribble_task>0] = 1
        
    scribble = np.array(scribble_task)
    # print("scribble shape",  scribble.shape)

    return scribble



def get_fov_mask(image, scribble):

    np.random.seed(44)

    val_perc = 0.4

    region_val_size = [int(image.shape[0] * val_perc/2),int(image.shape[1] * val_perc/2)] #validation region
    mask_scribbles = np.sum(scribble, axis = -1)
    mask_scribbles[mask_scribbles>0] = 1
    mask_scribbles = ndimage.convolve(mask_scribbles, np.ones([5,5]), mode='constant', cval=0.0)

    val_center = np.random.multinomial(1, mask_scribbles.flatten()/np.sum(mask_scribbles.flatten()), size=1).flatten()
    ix_center = np.argmax(val_center)
    ix_row = int(np.floor(ix_center/image.shape[1]))
    ix_col = int(ix_center - ix_row * image.shape[1])
    # print(ix_center,ix_row,ix_col)

    row_low = np.maximum(ix_row-region_val_size[0],0)
    row_high = np.minimum(row_low+region_val_size[0], image.shape[0])
    row_low = np.maximum(row_high - 2*region_val_size[0],0)
    row_high = np.minimum(row_low+ 2*region_val_size[0], image.shape[0])

    col_low = np.maximum(ix_col-region_val_size[1],0)
    col_high = np.minimum(col_low+region_val_size[1], image.shape[1])
    col_low = np.maximum(col_high - 2*region_val_size[1],0)
    col_high = np.minimum(col_low+2*region_val_size[1], image.shape[1])
    # print(row_low,row_high,col_low,col_high)

    validation_mask = np.zeros([image.shape[0], image.shape[1]])
    validation_mask[row_low:row_high,
                    col_low:col_high] = 1

    return validation_mask 


def prepare_data(image_path, roi_path, scribble_rate=1.0):
    sample = {}
    sample['name'] = image_path
    print("image_path", image_path)
    img =  read_image(image_path)
    img = img.astype(np.float32)
    sample['image'] = percentile_normalization(img, pmin=1, pmax=98, clip = False)


    sample['label'] = readRoiFile(sample['image'], roi_path)
    eroded_label = erosion_labels(sample['label'], radius_pointer=1)
    mask = eroded_label.copy()
    mask[mask>0] = 1
    sample['mask'] = mask
    # sample['scribble'] = get_scribble(mask, total_labels=len(np.unique(sample['label'])))
    # sample['scribble'] = get_scribble(mask, total_labels=30)
    sample['scribble'] = get_scribble(sample['label'], scribble_rate=scribble_rate)
    sample['fov_mask'] = get_fov_mask(sample['image'], sample['scribble'])

    return sample


def prepare_data_test(image_path):
    
    sample = {}
    sample['name'] = image_path
    img =  read_image(image_path)
    img = img.astype(np.float32)
    sample['image'] = percentile_normalization(img, pmin=1, pmax=98, clip = False)

    return sample



def save_model(model, path):
    torch.save(model, path)

def load_model(path):
    model = torch.load(path)
    model.eval()

class DataProcessor():
    def __init__(self, data_dir):
        self.data_dir = data_dir 

    def get_file_list(self):
        
        # data_dir = '/nadeem_lab/Gunjan/data/Vectra_WC_2CH_tiff_full_labels/'
        train_paths = []
        test_paths = []

        for path in glob.iglob(f'{self.data_dir}/*.tif'):           ## modify for png images
            if not os.path.exists(path):
                print("path does not exists")
            # else:
            #     print("path : ", path)

            name = path.split('/')[-1].split('.')[0]
            print("name: ",name)
            rpath = self.data_dir + 'labels/final/' + name + '.zip'
                        
            if os.path.exists(rpath):
                train_paths.append((path, rpath))
                print("roi path : ", rpath)
            else:
                test_paths.append(path)
                print("test path : ", path)

        return train_paths, test_paths
    




def plot_sample(sample, output_file_path=None):

    plt.figure(figsize=(10,10))
    plt.subplot(1,5,1)
    plt.title("image")
    plt.imshow(sample['image'][:,:,0])
    plt.subplot(1,5,2)
    plt.imshow(sample['mask'])

    plt.subplot(1,5,3)
    plt.title('foreground')
    plt.imshow(sample['scribble'][:,:,0])

    plt.subplot(1,5,4)
    plt.title('background')
    plt.imshow(sample['scribble'][:,:,1])

    plt.subplot(1,5,5)
    plt.title('fov_mask')
    plt.imshow(sample['fov_mask'])

    if output_file_path:
        plt.savefig(output_file_path)

    plt.close()


def plot_patch_sample(patch_sample, output_file_path=None):
    # print("patch_sample['input'] shape: ", patch_sample['input'].shape)
    # print("patch_sample['target'] shape: ", patch_sample['target'].shape)
    # print("patch_sample['mask'] shape: ", patch_sample['mask'].shape)
    # print("patch_sample['scribble'] shape: ", patch_sample['scribble'].shape)
    # print("patch_sample['scribble'] shape: ", patch_sample['input'].shape)

    plt.figure(figsize=(10,5))
    plt.subplot(4,2,1)
    plt.title("input ch1")
    plt.imshow(patch_sample['input'][0,:,:])

    plt.subplot(4,2,2)
    plt.title("input ch2")
    plt.imshow(patch_sample['input'][1,:,:])

    plt.subplot(4,2,3)
    plt.title("target ch1")
    plt.imshow(patch_sample['target'][0,:,:])

    plt.subplot(4,2,4)
    plt.title("target ch2")
    plt.imshow(patch_sample['target'][1,:,:])

    plt.subplot(4,2,5)
    plt.title("mask ch1")
    plt.imshow(patch_sample['mask'][0,:,:])

    plt.subplot(4,2,6)
    plt.title("mask ch2")
    plt.imshow(patch_sample['mask'][1,:,:])

    plt.subplot(4,2,7)
    plt.title('foreground')
    plt.imshow(patch_sample['scribble'][0,:,:])

    plt.subplot(4,2,8)
    plt.title('background')
    plt.imshow(patch_sample['scribble'][1,:,:])

    if output_file_path:
        plt.savefig(output_file_path)

    plt.close()



def plot_predictions(data, predictions, output_file_path=None):

    pass 
