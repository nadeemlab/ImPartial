import numpy as np
import glob 
import os 
import sys 

sys.path.append('../')


import matplotlib.pyplot as plt
import sys
import glob
# sys.path.append("../")
# import tissuenet
from PIL import Image
from collections import Counter

def func():
    dir = "/nadeem_lab/Gunjan/code/Cellpose2.0/impartial_tissunet_data/tissuenet_orig_npz/"
    file = os.path.join(dir, "tissuenet_v1.0_train.npz")

    npz = np.load(file)
    print(npz.files)

    image, label, tissue_list, platform_list = npz['X'], npz['y'], npz['tissue_list'], npz['platform_list']

    valid_tissues = np.unique(tissue_list)
    valid_platforms = np.unique(platform_list)

    print("Valid tissues: ", valid_tissues)
    print(valid_platforms)

    values, counts = np.unique(platform_list, return_counts=True)
    print(counts)

    freq = Counter(platform_list)
    print(freq)

    vectra_idx = []
    mibi_idx = []
    for i in range(0, image.shape[0]):

        if platform_list[i] == 'vectra':
            vectra_idx.append(i)
        
        if platform_list[i] == 'mibi':
            mibi_idx.append(i)

    print('vectra_idx: ', vectra_idx)
    print('vectra_idx len: ', len(vectra_idx))

    print('mibi_idx len: ', len(mibi_idx))
    print('mibi_idx: ', mibi_idx)

    # save_vetra_indx_filname = '/nadeem_lab/Gunjan/code/Cellpose2.0/impartial_tissunet_data/vectra_idx.npy'
    # np.save(save_vetra_indx_filname, vectra_idx)



# --------------------------------
# generate / prepare dataset 


def get_train_data(data_dir):
    ### old code : with old un-eroded labels
    image_list = []
    label_list = [] 

    img_list = glob.glob(os.path.join(data_dir, 'combined/', 'TN_train_image*.tiff'))

    vectra_idx_path = '/nadeem_lab/Gunjan/code/Cellpose2.0/impartial_tissunet_data/vectra_idx.npy'
    vectra_indx = np.load(vectra_idx_path)
    vectra_img_list = []
    for img in img_list:
        # print("images : ", img) # TODO 

        img_idx = img.split('/nadeem_lab/Gunjan/code/Cellpose2.0/impartial_tissunet_data/combined/TN_train_image')[-1].split('.tiff')[0]
        # print("img_idx: ", img_idx)

        if int(img_idx) in vectra_indx:
            # print("selected image: ", img)
            vectra_img_list.append(img)
                    
    image_list = vectra_img_list

    for image_list_item in image_list:
        img_prefix = image_list_item.split('/')[-1].split('.')[0]
        label_prefix = img_prefix + '_cp_masks.tif'
        label_list_item = os.path.join(data_dir, 'combined_labels', label_prefix)
        label_list.append(label_list_item)
        # print("label_list_item : ", label_list_item)
    
    return image_list, label_list


# --------------------------------

def train_data(data_dir):

    image_list = []
    label_list = [] 

    img_list = glob.glob(os.path.join(data_dir, 'combined/', 'TN_train_image*.tiff'))

    vectra_idx_path = '/nadeem_lab/Gunjan/code/Cellpose2.0/impartial_tissunet_data/vectra_idx.npy'
    vectra_indx = np.load(vectra_idx_path)

    for img in img_list:
        # print("image : ", img) # TODO 

        img_idx = img.split(os.path.join(data_dir, 'combined/', 'TN_train_image'))[-1].split('.tiff')[0]
        # print("img_idx: ", img_idx)

        if int(img_idx) in vectra_indx:
            # print("selected image: ", img)
            image_list.append(img)
                    
            img_prefix = img.split('/')[-1].split('.')[0]
            # print("img_prefix : ", img_prefix)

            label_prefix = img_prefix + '_cp_masks.tif'
            label_list_item = os.path.join(data_dir, 'combined_labels_erode/', label_prefix)
            label_list.append(label_list_item)
            # print("label_list_item : ", label_list_item)
        
    return image_list, label_list

# --------------------------------


def test_data(data_dir):
    # generate / prepare test dataset 
    image_list = []
    label_list = [] 

    img_list = glob.glob(os.path.join(data_dir, 'combined/', 'TN_test_image*.tiff'))
    print("test image dataset len: ", len(img_list))

    vectra_test_idx_path = '/nadeem_lab/Gunjan/code/Cellpose2.0/impartial_tissunet_data/vectra_test_idx.npy'
    vectra_test_indx = np.load(vectra_test_idx_path)
    vectra_test_img_list = []

    for img in img_list:
        img_idx = img.split('/nadeem_lab/Gunjan/code/Cellpose2.0/impartial_tissunet_data/combined/TN_test_image')[-1].split('.tiff')[0]
        # print("img_idx: ", img_idx)

        if int(img_idx) in vectra_test_indx:
            # print("selected image: ", img)
            vectra_test_img_list.append(img)
                    
    image_list = vectra_test_img_list

    print("test vectra image dataset len: ", len(image_list))

    for image_list_item in image_list:
        img_prefix = image_list_item.split('/')[-1].split('.')[0]
        label_prefix = img_prefix + '_cp_masks.tif'
        label_list_item = os.path.join(data_dir, 'combined_labels_erode/', label_prefix)
        label_list.append(label_list_item)

    print("test label dataset len: ", len(label_list))

    return image_list, label_list