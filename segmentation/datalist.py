import numpy as np
import glob 
import os 
from collections import Counter


def tissuenet_dataset_filter(train_val_test='train', platform=None):

    tissue_net_dataset_npz_file = f"/nadeem_lab/Gunjan/code/Cellpose2.0/impartial_tissunet_data/tissuenet_orig_npz/tissuenet_v1.0_{train_val_test}.npz"
    npz = np.load(tissue_net_dataset_npz_file)

    image, label, tissue_list, platform_list = npz['X'], npz['y'], npz['tissue_list'], npz['platform_list']

    valid_tissues = np.unique(tissue_list)
    valid_platforms = np.unique(platform_list)

    print("Valid tissues: ", valid_tissues)
    print("Valid platforms: ", valid_platforms)

    freq = Counter(platform_list)
    print("Platforms frequency: ", freq)

    idx = []
    for i in range(0, image.shape[0]):

        if platform == None:
            idx.append(i)

        else:
            if platform_list[i] == platform:
                idx.append(i)

    return idx


def get_tissuenet_datalist(data_dir, train_val_test='train', platform='vectra'):

    image_list = []
    label_list = [] 

    img_list = glob.glob(os.path.join(data_dir, 'combined/', f'TN_{train_val_test}_image*.tiff'))

    data_list_idx = tissuenet_dataset_filter(train_val_test=train_val_test, platform=platform)

    for img in img_list:
        img_idx = img.split(os.path.join(data_dir, 'combined/', f'TN_{train_val_test}_image'))[-1].split('.tiff')[0]

        if int(img_idx) in data_list_idx:
            # print("selected image: ", img)
            image_list.append(img)
                    
            img_prefix = img.split('/')[-1].split('.')[0]
            # print("img_prefix : ", img_prefix)

            label_prefix = img_prefix + '_cp_masks.tif'
            label_list_item = os.path.join(data_dir, 'combined_labels_erode/', label_prefix)
            label_list.append(label_list_item)
            # print("label_list_item : ", label_list_item)
        
    return image_list, label_list


if __name__ == "__main__":

    data_list_idx = tissuenet_dataset_filter(train_val_test='train', platform=None)
    print("Train, None", len(data_list_idx))

    data_list_idx = tissuenet_dataset_filter(train_val_test='val', platform='vectra')
    print("Val, vectra", len(data_list_idx))

    data_list_idx = tissuenet_dataset_filter(train_val_test='test', platform='vectra')
    print("Test, vectra", len(data_list_idx))

    data_list_idx = tissuenet_dataset_filter(train_val_test='train', platform='mibi')
    print("Train, vectra", len(data_list_idx))
