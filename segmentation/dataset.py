import numpy as np
import glob 
import os 
import tifffile as tiff 
import sys

from torch.utils.data import Dataset

sys.path.append('../')
from dataprocessing.utils import read_label, percentile_normalization, erosion_labels

class ImageSegDatasetTiff(Dataset):

    def __init__(self, data_dir, img_list, label_list, transform):
        self.data_dir = data_dir

        self.img_list = img_list
        self.label_list = label_list

        self.transform = transform

    def __len__(self):
        return len(self.img_list) 

    def __getitem__(self, idx):
        img_fname = self.img_list[idx]
        label_fname = self.label_list[idx]

        image = tiff.imread(img_fname)
        idx = np.argmin(image.shape)
        image = np.moveaxis(image, idx, -1)  
        
        label = read_label(label_fname, image.shape)
        # label0 = erosion_labels(label,radius_pointer=1)

        mask = (label > 0).astype(np.float32)
        
        X = np.array(image).astype(int)
        X = percentile_normalization(X, pmin=1, pmax=98, clip = False)
        X = X.astype(np.float32)

        if len(X.shape) <= 2:
            X = X[...,np.newaxis]

        if self.transform:
            X = self.transform(X)
            mask = self.transform(mask)

        return img_fname, X, mask


class ImageDatasetTiff(Dataset):

    def __init__(self, data_dir, img_list, transform):
        self.data_dir = data_dir

        self.img_list = img_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list) 

    def __getitem__(self, idx):
        img_fname = self.img_list[idx]

        image = tiff.imread(img_fname)
        idx = np.argmin(image.shape)
        image = np.moveaxis(image, idx, -1)  
                
        X = np.array(image).astype(int)
        X = percentile_normalization(X, pmin=1, pmax=98, clip = False)
        X = X.astype(np.float32)

        if len(X.shape) <= 2:
            X = X[...,np.newaxis]

        if self.transform:
            X = self.transform(X)

        return img_fname, X


class ImageDenoiseDatasetTiff(Dataset):

    def __init__(self, data_dir, img_list, transform):
        self.data_dir = data_dir

        self.img_list = img_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list) 

    def __getitem__(self, idx):
        img_fname = self.img_list[idx]

        image = tiff.imread(img_fname)
        idx = np.argmin(image.shape)
        image = np.moveaxis(image, idx, -1)  
                
        X = np.array(image).astype(int)
        X = percentile_normalization(X, pmin=1, pmax=98, clip = False)
        X = X.astype(np.float32)

        if len(X.shape) <= 2:
            X = X[...,np.newaxis]

        if self.transform:
            X = self.transform(X)

        return img_fname, X, X









############### Delete Later
class CPTissuenetDatasetTiff(Dataset):

    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.img_list = glob.glob(os.path.join(data_dir, 'combined/', 'TN_train_image*.tiff'))
        print("data_dir :", data_dir)
        # print("initial img_list: ", self.img_list)
        vectra_idx_path = '/nadeem_lab/Gunjan/code/Cellpose2.0/impartial_tissunet_data/vectra_idx.npy'
        vectra_indx = np.load(vectra_idx_path)
        # print("vectra_indx: ", vectra_indx)

        vectra_img_list = []
        print("vectra_img_list: ", type(vectra_img_list))
        print("vectra_img_list: ", vectra_img_list)

        for img in self.img_list:
            img_idx = img.split('/nadeem_lab/Gunjan/code/Cellpose2.0/impartial_tissunet_data/combined/TN_train_image')[-1].split('.tiff')[0]
            # print("img_idx: ", img_idx)

            if int(img_idx) in vectra_indx:
                # print("selected image: ", img)
                vectra_img_list.append(img)
                

        # self.img_list = None #choose vectra
        self.img_list = vectra_img_list
        # print("final img_list: ", self.img_list)

        # print("Images list: ", self.img_list)
        self.transform = transform

    def __len__(self):
        return len(self.img_list) 

    def __getitem__(self, idx):
        img_fname = self.img_list[idx]

        img_prefix = img_fname.split('/')[-1].split('.')[0]
        label_prefix = img_prefix + '_cp_masks.tif'

        # print("img_fname: ", img_fname)
        # print("img_prefix: ", img_prefix)
        # print("label_prefix: ", label_prefix)

        label_path = os.path.join(self.data_dir, 'combined_labels', label_prefix)
        # print("label_path: ", label_path)

        image = tiff.imread(img_fname)
        idx = np.argmin(image.shape)
        image = np.moveaxis(image, idx, -1)  

        label = read_label(label_path, image.shape)

        mask = (label > 0).astype(np.long)
        
        X = np.array(image).astype(int)
        X = percentile_normalization(X, pmin=1, pmax=98, clip = False)
        X = X.astype(np.float32)

        if len(X.shape) <= 2:
            X = X[...,np.newaxis]

        if self.transform:
            X = self.transform(X)
            mask = self.transform(mask)

        return img_fname, X, mask
