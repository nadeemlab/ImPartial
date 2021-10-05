import numpy as np
import copy
import os
from skimage.util.shape import view_as_windows

import torch
from torch.utils.data import Dataset

class ImageBlindSpotDataset(Dataset):

    """ reading from Pandas file """
    def __init__(self, pd, data_dir, input_file_tag='input_file',      # todo gs
                 scribble_file_tag='scribble_file', transform=None, validation=False,
                 shift_crop=35, p_scribble_crop=0.5, ratio=0.95,
                 size_window=(10,10), patch_size=(128,128), npatch_image=8):

        self.data_dir = data_dir        # todo gs
        self.inputfiles = pd[input_file_tag].values
        self.scribblefiles = pd[scribble_file_tag].values
        self.patch_size = patch_size

        self.transform = transform
        self.validation = validation

        #sample patch
        self.shift_crop = shift_crop
        self.p_scribble_crop = p_scribble_crop
        self.npatch_image = npatch_image

        #n2v
        self.ratio = ratio
        self.size_window = size_window

        self.data_list = []

    def sample_patches_data(self, npatches_total=np.inf):

        self.data_list = []
        input_files_idx = np.arange(len(self.inputfiles))
        np.random.shuffle(input_files_idx)
        npatches = 0
        for idx in input_files_idx:
            import time
            time_list = []
            time_list.append(time.perf_counter())

            # input
            npz_read = np.load(os.path.join(self.data_dir, self.inputfiles[idx]))       # todo gs
            X = npz_read['image']
            if len(X.shape) <= 2:
                X = X[..., np.newaxis]

            # scribbles
            npz_read = np.load(os.path.join(self.data_dir, self.scribblefiles[idx]))        # todo gs
            S = npz_read['scribble']
            fov_mask = npz_read['val_mask']
            if not self.validation:
                fov_mask = 1 - fov_mask

            time_list.append(time.perf_counter())
            # print('loading images : ', time_list[-1]-time_list[-2])

            # prob mask
            probability_map = np.sum(S, axis=-1)
            probability_map[probability_map > 0] = self.p_scribble_crop / np.sum(probability_map > 0)
            probability_map[probability_map == 0] = (1 - self.p_scribble_crop) / np.sum(probability_map == 0)

            time_list.append(time.perf_counter())
            # print('prob mask : ',time_list[-1]-time_list[-2])

            # crop patch
            Xcrop, Scrop = self.random_crop(X, S, fov_mask * probability_map)

            time_list.append(time.perf_counter())
            # print('crop  : ', time_list[-1] - time_list[-2])

            for i in range(Xcrop.shape[0]):
                self.data_list.append([Xcrop[i,...], Scrop[i,...]])

            npatches += self.npatch_image
            if npatches > npatches_total:
                break

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        Xcrop, Scrop = self.data_list[idx]
        if self.ratio < 1:
            Xout, mask = self.generate_mask(copy.deepcopy(Xcrop))
        else:
            Xout = Xcrop
            mask = np.zeros_like(Xout)
        data = { 'input': Xout, 'target': Xcrop ,'mask': mask, 'scribble': Scrop }
        if self.transform:
            data = self.transform(data)

        return data


    def generate_mask(self, input):
        #input size: patches x h x w x channel

        ratio = self.ratio
        size_window = self.size_window
        size_data = input.shape
        num_sample = int(size_data[0] * size_data[1] * (1 - ratio))
        # num_sample = int(size_data[1] * size_data[2] * (1 - ratio))

        mask = np.ones(size_data)
        output = np.array(input)

        for ich in range(size_data[2]):
            idy_msk = np.random.randint(0, size_data[0], num_sample)
            idx_msk = np.random.randint(0, size_data[1], num_sample)

            idy_neigh = np.random.randint(-size_window[0] // 2 + size_window[0] % 2, size_window[0] // 2 + size_window[0] % 2, num_sample)
            idx_neigh = np.random.randint(-size_window[1] // 2 + size_window[1] % 2, size_window[1] // 2 + size_window[1] % 2, num_sample)

            idy_msk_neigh = idy_msk + idy_neigh
            idx_msk_neigh = idx_msk + idx_neigh

            idy_msk_neigh = idy_msk_neigh + (idy_msk_neigh < 0) * size_data[0] - (idy_msk_neigh >= size_data[0]) * size_data[0]
            idx_msk_neigh = idx_msk_neigh + (idx_msk_neigh < 0) * size_data[1] - (idx_msk_neigh >= size_data[1]) * size_data[1]

            id_msk = (idy_msk, idx_msk, ich)
            id_msk_neigh = (idy_msk_neigh, idx_msk_neigh, ich)

            output[id_msk] = input[id_msk_neigh]
            mask[id_msk] = 0.0

        return output, mask

    def random_crop(self,X,S,probability_mask):

        h, w = X.shape[:2]
        new_h, new_w = self.patch_size

        # print(aux.shape)
        ix = np.argmax(np.random.multinomial(1, probability_mask.flatten() / np.sum(probability_mask.flatten()),
                                             size=self.npatch_image), axis = 1)
        # print(ix)
        center_h = ix // w
        center_w = ix - center_h * w
        # print(center_h,center_w)
        # add random shift to the center
        sign = np.random.binomial(1, 0.5,size = self.npatch_image)
        value = np.random.randint(self.shift_crop,size = self.npatch_image)
        center_h = center_h + value * (1 - sign) + -1 * sign * value
        sign = np.random.binomial(1, 0.5,size = self.npatch_image)
        value = np.random.randint(self.shift_crop,size = self.npatch_image)
        center_w = center_w + value * (1 - sign) + -1 * sign * value

        max_h = np.minimum(center_h + new_h // 2, h)  # minimum between border and max value
        min_h = list(np.maximum(max_h - new_h, 0)) # maximum between border and min value

        max_w = np.minimum(center_w + new_w // 2, w)  # minimum between border and max value
        min_w = list(np.maximum(max_w - new_w, 0))  # maximum between border and min value

        
        patch_size = (new_h, new_w, X.shape[2])
        X_patches = view_as_windows(X,patch_size)
        patch_size = (new_h, new_w, S.shape[2])
        S_patches = view_as_windows(S, patch_size)

        return X_patches[min_h, min_w, 0, ...], S_patches[min_h, min_w, 0, ...]

class ImageSegDataset(Dataset):

    """reading from Pandas file.    """
    def __init__(self, pd, data_dir, input_file_tag='input_file', transform=None): # todo gs

        self.data_dir = data_dir       # todo gs
        self.inputfiles = pd[input_file_tag].values
        self.transform = transform

    def __len__(self):
        return len(self.inputfiles)

    def __getitem__(self, idx):

        #input
        npz_read = np.load(os.path.join(self.data_dir, self.inputfiles[idx]))     # todo gs
        X = npz_read['image']
        if len(X.shape) <= 2:
            X= X[...,np.newaxis]
        Y = npz_read['label']
        if len(Y.shape)<=2:
            Y= Y[...,np.newaxis]

        data = {'input': X, 'label': Y}

        if self.transform:
            data = self.transform(data)

        return data


class RandomFlip(object):
    def __call__(self, data):
        # Random Left or Right Flip

        # for key, value in data:
        #     data[key] = 2 * (value / 255) - 1
        #
        # return data
        input, target, scribble, mask = data['input'], data['target'], data['scribble'], data['mask']

        # print(input.shape)
        if np.random.rand() > 0.5:
            input = np.fliplr(input)
            scribble = np.fliplr(scribble)
            mask = np.fliplr(mask)
            target = np.fliplr(target)

        if np.random.rand() > 0.5:
            input = np.flipud(input)
            scribble = np.flipud(scribble)
            mask = np.flipud(mask)
            target = np.flipud(target)

        return {'input': input, 'target': target, 'scribble': scribble, 'mask': mask}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self,dim_data = 3):
        self.dim_data = dim_data

    def __call__(self, data):
        # Swap color axis because numpy image: H x W x C
        #                         torch image: C x H x W

        # for key, value in data:
        #     data[key] = torch.from_numpy(value.transpose((2, 0, 1)))
        #
        # return data
        output = {}
        if self.dim_data == 3:
            for key in data.keys():
                output[key] = torch.from_numpy(data[key].transpose((2, 0, 1)).astype(np.float32))

        if self.dim_data == 4:
            for key in data.keys():
                output[key] = torch.from_numpy(data[key].transpose((0, 3, 1, 2)).astype(np.float32))
        return output


class Normalize(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        input = data['input']
        target = data['target']

        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std

        data['input'] = input
        data['target'] = target
        return data

class ToNumpy(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        # Swap color axis because numpy image: H x W x C
        #                         torch image: C x H x W

        # for key, value in data:
        #     data[key] = value.transpose((2, 0, 1)).numpy()
        #
        # return data

        return data.to('cpu').detach().numpy().transpose(0, 2, 3, 1)

        # input, label = data['input'], data['label']
        # input = input.transpose((2, 0, 1))
        # label = label.transpose((2, 0, 1))
        # return {'input': input.detach().numpy(), 'label': label.detach().numpy()}

class Denormalize(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        data = self.std * data + self.mean
        return data

