import numpy as np
import copy
import torch
import random
from skimage.util.shape import view_as_windows

from torch.utils.data import Dataset

class ImageBlindSpotDataset(Dataset):

    def __init__(self, 
                 data,
                 validation=False,
                 transform=None,
                 shift_crop=35, p_scribble_crop=0.5, ratio=0.95,
                 size_window=(10,10), 
                 patch_size=(256,256), 
                 npatch_image=8):

        self.patch_size = patch_size
        self.transform = transform

        #sample patch
        self.shift_crop = shift_crop
        self.p_scribble_crop = p_scribble_crop
        self.npatch_image = npatch_image

        #n2v
        self.ratio = ratio
        self.size_window = size_window

        self.validation = validation
        self.data = data 
        self.data_list = []

    def sample_patches_data(self, npatches_total=np.inf):

        self.data_list = []
        for idx in range(0, len(self.data)):

            X = self.data[idx]['image']
            S = self.data[idx]['scribble']
            fov_mask = self.data[idx]['fov_mask']

            if len(X.shape) <= 2:
                X = X[..., np.newaxis]

            if not self.validation:
                fov_mask = 1 - fov_mask

            # prob mask
            probability_map = np.sum(S, axis=-1)
            probability_map[probability_map > 0] = self.p_scribble_crop / np.sum(probability_map > 0)
            probability_map[probability_map == 0] = (1 - self.p_scribble_crop) / np.sum(probability_map == 0)

            # crop patch
            # Xcrop, Scrop = self.random_crop(X, S, fov_mask * probability_map)
            Xcrop, Scrop = self.random_crop(X, S, probability_map) # No fov_mask

            # Vectorized: extend data_list with all patches at once
            self.data_list.extend([[x, s] for x, s in zip(Xcrop, Scrop)])

    def __len__(self):

        return len(self.data_list)

    def __getitem__(self, idx):

        Xcrop, Scrop = self.data_list[idx]
        if random.random() < 0.9: # TODO: re-check this
            # Xout, mask = self.generate_mask(copy.deepcopy(Xcrop))
            Xout, mask = self.generate_mask(Xcrop)
        else:
            Xout = Xcrop
            mask = np.zeros_like(Xout)

        data = { 'input': Xout, 'target': Xcrop ,'mask': mask, 'scribble': Scrop }
        if self.transform:
            data = self.transform(data)

        return data


    def generate_mask(self, input):
        #input size: patches x h x w x channel

        size_window = self.size_window
        size_data = input.shape
        num_sample = int(size_data[0] * size_data[1] * (1 - self.ratio))

        mask = np.ones(size_data)
        output = input.copy()

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

    def random_crop(self, X, S, probability_mask):

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
        sign = np.random.binomial(1, 0.5, size = self.npatch_image)
        value = np.random.randint(self.shift_crop,size = self.npatch_image)
        center_h = center_h + value * (1 - sign) + -1 * sign * value
        sign = np.random.binomial(1, 0.5, size = self.npatch_image)
        value = np.random.randint(self.shift_crop,size = self.npatch_image)
        center_w = center_w + value * (1 - sign) + -1 * sign * value

        max_h = np.minimum(center_h + new_h // 2, h)  # minimum between border and max value
        min_h = list(np.maximum(max_h - new_h, 0)) # maximum between border and min value

        max_w = np.minimum(center_w + new_w // 2, w)  # minimum between border and max value
        min_w = list(np.maximum(max_w - new_w, 0))  # maximum between border and min value

        
        patch_size = (new_h, new_w, X.shape[2])
        X_patches = view_as_windows(X, patch_size)
        patch_size = (new_h, new_w, S.shape[2])
        S_patches = view_as_windows(S, patch_size)

        return X_patches[min_h, min_w, 0, ...], S_patches[min_h, min_w, 0, ...]


class ImageSegDataset(Dataset):

    def __init__(self, data,
                 transform=None): 

        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        X = self.data[idx]['image']
        if len(X.shape) <= 2:
            X= X[...,np.newaxis]

        Y = self.data[idx]['label']
        if len(Y.shape) <= 2:
            Y= Y[..., np.newaxis]

        data = {'input': X, 'label': Y}

        if self.transform:
            data = self.transform(data)

        data['image_name'] = self.data[idx]['name']
        return data

class ImageDataset(Dataset):

    def __init__(self, data,
                 transform=None): 

        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        X = self.data[idx]['image']
        if len(X.shape) <= 2:
            X= X[...,np.newaxis]

        data = {'input': X}

        if self.transform:
            data = self.transform(data)

        data['image_name'] = self.data[idx]['name']
        return data

class RandomFlip(object):
    def __call__(self, data):
        input, target, scribble, mask = data['input'], data['target'], data['scribble'], data['mask']

        if np.random.rand() > 0.4:
            input = np.fliplr(input)
            scribble = np.fliplr(scribble)
            mask = np.fliplr(mask)
            target = np.fliplr(target)

        if np.random.rand() > 0.4:
            input = np.flipud(input)
            scribble = np.flipud(scribble)
            mask = np.flipud(mask)
            target = np.flipud(target)

        return {'input': input, 'target': target, 'scribble': scribble, 'mask': mask}


class RandomRotate(object):
    def __call__(self, data):
        input, target, scribble, mask = data['input'], data['target'], data['scribble'], data['mask']

        k = np.random.randint(0, 4)
        input = np.rot90(input, k=k)
        scribble = np.rot90(scribble, k=k)
        mask = np.rot90(mask, k=k)
        target = np.rot90(target, k=k)

        return {'input': input, 'target': target, 'scribble': scribble, 'mask': mask}


class RandomPermuteChannel(object):
    def __call__(self, data):

        input, target, scribble, mask = data['input'], data['target'], data['scribble'], data['mask']
        n_channels = input.shape[-1]

        # Random channel permutation
        permuted_channels = np.random.permutation(n_channels)
        input = input[..., permuted_channels]
        target = target[..., permuted_channels]
        mask = mask[..., permuted_channels]

        if np.random.rand() < 0.0:
            # Randomly choose which channel to drop
            channel_to_drop = np.random.randint(0, n_channels)
            input[..., channel_to_drop] = 0.0
            target[..., channel_to_drop] = 0.0
            mask[..., channel_to_drop] = 0.0

        return {'input': input, 'target': target, 'scribble': scribble, 'mask': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, dim_data = 3):
        self.dim_data = dim_data

    def __call__(self, data):
        # Swap color axis because numpy image: H x W x C
        #                         torch image: C x H x W

        output = {}
        keys = list(data.keys())
        if self.dim_data == 3:
            for key in keys:
                output[key] = torch.from_numpy(data[key].transpose((2, 0, 1)).astype(np.float32))

        if self.dim_data == 4:
            for key in keys:
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

