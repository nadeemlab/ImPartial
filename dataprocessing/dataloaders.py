import math
import os
import copy
import random

import torch
from torch.utils.data import Dataset
import numpy as np
from skimage.util.shape import view_as_windows


class ImageBlindSpotDataset(Dataset):
    """ reading from Pandas file """

    def __init__(self, pd, data_dir, input_file_tag='input_file',  # todo gs
                 scribble_file_tag='scribble_file', transform=None, validation=False,
                 shift_crop=35, p_scribble_crop=0.5, ratio=0.95,
                 size_window=(10, 10), patch_size=(128, 128), npatch_image=8):
        self.data_dir = data_dir  # todo gs
        self.inputfiles = pd[input_file_tag].values
        self.scribblefiles = pd[scribble_file_tag].values
        self.patch_size = patch_size

        self.transform = transform
        self.validation = validation

        # sample patch
        self.shift_crop = shift_crop
        self.p_scribble_crop = p_scribble_crop
        self.npatch_image = npatch_image

        # n2v
        self.ratio = ratio
        self.size_window = size_window

        self.data_list = []

    def sample_patches_data(self, npatches_total=np.inf):
        self.data_list = sample_from_files(
            data_dir=self.data_dir,
            inputfiles=self.inputfiles,
            scribblefiles=self.scribblefiles,
            validation=self.validation,
            p_scribble_crop=self.p_scribble_crop,
            patch_size=self.patch_size,
            shift_crop=self.shift_crop,
            npatches_total=npatches_total
        )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return transform_patch(
            patch=self.data_list[idx],
            ratio=self.ratio,
            size_window=self.size_window,
            transform=self.transform
        )

    def generate_mask(self, input):
        # input size: patches x h x w x channel
        return GenerateMask(self.ratio, self.size_window)(input)


def transform_patch(patch, ratio, size_window, transform):
    Xcrop, Scrop = patch
    if ratio < 1:
        Xout, mask = GenerateMask(ratio, size_window)(Xcrop)
    else:
        Xout = Xcrop
        mask = np.zeros_like(Xout)
    data = {'input': Xout, 'target': Xcrop, 'mask': mask, 'scribble': Scrop}
    if transform:
        data = transform(data)

    return data


def sample_from_files(
        data_dir,
        inputfiles,
        scribblefiles,
        validation,
        p_scribble_crop,
        patch_size,
        shift_crop,
        npatches_total
):
    images = [np.load(os.path.join(data_dir, f))["image"] for f in inputfiles]
    scribbles = [np.load(os.path.join(data_dir, f))["scribble"] for f in scribblefiles]
    fov_masks = [np.load(os.path.join(data_dir, f))["val_mask"] for f in scribblefiles]

    return sample_patches(
        images,
        scribbles,
        fov_masks,
        validation,
        p_scribble_crop,
        patch_size,
        shift_crop,
        npatches_total
    )


def sample_patches(
        images,
        scribbles,
        fov_masks,
        validation,
        p_scribble_crop,
        patch_size,
        shift_crop,
        npatches_total
):
    # number of patches to extract per image
    npatch_image = math.ceil(npatches_total / len(scribbles))

    data_list = []
    for image, scribble, mask in sorted(zip(images, scribbles, fov_masks), key=lambda k: random.random()):
        if len(image.shape) <= 2:
            image = mask[..., np.newaxis]

        if not validation:
            mask = 1 - mask

        # make sure we generate exactly `npatches_total` patches
        npatch_image = min(npatch_image, npatches_total - len(data_list))

        # crop patch
        image_patches, scribble_patches = random_crop(
            X=image,
            S=scribble,
            probability_mask=mask * compute_probability_map(scribble, p_scribble_crop),
            patch_size=patch_size,
            npatch_image=npatch_image,
            shift_crop=shift_crop
        )

        for i in range(image_patches.shape[0]):
            data_list.append([image_patches[i, ...], scribble_patches[i, ...]])

    return data_list


def compute_probability_map(scribble, p_scribble_crop):
    probability_map = np.sum(scribble, axis=-1)
    probability_map[probability_map > 0] = p_scribble_crop / np.sum(probability_map > 0)
    probability_map[probability_map == 0] = (1 - p_scribble_crop) / np.sum(probability_map == 0)

    return probability_map


def random_crop(X, S, probability_mask, patch_size, npatch_image, shift_crop):
    h, w = X.shape[:2]
    new_h, new_w = patch_size

    ix = np.argmax(
        np.random.multinomial(
            n=1,
            pvals=probability_mask.flatten() / np.sum(probability_mask.flatten()),
            size=npatch_image
        ),
        axis=1
    )

    center_h = ix // w
    center_w = ix - center_h * w

    # add random shift to the center
    sign = np.random.binomial(1, 0.5, size=npatch_image)
    value = np.random.randint(shift_crop, size=npatch_image)
    center_h = center_h + value * (1 - sign) + -1 * sign * value

    sign = np.random.binomial(1, 0.5, size=npatch_image)
    value = np.random.randint(shift_crop, size=npatch_image)
    center_w = center_w + value * (1 - sign) + -1 * sign * value

    max_h = np.minimum(center_h + new_h // 2, h)  # minimum between border and max value
    min_h = list(np.maximum(max_h - new_h, 0))  # maximum between border and min value

    max_w = np.minimum(center_w + new_w // 2, w)  # minimum between border and max value
    min_w = list(np.maximum(max_w - new_w, 0))  # maximum between border and min value

    patch_size = (new_h, new_w, X.shape[2])
    X_patches = view_as_windows(X, patch_size)

    patch_size = (new_h, new_w, S.shape[2])
    S_patches = view_as_windows(S, patch_size)

    return X_patches[min_h, min_w, 0, ...], S_patches[min_h, min_w, 0, ...]


class ImageSegDataset(Dataset):
    """reading from Pandas file.    """

    def __init__(self, pd, data_dir, input_file_tag='input_file', transform=None):  # todo gs

        self.data_dir = data_dir  # todo gs
        self.inputfiles = pd[input_file_tag].values
        self.transform = transform

    def __len__(self):
        return len(self.inputfiles)

    def __getitem__(self, idx):

        # input
        npz_read = np.load(os.path.join(self.data_dir, self.inputfiles[idx]))  # todo gs
        X = npz_read['image']
        if len(X.shape) <= 2:
            X = X[..., np.newaxis]
        Y = npz_read['label']
        if len(Y.shape) <= 2:
            Y = Y[..., np.newaxis]

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


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, dim_data=3, keys=None, exclude_keys=None):
        self.dim_data = dim_data
        self.keys = [keys] if isinstance(keys, str) else keys
        self.exclude_keys = [exclude_keys] if isinstance(exclude_keys, str) else exclude_keys

    def __call__(self, data):
        # Swap color axis because numpy image: H x W x C
        #                         torch image: C x H x W

        res = copy.deepcopy(data)

        # filter data.keys using defined keys
        keys = data.keys()
        if self.keys:
            keys = [k for k in keys if k in self.keys]

        if self.exclude_keys:
            keys = [k for k in keys if k not in self.exclude_keys]

        if self.dim_data == 3:
            for key in keys:
                res[key] = torch.from_numpy(data[key].transpose((2, 0, 1)).astype(np.float32))

        if self.dim_data == 4:
            for key in keys:
                res[key] = torch.from_numpy(data[key].transpose((0, 3, 1, 2)).astype(np.float32))

        return res


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


def blind_spot_patch(input, ratio=0.95, size_window=(10, 10)):
    # input size: patches x h x w x channel
    window_height, window_width = size_window

    batch_size, height, width, channels = input.shape
    num_sample = int(height * width * (1 - ratio))

    mask = np.ones(input.shape)
    output = np.array(input)

    for ich in range(channels):
        idy_msk = np.random.randint(0, height, num_sample)
        idx_msk = np.random.randint(0, width, num_sample)

        idy_neigh = np.random.randint(
            -window_height // 2 + window_height % 2,
            window_height // 2 + window_height % 2,
            num_sample
        )
        idx_neigh = np.random.randint(
            -window_width // 2 + window_width % 2,
            window_width // 2 + window_width % 2,
            num_sample
        )

        idy_msk_neigh = idy_msk + idy_neigh
        idx_msk_neigh = idx_msk + idx_neigh

        idy_msk_neigh = idy_msk_neigh + (idy_msk_neigh < 0) * height - (idy_msk_neigh >= height) * height
        idx_msk_neigh = idx_msk_neigh + (idx_msk_neigh < 0) * width - (idx_msk_neigh >= width) * width

        output[:, idy_msk, idx_msk, ich] = input[:, idy_msk_neigh, idx_msk_neigh, ich]
        mask[:, idy_msk, idx_msk, ich] = 0.0

    return output, mask
