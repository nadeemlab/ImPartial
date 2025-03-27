import math
import random
import torch
import numpy as np
from skimage.util.shape import view_as_windows

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
            image = image[..., np.newaxis]

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
    probability_map = np.sum(scribble, axis=-1).astype(np.float64)
    probability_map[probability_map > 0] = p_scribble_crop / np.sum(probability_map > 0)
    probability_map[probability_map == 0] = (1 - p_scribble_crop) / np.sum(probability_map == 0)

    return probability_map


def random_crop(X, S, probability_mask, patch_size, npatch_image, shift_crop):
    h, w = X.shape[:2]
    new_h, new_w = patch_size

    ix = np.argmax(
        np.random.multinomial(
            n=1,
            pvals=probability_mask.flatten() / np.sum(probability_mask),
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


def blind_spot_patch(input, ratio=0.95, size_window=(10, 10)):
    # input size: (patches x h x w x channel) or (h x w x channel)
    window_height, window_width = size_window

    channels = input.shape[-1]
    width = input.shape[-2]
    height = input.shape[-3]
    num_sample = int(height * width * (1 - ratio))

    if isinstance(input, torch.Tensor):
        mask = torch.ones_like(input)
        output = input.clone().detach()
    else:
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

        if len(input.shape) > 3:
            output[:, idy_msk, idx_msk, ich] = input[:, idy_msk_neigh, idx_msk_neigh, ich]
            mask[:, idy_msk, idx_msk, ich] = 0.0
        else:
            output[idy_msk, idx_msk, ich] = input[idy_msk_neigh, idx_msk_neigh, ich]
            mask[idy_msk, idx_msk, ich] = 0.0


    return output, mask
