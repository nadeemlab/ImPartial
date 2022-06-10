import os
import random

import roifile
from roifile import ImagejRoi
import cv2 as cv
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import morphology, measure


def rois_to_mask(zip_path, size, sample_rate=1):
    rois = roifile.roiread(zip_path)
    mask = np.zeros(size).astype(np.uint8)

    for roi in random.sample(rois, int(len(rois) * sample_rate)):
        if roi.integer_coordinates:
            coord = roi.integer_coordinates
            coord[:, 0] += roi.left
            coord[:, 1] += roi.top

        elif roi.multi_coordinates is not None:
            coord = ImagejRoi.path2coords(roi.multi_coordinates)[0]

        else:
            raise RuntimeError("ROI type not supported")

        contour = np.asarray(coord).astype(np.int32)

        cv.fillPoly(mask, pts=[contour], color=255)

    return mask


def rois_to_labels(zip_path, size, sample_rate=1):
    rois = roifile.roiread(zip_path)

    roi_samples = random.sample(rois, int(len(rois) * sample_rate))
    roi_contours = [get_contour(roi) for roi in roi_samples]

    return np.stack((
        generate_foreground_scribble(roi_contours, size),
        generate_background_scribble(roi_contours, size)
    ), 2)


def generate_foreground_scribble(contours, size):
    foreground_scribble = np.zeros(size).astype(np.uint8)

    for c in contours:
        mask = np.zeros(size).astype(np.uint8)
        cv.fillPoly(mask, pts=[c], color=1)

        eroded = morphology.binary_erosion(mask, footprint=np.ones((5, 5))).astype(np.uint8)
        skeletonized = morphology.skeletonize(eroded).astype(np.uint8)

        foreground_contours = np.zeros(eroded.shape)
        for c in measure.find_contours(eroded):
            c = c.astype(np.uint32)
            foreground_contours[c[:, 0], c[:, 1]] = 1

        foreground_scribble += skeletonized + foreground_contours.astype(np.uint8)

    return np.clip(foreground_scribble, 0, 1)


def generate_background_scribble(contours, size):
    mask = np.zeros(size).astype(np.uint8)
    for c in contours:
        cv.polylines(mask, pts=[c], color=1, isClosed=True)

    return mask


def get_contour(roi):
    if roi.integer_coordinates is not None:
        coord = roi.integer_coordinates
        coord[:, 0] += roi.left
        coord[:, 1] += roi.top

    elif roi.multi_coordinates is not None:
        coord = ImagejRoi.path2coords(roi.multi_coordinates)[0]

    else:
        raise RuntimeError("ROI type not supported")

    return np.asarray(coord).astype(np.int32)


def read_files(path):
    filenames = os.listdir(path)

    dataset = []
    for fn in filenames:
        name, ext = os.path.splitext(fn)
        if ext.lower() == ".png" and "-" in name:
            prefix, _ = fn.split("-")

            dataset.append({
                "image": fn,
                "zip": next(f for f in filenames if f.startswith(prefix) and f.endswith(".zip"))
            })

    return dataset


def generate_tiles(img, tile_size):
    rows = int(img.height / tile_size)  # Number of tiles in the row
    cols = int(img.width / tile_size)  # Number of tiles in the column

    # Generating the tiles
    for i in range(cols):
        for j in range(rows):
            yield img.crop((
                i * tile_size, j * tile_size,
                i * tile_size + tile_size,
                j * tile_size + tile_size
            ))


def validation_mask(scribble, val_split):
    """
    Compute validation sample region mask
    """
    scribble_mask = scribble.copy()
    scribble_mask[scribble_mask > 0] = 1

    scribble_mask = ndimage.convolve(scribble_mask, np.ones([5, 5]), mode='constant', cval=0.0)

    # remove borders
    val_size = [int(scribble_mask.shape[0] * val_split / 2),
                int(scribble_mask.shape[1] * val_split / 2)]  # validation region
    # scribble_mask[-val_size[0]:, :] = 0
    # scribble_mask[:, -val_size[1]:] = 0
    # scribble_mask[0:val_size[0], :] = 0
    # scribble_mask[:, 0:val_size[1]] = 0

    val_center = np.random.multinomial(
        n=1,
        pvals=scribble_mask.flatten() / np.sum(scribble_mask),
        size=1
    ).flatten()

    center = np.argmax(val_center)
    row = np.clip(int(np.floor(center / scribble_mask.shape[1])), val_size[1], scribble_mask.shape[1] - val_size[1])
    col = np.clip(int(center - row * scribble_mask.shape[1]), val_size[1], scribble_mask.shape[1] - val_size[1])

    mask = np.zeros(scribble_mask.shape)
    mask[row - val_size[0]:row + val_size[0], col - val_size[1]:col + val_size[1]] = 1

    return mask.astype(np.uint8)


def transform_dataset(dataset_dir, output_dir):
    """
    Transform PNG images of any size with corresponding zip files
    containing a list of ImageJ ROIs files into a
    list of 400 x 400 images with the corresponding
    binary mask label built from the ROIs zip
    """
    os.makedirs(os.path.join(output_dir, "labels", "final"), exist_ok=True)

    dataset = []

    for f in read_files(dataset_dir):
        img = Image.open(os.path.join(dataset_dir, f["image"]))
        mask = rois_to_mask(
            zip_path=os.path.join(dataset_dir, f["zip"]),
            size=img.size
        )

        if img.size > (400, 400):
            for i, m in zip(generate_tiles(img, 400), generate_tiles(mask, 400)):
                dataset.append({"image": i, "label": m})

        else:
            dataset.append({"image": img, "label": mask})

    for i, d in enumerate(dataset):
        d["image"].save(os.path.join(output_dir, f"image{i}.png"))
        d["label"].save(os.path.join(output_dir, "labels", "final", f"image{i}.png"))


def generate_validation_masks(dataset_dir):
    masks_dir = os.path.join(dataset_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    labels_dir = os.path.join(dataset_dir, "labels")
    labels = [f for f in os.listdir(labels_dir) if os.path.splitext(f)[1] == ".png"]

    for l in labels:
        label_path = os.path.join(dataset_dir, "labels", l)
        label = Image.open(label_path)

        mask = validation_mask(label, 0.3)

        Image.fromarray(mask * 255).save(os.path.join(masks_dir, l))
