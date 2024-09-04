import os
import random

import roifile
from roifile import ImagejRoi
import cv2 as cv
import numpy as np
from PIL import Image
import tifffile as tiff

from scipy import ndimage
from skimage import morphology, measure


def percentile_normalization(image, pmin=1, pmax=98, clip = False):
    # Normalize the image using percentiles
    lo, hi = np.percentile(image, (pmin, pmax))
    image_norm_percentile = (image - lo) / (hi - lo)
    
    if clip:
        image_norm_percentile[image_norm_percentile>1] = 1
        image_norm_percentile[image_norm_percentile<0] = 0
        
    return image_norm_percentile


def read_image(path):
    extension = os.path.splitext(path)[-1][1:].lower()

    if extension == "png":
        image = np.array(Image.open(path))
    elif extension in ("tiff", "tif"):
        image = tiff.imread(path)
    else:
        raise RuntimeError(f"File type '{extension}' not supported from path '{path}'")

    if len(image.shape) == 2:
        image = image[np.newaxis, ...]

    idx = np.argmin(image.shape)
    image = np.moveaxis(image, idx, -1)

    return image


def read_label(path, image_shape):
    extension = path.split(".")[-1].lower()

    if extension == "tiff" or extension == "tif":
        label = np.array(tiff.imread(path))

    if extension == "zip" or extension == "ZIP":
        roi = roifile.roiread(path)
        label = np.zeros((image_shape[0], image_shape[1])).astype(np.int32)
        convert_roi_to_label(label, roi)

    label = (label).astype(np.float32)

    return label


def convert_roi_to_label(label, rois):

    for i in range(0, len(rois)):
        # coord = rois[i].integer_coordinates
        # top = rois[i].top
        # left = rois[i].left
        # coord[:, 0] = coord[:, 0] + left
        # coord[:, 1] = coord[:, 1] + top

        # contour = []
        # for j in range(0, len(coord)):
        #     contour.append([ coord[j][0], coord[j][1] ])

        # contour = np.asarray(coord).astype(np.int32)
        contour = get_contour(rois[i])

        # cv.drawContours(img, [contour], -1, (0,255,0), 1)
        # cv.drawContours(label, [contour], -1, (i), 1)
        cv.fillPoly(label, pts=[contour], color=i)

def rois_to_mask(zip_path, size, sample_rate=1):
    rois = roifile.roiread(zip_path)
    mask = np.zeros(size).astype(np.uint8)

    for roi in random.sample(rois, int(len(rois) * sample_rate)):
        if roi.integer_coordinates is not None:
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


def rois_to_labels(zip_path, size, sample_rate=1.0):
    rois = roifile.roiread(zip_path)

    roi_samples = random.sample(rois, int(len(rois) * sample_rate))
    roi_contours = [get_contour(roi) for roi in roi_samples]

    label_mask = np.zeros(size).astype(np.uint8)
    for c in roi_contours:
        cv.fillPoly(label_mask, pts=[c], color=1)
    label_mask[label_mask > 0] = 1

    return np.stack((
        generate_foreground_scribble(roi_contours, size),
        # 1-label_mask
        # generate_background_scribble(roi_contours, size)*(1-label_mask)
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

    # mask = morphology.skeletonize(mask)
    return mask


def get_contour(roi):
    new_coord = []
    if roi.integer_coordinates is not None:
        coord = roi.integer_coordinates
        for i in range(len(coord)):
            new_coord.append([coord[i, 0] + roi.left, coord[i, 1] + roi.top])

        return np.asarray(new_coord).astype(np.int32)

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
    region_val_size = [int(scribble_mask.shape[0] * val_split / 2),
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

    ix_center = np.argmax(val_center)
    ix_row = int(np.floor(ix_center/scribble_mask.shape[1]))
    ix_col = int(ix_center - ix_row * scribble_mask.shape[1])
    # print(ix_center,ix_row,ix_col)
    
    row_low = np.maximum(ix_row   - region_val_size[0]  , 0)
    row_high = np.minimum(row_low + region_val_size[0]  , scribble_mask.shape[0])
    row_low = np.maximum(row_high - 2*region_val_size[0], 0)
    row_high = np.minimum(row_low + 2*region_val_size[0], scribble_mask.shape[0])
    
    col_low = np.maximum(ix_col   - region_val_size[1]  , 0)
    col_high = np.minimum(col_low + region_val_size[1]  , scribble_mask.shape[1])
    col_low = np.maximum(col_high - 2*region_val_size[1], 0)
    col_high = np.minimum(col_low + 2*region_val_size[1], scribble_mask.shape[1])
    # print(row_low, row_high, col_low, col_high)
    
    validation_mask = np.zeros([scribble_mask.shape[0], scribble_mask.shape[1]])
    validation_mask[row_low:row_high,
                    col_low:col_high] = 1

    return validation_mask.astype(np.uint8)


    # center = np.argmax(val_center)
    # row = np.clip(int(np.floor(center / scribble_mask.shape[1])), val_size[1], scribble_mask.shape[1] - val_size[1])
    # col = np.clip(int(center - row * scribble_mask.shape[1]), val_size[1], scribble_mask.shape[1] - val_size[1])

    # mask = np.zeros(scribble_mask.shape)
    # mask[row - val_size[0]:row + val_size[0], col - val_size[1]:col + val_size[1]] = 1

    # return mask.astype(np.uint8)



"""
    ### validation sample region mask ###
    region_val_size = [int(image.shape[0] * val_perc/2),int(image.shape[1] * val_perc/2)] #validation region
    mask_scribbles = np.sum(scribble,axis = -1)
    mask_scribbles[mask_scribbles>0] = 1
    from scipy import ndimage
    mask_scribbles = ndimage.convolve(mask_scribbles, np.ones([5,5]), mode='constant', cval=0.0)
    #remove borders
#     mask_scribbles[-region_val_size[0]:,:] = 0
#     mask_scribbles[:,-region_val_size[1]:] = 0
#     mask_scribbles[0:region_val_size[0],:] = 0
#     mask_scribbles[:,0:region_val_size[1]] = 0
    val_center = np.random.multinomial(1, mask_scribbles.flatten()/np.sum(mask_scribbles.flatten()), size=1).flatten()
    ix_center = np.argmax(val_center)
    ix_row = int(np.floor(ix_center/image.shape[1]))
    ix_col = int(ix_center - ix_row * image.shape[1])
    print(ix_center,ix_row,ix_col)
    
    row_low = np.maximum(ix_row-region_val_size[0],0)
    row_high = np.minimum(row_low+region_val_size[0],image.shape[0])
    row_low = np.maximum(row_high - 2*region_val_size[0],0)
    row_high = np.minimum(row_low+ 2*region_val_size[0],image.shape[0])
    
    col_low = np.maximum(ix_col-region_val_size[1],0)
    col_high = np.minimum(col_low+region_val_size[1],image.shape[1])
    col_low = np.maximum(col_high - 2*region_val_size[1],0)
    col_high = np.minimum(col_low+2*region_val_size[1],image.shape[1])
    print(row_low,row_high,col_low,col_high)
    
    validation_mask = np.zeros([image.shape[0],image.shape[1]])
    validation_mask[row_low:row_high,
                    col_low:col_high] = 1

"""

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


def compute_entropy(output: np.ndarray):
    """
    Computes the entropy of a probability map.
    Args:
        output (w, h): prediction of the model

    Returns:
        entropy (w, h): entropy of the probability map
    """
    # output = output[task]['class_segmentation'][0, ix_class, ...]
    res = -output * np.log(np.maximum(output, 1e-5))
    res += -(1 - output) * np.log(np.maximum(1 - output, 1e-5))

    return res
