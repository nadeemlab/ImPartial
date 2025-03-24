import os
import cv2
import numpy as np
from roifile import ImagejRoi
from roifile import ImagejRoi, roiwrite
from scipy.ndimage import find_objects

def masks_to_outlines(masks):
    """Get outlines of masks as a 0-1 array.

    Args:
        masks (int, 2D or 3D array): Size [Ly x Lx] or [Lz x Ly x Lx], where 0=NO masks and 1,2,...=mask labels.

    Returns:
        outlines (2D or 3D array): Size [Ly x Lx] or [Lz x Ly x Lx], where True pixels are outlines.
    """
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError("masks_to_outlines takes 2D or 3D array, not %dD array" %
                         masks.ndim)
    outlines = np.zeros(masks.shape, bool)

    if masks.ndim == 3:
        for i in range(masks.shape[0]):
            outlines[i] = masks_to_outlines(masks[i])
        return outlines
    else:
        slices = find_objects(masks.astype(int))
        for i, si in enumerate(slices):
            if si is not None:
                sr, sc = si
                mask = (masks[sr, sc] == (i + 1)).astype(np.uint8)
                contours = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
                pvc, pvr = np.concatenate(contours[-2], axis=0).squeeze().T
                vr, vc = pvr + sr.start, pvc + sc.start
                outlines[vr, vc] = 1
        return outlines


def dilate_masks(masks, n_iter=5):
    """Dilate masks by n_iter pixels.

    Args:
        masks (ndarray): Array of masks.
        n_iter (int, optional): Number of pixels to dilate the masks. Defaults to 5.

    Returns:
        ndarray: Dilated masks.
    """
    dilated_masks = masks.copy()
    for n in range(n_iter):
        # define the structuring element to use for dilation
        kernel = np.ones((3, 3), "uint8")
        # find the distance to each mask (distances are zero within masks)
        dist_transform = cv2.distanceTransform((dilated_masks == 0).astype("uint8"),
                                               cv2.DIST_L2, 5)
        # dilate each mask and assign to it the pixels along the border of the mask
        # (does not allow dilation into other masks since dist_transform is zero there)
        for i in range(1, np.max(masks) + 1):
            mask = (dilated_masks == i).astype("uint8")
            dilated_mask = cv2.dilate(mask, kernel, iterations=1)
            dilated_mask = np.logical_and(dist_transform < 2, dilated_mask)
            dilated_masks[dilated_mask > 0] = i
    return dilated_masks


def outlines_list_single(masks):
    """Get outlines of masks as a list to loop over for plotting.

    Args:
        masks (ndarray): masks (0=no cells, 1=first cell, 2=second cell,...)

    Returns:
        list: List of outlines as pixel coordinates.

    """
    outpix = []
    for n in np.unique(masks)[1:]:
        mn = masks == n
        if mn.sum() > 0:
            contours = cv2.findContours(mn.astype(np.uint8), mode=cv2.RETR_EXTERNAL,
                                        method=cv2.CHAIN_APPROX_NONE)
            contours = contours[-2]
            cmax = np.argmax([c.shape[0] for c in contours])
            pix = contours[cmax].astype(int).squeeze()
            if len(pix) > 4:
                outpix.append(pix)
            else:
                outpix.append(np.zeros((0, 2)))
    return outpix



def save_rois(masks, file_name):
    """ save masks to .roi files in .zip archive for ImageJ/Fiji

    Args:
        masks (np.ndarray): masks output from Cellpose.eval, where 0=NO masks; 1,2,...=mask labels
        file_name (str): name to save the .zip file to
    
    Returns:
        None
    """
    outlines = outlines_list_single(masks)
    nonempty_outlines = [outline for outline in outlines if len(outline)!=0]
    if len(outlines)!=len(nonempty_outlines):
        print(f"empty outlines found, saving {len(nonempty_outlines)} ImageJ ROIs to .zip archive.")
    rois = [ImagejRoi.frompoints(outline) for outline in nonempty_outlines]
    # file_name = os.path.splitext(file_name)[0] + '_rois.zip'


    # Delete file if it exists; the roifile lib appends to existing zip files.
    # If the user removed a mask it will still be in the zip file
    if os.path.exists(file_name):
        os.remove(file_name)

    roiwrite(file_name, rois)