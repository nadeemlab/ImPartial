import base64
import io
import logging
import os
import json
import sys
import skimage 
import numpy as np

from typing import Any, Callable, Dict, List, Sequence, Union
from scipy import ndimage
from PIL import Image
from roifile import ImagejRoi
from skimage import measure

from lib.transforms import AggregateComponentOutputs, ComputeEntropy, GetImpartialOutputs, DisplayOuputs, DisplayPredictions
from monai.config import PathLike
from monai.data import ImageReader
from monai.data.image_reader import _copy_compatible_dict, _stack_images
from monai.transforms import (
    Activationsd,
    EnsureChannelFirstd,
    LoadImaged,
    ScaleIntensityRangePercentilesd,
    ToNumpyd,
    ToTensord,
)

from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.basic_infer import BasicInferTask

from general.evaluation import get_performance
from dataprocessing.utils import read_image, read_label, percentile_normalization
from general.outlines import dilate_masks, masks_to_outlines


logger = logging.getLogger(__name__)


class Impartial(BasicInferTask):
    """
    This provides Inference Engine for pre-trained Impartial model.
    """
    def __init__(
        self,
        path,
        iconfig,
        network=None,
        roi_size=(400, 400),
        type=InferType.SEGMENTATION,
        labels=None,
        dimension=2,
        label_colors=None,
        description="A pre-trained ImPartial segmentation model",
    ):
        self.label_colors = label_colors
        self.iconfig = iconfig
        super().__init__(
            path=path,
            network=network,
            roi_size=roi_size,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            config={"label_colors": label_colors},
            input_key="image"
        )

    def info(self) -> Dict[str, Any]:
        d = super().info()
        d["pathology"] = True
        return d

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        return [
            LoadImaged(keys="image", reader=ImpartialImageReader),
            # Note: do normalization duing image loading
            # ScaleIntensityRangePercentilesd(keys="image", lower=1, upper=98, b_min=0, b_max=1, clip=True),
            ToTensord(keys="image"),
            EnsureChannelFirstd(keys="image", channel_dim=-1)
        ]

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            GetImpartialOutputs(iconfig=self.iconfig),
            #DisplayOuputs(iconfig=self.iconfig, output_dir="/tmp/vectra_datalist_out_2024-08-10"),
            # Activationsd(keys="output", softmax=True),
            AggregateComponentOutputs(keys="output", iconfig=self.iconfig),
            ComputeEntropy(),
            # AsDiscreted(keys="output", threshold=data.get("threshold", 0.5)),
            ToNumpyd(keys=("output", "entropy")),
            #DisplayPredictions(iconfig=self.iconfig, output_dir="/tmp/vectra_datalist_2024-08-10"),
        ]

    def writer(self, data, extension=None, dtype=None):
        writer = ZIPFileWriter(
            label=self.output_label_key,
            json=self.output_json_key,
            threshold=data.get("threshold", 0.5)
        )
        return writer(data)


def get_outlines(prob_map, threshold=0.9):
        
    # use prob_map to convert into outline
            # threshold & save 
 
    # threshold = 0.70
    out_mask = (prob_map > threshold).astype(np.uint8) * 255
    out_mask = Image.fromarray(out_mask)
    # out_mask.save(png_mask_path)
    
    labels_pred, _ = ndimage.label(out_mask)
    labels_pred = skimage.morphology.remove_small_objects(labels_pred, min_size=5)
    labels_pred = dilate_masks(labels_pred, n_iter=1)
    print("labels_pred: ", labels_pred.shape, labels_pred.min(), labels_pred.max())
    mask_pred = (labels_pred > 0) * 1.0
    print("mask_pred: ", mask_pred.shape, mask_pred.min(), mask_pred.max())
    
    # labels_pred_img = Image.fromarray(labels_pred)
    # labels_pred_img.save("/home/ubuntu/git-2024-03-18/labels_pred_img.png")
    
    # mask_pred_img = Image.fromarray((mask_pred * 255).astype(np.uint8))
    # mask_pred_img.save("/home/ubuntu/git-2024-03-18/mask_pred.png")
    
    # mask_outlines = masks_to_outlines(labels_pred)

    # noutX, noutY = np.nonzero(mask_outlines)
    # mask_outlines_img = np.zeros((out_mask.height, out_mask.width, 3), dtype=np.uint8)
    # mask_outlines_img[noutX, noutY] = np.array([255, 0, 0])
    # mask_outlines_img = Image.fromarray(mask_outlines_img)
    # mask_outlines_img.save("/home/ubuntu/git-2024-03-18/test_outline.png")
    
    return mask_pred


def pil_to_b64(i):
    buff = io.BytesIO()
    i.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")


def np_to_b64(i):
    return pil_to_b64(Image.fromarray((i * 255).astype(np.uint8)))


class ZIPFileWriter:
    def __init__(self, label, json, threshold):
        self.label = label
        self.json = json
        self.threshold = threshold

    def __call__(self, data):

        base_dir, input_file = os.path.split(data["image_path"])
        output_dir = os.path.join(base_dir, "outputs/final/")
        
        os.makedirs(output_dir, exist_ok=True)

        prob_map = data["output"]

        label_path = os.path.join(base_dir, "labels/final/", f"{os.path.splitext(input_file)[0]}.zip")
        print("Infers/impartial.py, label_path::", label_path)
        metrics = {}

        # if os.path.exists(label_path):
        #     label_gt = read_label(label_path, (data["image"].shape[1], data["image"].shape[2]))
        #     label_gt = label_gt.astype(int)

        #     metrics = get_performance(label_gt=label_gt, y_pred=prob_map, threshold=0.5, iou_threshold=0.5)

        output_path = os.path.join(output_dir, f"{os.path.splitext(input_file)[0]}.json")
        dilated_mask = get_outlines(prob_map, threshold=0.95)
        # res = {"output": prob_map.tolist(), "entropy": data["entropy"].tolist(), "metrics": metrics}
        res = {"output": dilated_mask.tolist(), "entropy": data["entropy"].tolist(), "metrics": metrics}

        # # TODO: Delete laterx
        # output_path2 = os.path.join(output_dir, f"{os.path.splitext(input_file)[0]}_test.json")
        # roi_zip_path = os.path.join(output_dir, f"{os.path.splitext(input_file)[0]}_roi.zip")

        # with open(output_path2, 'w') as fp:
        #     json.dump(res, fp)
        
        # threshold = 0.95
        
        # for contour in measure.find_contours((prob_map > threshold).astype(np.uint8), level=0.9999):
        #     roi = ImagejRoi.frompoints(np.round(contour)[:, ::-1])
        #     roi.tofile(roi_zip_path)
        # # TODO: Delete later ^^


        with open(output_path, 'w') as fp:
            json.dump(res, fp)

        return output_path, {} 

class ImpartialImageReader(ImageReader):
    def __init__(self):
        super().__init__()

    def verify_suffix(self, filename):
        extension = filename.split(".")[-1]

        if extension == "tiff" or extension == "tif" or extension == "png" or extension == "PNG" :
            return True
        else:
            return False

    def read(self, data: Union[Sequence[PathLike], PathLike], **kwargs):

        path = data[0]
        img = read_image(path)
        img = percentile_normalization(img, pmin=1, pmax=98, clip=False)
        return img

    def get_data(self, img):
        
        img_array: List[np.ndarray] = []
        compatible_meta: Dict = {}

        img_array.append(img)

        header = self._get_meta_dict(img)
        header["spatial_shape"] = self._get_spatial_shape(img)
        header["original_channel_dim"] = "no_channel" if len(img.shape) == len(header["spatial_shape"]) else -1
        _copy_compatible_dict(header, compatible_meta)

        return _stack_images(img_array, compatible_meta), compatible_meta

    def _get_meta_dict(self, img) -> Dict:
        """
        Get the all the meta data of the image and convert to dict type.
        Args:
            img: a PIL Image object loaded from an image file.

        """
        return {"format": 'tiff', "mode": '64', "width": img.shape[0], "height": img.shape[1]}

    def _get_spatial_shape(self, img):
        """
        Get the spatial shape of image data, it doesn't contain the channel dim.
        Args:
            img: a PIL Image object loaded from an image file.
        """
        print("INFER::_get_spatial_shape::", img.shape)
        return np.asarray((img.shape[0], img.shape[1]))
