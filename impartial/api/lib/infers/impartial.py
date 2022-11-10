import base64
import io
import logging
import os
from typing import Any, Callable, Dict, List, Sequence, Union

import numpy as np
from dataprocessing.utils import read_image
from lib.transforms import AggregateComponentOutputs, ComputeEntropy, GetImpartialOutputs
from monai.config import PathLike
from monai.data import ImageReader
from monai.data.image_reader import _copy_compatible_dict, _stack_images
from monai.transforms import (
    Activationsd,
    AsChannelFirstd,
    LoadImaged,
    ScaleIntensityRangePercentilesd,
    ToNumpyd,
    ToTensord,
)
from PIL import Image
from roifile import ImagejRoi
from skimage import measure

from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.basic_infer import BasicInferTask

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
            ScaleIntensityRangePercentilesd(
                keys="image",
                lower=1,
                upper=98,
                b_min=0,
                b_max=1,
                clip=True
            ),
            ToTensord(keys="image"),
            AsChannelFirstd(keys="image")
        ]

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            GetImpartialOutputs(iconfig=self.iconfig),
            Activationsd(keys="output", softmax=True),
            AggregateComponentOutputs(keys="output", iconfig=self.iconfig),
            ComputeEntropy(),
            # AsDiscreted(keys="output", threshold=data.get("threshold", 0.5)),
            ToNumpyd(keys=("output", "entropy"))
        ]

    def writer(self, data, extension=None, dtype=None):
        writer = ZIPFileWriter(
            label=self.output_label_key,
            json=self.output_json_key,
            threshold=data.get("threshold", 0.5)
        )
        return writer(data)


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
        output_dir = os.path.join(base_dir, "outputs")
        
        os.makedirs(output_dir, exist_ok=True)

        entropy = -data["output"] * np.log(np.maximum(data["output"], 1e-5))
        entropy += -(1-data["output"]) * np.log(np.maximum(1-data["output"], 1e-5))

        prob_map = data["output"]

        output_path = os.path.join(output_dir, f"{os.path.splitext(input_file)[0]}.zip")
        for contour in measure.find_contours((prob_map > self.threshold).astype(np.uint8), level=0.9999):
            roi = ImagejRoi.frompoints(np.round(contour)[:, ::-1])
            roi.tofile(output_path)

        return output_path, {"output": prob_map.tolist(), "entropy": data["entropy"].tolist()}


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
        return read_image(path)

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
