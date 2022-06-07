import os
import io
import base64
import logging
from typing import Any, Callable, Dict, Sequence, List

import numpy as np
from PIL import Image
from skimage import measure
from roifile import ImagejRoi

from monai.data import PILReader
from monai.data.image_reader import _copy_compatible_dict, _stack_images
from monai.transforms import (
    LoadImaged,
    ScaleIntensityRangePercentilesd,
    ToTensord,
    AddChanneld,
    Activationsd,
    AsDiscreted,
    ToNumpyd
)
from monai.utils import ensure_tuple
from monailabel.interfaces.tasks.infer import InferTask, InferType

from lib.transforms import GetImpartialOutputs, AddForegroundOutput


logger = logging.getLogger(__name__)


class Impartial(InferTask):
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
            LoadImaged(keys="image", reader=PNGReader),
            ScaleIntensityRangePercentilesd(
                keys="image",
                lower=1,
                upper=98,
                b_min=0,
                b_max=1,
                clip=True
            ),
            ToTensord(keys="image"),
            AddChanneld(keys="image")
        ]

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            GetImpartialOutputs(keys="pred", iconfig=self.iconfig),
            Activationsd(keys="output", softmax=True),
            AddForegroundOutput(keys="output", iconfig=self.iconfig),
            AsDiscreted(keys="output", threshold=0.5),
            ToNumpyd(keys="output")
        ]

    def writer(self, data, extension=None, dtype=None):
        writer = PNGWriter(label=self.output_label_key, json=self.output_json_key)
        return writer(data)


def pil_to_b64(i):
    buff = io.BytesIO()
    i.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")


class PNGWriter:
    def __init__(self, label, json):
        self.label = label
        self.json = json

    def __call__(self, data):
        base_dir, input_file = os.path.split(data["image_path"])
        output_dir = os.path.join(base_dir, "outputs")

        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f"{os.path.splitext(input_file)[0]}.zip")

        img = (data["output"] * 255).astype(np.uint8)

        for contour in measure.find_contours(img, level=0.9999):
            roi = ImagejRoi.frompoints(np.round(contour)[:, ::-1])
            roi.tofile(output_path)

        return output_path, {"b64_image": pil_to_b64(Image.fromarray(img))}


class PNGReader(PILReader):
    def get_data(self, img):
        """
        Extract data array and meta data from loaded image and return them.
        This function returns two objects, first is numpy array of image data, second is dict of meta data.
        It computes `spatial_shape` and stores it in meta dict.
        When loading a list of files, they are stacked together at a new dimension as the first dimension,
        and the meta data of the first image is used to represent the output meta data.

        Args:
            img: a PIL Image object loaded from a file or a list of PIL Image objects.

        """
        img_array: List[np.ndarray] = []
        compatible_meta: Dict = {}

        for i in ensure_tuple(img):
            header = self._get_meta_dict(i)
            header["spatial_shape"] = self._get_spatial_shape(i)
            data = np.asarray(i)
            img_array.append(data)
            header["original_channel_dim"] = "no_channel" if len(data.shape) == len(header["spatial_shape"]) else -1
            _copy_compatible_dict(header, compatible_meta)

        return _stack_images(img_array, compatible_meta), compatible_meta