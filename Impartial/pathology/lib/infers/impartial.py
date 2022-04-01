import base64
import logging
import os
import io
from typing import Any, Callable, Dict, Sequence

from PIL import Image
import numpy as np
from roifile import ImagejRoi
from skimage import measure

from dataprocessing.dataloaders import ToTensor
from lib.transforms import GetImpartialOutputs, LoadPNGFile, PercentileNormalization

from monailabel.interfaces.tasks.infer import InferTask, InferType

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
            LoadPNGFile(keys="image"),
            PercentileNormalization(keys="image"),
            ToTensor(keys="image")
        ]

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [GetImpartialOutputs(keys="image", iconfig=self.iconfig)]

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

        img = (data["output"][0, ...] < 0.507).astype(np.uint8)

        for contour in measure.find_contours(img, level=0.9999):
            roi = ImagejRoi.frompoints(np.round(contour)[:, ::-1])
            roi.tofile(output_path)

        return output_path, {"b64_image": pil_to_b64(Image.fromarray(img))}
