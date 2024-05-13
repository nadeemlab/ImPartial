import base64
import io
import logging
import os
import json
import numpy as np
import torch
from typing import Any, Callable, Dict, List, Sequence, Union
from PIL import Image
from roifile import ImagejRoi
from skimage import measure

from dataprocessing.utils import read_image, read_label
from general.evaluation import get_performance
from lib.transforms import GetImpartialOutputsFinal, AggregateComponentOutputs, ComputeEntropy, GetImpartialOutputs

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
from monai.inferers import Inferer

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
        # self.inferer = MCDropoutInferer(iconfig)


    def inferer(self, data=None) -> Inferer:
        return MCDropoutInferer(self.iconfig)
    
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
            EnsureChannelFirstd(keys="image", channel_dim=-1)
        ]

    # def post_transforms(self, data=None) -> Sequence[Callable]:
    #     return [
    #         GetImpartialOutputs(iconfig=self.iconfig),
    #         Activationsd(keys="output", softmax=True),
    #         AggregateComponentOutputs(keys="output", iconfig=self.iconfig),
    #         ComputeEntropy(),
    #         # AsDiscreted(keys="output", threshold=data.get("threshold", 0.5)),
    #         ToNumpyd(keys=("output", "entropy"))
    #     ]

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            GetImpartialOutputsFinal(iconfig=self.iconfig),
            ToNumpyd(keys=("output", "entropy"))
        ]

    def writer(self, data, extension=None, dtype=None):
        writer = ZIPFileWriter(
            label=self.output_label_key,
            json=self.output_json_key,
            threshold=data.get("threshold", 0.5)
        )
        return writer(data)


def to_np(t):
    return t.cpu().detach().numpy()

class MCDropoutInferer(Inferer):
    """
    MCDropoutInferer is the normal inference method that run model forward() directly.
    Usage example can be found in the :py:class:`monai.inferers.Inferer` base class.

    """

    def __init__(self, iconfig) -> None:
        Inferer.__init__(self)
        self.iconfig = iconfig
        print("MCDropoutInferer intialized ... ")

    def __call__(
        self, inputs: torch.Tensor, network: Callable[..., torch.Tensor], *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        
        # assume batch_size = 1 
        # 40, 12, h, w
        
        print("inputs size: ", inputs.size())
        print("Calling MCDropoutInferer ............... ############# ..............")
        predictions = np.empty((0, self.iconfig.n_output, inputs.size(-2), inputs.size(-1)))
        network.enable_dropout()
        print(' ....running mcdrop iterations: ', self.iconfig.MCdrop_it)
        # start_mcdropout_time = time.time()
        for it in range(self.iconfig.MCdrop_it):
            with torch.no_grad():
                out = to_np(network(inputs, *args, **kwargs))
                out = out[0] # assume batch_isze = 1 
                print("GS:: mc dropout out shape: ", out.shape)
                predictions = np.vstack((predictions, out[np.newaxis,...]))

        print("mc dropout predictions size: ", predictions.shape)

        out = predictions[np.newaxis, ...]
        print("mc drop out out shape : ", out.shape)
        out = torch.from_numpy(out)

        # # return predictions
        # out = network(inputs, *args, **kwargs)
        # print("single out shape : ", out.shape)

        # out = out[np.newaxis, ...]
        # print("single out shape : ", out.shape)

        return out
    
        
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

        if os.path.exists(label_path):
            label_gt = read_label(label_path, (data["image"].shape[1], data["image"].shape[2]))
            label_gt = label_gt.astype(int)

            metrics = get_performance(label_gt=label_gt, y_pred=prob_map, threshold=0.5, iou_threshold=0.5)

        output_path = os.path.join(output_dir, f"{os.path.splitext(input_file)[0]}.json")

        res = {"output": prob_map.tolist(), "entropy": data["entropy"].tolist(), "metrics": metrics}

        with open(output_path, 'w') as fp:
            json.dump(res, fp)
        
        # return output_path, {"output": prob_map.tolist(), "entropy": data["entropy"].tolist(), "metrics": metrics}
        # return output_path, {"output": prob_map.tolist(), "metrics": metrics}
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
