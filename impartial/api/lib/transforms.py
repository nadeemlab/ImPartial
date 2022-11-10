import copy
import logging
from typing import Tuple

import numpy as np
import torch
from dataprocessing.dataloaders import compute_probability_map, random_crop, blind_spot_patch
from dataprocessing.utils import compute_entropy
from impartial.Impartial_classes import ImPartialConfig
from monai.config import KeysCollection
from monai.transforms import MapTransform, Randomizable, Transform

logger = logging.getLogger(__name__)


class PercentileNormalization(MapTransform):
    def __init__(
            self,
            keys: KeysCollection,
            pmin: int = 1,
            pmax: int = 98,
            clip: bool = False,
    ):
        super().__init__(keys)
        self.pmin = pmin
        self.pmax = pmax
        self.clip = clip

    def __call__(self, data):
        res = copy.deepcopy(data)

        # Normalize the image using percentiles
        lo, hi = np.percentile(data["image"], (self.pmin, self.pmax))
        image_norm_percentile = (data["image"] - lo) / (hi - lo)

        if self.clip:
            image_norm_percentile[image_norm_percentile > 1] = 1
            image_norm_percentile[image_norm_percentile < 0] = 0

        res["image"] = image_norm_percentile

        return res


class SampleImage(MapTransform):
    def __init__(
            self,
            keys: KeysCollection,
            validation: bool,
            p_scribble_crop,
            npatch_image,
            patch_size,
            shift_crop
    ):
        super().__init__(keys)
        self.validation = validation
        self.p_scribble_crop = p_scribble_crop
        self.npatch_image = npatch_image
        self.patch_size = patch_size
        self.shift_crop = shift_crop

    def __call__(self, data):
        res = copy.deepcopy(data)

        image = data["image"]
        scribbles = data["label"]["scribbles"]
        mask = data["label"]["val_mask"]

        if len(image.shape) <= 2:
            image = image[..., np.newaxis]

        if not self.validation:
            mask = 1 - mask

        res["image_patches"], res["scribble_patches"] = random_crop(
            X=image,
            S=scribbles,
            probability_mask=mask * compute_probability_map(scribbles, self.p_scribble_crop),
            patch_size=self.patch_size,
            npatch_image=self.npatch_image,
            shift_crop=self.shift_crop
        )

        return res


class BlindSpotPatch(Randomizable, MapTransform):
    def __init__(
        self, keys: KeysCollection, ratio: float = 0.95, size_window: Tuple = (10, 10), input="input", mask="mask"
    ):
        super().__init__(keys)
        self.ratio = ratio
        self.size_window = size_window
        self.input = input
        self.mask = mask

    def __call__(self, data):
        d = dict(data)

        for key in self.key_iterator(d):
            input, mask = blind_spot_patch(d[key], ratio=self.ratio, size_window=self.size_window)

            input = torch.moveaxis(input, -1, 0) if isinstance(input, torch.Tensor) else np.moveaxis(input, -1, 0)
            mask = torch.moveaxis(mask, -1, 0) if isinstance(mask, torch.Tensor) else np.moveaxis(mask, -1, 0)

            d[self.input] = input
            d[self.mask] = mask
        return d


class GetImpartialOutputs(Transform):
    def __init__(self, iconfig: ImPartialConfig):
        self.iconfig = iconfig

    def __call__(self, data):
        d = dict(data)

        # tasks = self.iconfig.classification_tasks
        # d["output"] = outputs_by_task(tasks, torch.unsqueeze(d["pred"], 0))
        d["output"] = d["pred"][:4, ...]

        return d


class AggregateComponentOutputs(MapTransform):
    def __init__(
            self, keys: KeysCollection, iconfig: ImPartialConfig
    ):
        self.iconfig = iconfig
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)

        # assume 1 task
        assert len(self.iconfig.classification_tasks) == 1
        task = self.iconfig.classification_tasks["0"]

        # assume 1 class (+ background) components
        assert len(self.iconfig.classification_tasks["0"]["ncomponents"]) == 2
        step = task["ncomponents"][0]

        for key in self.key_iterator(d):
            d[key] = torch.sum(d[key][:step, ...], dim=0)
        return d


class ComputeEntropy(Transform):
    def __call__(self, data):
        d = dict(data)

        d["entropy"] = compute_entropy(data["output"].cpu())

        return d

