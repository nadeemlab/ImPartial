import logging
from typing import Optional, Union, List

import torch
from PIL import Image
from ignite.metrics import Loss
from monai.data import partition_dataset
from monai.transforms import SpatialCropd
from torch import Tensor
from torch.nn.modules.loss import _Loss
from monai.engines import SupervisedTrainer
from monai.handlers import CheckpointSaver, ROCAUC, IgniteMetric
from monai.inferers import SimpleInferer
from monailabel.interfaces.datastore import Datastore
from monailabel.tasks.train.basic_train import BasicTrainTask, Context

from Impartial.Impartial_functions import compute_impartial_losses
from dataprocessing.utils import rois_to_mask

from general.losses import seglosses, reclosses
from lib.transforms import RandomFlip, ToTensor, GetImpartialOutputs, BlindSpotPatch, PercentileNormalization

import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology

logger = logging.getLogger(__name__)
ENV = "prod"


class Impartial(BasicTrainTask):
    def __init__(
            self,
            model_dir,
            network,
            labels,
            iconfig,
            roi_size=(128, 128),
            max_train_interactions=10,
            max_val_interactions=5,
            description="Interactive deep learning whole-cell segmentation"
                        " and thresholding using partial annotations",
            **kwargs,
    ):
        self._network = network
        self.labels = labels
        self.iconfig = iconfig
        self.roi_size = roi_size
        self.max_train_interactions = max_train_interactions
        self.max_val_interactions = max_val_interactions
        super().__init__(
            model_dir=model_dir,
            description=description,
            train_save_interval=5,
            final_filename="model.pt",
            **kwargs
        )

    def network(self, context: Context):
        return self._network

    def optimizer(self, context: Context):
        return torch.optim.Adam(context.network.parameters(), 0.0001)

    def loss_function(self, context: Context):
        return ImpartialLoss(self.iconfig)

    def compute_weight_map(self, scribble):
        labels, num_labels = ndi.label(scribble)
        centers = ndi.center_of_mass(scribble, labels, range(1, num_labels + 1))

        w = np.zeros(scribble.shape)

        for (i, j) in centers:
            w[int(i), int(j)] = 255

        return ndi.gaussian_filter(w, sigma=10)

    def pre_process(self, request, datastore: Datastore):
        res = datastore.datalist().copy()

        for r in res:
            from_rois_zip = True
            if from_rois_zip:
                scribble = rois_to_mask(r["label"], size=(400, 400))
            else:
                scribble = np.array(Image.open(r["label"])).astype(np.uint8)

            use_ground_truth_labels = False
            if use_ground_truth_labels:
                r["foreground_scribble"] = (scribble / 255).astype(np.uint8)
                r["background_scribble"] = 1 - r["foreground_scribble"]
            else:
                background_scribble = np.zeros(scribble.shape)
                for c in measure.find_contours(scribble):
                    c = c.astype(np.uint32)
                    background_scribble[c[:, 0], c[:, 1]] = 1
                r["background_scribble"] = background_scribble.astype(np.uint8)
                r["foreground_scribble"] = morphology.binary_erosion(scribble, footprint=np.ones((5, 5))).astype(np.uint8)

            labels, num_labels = ndi.label(scribble)
            r["centers"] = ndi.center_of_mass(scribble, labels, range(1, num_labels + 1))

        return res

    def train_pre_transforms(self, context: Context):
        return [
            PercentileNormalization(keys="image"),
            BlindSpotPatch(keys="image"),
            # RandomFlip(keys="image"),
            ToTensor(keys=("input"))
        ]

    def train_post_transforms(self, context: Context):
        return [GetImpartialOutputs(keys="image", iconfig=self.iconfig)]

    def val_inferer(self, context: Context):
        return SimpleInferer()

    def val_additional_metrics(self, context: Context):
        def ot(d):
            return d[0]['pred'].flatten(), (d[0]['label'] > 0).flatten()

        # return {"val_roc_auc": ROCAUC(output_transform=ot), "val_loss": ImpartialLossMetric()}
        return {"val_roc_auc": ROCAUC(output_transform=ot)}

    def partition_datalist(self, context: Context, shuffle=False):
        ds = context.datalist
        val_split = context.request.get("val_split", 0.0)

        def clipped_center(center, rec_size, im_size):
            x_range = (
                max(rec_size[0]/2, center[0] - rec_size[0]/2),
                min(im_size[0] - rec_size[0]/2, center[0] + rec_size[0]/2)
            )
            y_range = (
                max(rec_size[1]/2, center[1] - rec_size[1]/2),
                min(im_size[1] - rec_size[1]/2, center[1] + rec_size[1]/2)
            )

            return np.random.randint(*x_range), np.random.randint(*y_range)

        samples = [dict(center=c, **d) for d in ds for c in d["centers"]]
        train_datalist, val_datalist = partition_dataset(
            data=samples,
            ratios=[1 - val_split, val_split],
            shuffle=True
        )

        def generate_patches(ds):
            patches = []

            keys = [
                "image",
                "background_scribble",
                "foreground_scribble"
            ]
            npatches = 10

            for d in ds:
                image = np.array(Image.open(d["image"])).astype(np.uint8)

                for i in range(npatches):
                    crop = SpatialCropd(
                        keys=keys,
                        roi_size=(128, 128),
                        roi_center=clipped_center(d["center"], rec_size=(128, 128), im_size=image.shape)
                    )

                    patches.append(crop({
                        "image": image[np.newaxis, ...],
                        "background_scribble": d["background_scribble"][np.newaxis, ...],
                        "foreground_scribble": d["foreground_scribble"][np.newaxis, ...],
                    }))

            return patches

        return generate_patches(train_datalist), generate_patches(val_datalist)

    def _create_evaluator(self, context: Context):
        if ENV == "prod":
            return None

        evaluator = super()._create_evaluator(context)
        evaluator.prepare_batch = impartial_prepare_val_batch

        return evaluator

    def train_key_metric(self, context: Context):
        return None

    def _create_trainer(self, context: Context):
        train_handlers: List = self.train_handlers(context)
        if context.local_rank == 0:

            checkpoint_saver = CheckpointSaver(
                save_dir=context.output_dir,
                save_dict={self._model_dict_key: context.network},
                save_interval=self._train_save_interval,
                save_final=True,
                final_filename=self._final_filename,
                save_key_metric=False,
                key_metric_filename=f"train_{self._key_metric_filename}"
                if context.evaluator
                else self._key_metric_filename,
                n_saved=5,
            )

            checkpoint_saver._interval_checkpoint.filename_pattern = "{name}.pt"

            train_handlers.append(checkpoint_saver)

        self._load_checkpoint(context, train_handlers)

        return SupervisedTrainer(
            device=context.device,
            max_epochs=context.max_epochs,
            train_data_loader=self.train_data_loader(context),
            network=context.network,
            optimizer=context.optimizer,
            loss_function=self.loss_function(context),
            prepare_batch=impartial_prepare_batch,
            inferer=self.train_inferer(context),
            amp=self._amp,
            postprocessing=self._validate_transforms(self.train_post_transforms(context), "Training", "post"),
            key_train_metric=self.train_key_metric(context),
            train_handlers=train_handlers,
            iteration_update=self.train_iteration_update(context),
            event_names=self.event_names(context),
        )


class ImpartialLoss(_Loss):
    def __init__(self, config, criterio_seg=None, criterio_rec=None):
        super(ImpartialLoss, self).__init__(reduction="none")
        self.config = config
        self.criterio_seg = criterio_seg or seglosses()
        self.criterio_rec = criterio_rec or reclosses()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        losses = compute_impartial_losses(
            out=input,
            input=target["input"],
            scribble=target["scribble"],
            mask=target["mask"],
            config=self.config,
            criterio_seg=self.criterio_seg,
            criterio_rec=self.criterio_rec
        )

        loss_batch = 0
        for k, w in self.config.weight_objectives.items():
            loss_batch += losses[k] * w

        return loss_batch


def impartial_prepare_batch(batchdata, device: Optional[Union[str, torch.device]] = None,
                            non_blocking: bool = False):
    if not isinstance(batchdata, dict):
        raise AssertionError("impartial_prepare_batch expects dict input data.")
    return (
        batchdata["input"].to(device=device, non_blocking=non_blocking),
        {
            "input": batchdata["input"].to(device=device, non_blocking=non_blocking),
            "scribble": {
                "classes": [batchdata["foreground_scribble"][:, 0, ...].to(device=device, non_blocking=non_blocking), ],
                "background": batchdata["background_scribble"][:, 0, ...].to(device=device, non_blocking=non_blocking),
            },
            "mask": batchdata["mask"].to(device=device, non_blocking=non_blocking)
        }
    )


def impartial_prepare_val_batch(batchdata, device: Optional[Union[str, torch.device]] = None,
                                non_blocking: bool = False):
    if not isinstance(batchdata, dict):
        raise AssertionError("impartial_prepare_val_batch expects dict input data.")
    return (
        batchdata["input"].to(device=device, non_blocking=non_blocking),
        batchdata["label"].to(device=device, non_blocking=non_blocking)
    )


class ImpartialPerformanceMetric(IgniteMetric):
    # TODO: implement a metric that compute all ImPartial metrics
    # computed in evaluation.get_performance()
    pass


class ImpartialLossMetric(Loss):
    def __init__(self):
        super().__init__(loss_fn=ImpartialLoss)
    # TODO overwrite the update() method so that it can compute
    # the loss using ImPartial outputs
