import collections
import logging
import random
import sys
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss

from ignite.metrics import Loss
from ignite.metrics.metric import reinit__is_reduced

from monai.engines import SupervisedTrainer
from monai.handlers import CheckpointSaver, IgniteMetricHandler, LrScheduleHandler
from monai.inferers import SimpleInferer
from monai.transforms import EnsureChannelFirstd, RandFlipd, ScaleIntensityRangePercentiles
from monailabel.interfaces.datastore import Datastore
from monailabel.tasks.train.basic_train import BasicTrainTask, Context

from Impartial_dataloaders import sample_patches
from Impartial_functions import compute_impartial_losses
from lib.transforms import BlindSpotPatch, GetImpartialOutputs, DisplayInputs

from general.losses import reclosses, seglosses
from dataprocessing.utils import read_image, rois_to_labels, validation_mask, percentile_normalization

logger = logging.getLogger(__name__)


class Impartial(BasicTrainTask):
    VAL_KEY_METRIC = "val_loss"

    def __init__(
            self,
            model_dir,
            network,
            labels,
            iconfig,
            roi_size=(128, 128), # TODO: (256, 256)
            description="Interactive deep learning whole-cell segmentation and thresholding using partial annotations",
            **kwargs,
    ):
        self._network = network
        self.labels = labels
        self.iconfig = iconfig
        self.roi_size = roi_size
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
        return torch.optim.Adam(context.network.parameters(), 0.0005, weight_decay=0.0001)


    def loss_function(self, context: Context):
        return ImpartialLoss(self.iconfig)

    # def lr_scheduler_handler(self, context: Context):
    #     # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(context.optimizer, mode="min")
    #     # return LrScheduleHandler(lr_scheduler, print_lr=True, step_transform=lambda x: x.state.output[0]["loss"])

    #     lr_scheduler = torch.optim.lr_scheduler.StepLR(context.optimizer, step_size=5, gamma=0.95)
    #     return LrScheduleHandler(lr_scheduler, print_lr=True)

    def pre_process(self, request, datastore: Datastore):
        
        datalist = datastore.datalist().copy()
        # scaler = ScaleIntensityRangePercentiles(lower=1, upper=98, b_min=0, b_max=1, clip=True)

        for d in datalist:
            path = d["image"]
            img = read_image(path=d["image"])
            img = img.astype(np.float32)
            d["image"] = percentile_normalization(img, pmin=1, pmax=98, clip=False)
            d["scribble"] = rois_to_labels(d["label"], size=(d["image"].shape[0], d["image"].shape[1]))
            # TODO: Add code for visualization of image, scribble

        return datalist

    def train_pre_transforms(self, context: Context):
        return [
            BlindSpotPatch(keys="image", input="input", mask="mask"),
            EnsureChannelFirstd(keys=("image", "scribble"), channel_dim=-1),
            # DisplayInputs(iconfig=self.iconfig, output_dir="/tmp/vectra_datalist_input/"),
            RandFlipd(keys=("image", "scribble", "input", "mask"), prob=0.5, spatial_axis=1),
            # DisplayInputs(iconfig=self.iconfig, output_dir="/tmp/vectra_datalist_input/")
        ]

    def train_post_transforms(self, context: Context):
        return [GetImpartialOutputs(iconfig=self.iconfig)]

    def val_inferer(self, context: Context):
        return SimpleInferer()

    def partition_datalist(self, context: Context, shuffle=False):
        datalist = context.datalist

        # val_split = context.request.get("val_split", 0.4)        
        val_split = 0.4     #was being overwritten from monai label earlier to 0.2

        images = [d["image"] for d in datalist]
        scribbles = [d["scribble"] for d in datalist]
        
        validation_masks = [
            validation_mask(
                scribble=np.sum(s, 2),
                val_split=val_split
            )
            for s in scribbles
        ]

        # TODO: Can write code to plot val masks here

        logger.info(f"iconfig.npatches_epoch:: {self.iconfig.npatches_epoch}")
        npatches_epoch = context.request["npatches_epoch"]
        nval_patches = int(val_split * npatches_epoch)
        ntrain_patches = npatches_epoch - nval_patches

        logger.info(f"partition_datalist :: npatches_epoch: {npatches_epoch}")
        logger.info(f"partition_datalist :: ntrain_patches: {ntrain_patches}")
        logger.info(f"partition_datalist :: nval_patches: {nval_patches}")

        train_patches = sample_patches(
            images=images,
            scribbles=scribbles,
            fov_masks=validation_masks,
            validation=False,
            p_scribble_crop=self.iconfig.p_scribble_crop,
            patch_size=self.iconfig.patch_size,
            shift_crop=self.iconfig.shift_crop,
            npatches_total=ntrain_patches
        )

        val_patches = sample_patches(
            images=images,
            scribbles=scribbles,
            fov_masks=validation_masks,
            validation=True,
            p_scribble_crop=self.iconfig.p_scribble_crop,
            patch_size=self.iconfig.patch_size,
            shift_crop=self.iconfig.shift_crop,
            npatches_total=nval_patches
        )

        def to_dict(ds):
            return [{"image": d[0], "scribble": d[1]} for d in ds]

        train_ds = to_dict(train_patches)
        val_ds = to_dict(val_patches)

        logger.info(f"End:: Partition dataset {len(train_ds)} + {len(val_ds)}")
        random.shuffle(train_ds)
        random.shuffle(val_ds)
        return train_ds, val_ds

    def _create_evaluator(self, context: Context):
        evaluator = super()._create_evaluator(context)
        evaluator.prepare_batch = impartial_prepare_batch

        return evaluator

    def train_key_metric(self, context: Context):
        return None

    def val_key_metric(self, context):
        return {"val_loss": ImpartialLossMetric(self.iconfig)}

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

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        losses = compute_impartial_losses(
            out=output,
            input=target["image"],
            scribble=target["scribble"],
            mask=target["mask"],
            config=self.config,
            criterio_seg=self.criterio_seg,
            criterio_rec=self.criterio_rec
        )
        # print("ImpartialLoss:  losses: keys(): ", losses.keys())

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
            "image": batchdata["image"].to(device=device, non_blocking=non_blocking),
            "scribble": {
                "classes": [batchdata["scribble"][:, 0, ...].to(device=device, non_blocking=non_blocking), ],
                "background": batchdata["scribble"][:, -1, ...].to(device=device, non_blocking=non_blocking),
            },
            "mask": batchdata["mask"].to(device=device, non_blocking=non_blocking)
        }
    )


#class ImpartialPerformanceMetric(IgniteMetric):
    # TODO: implement a metric that compute all ImPartial metrics
    # computed in evaluation.get_performance()
#    pass


class ImpartialLossMetric(Loss):
    def __init__(self, iconfig):
        super().__init__(loss_fn=ImpartialLoss(config=iconfig))

    @reinit__is_reduced
    def update(self, output: Sequence[Union[torch.Tensor, Dict]]) -> None:
        image = torch.stack([o["label"]["image"] for o in output], 0)
        mask = torch.stack([o["label"]["mask"] for o in output], 0)

        out = collections.defaultdict(list)
        for o in output:
            for i, c in enumerate(o["label"]["scribble"]["classes"]):
                out[i].append(c)
        scribble = {
            "classes": [torch.stack(c) for c in out.values()],
            "background": torch.stack([o["label"]["scribble"]["background"] for o in output], 0)
        }

        y_pred = torch.stack([o["pred"] for o in output], 0)
        y = {
            "image": image,
            "mask": mask,
            "scribble": scribble
        }

        average_loss = -self._loss_fn(y_pred, y).detach()

        if len(average_loss.shape) != 0:
            raise ValueError("loss_fn did not return the average loss.")

        n = self._batch_size(y_pred)
        self._sum += average_loss.to(self._device) * n
        self._num_examples += n
