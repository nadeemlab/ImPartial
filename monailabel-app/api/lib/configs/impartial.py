import logging
import os
import sys
from typing import Any, Dict, Optional, Union

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.train import TrainTask

import lib.infers
import lib.trainers

from . import Config_CH1, Config_CH2, Config_CH3

sys.path.append("../")
from impartial.general.networks import UNet

logger = logging.getLogger(__name__)


class Impartial(TaskConfig):
    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)

        # Labels
        self.labels = ["Nuclei"]
        self.label_colors = {"Nuclei": (0, 255, 255)}

        # Model Files
        self.path = [
            os.path.join(self.model_dir, f"pretrained_{name}.pt"),  # pretrained
            os.path.join(self.model_dir, f"{name}.pt"),  # published
        ]

        # ImPartial config
        nChannels = name.split('_')[1]
        # nChannels = conf.get("nChannels", "1")
        if nChannels == "1":
            self.iconfig = Config_CH1()
        if nChannels == "2":
            self.iconfig = Config_CH2()
        if nChannels == "3":
            self.iconfig = Config_CH3()

        # Network
        self.network = UNet(
            n_channels=self.iconfig.n_channels,
            n_classes=self.iconfig.n_output,
            depth=self.iconfig.unet_depth,
            base=self.iconfig.unet_base,
            activation=self.iconfig.activation,
            batchnorm=self.iconfig.batchnorm,
            # dropout=self.iconfig.drop_encoder_decoder,
            dropout=True,
            dropout_lastconv=self.iconfig.drop_last_conv,
            p_drop=self.iconfig.p_drop
        )

        # overwrite the last published model with
        # a brand new one
        # torch.save(self.network.state_dict(), self.path[1])

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        task: InferTask = lib.infers.Impartial(
            path=self.path,
            iconfig=self.iconfig,
            network=self.network,
            labels=self.labels,
            label_colors=self.label_colors
        )
        return task

    def trainer(self) -> Optional[TrainTask]:
        output_dir = os.path.join(self.model_dir, self.name)
        task: TrainTask = lib.trainers.Impartial(
            model_dir=output_dir,
            network=self.network,
            load_path=self.path[0],
            publish_path=self.path[1],
            labels=self.labels,
            iconfig=self.iconfig,
            description="Train ImPartial model",
            config={
                "max_epochs": self.iconfig.EPOCHS,
                "train_batch_size": self.iconfig.BATCH_SIZE,
                "val_batch_size": self.iconfig.BATCH_SIZE,
                "dataset_max_region": (10240, 10240),
                "npatches_epoch": self.iconfig.npatches_epoch,
                "dataset_limit": 0,
                "dataset_randomize": True,
                "early_stop_patience": self.iconfig.patience,
                "pretrained": True,
                "name": type(self.iconfig).__name__.lower()
            },
        )
        return task
