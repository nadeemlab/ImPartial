import os
import logging
from typing import Any, Dict, Optional, Union

import torch
from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer import InferTask
from monailabel.interfaces.tasks.train import TrainTask

from Impartial_classes import ImPartialConfig
from general.networks import UNet
import lib.infers
import lib.trainers

logger = logging.getLogger(__name__)


class BaseImpartialConfig(ImPartialConfig):
    def __init__(self, **kwargs):
        super().__init__(
            config_dic=kwargs
        )


class Vectra2ChConfig(BaseImpartialConfig):
    def __init__(self):
        super().__init__(
            unet_base=64,
            BATCH_SIZE=8,
            n_channels=2,
            classification_tasks={
                '0': {'classes': 1, 'rec_channels': [0], 'ncomponents': [2, 2]},
                '1': {'classes': 2, 'rec_channels': [1], 'ncomponents': [1, 1, 2]}
            }
        )


class Vectra2Ch1taskConfig(BaseImpartialConfig):
    def __init__(self):
        super().__init__(
            unet_base=64,
            BATCH_SIZE=8,
            n_channels=1,
            classification_tasks={
                '0': {'classes': 1, 'rec_channels': [0], 'ncomponents': [2, 2]}
            }
        )


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

        # Impartial config
        self.iconfig = Vectra2Ch1taskConfig()


        # Network
        self.network = UNet(
            n_channels=self.iconfig.n_channels,
            n_classes=self.iconfig.n_output,
            depth=self.iconfig.unet_depth,
            base=self.iconfig.unet_base,
            activation=self.iconfig.activation,
            batchnorm=self.iconfig.batchnorm,
            dropout=self.iconfig.dropout,
            dropout_lastconv=self.iconfig.drop_last_conv,
            p_drop=self.iconfig.p_drop
        )

        # overwrite the last published model with
        # a brand new one
        torch.save(self.network.state_dict(), self.path[1])

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
                "dataset_max_region": (10240, 10240),
                "dataset_limit": 0,
                "dataset_randomize": True,
                "early_stop_patience": self.iconfig.patience,
                "pretrained": False,
                "name": 'train02_Vectra2ch1task'
            },
        )
        return task
