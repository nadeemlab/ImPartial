import logging
import os
from distutils.util import strtobool
from typing import Dict

import lib.configs

from monailabel.config import settings
from monailabel.datastore.local import LocalDatastore
from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.datastore import Datastore
from monailabel.interfaces.tasks.infer import InferTask
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.utils.others.class_utils import get_class_names

logger = logging.getLogger(__name__)


class Impartial(MONAILabelApp):
    def __init__(self, app_dir, studies, conf):
        self.model_dir = os.path.join(app_dir, "model")

        configs = {}
        for c in get_class_names(lib.configs, "TaskConfig"):
            name = c.split(".")[-2].lower()
            configs[name] = c
        
        configs = {k: v for k, v in sorted(configs.items())}

        models = conf.get("models", "impartial_1,impartial_2,impartial_3")
        # models = conf.get("models")
        if not models:
            print("")
            print("---------------------------------------------------------------------------------------")
            print("Provide --conf models <name>")
            print("Following are the available models.  You can pass comma (,) seperated names to pass multiple")
            print(f"    all, {', '.join(configs.keys())}")
            print("---------------------------------------------------------------------------------------")
            print("")
            exit(-1)

        models = models.split(",")
        models = [m.strip() for m in models]
        
        self.models: Dict[str, TaskConfig] = {}
        for k in models:
            if self.models.get(k):
                continue
            else:
                model_type = k.split('_')[0]
                if model_type in configs:
                    v = configs[model_type]
                    logger.info(f"+++ Adding Model: {model_type} type: {k} => {v}")
                    self.models[k] = eval(f"{v}()")
                    self.models[k].init(k, self.model_dir, conf, None)

        logger.info(f"+++ Using Models: {list(self.models.keys())}")

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            name="ImPartial",
            description="Interactive deep-learning whole-cell segmentation "
                        "and thresholding using partial annotations",
        )

    def init_datastore(self) -> Datastore:
        logger.info(f"Init Datastore for: {self.studies}")

        return LocalDatastore(
            self.studies,
            extensions=['*.tif', '*.png'],
            auto_reload=settings.MONAI_LABEL_DATASTORE_AUTO_RELOAD,
        )

    def init_infers(self) -> Dict[str, InferTask]:
        infers: Dict[str, InferTask] = {}
        #################################################
        # Models
        #################################################
        for n, task_config in self.models.items():
            c = task_config.infer()
            c = c if isinstance(c, dict) else {n: c}
            for k, v in c.items():
                logger.info(f"+++ Adding Inferer:: {k} => {v}")
                infers[k] = v
        return infers

    def init_trainers(self) -> Dict[str, TrainTask]:
        trainers: Dict[str, TrainTask] = {}
        if strtobool(self.conf.get("skip_trainers", "false")):
            return trainers

        for n, task_config in self.models.items():
            t = task_config.trainer()
            if not t:
                continue

            logger.info(f"+++ Adding Trainer:: {n} => {t}")
            trainers[n] = t
        return trainers



"""
Example to run train/infer/scoring task(s) locally without actually running MONAI Label Server
"""

def main():
    os.putenv("MASTER_ADDR", "127.0.0.1")
    os.putenv("MASTER_PORT", "1234")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(process)s] [%(threadName)s] [%(levelname)s] (%(name)s:%(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    app_dir = os.path.dirname(__file__)
    studies = os.path.join(app_dir, "..", "..", "Data", "Vectra_WC_2CH_tiff")

    conf = {"models": "impartial_1,impartial_2,impartial_3"}
    app = Impartial(app_dir, studies, conf)

    # Train
    app.train(
        request={
            "model": "impartial_3",
            "max_epochs": 100,
            # "dataset": "CacheDataset",
        },
    )


if __name__ == "__main__":
    # export PYTHONPATH=ImPartial/impartial:ImPartial/impartial/api
    # python main.py
    main()
