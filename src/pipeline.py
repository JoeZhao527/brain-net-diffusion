from typing import Any, Dict, List, Optional, Tuple

import hydra
import pytorch_lightning as pl
import rootutils
import torch
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger
from omegaconf import OmegaConf
from omegaconf import DictConfig
import os

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import log
from src.train import train
from src.utils import (
    extras,
    get_metric_value
)

@hydra.main(version_base=None, config_path="../configs", config_name="acerta.yaml")
def main(cfg: DictConfig):
    if cfg.prev_stage:
        # Use specified resources paths
        if cfg.get("dependent_dir", None):
            dependent_dir = cfg.dependent_dir
        else:
            dependent_dir = os.path.join(
                os.path.dirname(cfg.paths.output_dir),
                f"data.fold={cfg.data.fold},experiment={cfg.prev_stage}"
            )

        # Use previous checkpoint if model relies on some checkpoints
        if cfg.model.get("ckpt_path", None):
            cfg.model.ckpt_path = os.path.join(dependent_dir, "checkpoints")

        # Use the original feature or use the prediction from previous stage
        if not cfg.get("use_original_data", False):
            cfg.data.store_path = os.path.join(dependent_dir, "predicts")

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    return metric_value

if __name__ == '__main__':
    main()