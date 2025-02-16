from typing import Any, Dict, Optional

import os
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, default_collate

from src import log
from data_prep.feature_store.featurize import preprocess


class AbideDataModule(LightningDataModule):
    """`LightningDataModule` for the ABIDE dataset.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        dataset_cls,
        data_dir: str = "data/",
        fold: int = 0,
        train_file: str = "train.csv",
        valid_file: str = "valid.csv",
        test_file: str = "test.csv",
        roi_dir: str = './data/functionals/cpac/filt_global/rois_cc200',
        store_path: str = './feature/rois_cc200/raw',
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `AbideDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.trn_data = pd.read_csv(os.path.join(f"{data_dir}/fold_{fold}", train_file))
        self.trn_data['split'] = "train"

        self.val_data = pd.read_csv(os.path.join(f"{data_dir}/fold_{fold}", valid_file))
        self.val_data['split'] = "valid"

        self.tst_data = pd.read_csv(os.path.join(f"{data_dir}/fold_{fold}", test_file))
        self.tst_data['split'] = "test"

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of ABIDE classes (2).
        """
        return 2

    def prepare_data(self) -> None:
        # Preprocesses data, contruct feature store
        if not os.path.exists(self.hparams.store_path):
            log.info(f"Feature is not ready, prepare feature store at {self.hparams.store_path}")
            preprocess(
                roi_dir=self.hparams.roi_dir,
                store_path=self.hparams.store_path
            )
        else:
            log.info(f"Using feature store at {self.hparams.store_path}")

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        dataset: Dataset = self.hparams.dataset_cls(
            store_path=self.hparams.store_path,
            data=self.trn_data,
            sampling=True
        )
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=default_collate
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        dataset: Dataset = self.hparams.dataset_cls(
            store_path=self.hparams.store_path,
            data=self.val_data
        )
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=default_collate
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        dataset: Dataset = self.hparams.dataset_cls(
            store_path=self.hparams.store_path,
            data=self.tst_data
        )
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=default_collate
        )

    def predict_dataloader(self) -> Any:
        """Create and return the predict dataloader.

        :return: The predict dataloader.
        """
        all_data = pd.concat([
            self.trn_data, self.val_data, self.tst_data
        ]).reset_index(drop=True)

        dataset: Dataset = self.hparams.dataset_cls(
            store_path=self.hparams.store_path,
            data=all_data
        )

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=default_collate
        )
    
    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass