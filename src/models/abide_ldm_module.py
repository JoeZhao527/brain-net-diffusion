from typing import Any, Dict, Tuple

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric
from tqdm import tqdm


class AbideLDMModule(LightningModule):
    """Latent diffusion training module
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        sample_num: int = 3,
        sample_test: bool = False,
        sample_wi_seed: bool = False,
        **kwargs
    ) -> None:
        """Initialize the module

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.train_ccl_loss = MeanMetric()
        self.train_recon_loss = MeanMetric()

        self.val_loss = MeanMetric()
        self.val_ccl_loss = MeanMetric()
        self.val_recon_loss = MeanMetric()

        self.test_loss = MeanMetric()
        self.test_ccl_loss = MeanMetric()
        self.test_recon_loss = MeanMetric()

        # for tracking best so far validation loss
        self.val_loss_best = MinMetric()
        self.val_ccl_loss_best = MinMetric()
        self.val_recon_loss_best = MinMetric()

        # for tracking best so far validation loss
        self.val_loss_best = MinMetric()
        self.val_ccl_best = MinMetric()
        self.val_recon_best = MinMetric()
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(**batch)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

    def model_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        loss, _, target = self.forward(batch)
        
        return loss, target

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, target = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss["loss"])
        self.train_recon_loss(loss["recon_loss"])
        self.train_ccl_loss(loss["ccl_loss"])

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/train_recon_loss", self.train_recon_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/train_ccl_loss", self.train_ccl_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # return loss or backpropagation will fail
        return loss["loss"]

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, target = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss["loss"])
        self.val_recon_loss(loss["recon_loss"])
        self.val_ccl_loss(loss["ccl_loss"])

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recon_loss", self.val_recon_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/ccl_loss", self.val_ccl_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        # get current val loss
        loss = self.val_loss.compute()
        recon_loss = self.val_recon_loss.compute()
        ccl_loss = self.val_ccl_loss.compute()

        # update best so far val loss
        self.val_loss_best(loss)
        self.val_recon_loss_best(recon_loss)
        self.val_ccl_loss_best(ccl_loss)

        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/recon_loss_best", self.val_recon_loss_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/ccl_loss_best", self.val_ccl_loss_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, target = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss["loss"])
        self.test_recon_loss(loss["recon_loss"])
        self.test_ccl_loss(loss["ccl_loss"])

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/recon_loss", self.test_recon_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/ccl_loss", self.test_ccl_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Any:
        prediction = []
        samples = []
        recons = []
        for _ in range(self.hparams.sample_num):
            # print(f"sample batch {batch_idx}, sample {sam}")
            if self.hparams.sample_wi_seed:
                samp, recon = self.net.sample_wi_seed(**batch)
            else:
                samp, recon = self.net.sample(
                    batch["edge"].shape,
                    batch["condition"]
                )

            samples.append(samp)
            recons.append(recon)
        
        for i in range(len(batch["id"])):
            sample_res = []
            recon_res = [batch["edge_mtx"][i]]
            # recon_res = []
            for j in range(self.hparams.sample_num):
                sample_res.append(samples[j][i])
                recon_res.append(recons[j][i])

            prediction.append({
                "id": batch["id"][i],
                "latent_repr": torch.stack(sample_res, dim=0),
                "reconstruct": torch.stack(recon_res, dim=0)
            })

        return prediction
        
    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}