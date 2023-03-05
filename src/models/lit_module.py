from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.retrieval import RetrievalMRR

from src.models.components.text_model import BertWrapper
from src.models.components.video_model import SpatioTemporalTransformer3D
from src.models.components.loss_fn import LongShortAlignmentLoss


class SpatioTemporalVideoTransformer(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        video_model: SpatioTemporalTransformer3D,
        text_model: BertWrapper,
        criterion: LongShortAlignmentLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.video_model = video_model
        self.text_model = text_model

        # loss function
        self.criterion = criterion
        
        # optimization
        self.optimizer = optimizer
        self.scheduler = scheduler

        # metric objects for calculating and averaging accuracy across batches
        self.train_video_retrieval = RetrievalMRR()
        self.val_video_retrieval = RetrievalMRR()
        self.test_video_retrieval = RetrievalMRR()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_video_retrieval_best = MaxMetric()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        video_features = self.video_model(x)
        text_features = self.text_model(y)
        return video_features, text_features

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_video_retrieval_best doesn't store accuracy from these checks
        self.val_video_retrieval_best.reset()

    def model_step(self, batch: Any):
        x, y = batch
        video_features, text_features = self.forward(x, y)
        loss = self.criterion(video_features, text_features)
        return loss, video_features, text_features

    def training_step(self, batch: Any, batch_idx: int):
        loss, video_features, text_features = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_video_retrieval(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_video_retrieval, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs from all batches of the epoch
        # this may not be an issue when training on mnist
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs

        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, video_features, text_features = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_video_retrieval(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_video_retrieval, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_video_retrieval.compute()  # get current val acc
        self.val_video_retrieval_best(acc)  # update best so far val acc
        # log `val_video_retrieval_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_video_retrieval_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, video_features, text_features = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_video_retrieval(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_video_retrieval, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.optimizer(params=self.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
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


if __name__ == "__main__":
    _ = SpatioTemporalVideoTransformer(None, None, None)
