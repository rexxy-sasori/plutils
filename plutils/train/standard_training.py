from typing import Callable

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics

from plutils.config.parsers import parse_optimization_config
from plutils.utils import debug_msg, none_check, attr_check


class StandardTrainingModule(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            usr_config=None,
            optimizer_config: dict = None,
            lr_scheduler_config: dict = None,
            loss_func: Callable = F.cross_entropy,
            *args, **kwargs
    ):
        super(StandardTrainingModule, self).__init__(*args, **kwargs)
        self.module = model
        self.loss_func = loss_func

        if usr_config is None:
            self.optimizer_config = optimizer_config
            self.lr_scheduler_config = lr_scheduler_config
            self.label_smoothing = none_check(self.optimizer_config.get('label_smoothing'))
        else:
            self.optimizer_config = usr_config.optimizer
            self.lr_scheduler_config = usr_config.lr_scheduler
            self.label_smoothing = attr_check(self.optimizer_config, 'label_smoothing', 0)

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

        self.save_hyperparameters("usr_config")

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def configure_optimizers(self):
        optimizer, lr_scheduler = parse_optimization_config(
            self.module,
            self.optimizer_config,
            self.lr_scheduler_config
        )

        if isinstance(lr_scheduler, optim.lr_scheduler.OneCycleLR):
            lr_scheduler_instance_config = {
                'scheduler': lr_scheduler,
                'interval': 'step',
                'frequency': 1
            }
            return [optimizer], [lr_scheduler_instance_config]
        else:
            return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        train_loss = self.loss_func(y_hat, y, label_smoothing=self.label_smoothing)

        self.train_acc(y_hat, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        self.log('train_loss', train_loss, on_step=True, on_epoch=True)

        return {'loss': train_loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        val_loss = self.loss_func(y_hat, y)

        self.val_acc(y_hat, y)
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True, logger=True)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        test_loss = self.loss_func(y_hat, y)
        self.test_acc(y_hat, y)
        self.log('test_acc', self.test_acc, on_step=True, on_epoch=True, logger=True)
        self.log('test_loss', test_loss, on_step=True, on_epoch=True, logger=True)


def run_standard_training(
        pl_module, data_module, num_epochs, ckpt_path,
        logger, strategy, callbacks, verbose
):
    if not isinstance(pl_module, StandardTrainingModule):
        msg = f'Wrap the model with {StandardTrainingModule.__name__} before start training'
        raise ValueError(msg)

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="auto",
        accumulate_grad_batches=1,
        benchmark=True,
        devices=-1,
        logger=logger,
        strategy=strategy,
        callbacks=callbacks,
    )

    try:
        debug_msg(f"loading ckpt from: {ckpt_path}", verbose)
        ckpt = torch.load(ckpt_path)
        pl_module.load_state_dict(ckpt['state_dict'])
        debug_msg(f"finished loading model states from {ckpt_path} !!", verbose)
    except FileNotFoundError:
        debug_msg(f'ckpt not found at {ckpt_path}, start train the model', verbose)
        trainer.fit(model=pl_module, datamodule=data_module)
    finally:
        trainer.test(model=pl_module, datamodule=data_module)
