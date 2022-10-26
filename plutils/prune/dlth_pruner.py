"""
Author: Hanfei Rex Geng

This python module implements dual lottery ticket hypothesis (DLTH).

Current: uniform sparsity level assignment

todo:
    - Layerwise sparsity level assignment
"""

import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
from pytorch_lightning import Trainer
from torch import nn

from plutils.analysis import get_pr_over_kp
from plutils.config.parsers import parse_logging, parse_callbacks, parse_strategy
from plutils.module import convert_module, GrowRegMaskConv2d, GrowRegMaskLinear
from plutils.prune.utils import find_targets, get_block_dim
from plutils.train.standard_training import StandardTrainingModule
from plutils.utils import rsetattr, debug_msg


class DlthPruner:
    def __init__(
            self,
            model: nn.Module,
            model_sparsity: float,
            block_policy: dict,
            info_extrusion_datamodule: pl.LightningDataModule,
            reg_ceiling: float = 1,
            update_reg_interval: int = 5,
            epislon_lambda: float = 1e-4,
            prune_first_layer: bool = True,
            prune_last_layer: bool = True,
            debug_on: bool = False,
            usr_config=None
    ):
        self.pruner_module = DlthPrunerModule(
            model, model_sparsity, block_policy,
            update_reg_interval, epislon_lambda,
            prune_first_layer, prune_last_layer,
        )

        self.info_extrusion_datamodule = info_extrusion_datamodule
        self.usr_config = usr_config

        num_devices = torch.cuda.device_count()
        if num_devices == 0:
            raise ValueError('No CUDA device found')

        num_global_iter_steps = (reg_ceiling / epislon_lambda * update_reg_interval)
        num_training_batches = int(len(info_extrusion_datamodule.train_dataloader()) / num_devices)
        self.info_extrusion_max_epochs = 1 + int(num_global_iter_steps / num_training_batches)
        msg = f'INFO extrusion epochs: {self.info_extrusion_max_epochs} = {num_global_iter_steps}/{num_training_batches}'
        debug_msg(msg, debug_on)

    def prune(self, finetune_datamodule):
        usr_config = self.usr_config
        logger = parse_logging(
            usr_config=usr_config,
            use_time_code=usr_config.trainer.use_time_code,
            name='info_extrusion'
        )
        info_extrusion_strategy = parse_strategy(usr_config.pruner.init_args.usr_config.trainer.strategy)

        self.info_extrusion(logger, info_extrusion_strategy)

        finetune_logger = parse_logging(save_dir=os.path.split(logger.root_dir)[0], name='finetune')
        strategy = parse_strategy(usr_config.trainer.strategy)
        callbacks = parse_callbacks(finetune_logger, usr_config, usr_config.trainer.persist_ckpt)

        self.finetune(finetune_logger, strategy, callbacks, finetune_datamodule, usr_config.trainer.epochs)

    def info_extrusion(self, logger, strategy):
        trainer = Trainer(
            max_epochs=self.info_extrusion_max_epochs, accelerator="auto",
            benchmark=True, devices=-1, logger=logger, strategy=strategy,
        )

        trainer.fit(model=self.pruner_module, datamodule=self.info_extrusion_datamodule)
        trainer.test(model=self.pruner_module, datamodule=self.info_extrusion_datamodule)

    def get_info_extruded_mask_model(self):
        model = self.pruner_module.model
        for n, m in self.pruner_module.target_layers.items():
            new_m = m.to_finetune_mode()
            rsetattr(model, n, new_m)

        return model

    def finetune(self, logger, strategy, callbacks, data_module, num_epochs):
        trainer = Trainer(
            max_epochs=num_epochs,
            accelerator="auto",
            accumulate_grad_batches=1,
            benchmark=True,
            devices=-1,
            logger=logger,
            strategy=strategy,
            callbacks=callbacks,
        )

        masked_model = self.get_info_extruded_mask_model()
        model = StandardTrainingModule(masked_model, self.usr_config)
        trainer.fit(model=model, datamodule=data_module)
        trainer.test(model=model, datamodule=data_module)


class DlthPrunerModule(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            model_sparsity: float,
            block_policy: dict,
            update_reg_interval: int = 5,
            epislon_lambda: float = 1e-4,
            prune_first_layer: bool = True,
            prune_last_layer: bool = True,
            *args, **kwargs,
    ):
        super(DlthPrunerModule, self).__init__(*args, **kwargs)
        self.model = model
        self.block_policy = block_policy

        self.update_reg_interval = update_reg_interval
        self.epislon_lambda = epislon_lambda

        self.target_layers = find_targets(model, prune_first_layer, prune_last_layer)
        self.sparsities = {n: model_sparsity for n, m in self.target_layers.items()}

        self.get_grow_reg_modules()

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def get_grow_reg_modules(self):
        for name, module in self.target_layers.items():
            new_m = convert_module(
                module, GrowRegMaskConv2d.convert, GrowRegMaskLinear.convert,
                get_block_dim(self.block_policy, name, module), self.sparsities[name]
            )

            rsetattr(self.model, name, new_m)
            self.target_layers[name] = new_m

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        global_iter_idx = self.global_step
        if global_iter_idx % self.update_reg_interval == 0:
            for name, module in self.target_layers.items():
                module.update_reg(self.epislon_lambda)

        max_pr_over_kp = max([get_pr_over_kp(m, 1 - m.pr_mask) for n, m in self.target_layers.items()])
        train_loss = F.cross_entropy(y_hat, y, label_smoothing=0.1)

        self.train_acc(y_hat, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        self.log('train_loss', train_loss, on_step=True, on_epoch=True)
        self.log('max_pr_over_kp', max_pr_over_kp, on_step=True, on_epoch=True)

        return {'loss': train_loss}

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.model(x)
        test_loss = F.cross_entropy(y_hat, y)
        self.test_acc(y_hat, y)
        self.log('test_acc', self.test_acc, on_step=True, on_epoch=True, logger=True)
        self.log('test_loss', test_loss, on_step=True, on_epoch=True, logger=True)

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def backward(self, loss, optimizer, optimizer_idx, *args, **kwargs) -> None:
        loss.backward(*args, **kwargs)
        for name, module in self.target_layers.items():
            module.apply_reg()

    def configure_optimizers(self):
        return optim.SGD(self.model.parameters(), lr=5e-2)
