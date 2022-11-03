"""
Author: Hanfei Rex Geng

This python module implements iterative magnitude pruning.

Current: Pruning rate scheduling with constant learning rate scheduling

todo:
    - Pruning rate scheduling with arbitrary learning rate scheduling
    - Weight state reset to early epochs (found in LTH)
"""

import copy
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics

from plutils.analysis import get_num_params, get_sparsity
from plutils.config.parsers import parse_strategy, parse_logging, parse_callbacks
from plutils.module import MaskLinear, MaskConv2d, convert_module
from plutils.prune.utils import (
    find_targets,
    exp_pruning_schedule,
    global_prune_model,
    local_prune_model,
    mag_score_func
)
from plutils.train.standard_training import run_standard_training, StandardTrainingModule
from plutils.utils import rsetattr


class ImpPruner:
    def __init__(
            self,
            model: nn.Module,
            model_sparsity: float,
            block_policy: dict,
            lr: float,
            global_prune: bool = False,
            prune_first_layer: bool = True,
            prune_last_layer: bool = True,
            sparsity_margin: float = 1e-4,
            enable_checkpoint: bool = False,
            ckpt_path: str = None,
            pruning_interval: int = 2,
            sparsity_step: int = 2,
            debug_on: bool = False,
            usr_config=None,
    ):
        self.usr_config = usr_config
        self.enable_checkpoint = enable_checkpoint
        self.ckpt_path = ckpt_path
        self.sparsity_margin = sparsity_margin
        self.debug_on = debug_on

        self.pretrain_module = StandardTrainingModule(model, usr_config)
        self.imp_module = ImpModule(
            model, model_sparsity, block_policy, lr,
            global_prune, prune_first_layer, prune_last_layer,
            pruning_interval, sparsity_step
        )

    def prune(self, data_module):
        """
        Main interface for pruner

        Example:
        usr_config = get_usr_config(path)

        seed_everything(usr_config.seed)

        model = parse_model(...)
        block_policy = parse_block_policy(...)
        data_module = parse_datamodule(...)

        pruner = ImpPruner(model, block_policy, ...)
        pruner.prune(data_module)

        :param data_module:
        :return:
        """
        usr_config = self.usr_config

        logger = parse_logging(usr_config=usr_config, use_time_code=usr_config.trainer.use_time_code, name='pretrain')
        pretrain_strategy = parse_strategy(usr_config.pruner.init_args.pretrain_strategy)
        callbacks = parse_callbacks(logger, usr_config, pl.callbacks, usr_config.trainer.persist_ckpt)

        run_standard_training(
            self.pretrain_module, data_module, self.usr_config.trainer.epochs,
            self.ckpt_path, logger, pretrain_strategy, callbacks, self.debug_on
        )

        sparsity_schedule_logger = parse_logging(
            save_dir=os.path.split(logger.root_dir)[0],
            name='run_sparsity_schedule'
        )

        run_sparsity_schedule_strategy = parse_strategy(usr_config.pruner.init_args.run_sparsity_schedule_strategy)
        self.run_sparsity_schedule(sparsity_schedule_logger, run_sparsity_schedule_strategy, data_module)

    def run_sparsity_schedule(self, logger, strategy, data_module):
        from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
        callbacks = [
            EarlyStopping(
                monitor='sparsity', mode='max',
                patience=self.usr_config.pruner.init_args.finetune_epochs,
                verbose=True, stopping_threshold=self.imp_module.model_sparsity - self.sparsity_margin
            )
        ]

        if self.enable_checkpoint:
            callbacks.append(
                ModelCheckpoint(
                    monitor='val_loss_density_product',
                    dirpath=logger.log_dir,
                    filename='{epoch}-{val_loss:.4f}-{val_acc:.4f}-{sparsity:.8f}',
                    save_top_k=1,
                    verbose=True
                )
            )

        trainer = pl.Trainer(
            max_epochs=1000000,  # don't worry, early stopping will always kill this
            accelerator="auto", benchmark=True, devices=-1, logger=logger, strategy=strategy,
            callbacks=callbacks,
            enable_checkpointing=self.enable_checkpoint
        )

        trainer.fit(model=self.imp_module, datamodule=data_module)
        trainer.test(model=self.imp_module, datamodule=data_module)


class ImpModule(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            model_sparsity: float,
            block_policy: dict,
            lr: float,
            global_prune: bool = False,
            prune_first_layer: bool = True,
            prune_last_layer: bool = True,
            pruning_interval: int = 2,
            sparsity_step: int = 2,
            pruning_rate_func=exp_pruning_schedule,
            *args, **kwargs
    ):
        super(ImpModule, self).__init__(*args, **kwargs)
        self.module = model
        self.model_sparsity = model_sparsity
        self.block_policy = block_policy
        self.pruning_interval = pruning_interval
        self.sparsity_step = sparsity_step
        self.global_prune = global_prune
        self.lr = lr
        self.total_param = get_num_params(model)

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

        self.pruning_rate_func = pruning_rate_func
        self.target_layers = find_targets(model, prune_first_layer, prune_last_layer)
        self.register_mask()
        self.rewind_state = copy.deepcopy(self.target_layers)

    def register_mask(self):
        for name, module in self.target_layers.items():
            new_m = convert_module(
                module, MaskConv2d.convert, MaskLinear.convert,
                torch.ones_like(module.weight.data)
            )

            rsetattr(self.module, name, new_m)
            self.target_layers[name] = new_m

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=self.lr)

    def rewind(self):
        for n in self.target_layers.keys():  # copy the data
            self.target_layers[n].weight.data = self.rewind_state[n].weight.data

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def on_epoch_start(self) -> None:
        current_epoch = self.current_epoch

        if current_epoch % self.pruning_interval == 0:
            current_pruning_rate = self.pruning_rate_func(
                current_epoch, s_f=self.model_sparsity,
                pruning_interval=self.pruning_interval,
                sparsity_step=self.sparsity_step
            )
            self.cut_weights(current_pruning_rate)

    def on_epoch_end(self) -> None:
        current_epoch = self.current_epoch

        if current_epoch % self.pruning_interval == 0:
            self.rewind_state = copy.deepcopy(self.target_layers)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.module(x)
        train_loss = F.cross_entropy(y_hat, y, label_smoothing=0.1)
        sparsity = get_sparsity(self.module)['sparsity']

        self.train_acc(y_hat, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        self.log('train_loss', train_loss, on_step=True, on_epoch=True)
        self.log('sparsity', sparsity, on_step=True, on_epoch=True)

        return {'loss': train_loss}

    def cut_weights(self, pruning_rate):
        if self.global_prune:
            local_prune_model(self.target_layers, pruning_rate, self.block_policy, mag_score_func)
        else:
            global_prune_model(self.target_layers, pruning_rate, self.total_param, self.block_policy, mag_score_func)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.module(x)
        val_loss = F.cross_entropy(y_hat, y)
        margin = self.model_sparsity - get_sparsity(self.module)['sparsity']
        self.val_acc(y_hat, y)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, logger=True)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, logger=True)
        self.log('val_loss_density_product', val_loss * margin, on_step=False, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.module(x)
        test_loss = F.cross_entropy(y_hat, y)
        sparsity = get_sparsity(self.module)['sparsity']
        self.test_acc(y_hat, y)
        self.log('test_acc', self.test_acc, on_step=True, on_epoch=True, logger=True)
        self.log('sparsity', sparsity, on_step=True, on_epoch=True, logger=True)
        self.log('test_loss', test_loss, on_step=True, on_epoch=True, logger=True)
