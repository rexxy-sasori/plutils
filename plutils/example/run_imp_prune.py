import argparse

import pytorch_lightning as pl

import plutils.data as datazoo
import plutils.models as modelzoo
from plutils.config.parsers import parse_model, parse_datamodule, parse_block_policy
from plutils.config.usr_config import get_usr_config
from plutils.prune.imp_pruner import ImpPruner

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--usr-config', type=str, required=True)
    command_args = parser.parse_args()
    usr_config = get_usr_config(command_args.usr_config)

    pl.seed_everything(usr_config.seed)

    model = parse_model(usr_config, modelzoo)
    block_policy = parse_block_policy(model, usr_config)
    data_module = parse_datamodule(usr_config, datazoo)

    pruner = ImpPruner(
        model=model, block_policy=block_policy,
        model_sparsity=usr_config.pruner.init_args.model_sparsity,
        lr=usr_config.pruner.init_args.lr,
        global_prune=usr_config.pruner.init_args.global_prune,
        prune_first_layer=usr_config.pruner.init_args.prune_first_layer,
        prune_last_layer=usr_config.pruner.init_args.prune_last_layer,
        pruning_interval=usr_config.pruner.init_args.pruning_interval,
        sparsity_step=usr_config.pruner.init_args.sparsity_step,
        ckpt_path=usr_config.pruner.init_args.ckpt_path,
        usr_config=usr_config
    )

    pruner.prune(data_module)
