import argparse

from pytorch_lightning import seed_everything

import plutils.data as datazoo
import plutils.models as modelzoo
from plutils.config.parsers import *
from plutils.config.usr_config import get_usr_config
from plutils.prune.dlth_pruner import DlthPruner

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--usr-config', type=str, required=True)
    command_args = parser.parse_args()
    usr_config = get_usr_config(command_args.usr_config)

    seed_everything(usr_config.seed)
    logger = parse_logging(usr_config, usr_config.trainer.use_time_code, name='info_extrusion')

    model = parse_model(usr_config, modelzoo)
    block_policy = parse_block_policy(model, usr_config)
    info_extrusion_strategy = parse_strategy(usr_config.pruner.init_args.trainer.strategy)

    data_module = parse_datamodule(usr_config, datazoo)
    info_extrusion_datamodule = parse_datamodule(usr_config.pruner.init_args, datazoo)

    pruner = DlthPruner(
        model=model, block_policy=block_policy,
        info_extrusion_datamodule=info_extrusion_datamodule,
        model_sparsity=usr_config.pruner.init_args.model_sparsity,
        reg_ceiling=usr_config.pruner.init_args.reg_ceiling,
        update_reg_interval=usr_config.pruner.init_args.update_reg_interval,
        epislon_lambda=usr_config.pruner.init_args.epislon_lambda,
        prune_first_layer=usr_config.pruner.init_args.prune_first_layer,
        prune_last_layer=usr_config.pruner.init_args.prune_last_layer,
    )

    pruner.prune(data_module)
