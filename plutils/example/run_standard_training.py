import argparse

from pytorch_lightning import seed_everything

import plutils.data as datazoo
import plutils.models as modelzoo
from plutils.config.usr_config import get_usr_config
from plutils.config.parsers import *
from plutils.train.standard_training import StandardTrainingModule, run_standard_training

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--usr-config', type=str, required=True)
    command_args = parser.parse_args()
    usr_config = get_usr_config(command_args.usr_config)

    seed_everything(usr_config.seed)

    model = parse_model(usr_config, modelzoo)
    data_module = parse_datamodule(usr_config, datazoo)

    logger = parse_logging(
        usr_config=usr_config,
        use_time_code=usr_config.trainer.use_time_code,
        name='standard_training'
    )

    strategy = parse_strategy(usr_config.trainer.init_args.training_strategy)
    callbacks = parse_callbacks(logger, usr_config, usr_config.trainer.persist_ckpt)
    training_module = StandardTrainingModule(model, usr_config)

    run_standard_training(
        pl_module=training_module,
        data_module=data_module,
        num_epochs=usr_config.trainer.num_epochs,
        ckpt_path=usr_config.model.ckpt_path,
        logger=logger,
        strategy=strategy,
        callbacks=callbacks,
        verbose=usr_config.verbose
    )

