from datetime import datetime

from pytorch_lightning.loggers import CSVLogger
from torch import optim

from plutils.config.usr_config import EmptyConfig
from plutils.utils import none_check, get_wd_nwd_params, attr_check


def parse_optimization_config(model, optimizer_config, lr_scheduler_config):
    if isinstance(optimizer_config, dict):
        weight_decay_val = none_check(optimizer_config['init_args'].get('weight_decay'), 0)
        reg_bn = none_check(optimizer_config.get('reg_bn'), False)
        params = model.parameters() if reg_bn else get_wd_nwd_params(model, weight_decay=weight_decay_val)

        OptimCls = getattr(optim, optimizer_config['name'])
        optimizer = OptimCls(params, **optimizer_config['init_args'])
        LRScheCls = getattr(optim.lr_scheduler, lr_scheduler_config['name'])
        lr_scheduler = LRScheCls(optimizer, **lr_scheduler_config['init_args'])
    else:
        weight_decay_val = attr_check(optimizer_config.init_args, 'weight_decay', 0)
        reg_bn = attr_check(optimizer_config.init_args, 'reg_bn', False)
        params = model.parameters() if reg_bn else get_wd_nwd_params(model, weight_decay=weight_decay_val)

        OptimCls = getattr(optim, optimizer_config.name)
        optimizer = OptimCls(params, **optimizer_config.init_args.__dict__)
        LRScheCls = getattr(optim.lr_scheduler, lr_scheduler_config.name)
        lr_scheduler = LRScheCls(optimizer, **lr_scheduler_config.init_args.__dict__)

    return optimizer, lr_scheduler


def parse_model(usr_config, modellib):
    model_cls = getattr(modellib, usr_config.module.name)
    model_obj = model_cls(usr_config)
    return model_obj


def parse_datamodule(usr_config, datalib):
    dm_initializer = getattr(datalib, usr_config.data.name)
    dm = dm_initializer(**usr_config.data.init_args.__dict__)
    return dm


def parse_logging(logger_cls=CSVLogger, usr_config=None, use_time_code=False, name=None, save_dir=None):
    def _parse_name(usr_config, usr_time_code, name):
        time = datetime.now()
        timecode = time.strftime("%m.%d.%Y.%H.%M.%S.%f")

        valid_logging_name = hasattr(usr_config, 'logging_name') and not isinstance(usr_config.logging_name,
                                                                                    EmptyConfig)
        valid_name = name is not None

        ret = 'default'
        if not valid_name and not valid_logging_name and not usr_time_code:
            return ret

        if valid_name:
            return f"{timecode}/{name}" if usr_time_code else name
        else:
            if valid_logging_name:
                return f"{timecode}/{usr_config.logging_name}" if usr_time_code else f"{usr_config.logging_name}"
            else:
                return f"{timecode}" if usr_time_code else ret

    logger_name = _parse_name(usr_config, use_time_code, name)
    log_directory = usr_config.log_directory if save_dir is None else save_dir

    logger = logger_cls(
        save_dir=log_directory,
        name=logger_name
    )

    return logger


def parse_callbacks(logger, usr_config, callbacks_modules, persist_ckpt=True):
    callbacks = [
        callbacks_modules.LearningRateMonitor(logging_interval='step')
    ]

    for idx in range(len(usr_config.trainer.callbacks)):
        callback_cls = getattr(callbacks_modules, usr_config.trainer.callbacks[idx].name)
        callback_args = usr_config.trainer.callbacks[idx].init_args.__dict__
        callbacks.append(callback_cls(**callback_args))

    if persist_ckpt:
        callbacks.append(
            callbacks_modules.ModelCheckpoint(
                monitor='val_loss',
                dirpath=logger.log_dir,
                filename='{epoch}-{val_loss:.2f}-{val_acc:.2f}',
                save_top_k=1,
                verbose=True
            )
        )

    return callbacks


def parse_strategy(strategy_config):
    import pytorch_lightning.strategies as strategies
    if isinstance(strategy_config, EmptyConfig):
        return None

    strategy_cls = getattr(strategies, strategy_config.name)
    return strategy_cls(**strategy_config.init_args.__dict__)
