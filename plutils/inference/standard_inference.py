import glob
import os

import pytorch_lightning as pl

from plutils.utils import debug_msg


def run_standard_inference(result_path, verbose):  # todo
    """
    A trained model contains the following:
        pl ckpt file optional
        metrics.csv required
        usr_config.yaml required
    """
    path, is_ckpt = get_ckpt_path(result_path, verbose)

    if is_ckpt:
        ckpt_path = path
        load_acc_from_ckpt(result_path, ckpt_path)
    else:
        csv_path = path
        load_acc_from_csv(csv_path)


def load_acc_from_ckpt(result_path, ckpt_path):
    yaml_path = os.path.join(result_path, 'usr_config.yaml')
    # parse data, and model
    # run inference

    trainer = pl.Trainer()


def load_acc_from_csv(csv_path):
    pass


def get_ckpt_path(result_path, verbose):
    ckpt_files_this = glob.glob(os.path.join(result_path, '*.ckpt'))
    ckpt_files_under_this = glob.glob(os.path.join(result_path, '*/*.ckpt'))
    if len(ckpt_files_this) == len(ckpt_files_under_this) == 0:
        debug_msg(f'no ckpt files found under {result_path}', verbose)
        ret = os.path.join(result_path, 'metrics.csv')
        assert os.path.exists(ret)
        return ret, False

    ckpt_files = ckpt_files_this + ckpt_files_under_this
    for p in ckpt_files:
        debug_msg(f'found ckpt file at {p}', verbose)

    ckpt_path = ckpt_files[0]
    debug_msg(f'Proceed to use the first one {ckpt_path}', verbose)
    return ckpt_path, True
