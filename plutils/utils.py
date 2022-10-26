import functools
import os

import torch
from torch import nn


def debug_msg(msg, verbose=False):
    if verbose:
        print(msg)


def get_wd_nwd_params(model: nn.Module, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    ret = [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}
    ]

    return ret


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def is_linear_transform_layer(layer):
    isconv = isinstance(layer, nn.Conv2d)
    isfc = isinstance(layer, nn.Linear)

    return isconv or isfc


def matrix_to_blocks(tensor, nrows_per_block, ncols_per_block):
    tensor_nrows, tensor_ncols = tensor.shape
    ret = tensor.reshape(tensor_nrows // nrows_per_block, nrows_per_block, -1, ncols_per_block)
    ret = torch.transpose(ret, 1, 2)
    ret = ret.reshape(-1, nrows_per_block, ncols_per_block)
    return ret, tensor_nrows // nrows_per_block, tensor_ncols // ncols_per_block


def blocks_to_matrix(blocks, num_blocks_row, num_blocks_col, nrows_per_block, ncols_per_block):
    ret = blocks.reshape(num_blocks_row, num_blocks_col, nrows_per_block, ncols_per_block)
    ret = torch.transpose(ret, 1, 2)
    ret = ret.reshape(num_blocks_row * nrows_per_block, num_blocks_col * ncols_per_block)
    return ret


def strip_module_in_module_name(name):
    break_down_name = name.split('.')
    break_down_name_wo_parallel = [s for s in break_down_name if s != 'module']
    name_wo_parallel = '.'.join(break_down_name_wo_parallel)
    return name_wo_parallel


def callback_exists(callbacks, callback_cls):
    for callback in callbacks:
        if isinstance(callback, callback_cls):
            return True

    return False


def make_directory(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def none_check(val, default=0):
    if val is not None:
        return val
    else:
        return default


def attr_check(obj, attr_name, default):
    if hasattr(obj, attr_name):
        return getattr(obj, attr_name)
    else:
        return default
