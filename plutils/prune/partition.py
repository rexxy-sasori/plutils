import itertools

import numpy as np
from torch import nn

from plutils.utils import is_linear_transform_layer

DEFAULT_BLOCK_SEARCH_SPACE = {
    'br': np.arange(1, 6000),
    'bc': np.arange(1, 6000),
}


def is_block_size_candidate_valid(candidate: int, dim: int):
    if candidate > dim:
        return False
    if dim % candidate != 0:
        return False
    else:
        return True


def conv_unstructured_single(cout, cin, hk, wk, search_space, tau_acc):
    return [(1, 1)]


def conv_unstructured_channel(cout, cin, hk, wk, search_space, tau_acc):
    return [(1, hk * wk)]


def conv_filter(ccout, cin, hk, wk, search_space, tau_acc):
    return [(1, cin * hk * wk)]


def conv_structured_channel(cout, cin, hk, wk, search_space, tau_acc):
    return [(cout, hk * wk)]


def conv_mixed_structured_channel_filter(cout, cin, hk, wk, search_space, tau_acc):
    flag = np.random.randint(0, 2)
    if flag == 0:
        return conv_structured_channel(cout, cin, hk, wk, search_space, tau_acc)
    else:
        return conv_filter(cout, cin, hk, wk, search_space, tau_acc)


def conv_mixed_unstructured_channel_filter(cout, cin, hk, wk, search_space, tau_acc):
    flag = np.random.randint(0, 2)
    if flag == 0:
        return conv_unstructured_channel(cout, cin, hk, wk, search_space, tau_acc)
    else:
        return conv_filter(cout, cin, hk, wk, search_space, tau_acc)


def conv_mix(cout, cin, hk, wk, search_space, tau_acc):
    return [
        (1, 1),  # unstructured
        (1, hk * wk),  # unstructured channel
        (cout, hk * wk),  # structured channel
        (1, cin * hk * wk),  # filter pruning
        (cout, 1),  # conv column pruning
    ]


def conv_block(cout, cin, hk, wk, search_space, tau_acc):
    possible_brs = np.array([b for b in search_space['br'] if is_block_size_candidate_valid(b, cout)])
    possible_bcs = np.array([b for b in search_space['bc'] if is_block_size_candidate_valid(b, cin * hk * wk)])

    possible_br_bc_set = list(itertools.product(possible_brs, possible_bcs))

    if 0 < tau_acc < 1:
        total_ele = cout * cin * hk * wk
        max_param_one_block = tau_acc * total_ele
    else:
        max_param_one_block = tau_acc

    br_bc_candidates = list(filter(lambda x: x[0] * x[1] <= max_param_one_block, possible_br_bc_set))
    if len(br_bc_candidates) == 0:
        br_bc_candidates = [(1, 1)]
    return br_bc_candidates


def fc_mix(num_row, num_column, search_space, tau_acc):
    return [
        (1, 1),
        (num_row, 1),
        (1, num_column)
    ]


def fc_unstructured_single(num_row, num_column, search_space, tau_acc):
    return [(1, 1)]


def fc_block(num_row, num_column, search_space, tau_acc):
    possible_brs = np.array([b for b in search_space['br'] if is_block_size_candidate_valid(b, num_row)])
    possible_bcs = np.array(np.array([b for b in search_space['bc'] if is_block_size_candidate_valid(b, num_column)]))
    possible_br_bc_set = list(itertools.product(possible_brs, possible_bcs))

    if 0 < tau_acc < 1:
        total_ele = num_row * num_column
        max_param_one_block = tau_acc * total_ele
    else:
        max_param_one_block = tau_acc
    # br_bc_candidates = possible_br_bc_set
    br_bc_candidates = list(filter(lambda x: x[0] * x[1] <= max_param_one_block, possible_br_bc_set))
    if len(br_bc_candidates) == 0:
        br_bc_candidates = [(1, 1)]
    return br_bc_candidates


CONV_PRUNING_FUNC = {
    'unstructured': conv_unstructured_single,
    'unstructured_channel': conv_unstructured_channel,
    'filter_only': conv_filter,
    'structured_channel': conv_structured_channel,
    'mixed_structured_channel_filter': conv_mixed_structured_channel_filter,
    'mixed_unstructured_channel_filter': conv_mixed_unstructured_channel_filter,
    'block': conv_block,
    'existing': conv_mix
}

FC_PRUNING_FUNC = {
    'unstructured': fc_unstructured_single,
    'block': fc_block,
    'existing': fc_mix
}


def get_block_search_space_for_conv(weight: np.array, mode: str = 'default', search_space={}, tau_acc=1):
    cout, cin, hk, wk = weight.shape
    return CONV_PRUNING_FUNC[mode](cout, cin, hk, wk, search_space, tau_acc)


def get_block_search_space_fc(weight: np.array, mode: str = 'default', search_space={}, tau_acc=1):
    num_out_features, num_in_features = weight.shape
    return FC_PRUNING_FUNC[mode](num_out_features, num_in_features, search_space, tau_acc)


def get_block_search_space_single_layer(
        layer: nn.Module,
        conv_mode: str = 'unstructured',
        fc_mode: str = 'unstructured',
        search_space={},
        tau_acc=1
):
    weight = layer.weight.data
    if isinstance(layer, nn.Conv2d):
        return get_block_search_space_for_conv(weight, conv_mode, search_space, tau_acc)
    elif isinstance(layer, nn.Linear):
        return get_block_search_space_fc(weight, fc_mode, search_space, tau_acc)


def get_search_space(usr_valid_brs=(), usr_valid_bcs=()):
    ret = {}
    if len(usr_valid_brs) == 0:
        ret['br'] = DEFAULT_BLOCK_SEARCH_SPACE['br']
    elif len(usr_valid_brs) != 0:
        ret['br'] = set(usr_valid_brs).intersection(set(DEFAULT_BLOCK_SEARCH_SPACE['br'].tolist()))

    if len(usr_valid_bcs) == 0:
        ret['bc'] = DEFAULT_BLOCK_SEARCH_SPACE['bc']
    elif len(usr_valid_brs) != 0:
        ret['bc'] = set(usr_valid_brs).intersection(set(DEFAULT_BLOCK_SEARCH_SPACE['bc'].tolist()))

    return ret


def get_block_search_space_model(
        model: nn.Module,
        conv_mode: str = 'unstructured',
        fc_mode: str = 'unstructured',
        usr_valid_brs=(),
        usr_valid_bcs=(),
        tau_acc=1,
        op_unique=False,
        filter_func=None
):
    ret = {}

    search_space = get_search_space(usr_valid_brs, usr_valid_bcs)

    for idx, (name, module) in enumerate(model.named_modules()):
        if idx == 0:
            continue

        if isinstance(module, nn.Sequential) or isinstance(module, nn.ModuleList):
            continue

        break_down_name = name.split('.')
        break_down_name_wo_parallel = [s for s in break_down_name if s != 'module']
        name_wo_parallel = '.'.join(break_down_name_wo_parallel)

        if is_linear_transform_layer(module):
            if op_unique:
                shape = tuple(module.weight.data.shape)
                if ret.get(shape) is None:
                    ret[shape] = get_block_search_space_single_layer(
                        module, conv_mode, fc_mode, search_space, tau_acc
                    )
            else:
                ret[name_wo_parallel] = get_block_search_space_single_layer(
                    module, conv_mode, fc_mode, search_space, tau_acc
                )

    if filter_func is not None:
        for key, vals in ret.items():
            ret[key] = list(filter(filter_func, vals))

    return ret
