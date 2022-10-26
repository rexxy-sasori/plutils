import numpy as np
import torch

from plutils.utils import matrix_to_blocks, blocks_to_matrix, is_linear_transform_layer, strip_module_in_module_name


def get_block_dim(block_policy, name, module):
    ret = block_policy.get(strip_module_in_module_name(name))
    if ret is not None:
        return ret
    else:
        ret = block_policy.get(tuple(module.weight.data.shape))
        if ret is not None:
            return ret
        else:
            raise KeyError('block policy is not hashed by neither module name nor module shape')


def exp_pruning_schedule(current_epoch, s_f, s_i=0, pruning_interval=2, sparsity_step=2) -> float:
    exp_term = (s_f / sparsity_step) * ((current_epoch + pruning_interval) // pruning_interval)
    s_t = s_f + (s_i - s_f) * np.exp(-exp_term)
    return s_t


def mask_fc_layer_by_magnitude(weight, block_dims, alpha, load_balance):
    br, bc = block_dims
    alpha = alpha

    blocks, num_blocks_row, num_blocks_col = matrix_to_blocks(weight, br, bc)
    weight_blocks_reshape = blocks.reshape(num_blocks_row, num_blocks_col, br, bc)

    if not load_balance:
        score = torch.norm(blocks, dim=(-2, -1)) ** 2
        _, sorted_indices = torch.sort(score)

        mask = torch.ones_like(blocks)
        mask[sorted_indices[0:int(num_blocks_row * num_blocks_col * alpha)], :, :] = 0
        mask = blocks_to_matrix(mask, num_blocks_row, num_blocks_col, br, bc)
    else:
        mask = torch.ones_like(weight_blocks_reshape)

    return mask


def mask_conv_layer_by_magnitude(weight, block_dims, alpha, load_balance):
    cout, cin, hk, wk = weight.shape
    unroll_weight = weight.reshape(cout, cin * hk * wk)
    unroll_mask = mask_fc_layer_by_magnitude(unroll_weight, block_dims, alpha, load_balance)
    return unroll_mask.reshape(cout, cin, hk, wk)


def find_targets(model, prune_first_layer, prune_last_layer):
    linear_transform_modules = {
        name: module for name, module in model.named_modules()
        if is_linear_transform_layer(module)
    }

    target_layers = {}
    for idx, (name, module) in enumerate(linear_transform_modules.items()):
        if idx == 0 and not prune_first_layer:
            continue
        if idx == len(linear_transform_modules) - 1 and not prune_last_layer:
            continue

        target_layers[name] = module

    return target_layers
