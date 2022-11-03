import numpy as np
import torch
import torch.nn as nn

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


def local_prune_model(targets, pruning_rate, block_policy, score_func):
    for name, module in targets.items():
        weight = module.weight.data
        blockdim = get_block_dim(block_policy, name, module)
        if isinstance(module, nn.Conv2d):
            cout, cin, hk, wk = weight.shape
            weight = weight.reshape(cout, cin * hk * wk)

        weight_blocks, num_blocks_row, num_blocks_col = matrix_to_blocks(weight, *blockdim)
        block_score, block_score_sep = score_func(weight_blocks)
        num_blocks_rm = int(num_blocks_row * num_blocks_col * pruning_rate)
        sorted_scores, _ = torch.sort(block_score)
        threshold = sorted_scores[num_blocks_rm]

        block_score = torch.sum(block_score_sep, dim=(-2, -1)) / (blockdim[0] * blockdim[1])
        block_mask = (block_score >= threshold).float()
        block_mask = block_mask.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(block_score_sep)
        mask = blocks_to_matrix(block_mask, num_blocks_row, num_blocks_col, blockdim[0], blockdim[1])
        if isinstance(module, nn.Conv2d):
            cout, cin, hk, wk = module.weight.data.shape
            mask = mask.reshape(cout, cin, hk, wk)

        module.mask = mask


def global_prune_model(targets, pruning_rate, total_param, block_policy, score_func):
    block_scores = []
    block_dim_map = []
    block_infos = {}
    for name, module in targets.items():
        weight = module.weight.data
        blockdim = get_block_dim(block_policy, name, module)
        if isinstance(module, nn.Conv2d):
            cout, cin, hk, wk = weight.shape
            weight = weight.reshape(cout, cin * hk * wk)

        weight_blocks, num_blocks_row, num_blocks_col = matrix_to_blocks(weight, *blockdim)
        block_score, block_score_sep = score_func(weight_blocks)
        block_scores.append(block_score)
        block_dim_map.append(blockdim[0] * blockdim[1] * torch.ones_like(block_score))
        block_infos[name] = (block_score_sep, num_blocks_row, num_blocks_col, blockdim)

    block_scores = torch.cat(block_scores)
    block_dim_map = torch.cat(block_dim_map)
    num_params_to_rm = int(total_param * pruning_rate)
    sorted_scores, sorted_indices = torch.sort(block_scores)
    param_cum = torch.cumsum(block_dim_map[sorted_indices], dim=0)
    cutoff_index = torch.where((num_params_to_rm < param_cum).float() == 1)[0][0]
    threshold = sorted_scores[cutoff_index]

    total_sparsity = 0
    for name, module in targets.items():
        block_score_sep, num_blocks_row, num_blocks_col, block_dim = block_infos[name]
        block_score = torch.sum(block_score_sep, dim=(-2, -1)) / (block_dim[0] * block_dim[1])
        block_mask = (block_score >= threshold).float()
        block_mask = block_mask.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(block_score_sep)
        mask = blocks_to_matrix(block_mask, num_blocks_row, num_blocks_col, *block_dim)
        if isinstance(module, nn.Conv2d):
            cout, cin, hk, wk = module.weight.data.shape
            mask = mask.reshape(cout, cin, hk, wk)

        layer_sparsity = 1 - mask.sum().item() / mask.numel()
        total_sparsity += module.weight.data.numel() / total_param * layer_sparsity
        module.mask = mask


def mag_score_func(weighted_blocks):
    block_score_sep = torch.sqrt(torch.square(weighted_blocks))
    _, blockrow, blockcol = weighted_blocks.shape
    block_score = torch.sum(block_score_sep, dim=(-2, -1)) / (blockrow * blockcol)
    return block_score, block_score_sep