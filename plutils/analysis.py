import torch
from torch import nn

from plutils.utils import is_linear_transform_layer


def get_num_params(model):
    if isinstance(model, nn.Module):
        ret = sum(
            [p.numel() for n, p in model.state_dict().items() if (p.dim() == 4 or p.dim() == 2) and 'weight' in n]
        )
    else:
        ret = sum([p.weight.data.numel() for p in model.values()])
    return ret


def get_sparsity(model: nn.Module):
    num_ones = 0
    num_total = 0
    for n, m in model.named_modules():
        if is_linear_transform_layer(m):
            if hasattr(m, 'mask'):
                num_ones += m.mask.sum().item()
            else:
                one_mask = (m.weight.data != 0).float()
                num_ones += one_mask.sum().item()

            num_total += m.weight.data.numel()

    density = num_ones / num_total
    sparsity = 1 - density
    return {'num_ones': num_ones, 'num_total': num_total, 'sparsity': sparsity}


def get_pr_over_kp(module: nn.Module, mask: torch.Tensor):
    if get_sparsity(module)['sparsity'] == 0:
        return 0
    else:
        kp_mask = mask
        pr_mask = 1 - kp_mask
        pr_weight_norm = torch.norm(pr_mask * module.weight.data) ** 2
        kp_weight_norm = torch.norm(kp_mask * module.weight.data) ** 2
        ret = torch.sqrt(pr_weight_norm / kp_weight_norm)
        return ret
