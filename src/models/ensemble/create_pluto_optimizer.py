import torch
import torch.nn as nn
from src.optim.warmup_cos_lr import WarmupCosLR


def create_pluto_optimizer(model, weight_decay, lr, warmup_epochs, epochs):
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.MultiheadAttention,
        nn.LSTM,
        nn.GRU,
    )
    blacklist_weight_modules = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
        nn.LayerNorm,
        nn.Embedding,
    )
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters():
            full_param_name = (
                "%s.%s" % (module_name, param_name) if module_name else param_name
            )
            if "bias" in param_name:
                no_decay.add(full_param_name)
            elif "weight" in param_name:
                if isinstance(module, whitelist_weight_modules):
                    decay.add(full_param_name)
                elif isinstance(module, blacklist_weight_modules):
                    no_decay.add(full_param_name)
            elif not ("weight" in param_name or "bias" in param_name):
                no_decay.add(full_param_name)
    param_dict = {
        param_name: param for param_name, param in model.named_parameters()
    }
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0
    assert len(param_dict.keys() - union_params) == 0

    optim_groups = [
        {
            "params": [
                param_dict[param_name] for param_name in sorted(list(decay))
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                param_dict[param_name] for param_name in sorted(list(no_decay))
            ],
            "weight_decay": 0.0,
        },
    ]

    # Get optimizer
    optimizer = torch.optim.AdamW(
        optim_groups, lr=lr, weight_decay=weight_decay
    )

    # Get lr_scheduler
    scheduler = WarmupCosLR(
        optimizer=optimizer,
        lr=lr,
        min_lr=1e-6,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
    )
    return optimizer, scheduler