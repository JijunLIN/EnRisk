import torch
import torch.nn as nn


class Router(nn.Module):
    """ 改进的路由网络（含熵正则化） """
    def __init__(self, input_dim, num_ensemble):
        super().__init__()
        self.gating = nn.Linear(input_dim, num_ensemble)
        # 初始化偏置使初始选择均匀
        nn.init.constant_(self.gating.bias, 1/num_ensemble)
        
    def forward(self, x, A):
        logits = self.gating(x[:,1:A])
        if self.training:  # 训练时添加Gumbel噪声
            logits = logits + torch.randn_like(logits) * 0.01
        return logits