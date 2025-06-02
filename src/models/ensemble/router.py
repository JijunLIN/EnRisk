import torch
import torch.nn as nn
import torch.nn.functional as F

class Router(nn.Module):
    """ 改进的路由网络（含熵正则化） """
    def __init__(self, input_dim, num_ensemble):
        super().__init__()
        self.gating = nn.Linear(input_dim, num_ensemble)
        # 初始化偏置使初始选择均匀
        nn.init.constant_(self.gating.bias, 1/num_ensemble)
        
    def forward(self, x):
        logits = self.gating(x)
        if self.training:  # 训练时添加Gumbel噪声
            logits = logits + torch.randn_like(logits) * 0.01
        return logits
    

class CapacityAwareRouter(nn.Module):
    def __init__(self, input_dim=128, num_ensemble=5, hidden_dim=256, capacity_factor=torch.inf):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),  
            #nn.Dropout(0.1),  # 防止过拟合
            nn.Linear(hidden_dim, num_ensemble)
        )
        self.temperature = nn.Parameter(torch.tensor(0.8))  # 初始温度
        self.capacity_factor = capacity_factor
        self.num_ensemble = num_ensemble
        
        # 更精细的初始化
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.constant_(self.net[-1].bias, 1/num_ensemble)
        
    def forward(self, x):
        # x形状: [bs, seq_len, 128]
        logits = self.net(x) / (self.temperature + 1e-8)
        
        if self.training:
            # 改进的容量感知机制
            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1)
                expert_load = probs.sum(dim=(0,1))
                capacity = self.capacity_factor * (x.shape[0] * x.shape[1]) / self.num_ensemble
                overload = F.relu(expert_load - capacity) / (capacity + 1e-6)
                
            # 添加自适应噪声
            noise_scale = torch.sigmoid(overload) * 0.2
            logits = logits + torch.randn_like(logits) * noise_scale
            
        return logits