from copy import deepcopy
import math
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType

import torch
import torch.nn as nn
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)
import torch.nn.functional as F

from src.feature_builders.pluto_feature_builder import PlutoFeatureBuilder

from src.models.pluto.layers.fourier_embedding import FourierEmbedding
from src.models.pluto.layers.transformer import TransformerEncoderLayer
from src.models.pluto.modules.agent_encoder import AgentEncoder
from src.models.pluto.modules.agent_predictor import AgentPredictor
from src.models.pluto.modules.map_encoder import MapEncoder
from src.models.pluto.modules.static_objects_encoder import StaticObjectsEncoder
from src.models.pluto.modules.planning_decoder import PlanningDecoder
from src.models.pluto.layers.mlp_layer import MLPLayer
from src.models.ensemble.router import Router, CapacityAwareRouter

# no meaning, required by nuplan
trajectory_sampling = TrajectorySampling(num_poses=8, time_horizon=8, interval_length=1)


class PlanningModelShared(TorchModuleWrapper):
    def __init__(
        self,
        dim=128,
        state_channel=6,
        polygon_channel=6,
        history_channel=9,
        history_steps=21,
        future_steps=80,
        encoder_depth=4,
        decoder_depth=4,
        drop_path=0.2,
        dropout=0.1,
        num_heads=8,
        num_modes=6,
        use_ego_history=False,
        state_attn_encoder=True,
        state_dropout=0.75,
        use_hidden_proj=False,
        cat_x=False,
        ref_free_traj=False,
        feature_builder: PlutoFeatureBuilder = PlutoFeatureBuilder(),
    ) -> None:
        super().__init__(
            feature_builders=[feature_builder],
            target_builders=[EgoTrajectoryTargetBuilder(trajectory_sampling)],
            future_trajectory_sampling=trajectory_sampling,
        )

        self.dim = dim
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.use_hidden_proj = use_hidden_proj
        self.num_modes = num_modes
        self.radius = feature_builder.radius
        self.ref_free_traj = ref_free_traj

        self.pos_emb = FourierEmbedding(3, dim, 64)

        self.agent_encoder = AgentEncoder(
            state_channel=state_channel,
            history_channel=history_channel,
            dim=dim,
            hist_steps=history_steps,
            drop_path=drop_path,
            use_ego_history=use_ego_history,
            state_attn_encoder=state_attn_encoder,
            state_dropout=state_dropout,
        )

        self.map_encoder = MapEncoder(
            dim=dim,
            polygon_channel=polygon_channel,
            use_lane_boundary=True,
        )

        self.static_objects_encoder = StaticObjectsEncoder(dim=dim)

        self.encoder_blocks = nn.ModuleList(
            TransformerEncoderLayer(dim=dim, num_heads=num_heads, drop_path=dp)
            for dp in [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        )
        self.norm = nn.LayerNorm(dim)

        self.planning_decoder = PlanningDecoder(
            num_mode=num_modes,
            decoder_depth=decoder_depth,
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=4,
            dropout=dropout,
            cat_x=cat_x,
            future_steps=future_steps,
        )

        if use_hidden_proj:
            self.hidden_proj = nn.Sequential(
                nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)
            )

        if self.ref_free_traj:
            self.ref_free_decoder = MLPLayer(dim, 2 * dim, future_steps * 4)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, data):
        agent_pos = data["agent"]["position"][:, :, self.history_steps - 1]
        agent_heading = data["agent"]["heading"][:, :, self.history_steps - 1]
        agent_mask = data["agent"]["valid_mask"][:, :, : self.history_steps]
        polygon_center = data["map"]["polygon_center"]
        polygon_mask = data["map"]["valid_mask"]

        bs, A = agent_pos.shape[0:2]

        position = torch.cat([agent_pos, polygon_center[..., :2]], dim=1)
        angle = torch.cat([agent_heading, polygon_center[..., 2]], dim=1)
        angle = (angle + math.pi) % (2 * math.pi) - math.pi
        pos = torch.cat([position, angle.unsqueeze(-1)], dim=-1)

        agent_key_padding = ~(agent_mask.any(-1))
        polygon_key_padding = ~(polygon_mask.any(-1))
        key_padding_mask = torch.cat([agent_key_padding, polygon_key_padding], dim=-1)

        x_agent = self.agent_encoder(data)
        x_polygon = self.map_encoder(data)
        x_static, static_pos, static_key_padding = self.static_objects_encoder(data)

        x = torch.cat([x_agent, x_polygon, x_static], dim=1)

        pos = torch.cat([pos, static_pos], dim=1)
        pos_embed = self.pos_emb(pos)

        key_padding_mask = torch.cat([key_padding_mask, static_key_padding], dim=-1)
        x = x + pos_embed

        for blk in self.encoder_blocks:
            x = blk(x, key_padding_mask=key_padding_mask, return_attn_weights=False)
        x = self.norm(x)

        

        ref_line_available = data["reference_line"]["position"].shape[1] > 0

        if ref_line_available:
            trajectory, probability = self.planning_decoder(
                data, {"enc_emb": x, "enc_key_padding_mask": key_padding_mask}
            )
        else:
            trajectory, probability = None, None

        out = {
            "trajectory": trajectory,
            "probability": probability,  # (bs, R, M)
            "x": x,
            "A": A,
            "agent_pos": agent_pos,
            "agent_heading": agent_heading
        }

        if self.use_hidden_proj:
            out["hidden"] = self.hidden_proj(x[:, 0])

        if self.ref_free_traj:
            ref_free_traj = self.ref_free_decoder(x[:, 0]).reshape(
                bs, self.future_steps, 4
            )
            out["ref_free_trajectory"] = ref_free_traj

        if not self.training:
            if self.ref_free_traj:
                ref_free_traj_angle = torch.arctan2(
                    ref_free_traj[..., 3], ref_free_traj[..., 2]
                )
                ref_free_traj = torch.cat(
                    [ref_free_traj[..., :2], ref_free_traj_angle.unsqueeze(-1)], dim=-1
                )
                out["output_ref_free_trajectory"] = ref_free_traj

            if trajectory is not None:
                r_padding_mask = ~data["reference_line"]["valid_mask"].any(-1)
                probability.masked_fill_(r_padding_mask.unsqueeze(-1), -1e6)

                angle = torch.atan2(trajectory[..., 3], trajectory[..., 2])
                out_trajectory = torch.cat(
                    [trajectory[..., :2], angle.unsqueeze(-1)], dim=-1
                )

                bs, R, M, T, _ = out_trajectory.shape
                flattened_probability = probability.reshape(bs, R * M)
                best_trajectory = out_trajectory.reshape(bs, R * M, T, -1)[
                    torch.arange(bs), flattened_probability.argmax(-1)
                ]

                out["output_trajectory"] = best_trajectory
                out["candidate_trajectories"] = out_trajectory
            else:
                out["output_trajectory"] = out["output_ref_free_trajectory"]
                out["probability"] = torch.zeros(1, 0, 0)
                out["candidate_trajectories"] = torch.zeros(
                    1, 0, 0, self.future_steps, 3
                )

        return out

class PlanningModelParrallel(TorchModuleWrapper):
    def __init__(
        self,
        dim=128,
        future_steps=80,
        feature_builder: PlutoFeatureBuilder = PlutoFeatureBuilder(),
    ) -> None:
        super().__init__(
            feature_builders=[feature_builder],
            target_builders=[EgoTrajectoryTargetBuilder(trajectory_sampling)],
            future_trajectory_sampling=trajectory_sampling,
        )

        self.dim = dim
        self.future_steps = future_steps


        self.agent_predictor = AgentPredictor(dim=dim, future_steps=future_steps)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
    
    def forward(self, x):
        prediction = self.agent_predictor(x)
        return prediction

class MoELayer(nn.Module):
    """ 稳定版MoE层（含梯度控制） """
    def __init__(self, dim, future_step, num_ensemble, feature_builder, k=2):
        super().__init__()
        self.num_ensemble = num_ensemble
        self.k = k
        self.experts = nn.ModuleList([PlanningModelParrallel(dim, future_step, feature_builder) for _ in range(num_ensemble)])
        self.router = CapacityAwareRouter(dim, num_ensemble)
        
        # 专家激活统计（用于负载均衡）
        self.register_buffer('expert_counts', torch.zeros(num_ensemble))
        
    def forward(self, x, A):
        # 1. 路由计算
        x_input = x[:, 1:A]
        router_logits = self.router(x_input)  # [bs, A-1, num_experts]
        probs = F.softmax(router_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, self.k, dim=-1)  # [bs, A-1, k]
        
        # 2. 更新专家激活统计（用于负载均衡损失）
        if self.training:
            expert_mask = F.one_hot(topk_indices, self.num_ensemble).float()
            self.expert_counts = 0.9 * self.expert_counts + 0.1 * expert_mask.sum(0).sum(0).sum(0)
        
        # 3. 稀疏激活 提前计算所有专家的输出（显存足够时）
        all_expert_outputs = torch.stack([expert(x_input) for expert in self.experts], dim=0)  # [num_experts, bs, A-1, 80, 6]
        all_expert_outputs = all_expert_outputs.permute(1, 2, 0, 3, 4)  # [bs, A-1, num_experts, 80, 6]

        # 4. 选择 topk 专家的输出
        selected_expert_outputs = torch.gather(
            all_expert_outputs,
            dim=2,
            index=topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 80, 6)
        )  # [bs, A-1, k, 80, 6]

        # 5. 加权求和
        weighted_outputs = topk_probs.unsqueeze(-1).unsqueeze(-1) * selected_expert_outputs  # [bs, A-1, k, 80, 6]
        output = weighted_outputs.sum(dim=2)  # [bs, A-1, 80, 6]

        
        # 4. 保存路由信息（用于梯度计算）
        self.last_router_logits = router_logits
        self.last_topk_indices = topk_indices

        return output, probs, all_expert_outputs
    
    def load_balancing_loss(self):
        """ 负载均衡损失 """
        if not self.training:
            return 0.0
        
        # 计算专家选择分布的熵
        expert_freq = self.expert_counts / (self.expert_counts.sum() + 1e-6)
        entropy = -torch.sum(expert_freq * torch.log(expert_freq + 1e-6))
        
        # 添加路由决策的熵正则化
        router_probs = F.softmax(self.last_router_logits, dim=-1)
        router_entropy = -torch.mean(torch.sum(router_probs * torch.log(router_probs + 1e-6), dim=-1))
        return -0.1 * entropy + 0.01 * router_entropy, expert_freq
    
    def apply_ghost_gradients(self, optimizer):
        """ 处理未激活专家的梯度（Ghost Gradients技术） """
        active_experts = set(self.last_topk_indices.unique().tolist())
        
        for expert_idx in range(self.num_ensemble):
            if expert_idx not in active_experts:
                # 为该专家的参数创建虚拟梯度
                for param in self.experts[expert_idx].parameters():
                    if param.grad is None:
                        param.grad = torch.randn_like(param) * 0.001  # 小噪声梯度


class PredictionMoEModel(TorchModuleWrapper):
    def __init__(
        self,
        dim=128,
        state_channel=6,
        polygon_channel=6,
        history_channel=9,
        history_steps=21,
        future_steps=80,
        encoder_depth=4,
        decoder_depth=4,
        drop_path=0.2,
        dropout=0.1,
        num_heads=8,
        num_modes=6,
        use_ego_history=False,
        state_attn_encoder=True,
        state_dropout=0.75,
        use_hidden_proj=False,
        cat_x=False,
        ref_free_traj=False,
        feature_builder: PlutoFeatureBuilder = PlutoFeatureBuilder(),
        num_ensemble=5,
        k=2,
    ) -> None:
        super().__init__(
            feature_builders=[feature_builder],
            target_builders=[EgoTrajectoryTargetBuilder(trajectory_sampling)],
            future_trajectory_sampling=trajectory_sampling,
        )
        self.history_steps = history_steps
        self.num_ensemble = num_ensemble

        self.shared = PlanningModelShared(dim, state_channel, polygon_channel,
                                                history_channel, history_steps, future_steps,
                                                encoder_depth, decoder_depth, drop_path,
                                                dropout, num_heads, num_modes, use_ego_history,
                                                state_attn_encoder, state_dropout, 
                                                use_hidden_proj, cat_x, ref_free_traj,
                                                feature_builder)  # 共享模型
        self.parallel = MoELayer(dim, future_steps, num_ensemble,feature_builder, k)  # MoE层


    def forward(self, data): 
        out = self.shared(data)
        prediction, probs, all_expert_outputs = self.parallel(out["x"], out["A"])
        out["prediction"] = prediction
        out["probs"] = probs  # [bs, na, 5]
        # out["all_expert_outputs"] = all_expert_outputs # [bs, na, 5, 80, 6]
        all_expert_outputs = all_expert_outputs.permute(2, 0, 1, 3, 4)# [5, bs, na, 80, 6]

        if not self.training:
            agent_pos = data["agent"]["position"][:, :, self.history_steps - 1]
            agent_heading = data["agent"]["heading"][:, :, self.history_steps - 1]
            bs, A = agent_pos.shape[0:2]
            output_prediction = torch.cat(
                [
                    prediction[..., :2] + agent_pos[:, 1:A, None],
                    torch.atan2(prediction[..., 3], prediction[..., 2]).unsqueeze(-1)
                    + agent_heading[:, 1:A, None, None],
                    prediction[..., 4:6],
                ],
                dim=-1,
                )
            # output_predictions.append(output_prediction)  # (bs, A-1, T, 2)
            out["output_prediction"] = output_prediction
            output_predictions = []
            for i in range(self.num_ensemble):
                prediction = all_expert_outputs[i]
                #print(prediction.shape)
                output_prediction_item = torch.cat(
                    [
                        prediction[..., :2] + agent_pos[:, 1:A, None],
                        torch.atan2(prediction[..., 3], prediction[..., 2]).unsqueeze(-1)
                        + agent_heading[:, 1:A, None, None],
                        prediction[..., 4:6],
                    ],
                    dim=-1,
                    )
                # output_predictions.append(output_prediction)  # (bs, A-1, T, 2)
                output_predictions.append(output_prediction_item)
            out["output_predictions"] = output_predictions
        return out
    