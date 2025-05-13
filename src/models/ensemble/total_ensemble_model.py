from copy import deepcopy
import math

import torch
import torch.nn as nn
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)

from src.feature_builders.pluto_feature_builder import PlutoFeatureBuilder

from src.models.pluto.pluto_model import PlanningModel

import numpy as np

# no meaning, required by nuplan
trajectory_sampling = TrajectorySampling(num_poses=8, time_horizon=8, interval_length=1)




class TotalEnsembleModel(TorchModuleWrapper):
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
    ) -> None:
        super().__init__(
            feature_builders=[feature_builder],
            target_builders=[EgoTrajectoryTargetBuilder(trajectory_sampling)],
            future_trajectory_sampling=trajectory_sampling,
        )
        self.num_ensemble = num_ensemble
        self.shared =  None
        self.parallel = nn.ModuleList([PlanningModel(dim, state_channel, polygon_channel,
                                                history_channel, history_steps, future_steps,
                                                encoder_depth, decoder_depth, drop_path,
                                                dropout, num_heads, num_modes, use_ego_history,
                                                state_attn_encoder, state_dropout, 
                                                use_hidden_proj, cat_x, ref_free_traj, feature_builder)
                                                for _ in range(num_ensemble)])
    
    def forward(self, data): # 只允许推理时调用
        exit()
        out = None
        reses = []
        for model in self.parallel:
            res = model(data)
            reses.append(res)
            if out is not None:
                for key, value in res:
                    out[key].append(value)
            else:
                out = {}
                for key, value in res:
                    out[key] = [value]

        for key, value in out:
            out[key] = torch.mean(torch.stack(value), dim=0)  #平均值方案作为推理主要输出
            
        if not self.training:
            assert "trajectories" not in out and "output_trajectory" in out
            out["trajectories"] = [o["output_trajectory"] for o in reses]
            out["output_predictions"] = [o["output_prediction"] for o in reses]
        return out