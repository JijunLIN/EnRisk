import logging
import os
from typing import Dict, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import (
    FeaturesType,
    ScenarioListType,
    TargetsType,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import MetricCollection
from src.metrics import MR, minADE, minFDE
from src.metrics.prediction_avg_ade import PredAvgADE
from src.metrics.prediction_avg_fde import PredAvgFDE
from src.optim.warmup_cos_lr import WarmupCosLR
from ..ensemble.create_pluto_optimizer import create_pluto_optimizer

from src.models.pluto.loss.esdf_collision_loss import ESDFCollisionLoss

logger = logging.getLogger(__name__)


class PredictionMoETrainer(pl.LightningModule):
    def __init__(
        self,
        model: TorchModuleWrapper,
        lr,
        weight_decay,
        epochs,
        warmup_epochs,
        use_collision_loss=True,
        use_contrast_loss=False,
        regulate_yaw=False,
        objective_aggregate_mode: str = "mean",
    ) -> None:
        """
        Initializes the class.

        :param model: pytorch model
        :param objectives: list of learning objectives used for supervision at each step
        :param metrics: list of planning metrics computed at each step
        :param batch_size: batch_size taken from dataloader config
        :param optimizer: config for instantiating optimizer. Can be 'None' for older models.
        :param lr_scheduler: config for instantiating lr_scheduler. Can be 'None' for older models and when an lr_scheduler is not being used.
        :param warm_up_lr_scheduler: config for instantiating warm up lr scheduler. Can be 'None' for older models and when a warm up lr_scheduler is not being used.
        :param objective_aggregate_mode: how should different objectives be combined, can be 'sum', 'mean', and 'max'.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.objective_aggregate_mode = objective_aggregate_mode
        self.history_steps = model.shared.history_steps
        self.use_collision_loss = use_collision_loss
        self.use_contrast_loss = use_contrast_loss
        self.regulate_yaw = regulate_yaw
        self.num_ensemble = self.model.num_ensemble

        self.radius = model.shared.radius
        self.num_modes = model.shared.num_modes
        self.mode_interval = self.radius / self.num_modes

        self.automatic_optimization = False

        if use_collision_loss:
            self.collision_loss = ESDFCollisionLoss()


    def on_fit_start(self) -> None:
        metrics_collection = MetricCollection(
            [
                minADE().to(self.device),
                minFDE().to(self.device),
                MR(miss_threshold=2).to(self.device),
                PredAvgADE().to(self.device),
                PredAvgFDE().to(self.device),
            ]
        )
        self.metrics = {
            "train": metrics_collection.clone(prefix="train/"),
            "val": metrics_collection.clone(prefix="val/"),
        }

    def _step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], prefix: str
    ) -> torch.Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.

        This is called either during training, validation or testing stage.

        :param batch: input batch consisting of features and targets
        :param prefix: prefix prepended at each artifact's name during logging
        :return: model's scalar loss
        """
        features, targets, scenarios = batch

        [opt_experts, opt_shared, opt_router] = self.optimizers()
        opt_experts.zero_grad()
        opt_shared.zero_grad()
        opt_router.zero_grad()
        
        res = self.model(features["feature"].data)

        losses = self._compute_objectives(res, features["feature"].data)
        metrics = self._compute_metrics(res, features["feature"].data, prefix)

        self._log_step(losses["loss"], losses, metrics, prefix)

        if self.training:
            moe_loss, expert_freq = self.model.parallel.load_balancing_loss()
            total_loss = losses["loss"] + moe_loss
            self.manual_backward(total_loss)
            opt_experts.step()
            opt_shared.step()
            opt_router.step()

            # 记录指标
            self.log_dict({
                'total_loss': total_loss,
                'moe_loss': moe_loss,
                'expert_util': len(self.model.parallel.last_topk_indices.unique()) / self.num_ensemble,
                'expert_freq_std': expert_freq.std(),
                'router/temp': self.model.parallel.router.temperature.data,
                'router/capacity_factor': self.model.parallel.router.capacity_factor
            }, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

    def on_train_epoch_end(self) -> None:
        for scheduler in self.lr_schedulers():
            scheduler.step()
        # 在前几个epoch禁用容量限制
        if self.current_epoch < self.warmup_epochs:
            self.model.parallel.router.capacity_factor = float('inf')
        else:
            self.model.parallel.router.capacity_factor = 1.2
        # 温度退火 (S型曲线)

        # 更平滑的温度退火
        current_ratio = self.current_epoch / self.trainer.max_epochs
        if current_ratio < 0.3:  # 前30%保持高温探索
            new_temp = 1.0
        elif current_ratio < 0.7:  # 30%-70%线性退火
            progress = (current_ratio - 0.3) / 0.4
            new_temp = 1.0 * (1 - progress) + 0.3 * progress
        else:  # 后30%保持低温
            new_temp = 0.3
            
        self.model.parallel.router.temperature.data.clamp_(max=new_temp)
        

    def _compute_objectives(self, res, data) -> Dict[str, torch.Tensor]:
        bs, _, T, _ = res["prediction"].shape

        if self.use_contrast_loss:
            train_num = (bs // 3) * 2 if self.training else bs
        else:
            train_num = bs

        trajectory, probability, prediction = (
            res["trajectory"][:train_num],
            res["probability"][:train_num],
            res["prediction"][:train_num],
        )
        ref_free_trajectory = res.get("ref_free_trajectory", None)

        targets_pos = data["agent"]["target"][:train_num]
        valid_mask = data["agent"]["valid_mask"][:train_num, :, -T:]
        targets_vel = data["agent"]["velocity"][:train_num, :, -T:]

        target = torch.cat(
            [
                targets_pos[..., :2],
                torch.stack(
                    [targets_pos[..., 2].cos(), targets_pos[..., 2].sin()], dim=-1
                ),
                targets_vel,
            ],
            dim=-1,
        )

        # planning loss
        ego_reg_loss, ego_cls_loss, collision_loss = self.get_planning_loss(
            data, trajectory, probability, valid_mask[:, 0], target[:, 0], train_num
        )
        if ref_free_trajectory is not None:
            ego_ref_free_reg_loss = F.smooth_l1_loss(
                ref_free_trajectory[:train_num],
                target[:, 0, :, : ref_free_trajectory.shape[-1]],
                reduction="none",
            ).sum(-1)
            ego_ref_free_reg_loss = (
                ego_ref_free_reg_loss * valid_mask[:, 0]
            ).sum() / valid_mask[:, 0].sum()
        else:
            ego_ref_free_reg_loss = ego_reg_loss.new_zeros(1)

        # prediction loss
        prediction_loss = self.get_prediction_loss(
            data, prediction, valid_mask[:, 1:], target[:, 1:]
        )

        if self.training and self.use_contrast_loss:
            contrastive_loss = self._compute_contrastive_loss(
                res["hidden"], data["data_n_valid_mask"]
            )
        else:
            contrastive_loss = prediction_loss.new_zeros(1)

        loss = (
            ego_reg_loss
            + ego_cls_loss
            + prediction_loss
            + contrastive_loss
            + collision_loss
            + ego_ref_free_reg_loss
        )

        return {
            "loss": loss,
            "reg_loss": ego_reg_loss.item(),
            "cls_loss": ego_cls_loss.item(),
            "ref_free_reg_loss": ego_ref_free_reg_loss.item(),
            "collision_loss": collision_loss.item(),
            "prediction_loss": prediction_loss.item(),
            "contrastive_loss": contrastive_loss.item(),
        }

    def get_prediction_loss(self, data, prediction, valid_mask, target):
        """
        prediction: (bs, A-1, T, 6)
        valid_mask: (bs, A-1, T)
        target: (bs, A-1, 6)
        """
        prediction_loss = F.smooth_l1_loss(
            prediction[valid_mask], target[valid_mask], reduction="none"
        ).sum(-1)
        prediction_loss = prediction_loss.sum() / valid_mask.sum()

        return prediction_loss

    def get_planning_loss(self, data, trajectory, probability, valid_mask, target, bs):
        """
        trajectory: (bs, R, M, T, 4)
        valid_mask: (bs, T)
        """
        num_valid_points = valid_mask.sum(-1)
        endpoint_index = (num_valid_points / 10).long().clamp_(min=0, max=7)  # max 8s
        r_padding_mask = ~data["reference_line"]["valid_mask"][:bs].any(-1)  # (bs, R)
        future_projection = data["reference_line"]["future_projection"][:bs][
            torch.arange(bs), :, endpoint_index
        ]

        target_r_index = torch.argmin(
            future_projection[..., 1] + 1e6 * r_padding_mask, dim=-1
        )
        target_m_index = (
            future_projection[torch.arange(bs), target_r_index, 0] / self.mode_interval
        ).long()
        target_m_index.clamp_(min=0, max=self.num_modes - 1)

        target_label = torch.zeros_like(probability)
        target_label[torch.arange(bs), target_r_index, target_m_index] = 1

        best_trajectory = trajectory[torch.arange(bs), target_r_index, target_m_index]

        if self.use_collision_loss:
            collision_loss = self.collision_loss(
                best_trajectory, data["cost_maps"][:bs, :, :, 0].float()
            )
        else:
            collision_loss = trajectory.new_zeros(1)

        reg_loss = F.smooth_l1_loss(best_trajectory, target, reduction="none").sum(-1)
        reg_loss = (reg_loss * valid_mask).sum() / valid_mask.sum()

        probability.masked_fill_(r_padding_mask.unsqueeze(-1), -1e6)

        cls_loss = F.cross_entropy(
            probability.reshape(bs, -1), target_label.reshape(bs, -1).detach()
        )

        if self.regulate_yaw:
            heading_vec_norm = torch.norm(best_trajectory[..., 2:4], dim=-1)
            yaw_regularization_loss = F.l1_loss(
                heading_vec_norm, heading_vec_norm.new_ones(heading_vec_norm.shape)
            )
            reg_loss += yaw_regularization_loss

        return reg_loss, cls_loss, collision_loss

    def _compute_contrastive_loss(
        self, hidden, valid_mask, normalize=True, tempreture=0.1
    ):
        """
        Compute triplet loss

        Args:
            hidden: (3*bs, D)
        """
        if normalize:
            hidden = F.normalize(hidden, dim=1, p=2)

        if not valid_mask.any():
            return hidden.new_zeros(1)

        x_a, x_p, x_n = hidden.chunk(3, dim=0)

        x_a = x_a[valid_mask]
        x_p = x_p[valid_mask]
        x_n = x_n[valid_mask]

        logits_ap = (x_a * x_p).sum(dim=1) / tempreture
        logits_an = (x_a * x_n).sum(dim=1) / tempreture
        labels = x_a.new_zeros(x_a.size(0)).long()

        triplet_contrastive_loss = F.cross_entropy(
            torch.stack([logits_ap, logits_an], dim=1), labels
        )
        return triplet_contrastive_loss

    def _compute_metrics(self, res, data, prefix) -> Dict[str, torch.Tensor]:
        """
        Computes a set of planning metrics given the model's predictions and targets.

        :param predictions: model's predictions
        :param targets: ground truth targets
        :return: dictionary of metrics names and values
        """
        # get top 6 modes
        trajectory, probability = res["trajectory"], res["probability"]
        r_padding_mask = ~data["reference_line"]["valid_mask"].any(-1)
        probability.masked_fill_(r_padding_mask.unsqueeze(-1), -1e6)

        bs, R, M, T, _ = trajectory.shape
        trajectory = trajectory.reshape(bs, R * M, T, -1)
        probability = probability.reshape(bs, R * M)
        top_k_prob, top_k_index = probability.topk(6, dim=-1)
        top_k_traj = trajectory[torch.arange(bs)[:, None], top_k_index]

        outputs = {
            "trajectory": top_k_traj[..., :2],
            "probability": top_k_prob,
            "prediction": res["prediction"][..., :2],
            "prediction_target": data["agent"]["target"][:, 1:],
            "valid_mask": data["agent"]["valid_mask"][:, 1:, self.history_steps :],
        }
        target = data["agent"]["target"][:, 0]

        metrics = self.metrics[prefix](outputs, target)
        return metrics

    def _log_step(
        self,
        loss,
        objectives: Dict[str, torch.Tensor],
        metrics: Dict[str, torch.Tensor],
        prefix: str,
        loss_name: str = "loss",
    ) -> None:
        """
        Logs the artifacts from a training/validation/test step.

        :param loss: scalar loss value
        :type objectives: [type]
        :param metrics: dictionary of metrics names and values
        :param prefix: prefix prepended at each artifact's name
        :param loss_name: name given to the loss for logging
        """
        self.log(
            f"loss/{prefix}_{loss_name}",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True if prefix == "train" else False,
        )

        for key, value in objectives.items():
            self.log(
                f"objectives/{prefix}_{key}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        if metrics is not None:
            self.log_dict(
                metrics,
                prog_bar=(prefix == "val"),
                on_step=False,
                on_epoch=True,
                batch_size=1,
                sync_dist=True,
            )

    def training_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during training.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "train")

    def validation_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during validation.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "val")

    def test_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during testing.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "test")

    def forward(self, features: FeaturesType) -> TargetsType:
        exit()
        """
        Propagates a batch of features through the model.

        :param features: features batch
        :return: model's predictions
        """
        return self.model(features)

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, _LRScheduler]]]:
        """
        Configures the optimizers and learning schedules for the training.

        :return: optimizer or dictionary of optimizers and schedules
        """
        all_optimizers = []
        all_schedulers = []
        opt, sdl = create_pluto_optimizer(self.model.parallel.experts, self.weight_decay, self.lr*0.8, self.warmup_epochs, self.epochs)
        all_optimizers.append(opt)
        all_schedulers.append(sdl)
        opt, sdl = create_pluto_optimizer(self.model.shared, self.weight_decay, self.lr, self.warmup_epochs, self.epochs)
        all_optimizers.append(opt)
        all_schedulers.append(sdl)
        opt, sdl = create_pluto_optimizer(self.model.parallel.router, self.weight_decay, self.lr*2, self.warmup_epochs, self.epochs)
        all_optimizers.append(opt)
        all_schedulers.append(sdl)
        return all_optimizers, all_schedulers

    # def on_before_optimizer_step(self, optimizer) -> None:
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print("unused param", name)
    def on_after_backward(self):
        """后向传播后的梯度处理"""
        # 梯度裁剪和归一化
        if not hasattr(self, '_grads_processed'):
            torch.nn.utils.clip_grad_norm_(self.model.shared.parameters(), max_norm=1.0)
            if hasattr(self.model, 'parallel'):
                self.model.parallel.apply_ghost_gradients(self.optimizers()[0])
                # 增强路由网络梯度
                for p in self.model.parallel.router.parameters():
                    if p.grad is not None:
                        p.grad = p.grad * 1.5
            self._grads_processed = True
