import os, csv, scipy.io
import time
from pathlib import Path
from typing import List, Optional, Type

import numpy as np
import numpy.typing as npt
import shapely
import torch
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import (
    DetectionsTracks,
    Observation,
)
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner,
    PlannerInitialization,
    PlannerInput,
    PlannerReport,
)
from nuplan.planning.simulation.planner.planner_report import MLPlannerReport
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType
from scipy.special import softmax

from src.feature_builders.nuplan_scenario_render import NuplanScenarioRender

from ..post_processing.emergency_brake import EmergencyBrake
from ..post_processing.trajectory_evaluator import TrajectoryEvaluator
from ..scenario_manager.scenario_manager import ScenarioManager
from .ml_planner_utils import global_trajectory_to_states, load_checkpoint


class PlutoPlanner(AbstractPlanner):
    requires_scenario: bool = True

    def __init__(
        self,
        planner: TorchModuleWrapper,
        scenario: AbstractScenario = None,
        planner_ckpt: str = None,
        render: bool = False,
        use_gpu=True,
        save_dir=None,
        candidate_subsample_ratio: int = 0.5,
        candidate_min_num: int = 1,
        candidate_max_num: int = 20,
        eval_dt: float = 0.1,
        eval_num_frames: int = 80,
        learning_based_score_weight: float = 0.25,
        use_prediction: bool = True,
    ) -> None:
        """
        Initializes the ML planner class.
        :param model: Model to use for inference.
        """
        self._render = render
        self._imgs = []
        self._scenario = scenario
        if use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self._use_prediction = use_prediction

        self._planner = planner
        self._planner_feature_builder = planner.get_list_of_required_feature()[0]
        self._planner_ckpt = planner_ckpt

        self._initialization: Optional[PlannerInitialization] = None
        self._scenario_manager: Optional[ScenarioManager] = None

        self._future_horizon = 8.0
        self._step_interval = 0.1
        self._eval_dt = eval_dt
        self._eval_num_frames = eval_num_frames
        self._candidate_subsample_ratio = candidate_subsample_ratio
        self._candidate_min_num = candidate_min_num
        self._topk = candidate_max_num

        # Runtime stats for the MLPlannerReport
        self._feature_building_runtimes: List[float] = []
        self._inference_runtimes: List[float] = []

        self._scenario_type = scenario.scenario_type

        # post-processing
        self._trajectory_evaluator = TrajectoryEvaluator(eval_dt, eval_num_frames)
        self._emergency_brake = EmergencyBrake()
        self._learning_based_score_weight = learning_based_score_weight

        self._timestamp = 0
        self.mat_data = {}

        self.log_dir = Path(save_dir + "/log")
        if render:
            self._scene_render = NuplanScenarioRender()
            if save_dir is not None:
                self.video_dir = Path(save_dir+"/video")
            else:
                self.video_dir = Path(os.getcwd())
            self.video_dir.mkdir(exist_ok=True, parents=True)

    @torch.no_grad()
    def _infer_model(self, features: FeaturesType) -> npt.NDArray[np.float32]:
        """
        Makes a single inference on a Pytorch/Torchscript model.

        :param features: dictionary of feature types
        :return: predicted trajectory poses as a numpy array
        """
        # Propagate model
        action = self._model_loader.infer(features)
        action = action[0].cpu().numpy()[0]

        return action.astype(np.float64)

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        torch.set_grad_enabled(False)

        #print("Model state_dict keys:", self._planner.state_dict().keys())
        #print("Checkpoint state_dict keys:", load_checkpoint(self._planner_ckpt).keys())
        
        if self._planner_ckpt is not None:
            self._planner.load_state_dict(load_checkpoint(self._planner_ckpt))

        self._planner.eval()
        self._planner = self._planner.to(self.device)

        self._initialization = initialization

        self._scenario_manager = ScenarioManager(
            map_api=initialization.map_api,
            ego_state=None,
            route_roadblocks_ids=initialization.route_roadblock_ids,
            radius=self._eval_dt * self._eval_num_frames * 60 / 4.0,
        )
        self._planner_feature_builder.scenario_manager = self._scenario_manager

        if self._render:
            self._scene_render.scenario_manager = self._scenario_manager

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def compute_planner_trajectory(
        self, current_input: PlannerInput
    ) -> AbstractTrajectory:
        """
        Infer relative trajectory poses from model and convert to absolute agent states wrapped in a trajectory.
        Inherited, see superclass.
        """
        start_time = time.perf_counter()
        self._feature_building_runtimes.append(time.perf_counter() - start_time)

        start_time = time.perf_counter()

        ego_state = current_input.history.ego_states[-1]
        self._scenario_manager.update_ego_state(ego_state)
        self._scenario_manager.update_drivable_area_map()

        planning_trajectory = self._run_planning_once(current_input)

        self._inference_runtimes.append(time.perf_counter() - start_time)

        return planning_trajectory

    def _run_planning_once(self, current_input: PlannerInput):
        # common metrics
        start = time.perf_counter()
        self._timestamp += 1 #self._step_interval
        risk_str = None

        # simulation
        ego_state = current_input.history.ego_states[-1]
        velocity = ego_state.dynamic_car_state.center_velocity_2d.magnitude()
        planner_feature = self._planner_feature_builder.get_features_from_simulation(
            current_input, self._initialization
        )
        planner_feature_torch = planner_feature.collate(
            [planner_feature.to_feature_tensor()]
        ).to_device(self.device)

        start_f = time.perf_counter()
        # plan
        out = self._planner.forward(planner_feature_torch.data)  # use forward of model

        end_f = time.perf_counter()
        forward = end_f - start_f

        # output
        candidate_trajectories = (
            out["candidate_trajectories"][0].cpu().numpy().astype(np.float64)
        )
        probability = out["probability"][0].cpu().numpy()
        # if total ensemble, give trajectories
        trajectories = None
        if "trajectories" in out:
            trajectories = [traj[0].cpu().numpy().astype(np.float64) for traj in out["trajectories"]]
            trajectories = np.stack(trajectories)

        predictions = None
        prediction = None
        MoE_prob = None
        if self._use_prediction:
            if "output_predictions" in out:
                predictions = []
                for pre in out["output_predictions"]:
                    predictions.append(pre[0].cpu().numpy())
                prediction = out["output_prediction"][0].cpu().numpy()
                predictions = np.stack(predictions)
            elif "output_prediction" in out:
                prediction = out["output_prediction"][0].cpu().numpy()

        if "probs" in out:
            MoE_prob = out['probs'][0].cpu().numpy()

        ref_free_trajectory = (
            (out["output_ref_free_trajectory"][0].cpu().numpy().astype(np.float64))
            if "output_ref_free_trajectory" in out
            else None
        )

        # variables:
        # candidate_trajectories: [traj1, traj2, ... ]
        # probability: [0.03, 0.2, ...]
        # ref_free_trajectory: [?]
        # trajectories: [trajs1, trajs2, ...] num_ensemble
        # predictions: [preds1, preds2, ...] num_ensemble
        # if pluto: candidate_trajectories, probability, prediction
        # if total_ensemble: ..., ..., prediction, predictions, trajectories
        # if prediction_ensemble: ..., ..., ..., ...
        # if MoE_ensemble: ..., ..., ..., predictions([pred1, ...]), todo

        # use probablity from the model as learning_based_score
        # trim top_k trajs: default
        candidate_trajectories, learning_based_score = self._trim_candidates(
            candidate_trajectories,
            probability,
            current_input.history.ego_states[-1],
            ref_free_trajectory,
        )
        # use trajs, traffic lights, agent, map, baseline to evaluate
        # a score based on rule (with no prediction)
        #rule_based_scores = []
        metrics = []
        #enrisk = None
        enrisk = cal_enrisk(trajectories, predictions, planner_feature.data, MoE_prob)

        rule_based_scores, weighted_metrics, multi_metrics, red_light = self._trajectory_evaluator.evaluate(
                candidate_trajectories=candidate_trajectories,
                init_ego_state=current_input.history.ego_states[-1],
                detections=current_input.history.observations[-1],
                traffic_light_data=current_input.traffic_light_data,
                agents_info=self._get_agent_info(
                    planner_feature.data, prediction, ego_state
                ),
                route_lane_dict=self._scenario_manager.get_route_lane_dicts(),
                drivable_area_map=self._scenario_manager.drivable_area_map,
                baseline_path=self._get_ego_baseline_path(
                    self._scenario_manager.get_cached_reference_lines(), ego_state
                ),
                enrisk=enrisk
            )
        #rule_based_scores.append(final_scores)
        # metrics.append(np.concatenate((weighted_metrics, multi_metrics), axis=0))
        #rule_based_scores = np.stack(rule_based_scores)
        # print(weighted_metrics, multi_metrics)
        metrics = np.concatenate((weighted_metrics, multi_metrics), axis=0)
        # final score is rulebased + learnbased * weight
        # best_ensemble_index = np.argmin(rule_based_scores, axis=0)
        # out = rule_based_scores[best_ensemble_index, np.arange(rule_based_scores.shape[1])]
        #print(out)
        #print(np.min(rule_based_scores, axis=0))
        final_scores = (
            # rule_based_scores[best_ensemble_index, np.arange(rule_based_scores.shape[1])]
                rule_based_scores
             + self._learning_based_score_weight * learning_based_score
        )

        best_candidate_idx = final_scores.argmax()

        #best_candtraj_metrics = metrics[best_ensemble_index[best_candidate_idx], 
        #                                    :,best_candidate_idx] # (7)
        # brake at the last
        trajectory = self._emergency_brake.brake_if_emergency(
            ego_state,
            self._trajectory_evaluator.time_to_at_fault_collision(best_candidate_idx),
            candidate_trajectories[best_candidate_idx],
        )
        if_emergency = True
        # no emergency then give the "best" traj
        if trajectory is None:
            if_emergency = False
            trajectory = candidate_trajectories[best_candidate_idx, 1:]
            trajectory = InterpolatedTrajectory(
                global_trajectory_to_states(
                    global_trajectory=trajectory,
                    ego_history=current_input.history.ego_states,
                    future_horizon=len(trajectory) * self._step_interval,
                    step_interval=self._step_interval,
                    include_ego_state=False,
                )
            )
        if red_light:
            if_emergency = False

    
        end = time.perf_counter()
        elapsed = end - start
        # print(planner_feature.data.keys()) # dict_keys(['current_state', 'agent', 'agent_tokens', 'static_objects', 'map', 'reference_line', 'origin', 'angle'])
        # print(np.stack(predictions).shape)  (num_ensemble, agents, time_steps, (x,y,h,..))
        risk_dict = {"emergency": if_emergency,
                     "traj_var": cal_val(candidate_trajectories),
                     "prob_var": np.var(learning_based_score),
                     "traj_chosed": best_candidate_idx,
                     "final_score": final_scores[best_candidate_idx],
                     #"rule_based_scores": rule_based_scores,
                     "final_scores": final_scores,
                     "step_time": elapsed,
                     "forward_time": forward,
                     "metric" : metrics,
                     "velocity": velocity,
                     # "metrics" : best_candtraj_metrics,
                     "ttc": self._trajectory_evaluator._ttc,
                     "weighted_ttc": self._trajectory_evaluator._weighted_ttc,
                     "scenario_type": self._scenario.scenario_type
        }
        if enrisk is not None:
            risk_dict["enrisk"] = enrisk
        if MoE_prob is not None:
            risk_dict["probs"] = MoE_prob

        risk_str = None
        f"""Candidate traj var: {np.array2string(risk_dict['traj_var'], precision=6)}
        Candidate traj prob var: {np.array2string(risk_dict['prob_var'], precision=6)}
        Traj var: {risk_dict["enrisk"]["trajs_var"] if "enrisk" in risk_dict and "trajs_var" in risk_dict["enrisk"] else "None"}
        Traj Chosed: {risk_dict['traj_chosed']}     Final Score: {np.array2string(risk_dict['final_score'], precision=6)}
        Predictions var: {np.array2string(risk_dict["enrisk"]['preds_var'], precision=3, suppress_small=True, max_line_width=np.inf) if "enrisk" in risk_dict and "preds_var" in risk_dict["enrisk"] else "None"}
        TTC: {self._trajectory_evaluator._ttc}
        TTC_w": {self._trajectory_evaluator._weighted_ttc}
        """
#Candidate traj rule-based score var: {np.array2string(np.var(risk_dict['rule_based_scores'], axis=0), precision=6, suppress_small=False)}
        self.log_data_to_mat(risk_dict)

        if self._render:
            self._imgs.append(
                self._scene_render.render_from_simulation(
                    current_input=current_input,
                    initialization=self._initialization,
                    route_roadblock_ids=self._scenario_manager.get_route_roadblock_ids(),
                    scenario=self._scenario,
                    iteration=current_input.iteration.index,
                    planning_trajectory=self._global_to_local(trajectory, ego_state),
                    candidate_trajectories=self._global_to_local(
                        candidate_trajectories[rule_based_scores > 0], ego_state
                    ),
                    candidate_index=best_candidate_idx,
                    predictions=[prediction],
                    return_img=True,
                    risk=risk_str,
                    enrisk=enrisk
                )
            )


        return trajectory
    
    def log_data_to_mat(self, data):
        """
        将传入的字典数据保存为 MATLAB 的 .mat 文件，支持追加数据。

        参数:
        data (dict): 要保存的数据，字典的值可以是单个值、列表或嵌套结构。
        filename (str): 保存数据的 .mat 文件名。
        """
        id = str.split(self._planner_ckpt, "/")[-1]
        filename = f"{self.log_dir}/{self._scenario.log_name}_{self._scenario.token}_{id}.mat"
        #if os.path.exists(filename) and False:
        #    mat_data = scipy.io.loadmat(filename)
        #else:
        #    mat_data = {}

        # 生成唯一的变量名
        var_name = f'{self._timestamp}'

        # 将新数据保存到.mat文件中
        self.mat_data[var_name] = data
        if self._timestamp > 145:
            scipy.io.savemat(filename, self.mat_data)

    def _trim_candidates(
        self,
        candidate_trajectories: np.ndarray,
        probability: np.ndarray,
        ego_state: EgoState,
        ref_free_trajectory: np.ndarray = None,
    ) -> npt.NDArray[np.float32]:
        """
        give self_topk traj according to probability and transfer to global axis
        candidate_trajectories: (n_ref, n_mode, 80, 3)
        probability: (n_ref, n_mode)
        T timesteps  C (x, y, heading)
        """
        if len(candidate_trajectories.shape) == 4:
            n_ref, n_mode, T, C = candidate_trajectories.shape
            candidate_trajectories = candidate_trajectories.reshape(-1, T, C)
            probability = probability.reshape(-1)  # n_ref * n_mode

        sorted_idx = np.argsort(-probability)
        sorted_candidate_trajectories = candidate_trajectories[sorted_idx][: self._topk]
        sorted_probability = probability[sorted_idx][: self._topk]
        sorted_probability = softmax(sorted_probability)

        if ref_free_trajectory is not None:
            sorted_candidate_trajectories = np.concatenate(
                [sorted_candidate_trajectories, ref_free_trajectory[None, ...]],
                axis=0,
            )
            sorted_probability = np.concatenate([sorted_probability, [0.25]], axis=0)

        # to global
        origin = ego_state.rear_axle.array
        angle = ego_state.rear_axle.heading
        rot_mat = np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        )
        sorted_candidate_trajectories[..., :2] = (
            np.matmul(sorted_candidate_trajectories[..., :2], rot_mat) + origin
        )
        sorted_candidate_trajectories[..., 2] += angle

        sorted_candidate_trajectories = np.concatenate(
            [sorted_candidate_trajectories[..., 0:1, :], sorted_candidate_trajectories],
            axis=-2,
        )

        return sorted_candidate_trajectories, sorted_probability

    def _get_agent_info(self, data, predictions, ego_state: EgoState):
        """
        predictions: (n_agent, 80, 2 or 3)
        """
        current_velocity = np.linalg.norm(data["agent"]["velocity"][1:, -1], axis=-1)
        current_state = np.concatenate(
            [data["agent"]["position"][1:, -1], data["agent"]["heading"][1:, -1, None]],
            axis=-1,
        )
        velocity = None

        if predictions is None:  # constant velocity
            timesteps = np.linspace(0.1, 8, 80).reshape(1, 80, 1)
            displacement = data["agent"]["velocity"][1:, None, -1] * timesteps
            positions = current_state[:, None, :2] + displacement
            angles = current_state[:, None, 2:3].repeat(80, axis=1)
            predictions = np.concatenate([positions, angles], axis=-1)
            predictions = np.concatenate([current_state[:, None], predictions], axis=1)
            velocity = current_velocity[:, None].repeat(81, axis=1)
        elif predictions.shape[-1] == 2:
            predictions = np.concatenate(
                [current_state[:, None, :2], predictions], axis=1
            )
            diff = predictions[:, 1:] - predictions[:, :-1]
            start_end_dist = np.linalg.norm(
                predictions[:, -1, :2] - predictions[:, 0, :2], axis=-1
            )
            near_stop_mask = start_end_dist < 1.0
            angle = np.arctan2(diff[..., 1], diff[..., 0])
            angle = np.concatenate([current_state[:, None, -1], angle], axis=1)
            angle = np.where(
                near_stop_mask[:, None], current_state[:, 2:3].repeat(81, axis=1), angle
            )
            predictions = np.concatenate(
                [predictions[..., :2], angle[..., None]], axis=-1
            )
        elif predictions.shape[-1] == 3:
            predictions = np.concatenate([current_state[:, None], predictions], axis=1)
        elif predictions.shape[-1] == 5:
            velocity = np.linalg.norm(predictions[..., 3:5], axis=-1)
            predictions = np.concatenate(
                [current_state[:, None], predictions[..., :3]], axis=1
            )
            velocity = np.concatenate([current_velocity[:, None], velocity], axis=-1)
        else:
            raise ValueError("Invalid prediction shape")

        # to global
        predictions_global = self._local_to_global(predictions, ego_state)

        if velocity is None:
            velocity = (
                np.linalg.norm(np.diff(predictions_global[..., :2], axis=-2), axis=-1)
                / 0.1
            )
            velocity = np.concatenate([current_velocity[..., None], velocity], axis=-1)

        return {
            "tokens": data["agent_tokens"][1:],
            "shape": data["agent"]["shape"][1:, -1],
            "category": data["agent"]["category"][1:],
            "velocity": velocity,
            "predictions": predictions_global,
        }

    def _get_ego_baseline_path(self, reference_lines, ego_state: EgoState):
        init_ref_points = np.array([r[0] for r in reference_lines], dtype=np.float64)

        init_distance = np.linalg.norm(
            init_ref_points[:, :2] - ego_state.rear_axle.array, axis=-1
        )
        nearest_idx = np.argmin(init_distance)
        reference_line = reference_lines[nearest_idx]
        baseline_path = shapely.LineString(reference_line[:, :2])

        return baseline_path

    def _local_to_global(self, local_trajectory: np.ndarray, ego_state: EgoState):
        origin = ego_state.rear_axle.array
        angle = ego_state.rear_axle.heading
        rot_mat = np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        )
        position = np.matmul(local_trajectory[..., :2], rot_mat) + origin
        heading = local_trajectory[..., 2] + angle

        return np.concatenate([position, heading[..., None]], axis=-1)

    def _global_to_local(self, global_trajectory: np.ndarray, ego_state: EgoState):
        if isinstance(global_trajectory, InterpolatedTrajectory):
            states: List[EgoState] = global_trajectory.get_sampled_trajectory()
            global_trajectory = np.stack(
                [
                    np.array(
                        [state.rear_axle.x, state.rear_axle.y, state.rear_axle.heading]
                    )
                    for state in states
                ],
                axis=0,
            )

        origin = ego_state.rear_axle.array
        angle = ego_state.rear_axle.heading
        rot_mat = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        position = np.matmul(global_trajectory[..., :2] - origin, rot_mat)
        heading = global_trajectory[..., 2] - angle

        return np.concatenate([position, heading[..., None]], axis=-1)

    def generate_planner_report(self, clear_stats: bool = True) -> PlannerReport:
        """Inherited, see superclass."""
        report = MLPlannerReport(
            compute_trajectory_runtimes=self._compute_trajectory_runtimes,
            feature_building_runtimes=self._feature_building_runtimes,
            inference_runtimes=self._inference_runtimes,
        )
        if clear_stats:
            self._compute_trajectory_runtimes: List[float] = []
            self._feature_building_runtimes = []
            self._inference_runtimes = []

        if self._render:
            import imageio
            id = str(self._planner_ckpt.split("/")[-1])
            imageio.mimsave(
                self.video_dir
                / f"{self._scenario.log_name}_{self._scenario.token}_{id}.mp4",
                self._imgs,
                fps=10,
            )
            print("\n video saved to ", self.video_dir / "video.mp4\n")

        return report


def cal_enrisk(trajectories, predictions, data, probs):
    #n, Na, Nt, 5 = predictions.shape
    #n, Nt, 3 = trajectories.shape
    # probs: normalized probabilities
    enrisk = {}
    if trajectories is not None:
        enrisk["trajs_var"] = cal_val(trajectories)
    
    if predictions is not None and  predictions.shape[0] > 1 and predictions.shape[1] > 1:
        if probs is None:
            enrisk["preds_var"] = np.mean(np.mean(np.var(predictions, axis=0), axis=2), axis=1)
        else:
            # 对每个智能体 Na 独立计算
            weighted_vars = []
            for na in range(predictions.shape[1]):
                # 获取当前智能体的 predictions [n, Nt, 5] 和 probs [n,]
                preds_na = predictions[:, na, :, :]  # shape [n, Nt, 5]
                probs_na = probs[na, :]              # shape [n,]
                
                # 计算加权方差 [Nt, 5]
                weighted_var_na = np.average(
                    (preds_na - np.mean(preds_na, axis=0))**2,
                    axis=0,
                    weights=probs_na
                )
                
                # 对时间和特征取平均 [1,]
                weighted_vars.append(np.mean(weighted_var_na))
            enrisk["preds_var"] = np.array(weighted_vars)  # shape [Na,]
        if len(data["agent_tokens"][1:]) > 0:
            enrisk["preds_risk"] = {}
            for idx, token in enumerate(data["agent_tokens"][1:]):
                enrisk["preds_risk"][token] = enrisk["preds_var"][idx]
    if len(enrisk) == 0:
        enrisk = None
    # print(enrisk)
    return enrisk


def cal_val(data):
    """
    give val: \sum_{i=0}^{m}\sum_{j=0}^{n}{(x_{ij}-x_{i mean})^2}
        data: [m, n]
    """
    assert data.shape[-1] > 1 and data.shape[-2] > 1, f"_cal_val: can't handle data shape{data.shape}" 
    data = data.reshape(-1, data.shape[-2], data.shape[-1])
    res = np.mean(np.var(data, axis=0))
    return res

def find_none(data, path=""):
    if isinstance(data, dict):
        for k, v in data.items():
            new_path = f"{path}.{k}" if path else k
            find_none(v, new_path)
    elif isinstance(data, np.ndarray) and data.dtype == object:
        for i, item in enumerate(data):
            find_none(item, f"{path}[{i}]")
    elif data is None:
        print(f"发现 None 值在路径: {path}")