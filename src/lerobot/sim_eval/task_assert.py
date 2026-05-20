from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from .common import WalkerS2sim
from .terminal import check_step_terminal, task1_check_parts_out_of_workspace, task2_check_all_parts_lost, task3_check_terminal, task4_check_box_poses_terminal
from .scoring import (
    task1_check_parts_in_box,
    task1_check_parts_in_lift,
    task1_time_out_check,
    task2_check_parts_grabbed,
    task2_check_parts_in_correct_bin,
    task2_calculate_total_score,
    task3_calculate_specific_score,
    task3_time_score,
    task4_check_box_joints_success,
    task4_check_short_edge_close_score,
    task4_check_long_edge_close_score,
    task4_time_score,
    task4_get_bimanual_collaboration_factor,
    task4_calculate_total_score,
)
import logging

logger = logging.getLogger(__name__)


class TaskAssertion:
    """
    断言基类。子类实现 __call__，返回:
        (is_success, terminal, reason, metrics)
    - is_success: 任务是否完成
    - terminal:   是否触发终止（超时或不可恢复的失败）
    """
    def __call__(
        self,
        robot: WalkerS2sim,
        step: int,
        action: np.ndarray | torch.Tensor | None = None,
    ) -> tuple[bool, bool, str, dict]:
        raise NotImplementedError
    
    
class Task4EpisodeActionTracker:
    """记录 episode 内每一步 action，并统计双臂同时运动情况。"""

    def __init__(self):
        self.action_list: list[np.ndarray] = []

    def reset(self) -> None:
        self.action_list.clear()

    def add_action(self, action: np.ndarray | torch.Tensor | None) -> None:
        if action is None:
            return
        if isinstance(action, torch.Tensor):
            arr = action.detach().float().cpu().numpy().reshape(-1)
        else:
            arr = np.asarray(action, dtype=np.float32).reshape(-1)
        self.action_list.append(arr)

    def get_bimanual_collaboration_stats(
        self,
        movement_eps: float = 1e-3,
        co_move_ratio_threshold: float = 0.1,
    ) -> dict[str, Any]:
        """
        通过相邻 step 的 action 变化量判断“同时运动”：
        - 左臂: action[:7]
        - 右臂: action[7:14]
        """
        if len(self.action_list) < 2:
            return {
                "is_bimanual_collaboration": False,
                "co_move_steps": 0,
                "active_steps": 0,
                "co_move_ratio": 0.0,
                "left_only_steps": 0,
                "right_only_steps": 0,
                "valid_action_steps": len(self.action_list),
            }

        co_move_steps = 0
        active_steps = 0
        left_only_steps = 0
        right_only_steps = 0

        for idx in range(1, len(self.action_list)):
            prev = self.action_list[idx - 1]
            cur = self.action_list[idx]
            if prev.size < 14 or cur.size < 14:
                continue

            delta = np.abs(cur[:14] - prev[:14])
            left_move = bool(np.any(delta[:7] > movement_eps))
            right_move = bool(np.any(delta[7:14] > movement_eps))

            if left_move or right_move:
                active_steps += 1
            if left_move and right_move:
                co_move_steps += 1
            elif left_move:
                left_only_steps += 1
            elif right_move:
                right_only_steps += 1

        co_move_ratio = float(co_move_steps / active_steps) if active_steps > 0 else 0.0
        is_bimanual_collaboration = bool(co_move_steps > 0 and co_move_ratio >= co_move_ratio_threshold)

        return {
            "is_bimanual_collaboration": is_bimanual_collaboration,
            "co_move_steps": int(co_move_steps),
            "active_steps": int(active_steps),
            "co_move_ratio": float(co_move_ratio),
            "left_only_steps": int(left_only_steps),
            "right_only_steps": int(right_only_steps),
            "valid_action_steps": int(len(self.action_list)),
        }
        
        
class Taks1EpisodePartsTracker:
    """记录 episode 内每一步零件的位姿，并统计是否有零件掉落（task1 以外的其他任务可复用该类进行零件跟踪）。"""

    def __init__(self):
        self.parts_poses_list: list[dict[str, Any]] = []
        # 记录每个零件的 XYZ 坐标轨迹，格式：{prim_path: [[x1,y1,z1], [x2,y2,z2], ...]}
        self.parts_trajectory: dict[str, list[list[float]]] = {}

    def reset(self) -> None:
        self.parts_poses_list.clear()
        self.parts_trajectory.clear()

    def add_parts_poses(self, parts_poses_dict: dict[str, Any]) -> None:
        """添加一步的零件位姿数据，并更新轨迹"""
        self.parts_poses_list.append(parts_poses_dict)

        # 更新每个零件的轨迹
        for part_info in parts_poses_dict:
            prim_path = part_info['prim_path']
            position = part_info['position']  # [x, y, z]
            if prim_path not in self.parts_trajectory:
                self.parts_trajectory[prim_path] = []
            self.parts_trajectory[prim_path].append(position)

    def check_parts_moved(self, movement_threshold: float = 0.001) -> dict[str, bool]:
        """检查每个零件是否有运动

        Args:
            movement_threshold: 最小运动阈值（米），默认 1mm

        Returns:
            dict: {prim_path: True/False}，True 表示有运动
        """
        moved_result = {}
        for prim_path, trajectory in self.parts_trajectory.items():
            if len(trajectory) < 2:
                moved_result[prim_path] = False
                continue

            # 计算起点到终点的位移
            start_pos = trajectory[0]
            end_pos = trajectory[-1]
            displacement = np.sqrt(
                (end_pos[0] - start_pos[0])**2 +
                (end_pos[1] - start_pos[1])**2 +
                (end_pos[2] - start_pos[2])**2
            )
            moved_result[prim_path] = displacement > movement_threshold

        return moved_result

    def get_all_parts_moved(self, movement_threshold: float = 0.001) -> bool:
        """检查是否所有零件都有运动

        Returns:
            bool: True 表示所有零件都有运动
        """
        moved_result = self.check_parts_moved(movement_threshold)
        return all(moved_result.values()) if moved_result else False

class NoOpAssertion(TaskAssertion):
    """不做任何判断，仅靠超时终止 episode。适用于 task3 暂无判据的情况。"""
    def __call__(
        self,
        robot: WalkerS2sim,
        step: int,
        action: np.ndarray | torch.Tensor | None = None,
        extra_info: any = None,
    ) -> tuple[bool, bool, str, dict]:
        terminal = check_step_terminal(step, robot.config.task_name.upper())
        reason = "达到最大步数" if terminal else ""
        return False, terminal, reason, {}

class Task1Assertion(TaskAssertion):
    """
    Task1（抓取-放置）成功判据：
    检查零件是否已进入目标箱体区域。
    """
    def __init__(
        self,
        lift_height: float,
        workspace_limits: dict[str, tuple[float, float]],
        box_half_size: tuple[float, float, float],
        lift_score_per_part: int,
        box_score_per_part: int,
        max_parts: int,
        time_full_score: int,
        time_full_time_seconds: float,
        time_penalty_interval_seconds: float,
        time_penalty_per_interval: int,
        success_score_threshold: int,
        parts_movement_threshold: float,
    ):
        self._lift_height = float(lift_height)
        self._workspace_limits = workspace_limits
        if len(box_half_size) != 3:
            raise ValueError("task1_box_half_size 必须包含 3 个元素: [x, y, z]")
        self._box_half_size: tuple[float, float, float] = (
            float(box_half_size[0]),
            float(box_half_size[1]),
            float(box_half_size[2]),
        )
        self._lift_score_per_part = int(lift_score_per_part)
        self._box_score_per_part = int(box_score_per_part)
        self._max_parts = int(max_parts)
        self._time_full_score = int(time_full_score)
        self._time_full_time_seconds = float(time_full_time_seconds)
        self._time_penalty_interval_seconds = float(time_penalty_interval_seconds)
        self._time_penalty_per_interval = int(time_penalty_per_interval)
        self._success_score_threshold = int(success_score_threshold)
        self._parts_movement_threshold = float(parts_movement_threshold) 
        self._lift_scored_parts: set[str] = set()
        self._box_scored_parts: set[str] = set()
        self._episode_start_time: float = time.time()
        self._last_step: int = 0
        self._parts_tracker = Taks1EpisodePartsTracker()  # 

    def _reset_episode_state(self) -> None:
        self._lift_scored_parts.clear()
        self._box_scored_parts.clear()
        self._episode_start_time = time.time()
        self._parts_tracker.reset()  # 新增

    def __call__(
        self,
        robot: WalkerS2sim,
        step: int,
        action: np.ndarray | torch.Tensor | None = None,
        extra_info: any = None
        
    ) -> tuple[bool, bool, str, dict]:
        metrics: dict = {}
        try:
            # episode 从 step=1 开始，重置累计计分状态。
            if step <= 1 or step < self._last_step:
                self._reset_episode_state()
            self._last_step = step

            parts_poses = robot._scene_builder.get_parts_world_poses()

            # 记录零件轨迹（新增）
            self._parts_tracker.add_parts_poses(parts_poses)


            terminal_out_of_workspace = task1_check_parts_out_of_workspace(
                parts_poses,
                self._workspace_limits,
            )
            terminal_time_out = check_step_terminal(step, extra_info.get("max_steps", 1000))
            terminal_no_movement = False
            parts_moved = False
            terminal = terminal_out_of_workspace or terminal_time_out or terminal_no_movement

            # 初始化分数为 0
            box_score = 0
            lift_score = 0
            time_score = 0
            elapsed_seconds = 0.0

            elapsed_seconds = max(0.0, time.time() - self._episode_start_time)  
            box_pos, box_ori = robot._scene_builder.boxes.get_world_poses()
            box_score, self._box_scored_parts = task1_check_parts_in_box(
                box_poses=(np.asarray(box_pos).squeeze(), np.asarray(box_ori).squeeze()),
                parts_poses_dict=parts_poses,
                scored_parts=self._box_scored_parts,
                box_half_size=self._box_half_size,
                score_per_part=self._box_score_per_part,
                max_parts=self._max_parts,
            )
            lift_score, self._lift_scored_parts = task1_check_parts_in_lift(
                parts_poses_dict=parts_poses,
                threshold_height=self._lift_height,
                scored_parts=self._lift_scored_parts,
                score_per_part=self._lift_score_per_part,
                max_parts=self._max_parts,
            )

            # 基础分数（抬升 + 入箱）
            total_score = int(box_score + lift_score)

            # success 时额外计算时间分数
            is_success = total_score >= self._success_score_threshold
            if is_success:
                time_score = task1_time_out_check(
                    elapsed_seconds=elapsed_seconds,
                    full_score=self._time_full_score,
                    full_time_seconds=self._time_full_time_seconds,
                    penalty_interval_seconds=self._time_penalty_interval_seconds,
                    penalty_per_interval=self._time_penalty_per_interval,
                )
                total_score = int(total_score + time_score)

                # 检查零件是否有运动
                parts_moved = self._parts_tracker.get_all_parts_moved(self._parts_movement_threshold)
                terminal_no_movement = not parts_moved

            # 如果零件没有运动，分数清零
            if terminal_no_movement:
                total_score = 0

            metrics = {
                "task1_lift_score": int(lift_score),
                "task1_box_score": int(box_score),
                "task1_time_score": int(time_score),
                "task1_total_score": int(total_score),
                "task1_elapsed_seconds": float(elapsed_seconds),
                "task1_lift_scored_count": int(len(self._lift_scored_parts)),
                "task1_box_scored_count": int(len(self._box_scored_parts)),
                "task1_lift_height": float(self._lift_height),
                "task1_box_half_size": [float(v) for v in self._box_half_size],
                "task1_max_parts": int(self._max_parts),
                "task1_success_score_threshold": int(self._success_score_threshold),
                "is_success": bool(is_success),
                "terminal": bool(terminal),
                "task1_terminal_out_of_workspace": bool(terminal_out_of_workspace),
                "task1_terminal_time_out": bool(terminal_time_out),
                "task1_parts_moved": bool(parts_moved),
                "task1_terminal_no_movement": bool(terminal_no_movement),
            }

            if is_success:
                reason = f"Success!!!，总分达到{self._success_score_threshold}"
            elif terminal_no_movement:
                reason = "Terminal!!!，零件未发生运动"
            elif terminal_out_of_workspace:
                reason = "Terminal!!!，零件超出工作空间"
            elif terminal_time_out:
                reason = "Terminal!!!，达到最大步数时间限制"
            else:
                reason = "进行中"
            return is_success, terminal, reason, metrics

        except Exception as e:
            logger.warning(f"Task1 断言异常: {e}")
            return False, False, f"断言异常: {e}", metrics

class Task2Assertion(TaskAssertion):
    """
    Task2（传送带分拣）成功判据：
    - 基础分 = 抓取分(grab) + 分拣分(sort)，满分 200
    - 成功：基础分 ≥ 门槛（默认 150，即 75%，参考 Task1）
    - 最终得分 = 分拣分（官方标准，max 100）
    - 终止：超时 / 所有零件掉落传送带以下
    """
    def __init__(
        self,
        conveyor_limits: dict[str, tuple[float, float]],
        bin_half_size: tuple[float, float, float],
        grab_score_per_part: int,
        sort_score_per_part: int,
        max_parts: int,
        success_score_threshold: int,
    ):
        self._conveyor_limits = conveyor_limits
        if len(bin_half_size) != 3:
            raise ValueError("task2_bin_half_size 必须包含 3 个元素: [x, y, z]")
        self._bin_half_size: tuple[float, float, float] = (
            float(bin_half_size[0]),
            float(bin_half_size[1]),
            float(bin_half_size[2]),
        )
        self._grab_score_per_part = int(grab_score_per_part)
        self._sort_score_per_part = int(sort_score_per_part)
        self._max_parts = int(max_parts)
        self._success_score_threshold = int(success_score_threshold)
        self._grab_scored_parts: set[str] = set()
        self._sort_scored_parts: set[str] = set()
        self._last_step: int = 0

    def _reset_episode_state(self) -> None:
        self._grab_scored_parts.clear()
        self._sort_scored_parts.clear()

    def __call__(
        self,
        robot: WalkerS2sim,
        step: int,
        action: np.ndarray | torch.Tensor | None = None,
        extra_info: any = None
    ) -> tuple[bool, bool, str, dict]:
        metrics: dict = {}
        try:
            # 新 episode 从 step=1 开始，重置累计计分状态。
            if step <= 1 or step < self._last_step:
                self._reset_episode_state()
            self._last_step = step

            # 获取传送带上的工件位姿
            parts_poses = robot._scene_builder.get_parts_world_poses()

            # 左右料箱：SceneBuilder 中 box_position 与 XFormPrim 一一对应；Task2 场景沿世界 x 排布，
            bin_positions, bin_orientations = robot._scene_builder.boxes.get_world_poses()
            bin_positions = np.asarray(bin_positions)
            bin_orientations = np.asarray(bin_orientations)
            n_bins = int(bin_positions.shape[0])
            if n_bins != 2:
                raise ValueError(f"Task2 需要 2 个料箱，当前为 {n_bins} 个")
            li, ri = np.argsort(bin_positions[:, 0])[:2]
            li, ri = int(li), int(ri)
            left_bin = (bin_positions[li], bin_orientations[li])
            right_bin = (bin_positions[ri], bin_orientations[ri])

            # 检查抓取得分
            grab_score, self._grab_scored_parts, grab_details = task2_check_parts_grabbed(
                parts_poses_dict=parts_poses,
                conveyor_limits=self._conveyor_limits,
                scored_parts=self._grab_scored_parts,
                score_per_part=self._grab_score_per_part,
                max_parts=self._max_parts,
            )

            # 检查分拣得分
            sort_score, self._sort_scored_parts, sort_details = task2_check_parts_in_correct_bin(
                parts_poses_dict=parts_poses,
                left_bin_pose=left_bin,
                right_bin_pose=right_bin,
                bin_half_size=self._bin_half_size,
                scored_parts=self._sort_scored_parts,
                score_per_part=self._sort_score_per_part,
                max_parts=self._max_parts,
            )

            # 基础分 = 抓取分 + 分拣分（满分 200），参考 Task1 两段式设计
            base_score = int(grab_score + sort_score)
            # 最终得分 = 分拣分（官方标准，max 100），抓取分仅用于成功判定
            total_score = int(sort_score)

            is_success = base_score >= self._success_score_threshold

            # 终止条件：超时 / 所有零件掉落传送带以下
            max_steps = extra_info.get("max_steps", 1000) if extra_info else 1000
            terminal_time = check_step_terminal(step, int(max_steps))
            terminal_lost = task2_check_all_parts_lost(
                parts_poses_dict=parts_poses,
                conveyor_z_min=float(self._conveyor_limits["z"][0]),
            )
            terminal = terminal_time or terminal_lost

            metrics = {
                "task2_grab_score": int(grab_score),
                "task2_sort_score": int(sort_score),
                "task2_base_score": int(base_score),
                "task2_total_score": int(total_score),
                "task2_grab_scored_count": int(len(self._grab_scored_parts)),
                "task2_sort_scored_count": int(len(self._sort_scored_parts)),
                "task2_max_parts": int(self._max_parts),
                "task2_success_score_threshold": int(self._success_score_threshold),
                "task2_grab_details": grab_details,
                "task2_sort_details": sort_details,
                "task2_terminal_time": bool(terminal_time),
                "task2_terminal_lost": bool(terminal_lost),
                "is_success": bool(is_success),
                "terminal": bool(terminal),
            }

            if is_success:
                reason = f"Success！！！，基础分{base_score}达到门槛{self._success_score_threshold}（分拣分{total_score}）"
            elif terminal_lost:
                reason = "Terminal！！！，所有零件掉落传送带以下"
            elif terminal_time:
                reason = "Terminal！！！，达到最大步数时间限制"
            else:
                reason = "进行中"
            return is_success, terminal, reason, metrics

        except Exception as e:
            logger.warning(f"Task2 断言异常: {e}")
            return False, False, f"断言异常: {e}", metrics


class Task3Assertion(TaskAssertion):
    def __init__(
        self,
        foam_pos: list[float],
        workspace_limits: dict,
        dist_threshold: float,
        height_threshold: float,
        success_score_threshold: int
    ):
        self.workspace_limits = workspace_limits
        self.dist_threshold = dist_threshold
        self.height_threshold = height_threshold
        self.success_score_threshold = success_score_threshold
        
        self.foam_center = np.array(foam_pos)
        self._episode_start_time = time.time()
        self.max_insertion_score = 0
        self._last_step = 0

    def _reset_episode_state(self) -> None:
        """重置每个 Episode 的状态，防止时间与分数带入下一轮"""
        self._episode_start_time = time.time()
        self.max_insertion_score = 0

    def __call__(self, robot, step, action=None, extra_info=None, **kwargs):
        # 识别新 Episode 并重置状态
        if step <= 1 or step < self._last_step:
            self._reset_episode_state()
        self._last_step = step

        # 计算当前已用时间（移到最前面，确保任何时候都有值）
        elapsed = time.time() - self._episode_start_time

        # 60步（约2秒）内零件可能抖动，不做死亡判定
        is_grace_period = (step < 60)
        parts_poses = robot._scene_builder.get_parts_world_poses()
        terminal_time = check_step_terminal(step, extra_info.get("max_steps", 10000) if extra_info else 10000)
        
        # 💡 修正 1：使用 task3 专用的边界检查，防止瞬间死亡
        terminal_out = False
        reason_out = ""
        if not is_grace_period:
            terminal_out, reason_out = task3_check_terminal(
                robot, parts_poses, self.foam_center, self.workspace_limits
            )
        
        # 💡 修正 2：使用初始化的阈值，不使用硬编码
        insertion_score, scored_parts = task3_calculate_specific_score(
            parts_poses_dict=parts_poses,
            dist_threshold=self.dist_threshold, 
            height_threshold=self.height_threshold
        )
        
        # 记录本次 Episode 达到的最高分
        self.max_insertion_score = max(self.max_insertion_score, insertion_score)
        is_success = self.max_insertion_score >= self.success_score_threshold
        
        # 计算时间奖励分（仅在成功后计算）
        total_score = self.max_insertion_score
        if is_success:
            total_score += task3_time_score(elapsed)
        
        # 判定是否终止
        terminal = terminal_time or terminal_out or is_success
        
        reason = "进行中"
        if is_success: reason = "成功完成"
        elif terminal_time: reason = "超时"
        elif terminal_out: reason = reason_out or "出界(零件掉落)"
        elif is_grace_period: reason = "环境初始化中"

        # 💡 关键修正：返回 sim_eval.py 要求的 4 个参数
        metrics = {
            "insertion_score": insertion_score,
            "max_insertion_score": self.max_insertion_score,
            "total_score": total_score,
            "elapsed_time": elapsed
        }

        return is_success, terminal, reason, metrics

class Task4Assertion(TaskAssertion):
    """
    Task4（装箱）成功判据：
    - 成功：盒子关节位置达到目标角度
    - 终止：盒子位姿偏离初始位姿 或 超时
    """

    def __init__(
        self,
        short_edge_targets: tuple[float, float],
        long_edge_targets: tuple[float, float],
        short_edge_joint_indices: tuple[int, int],
        long_edge_joint_indices: tuple[int, int],
        joint_threshold: float,
        action_movement_eps: float,
        co_move_ratio_threshold: float,
        success_hold_steps: int,
        short_edge_score_per_edge: int,
        long_edge_score_per_edge: int,
        time_full_score: int,
        time_full_time_seconds: float,
        time_penalty_interval_seconds: float,
        time_penalty_per_interval: int,
        single_arm_factor: float,
        bimanual_factor: float,
        box_pose_position_threshold: float,
        box_pose_orientation_threshold: float,
    ):
        self._short_edge_targets = np.asarray(short_edge_targets, dtype=np.float32).reshape(2)
        self._long_edge_targets = np.asarray(long_edge_targets, dtype=np.float32).reshape(2)
        self._short_edge_joint_indices = short_edge_joint_indices
        self._long_edge_joint_indices = long_edge_joint_indices
        self._joint_threshold = float(joint_threshold)
        self._action_movement_eps = float(action_movement_eps)
        self._co_move_ratio_threshold = float(co_move_ratio_threshold)
        self._success_hold = success_hold_steps
        self._short_edge_score_per_edge = short_edge_score_per_edge
        self._long_edge_score_per_edge = long_edge_score_per_edge
        self._time_full_score = time_full_score
        self._time_full_time_seconds = time_full_time_seconds
        self._time_penalty_interval_seconds = time_penalty_interval_seconds
        self._time_penalty_per_interval = time_penalty_per_interval
        self._single_arm_factor = single_arm_factor
        self._bimanual_factor = bimanual_factor
        self._box_pose_position_threshold = box_pose_position_threshold
        self._box_pose_orientation_threshold = box_pose_orientation_threshold
        self._consecutive_success = 0
        self._episode_start_time = time.time()
        self._last_step: int = 0
        self._action_tracker = Task4EpisodeActionTracker()

    @property
    def target_box_joints(self) -> np.ndarray:
        return np.asarray([
            [
                float(self._long_edge_targets[0]),
                float(self._long_edge_targets[1]),
                float(self._short_edge_targets[0]),
                float(self._short_edge_targets[1]),
            ]
        ], dtype=np.float32)

    def _reset_episode_state(self) -> None:
        self._episode_start_time = time.time()
        self._consecutive_success = 0
        self._action_tracker.reset()

    def __call__(
        self,
        robot: WalkerS2sim,
        step: int,
        action: np.ndarray | torch.Tensor | None = None,
        extra_info: any = None,
    ) -> tuple[bool, bool, str, dict]:
        if step <= 1 or step < self._last_step:
            self._reset_episode_state()
        self._last_step = step
        self._action_tracker.add_action(action)

        # 终止条件1：盒子位姿偏离
        cur_pos, cur_ori = robot._scene_builder.box_articulation.get_world_poses()
        init_pos = robot._scene_builder._box_initial_world_pos
        init_ori = robot._scene_builder._box_initial_world_ori
        current_pose = np.concatenate([cur_pos.squeeze(), cur_ori.squeeze()])
        target_pose = np.concatenate([init_pos.squeeze(), init_ori.squeeze()])
        terminal_pose = task4_check_box_poses_terminal(
            current_pose,
            target_pose,
            position_threshold=self._box_pose_position_threshold,
            orientation_threshold=self._box_pose_orientation_threshold,
        )

        # 终止条件2：超时
        terminal_time = check_step_terminal(step, extra_info.get("max_steps", 1000))

        terminal = terminal_pose or terminal_time

        # 成功条件：盒子关节角达到目标
        cur_joints = robot.get_box_joints()
        is_raw_success, error_info = task4_check_box_joints_success(
            cur_joints,
            self.target_box_joints,
            threshold=self._joint_threshold,
        )

        if is_raw_success:
            self._consecutive_success += 1
        else:
            self._consecutive_success = 0

        # 连续成功达到要求才算真正成功，避免机器人一碰最后一个边就退出
        is_success = self._consecutive_success >= self._success_hold
        if is_success:
            terminal = True

        # 计算短边分数
        short_edge_score, short_info = task4_check_short_edge_close_score(
            current_box_joints=cur_joints,
            short_edge_targets=self._short_edge_targets,
            joint_indices=self._short_edge_joint_indices,
            threshold=self._joint_threshold,
            score_per_edge=self._short_edge_score_per_edge,
        )

        # 计算长边分数
        long_edge_score, long_info = task4_check_long_edge_close_score(
            current_box_joints=cur_joints,
            long_edge_targets=self._long_edge_targets,
            joint_indices=self._long_edge_joint_indices,
            threshold=self._joint_threshold,
            score_per_edge=self._long_edge_score_per_edge,
        )

        # 计算时间分数
        elapsed_seconds = max(0.0, time.time() - self._episode_start_time)
        time_score = task4_time_score(
            elapsed_seconds=elapsed_seconds,
            full_score=self._time_full_score,
            full_time_seconds=self._time_full_time_seconds,
            penalty_interval_seconds=self._time_penalty_interval_seconds,
            penalty_per_interval=self._time_penalty_per_interval,
        )

        # 计算双臂协作分数
        collaboration_stats = self._action_tracker.get_bimanual_collaboration_stats(
            movement_eps=self._action_movement_eps,
            co_move_ratio_threshold=self._co_move_ratio_threshold,
        )

        # 根据是否存在双臂协作调整总分
        collaboration_factor = task4_get_bimanual_collaboration_factor(
            is_bimanual_collaboration=bool(collaboration_stats.get("is_bimanual_collaboration", False)),
            single_arm_factor=self._single_arm_factor,
            bimanual_factor=self._bimanual_factor,
        )

        # 计算总分：时间分数仅在 is_success 为 true 时才计算，否则时间分数为 0
        # 短边分数和长边分数始终计算
        if is_success:
            score_info = task4_calculate_total_score(
                short_edge_score=short_edge_score,
                long_edge_score=long_edge_score,
                time_score=time_score,
                collaboration_factor=collaboration_factor,
            )
            total_score = int(score_info.get("final_score", 0))
            task4_time_score_val = int(time_score)
        else:
            time_score = 0
            score_info = task4_calculate_total_score(
                short_edge_score=short_edge_score,
                long_edge_score=long_edge_score,
                time_score=time_score,
                collaboration_factor=collaboration_factor,
            )
            total_score = int(score_info.get("final_score", 0))
            task4_time_score_val = int(time_score)

        # 构造详细的评估指标，便于后续分析
        metrics = {
            "current_box_joints": cur_joints.tolist() if hasattr(cur_joints, "tolist") else cur_joints,
            "target_box_joints": self.target_box_joints.tolist(),
            "max_error": error_info.get("max_error"),
            "mean_error": error_info.get("mean_error"),
            "all_errors": error_info.get("all_errors"),
            "task4_short_edge_score": int(short_edge_score),
            "task4_long_edge_score": int(long_edge_score),
            "task4_time_score": task4_time_score_val,
            "task4_raw_score": int(score_info.get("raw_score", 0)),
            "task4_total_score": int(total_score),
            "task4_collaboration_factor": float(score_info.get("collaboration_factor", self._single_arm_factor)) if is_success else float(self._single_arm_factor),
            "task4_short_edge_closed": short_info.get("short_edge_closed", [False, False]),
            "task4_short_edge_errors": short_info.get("short_edge_errors", [float("inf"), float("inf")]),
            "task4_long_edge_closed": long_info.get("long_edge_closed", [False, False]),
            "task4_long_edge_errors": long_info.get("long_edge_errors", [float("inf"), float("inf")]),
            "task4_elapsed_seconds": float(elapsed_seconds),
            "task4_is_bimanual_collaboration": bool(collaboration_stats.get("is_bimanual_collaboration", False)),
            "task4_co_move_steps": int(collaboration_stats.get("co_move_steps", 0)),
            "task4_active_steps": int(collaboration_stats.get("active_steps", 0)),
            "task4_co_move_ratio": float(collaboration_stats.get("co_move_ratio", 0.0)),
            "task4_left_only_steps": int(collaboration_stats.get("left_only_steps", 0)),
            "task4_right_only_steps": int(collaboration_stats.get("right_only_steps", 0)),
            "task4_action_steps": int(collaboration_stats.get("valid_action_steps", 0)),
            "is_success": is_success,
            "terminal_pose": terminal_pose,
            "terminal_time": terminal_time,
            "terminal": terminal,
            "total_score": int(total_score),
        }

        reason = (
            "盒子关节达到目标，装箱成功" if is_success
            else ("盒子位姿偏离" if terminal_pose else ("超时" if terminal_time else "进行中"))
        )
        return is_success, terminal, reason, metrics


# ─────────────────────────────────────────────
# Task Assertion Registry
# ─────────────────────────────────────────────
TASK_ASSERTION_REGISTRY: dict[str, type[TaskAssertion]] = {
    "task1": Task1Assertion,
    "task2": Task2Assertion,
    "task3": Task3Assertion,
    "task4": Task4Assertion,
}


def create_task_assertion(task: str, args) -> TaskAssertion:
    """根据任务名和配置参数创建对应的断言实例。"""
    task = task.lower()

    if task == "task1":
        workspace_limits = getattr(args, "task1_workspace_limits")
        if not isinstance(workspace_limits, dict):
            raise ValueError("task1_workspace_limits 必须是包含 x/y/z 键的字典")
        return Task1Assertion(
            lift_height=float(getattr(args, "task1_lift_height")),
            workspace_limits={
                "x": (float(workspace_limits["x"][0]), float(workspace_limits["x"][1])),
                "y": (float(workspace_limits["y"][0]), float(workspace_limits["y"][1])),
                "z": (float(workspace_limits["z"][0]), float(workspace_limits["z"][1])),
            },
            box_half_size=tuple(getattr(args, "task1_box_half_size")),
            lift_score_per_part=int(getattr(args, "task1_lift_score_per_part")),
            box_score_per_part=int(getattr(args, "task1_box_score_per_part")),
            max_parts=int(getattr(args, "task1_max_parts")),
            time_full_score=int(getattr(args, "task1_time_full_score")),
            time_full_time_seconds=float(getattr(args, "task1_time_full_time_seconds")),
            time_penalty_interval_seconds=float(getattr(args, "task1_time_penalty_interval_seconds")),
            time_penalty_per_interval=int(getattr(args, "task1_time_penalty_per_interval")),
            success_score_threshold=int(getattr(args, "task1_success_score_threshold")),
            parts_movement_threshold=float(getattr(args, "task1_parts_movement_threshold")),
        )

    elif task == "task4":
        return Task4Assertion(
            short_edge_targets=tuple(getattr(args, "task4_short_targets")),
            long_edge_targets=tuple(getattr(args, "task4_long_targets")),
            short_edge_joint_indices=tuple(getattr(args, "task4_short_edge_joint_indices")),
            long_edge_joint_indices=tuple(getattr(args, "task4_long_edge_joint_indices")),
            joint_threshold=float(getattr(args, "task4_joint_threshold")),
            action_movement_eps=float(getattr(args, "task4_action_movement_eps")),
            co_move_ratio_threshold=float(getattr(args, "task4_co_move_ratio_threshold")),
            success_hold_steps=int(getattr(args, "task4_success_hold_steps")),
            short_edge_score_per_edge=int(getattr(args, "task4_short_edge_score_per_edge")),
            long_edge_score_per_edge=int(getattr(args, "task4_long_edge_score_per_edge")),
            time_full_score=int(getattr(args, "task4_time_full_score")),
            time_full_time_seconds=float(getattr(args, "task4_time_full_time_seconds")),
            time_penalty_interval_seconds=float(getattr(args, "task4_time_penalty_interval_seconds")),
            time_penalty_per_interval=int(getattr(args, "task4_time_penalty_per_interval")),
            single_arm_factor=float(getattr(args, "task4_single_arm_factor")),
            bimanual_factor=float(getattr(args, "task4_bimanual_factor")),
            box_pose_position_threshold=float(getattr(args, "task4_box_pose_position_threshold")),
            box_pose_orientation_threshold=float(getattr(args, "task4_box_pose_orientation_threshold")),
        )

    elif task == "task2":
        conveyor_limits = getattr(args, "task2_conveyor_limits")
        if not isinstance(conveyor_limits, dict):
            raise ValueError("task2_conveyor_limits 必须是包含 x/y/z 键的字典")
        for axis in ["x", "y", "z"]:
            if axis in conveyor_limits:
                conveyor_limits[axis] = [float(conveyor_limits[axis][0]), float(conveyor_limits[axis][1])]
        return Task2Assertion(
            conveyor_limits={
                "x": (float(conveyor_limits["x"][0]), float(conveyor_limits["x"][1])),
                "y": (float(conveyor_limits["y"][0]), float(conveyor_limits["y"][1])),
                "z": (float(conveyor_limits["z"][0]), float(conveyor_limits["z"][1])),
            },
            bin_half_size=tuple(getattr(args, "task2_bin_half_size")),
            grab_score_per_part=int(getattr(args, "task2_grab_score_per_part")),
            sort_score_per_part=int(getattr(args, "task2_sort_score_per_part")),
            max_parts=int(getattr(args, "task2_max_parts")),
            success_score_threshold=int(getattr(args, "task2_success_score_threshold")),
        )

    elif task == "task3":
        return Task3Assertion(
            foam_pos=list(getattr(args, "task3_foam_pos")),
            workspace_limits=getattr(args, "task3_workspace_limits"),
            dist_threshold=float(getattr(args, "task3_dist_threshold")),
            height_threshold=float(getattr(args, "task3_height_threshold")),
            success_score_threshold=int(getattr(args, "task3_success_score_threshold")),
        )

    else:
        raise ValueError(f"暂不支持的任务类型：{task}")
