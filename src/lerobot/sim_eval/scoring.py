import numpy as np
from typing import Any, Iterable

from .common import PartPoseDict, iter_part_dicts, safe_vec3


def _extract_semantics(part: PartPoseDict) -> str:
    """兼容语义来源：优先 semantics，其次 prim_path。"""
    sem = str(part.get("semantics", "")).strip().upper()
    if sem in {"A", "B"}:
        return sem

    prim_path = str(part.get("prim_path", "")).upper()
    if "PART_A" in prim_path or prim_path.endswith("A"):
        return "A"
    if "PART_B" in prim_path or prim_path.endswith("B"):
        return "B"
    return "UNKNOWN"



# ----------------- Task1 评估工具函数 -----------------

def task1_check_parts_in_box(
    box_poses: tuple[np.ndarray, np.ndarray],
    parts_poses_dict: Any,
    scored_parts: Iterable[str] | None = None,
    box_half_size: tuple[float, float, float] = (0.095, 0.145, 0.25),
    score_per_part: int = 10,
    max_parts: int = 4,
) -> tuple[int, set[str]]:
    """
    Task1 入箱评分（40分满分，每个工件10分）：
    1. 零件在箱体空间内；
    2. 按 semantics(A/B) 落在正确类别区域；
    3. 一个工件只计分一次。
    """
    current_scored_parts: set[str] = set(scored_parts or [])
    try:
        box_pos = safe_vec3(box_poses[0] if isinstance(box_poses, tuple) and len(box_poses) > 0 else None)
        if box_pos is None:
            raise ValueError(f"task1_check_parts_in_box 异常: 无效的 box_poses 输入，无法获取箱体位置")

        parts = iter_part_dicts(parts_poses_dict)
        hx, hy, hz = float(box_half_size[0]), float(box_half_size[1]), float(box_half_size[2])

        for idx, part in enumerate(parts):
            part_id = str(part.get("prim_path", f"part_{idx}"))
            if part_id in current_scored_parts:
                continue

            pos = safe_vec3(part.get("position"))
            if pos is None:
                continue

            rel = pos - box_pos
            in_box = (abs(float(rel[0])) <= hx) and (abs(float(rel[1])) <= hy) and (abs(float(rel[2])) <= hz)
            if not in_box:
                continue

            sem = _extract_semantics(part)
            # 将箱体按 y 轴分成 A/B 两个类别区域：A 在负半轴，B 在正半轴。
            correct_category = (sem == "B" and rel[1] >= 0.0) or (sem == "A" and rel[1] < 0.0)
            if not correct_category: 
                continue

            current_scored_parts.add(part_id)
            if len(current_scored_parts) >= int(max_parts):
                break

        score = min(int(score_per_part) * int(max_parts), len(current_scored_parts) * int(score_per_part))
        return score, current_scored_parts
    except Exception as e:
        raise Exception(f"task1_check_parts_in_box 异常: {e}")


def task1_check_parts_in_lift(
    parts_poses_dict: Any,
    threshold_height: float,
    scored_parts: Iterable[str] | None = None,
    score_per_part: int = 10,
    max_parts: int = 4,
) -> tuple[int, set[str]]:
    """
    Task1 抬升评分（40分满分，每个工件10分）。
    判定条件：零件 z 高度 >= 指定阈值；一个工件只计分一次。
    """
    current_scored_parts: set[str] = set(scored_parts or [])
    try:
        parts = iter_part_dicts(parts_poses_dict)
        threshold = float(threshold_height)

        for idx, part in enumerate(parts):
            part_id = str(part.get("prim_path", f"part_{idx}"))
            if part_id in current_scored_parts:
                continue

            pos = safe_vec3(part.get("position"))
            if pos is None:
                continue

            if float(pos[2]) >= threshold:
                current_scored_parts.add(part_id)
                if len(current_scored_parts) >= int(max_parts):
                    break

        score = min(int(score_per_part) * int(max_parts), len(current_scored_parts) * int(score_per_part))
        return score, current_scored_parts
    except Exception as e:
        raise Exception(f"task1_check_parts_in_lift 异常: {e}") 


def task1_time_out_check(
    elapsed_seconds: float,
    full_score: int = 20,
    full_time_seconds: float = 180.0,
    penalty_interval_seconds: float = 30.0,
    penalty_per_interval: int = 5,
) -> int:
    """
    Task1 时间评分（20分满分）：
    - 40秒内得20分
    - 每超过10秒扣5分
    - 最低0分
    """
    try:
        t = max(0.0, float(elapsed_seconds))
        if t <= float(full_time_seconds):
            return int(full_score)
        overtime = t - float(full_time_seconds)
        penalty_steps = int(np.ceil(overtime / float(penalty_interval_seconds)))
        penalty = min(int(full_score), penalty_steps * int(penalty_per_interval))
        return max(0, int(full_score) - penalty)
    except Exception as e:
        raise Exception(f"task1_time_out_check 异常: {e}")



# ----------------- Task2 评估工具函数 -----------------


def task2_check_parts_grabbed(
    parts_poses_dict: Any,
    conveyor_limits: dict[str, tuple[float, float]],
    scored_parts: Iterable[str] | None = None,
    score_per_part: int = 10,
    max_parts: int = 10,
) -> tuple[int, set[str], dict]:
    """
    Task2 抓取评分：检查工件是否脱离传送带。

    判定条件：
    - 工件当前 z 高度 > 传送带高度 + 阈值
    - 一个工件只计分一次

    Returns:
        score: 抓取总分
        scored_parts: 已计分的工件ID集合
        details: 包含每个工件抓取状态的详细信息
    """
    current_scored_parts: set[str] = set(scored_parts or [])
    details: dict[str, dict] = {}
    try:
        parts = iter_part_dicts(parts_poses_dict)
        z_min = float(conveyor_limits.get("z", (0, 0))[0])
        # 传送带高度 + 一定阈值（如 0.05m）视为脱离传送带
        grab_threshold = z_min + 0.05

        for idx, part in enumerate(parts):
            part_id = str(part.get("prim_path", f"part_{idx}"))
            if part_id in current_scored_parts:
                # 已计分的工件，更新状态
                pos = safe_vec3(part.get("position"))
                if pos is not None:
                    details[part_id] = {
                        "grabbed": True,
                        "z_height": float(pos[2]),
                        "grabbed_at_step": details.get(part_id, {}).get("grabbed_at_step", "already_scored"),
                    }
                continue

            pos = safe_vec3(part.get("position"))
            if pos is None:
                continue

            is_grabbed = float(pos[2]) > grab_threshold
            details[part_id] = {
                "grabbed": bool(is_grabbed),
                "z_height": float(pos[2]),
                "grab_threshold": float(grab_threshold),
            }

            if is_grabbed:
                current_scored_parts.add(part_id)
                details[part_id]["grabbed_at_step"] = "scored"
                if len(current_scored_parts) >= int(max_parts):
                    break

        score = min(int(score_per_part) * int(max_parts), len(current_scored_parts) * int(score_per_part))
        return int(score), current_scored_parts, details
    except Exception as e:
        raise Exception(f"task2_check_parts_grabbed 异常: {e}")


def task2_check_parts_in_correct_bin(
    parts_poses_dict: Any,
    left_bin_pose: tuple[np.ndarray, np.ndarray],
    right_bin_pose: tuple[np.ndarray, np.ndarray],
    bin_half_size: tuple[float, float, float] = (0.15, 0.10, 0.05),
    scored_parts: Iterable[str] | None = None,
    score_per_part: int = 10,
    max_parts: int = 10,
) -> tuple[int, set[str], dict]:
    """
    Task2 分拣评分：检查工件是否在正确类别的料箱内。

    判定条件：
    - 工件在料箱空间内
    - A工件应在左侧料箱，B工件应在右侧料箱
    - 一个工件只计分一次

    Args:
        parts_poses_dict: 工件位姿字典
        left_bin_pose: 左料箱位姿
        right_bin_pose: 右料箱位姿
        bin_half_size: 料箱半尺寸
        scored_parts: 已计分的工件ID集合
        score_per_part: 每个工件得分
        max_parts: 最大工件数

    Returns:
        score: 分拣总分
        scored_parts: 已计分的工件ID集合
        details: 包含每个工件分拣状态的详细信息
    """
    current_scored_parts: set[str] = set(scored_parts or [])
    details: dict[str, dict] = {}
    try:
        left_bin_pos = safe_vec3(left_bin_pose[0] if isinstance(left_bin_pose, tuple) and len(left_bin_pose) > 0 else None)
        right_bin_pos = safe_vec3(right_bin_pose[0] if isinstance(right_bin_pose, tuple) and len(right_bin_pose) > 0 else None)

        if left_bin_pos is None or right_bin_pos is None:
            raise ValueError("task2_check_parts_in_correct_bin 异常: 无效的料箱位置")

        parts = iter_part_dicts(parts_poses_dict)
        hx, hy, hz = float(bin_half_size[0]), float(bin_half_size[1]), float(bin_half_size[2])

        for idx, part in enumerate(parts):
            part_id = str(part.get("prim_path", f"part_{idx}"))
            if part_id in current_scored_parts:
                # 已计分的工件，更新状态
                pos = safe_vec3(part.get("position"))
                if pos is not None:
                    details[part_id] = {
                        "sorted": True,
                        "bin_side": details.get(part_id, {}).get("bin_side", "unknown"),
                        "is_correct": True,
                    }
                continue

            pos = safe_vec3(part.get("position"))
            if pos is None:
                continue

            sem = _extract_semantics(part)
            if sem == "UNKNOWN":
                continue

            # 检查是否在左料箱内
            rel_left = pos - left_bin_pos
            in_left_bin = (abs(float(rel_left[0])) <= hx) and (abs(float(rel_left[1])) <= hy) and (abs(float(rel_left[2])) <= hz)

            # 检查是否在右料箱内
            rel_right = pos - right_bin_pos
            in_right_bin = (abs(float(rel_right[0])) <= hx) and (abs(float(rel_right[1])) <= hy) and (abs(float(rel_right[2])) <= hz)

            # A工件应在左料箱，B工件应在右料箱
            is_correct = (sem == "A" and in_left_bin) or (sem == "B" and in_right_bin)
            in_any_bin = in_left_bin or in_right_bin

            details[part_id] = {
                "sorted": bool(is_correct),
                "bin_side": "left" if in_left_bin else ("right" if in_right_bin else "none"),
                "is_correct": bool(is_correct),
                "semantics": sem,
            }

            if is_correct:
                current_scored_parts.add(part_id)
                if len(current_scored_parts) >= int(max_parts):
                    break

        score = min(int(score_per_part) * int(max_parts), len(current_scored_parts) * int(score_per_part))
        return int(score), current_scored_parts, details
    except Exception as e:
        raise Exception(f"task2_check_parts_in_correct_bin 异常: {e}")


def task2_calculate_total_score(
    grab_score: int,
    sort_score: int,
    max_grab_score: int = 100,
    max_sort_score: int = 100,
) -> dict:
    """
    Task2 总分汇总（官方标准）。

    官方评分标准：
    - 抓取成功 + 正确分拣 = 10分/个
    - 抓取成功但分拣错误 = 0分（不给分拣分）
    - 最大总分 100 分（10个工件，每个正确分拣得10分）

    计算规则：
    - grab_score: 抓取得分（仅统计用，不计入总分）
    - sort_score: 分拣得分（正确分拣的数量 × 10分）
    - total_score: 最终总分 = sort_score（只计算正确分拣）
    """
    # 官方标准：总分只计算正确分拣的数量
    total_score = int(sort_score)
    total_score = int(max(0, min(100, total_score)))
    return {
        "grab_count": int(grab_score / 10),      # 抓取成功数量（统计）
        "sort_count": int(sort_score / 10),      # 正确分拣数量
        "grab_score": int(grab_score),           # 抓取得分（仅统计）
        "sort_score": int(sort_score),           # 分拣得分（实际得分）
        "total_score": total_score,              # 最终总分
        "max_grab_score": int(max_grab_score),
        "max_sort_score": int(max_sort_score),
    }


# ------------------------task3评估工具------------------------------
def task3_check_insertion_success(
    parts_poses_dict: Any,
    foam_pose: np.ndarray,  # 泡棉的实时位姿 [x, y, z, qx, qy, qz, qw]
    local_slots_offsets: dict, # 槽位相对于泡棉中心的偏移量
    dist_threshold: float = 0.025,
    height_threshold: float = 0.015,
) -> tuple[int, set[str]]:
    """
    Task3 嵌装评分优化版：基于泡棉实时位姿进行坐标变换。
    """
    from scipy.spatial.transform import Rotation as R
    
    parts = iter_part_dicts(parts_poses_dict)
    scored_parts = set()
    total_score = 0
    
    # 获取泡棉的旋转和平移
    foam_pos = foam_pose[:3]
    foam_rot = R.from_quat(foam_pose[3:]).as_matrix()
    
    for slot_id, offset in local_slots_offsets.items():
        # 计算该槽位在世界坐标系下的期望位置：W_pos = R_foam * local_offset + T_foam
        target_world_pos = foam_rot @ np.array(offset['offset']) + foam_pos
        target_type = offset['type']
        
        for part in parts:
            part_id = part.get("prim_path", "")
            if part_id in scored_parts: continue
            if _extract_semantics(part) != target_type: continue
            
            part_pos = safe_vec3(part.get("position"))
            if part_pos is None: continue
            
            # 1. 检查水平距离 (XY)
            horizontal_dist = np.linalg.norm(part_pos[:2] - target_world_pos[:2])
            # 2. 检查高度 (Z) - 必须足够低才算嵌入
            height_diff = abs(part_pos[2] - target_world_pos[2])
            
            if horizontal_dist < dist_threshold and height_diff < height_threshold:
                total_score += 15
                scored_parts.add(part_id)
                break
                
    return min(90, total_score), scored_parts

def task3_calculate_specific_score(
    parts_poses_dict: Any,
    dist_threshold: float = 0.05,
    height_threshold: float = 0.03
) -> tuple[int, set[str]]:
    """
    针对 Task3 的精确评分：
    02, 03, 04 物体 -> 第二组空槽 (Y=0.21)
    05, 06, 07 物体 -> 第一组空槽 (Y=0.41)
    """
    # 按照你的要求提供绝对坐标
    group1_slots = [np.array([0.54, 0.41, 1.04]), np.array([0.76, 0.41, 1.04]), np.array([0.98, 0.41, 1.04])]
    group2_slots = [np.array([0.54, 0.21, 1.04]), np.array([0.76, 0.21, 1.04]), np.array([0.98, 0.21, 1.04])]
    
    total_score = 0
    scored_parts = set()
    used_slots = set() 

    parts = list(iter_part_dicts(parts_poses_dict))
    
    for part in parts:
        prim_path = part.get("prim_path", "")
        pos = safe_vec3(part.get("position"))
        if pos is None: continue
        
        target_slots = []
        slot_prefix = ""
        # 判定归属组别
        if any(x in prim_path for x in ["05", "06", "07"]):
            target_slots = group1_slots
            slot_prefix = "G1_"
        elif any(x in prim_path for x in ["02", "03", "04"]):
            target_slots = group2_slots
            slot_prefix = "G2_"
        else:
            continue # 如果都不是，跳过该物体不计分
        
        for i, slot_pos in enumerate(target_slots):
            slot_id = f"{slot_prefix}{i}"
            if slot_id in used_slots: continue
            
            # 仅计算 XY 平面距离
            h_dist = np.linalg.norm(pos[:2] - slot_pos[:2])
            # 计算高度误差
            v_dist = abs(pos[2] - slot_pos[2])
            
            # 💡 修正 3：使用 <= 防止正好卡在边缘距离而判定失败
            if h_dist <= dist_threshold and v_dist <= height_threshold:
                total_score += 15
                scored_parts.add(prim_path)
                used_slots.add(slot_id)
                break # 该物体已成功入槽，停止遍历其他槽
                
    return min(90, total_score), scored_parts

def task3_time_score(elapsed: float) -> int:
    """在2分钟内完成得10分，每超30秒扣5分，扣完为止"""
    if elapsed <= 120.0: 
        return 10
    
    over_time = elapsed - 120.0
    # np.ceil 确保只要超过一点点（比如超了1秒），就开始算作扣1次（5分）
    penalty_steps = int(np.ceil(over_time / 30.0))
    penalty = penalty_steps * 5
    
    return max(0, 10 - penalty)


# ----------------- Task4 评估工具函数 -----------------
def task4_check_box_joints_success(
    current_box_joints: np.ndarray,
    target_box_joints: np.ndarray,
    threshold: float = 0.2
) -> tuple[bool, dict]:
    """检查盒子关节位置是否达到目标。"""
    current = np.asarray(current_box_joints, dtype=np.float32).reshape(-1)
    target = np.asarray(target_box_joints, dtype=np.float32).reshape(-1)
    errors = np.abs(current - target)
    is_success = bool(np.all(errors < threshold))

    return is_success, {
        "max_error": float(np.max(errors)),
        "mean_error": float(np.mean(errors)),
        "all_errors": errors.copy()
    }


def task4_check_short_edge_close_score(
    current_box_joints: np.ndarray,
    short_edge_targets: tuple[float, float] | list[float] | np.ndarray,
    joint_indices: tuple[int, int] = (2, 3),
    threshold: float = 0.2,
    score_per_edge: int = 15,
) -> tuple[int, dict]:
    """Task4 短边评分（总分 30，每个短边 15）。"""
    joints = np.asarray(current_box_joints, dtype=np.float32).reshape(-1)
    targets = np.asarray(short_edge_targets, dtype=np.float32).reshape(-1)

    closed: list[bool] = []
    errors: list[float] = []
    score = 0
    for i, j_idx in enumerate(joint_indices):
        err = abs(float(joints[j_idx]) - float(targets[i]))
        is_closed = err <= float(threshold)
        errors.append(err)
        closed.append(is_closed)
        if is_closed:
            score += int(score_per_edge)

    return int(min(int(score_per_edge) * len(joint_indices), score)), {
        "short_edge_closed": closed,
        "short_edge_errors": errors,
    }


def task4_check_long_edge_close_score(
    current_box_joints: np.ndarray,
    long_edge_targets: tuple[float, float] | list[float] | np.ndarray,
    joint_indices: tuple[int, int] = (0, 1),
    threshold: float = 0.2,
    score_per_edge: int = 15,
) -> tuple[int, dict]:
    """Task4 长边评分（总分 30，每个长边 15）。"""
    joints = np.asarray(current_box_joints, dtype=np.float32).reshape(-1)
    targets = np.asarray(long_edge_targets, dtype=np.float32).reshape(-1)

    closed: list[bool] = []
    errors: list[float] = []
    score = 0
    for i, j_idx in enumerate(joint_indices):
        err = abs(float(joints[j_idx]) - float(targets[i]))
        is_closed = err <= float(threshold)
        errors.append(err)
        closed.append(is_closed)
        if is_closed:
            score += int(score_per_edge)

    return int(min(int(score_per_edge) * len(joint_indices), score)), {
        "long_edge_closed": closed,
        "long_edge_errors": errors,
    }


def task4_time_score(
    elapsed_seconds: float,
    full_score: int = 40,
    full_time_seconds: float = 180.0,
    penalty_interval_seconds: float = 30.0,
    penalty_per_interval: int = 5,
) -> int:
    """
    Task4 时间评分：
    - 2 分钟内满分（默认 40）
    - 每超 10 秒扣 5 分，最低 0 分
    """
    t = max(0.0, float(elapsed_seconds))
    if t <= float(full_time_seconds):
        return int(full_score)
    overtime = t - float(full_time_seconds)
    penalty_steps = int(np.ceil(overtime / float(penalty_interval_seconds)))
    penalty = penalty_steps * int(penalty_per_interval)
    return int(max(0, int(full_score) - penalty))


def task4_get_bimanual_collaboration_factor(
    is_bimanual_collaboration: bool,
    single_arm_factor: float = 0.7,
    bimanual_factor: float = 1.0,
) -> float:
    """Task4 协同系数：单臂=0.7，双臂协同=1.0。"""
    return float(bimanual_factor if is_bimanual_collaboration else single_arm_factor)


def task4_calculate_total_score(
    short_edge_score: int,
    long_edge_score: int,
    time_score: int,
    collaboration_factor: float,
) -> dict:
    """Task4 总分汇总：raw_score = 短边 + 长边 + 时间，final_score = raw_score * collaboration_factor。"""
    raw_score = int(short_edge_score) + int(long_edge_score) + int(time_score)
    raw_score = int(max(0, min(100, raw_score)))
    final_score = int(round(raw_score * float(collaboration_factor)))
    final_score = int(max(0, min(100, final_score)))
    return {
        "raw_score": raw_score,
        "final_score": final_score,
        "collaboration_factor": float(collaboration_factor),
    }
