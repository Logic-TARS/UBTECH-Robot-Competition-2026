# Evaluation Scoring Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the container simulation evaluation scoring match the official task1-task4 scoring criteria exactly.

**Architecture:** Keep scoring calculations in `src/lerobot/sim_eval/scoring.py`, episode state and task lifecycle logic in `src/lerobot/sim_eval/task_assert.py`, and task-derived defaults in `src/lerobot/sim_eval/task_eval_config.py`. Add deterministic part metadata at the scene boundary so scoring never guesses class from unstable prim paths.

**Tech Stack:** Python, NumPy, pytest, Isaac Sim scene wrappers, existing `lerobot.sim_eval` container evaluation modules.

---

## Official Criteria Summary

Task1:
- Grasp: 4 parts, 10 points each, max 40. A part counts once after it is grasped and stably lifted at least 10 cm above its own baseline.
- Place: 4 parts, 10 points each, max 40. A part counts once after it is fully released from the gripper and its final resting position is inside the correct category bin space.
- Time: max 20. Full score at or below 180 seconds, subtract 5 for each started 30 second overtime interval, floor 0.

Task2:
- Follow: 8 parts, 2.5 points each, max 20. A part counts once when an end effector reaches within 10 cm of it.
- Grasp: 8 parts, 5 points each, max 40. A part counts once after it is grasped and stably lifted at least 10 cm above its own baseline.
- Place: 8 parts, 5 points each, max 40. A part counts once after it is fully released from the gripper and its final resting position is inside the correct category bin space.

Task3:
- Grasp: 6 parts, 7.5 points each, max 45. A part counts once after it is grasped and stably lifted at least 10 cm above its own baseline.
- Insert: 6 parts, 7.5 points each, max 45. A part counts once after it is fully released from the gripper and placed in the matching category slot.
- Time: max 10. Full score at or below 360 seconds, subtract 5 for each started 60 second overtime interval, floor 0.

Task4:
- Short edges: 2 faces, 15 points each, max 30. The end effector must contact the carton short edge and close it.
- Long edges: 2 faces, 15 points each, max 30. The end effector must contact the carton long edge and close it.
- Time: max 10. Full score at or below 180 seconds, subtract 5 for each started 30 second overtime interval, floor 0.

---

## File Structure

- Modify `Ubtech_sim/source/SceneBuilder.py`
  - Add stable per-part semantics to `get_parts_world_poses()`.
  - Preserve A/B labels through reset and randomization for task1, task2, and task3.

- Modify `src/lerobot/sim_eval/scoring.py`
  - Replace old scoring defaults with official constants.
  - Add reusable scoring helpers for time, lift stability, release/static checks, bin checks, slot checks, end-effector proximity, and carton edge contact plus closure.
  - Keep small task-specific wrapper functions for compatibility with `task_assert.py`.

- Modify `src/lerobot/sim_eval/task_assert.py`
  - Track initial part heights, per-step trajectories, release/static state, end-effector proximity, and task4 contact events.
  - Compute partial scores continuously but only award time score when the task reaches the official completion condition.
  - Remove task4 collaboration multiplier.

- Modify `src/lerobot/sim_eval/task_eval_config.py`
  - Replace all defaults with official max parts, per-part scores, time windows, penalty intervals, and success thresholds.

- Modify `Ubtech_sim/config/Conveyor_Sorting.yaml`
  - Change task2 part count from 10 total to 8 total by setting `part.num_parts: 4`.

- Create `tests/sim_eval/test_scoring.py`
  - Unit tests for official time scoring, lift scoring from baseline, category bin checks, slot checks, and task4 score composition.

- Create `tests/sim_eval/test_task_eval_config.py`
  - Unit tests for official default extraction from task YAML.

---

## Task 1: Add Scoring Unit Tests Before Code Changes

**Files:**
- Create: `tests/sim_eval/test_scoring.py`
- Create: `tests/sim_eval/test_task_eval_config.py`

- [ ] **Step 1: Create the scoring test file**

Add this file:

```python
# tests/sim_eval/test_scoring.py
import numpy as np

from src.lerobot.sim_eval import scoring


def part(path, xyz, sem="A", released=True, static=True):
    return {
        "prim_path": path,
        "position": list(xyz),
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "semantics": sem,
        "released": released,
        "static": static,
    }


def test_official_time_score_task1_boundaries():
    assert scoring.calculate_time_score(180.0, 20, 180.0, 30.0, 5) == 20
    assert scoring.calculate_time_score(180.1, 20, 180.0, 30.0, 5) == 15
    assert scoring.calculate_time_score(210.0, 20, 180.0, 30.0, 5) == 15
    assert scoring.calculate_time_score(210.1, 20, 180.0, 30.0, 5) == 10
    assert scoring.calculate_time_score(300.1, 20, 180.0, 30.0, 5) == 0


def test_official_time_score_task3_boundaries():
    assert scoring.calculate_time_score(360.0, 10, 360.0, 60.0, 5) == 10
    assert scoring.calculate_time_score(360.1, 10, 360.0, 60.0, 5) == 5
    assert scoring.calculate_time_score(420.1, 10, 360.0, 60.0, 5) == 0


def test_lift_score_requires_ten_cm_above_initial_height():
    parts = [
        part("/Root/Part_A0", [0.0, 0.0, 1.09], "A"),
        part("/Root/Part_B0", [0.0, 0.0, 1.10], "B"),
        part("/Root/Part_A1", [0.0, 0.0, 1.101], "A"),
    ]
    initial_heights = {
        "/Root/Part_A0": 1.00,
        "/Root/Part_B0": 1.00,
        "/Root/Part_A1": 1.00,
    }

    score, scored = scoring.check_parts_lifted_from_initial_height(
        parts_poses_dict=parts,
        initial_heights=initial_heights,
        scored_parts=set(),
        lift_delta=0.10,
        score_per_part=10,
        max_parts=4,
    )

    assert score == 20
    assert scored == {"/Root/Part_B0", "/Root/Part_A1"}


def test_task1_bin_score_requires_correct_category_released_and_static():
    box_pose = (np.array([1.2, 0.3, 1.05]), np.array([0.0, 0.0, 0.0, 1.0]))
    parts = [
        part("/Root/Part_A0", [1.2, 0.20, 1.05], "A", released=True, static=True),
        part("/Root/Part_B0", [1.2, 0.40, 1.05], "B", released=True, static=True),
        part("/Root/Part_A1", [1.2, 0.40, 1.05], "A", released=True, static=True),
        part("/Root/Part_B1", [1.2, 0.40, 1.05], "B", released=False, static=True),
    ]

    score, scored = scoring.task1_check_parts_in_box(
        box_poses=box_pose,
        parts_poses_dict=parts,
        scored_parts=set(),
        box_half_size=(0.19, 0.375, 0.18),
        score_per_part=10,
        max_parts=4,
        require_released=True,
        require_static=True,
    )

    assert score == 20
    assert scored == {"/Root/Part_A0", "/Root/Part_B0"}


def test_task2_follow_score_counts_each_part_once_for_either_end_effector():
    parts = [
        part("/Root/Part_A0", [0.0, 0.0, 0.0], "A"),
        part("/Root/Part_B0", [1.0, 1.0, 1.0], "B"),
    ]
    ee_poses = {
        "left": np.array([0.09, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "right": np.array([5.0, 5.0, 5.0, 0.0, 0.0, 0.0]),
    }

    score, scored, details = scoring.task2_check_end_effector_followed_parts(
        parts_poses_dict=parts,
        ee_poses=ee_poses,
        scored_parts=set(),
        distance_threshold=0.10,
        score_per_part=2.5,
        max_parts=8,
    )

    assert score == 2.5
    assert scored == {"/Root/Part_A0"}
    assert details["/Root/Part_A0"]["min_distance"] == 0.09


def test_task3_insert_score_is_seven_point_five_per_correct_slot():
    parts = [
        part("/Root/Part_A0", [0.54, 0.21, 1.04], "A", released=True, static=True),
        part("/Root/Part_B0", [0.54, 0.41, 1.04], "B", released=True, static=True),
        part("/Root/Part_A1", [0.98, 0.41, 1.04], "A", released=True, static=True),
    ]
    slots = [
        {"slot_id": "A0", "type": "A", "position": [0.54, 0.21, 1.04]},
        {"slot_id": "B0", "type": "B", "position": [0.54, 0.41, 1.04]},
    ]

    score, scored, used_slots = scoring.task3_check_parts_inserted_in_slots(
        parts_poses_dict=parts,
        slots=slots,
        scored_parts=set(),
        used_slots=set(),
        dist_threshold=0.05,
        height_threshold=0.03,
        score_per_part=7.5,
        max_parts=6,
        require_released=True,
        require_static=True,
    )

    assert score == 15.0
    assert scored == {"/Root/Part_A0", "/Root/Part_B0"}
    assert used_slots == {"A0", "B0"}


def test_task4_total_score_has_no_collaboration_multiplier():
    score_info = scoring.task4_calculate_total_score(
        short_edge_score=30,
        long_edge_score=30,
        time_score=10,
    )

    assert score_info == {"raw_score": 70, "final_score": 70}
```

- [ ] **Step 2: Create the task config test file**

Add this file:

```python
# tests/sim_eval/test_task_eval_config.py
from src.lerobot.sim_eval.task_eval_config import build_assertion_args_from_task_yaml


def test_task1_official_defaults():
    cfg = {
        "part": {"num_parts": 2},
        "box": {"box_scale": [[0.38, 0.75, 0.36]]},
        "grasp": {"lift_height": 0.17, "scatter_area": {"center": [0.75, 0.28, 1.04], "size": [0.30, 0.23]}},
    }
    args = build_assertion_args_from_task_yaml("task1", cfg)

    assert args.task1_lift_score_per_part == 10
    assert args.task1_box_score_per_part == 10
    assert args.task1_max_parts == 4
    assert args.task1_time_full_score == 20
    assert args.task1_time_full_time_seconds == 180.0
    assert args.task1_time_penalty_interval_seconds == 30.0
    assert args.task1_success_score_threshold == 80


def test_task2_official_defaults():
    cfg = {
        "part": {"num_parts": 4},
        "box": {"box_scale": [[0.38, 0.75, 0.36]]},
        "plane": {"plane_position": [[0.1, 0.26859, 1.25]]},
        "grasp": {"scatter_area": {"center": [0.12, 0.26859, 1.2], "size": [0.06, 0.04]}},
        "ConveyorBelt": {"ConveyorBelt_position": [[-0.133, 0.26859, 0.0]]},
    }
    args = build_assertion_args_from_task_yaml("task2", cfg)

    assert args.task2_max_parts == 8
    assert args.task2_follow_score_per_part == 2.5
    assert args.task2_grab_score_per_part == 5
    assert args.task2_sort_score_per_part == 5
    assert args.task2_success_score_threshold == 100


def test_task3_official_defaults():
    cfg = {
        "part": {"num_parts": 3},
        "box": {"box_position": [[0.28, 0.3, 1.04], [1.25, 0.3, 1.04]], "box_scale": [[0.5, 1.2, 0.3], [0.5, 1.2, 0.3]]},
        "foam": {"foam_position": [0.76, 0.3, 1.0]},
    }
    args = build_assertion_args_from_task_yaml("task3", cfg)

    assert args.task3_grab_score_per_part == 7.5
    assert args.task3_insert_score_per_part == 7.5
    assert args.task3_max_parts == 6
    assert args.task3_time_full_score == 10
    assert args.task3_time_full_time_seconds == 360.0
    assert args.task3_time_penalty_interval_seconds == 60.0
    assert args.task3_success_score_threshold == 90


def test_task4_official_defaults():
    args = build_assertion_args_from_task_yaml("task4", {})

    assert args.task4_short_edge_score_per_edge == 15
    assert args.task4_long_edge_score_per_edge == 15
    assert args.task4_time_full_score == 10
    assert args.task4_time_full_time_seconds == 180.0
    assert args.task4_time_penalty_interval_seconds == 30.0
    assert not hasattr(args, "task4_single_arm_factor")
    assert not hasattr(args, "task4_bimanual_factor")
```

- [ ] **Step 3: Run the new tests and verify they fail for the current implementation**

Run:

```bash
pytest tests/sim_eval/test_scoring.py tests/sim_eval/test_task_eval_config.py -q
```

Expected:
- Fails because `calculate_time_score`, `check_parts_lifted_from_initial_height`, `task2_check_end_effector_followed_parts`, and `task3_check_parts_inserted_in_slots` do not exist yet.
- Fails because current task config defaults still use old score and time values.

---

## Task 2: Add Stable Part Semantics At The Scene Boundary

**Files:**
- Modify: `Ubtech_sim/source/SceneBuilder.py`
- Test: `tests/sim_eval/test_scoring.py`

- [ ] **Step 1: Add semantic map initialization**

In `SceneBuilder.__init__`, add this field after `self.parts_prim_paths = []`:

```python
self.parts_semantics_by_path: Dict[str, str] = {}
```

- [ ] **Step 2: Add a helper that assigns semantics by creation order**

Add this method inside `SceneBuilder`:

```python
def _set_part_semantics_by_order(self, labels: List[str]) -> None:
    """Assign stable A/B labels to tracked parts using current path order."""
    self.parts_semantics_by_path = {}
    for prim_path, label in zip(self.parts_prim_paths, labels):
        normalized = str(label).strip().upper()
        if normalized in {"A", "B"}:
            self.parts_semantics_by_path[str(prim_path)] = normalized
```

- [ ] **Step 3: Assign semantics for task2 clone paths**

After `self.parts_prim_paths = clone_paths` in task2 setup, add:

```python
self._set_part_semantics_by_order(["A"] * num_parts + ["B"] * num_parts)
```

- [ ] **Step 4: Assign semantics for task1 Replicator paths**

After `self._extract_parts_prim_paths()` in task1 setup, add:

```python
self._set_part_semantics_by_order(["A"] * num_to_create + ["B"] * num_to_create)
```

- [ ] **Step 5: Assign semantics for task3 grouped paths**

After `self._extract_parts_prim_paths()` in task3 setup, add:

```python
parts_per_group = self.part_cfg.get("num_parts", 3)
self._set_part_semantics_by_order(["A"] * parts_per_group + ["B"] * parts_per_group)
```

- [ ] **Step 6: Preserve semantics after task1 reset randomization**

At the end of `_randomize_task1_assets`, after `self.parts_prim_paths = saved_paths`, add:

```python
self._set_part_semantics_by_order(["A"] * num_a + ["B"] * num_a)
```

- [ ] **Step 7: Preserve semantics after task3 reset randomization**

At the end of `_randomize_task3_assets`, after `self.parts_prim_paths = saved_paths`, add:

```python
parts_per_group = self.part_cfg.get("num_parts", 3)
self._set_part_semantics_by_order(["A"] * parts_per_group + ["B"] * parts_per_group)
```

- [ ] **Step 8: Return semantics from `get_parts_world_poses()`**

Change the dict appended in `get_parts_world_poses()` to include:

```python
"semantics": self.parts_semantics_by_path.get(str(prim_path), ""),
```

The final append block should be:

```python
results.append({
    "prim_path": prim_path,
    "position": pos,
    "orientation": [float(qi[0]), float(qi[1]), float(qi[2]), float(qr)],
    "semantics": self.parts_semantics_by_path.get(str(prim_path), ""),
})
```

- [ ] **Step 9: Run parser check**

Run:

```bash
python -m py_compile Ubtech_sim/source/SceneBuilder.py
```

Expected: exit code 0.

---

## Task 3: Replace Scoring Helpers With Official Reusable Functions

**Files:**
- Modify: `src/lerobot/sim_eval/scoring.py`
- Test: `tests/sim_eval/test_scoring.py`

- [ ] **Step 1: Add official generic helpers**

Add these functions near the top of `scoring.py`, after `_extract_semantics`:

```python
def _part_id(part: PartPoseDict, idx: int) -> str:
    return str(part.get("prim_path", f"part_{idx}"))


def _is_released(part: PartPoseDict, require_released: bool) -> bool:
    if not require_released:
        return True
    return bool(part.get("released", False))


def _is_static(part: PartPoseDict, require_static: bool) -> bool:
    if not require_static:
        return True
    return bool(part.get("static", False))


def calculate_time_score(
    elapsed_seconds: float,
    full_score: float,
    full_time_seconds: float,
    penalty_interval_seconds: float,
    penalty_per_interval: float,
) -> float:
    t = max(0.0, float(elapsed_seconds))
    full = float(full_score)
    if t <= float(full_time_seconds):
        return full
    overtime = t - float(full_time_seconds)
    penalty_steps = int(np.ceil(overtime / float(penalty_interval_seconds)))
    penalty = penalty_steps * float(penalty_per_interval)
    return max(0.0, full - penalty)


def check_parts_lifted_from_initial_height(
    parts_poses_dict: Any,
    initial_heights: dict[str, float],
    scored_parts: Iterable[str] | None = None,
    lift_delta: float = 0.10,
    score_per_part: float = 10.0,
    max_parts: int = 4,
) -> tuple[float, set[str]]:
    current_scored_parts: set[str] = set(scored_parts or [])
    parts = iter_part_dicts(parts_poses_dict)
    for idx, part in enumerate(parts):
        part_id = _part_id(part, idx)
        if part_id in current_scored_parts:
            continue
        pos = safe_vec3(part.get("position"))
        if pos is None or part_id not in initial_heights:
            continue
        if float(pos[2]) >= float(initial_heights[part_id]) + float(lift_delta):
            current_scored_parts.add(part_id)
            if len(current_scored_parts) >= int(max_parts):
                break
    score = min(float(score_per_part) * int(max_parts), len(current_scored_parts) * float(score_per_part))
    return score, current_scored_parts


def point_in_aabb(point: np.ndarray, center: np.ndarray, half_size: tuple[float, float, float]) -> bool:
    rel = point - center
    hx, hy, hz = float(half_size[0]), float(half_size[1]), float(half_size[2])
    return (
        abs(float(rel[0])) <= hx
        and abs(float(rel[1])) <= hy
        and abs(float(rel[2])) <= hz
    )
```

- [ ] **Step 2: Update `task1_check_parts_in_box` signature**

Add release/static arguments:

```python
require_released: bool = False,
require_static: bool = False,
```

Inside the loop, before adding the part to `current_scored_parts`, add:

```python
if not _is_released(part, require_released):
    continue
if not _is_static(part, require_static):
    continue
```

Keep the existing A/B split for task1 unless the official task config later provides explicit bin subregions. This preserves the existing category-bin model while enforcing release and final static state.

- [ ] **Step 3: Replace task1 time function**

Change `task1_time_out_check` to delegate to the generic helper:

```python
def task1_time_out_check(
    elapsed_seconds: float,
    full_score: int = 20,
    full_time_seconds: float = 180.0,
    penalty_interval_seconds: float = 30.0,
    penalty_per_interval: int = 5,
) -> int:
    return int(calculate_time_score(
        elapsed_seconds=elapsed_seconds,
        full_score=full_score,
        full_time_seconds=full_time_seconds,
        penalty_interval_seconds=penalty_interval_seconds,
        penalty_per_interval=penalty_per_interval,
    ))
```

- [ ] **Step 4: Add task2 follow scoring**

Add this function:

```python
def task2_check_end_effector_followed_parts(
    parts_poses_dict: Any,
    ee_poses: dict[str, Any],
    scored_parts: Iterable[str] | None = None,
    distance_threshold: float = 0.10,
    score_per_part: float = 2.5,
    max_parts: int = 8,
) -> tuple[float, set[str], dict]:
    current_scored_parts: set[str] = set(scored_parts or [])
    details: dict[str, dict] = {}
    ee_points: list[np.ndarray] = []
    for value in ee_poses.values():
        vec = safe_vec3(value)
        if vec is not None:
            ee_points.append(vec)

    for idx, part in enumerate(iter_part_dicts(parts_poses_dict)):
        part_id = _part_id(part, idx)
        pos = safe_vec3(part.get("position"))
        if pos is None or not ee_points:
            continue
        distances = [float(np.linalg.norm(pos - ee_pos)) for ee_pos in ee_points]
        min_distance = min(distances)
        followed = min_distance <= float(distance_threshold)
        details[part_id] = {"followed": followed, "min_distance": min_distance}
        if followed and part_id not in current_scored_parts:
            current_scored_parts.add(part_id)
            if len(current_scored_parts) >= int(max_parts):
                break

    score = min(float(score_per_part) * int(max_parts), len(current_scored_parts) * float(score_per_part))
    return score, current_scored_parts, details
```

- [ ] **Step 5: Change task2 grasp defaults**

Change `task2_check_parts_grabbed` defaults:

```python
score_per_part: float = 5.0,
max_parts: int = 8,
```

Replace the old conveyor `z_min + 0.05` test with initial-height based lifting by adding an optional argument:

```python
initial_heights: dict[str, float] | None = None,
lift_delta: float = 0.10,
```

When `initial_heights` is provided, use `initial_heights[part_id] + lift_delta`; otherwise keep the conveyor fallback for compatibility.

- [ ] **Step 6: Change task2 placement defaults**

Change `task2_check_parts_in_correct_bin` defaults:

```python
score_per_part: float = 5.0,
max_parts: int = 8,
require_released: bool = False,
require_static: bool = False,
```

Before scoring `is_correct`, require:

```python
if is_correct and (not _is_released(part, require_released) or not _is_static(part, require_static)):
    is_correct = False
```

- [ ] **Step 7: Add task3 official slot scoring**

Add this function:

```python
def task3_check_parts_inserted_in_slots(
    parts_poses_dict: Any,
    slots: list[dict],
    scored_parts: Iterable[str] | None = None,
    used_slots: Iterable[str] | None = None,
    dist_threshold: float = 0.05,
    height_threshold: float = 0.03,
    score_per_part: float = 7.5,
    max_parts: int = 6,
    require_released: bool = False,
    require_static: bool = False,
) -> tuple[float, set[str], set[str]]:
    current_scored_parts: set[str] = set(scored_parts or [])
    current_used_slots: set[str] = set(used_slots or [])

    for idx, part in enumerate(iter_part_dicts(parts_poses_dict)):
        part_id = _part_id(part, idx)
        if part_id in current_scored_parts:
            continue
        if not _is_released(part, require_released) or not _is_static(part, require_static):
            continue
        part_pos = safe_vec3(part.get("position"))
        if part_pos is None:
            continue
        sem = _extract_semantics(part)
        for slot in slots:
            slot_id = str(slot["slot_id"])
            if slot_id in current_used_slots:
                continue
            if str(slot["type"]).upper() != sem:
                continue
            slot_pos = safe_vec3(slot["position"])
            if slot_pos is None:
                continue
            horizontal_dist = float(np.linalg.norm(part_pos[:2] - slot_pos[:2]))
            height_diff = abs(float(part_pos[2]) - float(slot_pos[2]))
            if horizontal_dist <= float(dist_threshold) and height_diff <= float(height_threshold):
                current_scored_parts.add(part_id)
                current_used_slots.add(slot_id)
                break
        if len(current_scored_parts) >= int(max_parts):
            break

    score = min(float(score_per_part) * int(max_parts), len(current_scored_parts) * float(score_per_part))
    return score, current_scored_parts, current_used_slots
```

- [ ] **Step 8: Replace task3 time function**

Replace `task3_time_score` with:

```python
def task3_time_score(elapsed: float) -> int:
    return int(calculate_time_score(
        elapsed_seconds=elapsed,
        full_score=10,
        full_time_seconds=360.0,
        penalty_interval_seconds=60.0,
        penalty_per_interval=5,
    ))
```

- [ ] **Step 9: Replace task4 time and total score functions**

Change `task4_time_score` defaults:

```python
full_score: int = 10,
full_time_seconds: float = 180.0,
penalty_interval_seconds: float = 30.0,
penalty_per_interval: int = 5,
```

Replace `task4_calculate_total_score` with:

```python
def task4_calculate_total_score(
    short_edge_score: int,
    long_edge_score: int,
    time_score: int,
) -> dict:
    raw_score = int(short_edge_score) + int(long_edge_score) + int(time_score)
    final_score = int(max(0, min(100, raw_score)))
    return {
        "raw_score": final_score,
        "final_score": final_score,
    }
```

- [ ] **Step 10: Run scoring tests**

Run:

```bash
pytest tests/sim_eval/test_scoring.py -q
```

Expected: all tests in `test_scoring.py` pass.

---

## Task 4: Fix Official Defaults And Task2 Part Count

**Files:**
- Modify: `src/lerobot/sim_eval/task_eval_config.py`
- Modify: `Ubtech_sim/config/Conveyor_Sorting.yaml`
- Test: `tests/sim_eval/test_task_eval_config.py`

- [ ] **Step 1: Fix task1 defaults**

In `_build_task1_args`, use these official values:

```python
"task1_lift_score_per_part": int(evaluation.get("lift_score_per_part", 10)),
"task1_box_score_per_part": int(evaluation.get("box_score_per_part", 10)),
"task1_max_parts": int(evaluation.get("max_parts", task_config.get("part", {}).get("num_parts", 2) * 2)),
"task1_time_full_score": int(evaluation.get("time_full_score", 20)),
"task1_time_full_time_seconds": float(evaluation.get("time_full_time_seconds", 180.0)),
"task1_time_penalty_interval_seconds": float(evaluation.get("time_penalty_interval_seconds", 30.0)),
"task1_time_penalty_per_interval": int(evaluation.get("time_penalty_per_interval", 5)),
"task1_success_score_threshold": int(evaluation.get("success_score_threshold", 80)),
"task1_lift_delta": float(evaluation.get("lift_delta", 0.10)),
```

- [ ] **Step 2: Fix task2 defaults**

In `_build_task2_args`, set:

```python
max_parts = int(evaluation.get("max_parts", task_config.get("part", {}).get("num_parts", 4) * 2))
follow_score_per_part = float(evaluation.get("follow_score_per_part", 2.5))
grab_score_per_part = float(evaluation.get("grab_score_per_part", 5.0))
sort_score_per_part = float(evaluation.get("sort_score_per_part", 5.0))
```

Return these additional fields:

```python
"task2_follow_score_per_part": follow_score_per_part,
"task2_follow_distance_threshold": float(evaluation.get("follow_distance_threshold", 0.10)),
"task2_lift_delta": float(evaluation.get("lift_delta", 0.10)),
"task2_success_score_threshold": int(evaluation.get("success_score_threshold", 100)),
```

- [ ] **Step 3: Fix task3 defaults**

In `_build_task3_args`, add official values:

```python
"task3_grab_score_per_part": float(evaluation.get("grab_score_per_part", 7.5)),
"task3_insert_score_per_part": float(evaluation.get("insert_score_per_part", 7.5)),
"task3_max_parts": int(evaluation.get("max_parts", task_config.get("part", {}).get("num_parts", 3) * 2)),
"task3_lift_delta": float(evaluation.get("lift_delta", 0.10)),
"task3_time_full_score": int(evaluation.get("time_full_score", 10)),
"task3_time_full_time_seconds": float(evaluation.get("time_full_time_seconds", 360.0)),
"task3_time_penalty_interval_seconds": float(evaluation.get("time_penalty_interval_seconds", 60.0)),
"task3_time_penalty_per_interval": int(evaluation.get("time_penalty_per_interval", 5)),
"task3_success_score_threshold": int(evaluation.get("success_score_threshold", 90)),
```

- [ ] **Step 4: Fix task4 defaults**

In `_build_task4_args`, change time values and remove collaboration factor fields from the returned dict:

```python
"task4_time_full_score": int(evaluation.get("time_full_score", 10)),
"task4_time_full_time_seconds": float(evaluation.get("time_full_time_seconds", 180.0)),
"task4_time_penalty_interval_seconds": float(evaluation.get("time_penalty_interval_seconds", 30.0)),
"task4_time_penalty_per_interval": int(evaluation.get("time_penalty_per_interval", 5)),
"task4_contact_distance_threshold": float(evaluation.get("contact_distance_threshold", 0.05)),
```

- [ ] **Step 5: Change task2 config to 8 total parts**

In `Ubtech_sim/config/Conveyor_Sorting.yaml`, change:

```yaml
num_parts: 5
```

to:

```yaml
num_parts: 4
```

- [ ] **Step 6: Run config tests**

Run:

```bash
pytest tests/sim_eval/test_task_eval_config.py -q
```

Expected: all tests in `test_task_eval_config.py` pass.

---

## Task 5: Track Episode State Needed For Official Scoring

**Files:**
- Modify: `src/lerobot/sim_eval/task_assert.py`
- Test: `tests/sim_eval/test_scoring.py`

- [ ] **Step 1: Rename and generalize the part tracker**

Rename `Taks1EpisodePartsTracker` to `EpisodePartsTracker` and give it these fields:

```python
self.parts_poses_list: list[list[dict[str, Any]]] = []
self.parts_trajectory: dict[str, list[list[float]]] = {}
self.initial_heights: dict[str, float] = {}
self.last_positions: dict[str, list[float]] = {}
self.static_parts: set[str] = set()
self.released_parts: set[str] = set()
```

- [ ] **Step 2: Track initial heights and static parts**

Replace `add_parts_poses` with:

```python
def add_parts_poses(self, parts_poses_dict: list[dict[str, Any]], static_window: int = 10, static_threshold: float = 0.005) -> None:
    self.parts_poses_list.append(parts_poses_dict)
    for part_info in parts_poses_dict:
        prim_path = str(part_info["prim_path"])
        position = [float(v) for v in part_info["position"][:3]]
        if prim_path not in self.initial_heights:
            self.initial_heights[prim_path] = float(position[2])
        self.parts_trajectory.setdefault(prim_path, []).append(position)
        self.last_positions[prim_path] = position

        recent = self.parts_trajectory[prim_path][-int(static_window):]
        if len(recent) >= int(static_window):
            arr = np.asarray(recent, dtype=np.float32)
            max_delta = float(np.max(np.linalg.norm(arr - arr[-1], axis=1)))
            if max_delta <= float(static_threshold):
                self.static_parts.add(prim_path)
```

- [ ] **Step 3: Track released parts using gripper distance**

Add this method:

```python
def update_release_state(
    self,
    parts_poses_dict: list[dict[str, Any]],
    ee_poses: dict[str, Any],
    release_distance_threshold: float = 0.08,
) -> list[dict[str, Any]]:
    ee_points = []
    for value in ee_poses.values():
        vec = np.asarray(value, dtype=np.float32).reshape(-1)
        if vec.size >= 3:
            ee_points.append(vec[:3])

    enriched = []
    for part in parts_poses_dict:
        new_part = dict(part)
        part_id = str(part.get("prim_path", ""))
        pos = np.asarray(part.get("position", []), dtype=np.float32).reshape(-1)
        released = False
        if pos.size >= 3 and ee_points:
            min_distance = min(float(np.linalg.norm(pos[:3] - ee)) for ee in ee_points)
            released = min_distance > float(release_distance_threshold)
        if released:
            self.released_parts.add(part_id)
        new_part["released"] = part_id in self.released_parts
        new_part["static"] = part_id in self.static_parts
        enriched.append(new_part)
    return enriched
```

- [ ] **Step 4: Add a safe end-effector pose helper**

Add this function near the top of `task_assert.py`:

```python
def get_robot_ee_poses(robot: WalkerS2sim) -> dict[str, Any]:
    interface = getattr(robot, "_robot_interface", None)
    if interface is None or not hasattr(interface, "get_ee_poses"):
        return {}
    poses = interface.get_ee_poses()
    return poses if isinstance(poses, dict) else {}
```

- [ ] **Step 5: Run parser check**

Run:

```bash
python -m py_compile src/lerobot/sim_eval/task_assert.py
```

Expected: exit code 0.

---

## Task 6: Fix Task1 Assertion To Use Official Scoring

**Files:**
- Modify: `src/lerobot/sim_eval/task_assert.py`
- Test: `tests/sim_eval/test_scoring.py`

- [ ] **Step 1: Add task1 lift delta constructor argument**

In `Task1Assertion.__init__`, add:

```python
lift_delta: float,
```

Store it:

```python
self._lift_delta = float(lift_delta)
```

- [ ] **Step 2: Use `EpisodePartsTracker`**

Replace:

```python
self._parts_tracker = Taks1EpisodePartsTracker()
```

with:

```python
self._parts_tracker = EpisodePartsTracker()
```

- [ ] **Step 3: Enrich part poses before scoring**

In `Task1Assertion.__call__`, after fetching `parts_poses`, add:

```python
ee_poses = get_robot_ee_poses(robot)
self._parts_tracker.add_parts_poses(parts_poses)
parts_poses = self._parts_tracker.update_release_state(parts_poses, ee_poses)
```

- [ ] **Step 4: Score lift from initial height**

Replace `task1_check_parts_in_lift(...)` call with:

```python
lift_score, self._lift_scored_parts = check_parts_lifted_from_initial_height(
    parts_poses_dict=parts_poses,
    initial_heights=self._parts_tracker.initial_heights,
    scored_parts=self._lift_scored_parts,
    lift_delta=self._lift_delta,
    score_per_part=self._lift_score_per_part,
    max_parts=self._max_parts,
)
```

- [ ] **Step 5: Require released and static for box score**

In the `task1_check_parts_in_box` call, add:

```python
require_released=True,
require_static=True,
```

- [ ] **Step 6: Set official success threshold**

Keep `is_success = total_score >= self._success_score_threshold`, with `_success_score_threshold` now set to 80. Time score is added only when all grasp and place points are awarded.

- [ ] **Step 7: Remove no-movement score clearing**

Delete this block:

```python
if terminal_no_movement:
    total_score = 0
```

The official criteria do not require clearing a successful score because a separate movement heuristic says all parts did not move.

- [ ] **Step 8: Pass `lift_delta` from `create_task_assertion`**

In the `Task1Assertion(...)` constructor call, add:

```python
lift_delta=float(getattr(args, "task1_lift_delta")),
```

- [ ] **Step 9: Run task1-related tests**

Run:

```bash
pytest tests/sim_eval/test_scoring.py::test_lift_score_requires_ten_cm_above_initial_height tests/sim_eval/test_scoring.py::test_task1_bin_score_requires_correct_category_released_and_static -q
```

Expected: both tests pass.

---

## Task 7: Fix Task2 Assertion To Include Follow, Grasp, And Place

**Files:**
- Modify: `src/lerobot/sim_eval/task_assert.py`
- Test: `tests/sim_eval/test_scoring.py`

- [ ] **Step 1: Extend Task2 constructor**

Add constructor arguments:

```python
follow_score_per_part: float,
follow_distance_threshold: float,
lift_delta: float,
```

Store them:

```python
self._follow_score_per_part = float(follow_score_per_part)
self._follow_distance_threshold = float(follow_distance_threshold)
self._lift_delta = float(lift_delta)
self._follow_scored_parts: set[str] = set()
self._parts_tracker = EpisodePartsTracker()
```

- [ ] **Step 2: Reset new state**

In `_reset_episode_state`, add:

```python
self._follow_scored_parts.clear()
self._parts_tracker.reset()
```

- [ ] **Step 3: Enrich part poses and get end-effector poses**

In `Task2Assertion.__call__`, after `parts_poses = robot._scene_builder.get_parts_world_poses()`, add:

```python
ee_poses = get_robot_ee_poses(robot)
self._parts_tracker.add_parts_poses(parts_poses)
parts_poses = self._parts_tracker.update_release_state(parts_poses, ee_poses)
```

- [ ] **Step 4: Compute follow score**

Before grasp scoring, add:

```python
follow_score, self._follow_scored_parts, follow_details = task2_check_end_effector_followed_parts(
    parts_poses_dict=parts_poses,
    ee_poses=ee_poses,
    scored_parts=self._follow_scored_parts,
    distance_threshold=self._follow_distance_threshold,
    score_per_part=self._follow_score_per_part,
    max_parts=self._max_parts,
)
```

- [ ] **Step 5: Compute grasp score from initial height**

In `task2_check_parts_grabbed(...)`, pass:

```python
initial_heights=self._parts_tracker.initial_heights,
lift_delta=self._lift_delta,
```

- [ ] **Step 6: Require released and static for placement**

In `task2_check_parts_in_correct_bin(...)`, pass:

```python
require_released=True,
require_static=True,
```

- [ ] **Step 7: Sum official score**

Replace:

```python
base_score = int(grab_score + sort_score)
total_score = int(sort_score)
```

with:

```python
total_score = float(follow_score + grab_score + sort_score)
base_score = total_score
```

- [ ] **Step 8: Add follow metrics**

Add these metrics:

```python
"task2_follow_score": float(follow_score),
"task2_follow_scored_count": int(len(self._follow_scored_parts)),
"task2_follow_details": follow_details,
```

- [ ] **Step 9: Pass new constructor args**

In `create_task_assertion` for task2, add:

```python
follow_score_per_part=float(getattr(args, "task2_follow_score_per_part")),
follow_distance_threshold=float(getattr(args, "task2_follow_distance_threshold")),
lift_delta=float(getattr(args, "task2_lift_delta")),
```

- [ ] **Step 10: Run task2 follow test**

Run:

```bash
pytest tests/sim_eval/test_scoring.py::test_task2_follow_score_counts_each_part_once_for_either_end_effector -q
```

Expected: test passes.

---

## Task 8: Fix Task3 Assertion To Include Grasp, Insert, And Official Time

**Files:**
- Modify: `src/lerobot/sim_eval/task_assert.py`
- Modify: `src/lerobot/sim_eval/scoring.py`
- Test: `tests/sim_eval/test_scoring.py`

- [ ] **Step 1: Extend Task3 constructor**

Add constructor arguments:

```python
grab_score_per_part: float,
insert_score_per_part: float,
max_parts: int,
lift_delta: float,
time_full_score: int,
time_full_time_seconds: float,
time_penalty_interval_seconds: float,
time_penalty_per_interval: int,
```

Store them and initialize state:

```python
self.grab_score_per_part = float(grab_score_per_part)
self.insert_score_per_part = float(insert_score_per_part)
self.max_parts = int(max_parts)
self.lift_delta = float(lift_delta)
self.time_full_score = int(time_full_score)
self.time_full_time_seconds = float(time_full_time_seconds)
self.time_penalty_interval_seconds = float(time_penalty_interval_seconds)
self.time_penalty_per_interval = int(time_penalty_per_interval)
self._grab_scored_parts: set[str] = set()
self._insert_scored_parts: set[str] = set()
self._used_slots: set[str] = set()
self._parts_tracker = EpisodePartsTracker()
```

- [ ] **Step 2: Reset new task3 state**

In `_reset_episode_state`, add:

```python
self._grab_scored_parts.clear()
self._insert_scored_parts.clear()
self._used_slots.clear()
self._parts_tracker.reset()
```

- [ ] **Step 3: Add task3 slot builder**

Add this method to `Task3Assertion`:

```python
def _build_slots(self) -> list[dict]:
    return [
        {"slot_id": "A0", "type": "A", "position": [0.54, 0.21, 1.04]},
        {"slot_id": "A1", "type": "A", "position": [0.76, 0.21, 1.04]},
        {"slot_id": "A2", "type": "A", "position": [0.98, 0.21, 1.04]},
        {"slot_id": "B0", "type": "B", "position": [0.54, 0.41, 1.04]},
        {"slot_id": "B1", "type": "B", "position": [0.76, 0.41, 1.04]},
        {"slot_id": "B2", "type": "B", "position": [0.98, 0.41, 1.04]},
    ]
```

If official slot coordinates are later exposed by the scene, replace only this method and keep the scoring function unchanged.

- [ ] **Step 4: Enrich part poses before scoring**

In `Task3Assertion.__call__`, after fetching `parts_poses`, add:

```python
ee_poses = get_robot_ee_poses(robot)
self._parts_tracker.add_parts_poses(parts_poses)
parts_poses = self._parts_tracker.update_release_state(parts_poses, ee_poses)
```

- [ ] **Step 5: Compute grasp score**

Replace the current insertion-only logic with:

```python
grab_score, self._grab_scored_parts = check_parts_lifted_from_initial_height(
    parts_poses_dict=parts_poses,
    initial_heights=self._parts_tracker.initial_heights,
    scored_parts=self._grab_scored_parts,
    lift_delta=self.lift_delta,
    score_per_part=self.grab_score_per_part,
    max_parts=self.max_parts,
)
```

- [ ] **Step 6: Compute insert score**

Add:

```python
insert_score, self._insert_scored_parts, self._used_slots = task3_check_parts_inserted_in_slots(
    parts_poses_dict=parts_poses,
    slots=self._build_slots(),
    scored_parts=self._insert_scored_parts,
    used_slots=self._used_slots,
    dist_threshold=self.dist_threshold,
    height_threshold=self.height_threshold,
    score_per_part=self.insert_score_per_part,
    max_parts=self.max_parts,
    require_released=True,
    require_static=True,
)
```

- [ ] **Step 7: Compute official total**

Use:

```python
base_score = float(grab_score + insert_score)
is_success = base_score >= float(self.success_score_threshold)
time_score = 0
if is_success:
    time_score = int(calculate_time_score(
        elapsed_seconds=elapsed,
        full_score=self.time_full_score,
        full_time_seconds=self.time_full_time_seconds,
        penalty_interval_seconds=self.time_penalty_interval_seconds,
        penalty_per_interval=self.time_penalty_per_interval,
    ))
total_score = float(base_score + time_score)
```

- [ ] **Step 8: Update metrics**

Return:

```python
metrics = {
    "task3_grab_score": float(grab_score),
    "task3_insert_score": float(insert_score),
    "task3_time_score": int(time_score),
    "task3_total_score": float(total_score),
    "task3_grab_scored_count": int(len(self._grab_scored_parts)),
    "task3_insert_scored_count": int(len(self._insert_scored_parts)),
    "task3_used_slot_count": int(len(self._used_slots)),
    "elapsed_time": float(elapsed),
    "total_score": float(total_score),
}
```

- [ ] **Step 9: Pass new constructor args**

In `create_task_assertion` for task3, add all new fields from `args`:

```python
grab_score_per_part=float(getattr(args, "task3_grab_score_per_part")),
insert_score_per_part=float(getattr(args, "task3_insert_score_per_part")),
max_parts=int(getattr(args, "task3_max_parts")),
lift_delta=float(getattr(args, "task3_lift_delta")),
time_full_score=int(getattr(args, "task3_time_full_score")),
time_full_time_seconds=float(getattr(args, "task3_time_full_time_seconds")),
time_penalty_interval_seconds=float(getattr(args, "task3_time_penalty_interval_seconds")),
time_penalty_per_interval=int(getattr(args, "task3_time_penalty_per_interval")),
```

- [ ] **Step 10: Run task3 tests**

Run:

```bash
pytest tests/sim_eval/test_scoring.py::test_task3_insert_score_is_seven_point_five_per_correct_slot tests/sim_eval/test_scoring.py::test_official_time_score_task3_boundaries -q
```

Expected: both tests pass.

---

## Task 9: Fix Task4 Assertion To Remove Collaboration Multiplier And Add Contact Gate

**Files:**
- Modify: `src/lerobot/sim_eval/task_assert.py`
- Modify: `src/lerobot/sim_eval/scoring.py`
- Test: `tests/sim_eval/test_scoring.py`

- [ ] **Step 1: Remove collaboration factor constructor args**

Remove these parameters from `Task4Assertion.__init__`:

```python
single_arm_factor: float,
bimanual_factor: float,
```

Remove these fields:

```python
self._single_arm_factor = single_arm_factor
self._bimanual_factor = bimanual_factor
```

- [ ] **Step 2: Add contact threshold constructor arg**

Add:

```python
contact_distance_threshold: float,
```

Store:

```python
self._contact_distance_threshold = float(contact_distance_threshold)
self._contacted_short_edges: set[int] = set()
self._contacted_long_edges: set[int] = set()
```

- [ ] **Step 3: Add contact tracker reset**

In `_reset_episode_state`, add:

```python
self._contacted_short_edges.clear()
self._contacted_long_edges.clear()
```

- [ ] **Step 4: Add an edge contact helper**

Add this method to `Task4Assertion`:

```python
def _update_edge_contacts(self, robot: WalkerS2sim) -> None:
    ee_poses = get_robot_ee_poses(robot)
    ee_points = []
    for value in ee_poses.values():
        vec = np.asarray(value, dtype=np.float32).reshape(-1)
        if vec.size >= 3:
            ee_points.append(vec[:3])
    if not ee_points:
        return

    box_pos, _ = robot._scene_builder.box_articulation.get_world_poses()
    center = np.asarray(box_pos, dtype=np.float32).reshape(-1)[:3]
    short_edge_points = [
        center + np.array([0.0, -0.20, 0.12], dtype=np.float32),
        center + np.array([0.0, 0.20, 0.12], dtype=np.float32),
    ]
    long_edge_points = [
        center + np.array([-0.30, 0.0, 0.12], dtype=np.float32),
        center + np.array([0.30, 0.0, 0.12], dtype=np.float32),
    ]

    for idx, edge_point in enumerate(short_edge_points):
        if min(float(np.linalg.norm(edge_point - ee)) for ee in ee_points) <= self._contact_distance_threshold:
            self._contacted_short_edges.add(idx)
    for idx, edge_point in enumerate(long_edge_points):
        if min(float(np.linalg.norm(edge_point - ee)) for ee in ee_points) <= self._contact_distance_threshold:
            self._contacted_long_edges.add(idx)
```

This is an approximate contact gate based on end-effector proximity. If Isaac Sim contact reports are available in this workspace, replace the proximity body with contact sensor reads while keeping the same `_contacted_short_edges` and `_contacted_long_edges` outputs.

- [ ] **Step 5: Apply contact gate before edge scores**

After computing raw short and long edge scores, add:

```python
short_edge_score = sum(
    self._short_edge_score_per_edge
    for idx, closed in enumerate(short_info.get("short_edge_closed", []))
    if closed and idx in self._contacted_short_edges
)
long_edge_score = sum(
    self._long_edge_score_per_edge
    for idx, closed in enumerate(long_info.get("long_edge_closed", []))
    if closed and idx in self._contacted_long_edges
)
```

- [ ] **Step 6: Remove collaboration calculations**

Delete:

```python
collaboration_stats = self._action_tracker.get_bimanual_collaboration_stats(...)
collaboration_factor = task4_get_bimanual_collaboration_factor(...)
```

Remove collaboration metrics from the metrics dict.

- [ ] **Step 7: Use official total score**

Replace calls to `task4_calculate_total_score` with:

```python
score_info = task4_calculate_total_score(
    short_edge_score=short_edge_score,
    long_edge_score=long_edge_score,
    time_score=time_score if is_success else 0,
)
```

- [ ] **Step 8: Pass contact threshold and remove collaboration args**

In `create_task_assertion` for task4, remove `single_arm_factor` and `bimanual_factor`, then add:

```python
contact_distance_threshold=float(getattr(args, "task4_contact_distance_threshold")),
```

- [ ] **Step 9: Run task4 total score test**

Run:

```bash
pytest tests/sim_eval/test_scoring.py::test_task4_total_score_has_no_collaboration_multiplier -q
```

Expected: test passes.

---

## Task 10: Full Verification And Container Smoke Runs

**Files:**
- Read: `docs/run_eval.md`
- Run: local tests and optional container evaluation smoke tests.

- [ ] **Step 1: Run all new unit tests**

Run:

```bash
pytest tests/sim_eval -q
```

Expected: all tests pass.

- [ ] **Step 2: Compile modified Python files**

Run:

```bash
python -m py_compile \
  src/lerobot/sim_eval/scoring.py \
  src/lerobot/sim_eval/task_assert.py \
  src/lerobot/sim_eval/task_eval_config.py \
  Ubtech_sim/source/SceneBuilder.py
```

Expected: exit code 0.

- [ ] **Step 3: Run one container smoke episode for each task**

Run the following only after images and Isaac Sim runtime are available:

```bash
HEADLESS=1 ./run_eval.sh task1
HEADLESS=1 ./run_eval.sh task2
HEADLESS=1 ./run_eval.sh task3
HEADLESS=1 ./run_eval.sh task4
```

Expected:
- Each task writes an `episode_*.json` file under `eval_config/logs/sim_eval_container/` or the configured log directory.
- Metrics include official score keys:
  - task1: `task1_lift_score`, `task1_box_score`, `task1_time_score`, `task1_total_score`
  - task2: `task2_follow_score`, `task2_grab_score`, `task2_sort_score`, `task2_total_score`
  - task3: `task3_grab_score`, `task3_insert_score`, `task3_time_score`, `task3_total_score`
  - task4: `task4_short_edge_score`, `task4_long_edge_score`, `task4_time_score`, `task4_total_score`

- [ ] **Step 4: Inspect score ranges in generated episode JSON**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

for path in sorted(Path("eval_config/logs/sim_eval_container").glob("episode_*.json"))[-8:]:
    data = json.loads(path.read_text())
    task_metrics = data.get("metrics", {})
    score = data.get("score")
    print(path.name, "score=", score, "keys=", sorted(k for k in task_metrics if k.endswith("_score") or k.endswith("_total_score")))
PY
```

Expected:
- No task reports a score above its official maximum.
- Task4 maximum visible total is 70 because its listed criteria sum to 70.
- Task1 and task3 time scores appear only when completion criteria are met.

- [ ] **Step 5: Commit the repair**

Run:

```bash
git add \
  Ubtech_sim/source/SceneBuilder.py \
  Ubtech_sim/config/Conveyor_Sorting.yaml \
  src/lerobot/sim_eval/scoring.py \
  src/lerobot/sim_eval/task_assert.py \
  src/lerobot/sim_eval/task_eval_config.py \
  tests/sim_eval/test_scoring.py \
  tests/sim_eval/test_task_eval_config.py
git commit -m "fix: align sim eval scoring with official criteria"
```

Expected: commit succeeds after tests pass.

---

## Self-Review

Spec coverage:
- Task1 grasp, placement, and time criteria are covered by Tasks 3, 4, and 6.
- Task2 follow, grasp, placement, and part count criteria are covered by Tasks 3, 4, and 7.
- Task3 grasp, insertion, and time criteria are covered by Tasks 3, 4, and 8.
- Task4 edge closure, contact gate, time scoring, and removal of non-official collaboration multiplier are covered by Tasks 3, 4, and 9.

Known residual risk:
- Task4 contact gating uses an end-effector proximity approximation unless Isaac Sim contact sensor data is wired in. The plan keeps this isolated in `_update_edge_contacts()` so it can be replaced with true contact reports without changing score aggregation.
- Task1 category-bin split still uses the current box Y-axis split. If official bin geometry is later provided as separate A/B volumes, replace only `task1_check_parts_in_box()` region logic and keep release/static scoring unchanged.

Verification scope:
- Unit tests prove the scoring math and official defaults.
- Container smoke runs prove integration with the Isaac Sim evaluation loop.
