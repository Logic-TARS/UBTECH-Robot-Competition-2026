# Task1 评测说明：抓取并分区放入单箱

## 场景与有效参数

默认场景为 `Ubtech_sim/config/Part_Sorting.yaml`：2 个 A 件和 2 个 B 件（共 4 件），一个固定箱体，中心约为 `(1.20, 0.30, 1.05)`，箱体半尺寸按 `box_scale / 2` 得到 `(0.19, 0.375, 0.18)`。Task1 最多 10,000 step。

当前 YAML 没有 `evaluation`，实际默认参数为：每项每件 10 分、最多 4 件；抬升阈值是每件首次观测高度 + 0.10 m；成功基础门槛 80；时间满分 20（180 s 内，之后每超 30 s 向上取整扣 5，最低 0）。

## 每步判定和累计计分

```text
get_parts_world_poses + end-effector poses
  → EpisodePartsTracker：首次高度；最近 10 帧最大位移 ≤ 0.005 m 为 static；
    到所有末端距离曾大于 0.08 m 为 released（两个状态均为一次成立后永久保留）
  → 出界检查
  → 抬升、入箱的“一次性”累计分
  → 基础分达到 80 才加时间分并标记成功
```

| 项目 | 计分条件 | 分数 |
|---|---|---:|
| 抬升 | 每件 `z >= initial_z + 0.10`；同一件只计一次 | 10 × 最多 4 = 40 |
| 入箱 | 位置在以箱体中心为准的轴对齐盒内；A 必须在箱体局部 `y < 0`，B 必须在 `y >= 0`；且该件曾 released 且已 static | 10 × 最多 4 = 40 |
| 时间 | 仅当基础分 ≥80 时计算；180 s 内 20 分，每超过一个 30 s 区间扣 5 分 | 0–20 |
| episode 记录分 | `lift + box + time` | 最高 100 |

`prim_path` 是零件唯一 ID；语义优先读 `semantics`，否则从路径中的 `PART_A` / `PART_B` 推断。抬升分和入箱分是独立集合：一件只要曾抬升、之后正确入箱，两个分都可以保留，即使之后状态改变。

成功是基础分 `lift + box >= 80`，因此默认必须四件都分别抬升且正确放入；成功时 sim 立即结束该 episode。

## 失败和边界

- 任一零件不在 workspace x/y/z 区间内，断言返回 terminal。当前任务 YAML 未显式给 limits，代码的回退范围是 x `[0, 1.5]`、y `[-0.02, 0.58]`、z `[0.94, 1.14]`。
- `step >= max_steps` 时超时。外层循环本身在 `step < max_steps` 时调用断言，通常由外层超时结束，仍记录为 failed/timeout。
- 文档/代码中虽有“零件未运动终止”字段，但当前实现仅在已经成功后才计算它，而且 `terminal` 在此之前已固定；因此它**不会实际终止一个未成功 episode**。

## 输出重点

`metrics` 中的 `task1_lift_score`、`task1_box_score`、`task1_time_score`、`task1_total_score` 是结束分；`task1_*_scored_count` 可用于定位漏分；`task1_terminal_out_of_workspace` 是实际出界原因。
