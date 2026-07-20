# Task2 评测说明：传送带分类分拣

## 场景与有效参数

默认 `Conveyor_Sorting.yaml` 定义 A、B 各 4 件（共 8 件），两个固定料箱。代码按料箱**世界 x 坐标**排序：较小 x 为 left，较大 x 为 right。A 应放 left，B 应放 right。最大 10,000 step。

当前没有 `evaluation` 覆盖，实际默认：跟随 2.5/件（阈值 0.10 m）、抓取 5/件（相对首次高度 +0.10 m）、分拣 5/件、每类最多 8 件、成功门槛 100。三个项目相加正好最高 20 + 40 + 40 = 100。

## 实际评分逻辑

| 项目 | 判断条件 | 计分 |
|---|---|---:|
| 跟随 | 任一末端到零件欧氏距离 `<= 0.10 m` | 2.5 × 最多 8 = 20 |
| 抓取 | `z >= initial_z + 0.10 m`；首次高度来自本 episode 的首帧 | 5 × 最多 8 = 40 |
| 正确分拣 | A 在 left 或 B 在 right 的 AABB 内；且 released、static | 5 × 最多 8 = 40 |
| 总分/成功分 | `follow_score + grab_score + sort_score`；达到 `>=100` 即成功 | 最高 100 |

静止/释放的定义与 Task1 相同：静止为最近 10 帧最大位移不超过 0.005 m；released 为曾距所有末端超过 0.08 m。各项目各自的已计分集合是累积且不回退，故不要求“分拣得分的同一件此前一定已拿到抓取得分”。

料箱命中使用未旋转的轴对齐盒，默认半尺寸为 `(0.19, 0.375, 0.18)`；并不使用料箱姿态。

## 终止

- 所有有效零件的 `z < conveyor_drop_z` 时失败。默认 `conveyor_drop_z` 取 YAML 的传送带位置 z，即 `0.0`，而非工作平面 z=1.25。
- `step >= max_steps` 为超时（外层循环通常先在 10,000 step 结束）。
- 达到 100 分时 `is_success=True`，sim 先按成功结束；即使同一帧 `terminal_lost=True`，成功分支优先处理。

## 与旧 `eval_scoring.md` 的差异

旧文档称“跟随和抓取仅统计、总分是正确分拣数 ×10”，这与当前 `Task2Assertion` 不符。运行路径没有调用 `task2_calculate_total_score()`；实际使用的是三项直接相加，且正确分拣每件默认 5 分。应以本文件为准。

## 输出重点

查看 `task2_follow/grab/sort_score` 及三类 `*_scored_count`；`*_details` 中有每件的最近距离、高度阈值、所属料箱和语义。`task2_total_score` 是该 episode 的真实记录分。
