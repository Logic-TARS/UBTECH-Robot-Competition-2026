# GHRC 仿真评测：数据流与结果链路

本文描述当前仓库的实际运行链路（以 `src/lerobot/scripts/ghrc_eval_*.py` 与 `src/lerobot/sim_eval/` 为准）。任务特有的规则见 `eval_task1.md` 至 `eval_task4.md`。

## 运行组成

```text
评测编排器 (ghrc_eval_orchestrator.py)
  ├─ 读取飞书/本地选手镜像信息，拉取镜像
  ├─ 调用 run_eval.sh，按 task 列表启动一次双容器评测
  └─ 读取各 task 的 summary_*.json，以「任务成功率的算术平均」写回最终 0–100 分

sim-eval 容器 (ghrc_eval_sim.py)                  infer 容器 (ghrc_eval_infer.py)
  Isaac Sim / WalkerS2sim                           PolicyAdapter / 选手策略
  get_observation()                                 接收 observation
      │ flatten_obs()                                adapter.predict()
      │ observation (stream, 二进制) ───────────────► │
      │                                               │ action (stream, 二进制)
      ◄──────────────────────────────────────────────┘
  action_to_dict() → robot.send_action() → robot.step()
      │
      └─ TaskAssertion(robot, step, action) → score / success / terminal / metrics
```

控制通道与数据通道均为 WebSocket：控制端口默认 8765，流端口默认 8766。`start` 激活推理，`episode_end` 携带结果并使 infer 侧执行 policy reset，结束后 sim 还会执行 `robot.reset()`。消息以 `episode_id + step` 配对，infer 会丢弃非当前 episode 的观测。

## 单个 episode 的实际顺序

1. sim 发送 `start(episode_id)`。
2. 每个 step 先取仿真观测并 `flatten_obs`：状态只保留 14 个臂关节、4 个手指关节和 2 个夹爪值；相机使用 `observation.images.*` 键。
3. stream 协议将数值打包为二进制，图像 JPEG 编码后 LZ4/msgpack 传给 infer；infer 解码后调用 adapter，回传同一 `(episode_id, step)` 的动作。
4. sim 接收动作；Task4 的 18 维动作会补两个 0 到 20 维。动作被映射为 14 个臂关节、4 个手指关节、左右夹爪，随后推进一帧物理仿真。
5. 若启用 `enable_assertion`（`eval_sim.yaml` 当前为 `true`），断言读取最新场景状态并给出累计指标。成功或终止即结束 episode；否则继续到 `max_steps`。infer action 等待超过 `action_wait_timeout`（默认 10 秒）会记为 `error`。

## 配置优先级与默认值

- CLI `--task` 覆盖 `eval_config/*.yaml` 的 `task`。
- 未指定 `task_config_path` 时，依次使用 `Ubtech_sim/config/Part_Sorting.yaml`、`Conveyor_Sorting.yaml`、`Foam_Inlaying.yaml`、`Packing_Box.yaml`。
- sim 在初始化时读取任务 YAML 的 `evaluation` 段；当前四个默认任务 YAML **均没有**该段，因此采用 `task_eval_config.py` 中的默认评分参数。
- max steps 与任务 YAML 的 `timelimit` 无关：task1–3 为 10,000，task4 为 2,000；可由 sim YAML 的 `max_steps` 覆盖。
- infer 使用 `task_policy_paths[task]` 或 `policy_path` 选择模型。若设置 `adapter_class`，则改由自定义 adapter 加载和推理。

## 结果与总成绩

sim 写入 `log_dir`（默认配置解析后为 `logs/sim_eval_container`）：

- `episode_0000.json`：`status`、步数、当前结束分数、reason 与完整 `metrics`。
- `summary_YYYYMMDD_HHMMSS.json`：各 episode 详情、成功/失败/异常数和 `success_rate`。

编排器只读取 `summary` 的 `success_rate`，每个任务转换为百分比，再对存在结果的任务做无权算术平均。也就是说，飞书中的最终成绩是**任务完成率平均**，不是四项任务 `episode.score` 的平均；episode 内的 0–100 分主要用于记录、调试和成功判定。
