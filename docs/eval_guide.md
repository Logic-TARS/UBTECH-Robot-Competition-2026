# 评估系统使用指南

## 1. 架构

双容器隔离评测，通过 WebSocket（msgpack + lz4）通信。

```
┌─── sim-eval 容器 ──────────────────────────────────────┐
│  Isaac Sim 仿真 + WalkerS2 机器人 + 断言评测 + 结果落盘  │
│  WebSocket Client → 连 infer                           │
└────────────────────┬───────────────────────────────────┘
                     │ control:8765 (start / episode_end / stop)
                     │ stream:8766  (observation / action)
┌────────────────────▼───────────────────────────────────┐
│ infer 容器                                              │
│  模型加载 → 观测推理 → 返回 action → policy.reset()     │
└────────────────────────────────────────────────────────┘
```

infer 不启 Isaac Sim app，用裸 Python 运行，避免和 sim-eval 争 GPU。

## 2. 前置条件

- Docker 镜像 `ghrc_2026:v0`
- 双 GPU（各独占一张）
- 基线模型已放置在 `challenge2026_baseline/` 下

## 3. 配置

`eval_config/` 下两个 YAML，任务名通过 `--task` CLI 覆盖。

### infer.yaml

```yaml
task: task4                       # 默认任务
device: cuda:0                    # infer 独占 GPU
adapter_type: lerobot             # 策略适配器类型
adapter_class:                    # 自定义适配器: pkg.module:ClassName
adapter_config: {}                # 传给 adapter.load() 的额外配置
policy_type: act                  # act | pi0 | diffusion
websocket_server_host: 0.0.0.0
websocket_control_port: 8765
websocket_stream_port: 8766
heartbeat_interval: 5.0
connection_timeout: 30.0
```

### sim_eval.yaml

```yaml
task: task4                       # 默认任务
device: auto
num_episodes: 1
print_every: 20
enable_assertion: true
auto_start: true                  # 跳过按 Enter 等待
log_dir: logs/sim_eval_container
action_wait_timeout: 10.0
reset_retries: 3
sim_infer_host: localhost
sim_infer_control_port: 8765
sim_infer_stream_port: 8766
heartbeat_interval: 5.0
connection_timeout: 30.0
```

max_steps、task_text、model_path 在各任务常量中有预设默认值，无需配置。

## 4. 运行

```bash
./run_eval.sh task4 --mode dual    # 单任务
./run_eval.sh all --mode dual      # 全部 4 个任务
```

## 5. PolicyAdapter 接口

接入自定义算法需实现以下接口（参考 `src/lerobot/sim_eval/policy_adapter.py`）：

```python
class PolicyAdapter(abc.ABC):
    def load(self, model_path, device, config) -> None: ...
    def predict(self, obs, context) -> Tensor: ...
    def reset(self, reset_context=None) -> None: ...
    def close(self) -> None: ...
```

- `InferenceContext`: task, task_text, episode_id, step, timestamp
- `ResetContext`: episode_id, status, reason, metrics, step

在 `infer.yaml` 中配置 `adapter_class: pkg.module:ClassName` 即可接入。

## 6. 结果

`logs/sim_eval_container/` 下生成：

- `episode_XXXX.json` — 单 episode 详情 (status, score, metrics)
- `summary_YYYYMMDD_HHMMSS.json` — 汇总 (success_rate, total_steps)

status: `success` | `failed` | `timeout` | `error`
