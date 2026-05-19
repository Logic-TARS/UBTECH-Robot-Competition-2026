# 评估系统使用指南

## 1. 架构

双容器隔离评测，通过 WebSocket（msgpack 序列化 + lz4 压缩）通信。

```
┌─── sim-eval 容器 ──────────────────────────────────────────┐
│  • Isaac Sim 仿真引擎                                       │
│  • WalkerS2sim 机器人接口 (robot.get_observation / send_action) │
│  • Task1-4 断言评测 + 计分 + 结果落盘                        │
│  • WebSocket Client → 连 infer                             │
└───────────────────┬────────────────────────────────────────┘
                    │ control:8765 (start / episode_end / stop)
                    │ stream:8766  (observation / action)
┌───────────────────▼────────────────────────────────────────┐
│ infer 容器                                                  │
│  • 策略模型加载 (ACT / PI0 / 自定义)                         │
│  • 观测预处理 → 推理 → 返回 action                          │
│  • policy.reset() 在 episode 结束时触发                      │
└────────────────────────────────────────────────────────────┘
```

## 2. 前置条件

- Docker 镜像 `ghrc_2026:v0`（`docker build -t ghrc_2026:v0 .`）
- GPU 可用（`nvidia-smi`）
- X11 显示（非 headless 模式需 `DISPLAY` 环境变量）
- 基线模型目录

## 3. 配置文件

位于 `eval_config/`，每任务两个 YAML：

### sim-eval 侧 (`task{N}_sim_eval.yaml`)
```yaml
task: task4
task_text: packing box
task_config_path: /workspace/.../Ubtech_sim/config/Packing_Box.yaml
device: auto
num_episodes: 10
max_steps: 2000
enable_assertion: true
auto_start: true                             # 跳过按 Enter 等待

sim_infer_host: localhost                    # infer 容器地址
sim_infer_control_port: 8765
sim_infer_stream_port: 8766
```

### infer 侧 (`task{N}_infer.yaml`)
```yaml
task: task4
task_text: packing box
device: cuda:0
adapter_type: lerobot                        # 或自定义 adapter
policy_type: act
policy_path: /workspace/.../pretrained_model
websocket_server_host: 0.0.0.0
websocket_control_port: 8765
websocket_stream_port: 8766
```

任务断言参数从 `Ubtech_sim/config/*.yaml` 的 `evaluation` 段自动读取，不写在 `eval_config` 里。

## 4. 运行

```bash
# 单任务
./run_eval.sh task4

# 全部 4 个任务
./run_eval.sh all

# 也可手动两阶段
# 终端1 — infer
docker run -d --rm --name sim-infer \
    --entrypoint /bin/bash --privileged --network host --gpus all --shm-size=8g \
    -v $(pwd):/workspace/...:rw \
    -v /path/to/models:/workspace/.../challenge2026_baseline:rw \
    ghrc_2026:v0 -c '...sim_infer_container...'

# 终端2 — sim-eval
docker run --rm --name sim-eval \
    --entrypoint /bin/bash --privileged --network host --gpus all --shm-size=8g \
    -v $(pwd):/workspace/...:rw \
    -v /path/to/models:/workspace/.../challenge2026_baseline:rw \
    -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    ghrc_2026:v0 -c '...sim_eval_container...'
```

关键点：
- `--entrypoint /bin/bash` 覆盖镜像默认 `runheadless.sh`
- 用 `/isaac-sim/python.sh` 执行 Python，pip 不在默认 PATH
- 模型目录通过 `-v` bind mount，不用软链接

## 5. 源码结构

```
src/lerobot/
├── scripts/
│   ├── sim_eval_container.py    # sim-eval 容器入口
│   └── sim_infer_container.py   # infer 容器入口
├── common/sim_eval/
│   ├── common.py                # build_robot, load_policy, obs_to_tensor
│   ├── const.py                 # 任务常量
│   ├── container_config.py      # 双容器配置加载
│   ├── logger.py                # EpisodeResult, InferenceSummary
│   ├── policy_adapter.py        # 策略适配器 (LeRobot + 自定义接口)
│   ├── protocol.py              # WebSocket 消息协议 (msgpack+lz4)
│   ├── success.py               # 计分逻辑
│   ├── task_assert.py           # Task1-4 断言类 (Episode 生命周期)
│   ├── task_eval_config.py      # 从 Ubtech_sim YAML 读断言参数
│   ├── terminal.py              # 终止条件检查
│   ├── websocket_client.py      # WebSocket 客户端 (sim-eval 侧)
│   └── websocket_server.py      # WebSocket 服务端 (infer 侧)
├── configs/policies.py          # Policy 配置加载 (from_pretrained)
└── robots/walker_s2_sim/        # WalkerS2 机器人接口

eval_config/                # 运行时配置
run_eval.sh                      # 一键评估脚本
```

## 6. 适配修改

| 文件 | 修改 | 原因 |
|------|------|------|
| `policies.py` | `from_pretrained`: 先 pop `"type"` 再 draccus.parse | config.json 顶层 `"type":"act"` 不被 ACTConfig 接受 |
| `common.py` | `.resolve()` → `.absolute()` | 避免软链接解析到宿主机路径 |
| `common.py` | `obs_to_tensor`: 图像 `(H,W,C)` → `(C,H,W)` permute | WalkerS2sim HWC，模型 CHW |
| `task_assert.py` | 初始化 `terminal_no_movement`, `parts_moved` | 变量条件内赋值条件外引用 |
| `task_assert.py` | Task3 max_steps 读 `extra_info` 替代硬编码 10000 | 配置文件限制不生效 |
| `walkers2sim.py` | 新增 `get_box_joints()` | Task4 断言需要读盒子关节 |
| `sim_eval_container.py` | 新增观测扁平化 + action dict 转换 | WalkerS2sim key ↔ LeRobot 标准 key |
| `sim_eval_container.py` | 新增 `auto_start` 支持 | 跳过按 Enter，后台运行 |
| `sim_infer_container.py` | 使用 `lerobot.common.sim_eval` 导入 | 容器内通过 pip install -e . 可用的包路径 |
| `Dockerfile` | 新增 msgpack, lz4 | WebSocket 消息序列化 + 压缩 |

## 7. 通信协议

### Control 通道 (8765)
| 消息 | 方向 | 作用 |
|------|------|------|
| `start` | sim-eval → infer | 标记新 episode 开始 |
| `episode_end` | sim-eval → infer | episode 结束，触发 policy.reset() |
| `stop` | sim-eval → infer | 停止 infer 服务 |

### Stream 通道 (8766)
| 消息 | 方向 | 帧内容 |
|------|------|---------|
| `observation` | sim-eval → infer | `observation.state` (20D), images (4×3×480×640) |
| `action` | infer → sim-eval | 18-20D 动作向量 |

编码：`dict → msgpack.packb() → lz4.frame.compress() → WebSocket`。

## 8. 策略适配器接口

自定义算法需实现 `PolicyAdapter`：
```python
class PolicyAdapter:
    def load(self, model_path, device, config) -> None: ...
    def predict(self, observation, context: InferenceContext) -> Tensor: ...
    def reset(self, reset_context: ResetContext | None = None) -> None: ...
    def close(self) -> None: ...
```

在 `infer_container.yaml` 中配置：
```yaml
adapter_type: custom
adapter_class: your_package.YourModule:YourAdapter
```

## 9. 结果输出

`simeval_container` 在每个 episode 后生成 `episode_XXXX.json`，评估完成后生成 `summary_YYYYMMDD_HHMMSS.json`。

```
批量推理汇总  [TASK4 / REMOTE]
成功：0  失败：1  异常：0
成功率    : 0.00%
平均步数  : 300.0
```

`status` 枚举：`success` | `failed` | `timeout` | `error` | `rerecord`
