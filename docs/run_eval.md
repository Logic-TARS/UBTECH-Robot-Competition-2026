# run_eval.sh — 双容器评估脚本

双容器隔离评测：infer 容器（CPU）运行策略推理，sim-eval 容器（GPU）运行 Isaac Sim 仿真，两者通过 WebSocket（msgpack + lz4）通信。

## 用法

```bash
./run_eval.sh task4    # 运行单个任务
./run_eval.sh all      # 按顺序运行全部 4 个任务 (task4 → task1 → task2 → task3)
```

## 前置条件

- 已构建 `ghrc-eval-infer:latest` 和 `ghrc-eval-sim:latest` 镜像
- 基线模型放置于 `challenge2026_baseline/`
- 非 headless 模式需要 X11 显示

### 构建镜像

```bash
docker build -f docker/Dockerfile.eval_infer -t ghrc-eval-infer:latest .
docker build -f docker/Dockerfile.eval_sim   -t ghrc-eval-sim:latest .
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `INFER_IMAGE` | `ghrc-eval-infer:latest` | 推理容器镜像 |
| `SIM_IMAGE` | `ghrc-eval-sim:latest` | 仿真容器镜像 |
| `ISAAC_CACHE_ROOT` | `~/.cache/isaac_sim_container` | Isaac Sim 缓存根目录 |
| `HF_CACHE` | `~/.cache/huggingface` | HuggingFace 缓存 |
| `HEADLESS` | `0` | 设为 `1` 开启无头模式 |
| `PIP_MIRROR` | `https://pypi.tuna.tsinghua.edu.cn/simple` | pip 镜像源 |

## 架构

```
┌─────────────────┐     WebSocket (msgpack+lz4)     ┌─────────────────┐
│  infer 容器 (CPU) │ ◄──────────────────────────────► │ sim-eval 容器 (GPU)│
│  kit 裸 Python    │                                 │  Isaac Sim        │
│  策略推理         │                                 │  仿真 + 评估       │
└────────┬────────┘                                  └────────┬─────────┘
         │                                                     │
    宿主机项目目录（rw 挂载）                            Isaac Sim 缓存
                                                        HF 模型缓存
```

## 执行流程

1. **启动 infer 容器**（后台运行，日志写入 `/tmp/eval_infer_<task>.log`）
   - 进入容器后先 `pip install -e .` + 安装依赖（lz4, msgpack, websockets, pillow）
   - 运行 `lerobot.scripts.ghrc_eval_infer`，等待 WebSocket 连接
2. **等待 infer 就绪** — 轮询 `http://localhost:8765/`，最多等待 120 秒
3. **启动 sim-eval 容器**（前台运行）
   - `pip install -e .` 安装源码
   - 运行 `lerobot.scripts.ghrc_eval_sim`，连接到 infer 的 WebSocket 进行协同评估
4. **sim-eval 完成后**，停止 infer 容器
5. **下一个任务**（`all` 模式）

## 任务列表

| 任务 | 说明 |
|------|------|
| task1 | 任务 1 |
| task2 | 任务 2 |
| task3 | 任务 3 |
| task4 | 任务 4 |

`all` 模式按 task4 → task1 → task2 → task3 顺序执行。

## 输出

sim-eval 容器的关键行会实时过滤输出（匹配 `Episode.*step=`、`SUCCESS`、`FAILED`、`Score` 等关键字），完整日志写入标准输出。

结果文件位于 `eval_config/logs/sim_eval_container/`：
- `episode_*.json` — 每个 episode 的详细数据
- `summary_*.json` — 汇总评分

## 配置

评估参数通过 YAML 配置文件控制：
- `eval_config/eval_infer.yaml` — 推理端配置：`device: cpu`、`policy_type`（act / pi0 / diffusion）
- `eval_config/eval_sim.yaml` — 仿真端配置：`device: auto`、`num_episodes`、`enable_assertion`

任务名通过 `--task` CLI 参数覆盖。

## 自定义算法接入

1. 参考 `src/lerobot/sim_eval/policy_adapter.py` 实现 `PolicyAdapter` 子类
2. 在 `eval_config/eval_infer.yaml` 中配置 `adapter_class` 指向你的实现
3. 运行 `./run_eval.sh <task>` 即可使用自定义策略评估
