
# `run_eval_cmd.sh` 使用说明

用于启动推理服务（infer）并在 Isaac Sim 中执行仿真评测（sim）。

---

## 基本用法

默认运行 `task4`：

```bash
./run_eval_cmd.sh
```

指定任务：

```bash
./run_eval_cmd.sh task1
./run_eval_cmd.sh task2
```

运行前需确保两个容器已经启动：

```bash
docker ps
```

应能看到：

```text
eval_infer_server
eval_sim_server
```

---

## 脚本执行流程

脚本整体流程如下：

```text
启动 infer websocket server
        ↓
等待 8765 端口 ready
        ↓
启动 Isaac Sim eval
        ↓
过滤关键日志输出
```

其中：

* infer 容器运行模型推理服务
* sim 容器运行 Isaac Sim 仿真
* 两者通过 websocket 通信

---

## 可配置参数

| 参数             | 默认值                | 作用         |
| ---------------- | --------------------- | ------------ |
| `TASK`         | `task4`             | 指定评测任务 |
| `PIP_MIRROR`   | 清华源                | pip 安装镜像 |
| `INFER_NAME`   | `eval_infer_server` | infer 容器名 |
| `SIM_NAME`     | `eval_sim_server`   | sim 容器名   |
| `CONTAINER_WS` | `/workspace/eval`   | 容器工作目录 |

---

## 环境变量覆盖方式

### 修改 pip 源

```bash
PIP_MIRROR=https://pypi.org/simple ./run_eval.sh
```

---

### 修改任务

```bash
./run_eval_cmd.sh task5
```

---

## 关键变量说明

| 变量                            | 说明                               |
| ------------------------------- | ---------------------------------- |
| `PYTHON=/isaac-sim/python.sh` | Isaac Sim 官方 Python 启动器       |
| `INFER_PY=...`                | 初始化 Isaac Sim 环境后启动 Python |
| `infer_args`                  | infer 配置参数                     |
| `sim_args`                    | sim 配置参数                       |

---

## `INFER_PY` 的作用

```bash
INFER_PY="source /isaac-sim/setup_python_env.sh && \
LD_PRELOAD=/isaac-sim/kit/libcarb.so \
/isaac-sim/kit/python/bin/python3"
```

作用：

1. 加载 Isaac Sim 环境变量
2. 预加载 `libcarb.so`
3. 使用 Isaac Sim 自带 Python

等价于：

```bash
source /isaac-sim/setup_python_env.sh
LD_PRELOAD=/isaac-sim/kit/libcarb.so \
/isaac-sim/kit/python/bin/python3
```

主要用于确保：

* omni
* carb
* pxr
* isaacsim

等模块能正常运行。

---

## 日志过滤逻辑

脚本最后：

```bash
grep -E "Episode.*step=|SUCCESS|FAILED|Score|Connected|INIT"
```

仅保留关键日志，例如：

```text
INIT
Connected
Episode step
SUCCESS
FAILED
Score
```

同时过滤 Isaac Sim 常见 warning：

```bash
grep -vE "Warning.*usd|omni.physicsschema"
```

避免日志刷屏。

---

## 常用调试命令

进入容器：

```bash
docker exec -it eval_infer_server bash
docker exec -it eval_sim_server bash
```

查看日志：

```bash
docker logs -f eval_infer_server
docker logs -f eval_sim_server
```

检查 websocket：

```bash
curl http://localhost:8765/
```

---

## 配置文件

| 文件                            | 作用         |
| ------------------------------- | ------------ |
| `eval_config/eval_infer.yaml` | 推理服务配置 |
| `eval_config/eval_sim.yaml`   | 仿真评测配置 |

脚本内部对应：

```bash
--config eval_config/eval_infer.yaml
--config eval_config/eval_sim.yaml
```

---

## 常见问题

| 问题                          | 原因                     |
| ----------------------------- | ------------------------ |
| `ModuleNotFoundError: omni` | Isaac 环境未初始化       |
| websocket 连接失败            | infer 未成功启动         |
| sim 崩溃                      | GPU / DISPLAY / 显存问题 |
| pip 安装失败                  | 网络或镜像源问题         |

---

## 推荐执行顺序

```bash
# 1. 启动容器
./start_container.sh

# 2. 执行评测
./run_eval.sh task4
```
