# GHRC Eval 双容器手动运行说明

原始脚本是自动启动两个容器并分别运行两个 Python 脚本：

- `infer` 容器：运行 `lerobot.scripts.ghrc_eval_infer`
- `sim` 容器：运行 `lerobot.scripts.ghrc_eval_sim`
- 两者通过 `host network` 上的 WebSocket 通信，默认端口是 `8765`

现在拆分为：先分别进入两个容器，再在容器内部手动启动对应 Python 脚本。

---

## 1. 文件说明

| 文件 | 作用 |
|---|---|
| `env.sh` | 公共环境变量和基础检查 |
| `enter_infer.sh` | 进入 infer 容器 |
| `enter_sim.sh` | 进入 sim 容器 |
| `run_infer_inside_container.sh` | 在 infer 容器内部运行 infer Python 脚本 |
| `run_sim_inside_container.sh` | 在 sim 容器内部运行 sim Python 脚本 |

建议把这几个文件放在项目根目录，也就是包含 `eval_config/`、`lerobot/`、`pyproject.toml` 或 `setup.py` 的目录。

---

## 2. 前置条件

确认 Docker 已安装，并且两个镜像已经存在：

```bash
docker image inspect ghrc-eval-infer:latest
docker image inspect ghrc-eval-sim:latest
```

给脚本增加执行权限：

```bash
chmod +x env.sh enter_infer.sh enter_sim.sh
chmod +x run_infer_inside_container.sh run_sim_inside_container.sh
```

---

## 3. 启动 infer 容器

打开第一个终端，在项目根目录执行：

```bash
./enter_infer.sh task4
```

进入容器后，执行：

```bash
./run_infer_inside_container.sh task4
```

或者直接运行 Python 命令：

```bash
cd /workspace/eval

/isaac-sim/python.sh -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e . --no-deps -q
/isaac-sim/python.sh -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple lz4 msgpack websockets pillow -q

source /isaac-sim/setup_python_env.sh
LD_PRELOAD=/isaac-sim/kit/libcarb.so /isaac-sim/kit/python/bin/python3 \
  -m lerobot.scripts.ghrc_eval_infer \
  --config eval_config/eval_infer.yaml \
  --task task4
```

看到 infer 服务启动后，保持这个终端不要关闭。

---

## 4. 确认 infer WebSocket 就绪

在宿主机另开一个终端执行：

```bash
curl -v http://localhost:8765/
```

只要端口能连上，说明 infer 容器已经监听成功。

如果连接失败，先检查 infer 容器日志或 infer 终端输出。

---

## 5. 启动 sim 容器

打开第二个终端，在项目根目录执行：

```bash
./enter_sim.sh task4
```

进入容器后，执行：

```bash
./run_sim_inside_container.sh task4
```

或者直接运行 Python 命令：

```bash
cd /workspace/eval

/isaac-sim/python.sh -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e . --no-deps -q

/isaac-sim/python.sh \
  -m lerobot.scripts.ghrc_eval_sim \
  --config eval_config/eval_sim.yaml \
  --task task4
```

sim 端会连接 infer 端的 WebSocket，并开始执行评估。

---

## 6. 运行其他任务

只需要把 `task4` 替换为对应任务名：

```bash
./enter_infer.sh task1
./run_infer_inside_container.sh task1
```

```bash
./enter_sim.sh task1
./run_sim_inside_container.sh task1
```

支持的任务通常是：

```bash
task1
task2
task3
task4
```

---

## 7. 常用环境变量

可以在宿主机启动容器前覆盖默认配置：

```bash
export INFER_IMAGE=ghrc-eval-infer:latest
export SIM_IMAGE=ghrc-eval-sim:latest
export HEADLESS=1
export PIP_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple
export ISAAC_CACHE_ROOT=$HOME/.cache/isaac_sim_container
export HF_CACHE=$HOME/.cache/huggingface
```

然后再执行：

```bash
./enter_infer.sh task4
./enter_sim.sh task4
```

---

## 8. 关闭容器

如果 infer 或 sim 容器没有自动退出，可以在宿主机执行：

```bash
docker rm -f eval_infer_task4
docker rm -f eval_sim_task4
```

---

## 9. 结果路径

评估结果默认在项目目录下：

```bash
eval_config/logs/sim_eval_container/
```

---

## 10. 推荐运行顺序

完整流程如下：

```bash
# 终端 1：启动 infer 容器
./enter_infer.sh task4

# infer 容器内部
./run_infer_inside_container.sh task4
```

```bash
# 终端 2：确认端口
curl -v http://localhost:8765/
```

```bash
# 终端 3：启动 sim 容器
./enter_sim.sh task4

# sim 容器内部
./run_sim_inside_container.sh task4
```
