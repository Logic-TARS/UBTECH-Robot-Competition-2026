可以把当前 `run_eval.sh` 里的逻辑拆成「手动进入容器 + 手动执行 Python」的方式。
下面给你一个更适合调试/开发的运行文档。

---

# GHRC Eval 手动运行文档

## 1. 启动容器

先启动两个容器：

```bash
./start_containers.sh
```

确认容器正常：

```bash
docker ps
```

应该能看到：

```text
eval_infer_server
eval_sim_server
```

---

# 2. 进入 infer 容器

```bash
docker exec -it eval_infer_server bash
```

进入后：

```bash
cd /workspace/eval
```

---

# 3. 安装 Python 环境（infer）

只需要第一次执行。

## 3.1 安装项目

```bash
/isaac-sim/python.sh -m pip install \
-i https://pypi.tuna.tsinghua.edu.cn/simple \
-e . \
--no-deps
```

## 3.2 安装依赖

```bash
/isaac-sim/python.sh -m pip install \
-i https://pypi.tuna.tsinghua.edu.cn/simple \
lz4 msgpack websockets pillow
```

---

# 4. 启动 infer server

在 infer 容器内执行：

```bash
source /isaac-sim/setup_python_env.sh

LD_PRELOAD=/isaac-sim/kit/libcarb.so \
/isaac-sim/kit/python/bin/python3 \
-m lerobot.scripts.ghrc_eval_infer \
--config eval_config/eval_infer.yaml \
--task task4
```

如果成功，会看到类似：

```text
WebSocket server started
Listening on 8765
```

这个终端保持不要关闭。

---

# 5. 新开终端进入 sim 容器

宿主机新开一个 terminal：

```bash
docker exec -it eval_sim_server bash
```

进入后：

```bash
cd /workspace/eval
```

---

# 6. 安装 sim 环境

只需第一次执行：

```bash
/isaac-sim/python.sh -m pip install \
-i https://pypi.tuna.tsinghua.edu.cn/simple \
-e . \
--no-deps
```

---

# 7. 运行 sim eval

在 sim 容器中执行：

```bash
/isaac-sim/python.sh \
-m lerobot.scripts.ghrc_eval_sim \
--config eval_config/eval_sim.yaml \
--task task4
```

---

# 8. 常用任务切换

例如：

```bash
--task task1
--task task2
--task task3
--task task4
```

---

# 9. 查看 infer 是否正常

宿主机执行：

```bash
curl http://localhost:8765/
```

如果端口通说明 infer 已启动。

---

# 10. 停止容器

退出后：

```bash
./stop_containers.sh
```

---

# 推荐调试方式

开发时建议：

## terminal 1

infer server

## terminal 2

sim evaluator

## terminal 3

实时查看日志：

```bash
docker logs -f eval_infer_server
```

或者：

```bash
docker logs -f eval_sim_server
```

---

# 推荐增加 alias（可选）

宿主机：

```bash
alias infer='docker exec -it eval_infer_server bash'
alias sim='docker exec -it eval_sim_server bash'
```

以后直接：

```bash
infer
sim
```

即可进入容器。
