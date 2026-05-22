# 评估系统使用指南

双容器隔离评测，WebSocket（msgpack + lz4）通信。infer 用 kit 裸 Python 跑 CPU 推理，sim-eval 用 Isaac Sim 独占 GPU。

## 前置条件

- 已构建 `ghrc-eval-infer` / `ghrc-eval-sim` 镜像
- 基线模型放置于 `challenge2026_baseline/`
- 12G 显存可正常运行

### 构建镜像

```bash
docker build -f docker/Dockerfile.eval_infer -t ghrc-eval-infer:latest .
docker build -f docker/Dockerfile.eval_sim   -t ghrc-eval-sim:latest .
```

## 配置

`eval_config/eval_infer.yaml` 和 `eval_config/eval_sim.yaml`，任务名通过 `--task` CLI 覆盖。关键字段：

- `eval_infer.yaml`: `device: cpu`（12G 显存配置），`policy_type`（act / pi0 / diffusion）
- `eval_sim.yaml`: `device: auto`，`num_episodes`，`enable_assertion`

## 运行

```bash
./run_eval.sh task4    # 单任务
./run_eval.sh all      # 全部 4 个任务
```

## 结果

`logs/sim_eval_container/` 下生成 `episode_*.json` 和 `summary_*.json`。

## 自定义算法接入

参考 `src/lerobot/sim_eval/policy_adapter.py`，实现 `PolicyAdapter` 后在 `eval_infer.yaml` 配置 `adapter_class` 即可。
