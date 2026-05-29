# Evaluation Guide

This project uses a dual-container evaluation pipeline over WebSocket (`msgpack + lz4`):

- `infer`: runs policy inference on CPU and exposes control/stream WebSocket services
- `sim-eval`: runs Isaac Sim on GPU, connects to `infer`, executes episodes, and writes evaluation results

The standard entrypoint is [run_eval.sh](D:/embodied_ai/challengeBaseline_newFramework/run_eval.sh:1).

## Prerequisites

- Docker is available on the host
- `ghrc-eval-infer:latest` and `ghrc-eval-sim:latest` have been built
- baseline weights are present under `challenge2026_baseline/`
- GPU resources are available for the `sim-eval` container

Build images:

```bash
docker build -f docker/Dockerfile.eval_infer -t ghrc-eval-infer:latest .
docker build -f docker/Dockerfile.eval_sim   -t ghrc-eval-sim:latest .
```

## Configuration

The runtime reads:

- `eval_config/eval_infer.yaml`
- `eval_config/eval_sim.yaml`

The task name in those YAML files is overridden by `./run_eval.sh <task>`.

Important fields:

- `eval_infer.yaml`
  - `policy_type`
  - `websocket_control_port`
  - `websocket_stream_port`
  - `device`
- `eval_sim.yaml`
  - `num_episodes`
  - `enable_assertion`
  - `action_wait_timeout`
  - `device`

## Run

Single task:

```bash
./run_eval.sh task4
```

All tasks:

```bash
./run_eval.sh all
```

Recommended preflight:

```bash
bash -n run_eval.sh
```

## Runtime Behavior

`run_eval.sh` is structured as a production-style runner:

1. Starts the `infer` container in the background
2. Waits until the configured WebSocket control/stream ports are actually listening
3. Starts the `sim-eval` container in the foreground
4. Streams filtered key logs to the terminal
5. Writes full per-task logs to a run-scoped directory
6. Cleans up containers automatically on success, failure, or interrupt

If `infer` exits early or does not expose its ports before timeout, the runner fails fast and prints the tail of the `infer` log.

## Logs

Each run gets a unique `RUN_ID` and a dedicated log directory:

```text
~/.cache/challenge_baseline_runner/<RUN_ID>/
└── task4/
    ├── infer.log
    └── sim.log
```

Default log root:

```text
~/.cache/challenge_baseline_runner
```

The runner prints:

- `run_id`
- run log root
- per-task `infer.log`
- per-task `sim.log`

Follow logs during execution:

```bash
tail -f ~/.cache/challenge_baseline_runner/<RUN_ID>/task4/infer.log
tail -f ~/.cache/challenge_baseline_runner/<RUN_ID>/task4/sim.log
```

Evaluation result JSON files are still written by `sim-eval` into its configured result directory, typically:

```text
logs/sim_eval_container/
```

## Environment Variables

Common runtime overrides:

```bash
export INFER_IMAGE=ghrc-eval-infer:latest
export SIM_IMAGE=ghrc-eval-sim:latest
export INFER_CONFIG=eval_config/eval_infer.yaml
export SIM_CONFIG=eval_config/eval_sim.yaml
export INFER_READY_TIMEOUT=300
export LOG_ROOT=$HOME/.cache/challenge_baseline_runner
export RUN_ID=manual-test-001
export HEADLESS=1
export RUNTIME_BOOTSTRAP=1
```

Notes:

- `LOG_ROOT`: must be writable by the current host user
- `RUN_ID`: allows deterministic log grouping
- `INFER_READY_TIMEOUT`: controls how long the runner waits for infer WebSocket readiness
- `HEADLESS=1`: default and recommended for servers, CI, and production runners
- `RUNTIME_BOOTSTRAP=1`: installs Python dependencies at runtime inside containers

For local desktop debugging with a visible window:

```bash
export HEADLESS=0
./run_eval.sh task4
```

## Production Recommendation

For production or CI usage:

```bash
export RUNTIME_BOOTSTRAP=0
```

and pre-install dependencies into the Docker images instead of relying on runtime `pip install`.

Reasons:

- faster startup
- fewer network-related failures
- more reproducible runs
- cleaner operational behavior

## Troubleshooting

### 1. No terminal logs after `启动 sim-eval 容器...`

The terminal only shows filtered key lines. Open the full log:

```bash
tail -f ~/.cache/challenge_baseline_runner/<RUN_ID>/task4/sim.log
```

### 2. Infer not ready / connection issues

Check whether the configured ports are occupied:

```bash
ss -ltnp | grep -E ':8765|:8766'
```

Check the infer log:

```bash
tail -n 200 ~/.cache/challenge_baseline_runner/<RUN_ID>/task4/infer.log
```

### 3. Log directory permission denied

Set a writable log root:

```bash
export LOG_ROOT=$HOME/.cache/challenge_baseline_runner
./run_eval.sh task4
```

### 4. Validate containers manually

Check running containers:

```bash
docker ps
docker ps -a
```

Inspect logs directly:

```bash
docker logs eval_infer_task4
docker logs eval_sim_task4
```

## Custom Policy Adapter

To use a custom policy implementation, follow [policy_adapter.py](D:/embodied_ai/challengeBaseline_newFramework/src/lerobot/sim_eval/policy_adapter.py:1) and configure `adapter_class` in `eval_infer.yaml`.
