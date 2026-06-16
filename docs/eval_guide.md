# GHRC Evaluation System User Guide

This document is intended for evaluation administrators, contestant teams, and local reproduction personnel. It covers the operation of the GHRC evaluation system, configuration boundaries, custom policy integration entry points, and automated orchestration workflows.

The evaluation system uses a dual-container isolation architecture: the `infer` container loads contestant policies and provides action inference services via WebSocket, while the `sim-eval` container starts Isaac Sim, executes tasks, collects results, and generates scoring logs. The two containers communicate via WebSocket using msgpack + lz4 encoded observation and action data.

---

## 1. Document Navigation

| Document | Applicable Scenario |
| --- | --- |
| This document | Evaluation system deployment, configuration, execution, troubleshooting, and automated orchestration |
| [Custom Policy Integration Guide](custom_policy.md) | Contestants needing to integrate custom policies, modify `adapter_class`, or verify action/observation interfaces |
| [External Algorithm Migration Example](external_algorithm_migration.md) | Contestants migrating an existing algorithm project folder into the eval repo, wrapped with `PolicyAdapter` |

For custom policy integration, read the [Custom Policy Integration Guide](custom_policy.md) first. If the policy code comes from another complete project directory, then continue with the [External Algorithm Migration Example](external_algorithm_migration.md).

---

## 2. System Responsibility Boundaries

| Module | Container | Primary Responsibility | Contestant May Modify? |
| --- | --- | --- | --- |
| `ghrc_eval_infer.py` | infer | Load config, load policy, start WebSocket inference service | Not recommended; extend via YAML and `adapter_class` instead |
| `PolicyAdapter` | infer | Convert evaluation observations to contestant model inputs and return actions | Allowed — add custom adapters |
| `ghrc_eval_sim.py` | sim-eval | Start simulation, connect to infer, execute episodes, decode actions | Not recommended |
| `src/lerobot/sim_eval` | infer / sim-eval | Communication protocol, container config, assertions, scoring logic | Modifying protocol, assertions, and scoring logic is prohibited |
| `ghrc_eval_orchestrator.py` | host | Batch pull images, invoke evaluation scripts, write back results | Maintained by the evaluation platform; contestants generally do not need to modify |

In the contest deliverables, contestants should integrate policies via:

- Standard LeRobot checkpoint: modify `policy_type`, `policy_path`, or `task_policy_paths` in `eval_config/eval_infer.yaml`.
- Non-standard policy or external project: add a custom `PolicyAdapter` and configure `adapter_class` in `eval_config/eval_infer.yaml`.

---

## 3. Prerequisites

| Item | Requirement |
| --- | --- |
| Images | `ghrc-eval-infer:latest` and `ghrc-eval-sim:latest` built |
| Models | LeRobot checkpoint or custom policy weights placed in a container-accessible path |
| GPU | `sim-eval` container requires an available NVIDIA GPU; 12 GB VRAM minimum |
| Resources | Simulation assets, task configs, and base dependencies prepared per contest instructions |
| Network | Host ports 8765 / 8766 not occupied by other processes |

### Building Images

```bash
docker build -f docker/Dockerfile.eval_infer -t ghrc-eval-infer:latest .
docker build -f docker/Dockerfile.eval_sim   -t ghrc-eval-sim:latest .
```

---

## 4. Configuration File Descriptions

| Configuration File | Purpose | Commonly Modified Items |
| --- | --- | --- |
| `eval_config/eval_infer.yaml` | infer container config | `task`, `device`, `policy_type`, `policy_path`, `task_policy_paths`, `adapter_class`, `adapter_config` |
| `eval_config/eval_sim.yaml` | sim-eval container config | `task`, `device`, `num_episodes`, `auto_start`, `enable_assertion`, WebSocket connection port |
| `eval_config/eval_orchestrator.yaml` | Automated orchestration config | Feishu spreadsheet, image registry, task list, timeout, results directory |

Task names can be overridden via CLI `--task` or script parameters. For official evaluation, passing the task name via script is recommended so the same config can be reused across `task1` through `task4`.

### 4.1 Multi-Task Policy Paths

The four tasks typically correspond to four different checkpoints. Do not rely on hardcoded default paths in source code; explicitly configure them in `eval_config/eval_infer.yaml`:

```yaml
policy_type: act
policy_path: null
require_task_policy_paths: true
task_policy_paths:
  task1: ../challenge2026_baseline/task1/act/pretrained_model
  task2: ../challenge2026_baseline/task2/act/pretrained_model
  task3: ../challenge2026_baseline/task3/act/pretrained_model
  task4: ../challenge2026_baseline/task4/act/pretrained_model
```

Path resolution rules:

- Absolute paths are used as-is.
- Relative paths are resolved relative to the YAML file's directory. For example, `../xxx` in `eval_config/eval_infer.yaml` resolves to `xxx` under the repository root.
- When `require_task_policy_paths: true`, missing an explicit path for the current task will raise an error, preventing accidental use of a default model.

### 4.2 Custom Adapter Configuration

Custom policies are integrated via `adapter_class`:

```yaml
adapter_class: my_team_policy.ghrc_adapter:MyAdapter
adapter_config:
  action_dim: 20
policy_type: null
policy_path: /workspace/eval/my_team_policy/checkpoints/best.pt
```

For detailed interfaces, the zero-action example, and external project migration, see the [Custom Policy Integration Guide](custom_policy.md).

### 4.3 Environment Variable Overrides

Common runtime environment variables:

```bash
export INFER_IMAGE=ghrc-eval-infer:latest
export SIM_IMAGE=ghrc-eval-sim:latest
export INFER_CONFIG=eval_config/eval_infer.yaml
export SIM_CONFIG=eval_config/eval_sim.yaml
export INFER_READY_TIMEOUT=300
export HEADLESS=1
```

| Variable | Description |
| --- | --- |
| `INFER_IMAGE` | infer container image |
| `SIM_IMAGE` | sim-eval container image |
| `INFER_CONFIG` | infer configuration file path |
| `SIM_CONFIG` | sim-eval configuration file path |
| `INFER_READY_TIMEOUT` | Maximum seconds to wait for infer WebSocket port readiness |
| `HEADLESS=1` | Headless mode, suitable for servers and CI |
| `HEADLESS=0` | Desktop debug mode, displays the Isaac Sim window |

---

## 5. Local Evaluation Execution

### 5.1 Running

```bash
./run_eval.sh task4
./run_eval.sh all
```

Execution flow:

1. Start the infer container in the background.
2. Wait for WebSocket control and data ports to become ready (default 8765 / 8766).
3. Start the sim-eval container in the foreground.
4. sim-eval connects to infer, sends observations step by step, and receives actions.
5. After each episode, evaluation results and logs are written.

### 5.2 Output Directories

| Output | Path |
| --- | --- |
| infer key logs | `/tmp/eval_infer_{task}.log` |
| sim-eval results | `logs/sim_eval_container/` |
| Automated orchestration cache | `logs/eval_cache.csv` |

---

## 6. Custom Policy Integration

The evaluation system only requires that the policy ultimately implements a unified inference interface: receive the current observation, return a one-dimensional action. Choose the integration method based on complexity:

| Integration Method | Applicable Scenario | Document |
| --- | --- | --- |
| LeRobot default adapter | Checkpoint conforms to LeRobot `from_pretrained` and `select_action` interface | [Custom Policy Integration Guide](custom_policy.md) |
| Custom `PolicyAdapter` | Custom PyTorch, ONNX, TensorRT, RL, planner, or hybrid algorithm | [Custom Policy Integration Guide](custom_policy.md) |
| External project migration | Policy comes from another complete project folder, need to preserve original directory structure | [External Algorithm Migration Example](external_algorithm_migration.md) |

The repository provides two minimal verification examples:

| Example | Purpose |
| --- | --- |
| `eval_config/eval_infer_zero_action.yaml` | Outputs all-zero actions to verify infer/sim communication and action decoding pipeline |
| `eval_config/eval_infer_external_random.yaml` | Uses an external project directory to output random actions, verifying project import, adapter loading, and migration structure |

Note: `curl http://localhost:8765/` returning `426 Upgrade Required` indicates the port is a WebSocket service, not a regular HTTP page. This usually means the infer service is listening — connect using sim-eval or a WebSocket client.

---

## 7. Troubleshooting

| Problem | What to Check |
| --- | --- |
| infer unresponsive after startup | Check `docker logs eval_infer_{task}` and `/tmp/eval_infer_{task}.log` |
| `curl` returns 426 | Normal; port 8765 is a WebSocket control port, not a browser HTTP service |
| Isaac Sim open but robot not moving | Check whether `auto_start` is `true` in `eval_config/eval_sim.yaml`; `false` waits for keyboard Enter |
| sim-eval connection failure | Confirm infer container is still running and ports 8765 / 8766 are mapped correctly |
| Policy path error | Check whether `policy_path` or `task_policy_paths` is a container-accessible path |
| `adapter_class` import failure | Confirm module path is importable, project directory is in `PYTHONPATH`, and each package directory contains `__init__.py` |
| Action dimension anomaly | Check whether `predict()` returns a one-dimensional `torch.Tensor`, `np.ndarray`, or `list[float]` |
| GPU out of memory | 12 GB VRAM is the minimum requirement; reduce `num_episodes` or use a GPU with more VRAM |

---

## 8. Automated Orchestration

The orchestration system reads contestant-submitted images from a Feishu multi-dimensional table, automatically pulls images, runs evaluations, collects results, and writes back scores.

### 8.1 Prerequisites

| Item | Requirement |
| --- | --- |
| Feishu App | Create an enterprise self-built app and obtain App ID / App Secret |
| Table Permissions | App added to the target multi-dimensional table |
| Image Registry | Evaluation machine has access to Docker Hub, ACR, or enterprise private registry |
| Local Environment | Docker, NVIDIA Container Toolkit, and evaluation images configured |

### 8.2 Feishu Configuration

1. Open the [Feishu Developer Console](https://open.feishu.cn), create an enterprise self-built app, and record the **App ID** and **App Secret**.
2. In the app details under Permission Management, select permissions and publish a new version.

| Permission | Purpose |
| --- | --- |
| `bitable:app` | Read multi-dimensional table |
| `base:record:update` | Write evaluation status and scores |

3. Open the Feishu table, select the menu in the upper-right corner, and add the document app.
4. Obtain `BITABLE_APP_TOKEN` and `TABLE_ID` from the table URL.

```text
https://xxx.feishu.cn/base/BITABLE_APP_TOKEN?table=TABLE_ID
```

Recommended table column structure:

| Column Name | Type | Description |
| --- | --- | --- |
| Team ID | Text | Unique identifier |
| Team Name | Text | Display name |
| Image Name | Link or Text | Contestant Docker image address |
| Review Status | Text | Initial value `评测中` |
| Score | Text | Automatically written by the system |

### 8.3 Orchestrator Configuration

`eval_config/eval_orchestrator.yaml` example:

```yaml
feishu:
  bitable_app_token: "xxx"
  table_id: "xxx"
registry:
  server: "your-registry.example.com"
eval:
  sim_image: "ghrc-eval-sim:latest"
  tasks: ["task4", "task1", "task2", "task3"]
  single_task_timeout: 7200
  results_dir: "logs/sim_eval_container"
```

### 8.4 Credential Configuration

```bash
cp .env.example .env
```

Fill in `.env`:

```bash
FEISHU_APP_ID=cli_xxxxxxxx
FEISHU_APP_SECRET=xxxxxxxx
DOCKER_USERNAME=xxxx
DOCKER_PASSWORD=xxxx
```

### 8.5 Manual Execution

```bash
source .env
export FEISHU_APP_ID FEISHU_APP_SECRET DOCKER_USERNAME DOCKER_PASSWORD
PYTHONPATH=src python -m lerobot.scripts.ghrc_eval_orchestrator
```

Common parameters:

| Parameter | Description |
| --- | --- |
| `--keep-images` | Keep contestant images after evaluation |
| `--skip-docker` | Skip image pull, suitable for local debugging |
| `--mock-eval` | Generate mock scores, suitable for Feishu write-back integration testing |
| `--mock` | Full mock mode, no Feishu or Docker access |
| `-v` | Verbose logging |

### 8.6 Scheduled Execution

```bash
crontab -e
```

```cron
*/30 * * * * source /path/to/.env && cd /path/to/project && export FEISHU_APP_ID FEISHU_APP_SECRET DOCKER_USERNAME DOCKER_PASSWORD && PYTHONPATH=src python -m lerobot.scripts.ghrc_eval_orchestrator >> /var/log/eval_orchestrator.log 2>&1
```

In production, use `flock` or a scheduler platform locking mechanism to avoid multiple evaluation tasks competing for the GPU.

### 8.7 Orchestration Statuses

| Status | Meaning |
| --- | --- |
| `评测中` | Pending evaluation; the orchestrator will pick up this record |
| `评测完成` | Evaluation succeeded; score written back |
| `评测失败` | Evaluation run exception or timeout |
| `镜像异常` | Image pull, login, or format exception |

Evaluated records are automatically skipped. To re-evaluate, manually change the status back to `评测中`.

### 8.8 Orchestrator Troubleshooting

| Problem | What to Check |
| --- | --- |
| Feishu read/write failure | Whether permissions are granted and published, whether the app is added to the table |
| Docker login failure | Whether `.env` credentials are correct, whether the registry address matches |
| Image pull timeout | First pull of a large image may take 5-10 minutes; check network and registry rate limits |
| Scheduled task not executing | Check `crontab -l` and `/var/log/eval_orchestrator.log` |
| Results not written back | Check whether summary JSON was generated under `logs/sim_eval_container/` |
