# External Algorithm Project Migration Example

This document explains how to place files, declare dependencies, integrate via `PolicyAdapter`, and verify the migration pipeline with a random-action example when migrating a non-LeRobot project into the GHRC evaluation repository.

For the custom policy interface specification, see the [GHRC Custom Policy Integration Guide](custom_policy.md). For the full evaluation workflow, see the [GHRC Evaluation System User Guide](eval_guide.md).

---

## 1. Applicability

This document applies to the following scenarios:

- The policy comes from another complete Python project directory.
- The policy is not a standard LeRobot checkpoint and cannot be directly loaded via `policy_type` + `policy_path`.
- The policy requires custom image preprocessing, state stitching, a planner, an external inference engine, or specialized dependencies.
- The policy can ultimately output the one-dimensional action required by the GHRC evaluation system.

Do not scatter external project code inside `src/lerobot`. It is recommended to keep the project's original directory structure and only add a lightweight GHRC adapter as a boundary layer.

---

## 2. Recommended Directory Structure

Place the external algorithm as an independent project folder at the repository root:

```text
challengeBaseline_newFramework/
├── my_team_policy/
│   ├── my_algorithm/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── planner.py
│   │   └── checkpoint_utils.py
│   ├── checkpoints/
│   │   └── best.pt
│   ├── __init__.py
│   └── ghrc_adapter.py
├── eval_config/
│   └── eval_infer.yaml
└── src/lerobot/
```

Key requirements:

| Item | Requirement |
| --- | --- |
| import | Project directory must be Python-importable |
| Package structure | Each level of Python package directory should contain `__init__.py` |
| adapter | Add `ghrc_adapter.py`, responsible only for GHRC interface conversion |
| Weights | Checkpoint placed in a container-accessible path and specified via `policy_path` or `adapter_config` |
| Dependencies | Additional dependencies written into Dockerfile, requirements, or pyproject config |

If the project directory is placed at the repository root and launched from `/workspace/eval`, the root directory is typically already in `sys.path`. If placed in a deeper path, set `PYTHONPATH` in the Dockerfile or launch script.

---

## 3. In-Repo Random Action Example

The repository provides an external project migration example where the network outputs random actions. This example is only for verifying the directory structure, imports, adapter loading, and action return pipeline — it does not represent task policy capability.

```text
challengeBaseline_newFramework/
├── external_policy_examples/
│   └── random_action_project/
│       ├── external_algo/
│       │   ├── __init__.py
│       │   └── random_network.py
│       ├── __init__.py
│       └── ghrc_adapter.py
└── eval_config/
    └── eval_infer_external_random.yaml
```

Example files:

| File | Description |
| --- | --- |
| `external_policy_examples/random_action_project/external_algo/random_network.py` | Simulates a policy network inside an external project |
| `external_policy_examples/random_action_project/ghrc_adapter.py` | GHRC `PolicyAdapter` wrapper layer |
| `eval_config/eval_infer_external_random.yaml` | infer configuration example |

---

## 4. External Network Example

`random_network.py` simulates a policy object inside an external project:

```python
class RandomActionNetwork:
    def forward(self, observation: dict) -> list[float]:
        return [
            self._rng.uniform(self.action_low, self.action_high)
            for _ in range(self.action_dim)
        ]
```

When migrating a real project, replace this with your own model loading, image preprocessing, state encoding, planner, or inference engine. Code inside the external project does not need to know about the GHRC WebSocket protocol — it only needs to be called by the adapter and return actions.

---

## 5. GHRC Adapter Example

`ghrc_adapter.py` converts GHRC observations into external project inputs and external project outputs into GHRC actions:

```python
from src.lerobot.sim_eval.policy_adapter import PolicyAdapter
from .external_algo.random_network import RandomActionNetwork


class ExternalRandomPolicyAdapter(PolicyAdapter):
    def load(self, model_path, device, config):
        self.network = RandomActionNetwork(
            action_dim=int(config.get("action_dim", 20)),
            action_low=float(config.get("action_low", -1.0)),
            action_high=float(config.get("action_high", 1.0)),
            seed=config.get("seed"),
        )

    def predict(self, observation, context):
        return self.network.forward(observation)

    def reset(self, reset_context=None):
        self.network.reset()

    def close(self):
        self.network = None
```

When migrating a real project, the adapter typically needs to handle:

| Method | Migration Responsibility |
| --- | --- |
| `load()` | Read `policy_path` and `adapter_config`, load checkpoint, initialize model, set device |
| `predict()` | Convert GHRC observation to external algorithm input, invoke model or planner, return one-dimensional action |
| `reset()` | Clear cross-episode state such as RNN hidden state, action chunk, history buffer, planner state |
| `close()` | Release GPU memory, file handles, inference engine resources |

---

## 6. YAML Configuration

Random action example configuration:

```yaml
adapter_class: external_policy_examples.random_action_project.ghrc_adapter:ExternalRandomPolicyAdapter
adapter_config:
  action_dim: 20
  action_low: -1.0
  action_high: 1.0
  seed: 2026
policy_type: null
policy_path: null
```

Real project configuration example:

```yaml
adapter_class: my_team_policy.ghrc_adapter:MyAdapter
adapter_config:
  action_dim: 20
  image_size: [224, 224]
  use_language: true
policy_type: null
policy_path: /workspace/eval/my_team_policy/checkpoints/best.pt
```

Field descriptions:

| Field | Description |
| --- | --- |
| `adapter_class` | `module.path:ClassName`, dynamically imported by `ghrc_eval_infer.py` using `importlib` |
| `adapter_config` | Custom parameter dictionary passed to `adapter.load(model_path, device, config)` |
| `policy_type` | Can be `null` when `adapter_class` is set; will not use the LeRobot default adapter |
| `policy_path` | Real model weight path; can be `null` if no weights are needed |

---

## 7. Running Verification

### 7.1 Verify Python import

Run at the repository root or inside the infer container:

```bash
python -c "from external_policy_examples.random_action_project.ghrc_adapter import ExternalRandomPolicyAdapter; print(ExternalRandomPolicyAdapter)"
```

For real projects, substitute your own adapter:

```bash
python -c "from my_team_policy.ghrc_adapter import MyAdapter; print(MyAdapter)"
```

### 7.2 Start infer

```bash
python -m lerobot.scripts.ghrc_eval_infer \
  --config eval_config/eval_infer_external_random.yaml \
  --task task4
```

### 7.3 Start sim-eval

```bash
python -m lerobot.scripts.ghrc_eval_sim \
  --config eval_config/eval_sim.yaml \
  --task task4
```

A random-action policy will generally not complete the task; failure or timeout is expected. The goal of this example is to verify:

- The external project directory can be imported.
- `adapter_class` can be dynamically loaded by infer.
- The one-dimensional action returned by `predict()` can be decoded by the sim side.
- `reset()` can be invoked at episode end.

---

## 8. Dependency and Image Requirements

If the external project has additional dependencies, they must be delivered together with the contestant image.

| Type | Handling |
| --- | --- |
| Python dependencies | Write into Dockerfile, requirements.txt, or pyproject.toml |
| System libraries | Write into Dockerfile; avoid installing at runtime |
| Large model weights | Place in image, mounted directory, or contest-sanctioned model path |
| Environment variables | Explicitly declare in launch script or container configuration |
| CUDA / TensorRT | Confirm compatibility with the evaluation base image version |

Before formal submission, it is recommended to complete a full `import -> infer startup -> sim connection -> episode reset` pipeline verification in an image consistent with the contest evaluation environment.

---

## 9. Prohibited Modifications

When migrating an external project, the following must not be modified for algorithm adaptation:

| Path or Logic | Reason |
| --- | --- |
| Communication, assertion, and scoring logic in `src/lerobot/sim_eval` | Affects evaluation consistency and fairness |
| Action decoding, episode execution, and result generation in `src/lerobot/scripts/ghrc_eval_sim.py` | Affects simulation behavior and scores |
| Task evaluation thresholds and success conditions | Affects unified contest standards |
