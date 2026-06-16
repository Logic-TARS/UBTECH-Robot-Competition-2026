# GHRC Custom Policy Integration Guide

This document is for contestant teams. It explains how to integrate a custom policy into the GHRC infer container and ensure the policy can be reliably invoked by the `sim-eval` container.

For the upstream evaluation workflow, see the [GHRC Evaluation System User Guide](eval_guide.md). If the policy comes from another complete algorithm project directory, read this document first, then continue with the [External Algorithm Migration Example](external_algorithm_migration.md).

---

## 1. Integration Methods Overview

| Integration Method | Applicable Scenario | What to Modify |
| --- | --- | --- |
| LeRobot default adapter | Checkpoint conforms to LeRobot `from_pretrained` and `select_action` interface | `eval_config/eval_infer.yaml` |
| Custom `PolicyAdapter` | Custom PyTorch, ONNX, TensorRT, RL, planner, or hybrid algorithm | Add an adapter file and configure `adapter_class` |
| External project migration | Original algorithm is a complete project directory; need to preserve its internal structure | Add project directory and `ghrc_adapter.py`; see [External Algorithm Migration Example](external_algorithm_migration.md) |

In the official evaluation, contestants should prioritize integrating policies via YAML and custom adapters. Communication protocols, assertions, and scoring logic must not be modified.

---

## 2. LeRobot Default Adapter

The LeRobot default adapter is suitable for standard LeRobot action policies. Configuration example:

```yaml
adapter_type: lerobot
adapter_class: null
policy_type: act
policy_path: null
require_task_policy_paths: true
task_policy_paths:
  task1: ../challenge2026_baseline/task1/act/pretrained_model
  task2: ../challenge2026_baseline/task2/act/pretrained_model
  task3: ../challenge2026_baseline/task3/act/pretrained_model
  task4: ../challenge2026_baseline/task4/act/pretrained_model
```

Loading chain:

```text
eval_infer.yaml
  -> LeRobotPolicyAdapter.load()
    -> load_policy(model_path, policy_type, device)
      -> get_policy_class(policy_type)
      -> PolicyClass.from_pretrained(model_path)
      -> policy.select_action(observation_batch)
```

### 2.1 Support Boundaries

The current default adapter is not equivalent to "supports all LeRobot policies." Direct usability requires all of the following conditions to be met:

| Condition | Requirement |
| --- | --- |
| policy class | `policy_type` must be discoverable by LeRobot `get_policy_class()` |
| checkpoint | Checkpoint must be loadable by `config_class.from_pretrained()` and `policy_cls.from_pretrained()` |
| inference interface | Policy must implement a working `select_action(batch)` |
| observation | Current evaluation observation keys must match the checkpoint's input features |
| action | Output must be a one-dimensional action vector decodable by the sim side |

Consider using a custom `PolicyAdapter` in the following cases:

- Policy requires additional tokenizer, processor, language tokens, or custom image preprocessing.
- Policy is not an action policy, e.g., a reward classifier.
- Policy's input keys, state concatenation method, camera naming, or action post-processing differs from the current evaluation environment.
- Migrating a non-LeRobot framework, external project, or self-developed inference engine.

---

## 3. Custom PolicyAdapter

A custom adapter is the recommended general-purpose extension method for GHRC. It converts the observation passed by the evaluation system into the contestant model's input and returns a one-dimensional action.

### 3.1 Interface Definition

```python
from src.lerobot.sim_eval.policy_adapter import PolicyAdapter, InferenceContext, ResetContext


class MyPolicyAdapter(PolicyAdapter):
    def load(self, model_path: str, device: str, config: dict) -> None:
        """Load model, weights, inference engine, and custom configuration."""
        ...

    def predict(self, observation: dict, context: InferenceContext):
        """Return a one-dimensional action based on the current observation."""
        ...

    def reset(self, reset_context: ResetContext | None = None) -> None:
        """Clear cross-step state after episode ends."""
        ...

    def close(self) -> None:
        """Release model, GPU memory, file handles, or inference engine resources."""
        ...
```

Interface responsibilities:

| Method | Invocation Timing | Required Behavior |
| --- | --- | --- |
| `load()` | Called once at infer service startup | Load model and enter inference mode; raise a clear exception on failure |
| `predict()` | Called at every simulation step | Return one-dimensional `torch.Tensor`, `np.ndarray`, or `list[float]` |
| `reset()` | Called at episode end or task reset | Clear RNN hidden state, action chunk, history buffer, planner state, etc. |
| `close()` | Called at infer service exit | Release GPU memory, file handles, TensorRT engine, etc. |

### 3.2 YAML Configuration

```yaml
adapter_type: lerobot
adapter_class: my_team_policy.ghrc_adapter:MyPolicyAdapter
adapter_config:
  action_dim: 20
  image_size: [224, 224]
  normalize_state: true

policy_type: null
policy_path: /workspace/eval/my_team_policy/checkpoints/best.pt
```

Field descriptions:

| Field | Required | Description |
| --- | --- | --- |
| `adapter_class` | Yes | Custom adapter class path in `module.path:ClassName` or `module.path.ClassName` format |
| `adapter_config` | No | Custom dictionary passed as-is to `load(model_path, device, config)` |
| `policy_path` | Depends on policy | Weight path or model directory; custom adapters may use as needed |
| `policy_type` | No | Not used when `adapter_class` is set; can be `null` |

The class pointed to by `adapter_class` must inherit from `PolicyAdapter`, or infer startup will fail.

---

## 4. Minimal Example: Zero-Action Policy

The repository provides a runnable minimal custom policy for verifying configuration, imports, WebSocket communication, and the action decoding pipeline:

| Item | Path |
| --- | --- |
| Example code | `src/lerobot/sim_eval/zero_action_policy.py` |
| Example config | `eval_config/eval_infer_zero_action.yaml` |

Core logic:

```python
class ZeroActionPolicyAdapter(PolicyAdapter):
    def load(self, model_path: str, device: str, config: dict) -> None:
        self.action_dim = int(config.get("action_dim", 20))
        self._action = [0.0] * self.action_dim

    def predict(self, observation: dict, context: InferenceContext) -> list[float]:
        return list(self._action)
```

YAML:

```yaml
adapter_class: src.lerobot.sim_eval.zero_action_policy:ZeroActionPolicyAdapter
adapter_config:
  action_dim: 20
policy_type: null
policy_path: null
```

Run inside the infer container:

```bash
python -m lerobot.scripts.ghrc_eval_infer \
  --config eval_config/eval_infer_zero_action.yaml \
  --task task4
```

A zero-action policy will generally not complete the task. This example is only for verifying that the evaluation pipeline can start, connect, and return actions correctly.

---

## 5. External Project Migration Entry Point

If the contestant is migrating another project directory — for example, a self-developed RL project, vision-language model project, or existing robot algorithm repository — it is recommended to keep the original project directory structure and only add a `ghrc_adapter.py` as the GHRC interface layer.

For a complete random-action external project example, see the [External Algorithm Migration Example](external_algorithm_migration.md):

| Item | Path |
| --- | --- |
| Example project | `external_policy_examples/random_action_project/` |
| Example config | `eval_config/eval_infer_external_random.yaml` |
| Migration guide | [external_algorithm_migration.md](external_algorithm_migration.md) |

This example retains the external project's own `external_algo/random_network.py` and outputs random actions via `ghrc_adapter.py`, used to verify:

- The external project directory can be Python-imported.
- `adapter_class` can correctly load the custom adapter.
- The one-dimensional action returned by `predict()` can be decoded by the sim side.
- `reset()` can be normally invoked at episode end.

---

## 6. Observation Format

The `observation` received by `predict()` is a dictionary with common keys as follows:

```python
{
    "observation.state": torch.Tensor,
    "observation.images.cam_high": torch.Tensor,
    "observation.images.cam_left_wrist": torch.Tensor,
    "observation.images.cam_right_wrist": torch.Tensor,
}
```

Notes:

- `state` is typically a one-dimensional robot state vector.
- Images are typically `torch.Tensor`; actual image keys depend on the task configuration and simulation output.
- Custom adapters should handle cropping, normalization, resizing, camera renaming, and batch dimension handling within `predict()`.
- Do not assume all policies have the same input format, especially models requiring language tokens or multi-modal processors.

---

## 7. Action Output Format

The `predict()` return value supports the following types:

| Type | Handling |
| --- | --- |
| `torch.Tensor` | Auto `detach()`, convert to float, move to CPU, and flatten to 1D |
| `np.ndarray` | Auto-converted to `torch.Tensor` |
| `list[float]` | Auto-converted to `torch.Tensor` |

Action requirements:

- Must be a one-dimensional action vector.
- Recommended to output 20-dimensional actions.
- For `task4`, outputting 18 dimensions is acceptable — the sim side will pad with 2 gripper control values on the right.
- Do not return batch dimensions, dictionaries, nested lists, or values outside the policy's expected range.

---

## 8. Prohibited Modifications

In official evaluation submissions, the following must not be modified for policy adaptation:

| Path | Reason |
| --- | --- |
| Communication protocol, assertions, and scoring logic in `src/lerobot/sim_eval` | Affects evaluation consistency |
| Action decoding and result generation logic in `src/lerobot/scripts/ghrc_eval_sim.py` | Affects simulation execution and scoring results |
| Task evaluation thresholds and success conditions | Affects competition fairness |

If policy loading needs to be extended, prioritize adding a custom `PolicyAdapter` or adding a wrapper layer in the contestant's project directory.

---

## 9. Integration Checklist

| Check Item | Requirement |
| --- | --- |
| Config file | `eval_config/eval_infer.yaml` or standalone example YAML is readable by infer |
| import | `python -c "from my_team_policy.ghrc_adapter import MyAdapter"` succeeds |
| Inheritance | Custom class inherits from `PolicyAdapter` |
| `load()` | Can load model, weights, and config, and enter eval inference mode |
| `predict()` | Returns a one-dimensional action for any valid observation input |
| Action dimension | Recommended 20 dims; 18 dims acceptable for task4 |
| `reset()` | Clears cross-step state after episode ends |
| Dependencies | Additional Python packages, system libraries, and weight files written into image or mounted path |
| Logging | Clear errors on load failure, missing weights, or dimension mismatches |

After completing the above checks, use the local evaluation workflow in the [GHRC Evaluation System User Guide](eval_guide.md) to start the full evaluation.
