# Global Humanoid Robot Challenge 2026 — Contestant Submission Specification

## 1. Overview

The evaluation system uses a **dual-container isolation architecture**. Contestants submit an **infer container** implementing policy inference.

| Container | Responsibility                                                                         | Provided By          |
| --------- | -------------------------------------------------------------------------------------- | -------------------- |
| sim-eval  | Isaac Sim simulation, observation collection, assertion evaluation, result persistence | Organizer            |
| sim-infer | Policy model loading, inference, policy reset                                          | **Contestant** |

The two containers communicate via **WebSocket** (msgpack serialization + lz4 compression):

- Control channel (8765): start / episode_end / stop
- Stream channel (8766): observation / action

---

## 2. System Architecture

```
┌─── sim-eval container (Organizer) ────────┐  WebSocket   ┌─── sim-infer container (Contestant) ──┐
│                                           │              │                                       │
│  Isaac Sim simulation                     │  ── obs ──→  │  PolicyAdapter                        │
│  robot.get_observation()                  │              │   ├─ load()      Load model           │
│  robot.send_action()                      │  ←─ action ── │   ├─ predict()   Infer action         │
│                                           │              │   ├─ reset()     Reset state          │
│  Task1-4 Assertion evaluation             │  ── ctrl ──→  │   └─ close()     Release resources   │
│  Scoring + result persistence             │              │                                       │
└───────────────────────────────────────────┘              └───────────────────────────────────────┘
```

The contestant infer container receives **LeRobot-format observations** (`observation.state` + `observation.images.*`) and does not need to be aware of Isaac Sim internals.

---

## 3. Required Contestant Interfaces

### 3.1 PolicyAdapter (Required)

Contestants must implement the `PolicyAdapter` abstract class with 4 methods:

```python
class PolicyAdapter(abc.ABC):

    @abc.abstractmethod
    def load(self, model_path: str, device: str, config: dict) -> None:
        """Load model weights. config contains runtime info such as task, task_text, etc."""

    @abc.abstractmethod
    def predict(self, observation: dict, context: InferenceContext) -> torch.Tensor:
        """Infer one step of action based on observation, returning a 1D tensor."""

    @abc.abstractmethod
    def reset(self, reset_context: ResetContext | None = None) -> None:
        """Reset internal policy state at episode end (action queue, history, etc.)."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release resources held by the adapter."""
```

### 3.2 Context Data Types

```python
@dataclass
class InferenceContext:
    task: str          # Task name, e.g., "task4"
    task_text: str     # Task description, e.g., "packing box"
    episode_id: int    # Current episode number
    step: int          # Current step number
    timestamp: float   # Observation sampling timestamp

@dataclass
class ResetContext:
    episode_id: int
    status: str = "reset"   # success / failed / timeout / error
    reason: str = ""        # Reason for episode end
    metrics: dict = {}      # Metrics for this episode
    step: int = 0
```

### 3.3 Default Implementation Reference

The competition provides `LeRobotPolicyAdapter`, which automatically handles observation tensor conversion, image normalization, dimension cropping, etc. Contestants may use it directly or inherit from it.

```python
from src.lerobot.sim_eval.policy_adapter import PolicyAdapter, LeRobotPolicyAdapter

class MyAdapter(LeRobotPolicyAdapter):
    # Inherit default implementation, override predict() only
    def predict(self, observation, context):
        # ... custom preprocessing + inference
        return action
```

### 3.4 Zero-Action Reference Example

The repository includes a minimal custom policy example so contestants can verify the interface and configuration:

- Code: `src/lerobot/sim_eval/zero_action_policy.py`
- Config: `eval_config/eval_infer_zero_action.yaml`

This example's `predict()` ignores the observation and always returns a 20-dimensional zero action:

```python
def predict(self, observation, context) -> list[float]:
    return [0.0] * 20
```

Run the example:

```bash
python -m lerobot.scripts.ghrc_eval_infer --config eval_config/eval_infer_zero_action.yaml --task task4
```

### 3.5 External Project Migration Reference

If migrating an existing algorithm project directory, it is recommended to keep the original project directory and add a `ghrc_adapter.py`:

```text
my_team_policy/
├── my_algorithm/        # Original project code
└── ghrc_adapter.py      # New: extends PolicyAdapter
```

The repository provides a random-action example:

- Code: `external_policy_examples/random_action_project/`
- Config: `eval_config/eval_infer_external_random.yaml`
- Documentation: `docs/external_algorithm_migration.md`

### 3.6 Configuration Method

The infer container selects adapters via YAML configuration:

```yaml
# eval_config/task4_infer.yaml
adapter_type: lerobot                    # Built-in LeRobot adapter
# Or specify a custom adapter
adapter_type: custom
adapter_class: my_package.MyModule:MyAdapter
policy_type: act
policy_path: /workspace/.../pretrained_model
```

---

## 4. Data Dimension Specification

### 4.1 Observation

Observations are transmitted via WebSocket using **LeRobot standard keys**:

```python
observation = {
    "observation.state": torch.Tensor,   # shape (20,) — joint positions + grippers
    "observation.images.head_left":  torch.Tensor,  # (3, 480, 640) float32, RGB
    "observation.images.head_right": torch.Tensor,
    "observation.images.wrist_left":  torch.Tensor,
    "observation.images.wrist_right": torch.Tensor,
    # task4 models may receive 18-dim state (no gripper control); adapter auto-crops
}
```

**observation.state structure (20 dimensions):**

| Index | Meaning               | Joint Names                                                     |
| ----- | --------------------- | --------------------------------------------------------------- |
| 0-6   | Left arm 7 joints     | L_shoulder_pitch/roll/yaw, L_elbow_roll/yaw, L_wrist_pitch/roll |
| 7-13  | Right arm 7 joints    | R_shoulder_pitch/roll/yaw, R_elbow_roll/yaw, R_wrist_pitch/roll |
| 14-17 | Gripper fingers       | L_finger1/2, R_finger1/2                                        |
| 18    | Left gripper control  | left_gripper                                                    |
| 19    | Right gripper control | right_gripper                                                   |

### 4.2 Action

The action returned by `predict()` is a **one-dimensional action vector**, with dimensions matching `output_features.action.shape` in the model's `config.json` (typically 18 or 20 dimensions).

**Requirements:**

- Supports `torch.Tensor`, `np.ndarray`, or `list[float]`
- Shape is `(20,)`; task4 may also return `(18,)` — the system will auto-pad with 2 gripper control values at the end
- No batch dimension
- The adapter already auto-squeezes; no manual handling needed

---

## 5. Submission Project Structure

```
<team_name>-sim-infer/
├── lerobot/             # LeRobot algorithm module
├── eval_config/         # infer container config (task{1-4}_infer.yaml)
├── challenge2026_baseline/  # Pretrained model weights
│   ├── Part_Sorting/act/pretrained_model
│   ├── Conveyor_Sorting/act/pretrained_model
│   ├── Foam_Inlaying/act/pretrained_model
│   └── Packing_Box/act/pretrained_model
├── Dockerfile
├── run.sh
└── pyproject.toml
```

### Packaging for Submission

```bash
cp -r . <team_name>-sim-infer
cd <team_name>-sim-infer
docker build -t <team_name>/sim-infer:v1 .
docker save <team_name>/sim-infer:v1 | gzip > <team_name>-sim-infer-v1.tar.gz
```

---

## 6. Interface Self-Check Checklist

| Interface                              | Required | Criteria                                                  |
| -------------------------------------- | -------- | --------------------------------------------------------- |
| `adapter.load(path, device, config)` | Yes      | Successfully loads model weights; policy enters eval mode |
| `adapter.predict(obs, ctx)`          | Yes      | Accepts valid obs, returns correct-dimension action       |
| `adapter.reset(ctx)`                 | Yes      | No exception; does not affect subsequent inference        |
| `adapter.close()`                    | Yes      | Releases GPU memory                                       |

---

## 7. FAQ

**Q: Can I use a non-LeRobot model?**
Yes. Inherit from `PolicyAdapter`, implement `load / predict / reset / close`. Use `adapter_class` in `infer.yaml` to specify your custom class.

**Q: Is task4's observation.state 18-dimensional?**
Task4 models were trained without gripper control, so the state is 18-dimensional. `LeRobotPolicyAdapter` auto-reads the expected dimension from `config.json` and crops accordingly; contestants do not need to handle this.

**Q: What image format is used?**
4 RGB cameras, shape `(3, 480, 640)`, float32, value range [0, 1].

**Q: Can I introduce additional dependencies?**
Yes. Declare them in the Dockerfile or pyproject.toml.
