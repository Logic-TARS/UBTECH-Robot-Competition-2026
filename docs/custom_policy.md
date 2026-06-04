# 自定义策略接入指南

评估系统支持两种方式接入自定义策略：简单切换 LeRobot 策略类型，或编写完整的自定义适配器。

## 方式一：切换 LeRobot 策略类型（只需改 YAML）

适用于标准 LeRobot 格式的策略（ACT / Pi0 / Diffusion）。修改 `eval_config/eval_infer.yaml`：

```yaml
policy_type: pi0                          # act / pi0 / diffusion
policy_path: /path/to/your/checkpoint     # 默认根据 task 自动推断，也可显式指定
```

底层链路：

```
YAML 配置 → LeRobotPolicyAdapter.load()
  → load_policy(model_path, policy_type, device)
    → PolicyClass.from_pretrained(model_path)
```

完成后直接运行 `./run_eval.sh <task>` 即可。

## 方式二：自定义 PolicyAdapter（不兼容 LeRobot 格式）

适用于自定义框架（RL 模型、ONNX、TensorRT 等）。需要实现自己的适配器类。

### 1. 实现适配器

继承 `PolicyAdapter` 并实现 4 个方法：

```python
# my_pkg/my_adapter.py
from src.lerobot.sim_eval.policy_adapter import PolicyAdapter, InferenceContext, ResetContext
import torch

class MyPolicyAdapter(PolicyAdapter):

    def load(self, model_path: str, device: str, config: dict) -> None:
        """加载模型权重。

        Args:
            model_path: 模型路径（来自 YAML 的 policy_path）
            device: 运行设备（来自 YAML 的 device），如 cuda:0 / cpu
            config: 额外配置字典（来自 YAML 的 adapter_config）
        """
        # 示例：加载自定义模型
        # self.model = MyModel()
        # self.model.load_state_dict(torch.load(model_path))
        # self.model.to(device)
        # self.model.eval()
        ...

    def predict(self, observation: dict, context: InferenceContext) -> torch.Tensor | list[float]:
        """执行推理，返回 action 向量。

        Args:
            observation: LeRobot 标准格式的观测字典，包含:
                - observation.state: 机器人状态张量
                - observation.images.*: 图像张量（如有）
            context: 推理上下文，包含 task / task_text / episode_id / step / timestamp

        Returns:
            torch.Tensor / np.ndarray / list[float]: 动作向量（一维）
        """
        # 示例：执行推理
        # state = observation["observation.state"]
        # action = self.model(state)
        # return action.detach().float().cpu()
        ...

    def reset(self, reset_context: ResetContext | None = None) -> None:
        """Episode 结束时重置策略内部状态。

        用于清理 RNN hidden state、chunk buffer 等跨 step 状态。
        当前默认实现即空操作，若无特殊需求可直接不覆写。

        Args:
            reset_context: 包含 episode_id / status / reason / metrics / step
        """
        ...

    def close(self) -> None:
        """释放适配器持有的资源（GPU 显存、文件句柄等）。"""
        ...
```

### 2. 配置 YAML

在 `eval_config/eval_infer.yaml` 中指定自定义适配器：

```yaml
adapter_class: my_pkg.my_adapter:MyPolicyAdapter   # 必填
adapter_config:                                     # 可选，会传入 load() 的 config
    custom_param: value
# 有 adapter_class 时 policy_type 不再强制要求
# 但仍可通过 policy_path 传入模型路径
```

`adapter_class` 支持两种写法：

| 写法 | 示例 |
|------|------|
| 冒号分隔 | `my_pkg.my_adapter:MyPolicyAdapter` |
| 点号分隔 | `my_pkg.my_adapter.MyPolicyAdapter` |

### 3. 加载链路

```
ghrc_eval_infer.py parse_args()
  → 读取 eval_infer.yaml
    → adapter_class 非空 → load_adapter_class(class_path)
      → importlib.import_module + 校验 PolicyAdapter 子类
    → adapter_class 为空 → create_policy_adapter(adapter_type="lerobot")
      → 走 LeRobotPolicyAdapter

SimInferContainer.initialize()
  → adapter.load(model_path, device, config)
  → 运行期间循环调用 adapter.predict() / adapter.reset()
  → 退出时 adapter.close()
```

## Action 输出格式

`predict()` 返回的 action 支持三种格式（`ghrc_eval_sim.py` 中 `_decode_action` 统一处理）：

| 类型 | 说明 |
|------|------|
| `torch.Tensor` | 一维张量，自动 detach → float → cpu → reshape(-1) |
| `np.ndarray` | 自动转为 torch.Tensor |
| `list[float]` | 自动转为 torch.Tensor |

task4 需要 18 维 action，不足 18 维会自动右侧补零；其他任务按实际维度直接发送给 robot。

## Observation 格式

`predict()` 收到的 `observation` 是 LeRobot 标准字典格式：

```python
{
    "observation.state": torch.Tensor,          # 机器人状态 (N,)
    "observation.images.cam_high": torch.Tensor, # 可选，图像 (C, H, W)
    # ... 其他图像 key
}
```

所有张量已在 `device` 上，无需手动迁移。
