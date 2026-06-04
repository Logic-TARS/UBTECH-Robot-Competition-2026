# run.sh — 交互式容器启动脚本

启动 Isaac Sim 开发容器，自动检测 GPU、X11、docker 权限，挂载缓存目录加速二次启动。

## 用法

```bash
./run.sh                 # 交互式启动（自动检测一切）
./run.sh --headless      # 无头模式（无 GPU 渲染，适合纯推理/远程服务器）
```

`--headless` 后的额外参数透传给 `docker run`，例如覆盖 entrypoint：

```bash
./run.sh --headless -- --entrypoint /bin/bash
```

## 前置条件

- Docker 已安装且当前用户有权限（脚本会自动 `sudo` 提权）
- NVIDIA GPU 驱动 + `nvidia-smi` 可用
- 非 headless 模式需要 X11 显示（`DISPLAY` 已设置且 `xset` 可连接）

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `IMAGE_NAME` | `ghrc_2026:v0` | Docker 镜像名 |
| `CONTAINER_NAME` | `isaac_sim_lerobot_ubt` | 容器名 |
| `HOST_WORKSPACE` | `run.sh` 所在目录 | 宿主机项目根目录 |
| `CONTAINER_WORKSPACE` | `/workspace/GlobalHumanoidRobotChallenge_2026_Baseline` | 容器内工作目录 |
| `SHM_SIZE` | `8g` | 共享内存大小 |
| `HEADLESS` | `0` | 设为 `1` 开启无头模式（等价于 `--headless`） |
| `ISAAC_CACHE_ROOT` | `~/.cache/isaac_sim_container` | Isaac Sim 缓存根目录 |
| `HF_CACHE` | `~/.cache/huggingface` | HuggingFace 模型缓存 |

示例：

```bash
IMAGE_NAME=my_image:v2 HEADLESS=1 ./run.sh
```

## 挂载目录

### 项目代码

宿主机项目根目录以 `rw` 模式挂载到容器内 `CONTAINER_WORKSPACE`，容器内源码修改会直接影响宿主机。

### 持久化缓存（加速二次启动）

| 宿主机 | 容器内 | 用途 |
|--------|--------|------|
| `ISAAC_CACHE_ROOT/cache/kit` | `/root/.cache/kit` | Isaac Kit 缓存 |
| `ISAAC_CACHE_ROOT/cache/ov` | `/root/.cache/ov` | Omniverse 缓存 |
| `ISAAC_CACHE_ROOT/cache/pip` | `/root/.cache/pip` | pip 包缓存 |
| `ISAAC_CACHE_ROOT/cache/glcache` | `/root/.cache/nvidia/GLCache` | OpenGL 着色器缓存 |
| `ISAAC_CACHE_ROOT/cache/computecache` | `/root/.cache/nvidia/ComputeCache` | CUDA 计算缓存 |
| `ISAAC_CACHE_ROOT/data` | `/root/.local/share/ov/data` | Omniverse 数据 |
| `ISAAC_CACHE_ROOT/documents` | `/root/Documents` | Isaac Sim 文档 |
| `HF_CACHE` | `/root/.cache/huggingface` | HuggingFace 模型 |

### 设备

自动检测并挂载：`/dev/bus/usb`、`/dev/input`、`/dev/video*`、`/run/udev`、`/var/run/dbus`。

## 容器内行为

容器启动后自动执行 `pip install -e . --no-deps -q`，使宿主机源码修改立即生效。完成后进入交互式 bash。

## 生命周期

- 每次启动前自动删除同名旧容器
- 容器配置了 `--restart unless-stopped`，宿主机重启后自动恢复
- 脚本退出时自动撤销 `xhost` 授权
