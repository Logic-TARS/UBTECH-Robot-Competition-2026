#!/usr/bin/env bash
# 用法:
#   ./enter_sim.sh task4
# 进入 sim 容器后，手动运行 docs/README_split_run.md 里的 sim Python 命令。

set -euo pipefail
source "$(dirname "$0")/env.sh"
prepare_common
print_env_summary

NAME="eval_sim_${TASK}"

docker rm -f "$NAME" >/dev/null 2>&1 || true

echo "[INFO] 进入 sim 容器: ${NAME}"
docker run --rm -it --name "$NAME" \
  --entrypoint /bin/bash \
  --privileged --network host --user root \
  --gpus all \
  --shm-size=8g \
  -v "${HOST_WS}:${CONTAINER_WS}:rw" \
  -w "${CONTAINER_WS}" \
  -v "${ISAAC_CACHE}/cache/kit:/root/.cache/kit:rw" \
  -v "${ISAAC_CACHE}/cache/ov:/root/.cache/ov:rw" \
  -v "${ISAAC_CACHE}/cache/pip:/root/.cache/pip:rw" \
  -v "${ISAAC_CACHE}/cache/glcache:/root/.cache/nvidia/GLCache:rw" \
  -v "${ISAAC_CACHE}/cache/computecache:/root/.cache/nvidia/ComputeCache:rw" \
  -v "${HF_CACHE}:/root/.cache/huggingface:rw" \
  -e NO_AT_BRIDGE=1 \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e XDG_RUNTIME_DIR=/tmp \
  -e "PYTHONPATH=${CONTAINER_WS}" \
  -e DISPLAY="${DISPLAY:-}" \
  -e QT_X11_NO_MITSHM=1 \
  -e TASK="${TASK}" \
  -e PIP_MIRROR="${PIP_MIRROR}" \
  -e PYTHON="${PYTHON}" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  "${SIM_IMAGE}"
