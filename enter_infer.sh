#!/usr/bin/env bash
# 用法:
#   ./enter_infer.sh task4
# 进入 infer 容器后，手动运行 docs/README_split_run.md 里的 infer Python 命令。

set -euo pipefail
source "$(dirname "$0")/env.sh"
prepare_common
print_env_summary

NAME="eval_infer_${TASK}"

docker rm -f "$NAME" >/dev/null 2>&1 || true

echo "[INFO] 进入 infer 容器: ${NAME}"
docker run --rm -it --name "$NAME" \
  --entrypoint /bin/bash \
  --privileged --network host --user root \
  --gpus all \
  --shm-size=8g \
  -v "${HOST_WS}:${CONTAINER_WS}:rw" \
  -w "${CONTAINER_WS}" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e TASK="${TASK}" \
  -e PIP_MIRROR="${PIP_MIRROR}" \
  -e PYTHON="${PYTHON}" \
  -e INFER_PY="${INFER_PY}" \
  "${INFER_IMAGE}"
