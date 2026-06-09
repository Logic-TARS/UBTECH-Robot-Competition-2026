#!/usr/bin/env bash
# 公共环境变量。两个入口脚本都会 source 这个文件。
set -euo pipefail

TASK="${TASK:-${1:-task4}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INFER_IMAGE="${INFER_IMAGE:-ghrc-eval-infer:latest}"
SIM_IMAGE="${SIM_IMAGE:-ghrc-eval-sim:latest}"

HOST_WS="${HOST_WS:-${SCRIPT_DIR}}"
CONTAINER_WS="${CONTAINER_WS:-/workspace/eval}"

ISAAC_CACHE="${ISAAC_CACHE_ROOT:-${HOME}/.cache/isaac_sim_container}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"

HEADLESS="${HEADLESS:-0}"
PIP_MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"

PYTHON="/isaac-sim/python.sh"
INFER_PY="source /isaac-sim/setup_python_env.sh && LD_PRELOAD=/isaac-sim/kit/libcarb.so /isaac-sim/kit/python/bin/python3"

infer_args() {
  echo "--config eval_config/eval_infer.yaml --task ${TASK}"
}

sim_args() {
  echo "--config eval_config/eval_sim.yaml --task ${TASK}"
}

prepare_common() {
  command -v docker >/dev/null 2>&1 || { echo "[ERROR] 未找到 docker" >&2; exit 1; }
  docker image inspect "$INFER_IMAGE" >/dev/null 2>&1 || { echo "[ERROR] 镜像 ${INFER_IMAGE} 不存在" >&2; exit 1; }
  docker image inspect "$SIM_IMAGE" >/dev/null 2>&1 || { echo "[ERROR] 镜像 ${SIM_IMAGE} 不存在" >&2; exit 1; }

  for d in cache/kit cache/ov cache/pip cache/glcache cache/computecache data documents; do
    mkdir -p "${ISAAC_CACHE}/${d}"
  done
  mkdir -p "${HF_CACHE}"

  if [[ "$HEADLESS" == "0" ]]; then
    xhost +local:docker >/dev/null 2>&1 || true
  fi
}

print_env_summary() {
  echo "[INFO] TASK=${TASK}"
  echo "[INFO] INFER_IMAGE=${INFER_IMAGE}"
  echo "[INFO] SIM_IMAGE=${SIM_IMAGE}"
  echo "[INFO] HOST_WS=${HOST_WS}"
  echo "[INFO] CONTAINER_WS=${CONTAINER_WS}"
}
