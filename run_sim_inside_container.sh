#!/usr/bin/env bash
# 在 sim 容器内部执行
set -euo pipefail

TASK="${TASK:-${1:-task4}}"
CONTAINER_WS="${CONTAINER_WS:-/workspace/eval}"
PIP_MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PYTHON="${PYTHON:-/isaac-sim/python.sh}"

cd "${CONTAINER_WS}"

${PYTHON} -m pip install -i "${PIP_MIRROR}" -e . --no-deps -q

${PYTHON} -m lerobot.scripts.ghrc_eval_sim \
  --config eval_config/eval_sim.yaml \
  --task "${TASK}"
