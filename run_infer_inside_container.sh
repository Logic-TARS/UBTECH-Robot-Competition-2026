#!/usr/bin/env bash
# 在 infer 容器内部执行
set -euo pipefail

TASK="${TASK:-${1:-task4}}"
CONTAINER_WS="${CONTAINER_WS:-/workspace/eval}"
PIP_MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PYTHON="${PYTHON:-/isaac-sim/python.sh}"

cd "${CONTAINER_WS}"

${PYTHON} -m pip install -i "${PIP_MIRROR}" -e . --no-deps -q
${PYTHON} -m pip install -i "${PIP_MIRROR}" lz4 msgpack websockets pillow -q

source /isaac-sim/setup_python_env.sh
LD_PRELOAD=/isaac-sim/kit/libcarb.so /isaac-sim/kit/python/bin/python3 \
  -m lerobot.scripts.ghrc_eval_infer \
  --config eval_config/eval_infer.yaml \
  --task "${TASK}"
