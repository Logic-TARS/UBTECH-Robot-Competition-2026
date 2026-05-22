#!/usr/bin/env bash
# 用法: ./run_eval.sh task4 | all
# 环境变量: IMAGE_NAME (默认 ghrc-eval-sim:latest), HEADLESS
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

TASK="${1:-all}"
ALL_TASKS=("task4" "task1" "task2" "task3")
[[ "$TASK" == "all" ]] && TASKS=("${ALL_TASKS[@]}") || TASKS=("$TASK")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-ghrc-eval-sim:latest}"
HOST_WS="${SCRIPT_DIR}"
CONTAINER_WS="/workspace/eval"
ISAAC_CACHE="${ISAAC_CACHE_ROOT:-${HOME}/.cache/isaac_sim_container}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
HEADLESS="${HEADLESS:-0}"
PIP_MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"

info "镜像: ${IMAGE_NAME}  |  任务: ${TASKS[*]}"

command -v docker &>/dev/null || { error "未找到 docker"; exit 1; }
docker image inspect "$IMAGE_NAME" &>/dev/null 2>&1 || { error "镜像 ${IMAGE_NAME} 不存在"; exit 1; }

for d in cache/kit cache/ov cache/pip cache/glcache cache/computecache data documents; do
    mkdir -p "${ISAAC_CACHE}/${d}"
done
mkdir -p "${HF_CACHE}"
[[ "$HEADLESS" == "0" ]] && xhost +local:docker &>/dev/null 2>&1 || true

PYTHON="/isaac-sim/python.sh"
INFER_PY="source /isaac-sim/setup_python_env.sh && LD_PRELOAD=/isaac-sim/kit/libcarb.so /isaac-sim/kit/python/bin/python3"

run_eval() {
    local task="$1"
    local infer_args="--config eval_config/eval_infer.yaml --task ${task}"
    local sim_args="--config eval_config/eval_sim.yaml --task ${task}"

    info "==========================================="
    info "评估: ${task}"
    info "==========================================="

    docker run --rm --name "eval_${task}" \
        --entrypoint /bin/bash \
        --privileged --network host --user root --gpus all --shm-size=8g \
        -v "${HOST_WS}:${CONTAINER_WS}:rw" -w "${CONTAINER_WS}" \
        -v "${ISAAC_CACHE}/cache/kit:/root/.cache/kit:rw" \
        -v "${ISAAC_CACHE}/cache/ov:/root/.cache/ov:rw" \
        -v "${ISAAC_CACHE}/cache/pip:/root/.cache/pip:rw" \
        -v "${ISAAC_CACHE}/cache/glcache:/root/.cache/nvidia/GLCache:rw" \
        -v "${ISAAC_CACHE}/cache/computecache:/root/.cache/nvidia/ComputeCache:rw" \
        -v "${HF_CACHE}:/root/.cache/huggingface:rw" \
        -e NO_AT_BRIDGE=1 -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e XDG_RUNTIME_DIR=/tmp \
        -e "PYTHONPATH=${CONTAINER_WS}" \
        -e DISPLAY="${DISPLAY:-}" -e QT_X11_NO_MITSHM=1 \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        "${IMAGE_NAME}" \
        -c "cd ${CONTAINER_WS}; ${PYTHON} -m pip install -i ${PIP_MIRROR} -e . --no-deps -q; ${PYTHON} -m pip install -i ${PIP_MIRROR} lz4 msgpack websockets pillow -q; echo '[INIT] infer 启动...'; ${INFER_PY} -m lerobot.scripts.ghrc_eval_infer ${infer_args} & sleep 20; echo '[INIT] sim-eval 启动...'; ${PYTHON} -m lerobot.scripts.ghrc_eval_sim ${sim_args}" 2>&1 \
        | grep -E "Episode.*step=|SUCCESS|FAILED|成功|失败|异常|汇总|Score|Connected|INIT" \
        | grep -vE "Warning.*usd|omni\.physicsschema" || true

    info "${task} 完成"
}

for t in "${TASKS[@]}"; do
    run_eval "$t"
done

[[ "$HEADLESS" == "0" ]] && xhost -local:docker &>/dev/null 2>&1 || true
info "全部完成！结果: logs/sim_eval_container/"
