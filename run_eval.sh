#!/usr/bin/env bash
# =============================================================================
#  评估脚本 — 支持两种模式
#
#  用法:
#    ./run_eval.sh task4                    # 默认单容器模式，评估 task4
#    ./run_eval.sh all                      # 依次评估全部 4 个任务
#    ./run_eval.sh task4 --mode dual        # 双容器模式（需多 GPU）
#
#  环境变量:
#    IMAGE_NAME       Docker 镜像名 (默认: ghrc_2026:v0)
#    HEADLESS         是否无头模式 (默认: 0)
# =============================================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

TASK="${1:-all}"
MODE="single"

shift || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        *) shift ;;
    esac
done

ALL_TASKS=("task4" "task1" "task2" "task3")
[[ "$TASK" == "all" ]] && TASKS=("${ALL_TASKS[@]}") || TASKS=("$TASK")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-ghrc_2026:v0}"
HOST_WS="${SCRIPT_DIR}"
CONTAINER_WS="/workspace/GlobalHumanoidRobotChallenge_2026_Baseline"
ISAAC_CACHE="${ISAAC_CACHE_ROOT:-${HOME}/.cache/isaac_sim_container}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
NETWORK="Ubtech_sim"
HEADLESS="${HEADLESS:-0}"

info "模式: ${MODE}  |  任务: ${TASKS[*]}  |  镜像: ${IMAGE_NAME}"

command -v docker &>/dev/null || { error "未找到 docker"; exit 1; }
docker image inspect "$IMAGE_NAME" &>/dev/null 2>&1 || { error "镜像 ${IMAGE_NAME} 不存在"; exit 1; }

for d in cache/kit cache/ov cache/pip cache/glcache cache/computecache data documents; do
    mkdir -p "${ISAAC_CACHE}/${d}"
done
mkdir -p "${HF_CACHE}"
[[ "$HEADLESS" == "0" ]] && xhost +local:docker &>/dev/null 2>&1 || true

PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
PIP_INIT="/isaac-sim/python.sh -m pip install -i ${PIP_MIRROR} -e . --no-deps -q && /isaac-sim/python.sh -m pip install -i ${PIP_MIRROR} lz4 msgpack -q"

# ========================== 共享挂载参数 ==========================
COMMON_MOUNTS=(
    -v "${HOST_WS}:${CONTAINER_WS}:rw"
    -w "${CONTAINER_WS}"
    -v "${ISAAC_CACHE}/cache/kit:/root/.cache/kit:rw"
    -v "${ISAAC_CACHE}/cache/ov:/root/.cache/ov:rw"
    -v "${ISAAC_CACHE}/cache/pip:/root/.cache/pip:rw"
    -v "${ISAAC_CACHE}/cache/glcache:/root/.cache/nvidia/GLCache:rw"
    -v "${ISAAC_CACHE}/cache/computecache:/root/.cache/nvidia/ComputeCache:rw"
    -v "${HF_CACHE}:/root/.cache/huggingface:rw"
    -e NO_AT_BRIDGE=1 -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e XDG_RUNTIME_DIR=/tmp
    -e "PYTHONPATH=${CONTAINER_WS}"
)

# ========================== 单容器双进程模式 ==========================
run_single() {
    local task="$1"
    local infer_cfg="eval_config/infer.yaml --task ${task}"
    local sim_cfg="eval_config/sim_eval.yaml --task ${task}"

    info "==========================================="
    info "评估: ${task}"
    info "==========================================="

    docker run --rm --name "eval_${task}" \
        --entrypoint /bin/bash \
        --privileged --network host --user root --gpus all --shm-size=8g \
        "${COMMON_MOUNTS[@]}" \
        -e DISPLAY="${DISPLAY:-}" -e QT_X11_NO_MITSHM=1 \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        "${IMAGE_NAME}" \
        -c "
            ${PIP_INIT}
            echo '[INIT] infer 启动...'
            /isaac-sim/python.sh -m lerobot.scripts.sim_infer_container --config ${infer_cfg} &
            sleep 40
            echo '[INIT] sim-eval 启动...'
            /isaac-sim/python.sh -m lerobot.scripts.sim_eval_container --config ${sim_cfg}
        " 2>&1 \
        | grep -E "Episode.*step=|SUCCESS|FAILED|成功|失败|异常|汇总|Score|Connected|segfault|Traceback|ModuleNotFound|INIT" \
        | grep -vE "Warning.*usd|omni\.physicsschema" || true

    info "${task} 完成"
}

# ========================== 双容器模式（需多 GPU） ==========================
run_dual() {
    local task="$1"
    local infer_cfg="eval_config/infer.yaml --task ${task}"
    local sim_cfg="eval_config/sim_eval.yaml --task ${task}"

    info "==========================================="
    info "评估: ${task} (双容器)"
    info "==========================================="

    docker network create "$NETWORK" 2>/dev/null || true
    docker rm -f sim-infer sim-eval &>/dev/null 2>&1 || true

    # 更新 sim-eval 配置指向容器名
    sed -i 's/sim_infer_host: .*/sim_infer_host: sim-infer/' "${sim_cfg}"

    # --- infer 容器 ---
    info "启动 sim-infer 容器..."
    docker run -d --rm --name sim-infer \
        --network "$NETWORK" \
        --entrypoint /bin/bash \
        --privileged --user root --gpus all --shm-size=8g \
        "${COMMON_MOUNTS[@]}" \
        "${IMAGE_NAME}" \
        -c "${PIP_INIT} && exec /isaac-sim/python.sh -m lerobot.scripts.sim_infer_container --config ${infer_cfg}"

    sleep 30
    if ! docker ps --format '{{.Names}}' | grep -q sim-infer; then
        error "sim-infer 启动失败"; docker logs sim-infer 2>&1 | grep -iE "error|traceback" | tail -5
        return 1
    fi
    info "sim-infer 就绪"

    # --- sim-eval 容器 ---
    info "启动 sim-eval 容器..."
    docker run --rm --name sim-eval \
        --network "$NETWORK" \
        --entrypoint /bin/bash \
        --privileged --user root --gpus all --shm-size=8g \
        "${COMMON_MOUNTS[@]}" \
        -e DISPLAY="${DISPLAY:-}" -e QT_X11_NO_MITSHM=1 \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        "${IMAGE_NAME}" \
        -c "${PIP_INIT} && exec /isaac-sim/python.sh -m lerobot.scripts.sim_eval_container --config ${sim_cfg}" 2>&1 \
        | grep -E "Episode.*step=|SUCCESS|FAILED|成功|失败|异常|汇总|Score|Connected|segfault|Traceback|ModuleNotFound|INIT" \
        | grep -vE "Warning.*usd|omni\.physicsschema" || true

    # 恢复配置
    sed -i 's/sim_infer_host: sim-infer/sim_infer_host: localhost/' "${sim_cfg}"
    docker rm -f sim-infer &>/dev/null 2>&1 || true
    info "${task} 完成"
}

# ========================== 主流程 ==========================
for t in "${TASKS[@]}"; do
    case "$MODE" in
        single) run_single "$t" ;;
        dual)   run_dual "$t" ;;
        *)      error "未知模式: ${MODE} (可选: single, dual)"; exit 1 ;;
    esac
done

[[ "$HEADLESS" == "0" ]] && xhost -local:docker &>/dev/null 2>&1 || true
info "全部完成！结果: logs/sim_eval_container/"
