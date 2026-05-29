#!/usr/bin/env bash
# 用法: ./run_eval.sh task4 | all
# 双容器模式: infer (CPU) 与 sim-eval (GPU) 各一个容器，WebSocket 通信
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

read_yaml_scalar() {
    local file="$1"
    local key="$2"
    local default_value="$3"
    local value
    value="$(
        awk -F':' -v key="$key" '
            $1 ~ "^[[:space:]]*" key "[[:space:]]*$" {
                sub(/^[[:space:]]+/, "", $2)
                sub(/[[:space:]]+$/, "", $2)
                print $2
                exit
            }
        ' "$file" 2>/dev/null || true
    )"
    [[ -n "${value}" ]] && echo "${value}" || echo "${default_value}"
}

is_port_listening() {
    local port="$1"
    if command -v ss &>/dev/null; then
        ss -lnt "( sport = :${port} )" 2>/dev/null | tail -n +2 | grep -q ":${port}[[:space:]]"
    else
        netstat -lnt 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]${port}$"
    fi
}

print_infer_log_tail() {
    local task="$1"
    local log_file="/tmp/eval_infer_${task}.log"
    if [[ -f "${log_file}" ]]; then
        error "infer 日志尾部 (${log_file}):"
        tail -n 80 "${log_file}" >&2 || true
    else
        error "未找到 infer 日志: ${log_file}"
    fi
}

wait_for_infer_ready() {
    local task="$1"
    local infer_name="$2"
    local control_port="$3"
    local stream_port="$4"
    local timeout_seconds="$5"
    local start_ts
    start_ts="$(date +%s)"

    while true; do
        if ! docker inspect -f '{{.State.Running}}' "${infer_name}" >/dev/null 2>&1; then
            error "infer 容器 ${infer_name} 已退出"
            print_infer_log_tail "${task}"
            return 1
        fi

        if is_port_listening "${control_port}" && is_port_listening "${stream_port}"; then
            info "infer 已就绪: control=${control_port}, stream=${stream_port}"
            return 0
        fi

        if (( $(date +%s) - start_ts >= timeout_seconds )); then
            error "等待 infer 就绪超时 (${timeout_seconds}s)，端口 ${control_port}/${stream_port} 未监听"
            print_infer_log_tail "${task}"
            return 1
        fi

        sleep 2
    done
}

TASK="${1:-all}"
ALL_TASKS=("task4" "task1" "task2" "task3")
[[ "$TASK" == "all" ]] && TASKS=("${ALL_TASKS[@]}") || TASKS=("$TASK")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFER_IMAGE="${INFER_IMAGE:-ghrc-eval-infer:latest}"
SIM_IMAGE="${SIM_IMAGE:-ghrc-eval-sim:latest}"
HOST_WS="${SCRIPT_DIR}"
CONTAINER_WS="/workspace/eval"
ISAAC_CACHE="${ISAAC_CACHE_ROOT:-${HOME}/.cache/isaac_sim_container}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
HEADLESS="${HEADLESS:-0}"
PIP_MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
INFER_CONFIG="${INFER_CONFIG:-eval_config/eval_infer.yaml}"
SIM_CONFIG="${SIM_CONFIG:-eval_config/eval_sim.yaml}"
INFER_CONTROL_PORT="$(read_yaml_scalar "${INFER_CONFIG}" "websocket_control_port" "8765")"
INFER_STREAM_PORT="$(read_yaml_scalar "${INFER_CONFIG}" "websocket_stream_port" "8766")"
INFER_READY_TIMEOUT="${INFER_READY_TIMEOUT:-300}"

info "infer: ${INFER_IMAGE}  |  sim: ${SIM_IMAGE}  |  任务: ${TASKS[*]}"

command -v docker &>/dev/null || { error "未找到 docker"; exit 1; }
docker image inspect "$INFER_IMAGE" &>/dev/null 2>&1 || { error "镜像 ${INFER_IMAGE} 不存在"; exit 1; }
docker image inspect "$SIM_IMAGE"   &>/dev/null 2>&1 || { error "镜像 ${SIM_IMAGE} 不存在"; exit 1; }

for d in cache/kit cache/ov cache/pip cache/glcache cache/computecache data documents; do
    mkdir -p "${ISAAC_CACHE}/${d}"
done
mkdir -p "${HF_CACHE}"
[[ "$HEADLESS" == "0" ]] && xhost +local:docker &>/dev/null 2>&1 || true

PYTHON="/isaac-sim/python.sh"
INFER_PY="source /isaac-sim/setup_python_env.sh && LD_PRELOAD=/isaac-sim/kit/libcarb.so /isaac-sim/kit/python/bin/python3"

run_eval() {
    local task="$1"
    local infer_args="--config ${INFER_CONFIG} --task ${task}"
    local sim_args="--config ${SIM_CONFIG} --task ${task}"
    local infer_name="eval_infer_${task}"
    local sim_name="eval_sim_${task}"

    info "==========================================="
    info "评估: ${task}"
    info "==========================================="

    docker rm -f "$infer_name" "$sim_name" &>/dev/null || true

    # 启动 infer 容器（CPU，后台）
    info "启动 infer 容器..."
    docker run --rm --name "$infer_name" \
        --entrypoint /bin/bash \
        --privileged --network host --user root --shm-size=8g \
        -v "${HOST_WS}:${CONTAINER_WS}:rw" -w "${CONTAINER_WS}" \
        "${INFER_IMAGE}" \
        -c "cd ${CONTAINER_WS}; ${PYTHON} -m pip install -i ${PIP_MIRROR} -e . --no-deps -q; ${PYTHON} -m pip install -i ${PIP_MIRROR} lz4 msgpack websockets pillow -q; ${INFER_PY} -m lerobot.scripts.ghrc_eval_infer ${infer_args}" \
        &>/tmp/eval_infer_${task}.log &

    # 等待 infer WebSocket 就绪
    info "等待 infer 就绪..."
    wait_for_infer_ready "${task}" "${infer_name}" "${INFER_CONTROL_PORT}" "${INFER_STREAM_PORT}" "${INFER_READY_TIMEOUT}"

    # 启动 sim-eval 容器（GPU，前台）
    info "启动 sim-eval 容器..."
    docker run --rm --name "$sim_name" \
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
        "${SIM_IMAGE}" \
        -c "cd ${CONTAINER_WS}; ${PYTHON} -m pip install -i ${PIP_MIRROR} -e . --no-deps -q; ${PYTHON} -m lerobot.scripts.ghrc_eval_sim ${sim_args}" 2>&1 \
        | grep -E "Episode.*step=|SUCCESS|FAILED|成功|失败|异常|汇总|Score|Connected|INIT" \
        | grep -vE "Warning.*usd|omni\.physicsschema" || true

    info "${task} 完成"
    docker stop "$infer_name" &>/dev/null || true
}

for t in "${TASKS[@]}"; do
    run_eval "$t"
done

[[ "$HEADLESS" == "0" ]] && xhost -local:docker &>/dev/null 2>&1 || true
info "全部完成！结果: eval_config/logs/sim_eval_container/"
