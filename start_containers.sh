#!/usr/bin/env bash
# 双容器启动脚本: infer (GPU) 与 sim-eval (GPU) 各一个容器，WebSocket 通信
# 容器启动后保持运行，通过 ./use_eval 执行评测命令
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFER_IMAGE="${INFER_IMAGE:-ghrc-eval-infer:latest}"
SIM_IMAGE="${SIM_IMAGE:-ghrc-eval-sim:latest}"
HOST_WS="${SCRIPT_DIR}"
CONTAINER_WS="/workspace/eval"
ISAAC_CACHE="${ISAAC_CACHE_ROOT:-${HOME}/.cache/isaac_sim_container}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
HEADLESS="${HEADLESS:-0}"

INFER_NAME="eval_infer_server"
SIM_NAME="eval_sim_server"

info "infer: ${INFER_IMAGE}  |  sim: ${SIM_IMAGE}"

command -v docker &>/dev/null || { error "未找到 docker"; exit 1; }
docker image inspect "$INFER_IMAGE" &>/dev/null 2>&1 || { error "镜像 ${INFER_IMAGE} 不存在"; exit 1; }
docker image inspect "$SIM_IMAGE"   &>/dev/null 2>&1 || { error "镜像 ${SIM_IMAGE} 不存在"; exit 1; }

for d in cache/kit cache/ov cache/pip cache/glcache cache/computecache data documents; do
    mkdir -p "${ISAAC_CACHE}/${d}"
done
mkdir -p "${HF_CACHE}"
[[ "$HEADLESS" == "0" ]] && xhost +local:docker &>/dev/null 2>&1 || true

# ---- 启动 infer 容器 ----
info "启动 infer 容器..."
docker rm -f "$INFER_NAME" &>/dev/null || true

docker run -dit \
    --name "$INFER_NAME" \
    --entrypoint /bin/bash \
    --privileged --network host --user root --gpus all --shm-size=8g \
    -v "${HOST_WS}:${CONTAINER_WS}:rw" -w "${CONTAINER_WS}" \
    "${INFER_IMAGE}"

# ---- 启动 sim 容器 ----
info "启动 sim 容器..."
docker rm -f "$SIM_NAME" &>/dev/null || true

docker run -dit \
    --name "$SIM_NAME" \
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
    "${SIM_IMAGE}"

info "容器启动完成"
echo
echo "infer: ${INFER_NAME}  |  sim: ${SIM_NAME}"
echo
echo "启动评测:  ./use_eval <task>"
echo "进入容器:  docker exec -it ${INFER_NAME} bash"
echo "           docker exec -it ${SIM_NAME} bash"
