#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-task4}"

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

INFER_NAME="eval_infer_server"
SIM_NAME="eval_sim_server"

CONTAINER_WS="/workspace/eval"

PYTHON="/isaac-sim/python.sh"

INFER_PY="source /isaac-sim/setup_python_env.sh && 
LD_PRELOAD=/isaac-sim/kit/libcarb.so 
/isaac-sim/kit/python/bin/python3"

PIP_MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"

infer_args="--config eval_config/eval_infer.yaml --task ${TASK}"
sim_args="--config eval_config/eval_sim.yaml --task ${TASK}"

info "启动 infer websocket server..."

docker exec -d "${INFER_NAME}" bash -c "
cd ${CONTAINER_WS};

${PYTHON} -m pip install -i ${PIP_MIRROR} -e . --no-deps -q;

${PYTHON} -m pip install 
-i ${PIP_MIRROR} 
lz4 msgpack websockets pillow -q;

${INFER_PY} 
-m lerobot.scripts.ghrc_eval_infer 
${infer_args}
"

info "等待 infer 就绪..."

for i in $(seq 1 60); do
curl -s --max-time 1 http://localhost:8765/ >/dev/null 2>&1 && break
sleep 2
done

info "启动 sim eval..."

docker exec -it "${SIM_NAME}" bash -c "
cd ${CONTAINER_WS};

${PYTHON} -m pip install 
-i ${PIP_MIRROR} 
-e . 
--no-deps -q;

${PYTHON} 
-m lerobot.scripts.ghrc_eval_sim 
${sim_args}
" 2>&1 | grep -E 
"Episode.*step=|SUCCESS|FAILED|成功|失败|异常|汇总|Score|Connected|INIT" 
| grep -vE 
"Warning.*usd|omni.physicsschema" || true

info "任务完成"
