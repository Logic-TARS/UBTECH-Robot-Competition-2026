#!/usr/bin/env bash

docker rm -f eval_infer_server eval_sim_server 2>/dev/null || true

xhost -local:docker &>/dev/null 2>&1 || true

echo "容器已停止"
