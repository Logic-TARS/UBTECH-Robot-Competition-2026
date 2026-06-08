# 评估系统使用指南

双容器隔离评测，WebSocket（msgpack + lz4）通信。infer 用 kit 裸 Python 跑 CPU 推理，sim-eval 用 Isaac Sim 独占 GPU。

---

## 前置条件

- 已构建 `ghrc-eval-infer` / `ghrc-eval-sim` 镜像
- 基线模型放置于 `challenge2026_baseline/`
- 12G 显存可正常运行

### 构建镜像

```bash
docker build -f docker/Dockerfile.eval_infer -t ghrc-eval-infer:latest .
docker build -f docker/Dockerfile.eval_sim   -t ghrc-eval-sim:latest .
```

---

## 评测引擎配置

`eval_config/eval_infer.yaml` 和 `eval_config/eval_sim.yaml`，任务名通过 `--task` CLI 覆盖。

- `eval_infer.yaml`: `device: cpu`，`policy_type`（act / pi0 / diffusion）
- `eval_sim.yaml`: `device: auto`，`num_episodes`，`enable_assertion`

### 环境变量

常用运行时覆盖：

```bash
export INFER_IMAGE=ghrc-eval-infer:latest
export SIM_IMAGE=ghrc-eval-sim:latest
export INFER_CONFIG=eval_config/eval_infer.yaml
export SIM_CONFIG=eval_config/eval_sim.yaml
export INFER_READY_TIMEOUT=300
export HEADLESS=1
```

- `HEADLESS=1`: 无头模式，适合服务器和 CI
- `HEADLESS=0`: 桌面调试模式，显示仿真窗口

### 单任务运行

```bash
./run_eval.sh task4    # 单任务
./run_eval.sh all      # 全部 4 个任务
```

### 运行时行为

1. 后台启动 infer 容器
2. 等待 WebSocket 端口就绪（默认 8765 / 8766）
3. 前台启动 sim-eval 容器
4. 终端输出关键日志行，完整日志写入 `/tmp/eval_infer_{task}.log`
5. 评测结果写入 `logs/sim_eval_container/`

### 自定义策略接入

参考 `src/lerobot/sim_eval/policy_adapter.py`，实现 `PolicyAdapter` 后在 `eval_infer.yaml` 配置 `adapter_class`。

### 故障排查

| 问题 | 检查项 |
|------|--------|
| infer 启动后无响应 | `docker logs eval_infer_{task}` 查看日志 |
| sim-eval 连接失败 | 确认 infer 的 8765/8766 端口已监听 |
| GPU 显存不足 | 12G 显存为最低要求，可减少 `num_episodes` |

---

## 自动化编排

编排系统从飞书表格读取选手提交的镜像，自动拉取、评测、回写分数。

### 前置条件

- 飞书管理员权限
- 镜像仓库（Docker Hub / ACR）访问账号

### 飞书配置

**1. 创建应用**

打开 [飞书开发者后台](https://open.feishu.cn)，创建企业自建应用，记录 **App ID** 和 **App Secret**。

**2. 开通权限**

应用详情 → 权限管理，勾选：

| 权限 | 用途 |
|------|------|
| `bitable:app` | 读取多维表格 |
| `base:record:update` | 写入分数 |

勾选后点右上角 **"发布新版本"**。

**3. 添加到表格**

打开飞书表格 → 右上角 ... → 更多 → 添加文档应用 → 搜索应用并添加。

**4. 获取表格 ID**

```
https://xxx.feishu.cn/base/BITABLE_APP_TOKEN?table=TABLE_ID
```

**5. 表格列结构**

| 列名 | 类型 | 说明 |
|------|------|------|
| 队伍编号 | 文本 | 唯一标识 |
| 队伍名称 | 文本 | 显示名 |
| 镜像名称 | 链接 | 选手 Docker 镜像地址 |
| 评审状态 | 文本 | 初始"评测中" |
| 得分 | 文本 | 系统自动写入 |

### 编排器配置

`eval_config/eval_orchestrator.yaml`：

```yaml
feishu:
  bitable_app_token: "xxx"
  table_id: "xxx"
registry:
  server: "your-registry.example.com"
eval:
  sim_image: "ghrc-eval-sim:latest"
  tasks: ["task4", "task1", "task2", "task3"]
  single_task_timeout: 7200
  results_dir: "logs/sim_eval_container"
```

### 凭据配置

```bash
cp .env.example .env
# 编辑填入：
#   FEISHU_APP_ID=cli_xxxxxxxx
#   FEISHU_APP_SECRET=xxxxxxxx
#   DOCKER_USERNAME=xxxx
#   DOCKER_PASSWORD=xxxx
```

### 运行

**手动运行：**

```bash
source .env
export FEISHU_APP_ID FEISHU_APP_SECRET DOCKER_USERNAME DOCKER_PASSWORD
PYTHONPATH=src python -m lerobot.scripts.ghrc_eval_orchestrator
```

**常用参数：**

| 参数 | 说明 |
|------|------|
| `--keep-images` | 评测后保留镜像 |
| `--skip-docker` | 跳过镜像拉取 |
| `--mock-eval` | 生成模拟分数（测试用） |
| `--mock` | 完整模拟模式，无需飞书/Docker |
| `-v` | 详细日志 |

**定时运行：**

```bash
crontab -e
```

```
*/30 * * * * source /path/to/.env && cd /path/to/project && export FEISHU_APP_ID FEISHU_APP_SECRET DOCKER_USERNAME DOCKER_PASSWORD && PYTHONPATH=src python -m lerobot.scripts.ghrc_eval_orchestrator >> /var/log/eval_orchestrator.log 2>&1
```

`flock` 锁防止并发执行。

### 工作流程

```
选手提交镜像 → 飞书表格新行（评审状态=评测中）
     │
     ▼
cron 每 30 分钟触发
     ├── 读取飞书，筛"评测中"记录
     ├── 登录镜像仓库
     ├── docker pull 选手镜像
     ├── 调用 run_eval.sh 执行 4 任务评测
     ├── 收集 summary JSON，计算得分
     └── 回写飞书：状态→评测完成，得分→写入
```

**状态说明：**

| 状态 | 含义 |
|------|------|
| 评测中 | 待评测 |
| 评测完成 | 评测成功 |
| 评测失败 | 评测异常 |
| 镜像异常 | 镜像拉取失败 |

已评测的记录自动跳过。需要重新评测时手动将状态改回"评测中"。

**本地缓存：** 每次读写飞书时同步更新 `logs/eval_cache.csv`，飞书故障时自动回退到缓存数据。

### 编排器故障排查

| 问题 | 检查项 |
|------|--------|
| 飞书读写失败 | 权限是否开通并发布、应用是否加到表格 |
| Docker 登录失败 | `.env` 凭据是否正确、registry 地址是否匹配 |
| 镜像拉取超时 | 镜像约 20GB，首次 5-10 分钟属正常 |
| 定时任务没跑 | `crontab -l` 检查，查看 `/var/log/eval_orchestrator.log` |
