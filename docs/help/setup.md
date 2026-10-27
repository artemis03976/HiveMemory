---
title: Setup and Run HiveMemory
status: current
owner: project
scope: installation-development-docker-startup-and-claude-code-connector
code_paths:
  - pyproject.toml
  - docker/Dockerfile
  - docker/docker-compose.yml
  - frontend/package.json
  - frontend/vite.config.ts
  - src/hivememory/server/__main__.py
  - src/hivememory/server/app.py
  - scripts/hivememory-dev.sh
  - scripts/hivememory-dev.env.example
  - integrations/.claude-plugin/marketplace.json
  - integrations/claude-code/
  - configs/workspace_actors.yaml
related_contracts:
  - docs/contracts/routes-and-events.md
related_docs:
  - docs/system/import-bus.md
last_reviewed: 2026-10-27
---

# 安装与启动 HiveMemory

HiveMemory 当前提供两条明确运行路径：Docker 将前端、后端与 Qdrant 组合为一套本地服务；开发模式则分别运行 Qdrant、Python 后端和 Vite 前端。在 WSL/Linux 上，开发模式也可以交给脚本管理，脚本另有不需要 Docker 的长期运行模式（第 4.5 节）。请先选择一条路径，不要把两套端口和启动命令混用。服务运行起来之后，可以按第 9 节把 Claude Code 的对话接入 HiveMemory。

## 1. 前置条件

共同要求：

- Git；
- 可用的 LLM Provider API key；
- 能够下载 Python、Node 或模型依赖的网络环境。

Docker 路径还需要 Docker Engine 与 Compose v2。开发路径需要：

- Python 3.12 或更高版本；
- Node.js 20 LTS 或更高版本，以及 npm；
- 可访问的 Qdrant，本文使用 Docker 单独启动。

首次构建或运行可能下载 Embedding/Reranker 模型，需要数 GB 磁盘空间和较长等待时间。GPU 不是当前默认运行的必需条件。

## 2. 获取源码与准备凭证

```text
git clone https://github.com/artemis03976/HiveMemory.git
cd HiveMemory
```

在 PowerShell 中复制环境变量模板：

```powershell
Copy-Item -LiteralPath 'configs/.env.example' -Destination '.env'
```

在 Bash 中：

```bash
cp configs/.env.example .env
```

至少把默认模型对应的 Provider key 改为真实值：

```env
HIVEMEMORY__PROVIDERS__DEEPSEEK__API_KEY=sk-...
HIVEMEMORY__PROVIDERS__DEEPSEEK__API_BASE=https://api.deepseek.com
```

默认 `configs/models.yaml` 将 `deepseek-chat` 标为默认模型，并通过 `provider: deepseek` 引用上述凭证。不要把真实密钥写入受版本控制的 `configs/config.yaml` 或 `configs/models.yaml`。其他配置方式见[配置指南](./configuration.md)。

## 3. 路径 A：Docker 一体化运行

从仓库根目录执行：

```text
docker compose -f docker/docker-compose.yml up -d --build
```

Compose 会启动：

- `hivememory-app`：FastAPI、构建后的 React 前端和配置挂载；
- `qdrant`：向量存储；
- 可选的 `qdrant-web-ui` 不会在默认 profile 中启动。

打开：

- Web UI：`http://localhost:8000`
- Liveness：`http://localhost:8000/health`
- Readiness：`http://localhost:8000/health/ready`

可选调试 UI：

```text
docker compose -f docker/docker-compose.yml --profile debug up -d
```

随后访问 `http://localhost:6335`。Qdrant HTTP/gRPC 分别暴露在 `6333/6334`。

Dockerfile 默认在镜像构建阶段预下载模型，经运行时同一套配置加载与模型服务完成，模型落在配置的 `cache_dir`（默认 `data/model_cache`），运行时直接使用，因此首次 `--build` 可能耗时很长；运行容器仍会在后台完成模型 warmup。Compose healthcheck 当前检查 `/health`，它只证明 Web 服务存活，不代表模型已经 ready。

## 4. 路径 B：本地开发

### 4.1 只启动 Qdrant

```text
docker compose -f docker/docker-compose.yml up -d qdrant
```

本地后端默认连接 `127.0.0.1:6333/6334`。

### 4.2 创建 Python 环境

PowerShell：

```powershell
python -m venv .venv
& '.\.venv\Scripts\Activate.ps1'
python -m pip install -e '.[dev]'
```

Bash：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

如果只运行服务、不需要测试工具，可以安装 `-e .`。

安装后可以核对当前代码版本：

```text
python -c "import hivememory; print(hivememory.__version__)"
```

### 4.3 启动后端

```text
hivememory-server
```

包脚本调用 `hivememory.server.__main__`，监听 `0.0.0.0:8769`。启动后检查：

```text
curl http://localhost:8769/health
curl http://localhost:8769/health/ready
```

`/health/ready` 在模型尚未完成后台预热时返回 HTTP 503 与 `warming_up`，这是可预期状态。若服务进程退出或长时间无法 ready，再进入排障。

### 4.4 启动前端

在另一个终端：

```text
cd frontend
npm ci
npm run dev
```

打开 `http://127.0.0.1:5173`。Vite 将相对 `/api` 请求和 WebSocket 代理到 `http://localhost:8769`。

### 4.5 用脚本管理本机服务（WSL/Linux）

`scripts/hivememory-dev.sh` 在 WSL/Linux 上管理 Qdrant、后端与前端三个进程，Qdrant 使用本机的可执行文件，不需要 Docker。首次使用前复制配置模板，并把 `QDRANT_BIN` 改为本机 Qdrant 可执行文件的路径：

```bash
cp scripts/hivememory-dev.env.example .hivememory/dev.env
```

脚本需要 `uv`、`curl` 与 `setsid`；前端由 nvm 提供的 Node 运行（版本见 `HIVEMEMORY_NODE_VERSION`）。

脚本有两种模式，共用同一组进程与端口：

| 命令 | 运行的服务 | 后端 | 前端 |
|:---|:---|:---|:---|
| `start [all\|qdrant\|backend\|frontend]` | 全部，或只启动指定的一个 | 开发模式：默认随代码改动自动重载 | Vite（`5173`） |
| `serve [--build]` | Qdrant 与后端 | 不重载，单进程，与 Docker 镜像中的运行方式相同 | 由后端在 `8769` 提供 `frontend/dist` 的构建产物 |

- `serve` 适合长期运行，例如让 Claude Code 连接器持续导入对话：自动重载会在每次代码改动时重启后端，内存中的短期话题随之结算，进程内的重复导入检查也会清空。`serve --build` 先构建前端再启动；不加 `--build` 时使用已有的构建产物，它可能早于当前代码。
- 开发模式下，在 `.hivememory/dev.env` 中设置 `HIVEMEMORY_BACKEND_RELOAD=0` 可以关闭自动重载。
- `stop` 与 `restart` 同样接受服务名，缺省为全部；`restart` 保持后端原来的模式。`status` 显示各服务的状态与后端的模式，`logs [服务]` 查看日志（位于 `.hivememory/logs`）。
- 停止后端时最多等待 60 秒才强制结束，可用 `HIVEMEMORY_BACKEND_STOP_TIMEOUT` 调整。后端关闭时要把内存中的短期话题结算为记忆；提前强制结束会丢失这些话题，其中包括连接器已得到确认、不会再重发的轮次。
- 同一时间只运行一个后端：后端已以另一种模式运行时，脚本拒绝启动并提示先执行 `stop backend`；`8769` 上已有不由脚本启动的后端时同样拒绝。启动失败时只停止本次启动的服务，之前已在运行的服务不受影响。
- 脚本只停止自己启动的进程；它发现 Qdrant 已在脚本之外运行时直接复用，`stop` 不会停止它。

## 5. 端口与进程对照

| 端口 | 开发模式 | 脚本的 `serve` 模式 | Docker 模式 |
|:---:|:---|:---|:---|
| `5173` | Vite 前端 | 未使用 | 未使用 |
| `8769` | 后端 | FastAPI + 构建后的前端 | 未暴露 |
| `8000` | 未使用 | 未使用 | FastAPI + 构建后的前端 |
| `6333` | Qdrant HTTP | Qdrant HTTP | Qdrant HTTP |
| `6334` | Qdrant gRPC | Qdrant gRPC | Qdrant gRPC |
| `6335` | 可选 debug UI | 未使用 | 可选 debug UI |

## 6. 健康检查的含义

- `GET /health`：进程能够响应 HTTP，并返回与 Python 包和 OpenAPI 一致的代码版本；该字段不替代 Git tag 的发布状态；
- `GET /health/ready`：Patchouli 的 Embedding 与可选 Reranker 已完成加载；未完成时返回 503；
- Docker `hivememory-app` healthcheck 只调用 `/health`；部署脚本若要求“可以接收完整模型请求”，应另外检查 `/health/ready`。

## 7. 开发验证

后端默认测试集：

```text
pytest
```

它按 `pyproject.toml` 默认排除 `live_llm`、`e2e` 和 `slow`。前端检查：

```text
cd frontend
npm run lint
npm run build
```

前端当前没有独立自动化组件测试；lint 与 production build 是主要静态门槛。

## 8. 最小验收清单

- [ ] 所选运行模式的进程或容器都在运行；
- [ ] Qdrant ready；
- [ ] `/health` 返回成功；
- [ ] `/health/ready` 最终返回 `ready`；
- [ ] 默认模型对应 Provider 已配置有效 API key；
- [ ] 开发模式可打开 `5173`，Docker 模式可打开 `8000`；
- [ ] Chat 请求能够收到 SSE 终态，而不只是打开静态页面；
- [ ] 页面中的 Memory、Agent 和 Settings 数据已确认来自真实后端而不是 mock fallback。

遇到问题时见[故障排查](./troubleshooting.md)。

## 9. 接入 Claude Code

仓库提供 Claude Code 插件 `hivememory-connector`：Claude Code 的每一轮对话结束后，插件把这一轮（用户消息、回复、Claude Code 本地记录的思考、工具调用与结果）导入 HiveMemory，记忆随日常使用积累。插件只向 HiveMemory 输入，不向 Claude Code 注入记忆；它的 hook 只做本地记录，网络提交在后台进程中完成，HiveMemory 未运行时 Claude Code 照常工作，未导入的轮次在之后被唤醒时按顺序补发。工作原理见 [Import Bus](../system/import-bus.md)。

前置条件：

- HiveMemory 后端已按第 3 节或第 4 节运行；
- Claude Code 所在环境可以运行 `python3`。插件只用 Python 标准库，不需要安装依赖，也不需要安装 HiveMemory 包。

### 9.1 安装插件

`integrations/.claude-plugin/marketplace.json` 声明了一个本地 marketplace `hivememory-local`，只供本机安装。在 Claude Code 中执行（把路径换成本机仓库的实际位置）：

```text
/plugin marketplace add <HiveMemory 仓库路径>/integrations
/plugin install hivememory-connector@hivememory-local
```

插件挂载 `Stop`、`StopFailure`、`SessionStart` 与 `SessionEnd` 四个 hook。安装时 Claude Code 保存的是插件的副本：仓库中的连接器更新后，需要在 `/plugin` 中更新该插件才会生效。

### 9.2 配置连接器

连接器从环境变量读取配置。hook 由 Claude Code 启动，因此在启动 Claude Code 的环境中设置：

| 环境变量 | 默认值 | 说明 |
|:---|:---|:---|
| `HIVEMEMORY_URL` | `http://localhost:8000` | HiveMemory 后端地址。默认值对应 Docker 路径；本地开发路径（含脚本的 `serve` 模式）的后端在 `8769`，需要设为 `http://localhost:8769` |
| `HIVEMEMORY_USER_ID` | 不设置 | 以哪个用户导入；不设置时使用服务端的默认用户 `default` |
| `HIVEMEMORY_AGENT_ID` | `claude_code` | 以哪个 Agent 导入；不能使用保留的 `system` |
| `HIVEMEMORY_CONNECTOR_STATE_DIR` | 见第 9.4 节 | 连接器状态、日志与拒绝记录的目录 |

例如本地开发路径：

```bash
export HIVEMEMORY_URL=http://localhost:8769
```

### 9.3 访问登记

导入请求同样经统一认证网关，请求的用户与 Agent 必须在 `configs/workspace_actors.yaml` 中获准进入 Workspace。随仓库发布的登记已经为默认用户 `default` 登记了 `claude_code`，使用默认配置时不需要修改。这条记录的 `allowed_operations` 为空：导入只要求准入，不需要任何 actor 操作，空白名单也使这个 Agent 不继承默认用户的记忆操作权限。

- `HIVEMEMORY_USER_ID` 设为 `default` 以外的用户时，需要为该用户登记记录，否则导入返回 403，连接器停止提交并保留进度；
- 更换 `HIVEMEMORY_AGENT_ID` 时，默认的用户级记录会覆盖新的 Agent；如果希望它同样只能导入，按 `claude_code` 那条记录的写法为它增加精确记录；
- 修改登记后重启 HiveMemory。登记规则见[配置指南](./configuration.md#51-访问登记)。

### 9.4 确认导入与排查

连接器的状态目录依次取 `HIVEMEMORY_CONNECTOR_STATE_DIR`、Claude Code 为插件提供的数据目录（`CLAUDE_PLUGIN_DATA`），都没有时使用 `~/.hivememory/claude-code-connector`。目录中：

- `connector.log`：每一轮的确认（`confirmed turn=... status=accepted`）、停止与拒绝，只记录标识与状态，不记录对话内容；
- `rejections.jsonl`：被放弃的轮次，记录会话与轮次标识、HTTP 状态码与错误码。

使用中需要知道的行为：

- 只导入插件首次运行之后开始的会话；安装前已存在的会话不回填，恢复这类会话后只导入新的轮次。
- 一轮在结束后才导入：正常结束为 `completed`，因 API 错误结束为 `failed`，被用户中断、被下一条消息打断或随会话结束而结束的为 `cancelled`。
- 连接器没有常驻进程，只在 hook 触发时被唤醒：一轮结束或会话结束时同步当前会话，会话开始（含恢复会话）时补发全部会话中积压的轮次。HiveMemory 停止期间积压的轮次，会在它恢复后的下一次唤醒时按顺序补发。
- HiveMemory 暂不可用（连接失败、503）或连接器配置有误（400、403、404）时，连接器保留进度、停止提交，修正后在下一次唤醒时补发；配置有误时日志中的 `stopped` 行带有状态码与错误码。
- 单轮内容超过服务端的请求体上限（413，默认 8 MiB）或格式被拒绝（422）时，这一轮写入 `rejections.jsonl` 后跳过，不阻塞后续轮次；同一轮连续三次得到 500 也会被跳过。
