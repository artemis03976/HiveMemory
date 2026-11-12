---
title: Setup and Run HiveMemory
status: current
owner: project
scope: installation-development-docker-startup-and-harness-plugin-entry
code_paths:
  - pyproject.toml
  - docker/Dockerfile
  - docker/docker-compose.yml
  - frontend/package.json
  - frontend/vite.config.ts
  - src/hivememory/server/__main__.py
  - src/hivememory/server/app.py
  - src/hivememory/server/management_credential.py
  - scripts/hivememory-dev.sh
  - scripts/hivememory-dev.env.example
related_contracts:
  - docs/contracts/routes-and-events.md
related_docs:
  - docs/help/claude-code-plugin.md
  - docs/help/deepseek-harness-plugin.md
  - docs/help/codex-plugin.md
  - docs/help/pi-plugin.md
last_reviewed: 2026-11-12
---

# 安装与启动 HiveMemory

只想使用 HiveMemory、不做开发时，在 Windows 上安装桌面客户端即可，见[使用桌面客户端](./desktop-client.md)。以下是从源码运行的方式。

HiveMemory 当前提供两条明确运行路径：Docker 将前端、后端与 Qdrant 组合为一套本地服务；开发模式则分别运行 Qdrant、Python 后端和 Vite 前端。在 WSL/Linux 上，开发模式也可以交给脚本管理，脚本另有不需要 Docker 的长期运行模式（第 4.5 节）。请先选择一条路径，不要把两套端口和启动命令混用。服务运行起来之后，可以按第 9 节把 Claude Code、DeepSeek Harness 等外部 harness 接入 HiveMemory：对话回流与记忆访问。

## 1. 前置条件

共同要求：

- Git；
- 可用的 LLM Provider API key；
- 能够下载 Python、Node 或模型依赖的网络环境。

Docker 路径还需要 Docker Engine 与 Compose v2。开发路径需要：

- Python 3.12 或更高版本；
- Node.js 20 LTS 或更高版本，以及 npm；
- 可访问的 Qdrant，本文使用 Docker 单独启动。

Qdrant 服务端与依赖中的 `qdrant-client` 保持同一小版本，当前为 1.19：Compose 使用 `qdrant/qdrant:v1.19.2`，脚本模式下 `QDRANT_BIN` 也应指向 1.19.x 的可执行文件。两者一起升级。已有数据时，Qdrant 只能逐个小版本升级，每个中间小版本都要换成它的最新补丁版本启动一次，完成存储迁移（例如从 1.16.3 依次经 1.17.1、1.18.3 再到 1.19.2），单节点也不能跳过。升级后不能降级，升级前先在 Qdrant 停止时备份存储目录或数据卷。

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

首次启动时从模板生成的 `configs/models.yaml` 将 `deepseek-chat` 标为默认模型，并通过 `provider: deepseek` 引用上述凭证。不要把真实密钥写入 `configs/config.yaml`、`configs/config.local.yaml` 或 `configs/models.yaml`。其他配置方式见[配置指南](./configuration.md)。

## 3. 路径 A：Docker 一体化运行

从仓库根目录执行：

```text
docker compose -f docker/docker-compose.yml up -d --build
```

Compose 会启动：

- `hivememory-app`：FastAPI、构建后的 React 前端和配置挂载；
- `qdrant`：向量存储。

打开：

- Web UI：`http://localhost:8000`
- Liveness：`http://localhost:8000/health`
- Readiness：`http://localhost:8000/health/ready`

Compose 只在宿主的本机地址上开放 `8000`；需要从其他机器访问时，自行把 `docker/docker-compose.yml` 中的端口映射改回 `8000:8000`。页面会自动取得管理凭据，在终端中调用管理接口见第 4.3 节。

Qdrant HTTP/gRPC 分别暴露在 `6333/6334`，Qdrant 镜像自带的管理界面位于 `http://localhost:6333/dashboard`。

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

包脚本调用 `hivememory.server.__main__`，默认只监听本机地址 `127.0.0.1:8769`；需要对外监听时，以环境变量 `HIVEMEMORY_BACKEND_HOST` 显式指定（例如 `0.0.0.0`）。启动后检查：

```text
curl http://localhost:8769/health
curl http://localhost:8769/health/ready
```

`/health/ready` 在模型尚未完成后台预热时返回 HTTP 503 与 `warming_up`，这是可预期状态；模型加载失败时返回 503 与 `model_load_failed`，并在 `failed_models` 中列出失败的模型。若服务进程退出、长时间无法 ready 或报告加载失败，再进入排障。

ready 表示嵌入与重排模型都至少成功加载过一次，不表示模型此刻在内存中。默认配置下模型空闲一段时间后卸载、归还内存（嵌入模型 30 分钟、重排模型 10 分钟），`/health/ready` 仍为 ready；卸载后的首次使用要重新加载模型，单个模型约需 1～3 秒，首次检索可能跳过重排。需要模型一直驻留时，把 `shared.embedding.default.residency` 与 `patchouli.retrieval.reranker.residency` 设为 `resident`（[配置](../system/configuration.md)）。

后端首次启动时生成管理凭据 `configs/management.token`（已被 `.gitignore` 排除）。`/api/` 下除对话导入之外的接口都要求它，前端会自动带上（第 4.4 节）；在终端中直接调用时需要自己带上，否则返回 401：

```bash
curl -H "Authorization: Bearer $(cat configs/management.token)" \
  http://127.0.0.1:8769/api/v1/memories
```

删除这个文件后重启后端会生成新的凭据，开发模式下的 Vite 不需要重启。外部 harness 的插件只调用 `/mcp` 与对话导入，不需要管理凭据。管理凭据只防误用：以同一系统用户运行的程序能读到这个文件（[认证与访问登记](../workspace/authentication.md#8-已知限制)）。

### 4.4 启动前端

在另一个终端：

```text
cd frontend
npm ci
npm run dev
```

打开 `http://127.0.0.1:5173`。Vite 将相对 `/api` 请求和 WebSocket 代理到 `http://localhost:8769`，转发时从 `configs/management.token` 读出管理凭据并附上；Vite 先于后端启动时，后端生成凭据之后的请求即可通过。

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

- `serve` 适合长期运行，例如让外部 harness 的插件持续导入对话：自动重载会在每次代码改动时重启后端，内存中的短期话题随之结算，进程内的重复导入检查也会清空。`serve --build` 先构建前端再启动；不加 `--build` 时使用已有的构建产物，它可能早于当前代码。
- 开发模式下，在 `.hivememory/dev.env` 中设置 `HIVEMEMORY_BACKEND_RELOAD=0` 可以关闭自动重载。
- `stop` 与 `restart` 同样接受服务名，缺省为全部；`restart` 保持后端原来的模式。`status` 显示各服务的状态与后端的模式，`logs [服务]` 查看日志（位于 `.hivememory/logs`）。
- 后端自己写 `backend.log`，每天零点后改名为 `backend.log.YYYY-MM-DD` 并另起当天的文件，归档不自动清理；`logs backend` 跟随当天的文件。后端在日志配置生效之前的输出（如导入错误与自动重载的提示）写入 `backend.stdout.log`，启动失败时脚本会把两份日志的末尾一并打印出来。
- 后端另写 `runtime-events.jsonl`，每行一个运行时事件，切分方式与 `backend.log` 相同。排查一个任务、进程或话题时，用 `python scripts/runtime_event_timeline.py task_id=<ID> .hivememory/logs/runtime-events.jsonl*` 按写入顺序列出它的事件（也可用 `process_id`、`topic_id`、`trace_id` 或 `agent_run_id`）。
- 停止后端时最多等待 60 秒才强制结束，可用 `HIVEMEMORY_BACKEND_STOP_TIMEOUT` 调整。后端关闭时结算内存中的短期话题，但不在关闭时生成记忆：生成工作保存到 `.hivememory/pending_generation.json`，下次启动后在后台生成，因此正常关闭通常几秒内完成。提前强制结束会丢失这些话题与工作，其中包括连接器已得到确认、不会再重发的轮次。
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
- [ ] 前端没有以 debug 模式（`VITE_DEBUG=true`）运行，页面中的 Memory、Agent 和 Settings 数据来自真实后端而不是 mock fallback。

遇到问题时见[故障排查](./troubleshooting.md)。

## 9. 接入外部 harness

服务运行起来之后，可以把外部 harness 以插件接入 HiveMemory：对话回流把每一轮已经结束的对话导入 HiveMemory，记忆工具让 harness 中的 agent 检索、读取、写入与更新记忆。新部署不连接任何 harness；HiveMemory 在 Windows 上运行时，在设置页的“插件”页为 harness 安装插件，HiveMemory 代为运行 harness 的插件命令、登记接入并交付凭据。每个 harness 的安装、确认与排查各有一份手册：

- [接入 Claude Code](./claude-code-plugin.md)：插件 `hivememory-connector`（`integrations/hivememory/`），接入身份 `harness:claude-code`；
- [接入 Codex](./codex-plugin.md)：同一个插件目录中的 Codex 插件，接入身份 `harness:codex`，只支持 Codex 与 HiveMemory 都在 Windows 上原生运行；
- [接入 pi](./pi-plugin.md)：同一个插件目录作为 pi 包安装的扩展，接入身份 `harness:pi`，只支持 pi 与 HiveMemory 都在 Windows 上原生运行；
- [接入 DeepSeek Harness](./deepseek-harness-plugin.md)：同一个插件目录链接安装到 dsh 桌面端，接入身份 `harness:dsh`；命令行版与不在 Windows 上运行的 HiveMemory 按手册手动安装。

长期接入时，本地开发路径建议用脚本的 `serve` 模式运行后端（第 4.5 节）。
