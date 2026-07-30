---
title: Setup and Run HiveMemory
status: current
owner: project
scope: installation-development-and-docker-startup
code_paths:
  - pyproject.toml
  - docker/Dockerfile
  - docker/docker-compose.yml
  - frontend/package.json
  - frontend/vite.config.ts
  - src/hivememory/server/__main__.py
  - src/hivememory/server/app.py
related_contracts:
  - docs/contracts/routes-and-events.md
last_reviewed: 2026-07-28
---

# 安装与启动 HiveMemory

HiveMemory 当前提供两条明确运行路径：Docker 将前端、后端与 Qdrant 组合为一套本地服务；开发模式则分别运行 Qdrant、Python 后端和 Vite 前端。请先选择一条路径，不要把两套端口和启动命令混用。

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

Dockerfile 默认在镜像构建阶段预下载模型，因此首次 `--build` 可能耗时很长；运行容器仍会在后台完成模型 warmup。Compose healthcheck 当前检查 `/health`，它只证明 Web 服务存活，不代表模型已经 ready。

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

## 5. 端口与进程对照

| 端口 | 开发模式 | Docker 模式 |
|:---:|:---|:---|
| `5173` | Vite 前端 | 未使用 |
| `8769` | `hivememory-server` 后端 | 未暴露 |
| `8000` | 未使用 | FastAPI + 构建后的前端 |
| `6333` | Qdrant HTTP | Qdrant HTTP |
| `6334` | Qdrant gRPC | Qdrant gRPC |
| `6335` | 可选 debug UI | 可选 debug UI |

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
