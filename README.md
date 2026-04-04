# HiveMemory

[English README](README_EN.md) | [项目设计文档](docs/PROJECT.md) | [环境搭建指南](docs/SETUP.md) | [开发路线图](docs/ROADMAP.md)

> 为 LLM Agent 设计的持久化记忆与知识共享系统
> *The Hippocampus for Artificial Intelligence*

HiveMemory 是一套面向 LLM Agent 工作流的持久化记忆系统，目标是解决长上下文遗忘、跨会话知识无法复用、以及多 Agent 协作中的信息孤岛问题。系统会将对话中的高价值信息沉淀为可检索、可更新、可复用的记忆，并通过统一协议将这些记忆重新注入到后续任务中。

当前仓库已经提供可运行的 Python 后端、前端开发界面、向量存储与缓存基础设施，以及围绕 Patchouli 体系构建的主动对话、被动消息摄入、记忆检索、话题管理和运行时配置能力。

## 发布状态

- 当前版本：`0.1.0`
- 发布阶段：测试版 / Alpha
- Python 要求：`>=3.12`
- 许可证：Apache-2.0

v0.1.0 的 README 以**当前实现情况**为准，重点提供真实可执行的本地启动路径和系统概览；更完整的设计背景与架构推演请参阅 [docs/PROJECT.md](docs/PROJECT.md)。

## 当前版本已提供的能力

### 对话与接入模式

- **主动模式（Active mode）**：通过 `POST /api/v1/chat` 提供 SSE 流式对话，由 `PatchouliSystem.chat_stream()` 驱动完整生成循环与 MTP 执行
- `POST /api/v1/chat` 支持在请求体中携带 `generation_options`（`model` / `temperature` / `top_p` / `max_tokens`）作为单次对话覆盖参数，不会写入全局配置文件
- **被动模式（Passive mode）**：通过 `POST /api/v1/ingest` 接收外部框架的离散消息，由 `PatchouliSystem.ingest()` 负责缓冲、分析、检索和后续记忆沉淀

### 记忆与话题管理

- 语义搜索与记忆列表：`GET /api/v1/memories`
- 单条记忆查询：`GET /api/v1/memories/{memory_id}`
- 记忆删除：`DELETE /api/v1/memories/{memory_id}`
- 活跃话题列表：`GET /api/v1/topics`
- 手动触发话题结算：`POST /api/v1/topics/{topic_id}/trigger`
- 从活跃池驱逐话题：`DELETE /api/v1/topics/{topic_id}`

### 配置与可观测性

- 当前运行时配置：`GET /api/v1/config`
- 更新并持久化运行时配置：`POST /api/v1/config`
- 查看默认配置：`GET /api/v1/config/defaults`
- WebSocket 日志流：`WS /api/v1/ws/logs`
- 健康检查：`GET /health`
- 就绪检查：`GET /health/ready`

### 核心能力

- Patchouli 三位一体架构：The Eye / Retrieval Familiar / Librarian Core
- 进程内通信总线：SystemBus
- MTP（Memory Tool Protocol）协议，支持 `SEARCH / READ / RUN / WRITE / UPDATE`
- 基于 Qdrant 的持久化记忆存储
- Dense + Sparse 的混合检索路径
- 前端开发界面（Vite + React）

## 架构概览

HiveMemory 当前实现围绕 **Patchouli System** 展开，它不是单纯的聊天接口，而是一套将实时检索与后台记忆整理解耦的运行时系统。

### Patchouli 的主要组成

- **PatchouliSystem**：开发者直接使用的顶层入口，负责连接 The Eye 与 PatchouliKernel
- **The Eye**：交互入口与网关，负责意图识别、查询重写、流量分发
- **PatchouliKernel**：系统编排器，负责初始化基础设施、注册服务、连接 SystemBus
- **Retrieval Familiar**：Hot Path 检索服务，负责混合检索、重排序、上下文渲染
- **Librarian Core**：Cold Path 记忆服务，负责话题感知、记忆提取、生命周期维护
- **KoakumaRuntime**：MTP 执行器，负责在生成过程中拦截并执行记忆工具调用

### 热路径 / 冷路径

- **Hot Path（热路径）**：追求低延迟，负责当前请求的检索与上下文注入
- **Cold Path（冷路径）**：异步执行，负责对话后的整理、总结、写入、更新、归档

这套拆分使 HiveMemory 能同时兼顾：

- 对当前对话的快速响应
- 对历史知识的持续沉淀与复用

### MTP：记忆工具协议

HiveMemory 提供一套进程内协议，让 Worker Agent 可以在生成过程中主动访问记忆层：

- `SEARCH`：模糊检索，返回候选记忆索引
- `READ`：读取具体记忆内容
- `RUN`：执行内核工具或记忆中的代码片段
- `WRITE`：主动提交新的记忆写入意图
- `UPDATE`：主动提交已有记忆的更新意图

协议格式为：

```text
⟪ VERB | TARGET | key="value" ⟫
```

更完整的架构背景、设计动机与术语说明见 [docs/PROJECT.md](docs/PROJECT.md)。

## API 概览

| 接口路径 | 方法 | 用途 |
| --- | --- | --- |
| `/health` | GET | 存活检查 |
| `/health/ready` | GET | 模型预热完成后的就绪检查 |
| `/api/v1/chat` | POST | SSE 流式主动对话 |
| `/api/v1/ingest` | POST | 被动消息摄入 |
| `/api/v1/memories` | GET | 语义搜索 / 列出记忆 |
| `/api/v1/memories/{memory_id}` | GET | 获取单条记忆 |
| `/api/v1/memories/{memory_id}` | DELETE | 删除单条记忆 |
| `/api/v1/topics` | GET | 获取活跃话题 |
| `/api/v1/topics/{topic_id}/trigger` | POST | 手动触发话题结算 |
| `/api/v1/topics/{topic_id}` | DELETE | 从活跃池删除话题 |
| `/api/v1/config` | GET / POST | 读取 / 更新运行时配置 |
| `/api/v1/config/defaults` | GET | 获取默认配置 |
| `/api/v1/ws/logs` | WS | 获取实时日志流 |

## 环境要求

运行本项目建议准备以下环境：

- Python 3.12+
- Node.js（建议使用较新的 LTS 版本，用于前端开发界面）
- Docker / Docker Compose（用于本地启动 Qdrant 和 Redis）
- 可用的 LLM API Key（例如 DeepSeek / OpenAI 兼容接口）

此外，Embedding 与 Reranker 模型在初次启动时可能需要下载与预热，因此服务启动成功并不等于模型已经 ready。

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/artemis03976/HiveMemory.git
cd HiveMemory
```

### 2. 启动基础设施

本仓库提供的 Docker Compose 会启动：

- Qdrant（HTTP `6333` / gRPC `6334`）
- Redis（`6379`）
- 可选的 Qdrant Web UI（`6335`，仅 debug profile）

```bash
docker-compose -f docker/docker-compose.yml up -d
```

如果你需要 Qdrant Web UI：

```bash
docker-compose -f docker/docker-compose.yml --profile debug up -d
```

> 注意：Compose 中 Redis 默认密码为 `hivememory_redis_pass`，而 `configs/.env.example` 中该值留空。如果直接使用仓库自带 Compose，请在环境变量中把 `HIVEMEMORY__REDIS__PASSWORD` 设置为 `hivememory_redis_pass`。

### 3. 配置环境变量

先复制环境变量模板：

```bash
cp configs/.env.example .env
```

然后按需修改 `.env`。其中：

- `.env` / 环境变量：主要放 API Key、Qdrant / Redis 地址、调试开关等
- `configs/config.yaml`：主要放业务逻辑和算法参数（检索、感知、生成、生命周期等）

至少需要检查：

- `HIVEMEMORY__LLM__WORKER__API_KEY`
- `HIVEMEMORY__LLM__LIBRARIAN__API_KEY`
- `HIVEMEMORY__REDIS__PASSWORD`
- `HIVEMEMORY__QDRANT__HOST` / `PORT`

### 4. 安装后端

推荐使用 `pyproject.toml` 定义的包安装方式：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

如果你需要运行测试或开发工具：

```bash
pip install -e ".[dev]"
```

### 5. 启动后端服务

推荐使用包脚本：

```bash
hivememory-server
```

默认后端地址为：

- `http://localhost:8769`

服务启动后可先检查：

```bash
curl http://localhost:8769/health
curl http://localhost:8769/health/ready
```

其中：

- `/health` 返回服务是否存活
- `/health/ready` 用于检查模型是否完成后台预热；若仍在预热，会返回 `503 warming_up`

### 6. 启动前端开发界面

```bash
cd frontend
npm install
npm run dev
```

当前前端开发服务器默认运行在：

- `http://127.0.0.1:5173`

前端开发代理默认转发 `/api` 到：

- `http://localhost:8769`

## 配置模型

HiveMemory 当前采用“环境变量 + YAML”分层配置：

### 环境变量

`configs/.env.example` 展示了推荐格式，环境变量统一使用 `HIVEMEMORY__` 前缀，例如：

- `HIVEMEMORY__LLM__WORKER__MODEL`
- `HIVEMEMORY__LLM__LIBRARIAN__API_KEY`
- `HIVEMEMORY__QDRANT__HOST`
- `HIVEMEMORY__REDIS__PASSWORD`
- `HIVEMEMORY__LOGGING__LEVEL`

### YAML 配置

[configs/config.yaml](configs/config.yaml) 定义默认运行参数，包括：

- `llm`：gateway / librarian / worker
- `embedding`
- `qdrant`
- `redis`
- `gateway`
- `perception`
- `generation`
- `retrieval`
- `lifecycle`
- `logging`

推荐做法是：

- 将密钥、地址、端口、环境切换放在环境变量中
- 将业务逻辑参数保留在 `config.yaml` 中

## 开发者入口

如果你希望直接在 Python 中集成系统，主入口是：

- `hivememory.patchouli.system.PatchouliSystem`

它当前提供两种主要接入方式：

- `chat()` / `chat_stream()`：主动模式，系统直接驱动生成与 MTP 循环
- `ingest()`：被动模式，适合接入 Discord Bot、微信机器人或其他外部框架

如果你只需要 HTTP 接口，可直接使用 FastAPI 服务；如果你要把 HiveMemory 嵌入已有 Agent 框架，通常从 passive ingest 模式开始会更自然。

## 项目结构

```text
HiveMemory/
├── configs/                 # 环境变量模板与主配置
├── docker/                  # 本地基础设施（Qdrant / Redis）
├── docs/                    # 项目设计与规划文档
├── frontend/                # React + Vite 前端开发界面
├── scripts/                 # 启动与辅助脚本
├── src/hivememory/
│   ├── core/                # 核心数据模型
│   ├── engines/             # Gateway / Retrieval / Perception / Generation / Lifecycle
│   ├── infrastructure/      # Storage / LLM / SystemBus / WebSocket
│   ├── patchouli/           # Patchouli 体系、Kernel、MTP、WorkerAgent
│   └── server/              # FastAPI 服务入口与路由
└── tests/                   # 单元测试、集成测试、端到端测试
```

## 开发说明

- 测试默认由 `pytest` 驱动，配置见 [pyproject.toml](pyproject.toml)
- 前端提供 `npm run dev` / `npm run build` / `npm run lint`
- 后端健康检查与 readiness 已内置在 API 中
- 日志可通过 WebSocket 推送到前端

## 文档索引

- [README_EN.md](README_EN.md) — 英文版 README
- [docs/PROJECT.md](docs/PROJECT.md) — 项目背景、架构设计、Patchouli 体系与 MTP 说明
- [docs/SETUP.md](docs/SETUP.md) — 环境搭建说明
- [docs/ROADMAP.md](docs/ROADMAP.md) — 版本规划与后续方向

## 贡献

欢迎提交 Issue 和 Pull Request。当前仓库正面向 v0.1.0 测试版收敛，提交改动前建议先核对实现与文档是否一致，避免把旧的启动方式、端口和配置说明重新写回文档。

## 许可证

本项目采用 Apache-2.0 许可证。详见 [LICENSE](LICENSE)。
