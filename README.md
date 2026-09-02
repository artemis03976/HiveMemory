# HiveMemory

[English README](README_EN.md) | [项目文档索引](docs/PROJECT.md) | [当前架构](docs/architecture/overview.md) | [安装与启动](docs/help/setup.md) | [开发路线图](docs/ROADMAP.md)

> 为 LLM Agent 设计的持久化记忆与知识共享系统
> *The Hippocampus for Artificial Intelligence*

HiveMemory 是一套面向 LLM Agent 的持久化记忆管理系统，目标是解决长上下文遗忘、跨会话知识无法复用、以及多 Agent 协作中的信息孤岛问题。系统会将对话中的高价值信息沉淀为可检索、可更新、可使用的记忆，并通过统一协议将这些记忆重新注入到后续任务中。

当前仓库已经提供可运行的 Python 后端、前端开发界面、向量存储与缓存基础设施，以及由顶层 HiveMemory System 编排 Gateway、Patchouli 和 Alice 三个同级子系统的 v0.6.1 发布基线。

## 发布状态

- 最新已发布标签：`v0.6.1`
- 当前发布基线：`v0.6.1`
- 代码与包版本：`0.6.1`
- Python 要求：`>=3.12`
- 许可证：Apache-2.0

当前系统设计见 [docs/architecture/overview.md](docs/architecture/overview.md)，全局文档入口见 [docs/PROJECT.md](docs/PROJECT.md)。

## 当前版本已提供的能力

### 对话与接入模式

- **主动模式（Active mode）**：通过 `POST /api/v1/chat` 提供 SSE 流式对话，由 `ChatApplicationService` 编排 Patchouli prepare/finalize 与 Alice Agent 执行
- `POST /api/v1/chat` 支持在请求体中携带 `generation_options`（`model` / `temperature` / `top_p` / `max_tokens`）作为单次对话覆盖参数，不会写入全局配置文件
- **被动模式（Passive mode）**：通过 `POST /api/v1/ingest` 接收外部框架的离散事件，由 System 层 `PassiveIngressService` 负责编排 Gateway 决策、缓冲、检索和 Patchouli 提交

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

- v0.6 子系统架构：顶层 `HiveMemorySystem` 与同级 Gateway、Patchouli、Alice
- Gateway 系统入口：命令、话题路由、查询分析、取消/超时和保守降级
- 进程内运行时总线：AsyncSystemBus / GlobalSystemBus / 子系统私有 bus
- MTP（Memory Tool Protocol）协议，支持 `SEARCH / READ / RUN / WRITE / UPDATE / CALL`
- 基于 Qdrant 的持久化记忆存储
- Dense + Sparse 的混合检索路径
- 前端开发界面（Vite + React）

## 架构概览

HiveMemory 当前实现围绕 **System / Service / Runtime** 分层展开。顶层 System 负责应用编排与全局路由，Gateway 负责入口决策，Patchouli 负责记忆域能力，Alice 负责 Agent 执行与 MTP/工具运行时。

### 主要运行时组成

- **HiveMemorySystem**：顶层宿主，装配全局总线、应用服务、Gateway、Patchouli 与 Alice
- **ChatApplicationService**：主动 chat 编排服务，负责 `prepare -> Alice run -> finalize`
- **GatewaySystem / GatewayRuntime**：入口决策子系统，负责系统指令、话题路由、查询分析与稳定决策投影
- **PatchouliSystem / PatchouliRuntime**：记忆子系统宿主与运行时，管理 retrieval、perception、generation、lifecycle 与 storage 能力
- **Retrieval Familiar**：Hot Path 检索服务，负责混合检索、重排序、上下文渲染
- **Librarian Core**：Cold Path 记忆服务，负责话题感知、记忆提取、生命周期维护
- **AliceSystem / AliceRuntime**：Agent 运行时子系统，持有 Agent runtime 与 Koakuma 工具 runtime
- **KoakumaRuntime**：Alice 在 Agent 生成过程中使用的 MTP/工具执行器

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
- `CALL`：挂起当前 frame 并委派给子 Agent

协议格式为：

```text
⟪ VERB | TARGET | key="value" ⟫
```

完整协议见 [docs/contracts/mtp.md](docs/contracts/mtp.md)，总体设计入口见 [docs/PROJECT.md](docs/PROJECT.md)。

## API 概览

| 接口路径                                | 方法         | 用途           |
| ----------------------------------- | ---------- | ------------ |
| `/health`                           | GET        | 存活检查         |
| `/health/ready`                     | GET        | 模型预热完成后的就绪检查 |
| `/api/v1/chat`                      | POST       | SSE 流式主动对话   |
| `/api/v1/ingest`                    | POST       | 被动消息摄入       |
| `/api/v1/memories`                  | GET        | 语义搜索 / 列出记忆  |
| `/api/v1/memories/{memory_id}`      | GET        | 获取单条记忆       |
| `/api/v1/memories/{memory_id}`      | DELETE     | 删除单条记忆       |
| `/api/v1/topics`                    | GET        | 获取活跃话题       |
| `/api/v1/topics/{topic_id}/trigger` | POST       | 手动触发话题结算     |
| `/api/v1/topics/{topic_id}`         | DELETE     | 从活跃池删除话题     |
| `/api/v1/config`                    | GET / POST | 读取 / 更新运行时配置 |
| `/api/v1/config/defaults`           | GET        | 获取默认配置       |
| `/api/v1/ws/logs`                   | WS         | 获取实时日志流      |

## 环境要求

运行本项目建议准备以下环境：

- Docker / Docker Compose（推荐，用于一键部署）
- 或手动搭建环境：
  - Python 3.12+
  - Node.js（用于前端开发）
  - 可用的 LLM API Key（例如 DeepSeek / OpenAI 兼容接口）

此外，Embedding 与 Reranker 模型在初次启动时可能需要下载与预热，因此服务启动成功并不等于模型已经 ready。

## 快速开始

### 方式一：Docker 一键部署 (推荐)

如果你只是想快速体验测试版，我们强烈推荐使用 Docker 一键部署：

```bash
# 1. 克隆仓库
git clone https://github.com/artemis03976/HiveMemory.git
cd HiveMemory

# 2. 复制并修改环境变量文件 (填入你的 LLM API Key)
cp configs/.env.example .env

# 3. 一键启动 (包含 Qdrant 和 HiveMemory 后端应用)
docker compose -f docker/docker-compose.yml up -d --build
```

启动成功后，直接在浏览器中打开 **<http://localhost:8000>** 即可开始使用完整的 Web 界面！

### 方式二：本地开发环境搭建

先复制环境变量模板：

```bash
cp configs/.env.example .env
```

然后按需修改 `.env`。其中：

- `.env` / 环境变量：主要放 API Key、Qdrant 地址、调试开关等
- `configs/config.yaml`：主要放业务逻辑和算法参数（检索、感知、生成、生命周期等）

至少需要检查：

- `HIVEMEMORY__PROVIDERS__DEEPSEEK__API_KEY`（或默认模型对应的 Provider）
- `configs/models.yaml` 中默认模型的 `id`、`litellm_model` 与 `provider`
- `HIVEMEMORY__PATCHOULI__STORAGE__HOST` / `PORT`

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
npm ci
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

- `HIVEMEMORY__PROVIDERS__DEEPSEEK__API_KEY`
- `HIVEMEMORY__GATEWAY__WORKFLOW__DEFAULT_REQUEST_TIMEOUT_MS`
- `HIVEMEMORY__PATCHOULI__STORAGE__HOST`
- `HIVEMEMORY__LOGGING__LEVEL`

### YAML 配置

[configs/config.yaml](configs/config.yaml) 定义默认运行参数，包括：

- `system`、`logging`、`scheduler`、`runtime_events` 与 `i18n`
- `shared`：Gateway/Librarian LLM 引用、Embedding 与 Provider 默认值
- `gateway`、`passive_ingress` 与 `memory_compiler`
- `patchouli`：storage、perception、generation、retrieval、lifecycle 与 artifacts
- `alice`：Agent Runtime 与 Koakuma

可用模型由 [configs/models.yaml](configs/models.yaml) 单独维护，Provider 密钥通过环境变量或 `configs/providers.secrets.yaml` 提供。

推荐做法是：

- 将密钥、地址、端口、环境切换放在环境变量中
- 将业务逻辑参数保留在 `config.yaml` 中

## 开发者入口

如果你希望直接在 Python 中集成系统，主入口是：

- `hivememory.system.system.HiveMemorySystem`

它提供两种主要接入方式：

- `chat()` / `chat_stream()`：主动模式，由 `ChatApplicationService` 协调 Patchouli 记忆准备、Alice Agent 执行与 Patchouli 后处理
- `ingest_event()` / `flush_ingressor()`：被动模式，适合接入 Discord Bot、微信机器人或其他外部框架

如果你只需要 HTTP 接口，可直接使用 FastAPI 服务；如果你要把 HiveMemory 嵌入已有 Agent 框架，通常从 passive ingest 模式开始会更自然。

## 项目结构

```text
HiveMemory/
├── configs/                 # 环境变量模板与主配置
├── docker/                  # Docker 一键部署（后端应用 + Qdrant）
├── docs/                    # 项目设计与规划文档
├── frontend/                # React + Vite 前端开发界面
├── scripts/                 # 启动与辅助脚本
├── src/hivememory/
│   ├── core/                # 核心数据模型
│   ├── engines/             # Gateway / Retrieval / Perception / Generation / Lifecycle
│   ├── infrastructure/      # Storage / LLM / WebSocket
│   ├── patchouli/           # Patchouli 记忆子系统与运行时
│   ├── alice/               # Alice Agent runtime 与 Koakuma MTP/工具 runtime
│   ├── system/              # 顶层 HiveMemory system、全局总线与应用服务
│   ├── prompts/             # System prompts 与 prompt 组装
│   └── server/              # FastAPI 服务入口与路由
└── tests/                   # 单元测试、集成测试、端到端测试
```

## 开发说明

- 测试默认由 `pytest` 驱动，配置见 [pyproject.toml](pyproject.toml)
- 前端提供 `npm run dev` / `npm run build` / `npm run lint`
- 后端健康检查与 readiness 已内置在 API 中
- 日志可通过 WebSocket 推送到前端

## 文档索引

- [README\_EN.md](README_EN.md) — 英文版 README
- [docs/PROJECT.md](docs/PROJECT.md) — 当前项目总览与全局文档索引
- [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md) — 文档分类、状态与维护规范
- [docs/architecture/overview.md](docs/architecture/overview.md) — 当前后端总体架构
- [docs/contracts/README.md](docs/contracts/README.md) — 跨子系统契约入口
- [docs/help/README.md](docs/help/README.md) — 安装、配置与排障入口
- [docs/ROADMAP.md](docs/ROADMAP.md) — 版本规划与后续方向

## 贡献

欢迎提交 Issue 和 Pull Request。当前仓库处于 v0.6.1 发布基线；提交行为变更时请遵守文档治理门禁，在开发工作明确收尾后再更新对应当前设计或契约文档，避免把仍在演进的设计稿或历史方案写入主干事实文档。

## 许可证

本项目采用 Apache-2.0 许可证。详见 [LICENSE](LICENSE)。
