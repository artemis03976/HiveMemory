# HiveMemory

[English README](README_EN.md) | [项目文档索引](docs/PROJECT.md) | [当前架构](docs/architecture/overview.md) | [安装与启动](docs/help/setup.md) | [开发路线图](docs/ROADMAP.md)

> 为 LLM Agent 设计的持久化记忆与知识共享系统
> *The Hippocampus for Artificial Intelligence*

HiveMemory 是一套面向 LLM Agent 的持久化记忆管理系统，目标是解决长上下文遗忘、跨会话知识无法复用、以及多 Agent 协作中的信息孤岛问题。系统会将对话中的高价值信息沉淀为可检索、可更新、可使用的记忆，并通过统一协议将这些记忆重新注入到后续任务中。

当前仓库已经提供可运行的 Python 后端、前端开发界面、向量存储与缓存基础设施，以及由顶层 HiveMemory System 装配 workspace、Gateway、Patchouli 与 Alice 的 v0.7.1 内容基线。v0.7.0 完成了向 Workspace 网络与任务进程架构的演进：任务请求经唯一注册入口登记为任务进程，Agent 的资源操作经 workspace 能力层逐次授权，执行者经 CPU 端口接入、可以替换。v0.7.1 以 plugin 模式接入了第一个外部 harness Claude Code：Claude Code 的对话经 Import Bus 回流进入记忆生成，Claude Code 的 agent 经 MCP 工具检索与读取记忆。

## 发布状态

- 最近已发布基线：`v0.7.1`（plugin 模式接入 Claude Code，见[收尾审计](docs/archive/plans/v0.7.1-release-closeout-audit.md)）
- 当前开发版本：`v0.7.2`（版本目标：生成质量，见 [ROADMAP](docs/ROADMAP.md)）
- 代码与包版本：`0.7.1`
- Python 要求：`>=3.12`
- 许可证：Apache-2.0

当前系统设计见 [docs/architecture/overview.md](docs/architecture/overview.md)，全局文档入口见 [docs/PROJECT.md](docs/PROJECT.md)。

## 当前版本已提供的能力

### 对话与接入模式

- **主动模式（Active mode）**：通过 `POST /api/v1/chat` 提供 SSE 流式对话。每次请求经 workspace 的唯一注册入口认证并登记为任务进程，由 `TaskProcessService` 驱动 Gateway 分析、Patchouli prepare/finalize 与经 CPU 端口的 Alice Agent 执行；`POST /api/v1/chat/stop` 停止进程，客户端断连按停止处理
- `POST /api/v1/chat` 支持在请求体中携带 `generation_options`（`model` / `temperature` / `top_p` / `max_tokens`）作为单次对话覆盖参数，不会写入全局配置文件
- **Import Bus（外部 harness 导入）**：通过 `POST /api/v1/import/turns` 导入外部 harness 一轮已结束交互的完整记录（用户消息、回复、思考、外部工具的调用与结果、结局）。请求经统一认证网关与导入授权，由 System 层的 Import Bus 编排 Gateway 话题落位、共用封口与 Patchouli 感知提交，交互应用完成后才确认；它只向记忆网络输入，不检索记忆，也不返回记忆上下文
- **MCP 记忆访问**：`POST /mcp` 是外部 harness 使用的 MCP 端点（streamable HTTP，无状态，只接受本机地址），提供只读工具 `memory_search` 与 `memory_read`。每次工具调用经统一认证网关取得一份请求级访问 context，按 Workspace 访问登记的白名单逐次授权，不建任务进程
- **Claude Code 插件**：`integrations/claude-code/`（插件 `hivememory-connector`，本地 marketplace `hivememory-local`，只用 Python 标准库）同时提供对话回流与记忆访问。连接器在轮次结束与会话开始、结束时把 Claude Code 会话记录中已结束的轮次导入 HiveMemory，网络提交在脱离 Claude Code 的后台进程中完成；插件声明的 MCP server 让 Claude Code 的 agent 检索与读取记忆。两者都以 Claude Code 自己的接入登记 `harness:claude-code` 认证，HiveMemory 一侧的 `HIVEMEMORY_CLAUDE_CODE_TOKEN` 与 Claude Code 一侧的 `HIVEMEMORY_TOKEN` 需配置同一份凭据；`configs/workspace_actors.yaml` 为 Agent `claude_code` 只登记了检索与读取。安装与配置见[安装与启动](docs/help/setup.md)
- **会话**：支持新建、恢复、重命名与删除；Alice 按模型预算自动精简旧工具结果并生成接力摘要，原始对话完整保留，可查看摘要与压缩边界；有输出的取消与失败轮次保留结局。会话与检查点仅在后端进程内保存，重启后消失；Topic 独立作为记忆材料。
- **系统指令**：Gateway 只解析命令，识别为命令的消息以命令终态结束、不进入普通对话；当前版本不执行内置命令。

### Workspace 与访问

- 两类访问登记（`configs/system_principals.yaml` 与 `configs/workspace_actors.yaml`）经两阶段认证，决定谁能以哪个 Agent 进入哪个 Workspace；Agent 的操作权限只由访问登记的 actor 操作白名单表达，Agent Profile 不承载权限；
- Agent 的资源操作（MTP 的 SEARCH、READ、RUN、WRITE、UPDATE、CALL）经 workspace 能力层逐次授权；CALL 派生的子 Agent 作为执行线程，以被调用方身份运行；
- Memory、Topic 和附件携带明确的 Workspace 归属，资源 owner 先校验归属，再按资源 policy 判断可见性；
- 写入意图登记与引用读取缓存由 workspace 持有，正式记忆变更会失效原子与 Profile 解析缓存；完整边界见 [Workspace](docs/workspace/README.md)。

### Chat 附件

- 通过 `POST /api/v1/workspace/assets` 上传 TXT、Markdown、DOCX，确定性解析后 READY 表示可显式选择进入 Chat；
- AttachmentCompiler 按预算编译上下文，成功 Interaction 记录使用关系，Memory CREATE/UPDATE 按需生成来源 Artifact；
- WorkspaceAsset 仅承诺当前进程内可用；完整链路见 [Chat 附件链路](docs/system/attachments.md)。

### 记忆、话题与 Agent 管理

- 记忆的创建、语义搜索与列表、读取、编辑、反馈与删除：`/api/v1/memories`
- 记忆生成任务的查询与取消：`/api/v1/memory-tasks`
- 活跃话题列表、手动结算与驱逐：`/api/v1/topics`
- Agent Profile 的创建与列表：`/api/v1/agents`

### 配置与可观测性

- 运行时配置的读取、更新与默认值：`/api/v1/config`
- 模型与 Provider 注册表：`/api/v1/models`、`/api/v1/providers`
- 运行时事件状态与 SSE 流：`/api/v1/runtime-events`
- WebSocket 日志流：`WS /api/v1/ws/logs`
- 健康检查与就绪检查：`GET /health`、`GET /health/ready`

### 核心能力

- 分层架构：顶层 `HiveMemorySystem` 装配同层的 workspace、Gateway、Patchouli 与 Alice
- Gateway 系统入口：命令解析、话题路由、查询分析、取消/超时和保守降级
- 进程内运行时总线：AsyncSystemBus / GlobalSystemBus / 子系统私有 bus
- MTP（Memory Tool Protocol）协议，支持 `SEARCH / READ / RUN / WRITE / UPDATE / CALL`
- 基于 Qdrant 的持久化记忆存储
- Dense + Sparse 的混合检索路径
- 前端开发界面（Vite + React）

## 架构概览

HiveMemory 当前实现按依赖分层组织。顶层 System 负责组合、门面与生命周期，共享运行时机制位于 `components`，认证、能力层与资产设施位于 `workspace`；Gateway 负责入口决策，Patchouli 负责记忆域能力，Alice 负责 Agent 执行与 MTP/工具运行时。

### 主要运行时组成

- **HiveMemorySystem**：组合根与门面，装配共享运行时、workspace 设施、应用服务、Gateway、Patchouli 与 Alice
- **Workspace 设施**：认证网关与访问登记、能力层（root API 与 actor API）、会话资源、读取视图与写入意图登记、WorkspaceAsset
- **TaskProcessService**：任务请求的唯一注册入口（`workspace.process`），认证后登记任务进程并以 `process_id` 统一进程标识，按 `Gateway -> prepare -> Actor 执行（经 CPU 端口，当前为 Alice） -> finalize` 四阶段编排；进程持有自己的 task 与输出通道，Actor 阶段内的执行线程由进程的线程表登记
- **GatewaySystem / GatewayRuntime**：入口决策子系统，负责系统指令解析、话题路由、查询分析与稳定决策投影
- **PatchouliSystem / PatchouliRuntime**：记忆子系统宿主与运行时，管理 retrieval、perception、generation、lifecycle 与 storage 能力
- **Retrieval Familiar**：Hot Path 检索服务，负责混合检索、重排序、上下文渲染
- **Librarian Core**：Cold Path 记忆服务，负责话题感知、记忆提取、生命周期维护
- **AliceSystem / AgentRuntime**：Agent 执行子系统，通过 CPU 端口交付运行结果，内核工厂装配 Agent loop 与 Koakuma 工具 runtime
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
- `RUN`：执行系统工具或记忆中的代码片段
- `WRITE`：主动提交新的记忆写入意图
- `UPDATE`：主动提交已有记忆的更新意图
- `CALL`：挂起当前 frame 并委派给子 Agent

协议格式为：

```text
⟪ VERB | TARGET | key="value" ⟫
```

完整协议见 [docs/contracts/mtp.md](docs/contracts/mtp.md)，总体设计入口见 [docs/PROJECT.md](docs/PROJECT.md)。

## API 概览

| 接口路径 | 方法 | 用途 |
|:---|:---|:---|
| `/health` | GET | 存活检查 |
| `/health/ready` | GET | 模型预热完成后的就绪检查 |
| `/api/v1/chat` | POST | SSE 流式主动对话 |
| `/api/v1/chat/stop` | POST | 停止运行中的任务进程 |
| `/api/v1/sessions` | GET / POST | 列出 / 新建会话 |
| `/api/v1/sessions/{session_id}` | GET / PATCH / DELETE | 读取全量历史 / 重命名 / 删除会话 |
| `/api/v1/import/turns` | POST | 导入外部 harness 一轮已结束的交互（Import Bus） |
| `/mcp` | POST | 外部 harness 经 MCP 工具检索与读取记忆 |
| `/api/v1/memories` | GET / POST | 语义搜索或列出记忆 / 创建记忆 |
| `/api/v1/memories/{memory_id}` | GET / PATCH / DELETE | 读取 / 编辑 / 删除单条记忆 |
| `/api/v1/memories/{memory_id}/feedback` | POST | 提交记忆反馈 |
| `/api/v1/memory-tasks` | GET | 列出记忆生成任务 |
| `/api/v1/memory-tasks/{task_id}` | GET | 查询单个任务 |
| `/api/v1/memory-tasks/{task_id}/cancel` | POST | 取消任务 |
| `/api/v1/topics` | GET | 获取活跃话题 |
| `/api/v1/topics/{topic_id}/settle` | POST | 手动结算话题 |
| `/api/v1/topics/{topic_id}` | DELETE | 从活跃池驱逐话题 |
| `/api/v1/agents` | GET / POST | 列出 / 创建 Agent Profile |
| `/api/v1/workspace/assets` | POST | 上传附件 |
| `/api/v1/models` | GET / POST | 列出 / 注册模型 |
| `/api/v1/models/{model_id}` | GET / PUT / DELETE | 读取 / 更新 / 删除模型 |
| `/api/v1/providers` | GET | 列出 Provider |
| `/api/v1/providers/{provider_name}` | PUT / DELETE | 更新 / 删除 Provider |
| `/api/v1/config` | GET / POST | 读取 / 更新运行时配置 |
| `/api/v1/config/defaults` | GET | 获取默认配置 |
| `/api/v1/runtime-events/status` | GET | 运行时事件状态 |
| `/api/v1/runtime-events/stream` | GET | 运行时事件 SSE 流 |
| `/api/v1/ws/logs` | WS | 获取实时日志流 |

请求身份经 `x-user-id` / `x-workspace-id` 请求头选择，Agent action 另带 `agent_id`；交接规则见[路由与事件契约](docs/contracts/routes-and-events.md#6-http-入口的身份交接与管理读取边界)。`/api/v1/import/turns` 与 `/mcp` 是外部 harness 的入口，另以 `X-Principal-ID` 声明接入来源、以 `Authorization: Bearer` 携带接入凭据，见 [Import Bus](docs/system/import-bus.md) 与 [MCP 记忆访问](docs/contracts/mcp.md)。

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
- `gateway`、`import_bus` 与 `memory_compiler`
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

- `process_service.register_process()` 与 `run_process()`：主动模式。先由注册入口完成认证并登记任务进程，取得进程句柄；再以 `run_process()` 订阅进程的流式输出，由 `TaskProcessService` 协调 Gateway 分析、Patchouli 记忆准备、Alice Agent 执行与 Patchouli 后处理
- `import_bus_service.import_turn()`：Import Bus，导入外部 harness 一轮已结束交互的完整记录；调用方需先经统一认证网关取得请求级访问 context（HTTP 入口 `POST /api/v1/import/turns` 已完成这一步）

如果你只需要 HTTP 接口，可直接使用 FastAPI 服务；如果你要让已有 Agent harness 的交互沉淀为 HiveMemory 的记忆，并让它读取这些记忆，通常从 `POST /api/v1/import/turns` 与 `/mcp` 开始更自然，Claude Code 插件（`integrations/claude-code/`）是现成的接入实现。

## 项目结构

```text
HiveMemory/
├── configs/                 # 环境变量模板与主配置
├── docker/                  # Docker 一键部署（后端应用 + Qdrant）
├── docs/                    # 项目设计与规划文档
├── frontend/                # React + Vite 前端开发界面
├── integrations/            # 外部 harness 连接器（Claude Code 插件与本地 marketplace）
├── scripts/                 # 启动与辅助脚本
├── src/hivememory/
│   ├── core/                # 核心数据模型、契约常量、访问值类型与端口协议
│   ├── config/              # 配置段模型与根配置加载
│   ├── components/          # 进程内运行时机制：总线、调度器、work queue、运行时事件
│   ├── engines/             # Gateway / Retrieval / Perception / Generation / Lifecycle 算法
│   ├── infrastructure/      # Storage / LLM / Embedding / 附件解析等外部技术适配
│   ├── workspace/           # 认证入口、能力层、读取视图、WorkspaceAsset 设施与任务进程（process）
│   ├── gateway/             # Gateway 入口决策子系统
│   ├── patchouli/           # Patchouli 记忆子系统与运行时
│   ├── alice/               # Alice Agent 执行子系统
│   ├── agent_runtime/       # 单 Agent 执行层与 Koakuma MTP/工具 runtime
│   ├── system/              # 组合根、门面、注册表与 Import Bus
│   ├── prompts/             # System prompts 与 prompt 组装
│   ├── server/              # FastAPI 服务入口与路由
│   └── mcp_server/          # 外部 harness 经 MCP 访问记忆的入口（由 server 接入 /mcp）
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

欢迎提交 Issue 和 Pull Request。v0.7.1 已发布，当前主干正在进行 v0.7.2 开发；提交行为变更时请遵守文档治理门禁，在开发工作明确收尾后再更新对应当前设计或契约文档，避免把仍在演进的设计稿或历史方案写入主干事实文档。

## 许可证

本项目采用 Apache-2.0 许可证。详见 [LICENSE](LICENSE)。
