---
title: 主配置的个人覆盖与热更新
status: archived
archived_at: 2026-11-09
implemented_by: 分支 feat/config-overrides-hot-reload（尚未提交）
superseded_by: docs/system/configuration.md
owner: system
target: 版本之外的工作（v1.0 范围内的配置热更新，候选清单 D-09）
scope: personal-config-overlay-models-template-dead-field-removal-single-retrieval-count-and-immediate-effect-for-exposed-settings
updates:
  - docs/system/configuration.md
  - docs/help/configuration.md
  - docs/frontend/management-views.md
  - docs/patchouli/retrieval.md
  - docs/gateway/analysis.md
  - docs/gateway/workflow.md
  - docs/gateway/README.md
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/mtp.md
  - docs/alice/mtp-runtime.md
  - docs/architecture/boundaries.md
code_paths:
  - src/hivememory/config/
  - src/hivememory/server/routers/config.py
  - src/hivememory/server/models/config.py
  - src/hivememory/system/
  - src/hivememory/core/protocol/gateway.py
  - src/hivememory/gateway/
  - src/hivememory/engines/retrieval/
  - src/hivememory/patchouli/services/retrieval.py
  - src/hivememory/workspace/process/runner.py
  - src/hivememory/workspace/capability/actor/memory.py
  - configs/config.yaml
  - configs/models.example.yaml
  - frontend/src/components/settings/
  - .gitignore
  - .dockerignore
related_docs:
  - docs/archive/plans/runtime-access-registration.md
  - docs/system/configuration.md
  - docs/help/configuration.md
  - docs/frontend/management-views.md
---

# 主配置的个人覆盖与热更新

> **状态**：Archived（2026-11-09，已实施，实施记录见第 10 节）。当前事实见 [System 配置](../../system/configuration.md#21-个人覆盖与保存)第 2.1 节与[配置指南](../../help/configuration.md#5-主配置与生效边界)第 5 节。以下为实施时的计划原文，链接已按归档位置改正。对应候选清单 D-09 与 [ROADMAP](../../ROADMAP.md) 第 5.5 节“配置热更新”的剩余部分，访问登记的部分已由[访问登记的运行时管理](./runtime-access-registration.md)完成。本计划的前提由 owner 于 2026-11-09 决定，见第 1.1 节。设置页的分组、表单与生效提示属于 owner 的前端 UI 工作，依据本计划定下的接口进行，不在本计划内。不绑定版本。

## 1. 背景和问题

主配置共 183 个叶子字段。2026-11-09 逐段核对了它们的使用方式：

| 使用方式 | 约多少项 | 例子 |
|:---|---:|:---|
| 每次请求都读 `system.config`，已经是热的 | 5 | `server_principal_id`、导入请求体上限 |
| 组件保留启动时的配置段，用到时读 | 约 70 | 检索阈值、Alice 会话压缩、附件限制、会话容量 |
| 启动时抄进派生对象 | 约 55 | Gateway 时限、队列策略、缓存容量 |
| 启动时据此建立资源或注册任务 | 约 40 | Qdrant 连接、嵌入与重排模型、调度任务、组件 LLM 参数 |
| 进程启动时只读一次 | 11 | 日志、默认语言 |

由此带来的问题：

- **保存几乎等于“下次启动生效”。** `POST /api/v1/config` 写回 `config.yaml` 并替换 `system.config`，但组件持有的是自己的配置段；门面的 setter 同步的 Patchouli 引用只在构造时读取。界面却提示“设置保存成功”。
- **有两处只生效一半。**
  - 运行时事件的状态接口按新配置报出容量，事件总线仍是旧的。
  - 检索结果的编译策略，MCP 按新配置，chat 与 MTP 仍按旧配置。
  - 另外，`server_principal_id` 每次请求都读，保存一个错值会立即锁死管理接口。
- **个人配置进入仓库。** `config.yaml`（设置页写回）与 `models.yaml`（模型页写回）都受版本控制。owner 定下个人配置不进仓库的规则，访问登记已按此处理。
- **有字段没有使用方，或者名不副实。**
  - 没有使用方：`system.name`、`system.debug`、`gateway.commands.enabled`。
  - 名不副实：
    - 嵌入模型名不含 bge-m3 时被换回 bge-m3；
    - 向量维度必须等于写死的 1024；
    - `scheduler.enabled` 不停调度器；
    - `alice.koakuma.enabled` 与 `mtp_prompt.enabled` 重复控制 MTP 教学。
- **检索条数有三个来源。**
  - chat 的预检索取 Gateway 检索计划里的 `top_k`（`gateway.user_query_analysis.default_top_k`）。
  - MTP 的 SEARCH 与 MCP 的检索回退到 `patchouli.retrieval.default_top_k`。
  - 名字最直观的 `patchouli.retrieval.retriever.top_k` 只在收到 0 时才用到，用户无法用它控制检索上限。
- **设置页的大部分表单与后端结构不一致**，读取接口也少返回五个段。表单对齐属于前端工作；后端要先给出完整的配置与生效信息。

### 1.1 owner 决定（2026-11-09）

1. `config.yaml` 采用本地覆盖：仓库中的 `config.yaml` 是随版本更新的默认值，个人改动写进不受版本控制的 `config.local.yaml`。
2. `models.yaml` 与访问登记相同，采用模板加本地文件。
3. 没有使用方的字段删除。
4. `patchouli.retrieval.retriever.top_k` 改为检索条数的唯一控制项。
5. 设置页分常用设置与“高级设置”栏位，暴露的项与分层见第 4.5 节。先定暴露哪些项，再按这些项判断“立即生效或重启后生效”。

## 2. 目标与非目标

### 2.1 目标

1. 个人对主配置的改动只写进 `config.local.yaml`，仓库中的 `config.yaml` 保持默认值；`models.yaml` 改为模板加本地文件。
2. 设置页暴露的项中，核实为“用到时读同一个配置对象”的字段，保存后立即生效；其余明确标为重启后生效，读取接口能列出哪些改动在等待重启。
3. 删除没有使用方与名不副实的字段，检索条数只由 `retriever.top_k` 控制。
4. 读取接口返回完整的配置树，供前端对齐表单。

### 2.2 非目标

- **设置页的界面重做**：分组、高级设置栏位、表单、生效提示与重启提示，由 owner 的前端 UI 工作完成。本计划只调整前端中绑定已删除字段的控件，使页面不报错。
- **重启或重新装配**：不提供“重启后端”入口，也不做进程内重新装配。
- **启动时抄写与建立资源的字段热生效**：这类字段一律重启后生效。
- **环境变量覆盖的字段**：环境变量的优先级高于覆盖文件，页面上修改这些字段在重启后不生效。本计划只在帮助文档中写明，不另做标记。
- **生命周期中按记忆类型区分的权重与加成、存储结构、附件解析的安全上限、缓存容量与目录**：不进设置页，仍在文件中修改。

## 3. 当前状态和缺口

| 环节 | 现在 | 本计划之后 |
|:---|:---|:---|
| 主配置来源 | 代码默认值 < `config.yaml` < `.env` < 环境变量 | 代码默认值 < `config.yaml` < `config.local.yaml` < `.env` < 环境变量 |
| 保存 | 整份配置写回 `config.yaml`，替换 `system.config` | 只把与默认值不同的字段写进 `config.local.yaml`；立即生效的字段原地写入运行中的配置对象；`system.config` 不再整体替换 |
| 读取 | 返回 `system.config`（运行中），少五个段 | 返回期望配置（下次启动将装载的配置），包含全部段，并列出等待重启的字段与可立即生效的字段 |
| 默认值 | 只有代码默认值 | 代码默认值加仓库中的 `config.yaml` |
| `models.yaml` | 受版本控制，模型页写回 | 生效文件不受版本控制，缺失时从 `models.example.yaml` 生成 |
| 检索条数 | 三个来源，`retriever.top_k` 不起作用 | 只有 `retriever.top_k` |

## 4. 目标设计

### 4.1 `config.local.yaml` 覆盖层

- **位置**：与生效的主配置文件同目录的 `config.local.yaml`，即 `HIVEMEMORY_CONFIG_PATH` 指向的文件所在目录，未设置时是仓库的 `configs/`。它加入 `.gitignore` 与 `.dockerignore`。
- **装载**：作为新的配置来源，优先级在 `config.yaml` 之上、`.env` 与环境变量之下，按层级合并（只写出的叶子覆盖下层）。覆盖文件同样拒绝未知段与写入密钥，规则与 `config.yaml` 相同。文件不存在表示没有个人改动。
- **写入**：保存时把提交的完整配置与“默认值”（代码默认值、`config.yaml`、`.env` 与环境变量，不含覆盖文件）逐叶比较，只把不同的叶子写进覆盖文件，以原子替换写入。与默认值相同的叶子不写；用户把一个字段改回默认值时，它从覆盖文件中消失。
- **恢复默认**：`GET /api/v1/config/defaults` 返回不含覆盖文件的默认值（含仓库中的 `config.yaml`，现在只含代码默认值）；前端把它提交回去，覆盖文件随之清空。
- **修正**：保存时构造配置不再去掉 `HIVEMEMORY_CONFIG_PATH`，提交中缺少的段从生效的配置文件补，而不是从仓库中的文件补。

### 4.2 `models.yaml` 模板加本地文件

- 仓库改为发布 `configs/models.example.yaml`，生效的 `configs/models.yaml` 加入 `.gitignore` 与 `.dockerignore`。`ModelRegistry` 在生效文件不存在时从模板复制一份，之后的读写只针对生效文件。
- 复制逻辑与访问登记共用一个函数，从 `config/access.py` 移到配置包中两者都能使用的位置。

### 4.3 删除的字段

| 字段 | 处理 |
|:---|:---|
| `system.name`、`system.debug`、`gateway.commands.enabled` | 删除（没有使用方） |
| `shared.embedding.default.model_name` | 删除，嵌入模型改为代码常量（现在只能是 bge-m3） |
| `patchouli.storage.vector_dimension` | 删除，建集合时取嵌入模型的维度常量；旧环境变量别名 `QDRANT__VECTOR_DIMENSION` 一并删除 |
| `scheduler.enabled` | 删除，各定时任务已有自己的开关 |
| `alice.koakuma.enabled` | 删除，MTP 教学只由 `alice.koakuma.mtp_prompt.enabled` 控制 |
| `gateway.user_query_analysis.default_top_k`、`patchouli.retrieval.default_top_k` | 删除，见第 4.4 节 |

各配置段拒绝未知键：配置文件中仍有这些键时启动失败。仓库中的 `config.yaml` 随本计划删除它们；自定义配置文件的部署按升级说明删除。

### 4.4 检索条数只由 `retriever.top_k` 控制

- Gateway 的检索计划 `RetrievalPlan` 不再携带条数，是否跳过检索只由 `mode` 表达；任务进程的预检索、MTP 的 SEARCH 与 MCP 的检索都不再传条数。
- 检索请求链路上的条数参数随之删除：
  - 操作请求 `RetrieveRequest.top_k`；
  - core 协议的 `RetrievalRequest.top_k`，即 actor API 交给 Patchouli 的检索请求；
  - Patchouli 检索路由的 `top_k`；
  - 检索引擎的 `default_top_k` 与 `retrieve(top_k)`。
- 检索器在融合与重排之后，按 `retriever.top_k` 截取最终结果；经过阈值过滤后可能更少。它是“用到时读”的字段，保存后立即生效。默认值 5 与现在相同，默认行为不变。
- 不受影响：
  - 管理页的语义检索走记忆列表接口，直接按请求的 `limit` 查中期存储，不经过检索管线；
  - 记忆生成的查重检索自己指定条数。

### 4.5 设置项的分层与生效方式

“立即”表示该字段的使用方在用到时读取同一个配置对象，保存后原地写入即生效；实施时逐项核实，核实不成立的改为重启。

**常用设置**

| 分组 | 设置项 | 配置字段 | 生效 |
|:---|:---|:---|:---|
| 模型 | Gateway 与帕秋莉使用的模型 | `gateway.llm.model_id`、`patchouli.llm.model_id` | 重启 |
| 记忆 | 检索条数上限 | `patchouli.retrieval.retriever.top_k` | 立即 |
| | 自动从对话中提取记忆 | `patchouli.generation.extractor.enabled` | 重启 |
| | 对话停多久后结算话题 | `patchouli.perception.idle_timeout_seconds` | 实施时核实 |
| 对话 | 自动压缩上下文 | `alice.compaction.enabled` | 立即 |
| 附件 | 上传大小上限 | `attachment_parser.max_raw_bytes` | 立即 |
| 系统 | 日志级别、默认语言 | `logging.level`、`i18n.default_language` | 重启 |

**高级设置**

| 分组 | 设置项 | 配置字段 | 生效 |
|:---|:---|:---|:---|
| 模型 | 采样参数 | `gateway.llm` 与 `patchouli.llm` 的 `temperature`、`top_p`、`max_tokens` | 重启 |
| | 嵌入与重排模型的驻留方式、空闲卸载时间 | `shared.embedding.default.residency`、`idle_timeout_seconds`；`patchouli.retrieval.reranker.residency`、`idle_timeout_seconds` | 重启 |
| 检索 | 相关度阈值、时间衰减 | `patchouli.retrieval.retriever.dense.score_threshold`、`enable_time_decay`、`time_decay_days` | 立即 |
| | 重排开关 | `patchouli.retrieval.reranker.enabled` | 重启 |
| | 注入上下文的记忆上限 | `memory_compiler.retrieval_context.strategy.max_memory_tokens` | 立即 |
| | 记忆的呈现方式 | `memory_compiler.retrieval_context.strategy.type` | 重启（切换类型会替换策略对象） |
| 生成 | 查重阈值 | `patchouli.generation.deduplicator.high_similarity_threshold`、`low_similarity_threshold` | 立即 |
| | 并发提取数 | `patchouli.generation.queue_max_concurrency` | 重启 |
| 话题与生命周期 | 同时驻留的话题数、遗忘速度 | `patchouli.perception.engine.max_resident_topics`、`patchouli.lifecycle.vitality_calculator.decay_lambda` | 重启 |
| | 低于多少活力时归档 | `patchouli.lifecycle.garbage_collector.low_watermark` | 立即 |
| | 自动归档的开关与周期 | `scheduler.tasks.enable_lifecycle_gc`、`lifecycle_gc_interval_hours` | 重启 |
| 对话 | 压缩触发比例、单轮最多工具调用轮数、默认上下文窗口 | `alice.compaction.trigger_ratio`、`alice.runtime.max_loop_iterations`、`default_context_window` | 立即 |
| 附件 | 注入上下文的附件上限 | `attachment_compiler.max_total_context_chars` | 立即 |
| 运行时 | Gateway 请求时限、导入等待应用的时限 | `gateway.workflow.default_request_timeout_ms`、`import_bus.apply_wait_seconds` | 重启 |
| | 会话数与每个会话的轮数上限 | `workspace.sessions.max_sessions_per_workspace`、`max_turns_per_session` | 立即 |
| 系统 | 日志文件、事件文件、运行页事件流 | `logging.file_path`、`event_file_path`、`runtime_events.enabled` | 重启 |
| | 存储地址与集合名 | `patchouli.storage.host`、`port`、`collection_name` | 只读展示 |

### 4.6 保存与生效

- **运行配置与期望配置**：
  - 运行配置是启动时装配的配置对象树，组件持有其中的配置段。启动之后不再整体替换，只有立即生效的字段被原地改写。
  - 期望配置是下次启动将装载的配置：代码默认值、`config.yaml`、覆盖文件、`.env` 与环境变量的合成结果。
- **立即生效的字段**：System 维护一份路径清单，内容为第 4.5 节标“立即”且已核实的字段。保存时，清单内发生变化的字段把新值写到运行配置中对应的对象上；清单外的字段只进入覆盖文件。
- **请求中途的变化**：原地写入发生在事件循环中，单个字段的读写是原子的。正在进行的一次检索可能前一个字段读到旧值、后一个读到新值，这一点写进当前文档。
- **两处只生效一半的问题**：运行配置不再整体替换，MCP 读取编译策略与运行时事件状态接口读取容量，读到的都是运行配置，与其他使用方一致。`server_principal_id` 同理：改动在重启后才生效，不会再立即锁死管理接口。
- **接口**：
  - `GET /api/v1/config`：返回完整的期望配置（密钥打码，覆盖全部配置段），另含 `immediate_fields`（可立即生效的字段路径）与 `pending_restart`（期望值与运行值不同、等待重启的字段路径）。
  - `POST /api/v1/config`：校验提交的完整配置，写覆盖文件，原地应用立即生效的字段，返回与 `GET` 相同的结构。
  - `GET /api/v1/config/defaults`：返回不含覆盖文件的默认值。

### 4.7 前端的最小调整

- 通用设置页删除“系统名称”“调试模式”与“版本”三个控件，前端配置类型删除对应字段。
- 内部引擎页的嵌入模型名改为不可编辑的说明。
- 其余表单、分层与生效提示留给 owner 的前端 UI 工作。

## 5. 受影响的系统、契约与数据

- **config**：新增覆盖文件来源与写入；删除第 4.3 节的字段与一个旧环境变量别名；模板复制函数移到共用位置。
- **System**：立即生效字段的清单与原地应用；`ModelRegistry` 从模板生成生效文件；门面的 `config` setter 删除（不再整体替换）。
- **server**：主配置三个接口的行为与响应结构（`ConfigResponse` 改为完整配置树加两份路径清单）。
- **core 协议**：`RetrievalPlan` 删除 `top_k`。
- **Gateway**：检索计划不再携带条数，投影随之调整。
- **workspace**：预检索不传条数；操作请求 `RetrieveRequest` 删除 `top_k`。
- **Patchouli 与 engines**：检索路由、管理检索请求与检索引擎删除条数参数，检索器按 `retriever.top_k` 截取。
- **配置文件**：
  - `config.yaml` 删除第 4.3 节的键；
  - `models.yaml` 改名为 `models.example.yaml`；
  - `.gitignore` 与 `.dockerignore` 加上 `config.local.yaml` 与 `models.yaml`。
- **前端**：第 4.7 节的最小调整。

## 6. 兼容与迁移

- **`config.yaml` 中的个人改动**：在原 checkout 中，`config.yaml` 当前没有未提交的改动，不需要迁移。曾在设置页保存过、或手工改过 `config.yaml` 的部署，升级时把与仓库版本不同的部分移到 `config.local.yaml`。升级说明给出做法。
- **`models.yaml`**：在原 checkout 中，文件留在原处，成为本机的生效文件。其他 checkout 拉取时 git 会删除它，下次启动从模板重建；在那里登记过模型的，先备份再拉取。
- **删除的字段**：配置文件中仍有这些键时启动失败，错误指出键名。仓库中的 `config.yaml` 随本计划删除；自定义配置文件与覆盖文件的部署按升级说明删除。
- **检索条数**：默认值不变。曾修改过 `gateway.user_query_analysis.default_top_k` 或 `patchouli.retrieval.default_top_k` 的部署，改为修改 `patchouli.retrieval.retriever.top_k`。
- **前端**：前端与后端在同一次发布中更新；旧前端读取已删除的字段时显示为空，不会报错。

## 7. 测试和验收标准

| 不变量 | 归属层 |
|:---|:---|
| 覆盖文件按层级合并，优先级在 `config.yaml` 之上、环境变量之下；覆盖文件中的未知段与密钥被拒绝 | 配置装载的现有单元测试 |
| 保存只把与默认值不同的叶子写进覆盖文件，改回默认值的字段从覆盖文件中消失，`config.yaml` 不被写；提交默认值后覆盖文件为空 | `tests/unit/server/routers/test_config.py` |
| 立即生效清单中的字段保存后，运行配置中对应的对象被改写，持有它的使用方下一次读取得到新值；清单外的字段不改写运行配置，出现在 `pending_restart` 中 | System 的单元测试；一条代表性用例经真实组件（检索器按新的 `top_k` 截取） |
| 立即生效清单中的每个路径都存在于配置模型中 | System 的单元测试 |
| 检索结果条数只由 `retriever.top_k` 决定，预检索、MTP SEARCH 与 MCP 检索得到同样的条数 | 检索引擎的单元测试；预检索与 MCP 各沿用现有的一条集成用例 |
| `models.yaml` 缺失时从模板生成，已有文件不被覆盖 | `ModelRegistry` 的单元测试 |

- 删除字段只删除守护它们的旧测试，不写“字段不再存在”的测试。旧配置键被拒绝，由现有的未知键拒绝测试覆盖。
- 手动验收：
  - 在设置页修改检索条数上限，下一次检索的条数立即改变；
  - 修改日志级别，接口列出它等待重启；
  - 恢复默认后 `config.local.yaml` 为空；
  - `git status` 中不出现 `config.yaml` 与 `models.yaml` 的改动。

验证命令：

```bash
pytest tests/unit/server tests/unit/system tests/unit/engines/retrieval tests/unit/gateway -q
pytest tests/unit tests/integration -v --tb=short \
  -m "not live_llm and not e2e and not slow" -n auto \
  --cov=hivememory --cov-report=term-missing --cov-fail-under=85
ruff check . && black --check .
uv run --no-sync mypy --warn-unused-ignores src scripts/typecheck_bus_routes.py
cd frontend && npm run lint && npm test && npm run build
```

## 8. 风险与待决问题

- **“立即”字段的核实**：第 4.5 节的“立即”来自逐段核对的结果。实施时逐项确认使用方持有的确实是同一个配置对象，并且在用到时读取；只要有一处抄写或缓存，就改为重启。`patchouli.perception.idle_timeout_seconds` 既有用到时读取，也有构造时抄写，实施时决定。
- **请求中途的变化**：见第 4.6 节，影响只在一次请求之内，不改变持久化的结果。
- **环境变量覆盖**：环境变量优先于覆盖文件，页面上的修改不生效，见第 2.2 节。
- **检索协议改动的范围**：`RetrievalPlan` 是 core 协议模型，Gateway 的决定投影、任务进程与相关测试都要随之调整。改动大，但都是删除。
- **待决**：无阻塞问题。路径清单的存放位置、响应字段名可在评审时调整。

## 9. 完成时必须更新的当前文档

- [System 配置](../../system/configuration.md)：
  - 配置来源与覆盖文件；
  - 主配置接口的行为；
  - 立即生效的字段与运行配置、期望配置；
  - 删除的字段与升级说明；
  - `models.yaml` 模板。
- [配置指南](../../help/configuration.md)：
  - 第 1 节文件表；
  - 第 4 节 Model Registry；
  - 第 5 节主配置与生效边界；
  - 第 6 节 Settings 页面限制。
- [管理页面](../../frontend/management-views.md)：主配置表单。
- 检索条数：[Patchouli 检索](../../patchouli/retrieval.md)、[Gateway 分析](../../gateway/analysis.md)、[Gateway 工作流](../../gateway/workflow.md)、[Gateway](../../gateway/README.md)、[子系统契约](../../contracts/subsystem-contracts.md)、[MTP 契约](../../contracts/mtp.md)、[MTP Runtime](../../alice/mtp-runtime.md)、[边界](../../architecture/boundaries.md)。
- 收尾时：
  - 本计划归档；
  - [ROADMAP](../../ROADMAP.md) 第 5.5 节“配置热更新”一行改为已完成；
  - [Plans 导航](../../plans/README.md)同步。

## 10. 实施记录

2026-11-09 在分支 `feat/config-overrides-hot-reload` 实施，第 4 节的设计按原样落地，路径清单放在 `system/running_config.py`，响应字段名沿用第 4.6 节。与计划不同或计划没有写到的地方：

- **覆盖文件带上策略的 `type`**：第 4.1 节写的是“只写不同的叶子”。实施中发现，类型不变的策略（如混合检索器）只写变化的叶子时，若下层配置缺少这一段，合并结果没有 `type`，判别联合无法确定结构而装载失败。现在类型不变时只写变化的叶子但带上 `type`，类型变化时整段写出；`pending_restart` 中切换了类型的策略整段算一个路径。
- **话题空闲结算时间改为立即生效**：第 4.5 节标为“实施时核实”。感知服务原先在构造时抄写 `idle_timeout_seconds`，改为在每次扫描时读取所持有的配置对象，与其他立即生效的字段一致。其余标“立即”的字段逐项核实，使用方都持有运行配置中的同一个对象并在用到时读取。
- **嵌入模型名改为代码常量**：推理模型服务原按配置的模型名区分实例，改为由构造参数显式给出模型名；嵌入服务固定使用 `Xenova/bge-m3`，集合维度取同一处的 1024，重排服务仍取配置中的模型名。
- **检索条数的取值范围**：`retriever.top_k`（以及单独作为检索器的 dense、sparse 的 `top_k`）至少为 1，0 会让检索静默返回空结果。单独作为检索器时，dense 与 sparse 的 `top_k`（默认 50）就是结果条数；混合检索（默认，也是随仓库发布的配置）的默认值 5 与此前相同。
- **测试隔离**：本机出现个人覆盖文件与生效的 `models.yaml` 之后，未指定主配置路径的测试会读到它们。测试的公共 fixture 让这类测试不读本机的覆盖文件，默认路径的模型注册表改到每个测试自己的临时目录、从模板生成。
- **`.env` 中的旧键**：随仓库发布的 `.env.example` 写有 `HIVEMEMORY__SYSTEM__DEBUG`，删除 `system.debug` 后，照它建立的 `.env` 会让启动失败。`.env.example` 删除了这一行，升级说明写明需要从 `.env` 中删除；原 checkout 的 `.env` 已经 owner 同意删除这一行。
- **前端**：除第 4.7 节的三处外，MTP 引擎页绑定 `koakuma.enabled` 的开关一并删除；配置接口的读取改为取响应中的 `config`。保存成功的提示仍不区分立即生效与等待重启，留给 owner 的前端 UI 工作。

验证：后端单元与集成测试（不含 `live_llm`、`e2e`、`slow`）全部通过；`ruff`、`black --check`、`mypy` 通过；前端 `npm run lint`、`npm test`、`npm run build` 通过。

尚未完成的手动验收（第 7 节）：在设置页修改检索条数上限、确认下一次检索条数改变，修改日志级别、确认接口列出它等待重启，恢复默认后覆盖文件被删除，以及 `git status` 中不出现 `config.yaml` 与 `models.yaml` 的改动。这些需要以本分支重启后端，由 owner 进行。
