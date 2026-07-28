---
title: System Architecture v4.0
status: archived
owner: project
scope: legacy-architecture
archived_at: 2026-07-28
superseded_by: docs/architecture/overview.md
---

> 本文记录第四次架构演进的阶段性收敛结果，已经被当前代码核验后的[系统架构概览](../../architecture/overview.md)和[系统边界](../../architecture/boundaries.md)替代。

# HiveMemory 第四次架构演进最终总纲

**文档状态**: Final (已收敛)\
**所属演进**: 第四次架构演进 / v4\
**文档定位**: 第四次架构演进的当前准则与最终结构说明。

***

## 1. 背景与结论

前三次架构演进中，HiveMemory 的主结构长期围绕 Patchouli 记忆域展开。随着 MTP、Agent 生成循环、运行时总线、维护调度与多 Agent 方向逐步出现，`patchouli/` 一度承担了超出记忆子系统的职责：

- 主动 chat 入口与完整生成循环
- MTP / tool runtime
- Agent frame 与子 Agent 调度
- 系统级运行时基础设施
- 记忆 prepare/finalize 与 interaction 后处理

第四次架构演进的核心结论是：

> HiveMemory 需要从“Patchouli 作为事实宿主”收敛为 `System -> Service -> Runtime` 分层，并将 Patchouli 与 Alice 定位为同级子系统。

最终结构如下：

```text
HiveMemorySystem
  -> ChatApplicationService
  -> PassiveIngressService
  -> MemoryApplicationService
  -> AgentApplicationService
  -> TopicApplicationService
  -> SystemReadinessService
  -> GlobalSystemBus
  -> GlobalMaintenanceScheduler
  -> PatchouliSystem
      -> PatchouliService
      -> Patchouli application services
      -> PatchouliRuntime
  -> AliceSystem
      -> AliceService
      -> AliceRuntime
          -> AgentRuntime
          -> KoakumaRuntime
```

其中：

- `System` 是宿主层，负责装配、生命周期、全局路由与全局调度。
- `Service` 是用例门面，负责对外能力语义与入口编排。
- `Runtime` 是子系统内部运行环境，负责组件图、私有总线、运行状态与内部能力边界。

***

## 2. 顶层 System

### 2.1 HiveMemorySystem

`HiveMemorySystem` 是项目级宿主，不再把长编排逻辑塞进自身。它负责：

- 创建并持有 `GlobalSystemBus`
- 创建并持有 `GlobalMaintenanceScheduler`
- 装配 `ChatApplicationService`、`PassiveIngressService`、`MemoryApplicationService`、`AgentApplicationService`、`TopicApplicationService` 与 `SystemReadinessService`
- 装配 `PatchouliSystem` 与 `AliceSystem`
- 管理 `start()` / `stop()` / `health()` 等生命周期入口
- 将 server 层依赖连接到稳定的应用服务入口

`HiveMemorySystem` 不负责：

- 记忆检索细节
- Agent 执行循环
- MTP 工具执行
- prompt 组装
- interaction 后处理
- chat、ingest、memory、agent、topic、readiness 等业务动作的直接门面

### 2.2 Application Services

`system/application/` 是系统级入口编排层。

`ChatApplicationService` 承担主动 chat 主链路：

```text
Patchouli.prepare_agent_run(...)
  -> Alice.run_agent(...) / run_agent_stream(...)
  -> Patchouli.finalize_agent_run(...)
```

`PassiveIngressService` 承担被动消息摄入链路，将外部事件归一化后交给 Patchouli 记忆域处理。

`MemoryApplicationService`、`AgentApplicationService` 与 `TopicApplicationService` 分别承接 HTTP 层的记忆、Agent Profile 与话题管理用例。它们负责 request/use case 层的数据转换与错误边界，不把 storage、runtime、librarian、buffer manager 等子系统内部对象暴露给 router。

`SystemReadinessService` 承接模型 warmup 与 readiness 检查。FastAPI lifespan 与 `/health/ready` 只调用该服务，不直接触碰 Patchouli runtime。

这些服务只通过 `GlobalSystemBus` 与全局公开路由访问子系统能力，不穿透到子系统私有 runtime。`HiveMemorySystem` 只暴露这些服务属性，作为组合根与生命周期宿主存在，不再提供 `chat()`、`chat_stream()`、`ingest_event()`、`manual_archive_topic()`、`warmup_models()` 等动作级 API。

### 2.3 Server Router 边界

server router 是 HTTP 协议适配层，职责限定为：

- 解析 FastAPI request 与 dependency
- 调用对应 application service
- 转换 response model
- 抛出 HTTP 层错误

router 不应直接访问：

- `system.patchouli`
- `system.alice`
- `system.patchouli.runtime`
- `system.patchouli.storage`
- `system.patchouli.librarian_core`
- 任意子系统 `_private` 成员

`server/deps.py` 提供面向 router 的窄依赖入口，例如 `get_chat_service()`、`get_ingress_service()`、`get_memory_service()`、`get_agent_service()` 与 `get_topic_service()`。这使 HTTP 层依赖稳定在系统级应用服务，而不是依赖完整 `HiveMemorySystem` 或子系统内部对象。

***

## 3. Patchouli 子系统

### 3.1 定位

Patchouli 是记忆子系统，不再是整个项目的事实宿主。它负责：

- The Eye / Gateway
- Retrieval Familiar
- Librarian Core
- 话题准备与上下文检索
- 记忆生成、感知与生命周期
- interaction 后处理
- 记忆能力公开路由

Patchouli 不负责：

- 主动 chat 顶层编排
- Agent 生成循环
- MTP 工具运行时
- 子 Agent frame 调度
- 系统级 generation cancel 注册表

### 3.2 PatchouliSystem

`PatchouliSystem` 是记忆子系统宿主，负责：

- 持有 `TheEye`
- 持有 `PatchouliRuntime`
- 持有 `PatchouliService`
- 持有 Patchouli application services
- 将公开记忆能力注册到 `GlobalSystemBus`
- 注册 Patchouli 维护任务
- 代理子系统生命周期

### 3.3 PatchouliRuntime

`PatchouliRuntime` 是记忆域运行时，负责：

- 初始化 storage、embedding、reranker、librarian LLM
- 构建 retrieval / perception / generation / lifecycle engines
- 持有 `RetrievalFamiliar` 与 `LibrarianCore`
- 持有 `PatchouliBus`
- 挂载 / 卸载 local routes
- 执行 warmup / health / shutdown drain

`PatchouliRuntime` 已取代旧 `PatchouliKernel` 的长期语义。旧 “kernel” 术语只应作为历史文档语境出现。

### 3.4 PatchouliService

`PatchouliService` 是记忆域用例门面，核心能力包括：

| 能力 | 说明 |
| :--- | :--- |
| `prepare_agent_run()` | 准备 AgentRunContext、话题上下文、预检索记忆、stream prelude |
| `finalize_agent_run()` | 从 Alice 结果生成 interaction payload，提交感知与记忆后处理 |
| `cleanup_prepared_agent_run()` | 流式异常时清理本轮准备状态 |
| `retrieve_for_gaze()` | 根据 EyeGazeResult 执行检索并渲染上下文 |
| `manual_archive_topic()` | 手动触发话题归档 |

PatchouliService 不再组装完整主 Agent prompt，也不再驱动 Agent loop。

### 3.5 Patchouli Application Services

`patchouli/application/` 是 Patchouli 子系统对外公开 API 的用例层。它位于 `PatchouliSystem` 与 `PatchouliRuntime` 之间，避免把所有公开路由方法继续堆叠到 `PatchouliService` 或 `PatchouliSystem` 上。

当前主要服务包括：

| 服务 | 职责 |
| :--- | :--- |
| `MemoryManagementService` | 记忆 CRUD、搜索、反馈、活力刷新等公开管理能力 |
| `AgentProfileManagementService` | Agent Profile 创建、查询与列表能力 |
| `TopicManagementService` | 活跃话题查询、归档、驱逐等话题管理能力 |
| `ModelReadinessService` | 模型 warmup 与 readiness 状态查询 |

这些服务可以调用 Patchouli 内部领域对象，但通过 `PatchouliSystem` 注册为公开路由后，对系统级 application service 呈现为 `GlobalSystemBus` 契约。这样可以同时保持子系统内部聚合清晰，并避免 router 或顶层 system 直接触碰 Patchouli runtime 细节。

***

## 4. Alice 子系统

### 4.1 定位

Alice 是 Agent runtime 子系统，负责 Agent 执行与工具运行时，不负责记忆域 prepare/finalize，也不升级为顶层 chat 门面。

### 4.2 AliceSystem

`AliceSystem` 是 Alice 子系统宿主，负责：

- 持有 `AliceRuntime`
- 持有 `AliceService`
- 将 Alice 公开能力注册到 `GlobalSystemBus`
- 管理 Alice 生命周期

### 4.3 AliceRuntime

`AliceRuntime` 是 Alice 的 runtime 聚合根，负责显式持有：

- `AliceBus`
- `AgentRuntime`
- `KoakumaRuntime`
- `AgentPromptAssembler`
- `MTPExecutor` adapter

AliceRuntime 的核心职责是组装 Agent 执行环境，而不是直接实现执行循环细节。

### 4.4 AgentRuntime

`AgentRuntime` 是 Agent 执行 runtime，负责：

- 主 Agent 执行
- 流式 / 非流式运行
- ExecutionFrame 管理
- FrameScheduler 调度
- WorkerAgent 调用
- CALL 派生子 Agent
- cancel_event 消费

它通过注入依赖访问：

- `local_bus`
- `prompt_assembler`
- `mtp_executor`
- `AgentRuntimeConfig`
- `FrameScheduler`
- `AgentProfileResolver`

它不依赖 `AliceRuntime`，也不直接持有 `KoakumaRuntime`。

### 4.5 KoakumaRuntime

`KoakumaRuntime` 是 MTP / Tool runtime，负责：

- MTP 指令解析
- MTP 权限检查
- syscall/tool 执行
- MTPResponse 格式化
- READ / SEARCH / RUN / WRITE / UPDATE / CALL 等协议语义

Koakuma 不负责：

- Agent loop
- frame 调度
- trace 长期缓存
- interaction state 累计
- AgentRuntime 回调

工具执行所需的身份、profile、depth 等权限信息由 `MTPExecutionContext` 随调用传入，不再通过旧式 set 方法写入 Koakuma 内部状态。

***

## 5. 主动 Chat 流程

主动 chat 已收敛为顶层应用服务编排：

```text
User / API
  -> server router
  -> ChatApplicationService.chat_stream(...)
      -> PATCHOULI_PREPARE_AGENT_RUN
          -> PatchouliService.prepare_agent_run(...)
          -> AgentRunContext
      -> ALICE_RUN_AGENT_STREAM
          -> AliceService.run_agent_stream(...)
          -> AliceRuntime
          -> AgentRuntime
      -> PATCHOULI_FINALIZE_AGENT_RUN
          -> PatchouliService.finalize_agent_run(...)
          -> InteractionPayload
```

非流式 `ChatApplicationService.chat()` 使用同一条三段式骨架，只是不产出 SSE prelude 与 token stream。

关键边界：

- Patchouli 准备记忆上下文，不执行 Agent loop。
- Alice 执行 Agent loop，不提交 interaction。
- Patchouli finalize 做后处理，不依赖 Koakuma trace 缓存。
- ChatApplicationService 持有 generation cancel 注册表。

***

## 6. Prompt 组装

prompt 组装已从 Patchouli 与 FrameScheduler 中收敛到：

```text
hivememory.prompts.assembler.AgentPromptAssembler
```

统一入口：

- `build_main_agent_messages(context: AgentRunContext)`
- `build_sub_agent_messages(profile, task, shared_context, depth)`

职责划分：

- Patchouli 准备 `AgentRunContext`，不组装完整 messages。
- AliceRuntime 在进入 AgentRuntime 前调用 assembler 组装主 Agent messages。
- FrameScheduler 只负责 frame 管理，子 Agent prompt 由 assembler 组装。
- 子 Agent 禁用 `CALL` 通过 MTP allowed verbs 白名单处理，不再通过文本裁剪 prompt。

***

## 7. Trace 与 Interaction 后处理

v4 最终链路不再依赖 Koakuma 内部 trace 缓存。

Agent loop 输出：

```text
TurnEvent -> ActionReducer -> AgentAction -> TraceReducer -> TraceItem
```

当前责任分布：

- `AgentRuntime` / `AgentLoopExecutor` 产出结构化 `turn_events`
- `PatchouliService.finalize_agent_run()` 聚合 `actions` 与 `mtp_traces`
- `SemanticFlowPerceptionLayer` 使用 payload 中已准备好的 `mtp_traces`
- `GenerationTranscriptBuilder` 可按需要跳过不适合摘要渲染的 trace 类型，但 trace 数据本身保持完整

因此：

- `ChatResult` 不再需要暴露 `mtp_commands_executed`
- `ChatResult` 不再作为 Koakuma trace 缓存输出载体
- WRITE / UPDATE / CALL 等事件可进入结构化事件链路，再由后处理阶段决定如何使用

***

## 8. 总线边界

v4 采用全局总线 + 子系统 local bus 的分层结构：

| 总线 | 归属 | 用途 |
| :--- | :--- | :--- |
| `GlobalSystemBus` | `HiveMemorySystem` | 跨子系统公开能力调用 |
| `PatchouliBus` | `PatchouliRuntime` | Patchouli 内部能力路由 |
| `AliceBus` | `AliceRuntime` | Alice 内部能力路由 |

约束：

- 顶层 application service 只使用 `GlobalSystemBus`。
- 子系统内部优先使用自己的 local bus。
- AgentRuntime 内部不直接访问 `GlobalSystemBus`。
- Koakuma 通过 Alice local bus 与公开 route 访问记忆能力。
- 公开 route 使用全局契约常量，不依赖子系统私有 `LocalRoutes` 命名。
- `GlobalRoutes` 由各子系统公开路由模块汇总生成，避免同时维护两份字符串常量。
- 系统级 application service 通过 `GlobalRoutes` 发起请求，不直接依赖子系统 service 或 runtime 实例。

当前公开路由边界包括：

- Patchouli prepare/finalize/cleanup 等 Agent run 后处理能力
- 记忆管理、Agent Profile 管理、话题管理能力
- 模型 warmup 与 readiness 查询能力
- Alice Agent run、stream run 与 cancel 能力

***

## 9. 配置归属

v4 后配置边界按 runtime 职责拆分：

- `AgentRuntimeConfig`：Agent loop、frame、递归深度、最大迭代等执行参数
- `KoakumaConfig`：MTP/tool runtime、协议能力、syscall 配置
- Patchouli 相关配置：retrieval、perception、generation、lifecycle、storage
- 顶层配置：server、scheduler、logging、子系统装配等

原则：

- AgentRuntime 不再读取 `config.koakuma` 作为执行配置。
- Koakuma 不再承担 Agent loop 运行时参数。
- prompt assembler 可接收 Koakuma prompt 子配置，但不访问 runtime、bus、storage。

***

## 10. 目录结构

v4 后主要目录语义如下：

```text
src/hivememory/
  system/
    system.py
    application/
      chat_service.py
      passive_ingress_service.py
      memory_service.py
      agent_service.py
      topic_service.py
      readiness_service.py
    runtime/
    contracts/
  server/
    deps.py
    routers/
  patchouli/
    system.py
    service.py
    application/
    runtime/
    services/
    contracts/
  alice/
    system.py
    service.py
    runtime/
      core.py
      koakuma.py
      models.py
      agent/
      syscalls/
  prompts/
    assembler.py
    system_prompt.py
    mtp.py
```

历史上的 `patchouli/kernel`、`AgentRuntimeHost`、Alice runtime 顶层重导出兼容文件均不再作为长期结构保留。

***

## 11. 已完成收敛项

当前 v4 关键设计已完成：

- 顶层 `HiveMemorySystem` 成立
- `HiveMemorySystem` 收敛为组合根与生命周期宿主，不再提供动作级业务 API
- `ChatApplicationService` 接管主动 chat 编排
- `PassiveIngressService` 成立
- `MemoryApplicationService`、`AgentApplicationService`、`TopicApplicationService` 接管对应 HTTP use case
- `SystemReadinessService` 接管模型 warmup 与 readiness 检查
- server router 改为依赖 application service，不再穿透访问 Patchouli / Alice 内部对象
- `PatchouliRuntime` 取代旧 kernel 语义
- `patchouli/application/` 承接记忆管理、Agent Profile 管理、话题管理与模型 readiness 等公开 API
- Patchouli local bus / local routes / shutdown drain 下沉到 runtime
- Alice 成为独立子系统
- `AliceRuntime` 显式持有 `AgentRuntime` 与 `KoakumaRuntime`
- `AgentRuntime` 不再依赖 `AliceRuntime`
- `AgentLoopExecutor` 通过依赖注入访问 frame scheduler、MTP executor、profile resolver 与 local bus
- Koakuma 旧 set 方法、trace 缓存与 interaction state 累计职责已清理
- prompt 组装集中到 `AgentPromptAssembler`
- trace 从结构化 `turn_events` 后处理生成
- `ChatResult` 去除 MTP trace/command 冗余字段
- FrameScheduler 脱离 runtime 依赖
- readiness / warmup 通过公开路由与 `SystemReadinessService` 完成，不再由 server 直接访问 Patchouli runtime

***

## 12. 后续可选收尾

v4 主体已完成，第四次架构演进的过渡性阶段文档已全部清理。剩余代码工作属于清理与文档同步：

- 继续清理旧文档中的 `PatchouliKernel`、`AgentRuntimeHost` 等历史术语
- 视需要将 agent profile 加载进一步抽为 repository / loader
- 为关键边界增加 import-scan 或 contract tests
- 继续收敛测试 fixture 与调试脚本中的内部访问路径，避免重新引入 router 到子系统内部对象的穿透依赖

这些工作不会改变 v4 的主架构结论。
