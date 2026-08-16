---
title: docs/mod 逐篇迁移审计
status: archived
owner: project
scope: documentation-migration-audit-docs-mod
archived_at: 2026-07-29
superseded_by:
  - docs/archive/plans/documentation-migration-inventory.md
  - docs/DOCUMENTATION.md
source_inventory: docs/archive/plans/documentation-migration-inventory.md
---

# `docs/mod` 逐篇迁移审计

本记录是原 `docs/mod/` 十八篇混合设计稿的逐篇审计与物理迁移门禁，不是新的后端设计入口。它回答五个问题：旧稿中的当前事实由谁承接、哪些动机和取舍仍有解释力、哪些历史主张必须拒绝、未完成内容应当进入 Plan/Todo/Idea 中的哪一类，以及最终文件应移动到哪里。

## 1. 审计口径

本批按照 [文档治理规范](../../DOCUMENTATION.md)逐篇核对当前代码、测试和现行文档：

1. **完成状态不能从旧稿标题推断**。只有代码、测试与当前文档共同支持的内容才可归为已完成；
2. **当前事实与设计理由同时承接**。现行文档不仅保留对象与调用顺序，也说明为什么拆分所有权、为什么采用 best-effort、为什么拒绝 God Object；
3. **未来工作按范围分类**。跨模块、有阶段和验收条件的工作进入 Plans；范围小且可独立关闭的缺陷才进入 Todo；没有形成实现承诺的开放方向不为迁移方便伪装成 Todo；
4. **历史稿允许保留未实现方案，但必须失去当前入口地位**。归档顶部统一提供 `superseded_by`，正文中的 Phase、类名和路径不再被当成事实；
5. **物理移动在承接完成后执行**。移动后同时修复当前文档、Roadmap、源码注释、历史稿互链和 Archive 索引。

本次代码核验覆盖 `agent_runtime/`、`alice/`、`patchouli/`、`engines/{memory_compiler,retrieval,generation,artifacts}/`、`gateway/`、`system/runtime/`、`system/services/passive/` 及其定向测试。

## 2. Alice 与 Agent Runtime

### 2.1 `AgentRuntimeBoundaryDesign.md`

**分类与动作：** 已完成的边界裁定，合并后归档至 [`agent-runtime-boundary.md`](./implementation/agent-runtime-boundary.md)。

**当前承接：** [Alice 总览](../../alice/README.md)、[Agent Runtime](../../alice/agent-runtime.md)和[多 Agent 编排](../../alice/orchestration.md)已经说明 Alice 是编排控制面，top-level `agent_runtime/` 是被 Alice 消费的共享单 Agent 执行层；执行层不实现 `SubsystemProtocol`，也不反向依赖 Alice。

**保留理念：** CPU/OS 隐喻仍用于解释“执行引擎与操作系统式编排不是一层”；`ExecutionFrame` 是可恢复执行 PCB；逻辑所有权应先稳定，再决定物理目录。这些理念能帮助读者识别把 CALL、frame 调度或 IPC 重新塞回 loop 的边界倒退。

**拒绝继承：** 旧稿中的三 Runtime 文件布局、旧类路径和“先不移动目录”的阶段性状态已经过时；当前物理目录已经完成分离。旧 `ChatResult`、Focus 字段和 Alice 内部装配细节也不是当前公共契约。

### 2.2 `AgentLoopDecouplingDesign.md`

**分类与动作：** 已完成实施稿，归档至 [`agent-loop-decoupling.md`](./implementation/agent-loop-decoupling.md)。

**当前承接：** Agent Runtime 文档描述单帧 loop、`FrameExecutionResult` 与 CALL trap；Orchestration 文档描述 profile 解析、子帧 fork、共享上下文、alias 收割、IPC 回填和主帧恢复。

**保留理念：** CALL 是从执行循环向编排层的 trap，不是 loop 内部递归；同一 `ExecutionFrame` 在恢复后继续推进，累计进度不能藏在不可重入局部变量中；流式子帧事件的命名空间与合并仍必须服从父 run 的编排边界。

**拒绝继承：** 源码行号、旧 `alice/runtime/agent/loop_executor.py` 路径、`ChatResult` 组装方案和旧测试位置均已失效。当前 Orchestrator 的已知子帧终态检查缺口以现行文档为准，不能用旧稿的目标状态掩盖。

### 2.3 `PendingAtomLifecycleDesign.md`

**分类与动作：** 已完成的生命周期设计，归档至 [`pending-atom-lifecycle.md`](./implementation/pending-atom-lifecycle.md)。

**当前承接：** [Alice PendingAtom](../../alice/pending-atom.md)已经保留 PendingAtom 是 store buffer、ACK 不等于持久化、三级 alias 解析、settlement 窗口、`SETTLED/FAILED/CANCELLED -> EXPIRED` 和以全局 run 周期推进回收的理由。

**保留理念：** 临时句柄必须允许 Agent 在物化完成前继续寻址，却不能把未完成意图伪装成正式 MemoryAtom；终态对象保留一个观察窗口后才释放，避免 redirect/discard 消息尚未被读取就消失。

**拒绝继承：** “CANCELLED 没有生产路径”已经过时。当前取消的 Agent run 会取消仍在飞行的 PendingAtom，Patchouli memory task 也会发布 cancelled settlement。仍真实存在的是 resolver 把 CANCELLED 退化为 `not_found`、取消原因未保存、没有 durable ledger/TTL/replay，以及旧 `PENDING/MATERIALIZING` 没有自动超时回收。

## 3. Patchouli、MemoryCompiler 与 v0.5.x

### 3.1 `MemoryCompilerIRDesign.md`

**分类与动作：** 已完成的 IR 形成记录，归档至 [`memory-compiler-ir.md`](./implementation/memory-compiler-ir.md)。

**当前承接：** [MemoryCompiler](../../patchouli/memory-compiler.md)完整描述 source builder、`MemoryUnitIR`、unit target、section/bundle IR、envelope、retrieval context 策略和当前调用点。

**保留理念：** IR 是 source normalization 与 target rendering 之间的内部语义边界；单项记忆表达和多项 envelope 是两种责任；Compiler 不能拥有检索、权限、持久化或运行时调度。`RUNNABLE_TOOL` 继续保留为显式 reserved target，而不是半实现能力。

**拒绝继承：** Phase 2A/2B/2C 的过渡结构、旧 handler 入口和“未来 RUN”字段不是当前保证。IR 不对外承诺为可持久化公共协议，format/citation/token budget 的现有不一致以当前限制为准。

### 3.2 `MemoryCompilerRetrievalRefactorPlan.md`

**分类与动作：** 已完成的边界重构，归档至 [`memory-compiler-retrieval-refactor.md`](./implementation/memory-compiler-retrieval-refactor.md)。

**当前承接：** [记忆检索](../../patchouli/retrieval.md)拥有召回、过滤、fusion、rerank 和访问副作用；MemoryCompiler 拥有 IR、target 与 Agent-facing envelope。当前 RetrievalEngine 已不持有旧 renderer 工厂。

**保留理念：** Retrieval 回答“选哪些、如何排序”，Compiler 回答“面向哪个任务如何表达”；让 Retrieval 返回结构化记忆结果、由调用方选择 target，避免 renderer 再次绑死检索策略和 prompt 格式。

**拒绝继承：** 旧 renderer 类名、Phase 2C 兼容注释和过渡调用签名不再继承。Compiler 不重新排序，Retrieval 也不能通过字符串输出反向拥有 Agent prompt。

### 3.3 `MemoryGenerationManagementEnhancementPlan.md`

**分类与动作：** 已完成的任务管理增强，归档至 [`memory-generation-management-enhancement.md`](./implementation/memory-generation-management-enhancement.md)。

**当前承接：** [记忆生成](../../patchouli/generation.md)已经说明 Coordinator 并发构造 active specs、每个 spec 独立失败、TaskController 的唯一终态、`wait_many/wait_all`、`asyncio.shield`、shutdown drain 超时后显式取消，以及 registry 与 backpressure 的当前限制。

**保留理念：** 一项 UPDATE 的 UUID/读取失败不应阻断同批 WRITE；等待超时不能偷偷接管或取消后台任务；关闭时既要停止新提交，也要给已经登记的任务一个明确排水窗口。

**拒绝继承：** 旧 `flush_all_for_shutdown` TODO、旧 service 路径和“串行构造全部 spec”的问题已失效。任务持久化、进程恢复、全局并发额度和 backpressure 仍未实现，已由 Local Work Queue Plan 承接，不能把本稿写成完整 durable queue。

### 3.4 `PatchouliSubsystemRefactorPlan.md`

**分类与动作：** 已完成/被后续实现细化的子系统重构，归档至 [`patchouli-subsystem-refactor.md`](./implementation/patchouli-subsystem-refactor.md)。

**当前承接：** [Patchouli 总览](../../patchouli/README.md)、[MemoryLibrary](../../patchouli/memory-library.md)、[感知](../../patchouli/perception.md)、[生成](../../patchouli/generation.md)、[检索](../../patchouli/retrieval.md)和[生命周期](../../patchouli/lifecycle.md)已经覆盖 System/Runtime/Application/Familiar/Coordinator/Library 的所有权与主要数据流。

**保留理念：** “Librarian”应当是整个 Patchouli 子系统的隐喻，而不是一个拥有所有存储、生成、检索和维护行为的 God Object；MemoryLibrary 是存储状态图的唯一组合边界；Familiar 是领域能力，Coordinator/Controller 是控制面；跨边界调用必须经过稳定 route 或应用服务。

**拒绝继承：** 旧 Phase 编号、LibrarianCore 迁移表、旧 local/public route 草案和 DB/Redis/SQL adapter 承诺不再是当前状态。当前长期层是 file-based，public application service、bridge 和 local bus 的真实接线以代码及现行文档为准。

### 3.5 `V0.5.0DataDurabilityAndAsyncColdPathPlan.md`

**分类与动作：** 已完成主体、剩余缺口已进入当前限制的版本实施稿，归档至 [`v0.5.0-data-durability-and-async-cold-path.md`](./implementation/v0.5.0-data-durability-and-async-cold-path.md)。

**当前承接：** [Artifacts](../../patchouli/artifacts.md)承接 Interaction/Document/Creation/Version artifact、hash 与 best-effort provenance；MemoryLibrary 承接三层存储和冷热迁移；Generation 承接 artifact -> upsert 顺序与异步冷路径。

**保留理念：** 当前 MemoryAtom 是可演化知识头部，artifact 是不可变证据/版本记录；原始材料与派生结论必须拥有不同身份；历史不应通过无限追加正文模拟；冷路径不能成为前台响应的强制等待项。

**拒绝继承：** 本地文件 artifact 不是完整数据耐久性，artifact 与 Qdrant 也不是原子事务；DocumentArtifact 模型存在不代表完整 Document Ingestion 已交付；短期 store、memory task 和 archive/revive 仍没有崩溃恢复事务。

### 3.6 `V0.5.1InfraCleanupPlan.md`

**分类与动作：** 已完成的基础设施清理，归档至 [`v0.5.1-infrastructure-cleanup.md`](./implementation/v0.5.1-infrastructure-cleanup.md)。

**当前承接：** [System 配置](../../system/configuration.md)承接 shared/subsystem config 与 registry；Artifacts 承接 `ArtifactEngine.noop()` 和三个 NoOp builder；[Agent Runtime](../../alice/agent-runtime.md)与 [MTP Runtime](../../alice/mtp-runtime.md)承接取消 checkpoint 和实际限制。

**保留理念：** 可选组件应以行为中性的 NoOp 实现收敛调用点，而不是让主链布满 `if component is not None`；取消 token 必须沿 Agent loop 传到 MTP 执行边界；配置按所有者组织，组合根负责唯一装配。

**拒绝继承：** 旧稿预想的单一 `NullArtifactEngine` 已由薄 `ArtifactEngine` 加 NoOp builders 实现，名称不是不变量。当前 Koakuma 的可变 `cancel_event` 仍有并发/陈旧值张力，同步 syscall 也不能强抢占，不能引用旧验收目标宣称完全取消安全。

### 3.7 `V0.5.2AsyncNativeAdaptationPlan.md`

**分类与动作：** 已实施记录，归档至 [`v0.5.2-async-native-adaptation.md`](./implementation/v0.5.2-async-native-adaptation.md)。

**当前承接：** Qdrant 由 `AsyncQdrantClient` 装配，GenerationEngine 和 Retrieval 链路公开 async 接口，Patchouli Familiar 通过 `await` 组合；当前边界分别由 MemoryLibrary、Generation 与 Retrieval 文档说明。

**保留理念：** 单一 async 调用链应由真正的异步存储/模型接口承担，不能长期用 `asyncio.to_thread()` 遮蔽同步客户端；生命周期与健康检查也必须服从同一个 event loop。

**拒绝继承：** v0.5.1/v0.5.2 是开发批次而非独立发布标签；“async-native”不等于任意操作都可立即取消，也不等于后台任务已经持久化。

## 4. System、Gateway 与 Runtime

### 4.1 `RuntimeEventPublishingRefactorDesign.md`

**分类与动作（后续更新）：** 当时曾迁移为 RuntimeEvent 生产端重构 Plan；随着 Publisher/Alice/Memory emitter 部分落地，当前规范已由 System/Contracts 承接，剩余工作缩减为 [Todo](../../todo/runtime-event-producer-migration.md)，原完整设计现已[归档](./runtime-event-publishing-refactor.md)。

**当前承接与缺口：** [System 可观测性](../../system/observability.md)已经承接 RuntimeEvent 是 best-effort 旁路、单一顶层扁平流、scope、payload 摘要和 observer 不驱动业务等稳定不变量；但 `RuntimeEventPublisher`、不可变 bind context、全域领域 emitter 与关键 payload 类型化并不存在。代码仍保留 `ChatApplicationService._emit_chat_event()`、`GatewayWorkflow._emit()`、`AliceRuntime._emit_agent_event()`、`MemoryGenerationTaskController._emit_memory_task_event()`、Scheduler/System lifecycle 私有发布方法。

**保留理念：** 生产端应分为领域 emitter、基础 publisher 与 sink/bus；Emitter 按稳定事件族而非 Python 类机械拆分；复杂取消/fallback/stream 终态保持显式发布，线性 started/completed/failed 才适合 operation observer。

**拒绝继承：** 不把旧稿误标为“已完成”，也不因为 Passive Ingress 已有 emitter 就宣称全域已收敛。Plan 不修改 wire format、不建立子系统私有观测总线，也不让事件驱动业务状态。

### 4.2 `V0.4.0RuntimeControlAndObservabilityPlan.md`

**分类与动作：** 已发布版本实施稿，归档至 [`v0.4.0-runtime-control-and-observability.md`](./implementation/v0.4.0-runtime-control-and-observability.md)。

**当前承接：** [System 应用服务](../../system/application-services.md)承接 chat run、取消请求—确认和 finalize/cleanup；[System Runtime](../../system/runtime-and-bus.md)承接进程内控制表和 Scheduler；System 可观测性承接 RuntimeEventBus/replay/gap；Generation 承接可查询、等待和取消的 memory task。

**保留理念：** chat run 与 background memory task 是两个生命周期；取消请求不等于取消终态；Agent 完成后 chat 仍可能 finalizing；事件与日志职责分离；观测流独立于功能总线并保持扁平，慢消费者不能阻断业务。

**拒绝继承：** 跨进程恢复、持久化 event log、同步 syscall 强抢占、私有子系统事件流和完整历史查询没有实现。前端具体窗口和旧 API 路径只作为历史交付轨迹保留。

### 4.3 `V0.6.0GatewaySystemDesign.md`

**分类与动作：** 已完成主体、被当前模块文档拆分承接的 Gateway 实施稿，归档至 [`v0.6.0-gateway-system.md`](./implementation/v0.6.0-gateway-system.md)。

**当前承接：** [Gateway 总览](../../gateway/README.md)、[固定工作流](../../gateway/workflow.md)、[分析](../../gateway/analysis.md)、[命令](../../gateway/commands.md)和 [System Passive Ingress](../../system/passive-ingress.md)覆盖子系统边界、固定 Step、冻结 state/outcome、deadline/cancel/fallback、上下文 Provider 与主动/被动入口。

**保留理念：** Gateway 是入口“观察与判断”层，不是系统中枢神经；一次分析只能受限复用为入口投影，不能升级为检索结果或持久化真相；固定 workflow 让每个能力拥有局部 timeout/fallback，而不是引入自由 DAG。

**拒绝继承：** 旧 Step/Engine 文件布局、`GatewayExecutionState` 草案字段、阶段编号和未落地配置不是当前 API。动态 branch、复合 intent、多任务执行和持久化 workflow 状态没有随主体重构自动完成，其中复合意图已经进入独立 Plan。

### 4.4 `V0.6.0GlobalCommandSystemDesign.md`

**分类与动作：** 已完成主体的命令系统设计，归档至 [`v0.6.0-global-command-system.md`](./implementation/v0.6.0-global-command-system.md)。

**当前承接：** [Gateway 全局命令](../../gateway/commands.md)已经说明 Registry、deterministic parser、Dispatcher、副作用边界、权限/确认、内置命令、短路结果和限制。

**保留理念：** 系统命令必须在普通话题/检索/Alice 前确定性识别；Parser 不执行副作用，Dispatcher 不重新解释自然语言；命令终态即使 rejected/failed/not implemented 也应短路普通 chat，避免控制指令被 LLM 猜测执行。

**拒绝继承：** 旧稿中规划的全部 runtime/debug handler、前端 command palette、配置热注入和自然语言命令别名没有全部落地；当前命令集合、权限与确认能力以代码和现行限制为准。

### 4.5 `V0.6.0PassiveIngressDesign.md`

**分类与动作：** 已完成主体的实施稿，归档至 [`v0.6.0-passive-ingress.md`](./implementation/v0.6.0-passive-ingress.md)。

**当前承接：** [System Passive Ingress](../../system/passive-ingress.md)承接外部事件、conversation key、去重、顺序 buffer、seal/outbox/retry、降级和 shutdown drain；Gateway workflow 承接 `PASSIVE_MEMORY` 入口策略。

**保留理念：** Passive Ingress 是外部对话记忆中间件，不是第二套 active chat runtime，也不是通用 Document Ingestion；它可以复用 Gateway 决策和 Patchouli 记忆能力，但绝不执行 command、Alice、MTP 或回复生成；accepted 不能被伪装成已经持久化。

**拒绝继承：** buffer/outbox 不归 Gateway，外部 transport 身份验证也不由 Gateway 决定。当前 outbox 仍是进程内、失败会重试但崩溃不可恢复，后续耐久化由 Work Queue Plan 承接。

### 4.6 `V0.6.0UserQueryAnalysisGen1TechDebt.md`

**分类与动作：** 当前事实与设计矛盾已并入 Gateway analysis，旧稿归档至 [`v0.6.0-user-query-analysis-gen1-tech-debt.md`](./implementation/v0.6.0-user-query-analysis-gen1-tech-debt.md)。本次不新建重复 Todo。

**当前承接：** Gateway analysis 已说明三层 Resolver、三种 fallback、memory signal 时机过早、共享调用风险、硬编码规则、私有 `sub_intents`、RetrievalPlan 权重消费有限和 config 边界未收敛；同时保留“指标先行、验证消费者后再拆 Engine”的演进原则。

**保留理念：** `memory_write_signal` 只是输入阶段预判，不能替 Patchouli 对完整 turn 的最终生成判断；候选能力拆分必须先验证 required input、consumer、failure policy、ordering 与 latency/cost；rewrite、keywords、signal 与 composite 质量应由可复现样本和脱敏指标判断。

**拒绝继承：** “第二代规划”尚未形成独立版本、实现阶段和跨系统验收，范围也明显大于小型 Todo；因此不为了迁移清单机械创建 Todo。若后续数据证明需要完整改造，应新建 Plan；复合 intent 已由专门 Plan 承接。当前私有 `sub_intents` 和未消费权重不能被写成已交付能力。

## 5. 保留为当前 Plans 的未来工作

### 5.1 `V0.6.0CompositeIntentDecompositionDesign.md`

**分类与动作（后续更新）：** 当时曾迁移为复合意图分解 Plan；由于缺少版本承诺与证据门禁，现归类为[复合意图分解 Idea](../../ideas/composite-intent-decomposition.md)。

**当前差距：** 代码只有公共 `IntentType.COMPOSITE` 和 Engine 私有 `sub_intents`；Resolver 不提交 sub-intents，workflow 没有 `CompositeGatewayDecision`、branch execution、merge/fallback policy 或旧稿设想的 `CompositePlaceholder`。迁移时已删除这些占位字段“已经实现”的错误口径，并新增样本/指标门禁 Phase C0。

**保留理念：** 先冻结 composite envelope 与下游消费协议，再实现 LLM decomposition；不向下游暴露 `list[GatewayState]`；低置信度、解析失败、下游不支持和合并失败都必须能退回单主意图路径；Gateway 分解入口，Patchouli/Alice/Job Runtime 仍拥有各自执行语义。

### 5.2 `V0.6.1LocalWorkQueueRuntimePlan.md`

**分类与动作：** 先迁移为 v0.6.1 当前 Plan，完成后归档为 [v0.6.1 Local Work Queue Runtime](./v0.6.1-local-work-queue-runtime.md)。

**当前差距：** Passive outbox 与 memory task 都是进程内专用实现；memory task 提交后直接 `asyncio.create_task()`，没有 pending queue、配额、backpressure 或 durable retry runner；Scheduler 只回答“何时触发”，不是业务 work store。

**保留理念：** 共享机械生命周期，不共享业务队列；一套 runtime、多条 lane，分别维护 ordering、retry、幂等、取消、成功与可见性语义；基础设施不 import Patchouli/Alice payload；首期优先单机可靠性，不提前引入分布式队列或 DAG。

### 5.3 `RuntimeEventPublishingRefactorDesign.md`

本文件的 Plan 处理见 §4.1。它与 Work Queue 一样是当前实现之上的结构性改进，不能因旧路径位于 `mod/` 就失去计划身份。

## 6. 最终物理迁移

审计通过后，十五篇完成/被替代稿进入 `docs/archive/plans/implementation/`，三篇仍有效计划进入 `docs/plans/`：

```text
docs/mod/{15 completed or superseded records}
  -> docs/archive/plans/implementation/

docs/mod/RuntimeEventPublishingRefactorDesign.md
  -> docs/archive/plans/runtime-event-publishing-refactor.md

docs/mod/V0.6.0CompositeIntentDecompositionDesign.md
  -> docs/ideas/composite-intent-decomposition.md

docs/mod/V0.6.1LocalWorkQueueRuntimePlan.md
  -> docs/archive/plans/v0.6.1-local-work-queue-runtime.md
```

归档文件统一修正到当前文档、源码、测试和彼此历史稿的相对链接；Roadmap、Plans/Archive 索引、System/Gateway 入口、源码注释和迁移清单不再引用活动 `docs/mod/` 路径。

## 7. 验证门禁

- [x] 十八篇旧稿均给出事实承接、理念保留、拒绝项与最终分类；
- [x] MemoryGeneration 的 spec 隔离、并发构造、wait/shutdown drain 与 durable queue 缺口均以代码复核；
- [x] RuntimeEvent Publisher/Emitter 尚未实现的事实由直接生产点搜索复核，原归档分类已纠正为 Plan；
- [x] Composite Intent 的私有 `sub_intents` 与公共协议差距已复核并写回 Plan；
- [x] Query Analysis 第二代方向未被机械降格为小型 Todo；
- [x] 十五篇历史稿与三篇当前 Plan 完成物理迁移，原 `docs/mod/` 路径清空；
- [x] 当前入口、Roadmap、索引、源码注释与历史稿互链完成修复；
- [x] 严格 UTF-8、相对链接、旧路径、尾随空白、`git diff --check` 和定向测试完成验证。

最终结果：`docs/` 下共 138 篇 Markdown；除第 4～6 节审计已明确按原字节保留的损坏归档 `MemoryCompilerI18nMigrationPlan.md` 外，其余 137 篇的严格 UTF-8 解码与相对链接目标检查通过。RuntimeEvent、operation observer、Passive outbox、Gateway analysis、memory task、MemoryCompiler、Retrieval 与 Agent loop 共 197 项定向单元测试通过。该例外不是本批新产生的编码问题，本批没有读取其损坏正文或从中迁移当前事实。
