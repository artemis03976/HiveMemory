---
title: Patchouli
status: current
owner: patchouli
scope: memory-and-knowledge-subsystem
code_paths:
  - src/hivememory/patchouli/
  - src/hivememory/engines/perception/
  - src/hivememory/engines/generation/
  - src/hivememory/engines/retrieval/
  - src/hivememory/engines/lifecycle/
  - src/hivememory/engines/artifacts/
  - src/hivememory/engines/memory_compiler/
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/routes-and-events.md
  - docs/contracts/mtp.md
  - docs/architecture/boundaries.md
last_reviewed: 2026-08-18
---

# Patchouli

Patchouli 是 HiveMemory 的记忆与知识子系统。若说 Gateway 决定一条输入应当如何进入系统，Alice 决定一次 Agent run 如何执行，那么 Patchouli 回答的就是另一组更缓慢、也更需要保持连续性的问题：哪些交互应形成长期资产，一条记忆如何被检索和使用，它在被修订、强化、衰减与归档时由谁维护真相。

项目借用“帕秋莉大图书馆”的形象，不是为了把所有记忆逻辑集中进一个万能馆长，而是为了强调知识所有权必须稳定。检索使魔可以找书，感知层可以整理尚未成册的交互，生成链可以提出新书或修订稿，生命周期能力可以决定何时移入冷藏库；但所有会改变长期记忆事实的路径，最终都必须回到 Patchouli 自己的存储、任务和版本边界内完成。

## 1. 设计定位

### 1.1 Patchouli 拥有什么

Patchouli 当前拥有：

- 活跃话题及其短期语义缓冲；
- `MemoryAtom` 的中期持久化、检索、创建、更新和删除；
- 长期冷存储及中期记忆的 archive/revive 状态转移；
- Interaction、Document、Memory Creation 与 Memory Version artifacts；
- Perception、Generation、Retrieval、Lifecycle 和 MemoryCompiler 能力；
- 后台记忆生成任务及其运行时终态；
- 作为记忆保存的 Agent Profile；
- Patchouli 内部 local routes，以及向 `GlobalSystemBus` 暴露的公开记忆能力。

这里的“拥有”不是指每个模型都必须定义在 `patchouli/` 目录下。`TurnRecord`、`MemoryAtom`、`PendingAtomSettlement` 等跨模块模型位于 `core`，底层算法仍位于 `engines`；所有权指的是谁决定这些对象如何进入长期状态、谁负责持久化与演化，以及发生冲突时由谁维护权威事实。

`v0.6.2` 候选设计拟把 WorkspaceAsset working set 与附件读取路由放在 Patchouli 的共享 Runtime 边界内，但该能力尚未实现。它不会成为每 Workspace 一套 Patchouli 实例，也不是 MemoryLibrary 的持久化第五层或 Artifact 的别名；MVP 仅承诺进程内生命周期，只有实际参与 Memory CREATE/UPDATE 的 representation 才在 Materialization 时提升为 Artifact。具体边界见 [Workspace MVP 初步设计](../ideas/workspace-mvp-chat-attachments-design.md)。

### 1.2 Patchouli 不拥有什么

Patchouli 不负责：

- 解释原始入口消息、识别系统命令或形成 `GatewayDecision`；
- 运行 Agent loop、控制模型生成、执行工具或编排子 Agent；
- 拥有 System 的 chat/passive ingress 用例、全局维护时钟或跨系统取消；
- 把检索结果无条件解释为正确事实；
- 把 `WRITE` / `UPDATE` 的即时 ACK 当作正式记忆已经持久化；
- 为任意不受信任内容提供执行沙箱。

Gateway、Alice 与 System 只能通过公开路由或显式用例交接使用 Patchouli，不能持有它的 Runtime、Familiar、Store 或 local bus。完整边界见[子系统公共契约](../contracts/subsystem-contracts.md)与[系统边界](../architecture/boundaries.md)。

## 2. 为什么不再保留一个 LibrarianCore

早期设计让 `LibrarianCore` 同时连接感知、检索、生成、生命周期和总线调用。它能够快速串起功能，却也把三种不同责任压在同一个对象里：领域算法、跨模块编排和运行时任务管理。结果是感知层可以直接触发生成，生成引擎可以决定持久化副作用，公开 API 又容易穿透到 Runtime 本体。

当前实现把这组责任拆成四层：

```text
PatchouliSystem
  -> public application services / PatchouliService
  -> PatchouliBridge
       -> GlobalSystemBus public routes
       -> PendingAtom global events
  -> PatchouliRuntime
       -> PatchouliBus + local route bindings
       -> Familiar / Coordinator / TaskController
       -> Engine / MemoryLibrary / ArtifactStore
```

- `PatchouliSystem` 是子系统容器，负责装配、公开桥接、维护任务注册和启停；
- application services 与 `PatchouliService` 表达公开用例，不直接持有 Runtime；
- `PatchouliRuntime` 创建基础设施、引擎、内部服务和 local routes；
- Familiar 承担领域用例，Coordinator 构造任务规范，TaskController 管理异步任务终态，Engine 专注计算，MemoryLibrary 管理存储。

这不是为了增加名词，而是为了让每一层只保留一种变化原因。算法可以独立测试，任务控制不必知道如何提取记忆，跨子系统接口也不再等同于内部方法集合。

## 3. 当前核心流程

### 3.1 主动 Agent 流程

```text
GatewayDecision
  -> Patchouli PREPARE_AGENT_RUN
       -> prepare topic
       -> retrieve memories
       -> MemoryCompiler compile retrieval context
       -> AgentRunContext + StreamPrelude
  -> Alice RUN / RUN_STREAM
  -> Patchouli FINALIZE_AGENT_RUN
       -> TurnEvent -> Action / semantic trace
       -> submit InteractionPayload to Perception
       -> WRITE / UPDATE materialize tasks -> generation tasks
       -> best-effort record retrieval HIT
```

Prepare 只准备运行上下文，不运行 Alice。Finalize 消费已经完成的 `AgentRunResult`，先等待本轮结构化事实成功摄入话题；该 applied gate 锁定 Chat completed，之后在同一个 Active continuation 内并行执行 MTP 物化接纳与 best-effort retrieval HIT，不反向改写 Chat 终态。HIT 只做单批去重，不自动 retry，也不提供跨 finalize 去重。若 prepare 已创建新话题而 run 没有走到 finalize，System 会调用 cleanup 删除仍为空的话题。

`WRITE` / `UPDATE` 的 ACK 只代表 Alice 已登记一个 PendingAtom。Patchouli 完成生成、去重、artifact 挂载和中期存储写入后，才通过 settlement 把 pending alias 投影为 canonical alias/UUID 或 discard/failure/cancel 终态。

### 3.2 被动摄入与话题结算

Passive Ingress 的去重、顺序缓冲、seal、retry 和降级属于 System。完成封口的一轮交互以结构化 `InteractionPayload` 进入 Patchouli 后，Perception 才负责将它变为 `TurnRecord` / `LogicalBlock`，加入目标话题并执行折叠或结算策略。

```text
System Passive Ingress
  -> Patchouli SUBMIT_INTERACTION
  -> PerceptionFamiliar
  -> SemanticFlowPerceptionLayer
  -> ShortTermMemoryStore / SemanticBuffer
  -> idle | LRU | shutdown | manual settle
  -> TopicMaterializeTask
  -> background Generation task
```

Patchouli 不为被动入口重新运行 Alice，也不重新分析 Gateway。完整入口语义见[被动摄入](../system/passive-ingress.md)，感知内部设计见[感知与短期话题](./perception.md)。

### 3.3 检索与上下文编译

Retrieval 负责根据身份与查询找到候选 `MemoryAtom`，MemoryCompiler 再按用途把记忆编译成检索上下文、MTP READ 响应、共享上下文或 embedding 文本。二者刻意分开：相关性排序不应决定 prompt 版式，渲染策略也不应反向改变检索状态。

```text
RetrievalRequest
  -> QueryFilters
  -> dense / sparse recall
  -> fusion
  -> optional rerank
  -> MemoryAtom[]
  -> MemoryCompiler(target, strategy, language)
  -> task-specific text view
```

## 4. 运行、维护与关闭

`PatchouliSystem.start()` 先确保 Qdrant 可用，再挂载 local routes、公开 bridge 和维护任务。模型预热是独立能力；预热失败会退化到首次请求懒加载，不阻止子系统启动。

Patchouli 向全局调度器注册两个业务任务：

- `perception_idle_flush`：扫描空闲话题并提交结算任务；
- `memory_gardening`：刷新生命力并归档低生命力中期记忆。

调度器只拥有时钟、非重入与停止语义，Perception 和 Lifecycle 仍拥有业务规则。关闭时 Patchouli 先结算全部非空活跃话题，再等待当时仍运行的记忆任务；超过 `generation_wait_timeout_seconds` 的任务会被请求取消，随后才卸载 bridge 与 local routes。

## 5. 当前设计文档

- [MemoryLibrary 与存储层](./memory-library.md)：短期、中期、长期和 artifact store 的所有权与状态转移；
- [Artifacts 与来源追踪](./artifacts.md)：原始交互、外源文档、创建记录和版本快照；
- [感知与短期话题](./perception.md)：结构化摄入、SemanticBuffer、Page Folding 与结算矩阵；
- [记忆生成](./generation.md)：三种生成模式、控制面/数据面、去重、任务和 PendingAtom settlement；
- [记忆检索](./retrieval.md)：身份过滤、Dense/Sparse 召回、融合、重排和访问副作用；
- [记忆生命周期](./lifecycle.md)：生命力、强化事件、gardening、archive 与 revive；
- [MemoryCompiler](./memory-compiler.md)：统一 IR、编译 target、envelope 与 token 策略。

跨系统 route、event 与错误不在本目录重复定义，分别以[公开路由与事件](../contracts/routes-and-events.md)、[子系统公共契约](../contracts/subsystem-contracts.md)和[跨边界错误模型](../contracts/error-model.md)为准。

## 6. 代码入口

| 责任 | 当前入口 |
|:---|:---|
| 子系统容器与启停 | `src/hivememory/patchouli/system.py` |
| 公开 prepare/finalize 门面 | `src/hivememory/patchouli/service.py` |
| 公开 route/event 桥接 | `src/hivememory/patchouli/runtime/bridge.py` |
| 内部装配与 local routes | `src/hivememory/patchouli/runtime/core.py`、`route_bindings.py` |
| 公开管理用例 | `src/hivememory/patchouli/application/` |
| 内部业务服务 | `src/hivememory/patchouli/services/`、`control/` |
| 存储协调 | `src/hivememory/patchouli/memory_library/` |
| 领域算法 | `src/hivememory/engines/{perception,generation,retrieval,lifecycle,artifacts,memory_compiler}/` |

## 7. 当前限制与设计张力

- 记忆生成任务和终态 registry 都是进程内状态，默认只保留最近 50 个终态任务；进程重启后不能恢复，也不是持久化 Job Queue；
- 短期话题默认只在内存中，关闭依赖 shutdown drain 尽力结算；异常退出仍可能丢失未结算 blocks；
- Artifact 是可选旁路，写入失败目前不会阻止 `MemoryAtom` 持久化，因此“记忆存在”不保证“来源链完整”；
- 中期与长期存储的 archive/revive 是顺序 I/O，不具备跨存储事务；
- Perception 的 token overflow 目前只形成摘要并清空当前 blocks，不生成长期记忆，配置中的 recent-block 保留量也尚未接入；
- Retrieval 与 MemoryCompiler 已经解耦，但若干过滤字段、预算口径和保留 target 仍存在实现缺口，详见各模块文档；
- `engines/` 仍是代码上的算法目录，不再是独立子系统或并行文档真相源。

这些限制属于当前真实边界。未来设计只有在进入 Plans 并落地后，才能改写本目录中的当前能力描述。
