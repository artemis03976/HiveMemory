---
title: System Boundaries
status: current
owner: system
scope: subsystem-ownership-and-dependencies
code_paths:
  - src/hivememory/system/
  - src/hivememory/gateway/
  - src/hivememory/patchouli/
  - src/hivememory/alice/
  - src/hivememory/core/protocol/
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/routes-and-events.md
last_reviewed: 2026-08-05
---

# 系统边界与所有权

本文定义 System、Gateway、Patchouli、Alice 之间当前生效的责任、状态所有权和依赖方向。具体公开输入输出由[子系统契约](../contracts/subsystem-contracts.md)定义。

这里的“边界”不是为了让目录看起来整齐，也不是把一套进程内实现包装成微服务。它要解决的是一个更直接的问题：当一次交互同时经过入口判断、记忆检索、Agent 执行和长期沉淀时，究竟由谁作决定、由谁保存状态、又由谁对失败负责。如果这个问题没有唯一答案，同一份状态就会在多个 Runtime 中出现副本，业务顺序会散落到 HTTP router、事件订阅者和领域对象里，局部修复最终会改变整条链路的语义。

当前边界是在项目多次演进和解耦中逐步形成的。早期 Gateway、Patchouli、Alice 的部分职责曾经依附于更大的引擎或对象图；随着命令、被动摄入、PendingAtom、取消和观测能力加入，直接持有另一个子系统 Runtime 的方式开始制造循环依赖，也让“谁是权威所有者”变得模糊。因此，本文保留的不只是当前目录关系，更是这些关系背后的约束：跨边界只传递完成一次交接所需的公共事实，领域状态留在其所有者内部，顶层用例由 System 明确编排。

## 1. 如何判断边界

一个能力应该属于哪个子系统，首先看它在回答什么问题，而不是看哪个模块最方便调用它：

- System 回答“这次用例按什么顺序运行、如何取消和收尾”；
- Gateway 回答“输入意味着什么、接下来应该采取什么入口决策”；
- Patchouli 回答“长期知识是什么、如何检索、提交和演化”；
- Alice 回答“Agent 如何在本轮上下文中行动、调用工具并形成执行结果”。

这四类问题对应四种不同的状态寿命。System 持有一次用例及进程生命周期的控制状态；Gateway 的分析状态只服务于形成本次不可变决策；Patchouli 持有跨会话延续的记忆、话题和生成任务；Alice 持有一次 run 内的 frame、工具调用和临时别名。状态寿命不同，是职责不能简单合并的根本原因。

## 2. 边界原则

1. **System 编排，子系统执行**：跨多个子系统的用户用例由 System 应用服务编排，领域行为留在其所有者内部。
2. **公开路由跨边界，local bus 留在边界内**：调用方不能依赖另一个子系统的 local route 或 Runtime 组件。
3. **模型应依赖中立**：跨边界模型放在 `core/protocol` 或明确的公共 contract 模块中，不暴露具体引擎对象。
4. **状态只有一个所有者**：其他模块可以读取投影或发送命令，不能并行维护同一状态的第二份权威副本。
5. **观测不反向控制业务**：RuntimeEvent 可以描述运行过程，但订阅者和观测失败不能改变业务终态。

公共模型在这里相当于一张“交接单”：它应说明上一阶段已经确认了什么、下一阶段可以依赖什么，却不允许接收方通过模型继续操纵发送方的内部对象。将公共模型做成 frozen 或依赖中立结构，目的正是防止 workflow state、存储客户端和引擎实体沿调用链泄漏，最终形成无法辨认的共享内部状态。

local bus 则是一个子系统内部的组合机制。它允许所有者替换内部实现，却不承诺跨子系统稳定性。一旦其他子系统直接依赖 local route，所谓内部重构就会变成隐蔽的公共契约变更，因此跨边界能力必须显式提升为公共 route、公共模型或全局事件。

## 3. 责任矩阵

| 边界 | 负责 | 明确不负责 |
|:---|:---|:---|
| System | 组合根、应用服务、生命周期、全局总线、运行控制、被动摄入、调度、注册表 | 查询分析、记忆算法、Agent loop、MTP 具体执行 |
| Gateway | 入口拦截、命令、话题/查询分析、检索计划、保守降级 | 记忆存储、检索执行、回复生成、interaction 提交 |
| Patchouli | 记忆/话题/Profile、检索、感知、生成、生命周期、prepare/finalize | 入口命令、顶层 chat 编排、Agent 生成循环 |
| Alice | Agent run、frame 编排、MTP/工具执行、PendingAtom 运行时 | 长期记忆所有权、Gateway 分析、HTTP 与顶层 chat 生命周期 |
| Core contracts | 依赖中立的数据模型、协议枚举和稳定常量 | 业务编排、I/O 与运行时状态 |

## 4. System 边界

System 是舞台管理者：它知道一次完整用例需要哪些参与者、按什么顺序登场，以及在取消和异常时如何收尾；它并不替参与者完成领域工作。把编排集中在 System，可以让 HTTP、CLI 或未来其他入口共享同一条业务链，也避免 transport adapter 成为第二套业务实现。

### 4.1 拥有的状态

- `HiveMemorySystem` 启停状态；
- chat generation registry，以及其中的 phase、outcome、首次 stop reason 与当前可中断阶段 task 引用；
- Passive Ingress 的去重、turn buffer、outbox 与 drain 状态；
- `RuntimeEventBus` 有界事件缓冲；
- 全局维护任务注册与调度状态；
- Provider / Model 注册信息。

### 4.2 允许的依赖

System 可以依赖各子系统的公开宿主类型完成装配，也可以通过全局路由调用公共能力。HTTP router 依赖 System 应用服务，而不是直接依赖子系统 Runtime。

### 4.3 禁止的越界

- 在 server router 中重新实现 prepare/run/finalize；
- 从应用服务直接调用另一个子系统的 Familiar、Controller 或 local bus；
- 把 RuntimeEvent 当成可靠命令或业务返回值；
- 由 Scheduler 实现 Patchouli 维护算法。

## 5. Gateway 边界

Gateway 的价值在于把不稳定、可能降级的入口分析收敛为一个可供下游执行的决定。它可以使用话题候选和记忆上下文帮助判断，但不能因为“看见了知识”就取得知识所有权，也不能因为“决定了下一步”就亲自执行下一步。这样才能让入口策略演进，而不牵动存储和 Agent loop。

### 5.1 输入与输出

Gateway 的唯一公开业务入口是 `gateway.public.process`。输入包括消息、`Identity`、`GatewayIngressMode` 以及可选 `request_timeout_ms`；输出为命令终态或普通决策终态。Chat application 通过取消自己创建的 Gateway task 中断调用，不向 Gateway 传递控制句柄。

`GatewayDecision` 只表达下游需要的稳定事实：目标话题、查询重写、关键词、记忆写入信号、检索计划和主意图。Workflow 的 step、snapshot、fallback 原因与内部分析对象不越过公共边界。

### 5.2 降级边界

- 候选话题失败：使用空候选集；
- 话题路由失败：保守选择 `NEW_TOPIC`；
- 已路由话题加载失败：使用空话题上下文；
- 查询分析失败：保留原始查询，使用 `RAG + HYBRID` 和默认 `top_k`；
- step 投影、状态提交或终态不变量失败：不使用局部 fallback，整个 workflow 失败；
- 外层 task cancellation：原生 `asyncio.CancelledError` 直接传播，不转为 Gateway 业务错误或 fallback；
- 总 deadline 耗尽：只有剩余步骤都有安全 fallback 时才能继续形成保守终态，否则抛出 `GatewayTimeoutError`。

这些降级并不是把所有异常吞掉。候选集、辅助上下文和单项分析失败时，Gateway 仍可能形成保守而完整的决定；一旦 workflow 无法满足终态不变量，再继续返回“看起来可用”的结果只会把程序错误伪装成正常决策，因此必须让整次处理失败。

### 5.3 模式边界

`ACTIVE_CHAT` 允许命令识别和分发；`PASSIVE_MEMORY` 禁止命令分支。两种模式共用决策 workflow，但后续由不同的 System 应用服务消费。

## 6. Patchouli 边界

Patchouli 是长期知识事实的核心。检索、话题、Profile、Interaction 和记忆生成虽然处于不同流程，但共同决定“系统以后相信什么、能够找回什么”。这些状态必须在同一个所有权边界内演化，否则短期执行结果会绕过感知、来源和生命周期规则，直接成为另一份长期事实。

### 6.1 拥有的状态

- MemoryAtom、索引、payload、版本与 artifact 引用；
- 活跃话题、语义缓冲和 interaction；
- Agent Profile 的长期表示；
- 记忆生成任务与维护状态；
- 检索缓存、生命周期统计和引用/反馈记录。

### 6.2 Prepare / Finalize 边界

`prepare_agent_run` 把 Gateway 决策转换为 `PreparedAgentRun`：解析真实话题、Agent Profile、检索结果、MemoryCompiler 文本与 stream prelude。

`finalize_agent_run` 只接收 `PreparedAgentRun + AgentRunResult`，由 Patchouli 自己构造 `InteractionPayload`、归约 trace、提交感知链并调度 materialize task。

如果 System 未能完成 finalize，只能调用 cleanup 请求 Patchouli 清理预创建空话题，不能自行修改话题状态。

prepare/finalize 把“为本次执行准备记忆视图”和“把完成后的交互提交回长期系统”放在 Patchouli 两端，中间只让 Alice 消费一个本轮快照。这一设计允许 Alice 专注执行，又确保长期状态的创建与结算仍经过 Patchouli。cleanup 只是对 prepare 阶段临时副作用的补偿，不是跨子系统事务回滚：已经存在或已经产生内容的长期状态不会因为本轮执行失败而被调用方撤销。

### 6.3 禁止的越界

- Patchouli 不调用 Alice 生成回复；
- Patchouli 不重新执行 Gateway 查询分析；
- 外部调用方不直接操作 MemoryLibrary、Familiar 或生成 Controller；
- Patchouli local PendingAtom 事件只有经 Bridge 转发后才是全局事件。

## 7. Alice 边界

Alice 是知识的使用者和行动者。它可以在一次 run 中读取记忆、建立 PendingAtom、执行工具和调度子 frame，但这些都是为了完成当前任务而存在的运行时视图。将临时执行状态与长期知识分开，既能让 Agent 在单轮内保持连贯，也能避免失败、取消或半完成的 run 直接污染正式记忆。

### 7.1 拥有的状态

- 一次 Agent run 的 frame、消息、turn events 和终态；
- Agent loop 的迭代与流式执行资源；task cancellation 的业务裁决属于 System，Alice 只负责原生传播与本地 unwind；
- Koakuma 的 MTP parser、权限检查、alias cache 与 syscall registry；
- PendingAtom 在当前运行期内的别名、redirect 和 terminal view；
- CALL 的父子 frame 调度。

### 7.2 依赖方向

Alice 接收 Patchouli 准备好的 `AgentRunContext`。需要检索、别名读取、Profile 或引用记录时，经映射到 Alice local bus 的全局公开路由访问 Patchouli。

Patchouli 结算 PendingAtom 后，通过全局事件通知 Alice 更新运行时视图。Alice 不以此取得正式记忆所有权。

### 7.3 禁止的越界

- Alice 不持有 Patchouli Runtime、Service 或存储客户端；
- MTP WRITE/UPDATE 不在 Koakuma 内直接写入正式记忆；
- Alice 不决定一个已完成 run 是否应被提交到感知层；
- CALL 不能从深度大于等于 1 的子 frame 再递归发起。

## 8. 数据所有权

| 数据或状态 | 权威所有者 | 跨边界形式 |
|:---|:---|:---|
| `GatewayExecutionState` | Gateway | 不公开；只投影 `GatewayProcessResult` |
| `GatewayDecision` | Gateway 形成，调用链只读消费 | frozen 公共模型 |
| `PreparedAgentRun` | Patchouli | dataclass，只供本轮 System/Alice 协作 |
| `AgentRunContext` | Patchouli 组装，Alice 消费 | Pydantic 公共模型 |
| `AgentRunResult` | Alice | Pydantic 公共模型 |
| `InteractionPayload` | Patchouli 组装并消费 | 公共协议模型，不由 router 拼装 |
| `MemoryAtom` / Topic | Patchouli | 公共模型或受控路由返回值 |
| PendingAtom 运行时状态 | Alice | 结算事件从 Patchouli 回传 |
| chat / passive run 控制 | System | 应用服务内部状态与 RuntimeEvent 投影 |
| RuntimeEvent | System 观测设施 | best-effort 事件信封 |

## 9. 允许的调用方向

```text
Server adapters -> System application services
System application services -> GlobalSystemBus public routes
Gateway -> Patchouli public read routes (话题上下文)
Alice -> Patchouli public read/citation routes (MTP)
Patchouli -> GlobalEvents -> Alice (PendingAtom 结算通知)
Subsystem -> RuntimeEventSink (观测旁路)
```

任何反向调用都需要先确认是否在制造循环所有权。新跨边界能力应优先扩展公共 route/model，而不是注入对方 Runtime 对象。

## 10. 公共模型的当前限制

`core/protocol/models.py` 仍保留部分历史兼容：`QueryFilters` 从 Retrieval 引擎重导出，`TopicData` 在运行时以 `Any` 避免循环导入。这不改变当前调用契约，但说明公共模型的依赖中立化尚未完全结束。修改这些模型时必须同时运行 core、Gateway、Patchouli、Alice 与 System 契约测试。

## 11. 如何发现边界矛盾

评审新能力时，可以先用以下问题检查它是否正在破坏边界。任何一个问题得到肯定答案，都不一定意味着实现必然错误，但必须给出新的所有权理由并同步修改契约：

1. 这项改动是否在另一个子系统中建立了同一长期状态的第二份权威副本？缓存和运行时投影是否被误当成正式事实？
2. 调用方是否必须直接持有另一个子系统的 Runtime、Service、Controller、存储客户端或 local bus 才能完成工作？
3. 这次交接需要一个确定返回值，却为了“解耦”改成了无人对结果负责的 Pub/Sub 事件吗？反过来，一个纯通知是否不必要地阻塞了发布者？
4. 新的公共模型是在传递稳定事实，还是把可变 workflow state、引擎实体或回调能力泄漏给了接收方？
5. fallback 是否仍能形成满足不变量的保守结果，还是正在掩盖装配错误、投影错误或程序缺陷？
6. WRITE/UPDATE 返回的 PendingAtom ACK 是否被当成正式记忆已经持久化？结算事件是否被误解为 Alice 取得了长期所有权？
7. RuntimeEvent、日志或 UI 观测是否开始反向决定业务是否成功？

这些检查的目的，是尽早暴露设计矛盾，而不是等循环依赖或数据分叉已经出现后再靠目录移动修复。

## 12. 边界变更要求

以下变化必须同步更新本文和 Contracts：

- 状态所有者变化；
- 新增跨子系统直接依赖；
- 公开 route、事件、输入输出或终态变化；
- prepare/finalize、取消、清理或启停顺序变化；
- local 能力升级为公共能力；
- RuntimeEvent 开始参与业务控制。

主要验证入口：`tests/unit/system/contracts/`、`tests/unit/system/application/`、`tests/unit/gateway/`、`tests/unit/patchouli/`、`tests/unit/alice/`。
