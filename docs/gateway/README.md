---
title: Gateway
status: current
owner: gateway
scope: ingress-decision-subsystem
code_paths:
  - src/hivememory/gateway/
  - src/hivememory/engines/gateway/
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/routes-and-events.md
  - docs/architecture/boundaries.md
related_docs:
  - docs/architecture/workspace.md
last_reviewed: 2026-09-01
---

# Gateway

Gateway 是 HiveMemory 的入口决策子系统。它面对一条尚未被系统解释的消息，判断这条消息是否是系统指令、应归入哪个话题、应如何重写和检索，以及是否表现出值得写入记忆的信号；随后，它把这些判断一次性投影为命令终态或依赖中立的 `GatewayDecision`。

项目早期把这类能力称为“真理之眼（The Eye）”。这个比喻表达的是 Gateway 应当在系统入口形成统一视角，而不是赋予它全知全能的控制权。Gateway 可以“看见并判断”，但不执行检索、不生成回复、不持久化记忆，也不拥有 Agent 运行状态。执行仍分别属于 Patchouli、Alice 和 System 应用层。

## 1. 为什么 Gateway 成为独立子系统

旧实现把入口分析放在 Patchouli 的装配和 prepare 链路中，并让一步 LLM router 同时承担话题路由、指代消解、关键词提取、意图判断和记忆价值预判。这带来两个相互强化的问题：

1. 入口决策被记忆子系统反向托管，主动 chat 必须先进入 Patchouli 才能知道下一步做什么；
2. 多种语义判断共用一个不透明输出，任何局部退化都会污染整条决策，而下游很难区分哪一项能力失败。

当前设计把 Gateway 提升为与 Patchouli、Alice 平级的 `GatewaySystem`，并用固定 workflow 明确每一步的输入、输出、超时和 fallback。这样做不是为了增加一层抽象，而是为了让入口决策拥有清晰所有者，同时让记忆与 Agent 执行不再依赖 Gateway 的私有状态。

## 2. 当前职责与非职责

Gateway 当前负责：

- 区分 `ACTIVE_CHAT` 与 `PASSIVE_MEMORY` 两种入口策略；
- 在主动入口中识别、解析、授权和分发系统指令；
- 读取候选话题快照并选择已有话题或 `NEW_TOPIC`；
- 对路由后的用户输入生成意图、query rewrite、关键词、记忆写入信号和检索计划；
- 对可恢复的上下文或模型能力失败应用局部保守降级；
- 处理整次请求的 deadline 与 RuntimeEvent 观测；外层 task cancellation 原样传播，不建立 Gateway 私有取消协议；
- 只通过 `gateway.public.process` 暴露稳定公共结果。

Gateway 不负责：

- 执行向量/关键词检索或解释检索结果；
- 结算、压缩或删除话题，或创建、更新、归档长期记忆；
- 运行 Alice、MTP、工具或子 Agent；
- 生成自然语言回复；
- 保存可恢复的 workflow/job 状态；
- 替 transport 验证身份真实性，或替下游决定最终记忆是否应当持久化。

尤其需要区分 `memory_write_signal` 与“写入命令”：前者只是入口阶段根据用户输入形成的预判，Patchouli 仍拥有长期记忆生成、验证和持久化的最终决定权。

Gateway 的公开处理入口携带完整 `IdentityScope`，并将它交给需要 Workspace 上下文的 Patchouli route。命令 allowlist 与查询分析等局部能力可以只消费 `ActorIdentity` 投影；这不代表 Gateway 拥有 Workspace 资源授权，也不替下游重新定义 ownership。

### 2.1 一次分析、受限复用

旧 Gateway 文档中“Compute Once, Use Everywhere”的动机仍然成立：如果检索、话题感知和记忆生成各自重新解释同一条入口消息，就会产生额外的模型调用、延迟和彼此矛盾的判断。当前实现因此在一次请求内形成一份冻结的 `GatewayDecision`，让下游共享同一份入口投影：`rewritten_query`、`search_keywords` 和 `retrieval_plan` 可供 Patchouli 派生检索请求，`target_topic_id` 供话题准备使用，`memory_write_signal` 供后续记忆流程作为预判参考。

这里的“复用”有明确边界。复用的是 Gateway 对入口消息的提示、路由和分析结果，不是把 Gateway 的结果升级成检索结果、记忆价值裁决或持久化真相。Patchouli 仍要根据自己的身份、过滤器、存储状态和生成规则决定检索与写回；Alice 仍只消费准备好的运行上下文；任何 fallback 或 `UNKNOWN` 信号都必须在下游按本域语义处理。这样既避免重复解释，也避免一个入口 DTO 重新吞并各子系统的所有权。

## 3. 运行结构

```text
GatewaySystem
  -> GatewayRuntime
       -> GatewayBus（子系统本地路由）
       -> GatewayContextProvider
       -> RuleInterceptor / CommandRegistry / Dispatcher
       -> TopicRouterEngine
       -> UserQueryAnalysisResolver
       -> GatewayWorkflow
  -> GatewayService.process
  -> gateway.public.process
```

`GatewaySystem` 挂载生命周期和公开 route；`GatewayRuntime` 拥有本地依赖与 workflow；`GatewayService` 只处理请求级 timeout 收敛并把执行委托给 workflow。Gateway 读取 Patchouli 话题上下文时只通过 `GlobalSystemBus` 公共 route，不持有 Patchouli Runtime 或 Service。

## 4. 两种入口模式

`ACTIVE_CHAT` 允许命令短路，也允许普通决策。命令一旦命中，无论执行成功、被拒绝还是尚未实现，都会形成 `GatewayCommandOutcome`，不再进入话题、检索或 Alice 链路。

`PASSIVE_MEMORY` 禁止系统命令，只能形成 `GatewayDecisionOutcome`。它复用同一套话题与查询分析能力，但被动摄入的 buffer、去重、outbox、Patchouli submit 和降级响应属于 [System Passive Ingress](../system/passive-ingress.md)，不属于 Gateway。

完整跨子系统语义见[子系统公共契约](../contracts/subsystem-contracts.md)和[公开路由与事件](../contracts/routes-and-events.md)。

## 5. 当前设计文档

- [固定工作流](./workflow.md)：拓扑、状态提交、终态投影、deadline、取消与 fallback；
- [话题与查询分析](./analysis.md)：两阶段上下文、Topic Router、第一代 Resolver 与技术债；
- [全局命令](./commands.md)：Registry、Parser、Dispatcher、权限和命令短路。

原 `docs/engines/gateway.md` 与 `docs/mod/` 中的 Gateway 实施稿已经完成审计并移入 Archive；复合意图计划则进入 Plans。它们保留了演化过程和未落地设想，但不再占用当前入口；逐篇结论见 [`docs/mod` 迁移记录](../archive/plans/documentation-migration-audit-docs-mod.md)，当前事实以代码、测试、本目录文档和 Contracts 为准。
