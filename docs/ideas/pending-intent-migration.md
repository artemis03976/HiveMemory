---
title: 写入意图（PendingAtom）的后续方向
status: idea
horizon: candidate
owner: project
scope: pending-intent-reconciliation-retention-visibility-and-external-contract
related_docs:
  - docs/workspace/README.md
  - docs/alice/pending-atom.md
  - docs/patchouli/generation.md
  - docs/governance/reliability/durability-and-recovery.md
  - docs/governance/reliability/idempotency-and-retry.md
  - docs/ideas/external-actor-registration-and-runtime-access.md
  - docs/ideas/external-session-and-topic-projection.md
  - docs/ideas/long-running-agent-intra-turn-context-folding.md
last_reviewed: 2026-10-19
---

# 写入意图（PendingAtom）的后续方向

## 0. 文档性质

本文收集写入意图体系尚未决定的方向，不描述当前行为，也不代表排期。登记、派发、回读与结算的当前设计见 [读取视图与写入意图](../workspace/read-view-and-write-intents.md)、[PendingAtom](../alice/pending-atom.md)、[Patchouli 生成](../patchouli/generation.md)与 [MTP 契约](../contracts/mtp.md)。耐久性与幂等的成熟度、缺口和工作包由[耐久性与故障恢复](../governance/reliability/durability-and-recovery.md)与[幂等与重试](../governance/reliability/idempotency-and-retry.md)治理文档维护，本文只讨论设计路径。

## 1. 问题背景

PendingAtom 解决所有 Actor 共有的资源问题：Actor 明确提出 WRITE/UPDATE，而正式 Memory 由后台异步生成时，如何在物化前读回意图、在结算后定位 canonical 结果，并避免读写不一致。它不是 Alice 专属机制，外部 Actor 接入后会遇到同样的问题。

以下各方向共享三条设计约束：

- 登记是唯一的意图状态机，不给外部 Actor、生成侧或 adapter 复制状态机；
- 结算事件只是加速，终态必须能从权威的任务与领域结果核对，不能等订阅者收到通知才变正确；晚到或重复的结果不改写已经确定的终态；
- 知道 Pending 不授予目标内容的读取权，Pending 与 canonical 的可见性分别判断，不能经结算跳转泄露不可读目标的内容或元数据。

## 2. 结算对账与结果未知

问题：登记只经 Patchouli 发布的结算事件推进终态。事件丢失，或派发时无法确认是否已被接纳（总线异常、接纳应答丢失），句柄会停在 MATERIALIZING：未被接纳的意图永远等不到结算，已被接纳的只能等事件。

可能路径：

- **按 intent_id 查询权威结果**：生成任务以 intent_id 为幂等身份，登记可以在回读时、派发结果未知之后或周期性地按 intent_id 查询任务终态并补齐。需要一条只读的查询路由，查询方不持有生成的内部状态；
- **结算游标与重放**：结算结果持久化后按游标重放补齐，依赖第 4 节的持久化。

影响：外部 Actor 在响应丢失后的查询（第 7 节）依赖同一机制。

## 3. 句柄寿命、容量与过期

问题：终态句柄保留到进程重启，登记随运行持续增长；`EXPIRED` 枚举保留在模型中，但不会产生。owner（2026-09-28）指出结算后句柄的生命周期需要重新设计。

EXPIRED 是设计中的永久终态，SETTLED、FAILED、CANCELLED 都可以迁入。它原先由 run 结束触发：上一轮已结束的意图在下一轮 run 以 completed 结束时标为 EXPIRED（见[写入意图登记归档计划](../archive/plans/v0.7.0-intent-registry-and-read-cache.md)）；登记与任务进程解耦、改为实时派发后，这个触发点不复存在，当前缺少的是转入 EXPIRED 的机制，而不是终态本身。

读取侧的现状（2026-10-18 核对）与下方约束有两处出入，过期机制需要一并决定：

- 读取视图把 EXPIRED 与 CANCELLED 都按不存在交付（`workspace/resolution/memory.py` 的 `_resolve_pending`），而引用解析结果的 `expired` 种类、MTP READ/RUN 与编译器中对应的分支和文案都已存在、只是走不到；
- 读取视图只把在途意图原样交给编译器，终态改由引用解析结果表达，因此编译器中终态写入意图的渲染分支（`pending_read_settled`、`pending_read_failed`、`pending_read_cancelled` 等）在生产中走不到；终态的交付形态确定后再决定这些分支的去留。

可能路径：结算后保留期满转为 EXPIRED；登记设容量上限，满时拒绝新登记，与生成队列的容量拒绝一致；或与第 4 节的持久化一起设计。

约束：

- discarded、failed、cancelled、expired 明确区分，过期可以解释，不伪装成不存在或仍在生成；
- 终态句柄不按缓存淘汰处理；未结算的意图不能因容量被静默丢弃。

## 4. 持久化与跨重启

登记、接纳与结算都只在进程内，重启后丢失，ACK 不是耐久承诺。耐久性等级、权威记录与恢复入口由[耐久性与故障恢复](../governance/reliability/durability-and-recovery.md)维护。本文补充两条设计约束：接纳后的断连或关闭不撤销已接纳的工作；恢复时不能重新派发已经结算的意图。引入持久化需要单独规划迁移。

## 5. 可见性与 policy

问题：意图不设 policy，同 Workspace 内持有别名即可回读（UPDATE 意图跟随基础原子的可读性）。别名由标题或内容片段加 4 位十六进制后缀组成，“只有写入者知道别名”是惯例，不是访问控制，真正的边界是 Workspace。`WriteFocus` 只有 content、reason、title，不携带目标 policy。

需要回答：

- WRITE 若能声明目标 policy，落库前的意图可见性是否跟随该 policy；
- 增加意图 policy 时，不要求外部构造 RuntimeScope，也不把整个 scope 相等重新变成权限条件；
- 读取正式原子时，是否提示调用者自己仍在进行的 UPDATE 修订；不提示时，Pending 正文不能静默叠加到已提交版本上。

## 6. UPDATE 的基础版本

问题：UPDATE 登记时解析可读的正式基础原子，生成侧构建任务时按 UUID 读取当时的最新版本；意图不携带登记时的基础版本，并发 UPDATE 不检测版本冲突。

可能路径：

- 意图记录登记时的基础版本，生成时不一致则显式冲突，结算为失败并说明原因；
- 按最新版本合并，在 changelog 中记录基础已经漂移。

版本冲突的判定仍归 canonical 更新规则；空交互不免除目标与授权校验。

## 7. 外部 Actor 的提交与回读契约

[外部 Actor](./external-actor-registration-and-runtime-access.md)需要在 Alice 之外使用同一套登记与读取，尚需决定：

- **wire**：提交、回读与结算状态的协议投影，core `ReferenceResolution` 的外部表示，以及缺少目标身份的旧结算载荷如何兼容；
- **收据**：区分已登记、已被生成队列接纳与 canonical 已产生，不返回模糊的整体 success；接纳失败后的回读与重试契约；
- **幂等**：客户端幂等键与 intent_id 的映射；同键同载荷定位同一提交，同键异载荷返回 conflict，响应丢失后可以查询（依赖第 2 节）；重复提交不产生第二个 canonical；
- **归属**：提交的 scope 必须同时匹配访问 context 与已登记意图的归属，不能借提交方法挪用他人的 intent_id；外部不依赖 RuntimeScope；
- **观察**：记忆任务的观察（root API）不自动授予意图内容或最终 Memory 的读取权。

## 8. 生成材料

问题：主动生成的背景材料只取话题过去若干轮与 `state_summary`，当前一轮不参与（[Patchouli 生成](../patchouli/generation.md)第 3.2 节）。长时间执行中途的显式保存因此看不到本轮的上下文。

方向（owner，2026-10-13）：对话以流式方式同步到 Patchouli 之后，材料变为之前全部的 Interaction 记录加上当前轮已流式接收的消息，见[长时间运行 Agent 的 Turn 内上下文折叠](./long-running-agent-intra-turn-context-folding.md)开头的方向。由进程把当前一轮未闭合的执行记录作为快照随意图提交是另一条路径，未采用；Topic 继续组织生成材料；ConversationSession 可承载原始记录，由进程/入口交接到 Topic，不作为生成时反向读取的会话或 CPU 视图。全链路来源保存见 [Raw Evidence](./PatchouliPageFoldingRawEvidenceDesign.md)。

约束：

- 先取得资料契约所选时点的全部可用材料，再按生成预算编译，不让 adapter 截取最近几条代替领域读取；已折叠且未保存的原文无法恢复，原文保全归[全链路 Raw Evidence](./PatchouliPageFoldingRawEvidenceDesign.md)，有界视图归[短期记忆与上下文压缩](./short-term-memory-and-context-compaction.md)；
- Topic 缺失、越权、为空与读取失败分别表达，缺失不等同于空材料；
- 生成侧只提供材料约束，不决定是否创建空 Topic；资料身份、绑定、快照时点与保留/释放见[外部会话与 Topic 投影](./external-session-and-topic-projection.md)第 4.4 节与第 8 节；
- 意图接纳后物化失败不回滚已应用的交互，Topic 清理不影响已接纳任务依赖的资料。
