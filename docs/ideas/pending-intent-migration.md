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
last_reviewed: 2026-11-12
---

# 写入意图（PendingAtom）的后续方向

## 0. 文档性质

2026-11-12：第 2、3、6 节与第 4 节中写入句柄的部分经计划[写入句柄的可靠性](../archive/plans/write-intent-handle-reliability.md)实施（PR #221），这三节改为已实施的说明；第 4、8 节其余的部分仍是候选方向。

本文收集写入意图体系尚未决定的方向，不描述当前行为，也不代表排期。登记、派发、回读与结算的当前设计见 [读取视图与写入意图](../workspace/read-view-and-write-intents.md)、[PendingAtom](../alice/pending-atom.md)、[Patchouli 生成](../patchouli/generation.md)与 [MTP 契约](../contracts/mtp.md)。耐久性与幂等的成熟度、缺口和工作包由[耐久性与故障恢复](../governance/reliability/durability-and-recovery.md)与[幂等与重试](../governance/reliability/idempotency-and-retry.md)治理文档维护，本文只讨论设计路径。

## 1. 问题背景

PendingAtom 解决所有 Actor 共有的资源问题：Actor 明确提出 WRITE/UPDATE，而正式 Memory 由后台异步生成时，如何在物化前读回意图、在结算后定位 canonical 结果，并避免读写不一致。它不是 Alice 专属机制，外部 Actor 接入后会遇到同样的问题。

以下各方向共享三条设计约束：

- 登记是唯一的意图状态机，不给外部 Actor、生成侧或 adapter 复制状态机；
- 结算事件只是加速，终态必须能从权威的任务与领域结果核对，不能等订阅者收到通知才变正确；晚到或重复的结果不改写已经确定的终态；
- 知道 Pending 不授予目标内容的读取权，Pending 与 canonical 的可见性分别判断，不能经结算跳转泄露不可读目标的内容或元数据。

## 2. 结算对账与结果未知

**已实施**（2026-11-12，PR #221）：派发结局不明时 workspace 以同一意图重试已有的提交路由，提交路由按 `intent_id` 幂等，命中原任务；正常重启时登记写盘、读回的顺序保证结算事件有落点。owner 排除了按 `intent_id` 查询结局的只读路由，不让 workspace 因此依赖 Patchouli。仍然没有对账：重试后仍然不明或事件丢失的句柄保持 MATERIALIZING，直到保留期满。当前事实见[读取视图与写入意图](../workspace/read-view-and-write-intents.md#5-派发与结算)第 5 节，设计与取舍见[归档计划](../archive/plans/write-intent-handle-reliability.md)。

## 3. 句柄寿命、容量与过期

**已实施**（2026-11-12，PR #221）：句柄自登记起保留 7 天（`workspace.write_intents.retention_days`），不论状态，满期删除；读取登记中没有的 `draft_`/`rev_` 句柄得到 expired，说明句柄不存在或已过期。过期不是登记中的状态，`PendingAtomStatus.EXPIRED` 与编译器中终态写入意图的渲染分支已删除。当前事实见[读取视图与写入意图](../workspace/read-view-and-write-intents.md#31-写盘读回与保留期)第 3.1、4 节。

## 4. 持久化与跨重启

写入意图登记、未执行的生成工作与已结束任务的历史在正常关闭时写盘、下次启动读回（PR #213、PR #221），崩溃时一起丢失，ACK 不是耐久承诺。耐久性等级、权威记录与恢复入口由[耐久性与故障恢复](../governance/reliability/durability-and-recovery.md)维护。本文补充两条设计约束：接纳后的断连或关闭不撤销已接纳的工作；恢复时不能重新派发已经结算的意图。引入持久化需要单独规划迁移。

## 5. 可见性与 policy

2026-11-05 移入[身份与访问体系 Idea](./identity-and-access-model.md#写入意图的可见性与-policy)第 6 节，与资源可见性一起维护。

## 6. UPDATE 的基础版本

**已实施**（2026-11-12，PR #221）：UPDATE 登记时记下基础原子的内容版本，生成侧写回前重读时版本不同则不写回，以 `workspace.memory.update_base_changed` 失败，读取句柄时提示重新读取后再更新。当前事实见 [Patchouli 生成](../patchouli/generation.md#33-mode-c主动-update)第 3.3 节。

## 7. 外部 Actor 的提交与回读契约

**已关闭**（owner，2026-10-31）：本节原列的 wire、收据、幂等、归属与观察都是已有能力或已有决定，不构成待决问题，也不阻塞 v0.7.2 的 plugin 模式 WRITE 与 UPDATE。外部 Actor 与 Alice 使用同一组 actor 操作、同一份登记与读取视图：WRITE 与 UPDATE 产生 PendingAtom 句柄交还前台 actor，记忆生成完成后由结算事件触发重定向（[外部 Actor Idea](./external-actor-registration-and-runtime-access.md#v072plugin-模式开放-write-与-update)第 4 节的 v0.7.2 一步）；当前设计见[读取视图与写入意图](../workspace/read-view-and-write-intents.md)。本节最后版本见 commit `acf3ccd4`。

## 8. 生成材料

问题：主动生成的背景材料只取话题过去若干轮与 `state_summary`，当前一轮不参与（[Patchouli 生成](../patchouli/generation.md)第 3.2 节）。长时间执行中途的显式保存因此看不到本轮的上下文。

方向（owner，2026-10-13）：对话以流式方式同步到 Patchouli 之后，材料变为之前全部的 Interaction 记录加上当前轮已流式接收的消息，见[长时间运行 Agent 的 Turn 内上下文折叠](./long-running-agent-intra-turn-context-folding.md)开头的方向。由进程把当前一轮未闭合的执行记录作为快照随意图提交是另一条路径，未采用；Topic 继续组织生成材料；ConversationSession 可承载原始记录，由进程/入口交接到 Topic，不作为生成时反向读取的会话或 CPU 视图。全链路来源保存见 [Raw Evidence](./PatchouliPageFoldingRawEvidenceDesign.md)。

约束：

- 先取得资料契约所选时点的全部可用材料，再按生成预算编译，不让 adapter 截取最近几条代替领域读取；已折叠且未保存的原文无法恢复，原文保全归[全链路 Raw Evidence](./PatchouliPageFoldingRawEvidenceDesign.md)，有界视图归[短期记忆与上下文压缩](./short-term-memory-and-context-compaction.md)；
- Topic 缺失、越权、为空与读取失败分别表达，缺失不等同于空材料；
- 生成侧只提供材料约束，不决定是否创建空 Topic；资料身份、绑定、快照时点与保留/释放见[外部会话与 Topic 投影](./external-session-and-topic-projection.md)第 4.4 节与第 8 节；
- 意图接纳后物化失败不回滚已应用的交互，Topic 清理不影响已接纳任务依赖的资料。
