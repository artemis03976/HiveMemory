---
title: Conversation Compact Command
status: todo
owner: workspace-alice
scope: conversation-compact-command-for-cpu-context
related_docs:
  - docs/ideas/external-session-and-topic-projection.md
  - docs/ideas/task-process-table-and-registration-entry.md
  - docs/ideas/external-actor-registration-and-runtime-access.md
  - docs/ideas/PatchouliPageFoldingRawEvidenceDesign.md
  - docs/gateway/commands.md
  - docs/contracts/subsystem-contracts.md
last_reviewed: 2026-10-01
---

# 会话 `/compact` 指令

## 定位（owner，2026-10-01）

ConversationSession 落地后，前台对话与后台的记忆 Topic 管理是两项各自独立的工作：

- **compact 归属前台**：压缩的对象是 CPU 基于会话派生的上下文。用户发出的 `/compact` 指令，对外部 Actor 是触发其 harness 自带的压缩，对 Alice 则需要 Alice 自己实现压缩。
- **后台的记忆 Topic 管理继续使用 page folding**：主要面向 Topic 累计消息过长时，保证输入 Patchouli 记忆生成的资料不会导致上下文爆炸。这是另一个问题，候选设计见 [Page Folding Raw Evidence Idea](../ideas/PatchouliPageFoldingRawEvidenceDesign.md)，不在本 Todo 范围内。

本 Todo 由原“Topic `/compact` 系统指令接入”（`docs/todo/topic-compact-command-ingress.md`，重写前最后版本见 commit `74b5056`）重写。原问题——用户能从聊天前端用 `/compact` 压缩上下文——仍然有效，但形式发生了变化：压缩对象从 Patchouli 的 Topic 工作集改为 CPU 的会话上下文。原设计中经 Gateway dispatcher 执行命令、由前端传播 `current_topic_id`、调用 Patchouli Topic manual compact 的路径不再成立：dispatcher 已删除，“当前 Topic”概念将随前端回归会话模型取消。

## 问题与证据（2026-10-01 核对）

- 内置命令中没有 `/compact`。Gateway 只解析命令、不执行，内置命令暂时不可用；命令在任务进程中何时、由谁运行尚未决定（[任务进程 Idea](../ideas/task-process-table-and-registration-entry.md) Q-5a，[Gateway 全局命令](../gateway/commands.md)）。
- ConversationSession 尚未落地。Alice 每轮的历史目前来自 Topic 最近 5 个 block 的滑动窗口与 `state_summary`（`prompts/assembler.py`），还不存在独立的会话上下文。
- CPU 端口（`workspace.contracts.CPUPort`）只有执行入口 `execute`，没有压缩操作，也没有声明是否支持压缩的方式（[子系统公共契约](../contracts/subsystem-contracts.md)第 4 节）。
- Alice 没有会话内压缩，压缩算法的目标窗口约为 v0.7.1（[ROADMAP](../ROADMAP.md) 第 4.4.3 节）。外部 harness 经 controller 模式接入在 v0.7.1；HiveMemory 能否触发某个 harness 的压缩，取决于驱动它的协议或 SDK 是否提供这项能力。
- 会话操作中的压缩已决定由 CPU 负责，作用于 CPU 自己基于 Session 派生的上下文视图，Session 记录保持原样（[外部会话与 Topic 投影 Idea](../ideas/external-session-and-topic-projection.md#01-会话模型与-topic-池owner2026-09-28) 0.1）。

## 影响

- 用户目前没有压缩会话上下文的入口。v0.7.0 期间，长会话的上下文长度只能靠新建会话控制（外部会话 Idea 0.1 的分析）。
- 如果不区分前台与后台，`/compact` 容易被实现为压缩 Topic 工作集，混淆两种对象，也会让前台的压缩改动后台记忆生成的资料。

## 约束

- `/compact` 是确定性的系统控制消息：由 Gateway 解析，命中后短路，不进入检索、Actor 执行或记忆生成；`PASSIVE_MEMORY` 不得产生该命令。
- 压缩只作用于 CPU 的上下文视图，不改写 ConversationSession 记录，也不改写 Patchouli 的 Topic 及其 page folding 状态。
- 前端不得自行解析和执行 `/compact`；任务进程只经 CPU 端口与 CPU 交互，不出现针对具体 CPU 的分支。
- 调用方依赖结构化的 status 与 error code，不解析本地化 message。

## 依赖

- ConversationSession 与前端会话模型（外部会话与 Topic 投影方向，v0.7.0）；
- 命令的运行位置（命令系统接回，Q-5a）；
- Alice 的会话压缩算法（约 v0.7.1）；
- 外部 harness 的 controller 模式接入（v0.7.1）。

## 待决问题

只列选项，不代表倾向。

1. `/compact` 如何到达 CPU：
   - 经任务进程把命令交给该会话所用的 CPU；
   - 作为不建立进程的直接操作执行；
   - 其他。
2. CPU 端口如何表达压缩：
   - 在端口上增加压缩操作；
   - 在 CPU 描述中声明是否支持压缩，不支持时由进程返回稳定的拒绝结果；
   - 其他。
3. CPU 不支持压缩时，用户看到什么结果。
4. 压缩完成后前端如何感知：命令终态中的结构化 data，或其他方式。

## 完成条件

- 用户在会话中输入 `/compact` 后，当前会话所用 CPU 的上下文被压缩：Alice 使用自己的压缩实现，外部 Actor 触发其 harness 的压缩；
- 不支持压缩的 CPU 返回稳定的结构化结果，不伪装为成功；
- ConversationSession 记录、Topic 与 page folding 状态不因 `/compact` 改变；
- 命令短路与 `PASSIVE_MEMORY` 约束有测试守护；
- 落地时同步 [Gateway 全局命令](../gateway/commands.md)、[子系统公共契约](../contracts/subsystem-contracts.md)等事实文档。
