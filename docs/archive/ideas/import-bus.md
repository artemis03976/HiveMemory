---
title: Import Bus：只向网络输入的记忆来源
status: archived
archived_at: 2026-10-30
implemented_by: v0.7.1 Import Bus 重设计（archive/plans/v0.7.1-import-bus-redesign.md）
superseded_by: docs/system/import-bus.md
owner: project
scope: import-bus-follow-up-directions
code_paths:
  - src/hivememory/system/services/import_bus/
  - src/hivememory/server/routers/import_bus.py
  - integrations/claude-code/
related_docs:
  - docs/system/import-bus.md
  - docs/archive/plans/v0.7.1-import-bus-redesign.md
  - docs/ideas/fifth-generation-architecture.md
  - docs/ideas/external-actor-registration-and-runtime-access.md
  - docs/ideas/external-session-and-topic-projection.md
  - docs/ROADMAP.md
last_reviewed: 2026-10-30
---

# Import Bus：只向网络输入的记忆来源

> 已归档（2026-10-30，v0.7.1 收尾，owner 决定）。Import Bus 的当前事实见 [Import Bus](../../system/import-bus.md)。第 2 节的三个后续方向不再由本文跟踪，去向见文末“归档说明”；以下正文保留归档前的原文，链接已改为归档后的相对路径。

## 0. 文档性质

Import Bus 的重设计已于 2026-10-21 按[归档计划](../plans/v0.7.1-import-bus-redesign.md)实施：一次请求接收 harness 一轮已结束交互的完整记录，经统一认证网关认证，只做话题落位与价值信号，不检索、不返回记忆上下文，应用完成后才确认；Claude Code 连接器以插件形式提供。当前事实见 [Import Bus](../../system/import-bus.md)。本文只保留尚未实施的后续方向，不描述当前行为，也不代表排期；已决定并实施的问题（Topic 落位、价值信号、完整交互记录的接收范围、回流内容、生成视图、接入认证）随实施收口，决定与理由见归档计划。

**术语**（owner，2026-09-27）：“被动（Passive）”一词留给被动请求体系，即由条件触发创建任务进程的请求。原先称为“被动输入”的输入链路称 **Import Bus**；Gateway 的 `PASSIVE_MEMORY` 模式名保持不变。

## 1. owner 的前提

1. （2026-09-26）执行者可以任意替换之后，Import Bus 退化为只向网络输入：接收信息并转为记忆资产，不读取记忆，也不返回记忆上下文。2026-10-20 界定：为话题落位读取候选话题不在“读取记忆”之列。
2. （2026-09-26）两种接收方式：直接接收一份完整的交互记录；流式监听一个交互。两者经同一条提交路径进入记忆生成。第一种已实施，第二种推迟（前提 5）。
3. （2026-09-27、2026-09-28）Import Bus 不在核心全局拓扑上，作为独立功能演进。
4. （2026-09-27）外部 Actor 的 plugin 模式下，HiveMemory 以插件形式挂载在外部 harness 中，对话经 Import Bus 回流，不接管对话（[外部 Actor Idea](../../ideas/external-actor-registration-and-runtime-access.md#11-两种接入模式owner2026-09-27) 1.1）。
5. （2026-10-20）先实现完整交互记录的接收：harness 的一轮交互结束后，由连接器把该轮的完整记录交给 Import Bus。流式监听推迟，与 controller 模式下 Patchouli 的流式接收（[外部会话 Idea](../../ideas/external-session-and-topic-projection.md)的流式同步）一并设计。
   - 理由（owner）：流式接入时，需要经 hook 先取得用户消息送入 Gateway 路由；harness 响应很快时，模型的首条消息可能在路由完成之前到达，产生时序问题，流式模式要完善还有许多问题。在轮次结束处拦截完整消息，路由可以照常进行。
   - 分析：Import Bus 不返回记忆上下文后，在用户消息到达时提前路由没有收益，只剩时序成本。harness 的 hook 一般只在用户提交、工具调用、轮次结束等节点触发，不逐 token 推送回复正文；同步 hook 会让每条消息多一次 Gateway 的模型调用，异步 hook 的请求到达顺序没有保证。

## 2. 后续方向

以下方向都不阻塞当前计划，按需要再展开。

- **流式监听**：前提 2 的第二种接收方式。需要先解决前提 5 的时序问题，并与 controller 模式下 Patchouli 的流式接收共用设计；接收契约（与 harness 无关的完整轮次、外部标识、同一外部会话内按序）应保持不变，流式只改变连接器到 Import Bus 的交付方式。
- **历史导入另设通道**：Import Bus 只接收实时或刚结束的交互（2026-10-20 决定），连接器不回填接入之前已经存在的会话内容。历史对话导入有历史发生时间、批次、去重、不重放进当前活跃话题等独立语义，见 [ROADMAP](../../ROADMAP.md) 第 4.5 节。
- **第二个 harness 的连接器**：连接器按“与 harness 无关的接收契约、公共骨架（游标、确认后前进、重发去重、首次见到的旧会话不回填）、每个 harness 的读取器”分层，接入第二个 harness 时把公共骨架从 Claude Code 插件中抽出共用。2026-10-20 的候选 harness 现状：

  | | Claude Code（已接入） | Codex | pi |
  |:---|:---|:---|:---|
  | 会话记录 | `~/.claude/projects/<cwd>/` 下的 JSONL | `~/.codex/sessions`，可由 `CODEX_HOME` 隔离 | `~/.pi/agent/sessions/` 下的 JSONL 会话树 |
  | 唤醒方式 | hooks：`Stop`、`StopFailure`、`SessionStart`、`SessionEnd` | 主线源码已有同形的 hooks，含 `Stop`、`SessionStart`、`SessionEnd`、`Interrupt`，输入带 `transcript_path` 与 `turn_id`；旧的 `notify` 在每轮结束时调用外部程序（[openai/codex](https://github.com/openai/codex) 主线，发行版支持情况未核对） | 没有命令式的外部 hook，有进程内的 TypeScript 扩展，可订阅的事件未核对 |

  会话记录的位置见[接口调研快照](../../research/harness-interface-survey-snapshot.md) B1。会话记录格式都不是公开承诺的接口，读取器需要容忍不认识的条目，并固定测试过的版本。只在云端运行、本地没有会话记录的 harness，关闭持久化的模式（例如 Codex 的 ephemeral、pi 的 no-session），以及无法解析的存储，需要其他来源（harness 自己推送事件或导出接口）；接收契约不变，只更换连接器。

## 归档说明（2026-10-30）

owner 决定在 v0.7.1 收尾时归档本文。第 2 节的方向此后不再由本文跟踪：

| 方向 | 去向 |
|:---|:---|
| 流式监听 | 不单独跟踪；需要时与 controller 模式下 Patchouli 的流式接收一并设计（[外部会话 Idea](../../ideas/external-session-and-topic-projection.md)的流式同步），前提 5 的时序分析供参考 |
| 历史导入另设通道 | [ROADMAP](../../ROADMAP.md) 第 4.5 节的 v0.7.2 冷启动与历史对话导入 |
| 第二个 harness 的连接器 | 不单独跟踪；接入第二个 harness 时参考第 2 节的分层与 2026-10-20 的候选现状，连接器的现有分层见 [Import Bus](../../system/import-bus.md#81-组成)第 8.1 节 |
