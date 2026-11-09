---
title: Claude Code 唤醒后的续作导入
status: archived
archived_at: 2026-11-08
implemented_by: PR #203（fix/claude-code-wake-turns）
superseded_by: docs/system/import-bus.md
owner: project
target: v1.0 范围的 plugin 模式接入（不绑定具体版本号，owner 2026-11-04）
scope: claude-code-non-human-wake-turns-and-empty-message-topic-continuation
related_docs:
  - docs/system/import-bus.md
  - docs/gateway/analysis.md
  - docs/contracts/mcp.md
  - docs/help/claude-code-plugin.md
  - docs/ideas/task-process-table-and-registration-entry.md
  - docs/todo/cross-conversation-topic-placement.md
code_paths:
  - integrations/hivememory/connector/claude_code/
  - integrations/hivememory/connector/core/sync.py
  - src/hivememory/gateway/workflow/topology.py
  - tests/unit/connectors/
  - tests/unit/gateway/test_workflow.py
updates:
  - docs/system/import-bus.md
  - docs/gateway/analysis.md
  - docs/contracts/mcp.md
  - docs/help/claude-code-plugin.md
  - docs/ideas/task-process-table-and-registration-entry.md
last_reviewed: 2026-11-08
---

# Claude Code 唤醒后的续作导入

> 状态：Archived（2026-11-08 建立并实施，分支 `fix/claude-code-wake-turns`；当前事实见 [Import Bus](../../system/import-bus.md#83-轮次的切分与映射)第 8.3 节与 [Gateway 分析](../../gateway/analysis.md#3-topic-router只做选择)第 3 节，实施记录见第 10 节）。owner 决定（2026-11-08）：
>
> - 上一轮结束后，agent 被非人发起的消息唤醒时，开启新的一轮。
> - 这类轮次的用户消息为空时，服务端不交给 Gateway 的模型判断话题，直接续接上一轮所在的话题（方案 A）。没有采用方案 B，即等后台任务都交回之后再一起导入。
> - 后台通知（`task_notification`）：用户消息为空，通知本身不写入。
> - 子 Agent 交回（`peer`）：写为 `subagent_message` 事件，与 dsh 相同。
> - 派生会话（`sdk`）：作为独立会话导入，交接说明去掉 `<system-reminder>` 后作为用户消息。
>
> 标为“候选设计”的内容不是 owner 决定，可在计划评审时调整。

## 1. 背景和问题

候选清单中的 H-01，原记录见[任务进程 Idea](../../ideas/task-process-table-and-registration-entry.md#26-被动请求q-6)第 2.6 节，当时“按代码推断成立，未实测”。2026-11-08 已经实测确认。

**机制**：Claude Code 连接器的读取器只把 `turnOrigin` 为 `human` 的用户条目当作一轮的开始（`connector/claude_code/reader.py` 的 `_is_human_prompt`）。

1. agent 回复完、停下来时，`Stop` 为这一轮写结束记录，同步随即提交这一轮，游标前进到这一轮的末尾。
2. 之后，后台任务的完成通知、子 Agent 交回的报告或宿主注入的消息唤醒 agent 续作。
3. 续作写在游标之后，但前面没有一条人类消息开头。读取器把这些条目归入“游标之前、已经提交过的轮次”，结果它们永远不会被提交。
4. 续作结束时的 `Stop` 在连接器日志中记为 `ignored Stop: no unsubmitted turn after cursor`。

**实例**：2026-11-08 本机会话中，dsh 插件迁移那一轮在 21:57:38 导入；10 秒后，测试完成的通知唤醒 agent，agent 随后写下的工作总结没有进入 HiveMemory。

**规模**：本机 20 份会话记录中有三种开启运行的非人消息。它们都在 agent 空闲时到达，之后的工作全部丢失：

| `turnOrigin` | 内容 | 次数 | 丢失的工具调用 | 丢失的正文段 |
|:---|:---|---:|---:|---:|
| `task_notification` | 后台命令完成、Monitor 事件 | 74 | 544 | 148 |
| `peer` | 后台子 Agent 交回报告，或其他会话发来的消息（条目标为 `isMeta`） | 42 | 145 | 84 |
| `sdk` | 宿主注入的消息：派生会话的第一条交接说明（8 次），以及之后的宿主事件，如 side session 的汇报、CI 监视事件（11 次） | 19 | 713 | 126 |

**`turnOrigin` 的含义**：只有开启一次模型运行的条目才带 `turnOrigin`。agent 正在工作时插入的同类消息不带这个字段，只有 `origin.kind`（例如 35 条 `task-notification`），并入正在进行的一轮，本来就不会丢。

**其他 harness 的做法**：

- dsh：父 Agent 空闲时，子 Agent 的结算通知开启新的一轮，用户消息为空，交回内容写为 `subagent_message` 事件（owner，2026-11-02）。
- pi：每一段运行都算一轮。
- Claude Code 是唯一按“人发出的消息”切分轮次的连接器。

**为什么不能直接交给 Gateway 路由**：Gateway 的话题路由提示词只看“用户输入”，空输入时模型没有依据，很可能每次都返回 `NEW_TOPIC`。后台通知与 Monitor 事件因此会各开一个话题，造成不连续和话题池容量问题。这与 dsh 当时把子 Agent 的内容单独送去路由时遇到的情况相同，也是选择方案 A 的原因。

## 2. 目标与非目标

目标：

- 被 `task_notification`、`peer`、`sdk` 唤醒之后的工作，作为独立的一轮回流 Import Bus；写入时补上的 `turn` 与回流时属于同一轮。
- 用户消息为空的一轮不经模型路由，续接上一轮所在的话题。这条规则在服务端，dsh 与 pi 中用户消息为空的轮次同样适用。
- 子 Agent 交回的报告作为 `subagent_message` 进入生成材料。
- 派生会话作为独立会话导入。

非目标：

- 回填已经丢失的续作。升级之后，每份会话记录至多补交游标之后的那一段（第 6 节）。
- 派生会话第一轮的跨会话话题路由：它和其他会话的第一轮一样，由 Gateway 按交接说明判断，相关问题见 [Todo](../../todo/cross-conversation-topic-placement.md)。
- 把一次 Monitor 产生的多次事件合并为一轮。
- Import Bus 的请求格式与 MCP 契约：空用户消息与 `subagent_message` 都已在现有契约内。
- dsh 插件与 pi 扩展的代码不改动，它们从服务端规则自然受益。

## 3. 当前状态和缺口

- **读取器**：`_is_human_prompt` 只认 `turnOrigin == "human"`，并且 `isMeta` 条目整体跳过，所以 `peer` 条目连事件都不会产生。`last_turn_id`（`Stop` 写结束记录时用）只扫描人类消息；`turn_id_at`（游标校验）只认人类消息。
- **hook**：`PreToolUse` 以读取器切出的最后一轮构造 `turn`，切分规则改了它就跟着改；`Stop` 与 `StopFailure` 通过 `last_turn_id` 找轮次。
- **同步**：一轮在提交时游标前进到它的末尾，读取器在游标之前切出的轮次只作为最近上文。
- **Gateway**：话题路由失败时，`_continue_previous_topic`（`gateway/workflow/topology.py`）续接上一轮的话题，前提是那个话题仍在候选中，否则以 `NEW_TOPIC` 兜底。用户消息为空时目前仍然调用模型。
- **Import Bus**：上一轮所在的话题，来自最近一轮的外部标识（写入时登记的话题，或它的交互进度）。导入请求允许空用户消息，事件种类已有 `subagent_message`。

## 4. 设计

### 4.1 一轮的开始

一轮从以下两类条目之一开始，非主链路的条目（`isSidechain`）照旧跳过：

- 人类消息：与现在相同。
- 唤醒条目：`user` 条目，`turnOrigin` 为 `task_notification`、`peer` 或 `sdk`，内容不是工具结果。`isMeta` 的 `peer` 条目同样算唤醒条目。

规则：

- 唤醒条目开启的一轮，轮次标识取它的 `uuid`，外部会话标识的取法与人类消息相同。它之前的一轮以 `closed_by_next_turn` 结束。正常情况下 `Stop` 已为那一轮写了结束记录，结局取已保存的记录。
- 只有开启运行的条目带 `turnOrigin`，所以“有 `turnOrigin`”就是“开启新的一轮”的判据，读取器不另外追踪 agent 是否空闲。
- agent 正在工作时插入的消息不带 `turnOrigin`，只有 `origin.kind`，并入正在进行的一轮（第 4.2 节）。

### 4.2 用户消息与事件

| 条目 | 用户消息 | 事件 |
|:---|:---|:---|
| `task_notification` 唤醒 | 空 | 不写入通知 |
| `peer` 唤醒 | 空 | 写一个 `subagent_message`，内容取 `origin.handbackReportText`，没有时取 `origin.body`，再没有时取消息正文（候选设计） |
| `sdk` 唤醒，且是这份会话记录中的第一轮 | 消息正文去掉 `<system-reminder>…</system-reminder>` | 无 |
| `sdk` 唤醒，之后的轮次（宿主事件） | 空 | 不写入（候选设计：这些是宿主的状态通知，与后台通知同样处理；它们是计划时新发现的，不在 owner 的决定之内） |
| 并入进行中一轮的 `peer` 消息（`origin.kind` 为 `peer`） | — | 在出现位置写一个 `subagent_message`，与 dsh 相同 |
| 并入进行中一轮的其他消息 | — | 与现在相同，不读取 |

`subagent_message` 不并入用户消息，原因与 dsh 相同：话题路由读取用户消息全文，而交回的报告可能很长。

### 4.3 hook 与游标

- `Stop`、`StopFailure` 为游标之后最后开启的一轮写结束记录。候选设计：`last_turn_id` 改为取读取器在游标之后切出的最后一轮，不再单独扫描人类消息，切分规则因此只维护一处。
- `turn_id_at` 同时认唤醒条目。它只在已知的轮次起点上被调用，校验游标时不需要知道 agent 是否空闲。
- `PreToolUse` 不改动：它取读取器切出的最后一轮，续作中的写入因此落在续作这一轮，服务端推导出与回流相同的交互标识。

### 4.4 服务端：空用户消息续接上一轮的话题

Gateway 的话题路由步骤中，用户消息去掉空白后为空时，不调用 Topic Router：

- 最近一轮带有已知话题，并且该话题仍在本次候选中，直接路由到这个话题，标题与摘要为空；
- 否则以 `NEW_TOPIC` 新建话题，标题与摘要为空，也不让模型为空消息起标题。

这条规则复用现有续接兜底的判定。它不算降级，步骤不标记为 fallback。导入时的路由与写入时确定话题都经过这一步，所以续作中的写入与回流落在同一个话题。

适用范围：dsh 中被结算通知开启的轮次、pi 中由其他扩展的消息开始的运行（用户消息同样为空），都会改为续接上一轮的话题。任务进程的用户消息不会为空，不受影响。

## 5. 受影响的系统、契约和数据

- **集成**：`connector/claude_code/reader.py`（切分、用户消息与事件、`last_turn_id`、`turn_id_at`）、`connector/claude_code/hooks.py`（只在 `last_turn_id` 的用法有变化时调整）。
- **服务端**：`gateway/workflow/topology.py` 的话题路由步骤。公开路由、事件、错误语义与导入请求格式都不变。
- **数据**：每份会话记录多出唤醒开启的轮次。这类轮次没有助手输出时，按现有规则跳过，不提交。

## 6. 兼容与迁移

- 升级之后第一次同步时，游标之后如果有续作，会作为新的一轮补交一次。更早丢失的续作在游标之前，只作为最近上文，不回填。
- 插件更新的方式与平常相同：`claude plugin update hivememory-connector@hivememory-local`，然后重开会话。服务端规则随后端重启生效；先更新插件、后重启后端时，空消息轮次在这段时间内仍由模型路由。
- 不迁移数据。

## 7. 测试和验收

| 不变量 | 归属层 |
|:---|:---|
| 三种唤醒条目开启新的一轮，轮次标识取它的 `uuid`；不带 `turnOrigin` 的插入消息并入进行中的一轮；各类条目的用户消息与 `subagent_message` 事件按第 4.2 节；`turn_id_at` 认唤醒条目 | `tests/unit/connectors/test_claude_code_reader.py` |
| `Stop` 为唤醒开启的一轮写结束记录；`PreToolUse` 在续作中补上的 `turn` 指向这一轮，与之后的导入是同一个交互标识 | `tests/unit/connectors/test_claude_code_hooks.py` |
| 上一轮在 `Stop` 时已经提交、游标已经前进之后，续作仍作为新的一轮提交 | `tests/unit/connectors/test_sync.py` |
| 用户消息为空时不调用 Topic Router：上一轮的话题仍在候选中就续接，否则新建无标题话题；非空消息照常路由 | `tests/unit/gateway/test_workflow.py` |
| 唤醒条目的识别与 Claude Code 实际写出的字段一致 | 读取器测试中以本机会话记录的结构构造样例（`tests/unit/connectors/transcripts.py`） |

```bash
pytest tests/unit/connectors tests/unit/gateway -q
pytest tests/integration/system/test_import_bus_submission.py -q
ruff check . && black --check .
```

本机验收（Claude Code 桌面端，本机 HiveMemory）：

- 让 agent 以后台方式运行一条命令，在完成通知唤醒后继续工作：连接器日志中出现这一段续作的 `confirmed`，后端日志显示它没有调用 Topic Router，落在上一轮的话题。
- 后台子 Agent 交回报告之后的续作，以 `subagent_message` 进入同一话题的材料。
- 派生一个会话、在它自己完成工作之后：这段工作作为独立会话导入。

## 8. 风险与已知限制

- **依赖未公开的格式**：`turnOrigin` 的取值、`isMeta` 的 `peer` 条目，以及 `origin.handbackReportText` 都不是 Claude Code 公开承诺的格式。字段变化时唤醒条目不再被识别，退回到续作丢失的状态。
- **Monitor 的轮次很碎**：一次 Monitor 调用可能产生多次事件，每次唤醒都是一轮。这些轮次续接在同一话题中，不新增话题，但交互数会增加。
- **派生会话的话题可能归错**：它的第一轮只凭交接说明路由，可能落进别的会话的话题（[Todo](../../todo/cross-conversation-topic-placement.md)），也可能新开一个话题。
- **长时间后台任务**：上一轮的话题已经结算或被驱逐时，续作会新开一个无标题的话题。
- **字段缺失的版本**：唤醒条目在旧版 Claude Code 中可能不带 `turnOrigin`，这类续作仍然丢失。本机记录中带 `turnOrigin` 的唤醒最早来自 2.1.304。

## 9. 完成时必须更新的当前文档

- [Import Bus](../../system/import-bus.md)：
  - 第 8.2 节（`Stop` 取最后开启的一轮）、第 8.3 节（唤醒条目、用户消息与事件）；
  - 第 4.5 节，或在 Gateway 文档中说明空用户消息的路由；
  - 第 13 节已知限制。
- [Gateway 分析](../../gateway/analysis.md)第 3 节：空用户消息续接上一轮的话题。
- [MCP 记忆访问](../../contracts/mcp.md)第 6 节：`PreToolUse` 的当前一轮包括唤醒开启的一轮。
- [接入 Claude Code](../../help/claude-code-plugin.md)的使用中的行为。
- [任务进程 Idea](../../ideas/task-process-table-and-registration-entry.md)第 2.6 节中 plugin 模式的那一段：改为指向本计划。
- Plans 与 Archived Plans 索引。

## 10. 实施记录

分支 `fix/claude-code-wake-turns`（2026-11-08）：读取器 `connector/claude_code/reader.py`（开启一轮的判定 `_opens_turn`、用户消息、`peer` 写为 `subagent_message`）；Gateway 话题路由步骤 `gateway/workflow/topology.py`（空用户消息续接上一轮的话题，与路由失败的兜底共用 `_previous_topic_or_new`）；测试按第 7 节的归属层补充；第 9 节列出的当前文档。`hooks.py` 与 `sync.py` 不需要改动。

**实施中对设计的调整**：

- 第 4.3 节的候选设计（`last_turn_id` 改为取读取器切出的最后一轮）没有采用：`Stop` 同步运行，游标之后积压大量条目时完整切分太慢，`last_turn_id` 仍只解析用户条目的行。它与切分、`turn_id_at` 共用 `_opens_turn` 这一个判定，切分规则仍只维护一处。
- 第 4.2 节中并入进行中一轮的 `peer` 消息：本机记录中的 42 条 `peer` 条目都带 `turnOrigin`、都是唤醒条目，没有观察到运行中插入的情形。读取器对不开启新轮的 `peer` 条目同样写出 `subagent_message`，取报告先于 `isMeta` 的跳过判定，与开启新轮时的处理一致。
- 第 4.2 节的候选设计按原样实现：`peer` 的报告依次取 `origin.handbackReportText`、`origin.body`、消息正文；派生会话之后的宿主事件用户消息为空、不写入。

**验证**（2026-11-08，Windows 原生）：

- `pytest tests/unit/connectors -q`：89 passed；`pytest tests/unit/gateway -q`：84 passed。
- 以本机 20 份会话记录检查新的读取器：`task_notification` 74 次、`peer` 42 次、`sdk` 19 次唤醒都切出了带助手输出的一轮，`peer` 交回写出 42 个 `subagent_message`，8 份派生会话的第一轮以去掉 `<system-reminder>` 的交接说明作为用户消息；第 1 节的实例（测试完成通知唤醒后的工作总结）在本会话的会话记录中切成了独立的一轮。

**待验收**（第 7 节的本机验收，需要更新插件并重启后端）：后台命令完成后的续作 `confirmed` 且没有调用 Topic Router、落在上一轮的话题；子 Agent 交回之后的续作带 `subagent_message`；派生会话作为独立会话导入。
