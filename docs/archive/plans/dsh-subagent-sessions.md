---
title: dsh 子 Agent 会话不再独立导入，交回的消息进入父会话的一轮
status: archived
archived_at: 2026-11-02
implemented_by: 提交 4a16b46a（分支 fix/dsh-subagent-sessions）
superseded_by: docs/system/import-bus.md
owner: project
target: v0.7.3 之前（版本之外的工作）
scope: dsh-subagent-sessions-and-subagent-messages-in-turn-material
related_docs:
  - docs/archive/plans/dsh-plugin-integration.md
  - docs/system/import-bus.md
  - docs/contracts/mcp.md
  - docs/contracts/routes-and-events.md
  - docs/architecture/data-model.md
  - docs/patchouli/generation.md
  - docs/help/deepseek-harness-plugin.md
code_paths:
  - integrations/deepseek-harness/
  - src/hivememory/core/models/interaction.py
  - src/hivememory/system/services/import_bus/models.py
  - src/hivememory/engines/perception/material.py
  - src/hivememory/i18n/prompts.py
  - src/hivememory/patchouli/control/interaction_submission.py
  - tests/unit/connectors/test_deepseek_harness_plugin.py
updates:
  - docs/system/import-bus.md
  - docs/contracts/routes-and-events.md
  - docs/contracts/mcp.md
  - docs/contracts/subsystem-contracts.md
  - docs/architecture/data-model.md
  - docs/patchouli/generation.md
  - docs/help/deepseek-harness-plugin.md
  - docs/ROADMAP.md
  - docs/plans/README.md
last_reviewed: 2026-11-02
---

# dsh 子 Agent 会话不再独立导入，交回的消息进入父会话的一轮

> 状态：Archived（2026-11-02，实现、测试与离线回放完成，事实文档收尾后归档；owner 决定此时收尾，派生子 Agent 的调研在日常使用中验收）。当前事实见 [Import Bus](../../system/import-bus.md#93-轮次的切分与映射)第 9.3 节、[数据模型](../../architecture/data-model.md#49-交互记录turnevent-与-interactionpayload)第 4.9 节与[记忆生成](../../patchouli/generation.md)。以下为实施时的计划原文。原状态：Active（2026-11-02 建立并开始实施）。owner 决定（2026-11-02）：dsh 子 Agent 的会话独立导入属于实现漂移，改为只导入顶层会话，子 Agent 的内部过程不进入生成材料；子 Agent 交回的最终输出保留在父会话这一轮的材料中，以交互记录中新增的一种事件表达，来源只取 dsh 的结算通知。

## 1. 背景和问题

dsh 插件接入（PR #158）把子 Agent 的会话作为独立的外部会话导入，子 Agent 中的写入落在它自己的一轮。这在[接入计划](./dsh-plugin-integration.md)第 8 节记为候选设计，同一处记了另一种做法：与 Claude Code 相同，只导入顶层会话。

owner 在 2026-11-01 与 11-02 经 dsh 做了两次仓库调研，主 Agent 各派生 5 个子 Agent，观察到话题被挤掉与记忆碎片化。核对插件日志、后端日志与 dsh 本地会话记录，得到以下事实：

- **每个子 Agent 占一个新话题**：子会话的第一轮没有上文，Gateway 5 次都判为新话题。驻留话题上限 `perception.engine.max_resident_topics` 为 5，新建话题会挤掉最近最少访问的话题并提前结算。第一次调研挤掉了另一个 dsh 会话刚建立的话题，提前结算出 3 条记忆；两次调研中主会话一轮建立话题时，都挤掉了先完成的子 Agent 话题。
- **碎片来自子 Agent 轨迹的被动提取**：两次调研中子 Agent 都有记忆工具，但都没有调用；写入全部来自主会话（第一次 4 条写入，第二次 7 条写入与 2 条更新；主 Agent 报告的"1+4"是第二次第一轮的 5 条写入）。每个子 Agent 话题单独结算，多条提取把一次深挖拆成细碎的记忆：检索子 Agent 的话题两次分别结算出 9 条与 8 条，粒度与主 Agent 的综述不一致，主 Agent 也不知道它们存在。
- **父会话的一轮里没有子 Agent 交回的结果**：子 Agent 交回父会话的消息在 dsh 中是来源为 `agent-message` 与 `subagent-settled` 的 `user/message`，插件只读来源为 `user` 的消息，二者都被当作 harness 注入的上下文丢弃。子 Agent 的发现进入记忆的唯一途径因此是子会话的独立导入。主话题自己的被动提取表现正常：第二次调研中它新建的 6 条随后都并入了主 Agent 的写入。

owner 判断（2026-11-02）当前做法是实现漂移：子 Agent 只是临时线程，要记录的内容应当自己调用记忆写入，它的内部过程不进入生成材料，Alice 的 CALL 也是如此；子 Agent 交回的最终输出通常总结了自己做了什么，信息足够，应当保留在父会话这一轮的材料中。

## 2. 目标与非目标

目标：

- dsh 子 Agent 的会话不再导入，不再建立话题；子 Agent 的内部过程不进入生成材料。
- 子 Agent 交回父会话的消息进入父会话所在一轮的生成材料，全文保留，按发生顺序排列，不进入 Gateway 的话题路由。
- 子 Agent 中的写入与更新落在顶层会话的一轮，与这一轮回流时所在的话题相同。

非目标（讨论中发现，本计划不处理）：

- plugin 模式下主 Agent 感知子 Agent 的写入（Alice 在 CALL 收尾时把子帧收到 ACK 的 alias 放入 CALL 产物）：两次调研中子 Agent 没有写入，先观察本计划完成后的使用再设计。
- Claude Code 子代理的结果作为 Task 调用的结局导入，生成材料中只保留 800 字符；是否改为本计划的子 Agent 消息，留到 Claude Code 一侧出现同类问题时讨论。
- Alice 的 CALL 在生成材料中不渲染，子帧的回复因此不进入材料，与本计划保留子 Agent 消息的做法不同；本计划不改 Alice。
- 驻留话题上限 5 由所有 harness 的会话共用。本计划之后子 Agent 不再占用话题，是否调整上限另议。
- 已经由子会话导入生成的交互、话题与记忆不迁移、不清理。

## 3. 当前状态和缺口

dsh 一侧的事实，依据桌面端自带的 dsh `0.2.1-rc.5` 源码（`dsh-subagent`、`dsh-tool-subagent`）与 owner 的会话记录核对（2026-11-02）：

- 进程内 spawn 与 fork 的子 Agent 都有自己的持久会话，会话头带 `origin: "subagent"`、`parentSession`（直接父会话）、`delegationDepth` 与 `isSeeded`。顶层会话的会话头没有 `origin` 与 `parentSession`，`delegationDepth` 为 0。委托深度默认为 1，可以配置得更深。
- `subagent` 工具立即返回 `started subagent <id>`，子 Agent 在后台运行。子 Agent 交回父会话的消息有两种，都是 role 为 user 的 `user/message`，来源中带 `senderSessionId`：
  - `agent-message`（`form: relay`）：子 Agent 调用 `send_message` 发给父 Agent 的消息，正文以 `Agent <id> sent a message:` 开头。dsh 在每个可继续的子 Agent 的任务末尾附上指引，要求它结束前用 `send_message` 把自成一体的结果发给父 Agent。
  - `subagent-settled`（`form: notice`）：子 Agent 结算后 dsh 发出的通知，依次是结局一行（完成、被停止、失败等），子 Agent 最后输出中的非空文本（`Its closing message:`，没有时为 `It left no closing message.`），以及有时附带的结构化结果与诊断。
- 父 Agent 忙时，这两种消息并入它正在进行的一轮；父 Agent 空闲时，消息开启新的一轮，这一轮没有人发出的消息。
- owner 两次调研的 10 个子 Agent，两种消息都落在派生它们的那一轮中。其中 9 个的两种消息是同一份报告，1 个的 `send_message` 只是摘要、结算通知才是完整报告；单条消息 1.7 万到 2.7 万字符。

HiveMemory 一侧：

- **插件**：[`turns.js`](../../../integrations/deepseek-harness/lib/turns.js) 只把来源为 `user` 的消息拼入用户消息，其他来源一律不读；[`index.js`](../../../integrations/deepseek-harness/index.js) 对每个会话（含子会话）登记并同步；[`tools.js`](../../../integrations/deepseek-harness/lib/tools.js) 以调用方 Agent 自己的会话构造轮次信息，最后一轮已经结束时拒绝。
- **交互记录**（[数据模型](../../architecture/data-model.md)第 4.9 节）：`TurnEvent` 只有 `user_message`、`assistant_message`、`thought`、`tool_call` 四种，一轮只有一条用户消息；导入事件只接受后三种。
- **生成材料**（[记忆生成](../../patchouli/generation.md)）：用户消息与助手正文全文渲染，外部工具的结果按 `max_result_chars`（默认 800）截断，CALL 不渲染。Gateway 的话题路由读取用户消息全文，请求时限 8 秒，没有输入上限。

子 Agent 交回的消息因此没有合适的位置：并入用户消息，话题路由要读取调研轮的全部报告（一次 9 万到 12 万字符）；作为委托调用的结局，会被截到 800 字符。owner 选择（2026-11-02）在交互记录中新增一种事件。

## 4. 设计

注明日期的是 owner 的决定；其余是候选设计，不是 owner 决定，可在计划评审时调整。

### 4.1 子 Agent 消息事件

- `TurnEvent` 新增 kind `subagent_message`（名字为候选）：被委托的子 Agent 交回当前 Agent 的一条消息，`content` 是消息正文，不带其他字段。一轮可以有多条，与其他事件一起按序号排列；一轮仍只有一条用户消息。
- 导入事件接受 `subagent_message`，字段规则由 `TurnEvent` 执行，与其他事件相同。
- 生成材料把它渲染为一条 `[Subagent]: …`，全文保留，不受 `max_result_chars` 截断：它是 Agent 交回的结论，不是外部工具的原始输出。感知的 token 统计与 Page Folding 使用同一份渲染，因此一并计入。
- 三种生成模式的提取提示词（中英文各一份）在材料格式中加入这一条：子 Agent 交回的结果与通知是子 Agent 的结论，按助手正文对待；它依据的工具观察不在材料中。
- 用户消息与回复正文不包含它：`turn_reply_text` 只取助手正文，Gateway 的路由输入与最近轮次因此不变。
- `InteractionSubmission` codec 升为 v7：载荷可以包含新的事件种类，沿用载荷形状变化时升版本的做法。
- 主动链路（Alice）不产生这种事件，会话重放与前端的会话视图不为它增加处理。

### 4.2 dsh 插件

- **识别子会话**：会话头 `origin` 为 `subagent` 的会话是子会话。
- **不导入子会话**（owner，2026-11-02）：Agent 创建与轮次结束的监听跳过子会话，不登记、不同步。
- **子 Agent 交回的消息**（owner，2026-11-02，保留子 Agent 的最终输出，只取结算通知）：顶层会话一轮中来源为 `subagent-settled` 的消息按出现位置写为 `subagent_message` 事件，正文按现有规则取文本（图片与文件写占位）。dsh 加上的结局一行原样保留，说明消息来自哪个子 Agent、以什么结局结束。
  - 子 Agent 运行中用 `send_message` 发来的消息（`agent-message`）不读取：结算通知已经带有子 Agent 最后的输出，两者都读会让同一份报告重复进入材料，并让调研轮的话题越过折叠阈值（第 10 节的离线回放）。
  - 其他注入的上下文（技能正文、时间、后台任务通知等）仍不读取。
- **由结算通知开启的一轮**：用户消息为空，事件中有这条子 Agent 消息，照常导入。owner 的记录中尚未出现这种轮次。
- **子 Agent 中的写入与更新**（owner，2026-11-02，落在顶层会话的一轮）：沿会话头的 `parentSession` 逐级读取父会话的会话头，找到顶层会话，以顶层会话最近的一轮构造轮次信息，与这一轮回流时的同名字段相同，服务端因此推导出同一个交互标识。
  - 最近一轮已经结束也照用（候选）：这一轮已经应用时，服务端直接返回它实际所在的话题；已经提交、仍在应用时以 `apply_pending` 交回，由 Agent 重试；尚未回流时为它落位，回流时沿用落点（[Import Bus](../../system/import-bus.md)第 4.7 节）。这与 Claude Code 后台子代理的写入落在会话当时的当前一轮相同。
  - 顶层会话自己的调用不变：工具总在一轮之中执行，最后一轮已经结束时仍然拒绝。
- 插件的模块说明与注释随之改写，写出规则本身，不引用文档。

### 4.3 不变的部分

- Import Bus 的认证、串行、落位与提交流程不变，只多接受一种事件。
- Gateway、Patchouli 的话题与提交队列、MCP 入口不变。
- Claude Code 连接器不变。

## 5. 受影响的系统、契约和数据

- **core**：`TurnEvent` 的事件种类与字段规则。
- **System**：导入事件的种类，公开 HTTP 请求体的 `events[].kind` 多一个取值。
- **Patchouli**：生成材料的渲染、三种模式的提示词、`InteractionSubmission` codec v7。
- **集成**：dsh 插件的轮次投影、监听与写入的轮次信息；跨语言一致性测试。
- **数据**：不迁移。已导入的子会话交互、话题与记忆保留。插件状态目录中已登记的子会话状态保留：owner 的状态目录中 10 个子会话都已确认，监听跳过子会话之后不会再同步它们。

## 6. 兼容与迁移

- 插件发出新的事件种类，服务端必须先更新：旧服务端以 422 拒绝含 `subagent_message` 的一轮，插件会写入拒绝记录并跳过这一轮。owner 的插件链接到仓库目录，切换到含本改动的分支后，先重启 HiveMemory，再重新打开 dsh；帮助文档中"插件与 HiveMemory 需要一起更新"写明这个顺序。
- codec 升版本只影响进程内队列，重启时队列已排空，不需要迁移。

## 7. 测试和验收

不变量与归属层：

| 不变量 | 归属层 |
|:---|:---|
| `subagent_message` 只带正文，带调用字段时拒绝；一轮可以有多条，序号规则不变；回复正文不包含它 | `tests/unit/core/models/test_interaction.py` |
| 生成材料按序全文渲染子 Agent 消息，不受单条工具结果上限截断 | `tests/unit/engines/perception/test_material.py` |
| codec v7 往返保留子 Agent 消息 | `tests/unit/patchouli/control/test_interaction_submission.py` |
| 导入的子 Agent 消息经封口、提交与感知进入话题的材料（一个代表性用例） | `tests/integration/system/test_import_bus_submission.py` |
| 结算通知按位置写为子 Agent 消息事件，用户消息不变；`send_message` 发来的消息与其他注入的上下文不读取；由结算通知开启的一轮用户消息为空 | 插件 `test/turns.test.js`（现有"子 Agent 结束通知不读取"的用例按新契约改写） |
| 子会话中的写入以顶层会话最近一轮（进行中或已结束）构造轮次信息，多级委托同样找到顶层会话；顶层会话的规则不变 | 插件 `test/tools.test.js` |
| 插件写出的事件种类都是导入接受的种类 | `tests/unit/connectors/test_deepseek_harness_plugin.py` |

监听跳过子会话的接线在 `index.js` 中，现有插件测试不覆盖入口接线，由下面的离线回放与日常使用验证。

```bash
cd integrations/deepseek-harness && npm test
pytest tests/unit/core/models tests/unit/engines/perception tests/unit/patchouli/control tests/unit/system/services/import_bus tests/unit/connectors -q
pytest tests/integration/system/test_import_bus_submission.py -q
```

验收：

- **离线回放**：以插件的投影读取 owner 两次调研的会话记录（主会话 `session-1b78a114…`、`session-4daa4db5…` 与各自的 5 个子会话）。子会话被识别并跳过；两个主会话的第一轮各有 5 条子 Agent 消息，导入请求通过服务端模型校验，渲染后的材料保留全部报告。记录这一轮材料的字符数与 token 估计，以及是否触发 Page Folding。
- **日常使用**：一次派生子 Agent 的调研只建立主会话的话题；子 Agent 中如有写入，落在主会话这一轮的话题。这项验证要等使用中自然发生，是否作为归档条件由 owner 决定。

## 8. 风险与已知限制

- **调研轮的材料变大**：一轮带 5 个子 Agent 时多出约 10 万到 12 万字符，会更早触发 Page Folding（阈值 131072 token），结算时一次提取的输入也更大。owner 的两次调研中话题合计约 11.9 万 token，仍在阈值以下（第 10 节）。
- **子 Agent 的内部过程不再进入任何材料**：读了哪些文件、工具输出了什么都不保留。子 Agent 没有写入、结束时的输出也没有提到的发现会丢失，这是 owner 接受的代价。
- **只用 `send_message` 交回的结果会丢失**：子 Agent 把完整结果只经 `send_message` 发给父 Agent、结束时只说"已完成"时，材料中只有这句话。owner 的记录中 10 个子 Agent 的结算通知都带完整报告。
- **写入可能落在之后的一轮**：子 Agent 在顶层会话这一轮结束、用户又发出下一条消息之后才写入时，写入落在那时最近的一轮，与 Claude Code 后台子代理相同。
- **依赖 dsh 会话头**：识别与向上查找依赖 dsh 会话头的 `origin` 与 `parentSession`，跨版本可能变化；插件限定在 dsh 0.2.x。
- **由结算通知开启的一轮**：父 Agent 空闲时，子 Agent 的结算通知开启新的一轮。这一轮的用户消息为空，话题路由只能依据最近几轮。

## 9. 完成时必须更新的当前文档

全部已更新（2026-11-02）：

- [Import Bus](../../system/import-bus.md)：第 9.3 节的用户消息与子 Agent 两条，加入子 Agent 消息事件；第 11 节已知限制中 dsh 子会话独立导入一条；
- [路由与事件](../../contracts/routes-and-events.md)：导入请求体的 `events[].kind`；
- [MCP 记忆访问](../../contracts/mcp.md)：第 6 节 dsh 插件写入与更新的轮次信息，第 8 节已知限制；
- [子系统契约](../../contracts/subsystem-contracts.md)：Interaction codec v7；
- [数据模型](../../architecture/data-model.md)：第 4.9 节的事件种类表；
- [记忆生成](../../patchouli/generation.md)：材料渲染表与提示词的材料格式说明；
- [接入 DeepSeek Harness](../../help/deepseek-harness-plugin.md)：第 7 节哪些会话会导入、子 Agent 中的写入，第 9 节服务端与插件的更新顺序；
- [ROADMAP](../../ROADMAP.md)、[Plans](../../plans/README.md) 与[归档计划索引](./README.md)。

## 10. 实施记录

**实现**（2026-11-02，分支 `fix/dsh-subagent-sessions`，尚未提交审查）：

- 服务端：`TurnEvent` 新增 `subagent_message`，只带原文；导入事件接受这一种类；生成材料渲染为全文的 `[Subagent]: …`；三种生成模式的提取提示词（中英文）加入这一条；`InteractionSubmission` codec 升为 v7。
- 插件：`isSubagentSession` 按会话头识别子会话；两个监听跳过子会话；顶层会话中来源为 `subagent-settled` 的结算通知按位置写为 `subagent_message` 事件，`send_message` 发来的消息不读取；子会话中的写入沿 `parentSession` 找到顶层会话，取它最近的一轮。
- 测试按第 7 节写入归属层的现有文件；插件中"子 Agent 结束通知不读取"的旧用例按新契约改写为两条用例（结算通知按位置写为事件、`send_message` 发来的消息不读取，由结算通知开启的一轮用户消息为空）。去掉材料渲染、子会话写入的分支或子 Agent 消息的投影时，对应用例失败（已逐一核对）。

**验证**：`npm test`（40 通过）；`ruff check .`、`black --check .`、`mypy`（444 个文件）通过；`pytest tests/unit tests/integration -m "not live_llm and not e2e and not slow" -n auto --cov=hivememory --cov-fail-under=85`：3178 通过，覆盖率 94.63%，警告只有 starlette 测试客户端的第三方弃用提示。

**离线回放**（第 7 节，2026-11-02）：以插件的投影读取 owner 两次调研的持久会话记录，服务端 `ImportTurnRequest` 校验全部通过。最初的实现两种消息都读，回放后改为只读结算通知（owner，2026-11-02）；下表同时列出两种做法与都不读时的话题合计。

| 主会话 | 子会话 | 第一轮子 Agent 消息 | 第一轮材料 | 话题合计（只读结算通知 / 两种都读 / 都不读） |
|:---|:---:|:---:|:---:|:---:|
| `session-1b78a114…`（2 轮） | 5 个，全部识别并跳过 | 5 条，115571 字符 | 428368 字符，约 110723 token | 约 119044 / 147870 / 90126 token |
| `session-4daa4db5…`（3 轮） | 5 个，全部识别并跳过 | 5 条，95632 字符 | 346402 字符，约 97211 token | 约 119691 / 142073 / 90960 token |

- 第一轮材料的其余部分主要是主 Agent 的思考（9 万到 14 万字符）与工具调用（`write`、`edit` 的完整参数，按上限截断的读取结果等）。
- 两种都读时，两个话题都超过折叠阈值 131072 token；话题中的块多于保留数（2）时第一轮会被折叠（第二个会话的第三轮之后即是如此），按 Page Folding 的现有语义，被折叠的轮次只以接力摘要进入结算时的生成。只读结算通知时，两个话题都低于阈值。
- 10 个子 Agent 的结算通知都带完整报告：9 个与 `send_message` 的内容相同，1 个的 `send_message` 只是摘要。

**收尾**（2026-11-02）：自查代码改动，没有会改变公共设计的问题；第 9 节的当前文档已更新，Interaction codec 版本同时在[身份与授权](../../workspace/identity-and-authorization.md)与[数据模型](../../architecture/data-model.md)中改为 v7。日常使用验收（第 7 节）在归档后进行。
