---
title: 短期记忆：话题的进展及其维护
status: idea
horizon: current
serves_version: v0.8.1
owner: patchouli-workspace-system
scope: v0.8.1-short-term-memory-topic-work-state-lifecycle-folding-and-push
code_paths:
  - src/hivememory/core/models/topic.py
  - src/hivememory/memory_library/stores.py
  - src/hivememory/memory_library/topic_log.py
  - src/hivememory/patchouli/services/perception.py
  - src/hivememory/patchouli/services/topic_working_set.py
  - src/hivememory/engines/perception/
  - src/hivememory/i18n/prompts.py
  - src/hivememory/workspace/capability/root/topic.py
  - src/hivememory/workspace/capability/actor/memory.py
  - src/hivememory/workspace/resolution/memory.py
  - src/hivememory/mcp_server/server.py
  - src/hivememory/engines/gateway/topic_router.py
  - integrations/hivememory/
related_current:
  - docs/patchouli/perception.md
  - docs/patchouli/generation.md
  - docs/alice/agent-runtime.md
  - docs/workspace/sessions.md
  - docs/workspace/capability.md
  - docs/system/import-bus.md
  - docs/contracts/mcp.md
related_ideas:
  - docs/ideas/memory-flow-short-to-mid-term.md
  - docs/ideas/memory-flow-write-path.md
  - docs/ideas/long-running-agent-intra-turn-context-folding.md
  - docs/ideas/external-session-and-topic-projection.md
  - docs/ideas/PatchouliPageFoldingRawEvidenceDesign.md
related_docs:
  - docs/research/agent-context-compaction-research.md
related_todos:
  - docs/todo/conversation-compact-command.md
last_reviewed: 2026-11-26
---

# 短期记忆：话题的进展及其维护

## 0. 文档性质

本文是 v0.8.1 的设计：短期记忆以话题为载体，话题的进展回答“进行到哪一步”；包括它由什么组成、什么时候写、话题的生命周期与存储、折叠，以及前台在什么时候用到它。正文是已定的设计，候选设计标“候选”；已经实现的部分以当前事实文档为准，本文只链接。

相邻的设计各在一处维护：一轮进行中的事件接收、封口与恢复见[长 Turn Idea](./long-running-agent-intra-turn-context-folding.md)；原始资料的保存见[全链路原始资料保存](./PatchouliPageFoldingRawEvidenceDesign.md)；沉淀时写什么、主动写入能拿到什么材料，属于短期到中期的过渡，见[写入链路](./memory-flow-write-path.md)与 [v0.8 总纲](./memory-flow-short-to-mid-term.md)；会话与话题的交接与路由见[外部会话 Idea](./external-session-and-topic-projection.md)。

## 1. 目标与原则

### 1.1 要解决的问题

- 长上下文的遗忘与注意力衰减是短期记忆的问题。各 harness 用会话压缩应对，但压缩会丢细节、让注意力漂移。
- 前台上下文不归 HiveMemory 管：plugin 模式下属于 harness，controller 模式下 Alice 有自己的会话压缩。两种模式共有的只有话题这一环，所以从话题入手。
- 话题若只被 Patchouli 用来生成记忆，对前台就没有直接用处：交互从会话流进话题，前台还要能把话题的进展读回来（第 7 节）。
- 后台的话题反而比前台的会话更不连贯：空闲结算会把一段连续的对话切成几个话题。日常实例 11 天中，77 个空闲结算的话题有 32 个在 2 小时内被同一会话续上，续上的 65 次里有 48 次开了新话题。

### 1.2 定位

- 短期记忆是一条机制链：前台上下文 → 会话压缩 → 话题 → 沉淀进中期，再从话题读回前台；链的各环与已经成立的跨层不变量见[系统架构概览](../architecture/overview.md#7-短期记忆话题是前台与中期之间由-hivememory-持有的一环)第 7 节。会话压缩属于 harness 与 Alice（第 8 节），本文设计的是话题这一环。
- 与中期同构：中期由整理过的记忆和作为证据的原始交互组成；短期由话题的进展（整理过、随交互推进而增长）和话题的交互记录（次级的完整记录）组成。
- 话题的进展回答“进行到哪一步”。前台的上下文按会话组织、会被压缩；话题按内容组织、原文完整。所以“先讨论 A、做了十几轮 B、再回到 A”时，前台丢掉或混在一起的东西，可以从 A 的话题里拿回来。
- 进展记录是索引：每一条指回原文，agent 可以照着它回查之前的交互。
- 话题提供信息，不控制前台上下文，与 plugin 模式下的被动地位相容。会话与话题不共享状态、互不控制，话题向前台的流动只是读出：这已是跨层不变量，见[系统架构概览](../architecture/overview.md#7-短期记忆话题是前台与中期之间由-hivememory-持有的一环)第 7 节。
- 话题的短期记忆不局限在轮次上：长 turn 下的短期记忆管理由本设计一并承担，写一条与折叠的切点都可以落在一轮之内。

### 1.3 原则

- 资料退出某个模型视图、原文是否继续保存、是否生成正式 Memory，是三个独立判断。话题也不等于“尚未写入 Memory 的队列”：主动 WRITE 可能已物化，相关交互仍留在话题中。
- 已整理的内容只往后追加，不被递推重写：多次整段重写会让细节漂移。
- 触发先用简单规则，规则定义在事件序列上，与事件什么时候到达无关。
- 话题可以是 agent 的一项工作，也可以是一段讨论或闲聊；进展的结构、触发与写法对 agent 工作流与对话流都成立，措辞不假设话题是工作：前者回答做到哪了，后者回答聊到哪了。

## 2. 组成与数据流

```text
会话（前台上下文；harness 或 Alice 压缩，第 8 节）
  │ 交互（plugin 模式在一轮结束后导入；一轮之内的接收见长 Turn Idea）
  ▼
话题
  ├─ 交互记录：完整原文，次级的完整记录；材料变长时折叠（第 6 节）
  └─ 进展记录：只增的条目，每一条指回原文；话题停下时写一条（第 3、4 节）
  │
  ├─ 推送：另一个会话被路由回来 → 话题的进展；会话压缩之后 → 进展记录的索引（第 7 节）
  ├─ 拉取：按句柄回查原文（第 7 节）
  ├─ 生命周期：停下够久 → 沉淀；容量满或明确结束 → 退役（第 5 节）
  └─ 存储：每个话题一条只追加的日志（第 5 节）
```

| 部分 | 所有者 | 本文位置 |
|:---|:---|:---|
| 交互记录、进展记录、折叠、话题的生命周期与存储 | Patchouli | 第 3～6 节 |
| 前台读取（拉取）的授权与入口 | Workspace 能力层 | 第 7 节 |
| 推送的时机与注入 | 连接器、Import Bus 与任务进程 | 第 7 节 |
| 会话压缩与检查点 | harness、Alice 与会话资源 | 第 8 节 |

## 3. 话题的进展

**设计**：

话题的进展 = 已整理的部分 + 未整理的原始交互记录，读话题的 agent 看到的就是这两部分：

- **已整理的部分**：进展记录，按时间排列的条目，每一条写自上一条以来话题有了什么进展（像过去 WIP 记忆那样的总结性语句），可以跨几轮，也可以只是一轮之中的一部分。只往后追加，旧条目不改写；不整段重写。
- **指回原文**：每一条用“轮次标识 + 事件序号区间”指回原文，与记忆引用是同一套坐标（[记忆生成](../patchouli/generation.md#22-段标签与记忆引用)）；区间可以落在一轮之内。
- **未整理的原始交互记录**：进展记录还没有覆盖到的交互原文。

合起来就是完整可用的短期记忆。还没写过一条的话题，进展就是它的原文，所以从未折叠的话题也能回答“进行到哪一步”。

进展记录的数据、写一条的输入（旧进展记录加新交互记录原文，输出一段总结）与它的理由已经实现，见[感知与短期话题](../patchouli/perception.md#13-话题原始数据与工作视图)第 1.3 节与[第 6 节](../patchouli/perception.md#6-memoryperceptionengine-与两个写作者)。

todo 与 plan 的渲染设计待定，取决于各家harness的实现细节与完成逻辑。

**现状与差距**：

- 话题的两种摘要与进展记录分开：`topic_summary` 在建话题时由 Gateway 写一次，之后不更新，话题路由一直用它；`fold_summary`（折叠摘要）只在折叠时产生，日常约 97% 的话题直到结算都没有它（[感知与短期话题](../patchouli/perception.md)）。
- 进展记录已实现（[感知与短期话题](../patchouli/perception.md#13-话题原始数据与工作视图)第 1.3 节）：每一条覆盖相邻的若干整轮，区间落在一轮之内的条目没有实现；进展记录经话题列表的 HTTP 响应交给话题池，经话题句柄交给 agent（第 7 节）。
- 段与引用坐标已经实现：段由一轮的事件按模型响应推出，工具结果挂在调用事件上，一段不会拆开调用与结果；记忆的版本记录保存“轮次标识 + 事件序号区间”（[记忆生成](../patchouli/generation.md#22-段标签与记忆引用)）。这些引用现在只有写入方。
- 规模：只按轮与轮之间的停下计，日常话题的进展记录中位 1 条，90% 不超过 3 条，最长 14 条（85 轮的话题）。

## 4. 什么时候写一条

**设计**：

- **轮与轮之间**：话题停下时写一条，即会话切到别的话题，或话题停下一段时间。交互连续进行时前台自己的上下文就够用，需要话题的进展的，是停下之后回来的人或另一个会话。话题停下 `progress_record_after_seconds` 之后写一条已经实现；沉淀看同一个访问时间、阈值更长，沉淀时通常已经写过一条，不另外再写（[感知与短期话题](../patchouli/perception.md#7-维护存储与启动时读回)第 7 节）。
- **一轮之内**：长 turn 可能一直不停，话题停下只是一种触发方式；段用来指回原文，但不必每段都写一条。一轮之内什么时候写一条暂缓设计（第 9 节搁置项）。
- 候选，有人读时补：新对话不多时，直接给“已整理的部分 + 未整理的原始交互记录”，不需要生成；新对话已经很长，才在读之前先写出新的一条。

**现状与差距**：

- 话题停下够久时写一条已实现（随附配置的 `progress_record_after_seconds` 为 1800 秒）；会话切到别的话题、一轮之内都没有触发。plugin 模式在一轮结束后才导入：Claude Code 在 `Stop` 钩子同步，连接器按最后一条助手条目的 `stop_reason` 判断这一轮是否写完（[Import Bus](../system/import-bus.md)）。
- 单个事件没有时间戳（[数据模型](../architecture/data-model.md#49-交互记录turnevent-与-interactionpayload)），“步与步之间长时间等待”无法从已存的事件算出。
- “会话切到了别的话题”由 Gateway 的话题路由判断：任务进程与 Import Bus 都把上一轮实际所在的话题交给 Gateway（`GatewaySessionTurn.topic_id`），这一轮的目标话题与它不同即是切走。差距在于这个结果不随提交交给 Patchouli：交互提交只带目标话题，Patchouli 只看到一轮应用到了哪个话题，看不到这个会话离开了哪个话题。候选：root API 提交交互时拿到的话题来源中已有上一轮的话题，随提交一并交出。
- 日常 121 个长轮次（每轮至少 72 个事件）中，83 个没有一次计划步骤完成；Claude 桌面端每轮都列自己的操作步骤，计划步骤完成在普通轮次里也频繁出现，不代表话题的工作单位。
- 成本：日常 11 天导入 881 轮，同一会话切到别的话题 132 次、话题内停下超过 30 分钟 41 次，轮与轮之间停下时写的次数约为每轮都写的四分之一。

## 5. 生命周期与存储：沉淀、退役与话题的日志

**设计**：

- 结算拆成沉淀与退役，话题的存储是每个话题一条只追加的日志：均已实现，设计与理由见[感知与短期话题](../patchouli/perception.md#4-触发原因与具名用例)第 4 节（沉淀与退役、沉淀进度在生成完成时推进、退役话题的日志保留到最后一次沉淀完成）、[第 7 节](../patchouli/perception.md#7-维护存储与启动时读回)（日志、关闭与启动时读回）与 [MemoryLibrary](../patchouli/memory-library.md#11-短期话题工作台)第 1.1 节（短期存储与 Port 契约）。
- 话题跨重启保存整个话题，本话题生成的记忆是话题的一部分。只保存进展记录的话，恢复后不能续写原文，重启照样切开正在进行的话题。
- 进展记录的一条同样是话题日志中的一种记录（`progress_entry`），在短期存储里有对应的写入方法，发生时即落盘：已实现，见 [MemoryLibrary](../patchouli/memory-library.md#11-短期话题工作台)第 1.1 节。
- 随退役执行的沉淀在执行时读本话题生成的记忆，看得到排在它前面的同话题沉淀生成的记忆：已实现，见[记忆生成](../patchouli/generation.md#42-本话题生成的记忆)第 4.2 节。

**现状与差距**：

- 沉淀与退役、话题的日志、跨重启与崩溃后读回均已实现（链接见上）；同一话题多次沉淀时，本话题经任何途径生成的记忆进入同一份“本话题生成的记忆”记录，记在话题的日志上，见[记忆生成](../patchouli/generation.md#42-本话题生成的记忆)第 4.2 节。
- 拆开之前的日常数据（11 天 89 次结算）：话题停下超过时限 77、关闭 11、LRU 1。

## 6. 折叠：只为材料服务

**设计**：

- 折叠是材料变长以后维护话题的一种手段，只为材料服务，不产出话题的进展。
- 什么时候折叠只看材料长度：话题的材料超过上限时折叠，与话题是否停下、有没有进展记录无关。“什么时候折叠”与“用什么材料生成进展记录”互不决定。
- 一旦折叠，就接受可能的细节丢失。一次折叠不隐式沉淀，不证明已经生成 Memory。
- 折叠摘要需要结构化。话题多次折叠时，新的折叠摘要由旧的折叠摘要与新折叠的轮次写成，会递推；它又是被折叠的轮次在材料中唯一的留存，这里丢了，读材料的生成都看不到。
- 折叠摘要是材料的一部分，读者是读材料的生成，不是前台。
- 候选，可以核对的结构：一部分由规则抽取（标识符、工具结果、失败诊断），一部分是判断类的栏（已定的决定、被否掉的方案、纠正），再加一道覆盖校验，保不住就不提交这次折叠。
- 候选：单轮超过上限时，在段的边界上折叠，不拆开调用与结果。

| 议题 | 候选 |
|:---|:---|
| 预算 | 同时约束已有折叠摘要、保留后缀与渲染开销；计数与块数联合使用，按实际的读取请求分别校验，不能把话题的软水位当作最终请求上限 |
| 接受与失败 | 候选摘要完整、有效且满足预算后才切换视图；失败保留上次有效的摘要与尚未可靠交接的材料，重试有界并检查是否实际释放空间 |
| 工具证据 | 从实际工具结果选取关键返回值、错误原因与验证结论，保留原文定位，不要求把所有日志重新内联 |
| 跨入口 | 按实际来源与能力处理材料；主动、被动或连接器名称不直接决定是否折叠 |
| 观测 | 记录资料规模、内联规模、摘要规模、耗时与失败原因；观测不替代领域结果 |

**现状与差距**：

- 工作视图的 token 超过 `fold_token_threshold` 时折叠，保留最近 `fold_retain_recent_blocks` 块，其余折叠进 `fold_summary`；随附配置为 131072 与 2，代码缺省阈值 32768；`LLMFoldSummaryWriter` 参照旧摘要写新摘要，失败回退为规则摘要（[感知与短期话题](../patchouli/perception.md)）。
- 结算时被折叠的轮次只以 `fold_summary` 进入生成，原文冻结进来源 Artifact、不参与生成（[记忆生成](../patchouli/generation.md)）。
- 折叠摘要的读者不只是生成：Gateway 的话题路由在没有 `topic_summary` 时用它、候选话题渲染为“状态”、查询理解读它；话题池详情把它标为“较早内容的摘要”，话题卡片与对话页的话题选择器把它当副标题；主动写入的材料、结算材料与下一次折叠读它。
- 折叠摘要的提示词（`get_fold_summary_prompt_text`）用固定的四段标题（核心目标、系统状态与已完成、约束与避坑、当前焦点），写明供 Worker Agent 使用，与“读者是读材料的生成”不符。日常 17 份折叠摘要中 16 份按四段写，文件路径多只保住 5% 以下；一份规则回退摘要把约 79 万字符压成 120 字符；单轮超大时每来一轮就折一次，最后两次摘要逐字相同。

## 7. 什么时候用：推送与拉取

**设计**：

- 短期记忆的目的是让回到话题的 agent 知道话题进行到哪一步：一项工作做到哪了，或一段对话聊到哪了。两个使用点：另一个会话发来的消息被路由回已有话题时；会话压缩完成后，agent 可能处于短暂的失忆窗口。其他时间，正在交互中的 agent 大概率不会突然失忆。
- 两个时点都由系统把短期记忆注入（推送）进前台的上下文。
- 拉取与会话压缩之后的推送已经实现：
  - 拉取：agent 经 actor API 的 READ 按话题句柄读话题，不新增 actor 操作，权限沿用 `memory.read`；句柄的形态、与 alias 的互斥、可见性与结局见[读取视图与写入意图](../workspace/read-view-and-write-intents.md#41-话题句柄)第 4.1 节，alias 的保留前缀见 [MemoryLibrary](../patchouli/memory-library.md#12-中期当前可检索书库)第 1.2 节，索引与一条原文的渲染见 [MemoryCompiler](../patchouli/memory-compiler.md#41-话题句柄的读取结果)第 4.1 节；
  - 压缩之后的推送：Claude Code 压缩会话之后，插件的钩子以导入结果带回、连接器记下的话题标识读出会话最近一轮所在话题的索引并注入；注入什么、为什么只注入索引、注入哪个话题与读不到时怎样，见 [Import Bus](../system/import-bus.md#82-hooks)第 8.2 节，MCP 一侧的约定见 [MCP 记忆访问](../contracts/mcp.md#31-读话题句柄)第 3.1 节；
  - 推送是从话题到会话的单向读出，见[系统架构概览](../architecture/overview.md#7-短期记忆话题是前台与中期之间由-hivememory-持有的一环)第 7 节的跨层不变量。
- 候选，第 1 个时点：plugin 模式下，在用户提交时先为这一轮确定话题（root API 的 `determine_topic`，导入时沿用、不再路由）；落在已有话题、且这个会话此前不在它上面时，注入它的进展，会话本来就在这个话题上则不注入。controller 模式下 Gateway 在 PREPARE 阶段先于 CPU 运行，可以直接注入。
- 候选：注入标明是历史资料、不是指令。用户提交时的注入随这条消息排在历史之后；压缩之后的注入发生在缓存本来已断的时候，由 harness 决定位置（Claude Code 放在对话开头），索引的开头一行已经这样写明。
- 候选：话题池也签发句柄，在话题池界面显示，需要先出样稿。
- 长 turn 中压缩常发生在一轮中间，索引能否覆盖这一轮前面的部分，取决于一轮之内写一条（暂缓）与一轮进行中接收已完成的事件。

**现状与差距**：

- 拉取与压缩之后的推送已实现（链接见上）；第 1 个时点没有实现：插件没有注册 `UserPromptSubmit`，plugin 模式下用户提交时没有轮前路由。
- 一个外部会话在哪个话题上，只记得最近一轮所在的那一个：连接器为每个会话记下导入结果带回的话题，随状态文件跨过后端重启（[Import Bus](../system/import-bus.md#84-结束判定与游标)第 8.4 节）；后端不保存外部会话。
- controller 模式下 Alice 的上下文不含话题（[Agent Runtime](../alice/agent-runtime.md#3-输入上下文与-prompt-组装)）；Alice 的 MTP READ 能读话题句柄，但没有地方发给它句柄。
- `determine_topic` 已供任务进程的线程表与 Import Bus 编排器使用，MCP 写入时以它在一轮结束前确定话题（[能力层](../workspace/capability.md)）。日常一次话题路由约 3 秒。
- Claude Code：`UserPromptSubmit` 与 `SessionStart` 都能返回 `additionalContext`；注入只保证交到模型手里，不保证模型会读；`PostCompact` 不能注入。Codex 的钩子是 `SessionStart`、`Stop`、`Interrupt` 与 `SessionEnd`；Codex、pi 与 DeepSeek Harness 的连接器不注入，它们能否注入内容没有核对。

## 8. 会话压缩（前台一侧）

### 8.1 已完成的范围

- 会话提供完整、原样积累的对话历史；CPU 自行压缩派生上下文，会话记录保持原样。Alice 的自动压缩、工具结果减重、增量接力摘要与逐帧输入守卫已经实现，算法见 [Agent Runtime](../alice/agent-runtime.md#31-会话上下文视图)。
- 压缩检查点属于会话资源，不属于 Agent 或 CPU 的进程内状态（[会话资源](../workspace/sessions.md)）。
- 用户的 `/compact` 对外部 Actor 触发其 harness 的压缩，对 Alice 使用 Alice 自己的实现；手动 `/compact` 要等命令系统接回（[命令 Todo](../todo/conversation-compact-command.md)）。
- 保留新建、恢复与会话内 compact，不设“新会话接续旧会话”的交接摘要：把旧会话完整历史带入新会话等同恢复。话题的进展按话题推送（第 7 节），不是会话的交接摘要。
- 真实 LLM 下接力摘要的事实保留、继续执行的质量与成本还没有评估。

### 8.2 两边的关系与不变量

- 两边的关系已是跨层不变量：会话压缩、折叠与进展记录各回答一个问题、各自触发，Alice 的接力摘要不进入话题，话题向前台只是读出，见[系统架构概览](../architecture/overview.md#7-短期记忆话题是前台与中期之间由-hivememory-持有的一环)第 7 节。
- 外部 harness 的 summary/checkpoint 若交给 HiveMemory，应保留产生者、来源与覆盖信息，未知覆盖范围不能驱动裁剪，也不能直接充当话题的折叠摘要或进展记录。
- 摘要范围与保留的原文后缀不重复覆盖；覆盖只能相对同一来源与顺序向前推进。
- 压缩、折叠与写进展记录都不改变授权、写入接纳或持久化事实；RuntimeEvent 只观测，不决定提交成功。

### 8.3 可共享的技术

可以共享 token 估算、完整交互边界、有界摘要更新、完成状态校验、覆盖范围校验与有限重试；先复用已有低层工具与端口，不预设统一的压缩子系统，跨 L3 子系统只消费公共 contracts。摘要策略、模型选择、预算、数据所有者、覆盖游标、接受条件与清理生命周期分别维护；使用同一算法不使两边成为同一状态。技术调研见 [Agent Harness 会话压缩技术调研](../research/agent-context-compaction-research.md)。

## 9. 开放问题

**第 1 个时点在 plugin 模式下的轮前路由**。要在用户提交时推送，就得先为这一轮确定话题，每条消息多一次路由。

- 选项：每条消息都先路由；只在会话的第一条消息、或会话停顿之后路由；异步路由，限时内没有结果就不注入。
- 影响：前者延迟最大、覆盖最全；后两者减少延迟，但会漏掉会话进行中被路由回别的话题的情形。

搁置：

- 会话切走这一触发：多个会话并发进行时，一个会话离开话题不等于话题停下，判断逻辑还没有设计；在此之前只在话题停下够久时写一条。
- 长话题的旧条目怎样处理（合并成阶段，或只给最近若干条）。
- 每次记忆生成前都把话题的全部交互冻结成一份交互快照，各份互相重叠；属于[全链路原始资料保存](./PatchouliPageFoldingRawEvidenceDesign.md)。
- 折叠摘要的具体字段。
- Codex、pi 与 DeepSeek Harness 能否注入内容。
- 一轮进行中接收已完成的事件，由[长 Turn Idea](./long-running-agent-intra-turn-context-folding.md)维护。
- 一轮之内写一条的触发：按量（自上一条以来新增内容超过阈值，只在单轮内生效）与计划完成一步，都暂缓设计；“步与步之间长时间等待”等事件有时间来源再看。
- Alice 的 `turn_context`（每轮重新编译的检索记忆与附件）排在全部历史之前，检索结果一变，其后的历史都失去前缀缓存；实际命中率没有测过。
- 代码中已实现的段类名 `TurnSegment`，与长 Turn Idea 第 2.3 节候选的 `TurnSegment`（容量单位）同名。

## 10. 非目标

- HiveMemory 不控制、不替代前台上下文与 harness 的压缩，不改写会话原文。
- 会话与话题不共享状态；会话压缩与折叠、写进展记录之间没有机制联系。
- 不为每段写一条进展记录，不整段重写进展记录。
- 不在本文设计沉淀时写什么与主动写入的材料（过渡）。
- 为轻量 bot 管理 prompt 的托管上下文，属于[长 Turn Idea](./long-running-agent-intra-turn-context-folding.md)第 8.3 节。
