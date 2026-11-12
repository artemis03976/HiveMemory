---
title: Frontend Chat View
status: current
owner: frontend
scope: chat-sessions-structured-events-and-workspace-vision
code_paths:
  - frontend/src/components/ChatLayout.tsx
  - frontend/src/components/chat/
  - frontend/src/stores/chat/
  - frontend/src/stores/topic/
  - frontend/src/stores/workspace/
  - frontend/src/services/chatApi.ts
  - frontend/src/services/sessionApi.ts
  - frontend/src/hooks/useAgents.ts
  - frontend/src/hooks/useTopicTitles.ts
  - frontend/src/utils/effectiveModelParams.ts
  - src/hivememory/server/routers/chat.py
  - src/hivememory/server/models/chat.py
related_contracts:
  - docs/contracts/routes-and-events.md
  - docs/contracts/mtp.md
  - docs/contracts/error-model.md
related_docs:
  - docs/workspace/README.md
  - docs/system/attachments.md
  - docs/frontend/state-and-transports.md
  - docs/gateway/workflow.md
last_reviewed: 2026-11-11
---

# Chat 视图

Chat 是 HiveMemory 前端的主工作面。它需要在同一轮交互中容纳自然语言、入口决策、记忆引用、MTP 动作、子 Agent 运行和后台记忆任务，却不能把这些状态压扁成一串难以理解的文本。当前实现以 fetch-based SSE 为主传输，将人类回答与机器动作拆成有序内容块，再把更重的观测信息放入右栏。

## 1. 视图分层

```text
会话栏                        中间栏                             右栏（Workspace Vision）
  会话列表                      会话标题与轮数                      引用记忆 · 条数
  （今天 / 昨天 / 更早）         消息时间线（MTP / 子 Agent 卡片）    记忆任务 · 条数
                                OmniInput                          模型配置
```

左侧是会话列表，中央展示服务端会话历史和当前流式回复，右侧以三个标签展示本轮引用的记忆、记忆生成任务与本会话的模型配置；后端日志与运行时事件在[全局运行页](./run-view.md)查看。会话历史由后端进程内的 ConversationSession 持有，浏览器仅恢复选择并重新读取历史；Topic 池独立放在 MemoryLibrary 的“短期话题池”整理视图。右栏标签和两栏的收起状态属于本地 UI 偏好；两栏的宽度、断点与抽屉见[应用壳与视觉系统](./application-shell.md)。

## 2. 发起一次对话

OmniInput 在发送前组装：

- 文本内容；
- 当前选中的 `agent_id`，默认 `omni_doll`（Chat 是 Agent action，必须携带具体 Agent，后端不回退到保留 `system`）；
- 当前 `session_id`（没有选择时由后端自动创建）；
- `enable_memory_retrieval`；
- 可选的 `model_selection`（对应后端的 `ModelSelection`）：模型注册表 ID `model_id`，以及开启覆盖时的 temperature、top_p、max_tokens；没有选择模型时不发送 `model_id`，沿用 Agent 的模型；
- 可选的 `topic_id`：用户以 `#` 为这一轮指定的话题（第 2.1 节），没有指定时不发送，由 Gateway 的话题路由决定。

用户导向基础选择（`user_id + workspace_id`）不在请求体中重复传递：`services/identity.ts` 统一以请求头 `x-user-id`/`x-workspace-id` 携带。server 解析身份声明，任务进程的注册入口经认证网关认证执行者与系统的 root 工作 actor，各阶段调用授权时才组装 `IdentityScope`。请求体中的 `session_id` 指定本次使用的会话资源，不参与身份构造；后端校验会话属于请求的 Workspace，且没有其他进程占用。发送时冻结当前会话与 Agent，防止流中切换目标。

`@` 菜单和 Agent 胶囊只改变本轮使用的 Agent，输入中的 `@name` 会在选中后被移除，不作为文本 mention 发送。可选列表来自 Profile 列表接口，只列出启用的 Agent，停用的 Agent 仍能在历史消息中显示名称；内置的 `omni_doll` 由后端代码提供、没有源记忆原子，不在接口结果中，前端把它固定放在列表首位，后端已有同名图纸时以后端为准（`hooks/useAgents.ts`）。Agent 管理页只展示后端真实存在的图纸。持久化的 Agent ID 若不再存在，前端回退到 `omni_doll`。

请求通过 `POST /api/v1/chat` 建立 fetch SSE。前端在收到 `process_id` 事件后，停止按钮会向 `POST /api/v1/chat/stop` 发出 best-effort 取消；stop 请求体只携带 `process_id`，身份选择仍来自统一请求头。后端把请求认证为 (user, `system`) 的工作 actor，对进程的 Workspace 执行 root 授权；授权拒绝与进程不存在都按 not found 处理。本地状态依次区分 preparing、streaming、cancelling、finalizing 和最终 completed/cancelled/failed。停止请求并不等于已经停止，仍需等待后端 `run_status` 或 `done` 给出终态；停止被后端拒绝时恢复为 streaming，可以再次停止。

### 2.1 `#` 话题引用

用户可以为下一轮指定话题，而不是交给 Gateway 的话题路由猜测：在词首输入 `#`（`#` 本身不留在消息中），或点击输入栏的“话题”按钮，弹出短期话题池中的话题列表（`components/chat/TopicPicker.tsx`），可按标题筛选，Enter 选中第一项，Esc 关闭。选中后输入框上方显示带 `#` 的话题标签，可以移除。

指定的话题保存在 chat store 的 `designatedTopic` 中，不持久化；发送时以请求的 `topic_id` 带上并随即清除，只作用于这一轮。记忆图书馆短期话题池的“在对话中引用”也设定它，并跳到对话页。

后端在话题仍在候选话题中时直接取用它，不调用话题路由模型（[Gateway 工作流](../gateway/workflow.md)）。选择与发送之间话题可能已被结算或驱逐，这时这一轮以错误码 `gateway.topic.designated_unavailable` 失败、不提交，回复显示“指定的话题已经结算或不在短期话题池中，请重新选择话题”。前端不把它自动改为路由，避免这一轮悄悄落到别的话题上。

## 3. SSE 事件投影

| 事件 | 前端作用 |
|:---|:---|
| `token` | 追加主 Agent 或子 Agent 文本 |
| `mtp_start` / `mtp_result` | 创建并更新 MTP 结构化卡片 |
| `sub_agent_start` / `sub_agent_end` | 创建子 Agent 区块并收束状态 |
| `memory_refs` | 替换右侧“引用记忆”列表 |
| `command_result` | 展示命令消息；当前内置命令暂时不可用（见 [Gateway 全局命令](../gateway/commands.md)） |
| `process_id` | 建立取消请求所需的任务进程标识，并回填 `session_id` |
| `run_status` | 驱动准备、流式、取消、收尾和失败状态 |
| `context_compaction` | 显示压缩进度，不生成消息或工具卡片 |
| `done` | 写入结局、可选最终文本和交互实际所在的话题 `topic_id`（有输出且交互已应用时才有）；不携带话题池或记忆任务 ID。无输出时移除本轮消息并把用户输入放回草稿 |
| `error` | 结束本地流并在回复中显示错误；交互提交失败与指定话题不可用两个错误码换成专门的提示（`stores/chat/failureMessage.ts`），其余显示后端给出的文案 |

`scope=sub` 的 token 和 MTP 事件进入相应 SubAgentBlock，不与主 Agent 正文混合。`ContentBlock[]` 保留文本、MTP 和子 Agent 的相对顺序，使卡片能够出现在动作真实发生的位置，而不是全部堆在消息末尾。

前端 `MTPVerb` 类型当前只列出 SEARCH、READ、RUN、WRITE、UPDATE，遗漏已经存在于后端契约中的 CALL。CALL 的流式视觉主要由子 Agent 事件驱动，因此界面仍能展示子 Agent 卡片，但 Agent 权限页面不能配置 CALL，属于跨页面模型偏差。

## 4. 会话恢复与 Topic 标注

ChatLayout 挂载时加载会话列表，并按浏览器保存的 `currentSessionId` 读取历史。会话栏按最近更新时间分为今天、昨天、更早三组（按本地日历日），每行写标题、更新时间（一周以内用相对时间，每分钟刷新）、轮数与是否运行中；重命名与删除在悬停或键盘聚焦时出现，运行中的会话不能删除。会话列表支持新建、选择、重命名和删除；中间栏头部显示会话标题与轮数。会话内可逐轮切换 Agent；删除会话不删除 Topic 或 Memory。

历史以 `ConversationTurn.payload` 的轮次事件为来源，按序号恢复正文、MTP 和 CALL 卡片：卡片的动词与目标取调用事件的 `verb`、`targets`，命令文本是调用原文（MTP 指令片段，指令前正文已是前一个正文块），状态与结果取同一个调用事件上的结局；无法解析的指令没有动词，显示为 UNKNOWN。WRITE/UPDATE 的 ACK 表示成功接纳；没有结局的调用（停止时仍挂起的 CALL）显示未完成，不继续转圈。CALL 只恢复根帧记录的结果，不编造子帧内部轨迹。回复有话题时在消息底部显示话题标题，交互没有应用成功的轮次话题为空、不显示；cancelled/failed 轮次带结局标记。

话题标题取自短期话题池：`hooks/useTopicTitles.ts` 在消息中出现尚未查过的话题时读取一次话题列表。话题仍在池中且有标题时，标题可以点击，跳到记忆图书馆的短期话题池并选中这个话题（选中项记在 topic store 的 `selectedTopicId`）；已结算、被驱逐或没有标题的话题显示“未命名话题”，悬停显示 `topic_id`，不可点击。

会话读取同时恢复服务端已接受的压缩检查点。消息以 `interaction_id` 匹配摘要与减重边界，在对应回复之后标记覆盖位置；摘要可展开，摘要与引用以纯文本展示，完整原文仍然可见。未知产生方或格式版本不展示标记。压缩开始时显示“正在压缩较早的上下文”，不会插入正文，也不会使无输出轮次变成已保存轮次。

收到压缩结束事件后，流关闭时另读会话取得检查点；done 不携带它，前端不把进度当作提交成功。此读取只更新检查点，保留当前回复的流式内容与失败提示；切换会话、清空或开始新一轮会使旧读取失效。取消后无输出也可读取旧历史的新检查点。读取失败提示重新选择会话刷新。

流式运行与历史加载期间禁止切换或删除当前会话；服务端仍负责跨窗口的会话互斥。读取历史返回 404 时清除失效选择并刷新列表；其他错误保留错误状态并禁止基于未加载的历史继续发送，允许重选重试。没有本地会话 ID 时仍可直接发送，由首个事件回填新会话。

`topic_info` 与 `pool_topics` 已删除。本轮话题由 `done.topic_id` 给出，恢复时读会话轮次的 `topic_id`（会话类型中可空）；两者都是交互应用后实际所在的话题（话题何时建立见[任务进程](../workspace/task-process.md#53-话题何时确定)）。Topic 不决定聊天历史。MemoryLibrary 经 Topic 列表接口读取短期话题池，保留 `settle` 与 `delete`：前者把内容交给记忆生成并结束 Topic 生命周期，后者结束生命周期而不触发记忆写入。失败时 Topic store 回滚乐观移除；Memory Library 的 `archive` 仍指中期记忆归档。

本轮 failed done 的 `failure_reason` 映射为四类提示：缩短当前输入或附件、历史无法压缩进预算时重试或新建会话、工具结果超预算后继续对话或缩小任务、提供商超窗后减少输入或选择更大窗口。稳定原因定义见[错误模型](../contracts/error-model.md#22-agent-run)，未知原因使用通用失败提示。已有部分回复保留并标记未完成；无输出时移除本轮消息、回填草稿，同时展示原因。失败码只存在本次前端运行状态，不写入会话 payload，重载历史后显示通用 failed 结局。

## 5. 消息时间线

### 5.1 滚动与空会话

打开会话或历史读取完成时，消息列表定位到最新一轮。停在底部（离底部 48px 以内，`utils/scrollFollow.ts`）时，新消息与流式输出自动跟随；用户向上翻过之后不再跟随，底部出现“回到最新”按钮，并标出此后新增的消息条数（“· N 条新消息”），点击回到底部并恢复跟随。滚动条是可见的细样式。

没有消息时，读取历史期间显示“少女祈祷中...”，其余情况显示欢迎区：当前 Agent 的名称、记忆预检索开关的状态与三条示例问题，点击示例把它填入输入框。

### 5.2 消息样式

用户消息是靠右的紧凑气泡，没有空格的长串（如 JSON）也会在中间换行，不撑出气泡。助手回复不包在带边框的卡片里：头部是 Agent 图标与名称、“回复中”或回复时间，正文与 MTP、子 Agent 卡片按内容块顺序排列，只有这两种卡片带框。

Agent 的名称与图标取自 Profile 列表（`hooks/useAgents.ts` 的 `resolveChatAgent`），不在列表中的 Agent 显示默认图标与 `agent_id`；静态的 mock Agent 只在 debug 模式的 Agent 管理页使用，不进入消息。MTP 卡片的动词徽标保留协议名（SEARCH、READ……），说明用中文（如“检索完成，……”），目标等标识符原样显示；没有完成的动作写出原因（“RUN 未完成：……”）。cancelled 轮次标为“已中断 · 回复未完成”，failed 轮次显示第 4 节的原因提示。

### 5.3 消息操作

回复结束后，消息底部右侧出现操作按钮：

- **复制**：复制这条回复的正文（各正文块的 Markdown 原文以空行连接），提示“已复制”。
- **👍/👎**：对这一轮预检索命中的记忆统一记录正/负反馈，逐条调用 `POST /api/v1/memories/{id}/feedback`。预检索命中只随 `memory_refs` 事件下发、不写入会话记录，所以按钮只出现在本页完成的最后一条回复上，且这一轮有命中时才出现；从会话历史加载的轮次不显示。
- **重新生成**：只出现在最后一条回复上，流式进行中不可用。以当前选中的 Agent 把这一轮的用户消息作为新的一轮再发送一次，原来的一轮保留在历史中（会话原文不删改）。它只重发消息文本：原轮次的附件与指定话题不随之重发，也不读取输入栏的记忆预检索开关与右栏的模型配置，这一轮按请求的缺省值运行（开启预检索、沿用 Agent 的模型）。

最后一条回复的按钮常驻，其余回复的复制按钮在悬停或键盘聚焦时出现。

## 6. 右栏（Workspace Vision）

右栏有引用记忆、记忆任务、模型配置三个标签：引用记忆的标签带本轮命中的条数，记忆任务的标签带进行中与等待中的任务数。选中的标签持久化在 `chat-ui-store` 中。

### 6.1 引用记忆

`memory_refs` 映射为本轮预检索命中的 MemoryAtom。每张卡片写类型图标块与类型名、别名（没有别名时取 ID 前缀）、标题与三行以内的摘要，底部是带标签的“生命力”“置信度”（悬停说明含义）与至多三个标签。悬停或聚焦卡片时出现逐条的 👍/👎，经 feedback API 记录这条记忆有没有帮助。新一轮开始时旧引用会被清空，避免把上轮证据错配给当前回答。预检索命中不带分数，卡片不显示相关度（[Todo](../todo/referenced-memory-relevance.md)）。

### 6.2 记忆任务

`memory.task.*` RuntimeEvent 在 WRITE/UPDATE 执行期间即可投影任务创建与状态变化，面板挂载时调用任务列表 API 补齐 best-effort 事件的遗漏；done 不再触发按任务 ID 刷新。任务按创建时间倒序排列，不显示“当前轮”标记或置顶（任务事件没有 process_id）。默认只列等待中与生成中的任务，勾选“显示已结束”后连已结束的一起显示；顶部一行写事件流的连接状态，并可刷新。

每个任务是一张卡片：名称或别名、带圆点的状态胶囊（等待中、生成中、完成、已取消、失败，请求取消后为“取消中”），底部一行写进入队列或开始的时间、完成时的用时，或失败的原因。点开卡片显示 task_id、topic_id、来源、别名与各时间点，未结束的任务可以在这里取消。task 数据本身不写入 localStorage，重载后重新以后端查询为准。memory task API 失败时只有 debug 模式换用 mock task，默认如实报错（[状态与传输](./state-and-transports.md#6-mock-降级边界)第 6 节）。

### 6.3 模型配置

模型配置是一张卡片，持久化在 `chat-runtime-config-store` 中。设置不按会话区分，作用于这个浏览器之后发出的每一轮，卡片上也这样写明：

- **模型**：从模型注册表中选择，或“跟随 Agent”（括号中写出 Agent 当前的模型），跟随时请求不带 `model_id`；
- **单独设置生成参数**：打开时以滑块设置温度、Top P 与最大输出，随请求下发；关闭时不下发这三项，卡片列出“当前生效”的值及来源（`utils/effectiveModelParams.ts`）：模型依次取本会话的选择、Agent Profile、注册表的默认模型；温度与 Top P 取 Agent Profile 的值，Profile 中为空时取模型定义的默认值；最大输出取模型定义。

## 7. 当前交互能力与限制

已经接线：

- Agent 选择和 `@` 快捷选择（只列出启用的 Agent）；
- `#` 话题引用；
- 记忆预检索开关；
- 本会话的模型选择与生成参数覆盖；
- Enter 发送、Shift+Enter 换行；
- 会话管理、历史恢复、逐轮话题标题、压缩进度与可展开摘要边界；
- 流式状态、停止生成；finalizing 期间停止按钮禁用，等待提交完成；
- 滚动跟随与“回到最新”；
- 消息的复制、反馈与重新生成；
- MTP 与子 Agent 卡片；
- 引用记忆、记忆任务与模型配置三个标签（日志与运行时事件在[全局运行页](./run-view.md)）；
- 附件上传（拖拽/Paperclip 进入同一串行队列，请求内解析到 READY/FAILED）、READY 附件的本轮选择与发送时冻结，选择随 `attachments` 字段提交。

限制：

- 输入草稿不跨刷新恢复；
- 消息的 👍/👎 只覆盖本页完成的最后一轮，历史轮次的预检索命中无从得知；
- 重新生成只重发消息文本，不带原附件与指定话题，也不读取当前的记忆预检索开关与模型配置；
- 引用记忆没有相关度分数。

## 8. 不变量与错误边界

- 只有后端事件可以把 MTP、子 Agent、task 或 run 标记为成功；
- `done` 之前断流按失败或取消处理，不能把已有 token 当作完整终态；已收到 `finalizing` 时后端继续提交，提交结束后才写入会话；前端改为重新读取会话历史并提示“本轮已保存”，不回填输入，提交尚未结束时读到的历史可能还没有这一轮；
- 开流之前被后端拒绝时，回复显示后端给出的原因，不显示为系统错误；只有连不上后端或流中断才是系统错误（[状态与传输](./state-and-transports.md#31-http-错误解析)）；
- 新一轮开始时清除旧引用，回复的话题来自 done 或服务端历史，标题只是从短期话题池取来的展示；
- 用户指定的话题被拒绝时这一轮失败，不改为自动路由；
- 消息反馈只作用于本页确知的预检索命中，不为历史轮次猜测命中了哪些记忆；
- 取消是请求—确认过程，UI 必须区分 cancelling 与 cancelled；
- 机器动作保留在有序结构块中，不能通过正则从最终自然语言“猜”出执行历史；
- 日志和 RuntimeEvent 是 best-effort 观测旁路，不参与 chat 成功判定。

## 9. 验证入口与限制

主要入口为 `frontend/src/services/sessionApi.ts`、`frontend/src/services/chatApi.ts`、`stores/chat/`、`components/ChatLayout.tsx`、`components/chat/`、`stores/workspace/` 和 `stores/topic/`；后端契约与路由测试位于 `tests/unit/server/routers/test_chat.py`、`test_topics.py`、`test_runtime_events.py`、`test_logs.py`，指定话题从请求到 Gateway、被拒绝时不提交的代表用例在 `tests/unit/workspace/process/test_gateway_chat_flow.py`。
滚动是否跟随、当前生效的模型参数与消息中 Agent 的解析分别由 `frontend/tests/scroll-follow.test.mjs`、`effective-model-params.test.mjs` 与 `chat-agents.test.mjs` 覆盖。
附件上传与选择的入口为 `services/attachmentApi.ts`、`stores/attachment/`、`components/chat/AttachmentQueue.tsx` 和 `hooks/useAttachmentUpload.ts`；链路事实见[Chat 附件链路](../system/attachments.md)。

会话行为测试位于 `frontend/tests/conversation-session.test.mjs`（`npm run test:session`），覆盖历史重建、服务端恢复、404、加载失败、发送绑定、无输出回填、晚到停止响应、开流前与停止时的后端拒绝，以及 finalizing 前后断连、压缩后刷新检查点、晚到读取隔离、失败原因提示与摘要纯文本展示；确定性 HTTP 链路见 `tests/e2e/system/test_chat_session.py`。会话没有跨后端重启持久化，也没有 SSE 断线续传；多窗口日志选举在浏览器挂起/崩溃情形下的可靠性仍未全面验证。状态所有权的完整矩阵见[状态、持久化与传输](./state-and-transports.md)。
