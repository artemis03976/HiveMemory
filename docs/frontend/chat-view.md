---
title: Frontend Chat View
status: current
owner: frontend
scope: chat-sessions-structured-events-and-workspace-vision
code_paths:
  - frontend/src/components/chat/
  - frontend/src/stores/chat/
  - frontend/src/stores/topic/
  - frontend/src/stores/workspace/
  - frontend/src/services/chatApi.ts
  - frontend/src/services/sessionApi.ts
  - src/hivememory/server/routers/chat.py
related_contracts:
  - docs/contracts/routes-and-events.md
  - docs/contracts/mtp.md
  - docs/contracts/error-model.md
related_docs:
  - docs/workspace/README.md
  - docs/system/attachments.md
  - docs/frontend/state-and-transports.md
last_reviewed: 2026-10-19
---

# Chat 视图

Chat 是 HiveMemory 前端的主工作面。它需要在同一轮交互中容纳自然语言、入口决策、记忆引用、MTP 动作、子 Agent 运行和后台记忆任务，却不能把这些状态压扁成一串难以理解的文本。当前实现以 fetch-based SSE 为主传输，将人类回答与机器动作拆成有序内容块，再把更重的观测信息放入左右侧栏。

## 1. 视图分层

```text
Context Sidebar              Chat Pane                     Workspace Vision
  会话列表                   消息时间线                     引用记忆
  模型/采样覆盖                 OmniInput                    Memory Runtime
                                MTP / Sub-Agent 卡片          日志与 RuntimeEvent
```

左侧提供会话管理与本轮配置，中央展示服务端会话历史和当前流式回复，右侧展示记忆与运行观测。会话历史由后端进程内的 ConversationSession 持有，浏览器仅恢复选择并重新读取历史；Topic 池独立放在 MemoryLibrary 的“短期话题池”整理视图。面板 tab 和折叠状态属于本地 UI 偏好。

## 2. 发起一次对话

OmniInput 在发送前组装：

- 文本内容；
- 当前选中的 `agent_id`，默认 `omni_doll`（Chat 是 Agent action，必须携带具体 Agent，后端不回退到保留 `system`）；
- 当前 `session_id`（没有选择时由后端自动创建）；
- `enable_memory_retrieval`；
- 可选的模型注册表 ID，以及 temperature、top_p、max_tokens 单轮覆盖。

用户导向基础选择（`user_id + workspace_id`）不在请求体中重复传递：`services/identity.ts` 统一以请求头 `x-user-id`/`x-workspace-id` 携带。server 解析身份声明，任务进程的注册入口经认证网关取得访问 context，各阶段授权时才组装 `IdentityScope`。请求体中的 `session_id` 指定本次使用的会话资源，不参与身份构造；后端校验会话属于请求的 Workspace，且没有其他进程占用。发送时冻结当前会话与 Agent，防止流中切换目标。

`@` 菜单和 Agent 胶囊只改变本轮使用的 Agent，输入中的 `@name` 会在选中后被移除，不作为文本 mention 发送。可选列表来自 Profile 列表接口；内置的 `omni_doll` 由后端代码提供、没有源记忆原子，不在接口结果中，前端把它固定放在列表首位，后端已有同名图纸时以后端为准（`hooks/useAgents.ts`）。Agent 管理页只展示后端真实存在的图纸。持久化的 Agent ID 若不再存在，前端回退到 `omni_doll`。

请求通过 `POST /api/v1/chat` 建立 fetch SSE。前端在收到 `process_id` 事件后，停止按钮会向 `POST /api/v1/chat/stop` 发出 best-effort 取消；stop 请求体只携带 `process_id`，身份选择仍来自统一请求头。后端以请求级 context 经进程控制授权，比对请求方与进程记录的驻留坐标；不可控与不存在都按 not found 处理。本地状态依次区分 preparing、streaming、cancelling、finalizing 和最终 completed/cancelled/failed。停止请求并不等于已经停止，仍需等待后端 `run_status` 或 `done` 给出终态。

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
| `done` | 写入结局、可选最终文本和本轮 `topic_id`；不携带话题池或记忆任务 ID。无输出时移除本轮消息并把用户输入放回草稿 |
| `error` | 结束本地流并显示错误 |

`scope=sub` 的 token 和 MTP 事件进入相应 SubAgentBlock，不与主 Agent 正文混合。`ContentBlock[]` 保留文本、MTP 和子 Agent 的相对顺序，使卡片能够出现在动作真实发生的位置，而不是全部堆在消息末尾。

前端 `MTPVerb` 类型当前只列出 SEARCH、READ、RUN、WRITE、UPDATE，遗漏已经存在于后端契约中的 CALL。CALL 的流式视觉主要由子 Agent 事件驱动，因此界面仍能展示子 Agent 卡片，但 Agent 权限页面不能配置 CALL，属于跨页面模型偏差。

## 4. 会话恢复与 Topic 标注

ChatLayout 挂载时加载会话列表，并按浏览器保存的 `currentSessionId` 读取历史。会话列表支持新建、选择、重命名和删除；TopBar 显示会话标题与轮数。会话内可逐轮切换 Agent；删除会话不删除 Topic 或 Memory。

历史以 `ConversationTurn.payload` 为来源，按事件顺序恢复正文、MTP 和 CALL 卡片，WRITE/UPDATE 的 ACK 表示成功接纳；未配对的动作显示未完成，不继续转圈。CALL 只恢复根帧记录的结果，不编造子帧内部轨迹。每条回复显示本轮 Topic 标注，cancelled/failed 轮次带结局标记。

会话读取同时恢复服务端已接受的压缩检查点。消息以 `interaction_id` 匹配摘要与减重边界，在对应回复之后标记覆盖位置；摘要可展开，摘要与引用以纯文本展示，完整原文仍然可见。未知产生方或格式版本不展示标记。压缩开始时显示“正在压缩较早的上下文”，不会插入正文，也不会使无输出轮次变成已保存轮次。

收到压缩结束事件后，流关闭时另读会话取得检查点；done 不携带它，前端不把进度当作提交成功。此读取只更新检查点，保留当前回复的流式内容与失败提示；切换会话、清空或开始新一轮会使旧读取失效。取消后无输出也可读取旧历史的新检查点。读取失败提示重新选择会话刷新。

流式运行与历史加载期间禁止切换或删除当前会话；服务端仍负责跨窗口的会话互斥。读取历史返回 404 时清除失效选择并刷新列表；其他错误保留错误状态并禁止基于未加载的历史继续发送，允许重选重试。没有本地会话 ID 时仍可直接发送，由首个事件回填新会话。

`topic_info` 与 `pool_topics` 已删除。本轮话题由 `done.topic_id` 给出，恢复时读会话轮次的 `topic_id`；Topic 不决定聊天历史。MemoryLibrary 经 Topic 列表接口读取短期话题池，保留 `settle` 与 `delete`：前者把内容交给记忆生成并结束 Topic 生命周期，后者结束生命周期而不触发记忆写入。失败时 Topic store 回滚乐观移除；Memory Library 的 `archive` 仍指中期记忆归档。

本轮 failed done 的 `failure_reason` 映射为四类提示：缩短当前输入或附件、历史无法压缩进预算时重试或新建会话、工具结果超预算后继续对话或缩小任务、提供商超窗后减少输入或选择更大窗口。稳定原因定义见[错误模型](../contracts/error-model.md#22-agent-run)，未知原因使用通用失败提示。已有部分回复保留并标记未完成；无输出时移除本轮消息、回填草稿，同时展示原因。失败码只存在本次前端运行状态，不写入会话 payload，重载历史后显示通用 failed 结局。

## 5. Workspace Vision

### 5.1 引用记忆

`memory_refs` 映射为本轮后端明确引用的 MemoryAtom。用户可以查看摘要与详情，并通过 feedback API 提交正/负反馈。新一轮开始时旧引用会被清空，避免把上轮证据错配给当前回答。

### 5.2 Memory Runtime

`memory.task.*` RuntimeEvent 在 WRITE/UPDATE 执行期间即可投影任务创建与状态变化，面板挂载时调用任务列表 API 补齐 best-effort 事件的遗漏；done 不再触发按任务 ID 刷新。任务按创建时间倒序排列，不显示“当前轮”标记或置顶（任务事件没有 process_id）。用户可以刷新、选择和取消 task；task 数据本身不写入 localStorage，重载后重新以后端查询为准。开发模式下 memory task API 失败可能使用 mock task，这一降级必须与真实执行状态区分。

### 5.3 Workspace Terminal

日志通过 `/api/v1/ws/logs` WebSocket 进入，RuntimeEvent 通过 `/api/v1/runtime-events/stream` EventSource 进入。Workspace store 维护过滤、trace/span 折叠与连接状态，但日志和事件不是业务账本，也不会跨刷新恢复。

为避免多个窗口各自占用连接，前端以 localStorage heartbeat 选举一个主窗口持有流连接，再通过 BroadcastChannel 把新日志与事件同步给其他窗口。这是浏览器级资源优化，不是跨进程可靠消费或 leader election 契约。

## 6. 当前交互能力与空壳入口

已经接线：

- Agent 选择和 `@` 快捷选择；
- 记忆预检索开关；
- 单轮模型/采样参数覆盖；
- Enter 发送、Shift+Enter 换行；
- 会话管理、历史恢复、逐轮 Topic 标注、压缩进度与可展开摘要边界；
- 流式状态、停止生成；finalizing 期间停止按钮禁用，等待提交完成；
- MTP 与子 Agent 卡片；
- 引用记忆、memory task 和 Workspace Terminal 日志面板；
- 附件上传（拖拽/Paperclip 进入同一串行队列，请求内解析到 READY/FAILED）、READY 附件的本轮选择与发送时冻结，选择随 `attachments` 字段提交。

尚未接线：

- `#` 话题引用按钮；
- assistant message 下方的复制、点赞和重新生成按钮；
- 跨刷新输入草稿恢复；
- 动态 Agent 头像/显示名，部分消息仍从静态 `MOCK_AGENTS` 解析视觉信息。

## 7. 不变量与错误边界

- 只有后端事件可以把 MTP、子 Agent、task 或 run 标记为成功；
- `done` 之前断流按失败或取消处理，不能把已有 token 当作完整终态；已收到 `finalizing` 时本轮已写入会话、后端继续提交，前端改为重新读取会话历史并提示“本轮已保存”，不回填输入；
- 新一轮开始时清除旧引用，回复 Topic 标注来自 done 或服务端历史；
- 取消是请求—确认过程，UI 必须区分 cancelling 与 cancelled；
- 机器动作保留在有序结构块中，不能通过正则从最终自然语言“猜”出执行历史；
- 日志和 RuntimeEvent 是 best-effort 观测旁路，不参与 chat 成功判定。

## 8. 验证入口与限制

主要入口为 `frontend/src/services/sessionApi.ts`、`frontend/src/services/chatApi.ts`、`stores/chat/`、`components/chat/`、`stores/workspace/` 和 `stores/topic/`；后端契约与路由测试位于 `tests/unit/server/routers/test_chat.py`、`test_topics.py`、`test_runtime_events.py`、`test_logs.py`。
附件上传与选择的入口为 `services/attachmentApi.ts`、`stores/attachment/`、`components/chat/AttachmentQueue.tsx` 和 `hooks/useAttachmentUpload.ts`；链路事实见[Chat 附件链路](../system/attachments.md)。

会话行为测试位于 `frontend/tests/conversation-session.test.mjs`（`npm run test:session`），覆盖历史重建、服务端恢复、404、加载失败、发送绑定、无输出回填、晚到停止响应，以及 finalizing 前后断连、压缩后刷新检查点、晚到读取隔离、失败原因提示与摘要纯文本展示；确定性 HTTP 链路见 `tests/e2e/system/test_chat_session.py`。会话没有跨后端重启持久化，也没有 SSE 断线续传；多窗口日志选举在浏览器挂起/崩溃情形下的可靠性仍未全面验证。状态所有权的完整矩阵见[状态、持久化与传输](./state-and-transports.md)。
