---
title: Frontend Chat Workspace
status: current
owner: frontend
scope: chat-topics-structured-events-and-kernel-vision
code_paths:
  - frontend/src/components/chat/
  - frontend/src/stores/chat/
  - frontend/src/stores/topic/
  - frontend/src/stores/kernel/
  - frontend/src/services/chatApi.ts
  - src/hivememory/server/routers/chat.py
related_contracts:
  - docs/contracts/routes-and-events.md
  - docs/contracts/mtp.md
  - docs/contracts/error-model.md
related_docs:
  - docs/architecture/workspace.md
last_reviewed: 2026-09-01
---

# Chat 工作区

Chat 是 HiveMemory 前端的主工作面。它需要在同一轮交互中容纳自然语言、入口决策、记忆引用、MTP 动作、子 Agent 运行和后台记忆任务，却不能把这些状态压扁成一串难以理解的文本。当前实现以 fetch-based SSE 为主传输，将人类回答与机器动作拆成有序内容块，再把更重的观测信息放入左右侧栏。

## 1. 工作区分层

```text
Context Sidebar              Chat Workspace                Kernel Vision
  Topic 概览                   消息时间线                     引用记忆
  模型/采样覆盖                 OmniInput                    Memory Runtime
                                MTP / Sub-Agent 卡片          日志与 RuntimeEvent
```

左侧描述本地选择和 Topic 池快照，中央承载本次尚未持久化的消息流，右侧展示由后端事件确认的记忆与运行观测。三者并不共享同一种生命周期：Topic、Memory 和 task 以后端为准，中央消息仅存在于当前浏览器运行期，面板 tab 和折叠状态则是本地 UI 偏好。

## 2. 发起一次对话

OmniInput 在发送前组装：

- 文本内容；
- 固定匿名 `user_id=default`；
- 当前选中的 `agent_id`，默认 `omni_doll`；
- `enable_memory_retrieval`；
- 可选的模型注册表 ID，以及 temperature、top_p、max_tokens 单轮覆盖。

`@` 菜单和 Agent 胶囊只改变本轮使用的 Agent，输入中的 `@name` 会在选中后被移除，不作为文本 mention 发送。持久化的 Agent ID 若不再存在，前端回退到 `omni_doll`。

请求通过 `POST /api/v1/chat` 建立 fetch SSE。前端在收到 generation ID 后，停止按钮会向 `POST /api/v1/chat/stop` 发出 best-effort 取消；本地状态依次区分 preparing、streaming、cancelling、finalizing 和最终 completed/cancelled/failed。停止请求并不等于已经停止，仍需等待后端 `run_status` 或 `done` 给出终态。

## 3. SSE 事件投影

| 事件 | 前端作用 |
|:---|:---|
| `token` | 追加主 Agent 或子 Agent 文本 |
| `mtp_start` / `mtp_result` | 创建并更新 MTP 结构化卡片 |
| `sub_agent_start` / `sub_agent_end` | 创建子 Agent 区块并收束状态 |
| `topic_info` | 更新本轮真实 Topic ID 与 Topic 池快照 |
| `memory_refs` | 替换右侧“引用记忆”列表 |
| `command_result` | 展示命令消息，并执行如 clear chat 的显式客户端动作 |
| `generation_id` | 建立取消请求所需的运行标识 |
| `run_status` | 驱动准备、流式、取消、收尾和失败状态 |
| `done` | 写入最终文本、task IDs、Topic 池与终态 |
| `error` | 结束本地流并显示错误 |

`scope=sub` 的 token 和 MTP 事件进入相应 SubAgentBlock，不与主 Agent 正文混合。`ContentBlock[]` 保留文本、MTP 和子 Agent 的相对顺序，使卡片能够出现在动作真实发生的位置，而不是全部堆在消息末尾。

前端 `MTPVerb` 类型当前只列出 SEARCH、READ、RUN、WRITE、UPDATE，遗漏已经存在于后端契约中的 CALL。CALL 的流式视觉主要由子 Agent 事件驱动，因此界面仍能展示子 Agent 卡片，但 Agent 权限页面不能配置 CALL，属于跨页面模型偏差。

## 4. Topic 的当前语义

ChatLayout 在挂载时读取后端 Topic 池。用户点击某个 Topic 后，当前实现只改变 `activeTopicId`、标题和高亮；它不会：

- 请求该 Topic 的消息历史；
- 恢复中央消息；
- 把选中的 Topic ID 作为下一次 chat 请求参数；
- 改变后端对新输入的 Gateway 路由判断。

本轮真正的 `currentTopicId` 只来自后端 `topic_info`。因此左侧 Topic 当前是“活跃上下文概览”，不是完整会话切换器。Topic 生命周期统一使用 `settle` 与 `delete` 两个后端术语：`settle` 将 Topic 内容交给记忆生成并结束当前 Topic 生命周期，`delete` 结束生命周期但不触发记忆写入；两者都可以先在本地乐观移除，失败后由 Topic store 按实际操作回滚。前端历史上的“Archive”显示不再作为 Topic 操作名称，`archive` 仅保留给中期记忆进入长期记忆库的操作。

## 5. Kernel Vision

### 5.1 引用记忆

`memory_refs` 映射为本轮后端明确引用的 MemoryAtom。用户可以查看摘要与详情，并通过 feedback API 提交正/负反馈。新一轮开始时旧引用会被清空，避免把上轮证据错配给当前回答。

### 5.2 Memory Runtime

`done.memory_task_ids` 和 RuntimeEvent 共同投影后台记忆生成任务。用户可以刷新、选择和取消 task；task 数据本身不写入 localStorage，重载后重新以后端查询为准。开发模式下 memory task API 失败可能使用 mock task，这一降级必须与真实执行状态区分。

### 5.3 Kernel Terminal

日志通过 `/api/v1/ws/logs` WebSocket 进入，RuntimeEvent 通过 `/api/v1/runtime-events/stream` EventSource 进入。Kernel store 维护过滤、trace/span 折叠与连接状态，但日志和事件不是业务账本，也不会跨刷新恢复。

为避免多个窗口各自占用连接，前端以 localStorage heartbeat 选举一个主窗口持有流连接，再通过 BroadcastChannel 把新日志与事件同步给其他窗口。这是浏览器级资源优化，不是跨进程可靠消费或 leader election 契约。

## 6. 当前交互能力与空壳入口

已经接线：

- Agent 选择和 `@` 快捷选择；
- 记忆预检索开关；
- 单轮模型/采样参数覆盖；
- Enter 发送、Shift+Enter 换行；
- 流式状态、停止生成；
- MTP 与子 Agent 卡片；
- 引用记忆、memory task 和 Kernel 日志面板。

尚未接线：

- `#` 话题引用按钮；
- assistant message 下方的复制、点赞和重新生成按钮；
- 输入草稿恢复；
- Topic 历史消息加载与真正切换；
- 消息/session 历史 API 与刷新恢复；
- 动态 Agent 头像/显示名，部分消息仍从静态 `MOCK_AGENTS` 解析视觉信息。

## 7. 不变量与错误边界

- 只有后端事件可以把 MTP、子 Agent、task 或 run 标记为成功；
- `done` 之前断流按失败或取消处理，不能把已有 token 当作完整终态；
- 新一轮开始时清除旧引用，Topic 只能由 `topic_info` 确认为本轮事实；
- 取消是请求—确认过程，UI 必须区分 cancelling 与 cancelled；
- 机器动作保留在有序结构块中，不能通过正则从最终自然语言“猜”出执行历史；
- 日志和 RuntimeEvent 是 best-effort 观测旁路，不参与 chat 成功判定。

## 8. 验证入口与限制

主要入口为 `frontend/src/services/chatApi.ts`、`stores/chat/`、`components/chat/`、`stores/kernel/` 和 `stores/topic/`；后端契约与路由测试位于 `tests/unit/server/routers/test_chat.py`、`test_topics.py`、`test_runtime_events.py`、`test_logs.py`。

当前 Chat 更接近“单次运行工作区”而非完整聊天产品：它没有服务端消息历史、账户边界、断线续传和跨刷新恢复，也没有证明多窗口选举在所有浏览器挂起/崩溃情形下可靠。状态所有权的完整矩阵见[状态、持久化与传输](./state-and-transports.md)。
