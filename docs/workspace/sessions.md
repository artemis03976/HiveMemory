---
title: 会话资源
status: current
owner: workspace
scope: conversation-session-ownership-history-and-compaction-checkpoint
code_paths:
  - src/hivememory/workspace/sessions/
  - src/hivememory/workspace/capability/root/session.py
  - src/hivememory/config/workspace.py
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/routes-and-events.md
related_docs:
  - docs/workspace/task-process.md
  - docs/alice/agent-runtime.md
  - docs/gateway/analysis.md
last_reviewed: 2026-11-06
---

# 会话资源

ConversationSession 是 workspace 的进程内资源，属于 Workspace、不属于 Agent。实际使用的对话上下文由会话提供并原样积累；Topic 继续作为记忆生成的材料组织方，与会话只有路由关联。

## 1. 所有权

`workspace/sessions/ConversationSessionService` 是 ConversationSession 的唯一状态所有者，组合根创建同一个实例并注入 root 管理能力、注册入口与进程执行器。会话以 `belong_to: WorkspaceIdentity` 归属 Workspace，不属于单个 Agent；每次请求独立认证并选择 Agent。会话管理只经 root API（[能力层](./capability.md#3-root-api)），列表、读取、重命名、删除均检查完整 Workspace 归属，越域与不存在统一隐藏。每轮的 `agent_id` 是展示标签，不参与授权；其他 Agent 的旧回复在提示词中带身份前缀，原有记忆可见性仍只约束直接资源访问。

## 2. 会话轮次与历史所有权

会话保存有序 `ConversationTurn`：进程关联 `interaction_id`、Agent 标签、唯一正文 `payload: InteractionPayload`、交互实际所在的话题 `topic_id` 与写入时间。`topic_id` 可以为空：交互提交被拒或应用没有成功时，这一轮仍写入会话，但不属于任何话题；会话 HTTP 响应的同名字段随之可空。下一轮 Gateway 话题路由失败时据此续接上一轮的话题，所以它取应用后实际所在的话题，而不是本轮开始时预期的话题。读取和追加都复制嵌套载荷，调用方不能改写历史。

载荷的交互内容只以事件集 `turn_events` 表达（以序号 0 的用户消息开头，工具调用连同结局记在同一个事件上，模型见[数据模型](../architecture/data-model.md)），另带本轮结局、实际使用的模型与附件。用户消息与回复正文（全部助手正文按序号以空行拼接）由载荷的只读属性从事件读出，不另存一份：会话标题、Gateway 的会话上文与会话 HTTP 的 `user_message`、`assistant_final_text` 都取自这两个属性。回复正文与流式 `done` 中 CPU 组装的最终回复可能只差分段间的空行。

它不是 Topic 的副本，三方各取所需：

- Alice 获取全量会话原文及检查点，自行构建预算受限的提示词视图（[Agent Runtime](../alice/agent-runtime.md#31-会话上下文视图)）；
- Gateway 获取最近 N 轮用户/助手文本（[Gateway 分析](../gateway/analysis.md)）；
- Topic 汇集可能来自多个会话的记忆材料，删除或结算 Topic 不删除会话。

## 3. 认领与容量

认证通过后注册入口同步创建或查找会话并认领，再登记进程；同一会话同时只允许一个进程，运行中不能删除，任何结局关闭后均释放认领。容量达到 `workspace.sessions` 上限时显式拒绝，不淘汰旧数据；失败注册回滚本次认领及自动创建的空会话。System 先关闭全部进程，再 `close_and_clear()`，此后读写拒绝。

## 4. 写入会话的时机

有输出的 completed/cancelled/failed 交互在 Actor 阶段结束、全部线程结束后封口并进入 FINALIZE；提交任务经 root API 提交交互、等待应用之后，才同步追加会话，话题取交互实际所在的话题。FINALIZE 期间会话里还没有这一轮，进程仍持有会话认领，同一会话的下一轮不会在此期间开始。会话原文与记忆提交相互独立：提交被拒或应用没有成功时照样追加，话题为空，进程随后按失败收口；已接纳写入也不回滚。无输出、进程异常与 FINALIZE 前交付方离开不追加。取消收口与提交任务的所有权见[任务进程](./task-process.md#8-封口部分输出与结算)。

## 5. 压缩检查点

会话另持有至多一份 CPU 中立的压缩检查点，结构见[子系统契约](../contracts/subsystem-contracts.md#421-会话压缩检查点)。摘要与工具减重属于执行视图，不替代或删除原始 `ConversationTurn`；Workspace 只负责存储和接受，不解释 CPU 的摘要算法。这使 CPU 可以替换，也避免 Alice 建立第二份跨 run 的会话状态。

`accept_checkpoint()` 是同步操作：校验完整 Workspace 归属、当前进程认领、两个非空游标都指向本会话已有轮次、减重边界不早于摘要覆盖，以及相对同一产生方与格式版本的旧检查点两个游标各自不回退。游标只在同一格式内可比：CPU 忽略无法解释的旧检查点并从头重建时，新检查点可以替换旧值，不会因游标更早而被永久拒绝。拒绝不改变已有值；进程记录告警后继续收尾。完整会话读取返回检查点、列表不返回，删除与关闭随会话清除。

进程在 Actor 阶段结束、全部线程结束之后，判断丢弃或无输出之前接受 CPU 返回结果中的检查点，同一次执行只接受一次；接受到正常封口之间没有 await。检查点只覆盖旧轮次，因此无输出、取消或交付方离开导致本轮不保存时仍可接受已完成的压缩；CPU 在停止后返回的取消结果同样可以携带它。普通执行异常没有返回结果，不接受任何候选值；进程 task 被硬取消而 CPU 把取消转为返回值时，先接受检查点再传播取消。压缩进度与 RuntimeEvent 都不是接受依据。

## 6. 已知限制

- 会话只在本进程内保存，不提供跨重启恢复或并发多进程存储；后端重启后前端需重新新建会话；会话持久化见[外部会话 Idea](../ideas/external-session-and-topic-projection.md)；
- 手动 `/compact` 随命令系统接回，见[会话 `/compact` 指令 Todo](../todo/conversation-compact-command.md)。

## 7. 代码与测试入口

- [`workspace/sessions/service.py`](../../src/hivememory/workspace/sessions/service.py)、[`RootSessionService`](../../src/hivememory/workspace/capability/root/session.py)、容量配置 [`config/workspace.py`](../../src/hivememory/config/workspace.py)；
- 测试：[`tests/unit/workspace/test_sessions.py`](../../tests/unit/workspace/test_sessions.py)、[`test_conversation_sessions.py`](../../tests/integration/workspace/test_conversation_sessions.py)（含提交后的轮次带交互实际所在的话题）、[`test_session_completion.py`](../../tests/integration/workspace/test_session_completion.py)、[`test_session_context.py`](../../tests/integration/workspace/test_session_context.py)、[`test_chat_session.py`](../../tests/e2e/system/test_chat_session.py)。
