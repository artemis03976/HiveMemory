---
title: Topic Compact Command Ingress
status: todo
owner: gateway-patchouli
scope: manual-topic-compact-slash-command-and-client-context
related_docs:
  - docs/todo/topic-content-emptiness-and-manual-lifecycle.md
  - docs/gateway/commands.md
  - docs/patchouli/perception.md
  - docs/governance/testing/test-design-standards.md
last_reviewed: 2026-08-22
---

# Topic `/compact` 系统指令接入

## 排期与非阻塞关系

本事项是 Topic manual compact 状态机分离完成后的独立用户入口工作，不阻塞 `topic-content-emptiness-and-manual-lifecycle.md` 的实现、验收或后续 P5 工作。

前置事项只需提供一个命名明确、可独立调用的 manual compact 用例，并保证该用例：

- `settle=False`；
- `compact=True`；
- `evict=False`；
- `retain_recent_blocks >= 1`；
- 不提交 memory generation task。

本 TODO 不重新实现或修改上述状态机，只负责让用户能够从聊天前端输入 `/compact`，并把该入口可靠地聚合到同一个 manual compact 用例。若前置事项尚未提供 HTTP compact 路由，本事项可以同时补齐适配器，但 HTTP 路由不构成 slash command 的内部实现。

## 当前缺口

Gateway 已有统一的系统指令 registry、parser、dispatcher 与 `command_result` 终态，`/compact` 应复用该链路，不能由前端自行判断输入字符串后绕开 Gateway。

当前仍缺少以下能力：

- 内置命令中没有 `/compact`；
- `ChatRequest` 没有携带前端当前选中的 Topic ID；
- command 分支在 Gateway 中早于普通 Topic routing 短路，因此 `/compact` 无法通过语义路由推测目标 Topic；
- `CommandDispatchInput` 与 `SystemCommandDispatcher` 当前只传递 actor `Identity`，没有把现成的 `WorkspaceAccessContext` 和独立的目标 Topic ID 传给资源写操作；
- Patchouli 尚未公开 manual compact 的统一 global route；
- 前端收到 compact 成功结果后没有刷新 Topic snapshot 的明确行为。

前端当前持有的 `currentTopicId` 只是最近一次聊天路由结果，不是服务端授权事实。它可以指定操作目标，但后端仍必须结合 `WorkspaceAccessContext` 重新验证资源边界。

## 冻结设计

### 1. 两个入口，一个用例

`/compact` 与 HTTP API 是不同协议适配器，不是两套业务逻辑：

```text
聊天输入 /compact
  -> Gateway CommandRegistry / SystemCommandDispatcher
  -> PATCHOULI_MANUAL_COMPACT_TOPIC global route
  -> Patchouli TopicManagementService.compact_topic()
  -> TOPIC_MANUAL_COMPACT local route

POST /topics/{topic_id}/compact
  -> Server Topic router / System TopicApplicationService
  -> PATCHOULI_MANUAL_COMPACT_TOPIC global route
  -> Patchouli TopicManagementService.compact_topic()
  -> TOPIC_MANUAL_COMPACT local route
```

唯一的业务聚合点是 Patchouli `TopicManagementService.compact_topic()`。它负责访问边界校验、调用 manual compact 原语并形成稳定结果；HTTP router、System TopicApplicationService、Gateway dispatcher 与前端均不得复制 retain、summary、generation 或 evict 规则。

禁止为了复用 HTTP 接口而在同一进程内发起 HTTP 自调用。两个入口应在 transport 以下汇入同一个 Patchouli public route。

### 2. `/compact` 的命令定义

注册公开内置命令：

```text
command_id = patchouli.topic.compact
primary_name = /compact
category = patchouli
route_target.kind = global_route
route_target.name = patchouli.public.manual_compact_topic
```

该输入是确定性的系统控制消息：命中后必须短路，不进入普通 Topic Router、retrieval、Alice、MTP 或 active memory generation。`PASSIVE_MEMORY` 仍不得触发该命令。

前端可以提供命令提示或自动补全，但不得维护一套独立的 `/compact` 解析和执行规则。未来若增加“压缩当前话题”按钮，按钮可以使用 HTTP API，并继续汇入同一用例。

### 3. 当前 Topic 的显式传播

为 active chat 请求增加可选的 `current_topic_id`，由前端 `chatStore.currentTopicId` 提供：

```text
chatStore.currentTopicId
  -> ChatRequest.current_topic_id
  -> ChatApplicationService
  -> GatewayService / GatewayWorkflow request context
  -> CommandDispatchInput
  -> SystemCommandDispatcher global route payload
  -> TopicManagementService.compact_topic(topic_id=...)
```

约束：

- `current_topic_id` 是命令操作目标，不是 `IdentityScope` 或 `WorkspaceAccessContext` 的组成部分；
- 不得把 Topic ID 塞入 identity 模型，也不得据此扩展 cache、queue、scheduler 或 registry 的命名域；
- 后端不能把客户端 Topic ID 当作授权凭据，必须使用同一次请求已经冻结的 `WorkspaceAccessContext` 访问 Topic；
- 没有 `current_topic_id` 时返回稳定的 `REJECTED` 结果，例如 `topic.compact.no_current_topic`，不得选择“最新 Topic”或让 LLM 猜测；
- Topic 不存在或不属于当前 Workspace 时，返回不泄漏跨域资源信息的稳定拒绝结果；
- 本事项不要求支持 `/compact <topic_id>`；如未来增加显式参数，仍必须经过相同访问校验。

Dispatcher 应传播既有 `WorkspaceAccessContext`，而不是仅从中降级取出 actor `Identity`。这只是把当前请求已有的访问上下文送达资源写边界，不改变 identity scope 的定义，也不引入新的 Workspace 分区。

### 4. 结果与前端同步

Patchouli manual compact 用例返回结构化结果，至少表达：

```text
topic_id
changed
retained_block_count
```

可以按现有 Topic snapshot 契约附带更新后的摘要信息，但前端不得依赖本地复算 compaction 结果。

用户可观察语义：

- 完成实际压缩时返回 `COMPLETED`，Topic 继续存在，前端刷新其 snapshot；
- blocks 数量不超过 retain 数时返回成功 no-op，`changed=False`，不重复生成 summary；
- 缺少当前 Topic、Topic 不存在或访问边界不匹配时返回稳定拒绝结果；
- compact 内部异常返回 `FAILED`，原 Topic 不应因该指令被 settle 或 evict；
- 命令结果通过现有 SSE `command_result` 展示，不伪装成普通 Agent 回复，也不把 `/compact` 写入 Topic 对话 blocks；
- 指令完成后前端应按结构化结果刷新 Topic store，不能靠解析本地化 message 判断是否成功。

如采用 `client_action` 触发刷新，应定义通用的 `refresh_topics` 动作；如直接消费 command result，则应按稳定的 `command_id/status/data` 处理。两种方式只能选择一种作为公共前端契约，不能同时维护互相竞争的刷新规则。

## 预期改动范围

后端可能涉及：

- `src/hivememory/gateway/commands/builtins.py`；
- `src/hivememory/gateway/commands/dispatcher.py`；
- `src/hivememory/gateway/workflow/state.py` 与 `topology.py`；
- `src/hivememory/gateway/service.py`；
- `src/hivememory/server/models/chat.py` 与 chat router；
- `src/hivememory/system/application/chat_service.py`；
- `src/hivememory/system/application/topic_service.py` 与 Topic router；
- `src/hivememory/system/contracts/route_names.py`、`routes.py`；
- `src/hivememory/patchouli/contracts/public_routes.py`、runtime bridge；
- `src/hivememory/patchouli/application/topic_management_service.py`；
- Patchouli manual compact local route 与既有 perception 能力的绑定。

前端可能涉及：

- `frontend/src/types/chat.ts`；
- `frontend/src/services/chatApi.ts`；
- `frontend/src/stores/chat/chatStore.ts` 与 command result reducer；
- Topic store 的 snapshot 刷新；
- 可选的 OmniInput 指令提示或自动补全。

明确不在范围内：

- 修改 manual compact 的 `settle / compact / evict` 决策；
- summary-only generation 或 raw evidence；
- Topic settle/delete 的生命周期修复；
- 把 `current_topic_id` 合并进 `IdentityScope`；
- 新增 Workspace cache/queue/scheduler/registry 分区；
- controller `wait_all`；
- 新的前端按钮或 `/compact <topic_id>` 参数形式。

## 测试计划

所有测试遵循 `docs/governance/testing/test-design-standards.md`，断言公开结果和 Topic 状态迁移，不以 mock 调用次数证明命令成功。

### Unit

- Gateway command registry 能确定性识别 `/compact`，并在 PASSIVE_MEMORY 中不把它作为可执行系统指令；
- 缺少 `current_topic_id` 时产生具体 `REJECTED` 结果，且不会进入普通 chat/Topic routing；
- command result 的成功、no-op、拒绝和失败映射具有稳定 status、error code 与结构化 data；
- Chat request 到 Gateway command context 的字段转换不会丢失 Topic ID 或把它写入 identity 模型；
- 前端收到成功结果后刷新 Topic snapshot，收到拒绝或失败时保留当前 Topic 选择并展示结构化错误。

每项测试必须指出其捕获的生产缺陷，例如“命令误入 LLM”“无当前 Topic 时误压缩最新 Topic”“成功后前端继续显示旧 summary”，不能只断言 definition 常量或 mock 参数。

### Integration

使用真实 `CommandRegistry + GatewayWorkflow + GlobalSystemBus + Patchouli TopicManagementService`，外围 summary/LLM 依赖使用确定性 fake：

- `/compact` 对显式当前 Topic 执行一次 manual compact，更新 summary、裁剪旧前缀、至少保留一个近期 block，并保留 Topic；
- no-op 路径不重复摘要、不 settle、不生成 memory task、不 evict；
- 越域或不存在的 Topic 被拒绝，其他 Workspace 的 Topic 状态不变；
- HTTP compact 与 `/compact` 对相同初始状态产生相同的 Topic 可观察终态，而不是各自维护不同规则。

这些测试不需要 `real_infra`、`live_llm` 或 `slow` 标记。

### Deterministic E2E

从 `/api/v1/chat` 提交 `/compact`，消费真实 SSE `command_result/done`，并通过公开 Topic 查询验证：目标 Topic 的 snapshot 已更新、Topic 仍存在、没有进入 Agent 生成链。该用例使用确定性 compaction fake 或协议级 fake LLM，不调用真实 Provider 和基础设施。

## 完成条件

- 该事项未被加入前置生命周期修复或 P5 的阻塞完成条件；
- `/compact` 由后端统一 command registry 识别，前端没有自建字符串执行分支；
- 当前 Topic ID 被作为独立请求目标显式传播，不进入 identity scope；
- command 资源写操作收到并使用完整 `WorkspaceAccessContext`；
- 缺少目标、目标不存在和 Workspace 不匹配均有稳定且不泄漏信息的拒绝结果；
- `/compact` 与 `POST /topics/{topic_id}/compact` 汇入同一个 Patchouli `compact_topic()` 用例；
- compact 的成功和 no-op 都不 settle、不生成记忆、不 evict，且至少保留一个近期 block；
- 前端按结构化 command result 刷新 Topic snapshot，不解析 message 文本；
- 指令不会进入 Topic Router、retrieval、Alice、MTP 或 active memory generation；
- unit、integration 与 deterministic e2e 以可观察行为验证上述契约，相关 PR 快速集通过。
