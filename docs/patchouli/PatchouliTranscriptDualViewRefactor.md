---
title: Legacy Patchouli Transcript Dual-View Refactor
status: superseded
owner: patchouli
scope: completed-structured-transcript-refactor
archived_at: 2026-07-28
superseded_by:
  - docs/patchouli/perception.md
  - docs/patchouli/generation.md
  - docs/patchouli/artifacts.md
---

> 本文保留 transcript 从字符串回填转向结构化事实的重构记录，已停止维护。当前 TurnRecord/LogicalBlock、GenerationContext 与 InteractionArtifact 的分工分别以[感知与短期话题](./perception.md)、[记忆生成](./generation.md)和[Artifacts](./artifacts.md)为准。

# Patchouli Transcript 重构设计与落地总结

**状态**: 当前权威文档  
**适用范围**: `patchouli` 对话主链路、`perception` 感知层、`generation` 记忆生成链路、被动 ingest 子系统

---

## 1. 文档定位

本文是当前 Patchouli transcript / 消息流重构的单一权威文档。

它合并并替代了此前两份文档：

- `PatchouliTranscriptDualViewRefactor.md`
- `PatchouliTranscriptPhase4CleanupPlan.md`

之所以合并，是因为：

- 早期设计草案仍保留大量旧链路描述，已经与代码实现不一致
- Phase 4 清理计划中的大部分事项已经完成，不再适合作为进行中文档继续存在
- 后续维护更需要一份“当前设计 + 已落地实现 + 剩余可选清理项”的统一说明

---

## 2. 这次重构到底解决了什么

本轮重构的核心目标，不是继续给旧字符串链路打补丁，而是建立统一的运行时结构化真相源，并让不同消费侧从同一份事实派生不同视图。

最终完成的关键目标包括：

1. 运行时直接采集结构化 `turn_events`，不再依赖最终文本的逆向解析
2. 将“一轮交互事实”和“感知层容器”解耦，明确模型分层
3. 统一主动模式与被动 ingest 模式的消息流入口
4. 让历史视图与记忆生成视图共享底层事实，但保留各自独立的渲染逻辑
5. 清理旧时代的 `assistant_message`、`context_messages`、旧 observer buffer 等兼容主路径

---

## 3. 最终设计结论

### 3.1 单一真相源

当前消息流以“单轮结构化事件”作为第一真相源：

- `TurnEvent`: 最小交互原子
- `AgentAction`: 由 `TurnEvent` 聚合出的动作
- `TraceItem`: 面向 perception / generation 的语义轨迹摘要
- `TurnRecord`: 一轮交互的内容真相记录

### 3.2 感知层容器

`LogicalBlock` 不再承担“万能 DTO”角色，而是收敛为：

- `TurnRecord` 作为内容真相
- 感知层元数据作为容器附加信息

可以简写为：

```text
LogicalBlock = TurnRecord + Perception Metadata
```

### 3.3 协议层入口

`InteractionPayload` 已不再属于 `engines.perception.models` 的领域模型，而是提升为 Patchouli 系统级协议模型：

- 定义位置：`src/hivememory/patchouli/protocol/models.py`
- 职责：承载主动 / 被动入口流向感知层的一轮交互传输协议

### 3.4 双视图策略

同一轮结构化事实当前派生出两类消费视图：

- 历史消息视图：偏向可重放、保留工具调用痕迹
- 记忆生成视图：偏向摘要化、便于生成长期记忆

它们共享底层事实，但不再共享同一种粗粒度字符串表示。

---

## 4. 当前模型分层

### 4.1 核心交互模型

位置：`src/hivememory/core/models/interaction.py`

当前承载：

- `TurnEvent`
- `AgentAction`
- `TraceItem`
- `TurnRecord`
- `ActionReducer`
- `TraceReducer`
- `Identity`

这一层表示项目级交互模型，不属于 perception 私有领域。

### 4.2 协议模型

位置：`src/hivememory/patchouli/protocol/models.py`

当前承载：

- `InteractionPayload`
- `EyeGazeResult`
- `ChatResult`
- 其他系统内部协议消息

这一层表示 Patchouli 体系内部模块间通信的协议契约。

### 4.3 感知层领域模型

位置：`src/hivememory/engines/perception/models.py`

当前仍由 perception 持有的核心模型：

- `LogicalBlock`
- `SemanticBuffer`
- `FlushEvent`
- `FlushReason`
- `ArchivePayload`

它们属于感知层的处理容器、缓冲与归档语义。

### 4.4 generation 视图模型

位置：`src/hivememory/engines/generation/models.py`

当前承载：

- `GenerationTurn`
- `GenerationContext`
- `GenerationRequest`

这部分已经从旧的 `context_messages` 文本链路切到结构化 generation 视图。

---

## 5. 当前端到端链路

### 5.1 主动模式

当前主链路如下：

```text
User
-> TheEye.gaze()
-> PatchouliKernel / LoopExecutor
-> 运行时采集 turn_events + final_text
-> PatchouliSystem 构造 InteractionPayload
-> LibrarianCore.submit_interaction()
-> SemanticFlowPerceptionLayer.route_and_ingest()
-> LogicalBlock(turn=TurnRecord, ...)
-> History / Generation 两类消费视图
```

关键点：

- `LoopExecutor` 直接产出结构化 `turn_events`
- `PatchouliSystem` 直接提交 `assistant_final_text + turn_events`
- 感知层不再接受 `assistant_message` fallback

### 5.2 被动 ingest 模式

当前被动模式已经接入同一条结构化主链：

```text
External Events
-> PassiveObserverIngressor
-> MessageTurnBuffer
-> InteractionPayload
-> LibrarianCore / Perception
```

关键点：

- `PassiveObserverIngressor` 已从 `TheEye` 中独立出来
- `MessageTurnBuffer` 会直接构建 `turn_events`
- 多 session 分桶通过 `PassiveSessionKey` 处理
- 被动模式也与主动模式一样提交结构化 `InteractionPayload`

---

## 6. 当前正式字段与事件语义

### 6.1 `InteractionPayload`

当前 `InteractionPayload` 的主字段为：

- `identity`
- `user_message`
- `rewritten_query`
- `assistant_final_text`
- `turn_events`
- `mtp_traces`
- `write_focus`
- `update_focus`
- `worth_saving`

已删除：

- `assistant_message`

### 6.2 `TurnEvent.kind`

当前正式事件类型为：

- `user_message`
- `assistant_message`
- `thought`
- `tool_call`
- `tool_result`
- `system_message`

说明：

- 代码里仍保留少量旧事件名兼容映射，便于旧数据或过渡测试平滑升级
- 这些兼容映射不再代表主路径设计

---

## 7. 两个消费视图的当前职责

### 7.1 历史消息视图

当前由 `HistoryTranscriptBuilder` 承担主职责。

目标：

- 从 `TurnRecord.turn_events` 重放出可用于历史上下文的消息序列
- 保留工具调用及必要身份前缀
- 在缺少结构化事件的历史数据上仅做有限 fallback

主路径：

```text
LogicalBlock.turn.turn_events
-> HistoryTranscriptBuilder
-> StreamMessage[]
```

### 7.2 记忆生成视图

当前由 `GenerationTranscriptBuilder` 承担主职责。

目标：

- 将 `LogicalBlock` 抽象成 generation 友好的结构化视图
- 保留用户问题、最终回复、语义轨迹、摘要上下文
- 不再依赖旧的 `context_messages`

主路径：

```text
LogicalBlock[]
-> GenerationTranscriptBuilder
-> GenerationContext / GenerationTurn
-> GenerationRequest
```

---

## 8. 已完成的关键清理

### 8.1 感知入口

- 删除 `InteractionPayload.assistant_message`
- 删除感知层对 `assistant_message` 的 fallback 消费
- 感知层只接受 `assistant_final_text + turn_events`

### 8.2 generation 侧

- 删除 `GenerationRequest.context_messages`
- generation 主路径改为 `GenerationContext`
- 历史文本消息不再作为 generation 主输入

### 8.3 历史视图

- `HistoryTranscriptBuilder` 成为历史重放主路径
- 历史视图从 `turn_events` 派生，不再依赖旧的 `clean_response` 主路径

### 8.4 模型归属

- `TurnEvent` / `AgentAction` / `TraceItem` / `TurnRecord` 已提升到 `core/models`
- `InteractionPayload` 已提升到 `patchouli/protocol`
- `LogicalBlock` 已收敛为 `turn: TurnRecord`

### 8.5 被动模式

- 旧 `gateway observer_buffer` 已删除
- 被动 observer 编排已独立为 `PassiveObserverIngressor`
- 被动模式与主动模式共用结构化 payload 主链

---

## 9. 仍保留的有限兼容层

当前仍保留少量“非主路径兼容层”，主要用于旧数据或测试迁移：

- `TurnEvent` 的旧事件名标准化映射
- `TraceReducer` 对旧 kind / 旧字段的兼容吸收
- `LogicalBlock` 对旧扁平字段构造的提升兼容
- 少量历史测试 helper 对 legacy 字段的探测逻辑

这些兼容层的意义是：

- 让重构能分阶段落地
- 降低对现有测试、旧 fixture、旧归档数据的破坏面

它们不再是推荐接口，也不应继续扩散到新代码。

---

## 10. 剩余可选清理项

如果后续要继续做“最终封口”，建议按如下顺序推进：

### 10.1 文档与测试语义清理

- 清理测试注释中残留的 `clean_response`、`assistant_text` 等旧术语
- 清理测试 helper 对 `user_block/response_block` 的旧探测逻辑
- 保证文档与当前代码路径一致

### 10.2 运行时兼容桥接收口

- 删除 `TurnEvent` 的旧名映射
- 删除 `TraceReducer` 的 compat 分支
- 删除 `LogicalBlock` 的扁平字段自动提升

### 10.3 API 面继续收紧

- 视需要继续收紧顶层公共导出
- 避免继续从宽泛入口导入系统内部模型

---

## 11. 推荐维护原则

后续若继续演进消息流，请遵守以下原则：

1. 新协议字段优先加在 `patchouli.protocol`，不要回塞到 perception 领域模型中
2. 新的交互事实优先落到 `TurnEvent` / `TurnRecord`，不要新增字符串回填主路径
3. 历史视图与 generation 视图只共享底层事实，不共享“凑出来的字符串表示”
4. 新代码不要再依赖 `assistant_message`、`clean_response`、`context_messages` 之类已退场的主路径字段
5. 若必须保留兼容逻辑，应显式标注为兼容层，并限制在最外圈

---

## 12. 一句话总结

这次 Patchouli transcript / 消息流重构已经完成从“文本逆向解析主导”到“结构化运行时事实主导”的迁移：

- `TurnEvent` 是事件真相
- `TurnRecord` 是单轮真相
- `InteractionPayload` 是系统协议
- `LogicalBlock` 是感知层容器
- 历史视图与 generation 视图共享事实，但各自独立渲染

后续工作的重点已经不再是主链路搭建，而是收紧剩余兼容层与清理旧术语。
