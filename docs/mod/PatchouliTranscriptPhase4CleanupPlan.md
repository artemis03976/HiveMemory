# Patchouli Transcript Phase 4 Cleanup Plan

## 1. 文档目标

本文件是 transcript 双视图重构的 **Phase 4 单独清理文档**。

与前序文档 [`PatchouliTranscriptDualViewRefactor.md`](file:///c:/Users/29305/Projects/HiveMemory/docs/mod/PatchouliTranscriptDualViewRefactor.md) 不同，这份文档不再讨论双视图如何建立，而是聚焦于：

- 清理已经完成迁移后的兼容层
- 收敛冗余字段与废弃接口
- 移除旧时代字符串回退路径
- 把“结构化事件 + 双 builder”真正收束为唯一主路径

由于本阶段涉及文件较多、兼容代码分布分散，因此单独开文档管理。

---

## 2. 当前阶段判断

截至 Phase 1 ~ Phase 3 完成后，系统已经具备以下稳定主路径：

- 历史视图：
  - `TurnEvent` / `turn_events`
  - `HistoryTranscriptBuilder`
- generation 视图：
  - `GenerationContext`
  - `GenerationTranscriptBuilder`
- 感知入口：
  - `InteractionPayload.assistant_final_text + turn_events`
  - `SemanticFlowPerceptionLayer` 结构化优先 ingest

但当前代码里仍存在大量“兼容壳”和“旧字段共存”：

- 历史视图仍保留 `clean_response` fallback
- generation 仍保留 `context_messages` fallback
- 感知层仍保留 `assistant_message + MTPLogParser` 文本降级路径
- `PatchouliSystem` 仍保留 `_reconstruct_raw_assistant_text()`
- `LogicalBlock` 同时保留 legacy 三元组字段与 Kernel v3 字段

这些兼容层在过渡期是必要的，但继续保留会带来以下问题：

- 模型语义持续模糊
- 新旧字段容易产生不一致
- 测试矩阵膨胀
- 调试时难以判断当前到底走的是哪条主路径

Phase 4 的目标就是**完成收束**。

---

## 3. Phase 4 总目标

### 3.1 目标

将当前“双视图 + 兼容层”的中间态，收敛为：

- 历史视图唯一主路径：`turn_events -> HistoryTranscriptBuilder`
- generation 视图唯一主路径：`GenerationContext -> GenerationTranscriptBuilder`
- 感知层唯一主路径：`assistant_final_text + turn_events`
- 逐步移除旧字段：
  - `assistant_message`
  - `context_messages`
  - `clean_response` 的主路径语义
  - `raw_response` 的主路径语义
  - `to_stream_messages()` / `_blocks_to_messages()` 这类旧投影接口

### 3.2 非目标

Phase 4 不是：

- 再次重构业务规则
- 改写 extractor / deduplicator 的核心逻辑
- 一次性删除所有 legacy 模式数据结构
- 立即重塑 `LogicalBlock(turn=TurnRecord, ...)`

Phase 4 应优先做**兼容层与废弃代码清理**，而不是继续开新抽象。

---

## 4. 当前待清理对象清单

### 4.1 感知入口兼容层

当前仍保留的旧路径：

- [`system.py`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/patchouli/system.py)
  - `_reconstruct_raw_assistant_text()`
  - `_chat_post_process()` 中仍构造 `assistant_message`
- [`models.py`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/engines/perception/models.py)
  - `InteractionPayload.assistant_message`
- [`semantic_flow_perception_layer.py`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/engines/perception/semantic_flow_perception_layer.py)
  - 结构化路径中的 parser 防御性 fallback
  - 纯文本降级路径 `MTPLogParser.parse(payload.assistant_message)`

### 4.2 历史视图兼容层

当前仍保留的旧路径：

- [`history_transcript_builder.py`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/engines/perception/history_transcript_builder.py)
  - `block.clean_response` fallback
- [`context_converter.py`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/engines/perception/context_converter.py)
  - 仍承接旧 block 的兼容路径

### 4.3 generation 兼容层

当前仍保留的旧路径：

- [`generation/models.py`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/engines/generation/models.py)
  - `GenerationRequest.context_messages`
- [`generation/engine.py`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/engines/generation/engine.py)
  - `_render_transcript()` 的 `context_messages` fallback
  - `_format_transcript(messages)`
  - `_extract_identity()` 对 `context_messages` 的 fallback
- [`librarian_core.py`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/patchouli/kernel/librarian_core.py)
  - `_blocks_to_messages()`

### 4.4 `LogicalBlock` 冗余字段

当前 `LogicalBlock` 同时保留：

- Legacy 结构：
  - `user_block`
  - `execution_chain`
  - `response_block`
- Kernel / 双视图结构：
  - `user_query`
  - `assistant_final_text`
  - `turn_events`
  - `semantic_traces`
  - `clean_response`
  - `raw_response`

这些字段中，以下最值得在 Phase 4 重新定义或清理：

- `clean_response`
  - 不再作为历史视图/generation 视图主字段
- `raw_response`
  - 不再作为事实真相源
- `assistant_message`
  - 不再作为 ingest 主输入

### 4.5 兼容性风险点

已明确纳入 Phase 4 的问题：

- `GenerationTranscriptBuilder` 对 legacy block 的 identity 继承仍不完整
  - 目前只回退了 `user_query` / `response_block.content`
  - 未回退 `user_block.identity` / `response_block.identity`
- 若直接删除 fallback，而测试和 README 仍引用旧接口，将出现文档与实现脱节

---

## 5. Phase 4 建议拆分

由于清理范围较大，不建议一次性完成。

建议拆成 4 个子阶段：

1. Phase 4A：generation 兼容层清理
2. Phase 4B：感知入口兼容层清理
3. Phase 4C：历史视图兼容层清理
4. Phase 4D：模型字段与 legacy 结构收敛

这样做的原因：

- generation 链路相对封闭，最适合作为第一批清理对象
- 感知入口位于主链路中心，风险更高，应放在 generation 清理之后
- `LogicalBlock` 字段收敛改动面最大，适合最后处理

---

## 6. Phase 4A：generation 兼容层清理

### 6.1 目标

让 generation 彻底只消费：

- `GenerationRequest.context`
- `GenerationContext`
- `GenerationTranscriptBuilder`

不再依赖：

- `GenerationRequest.context_messages`
- `LogicalBlock.to_stream_messages()`
- `LibrarianCore._blocks_to_messages()`

### 6.2 要处理的文件

- [`generation/models.py`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/engines/generation/models.py)
- [`generation/engine.py`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/engines/generation/engine.py)
- [`generation/generation_transcript_builder.py`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/engines/generation/generation_transcript_builder.py)
- [`patchouli/kernel/librarian_core.py`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/patchouli/kernel/librarian_core.py)
- generation README 和相关测试

### 6.3 具体清理项

#### A1. 收敛 `GenerationRequest`

目标：

- 删除 `context_messages`
- 删除围绕 `context_messages` 的兼容注释与判断

建议演进：

```python
class GenerationRequest(BaseModel):
    context: GenerationContext
    write_focus: Optional[WriteFocus] = None
    update_focus: Optional[UpdateFocus] = None
```

同步处理：

- 删除 `has_context` 的“空 turns 特判”与 `context_messages` 共存语义
- 统一规定：
  - `context` 始终存在
  - `context.turns` 可为空

#### A2. 清理 `MemoryGenerationEngine` 旧入口

删除：

- `_format_transcript(messages)`
- `_render_transcript()` 中的 `context_messages` fallback
- `_extract_identity()` 中对 `context_messages[0].identity` 的 fallback

改为：

- 只从 `request.context` 渲染 transcript
- identity 只从：
  - `write_focus.identity`
  - `update_focus.identity`
  - `context.turns[0].identity`
  三者中确定

#### A3. 修复 legacy block identity 兼容

这个问题已明确延期到 Phase 4。

建议在 [`generation_transcript_builder.py`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/engines/generation/generation_transcript_builder.py) 中补上：

- `block.identity`
- `block.user_block.identity`
- `block.response_block.identity`

的回退优先级。

推荐优先级：

1. `block.identity`
2. `block.response_block.identity`
3. `block.user_block.identity`
4. `Identity()`

#### A4. 删除 `_blocks_to_messages()`

文件：

- [`librarian_core.py`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/patchouli/kernel/librarian_core.py)

动作：

- 删除 `_blocks_to_messages()`
- 清理所有相关注释

### 6.4 测试要求

必补：

- `GenerationRequest` 去除 `context_messages` 后的构造测试
- `MemoryGenerationEngine` 仅依赖 `context` 的主路径测试
- `Mode B/C` 在空 `turns` 但有 focus 时仍可正常工作
- `GenerationTranscriptBuilder` 对 legacy identity fallback 的测试

推荐回归：

- `tests/unit/generation/test_generation_transcript_builder.py`
- `tests/unit/patchouli/kernel/test_librarian_core.py`
- 任何 `WRITE/UPDATE` 链路测试

---

## 7. Phase 4B：感知入口兼容层清理

### 7.1 目标

让感知层 ingest 彻底只依赖：

- `assistant_final_text`
- `turn_events`
- `mtp_traces` / `MTPTraceReducer`

不再依赖：

- `assistant_message`
- `_reconstruct_raw_assistant_text()`
- `MTPLogParser` 作为运行时主入口

### 7.2 要处理的文件

- [`patchouli/system.py`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/patchouli/system.py)
- [`engines/perception/models.py`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/engines/perception/models.py)
- [`engines/perception/semantic_flow_perception_layer.py`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/engines/perception/semantic_flow_perception_layer.py)
- 相关 perception README / 单测

### 7.3 具体清理项

#### B1. 删除 `InteractionPayload.assistant_message`

前提：

- 所有 ingest 调用都已稳定提供 `assistant_final_text + turn_events`

动作：

- 删除 `assistant_message`
- 清理 docstring / README 中的旧表述

#### B2. 删除 `_reconstruct_raw_assistant_text()`

文件：

- [`system.py`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/patchouli/system.py)

动作：

- 删除 `_reconstruct_raw_assistant_text()`
- `_chat_post_process()` 不再构造 `raw_assistant_text`
- `InteractionPayload` 只提交结构化字段

#### B3. 收敛 `SemanticFlowPerceptionLayer.ingest_payload()`

当前存在：

- 结构化优先路径
- 文本解析 fallback 路径

Phase 4 建议：

- 主代码删除 `payload.assistant_message` 相关逻辑
- 保留 `MTPLogParser` 仅服务于：
  - 离线迁移脚本
  - 旧数据导入
  - 特殊测试夹具

也就是说：

- `MTPLogParser` 退出运行时主链路
- 不一定要立即删除整个类

### 7.4 测试要求

必补：

- `InteractionPayload` 新构造签名测试
- `SemanticFlowPerceptionLayer` 在无 `assistant_message` 情况下正常 ingest
- `PatchouliSystem._chat_post_process()` 不再依赖 raw assistant 重建

推荐回归：

- `tests/unit/engines/perception/test_dual_path_ingestion.py`
- `tests/unit/system/test_chat_logic.py`
- `tests/unit/system/test_chat_stream_memory_refs_schema.py`

---

## 8. Phase 4C：历史视图兼容层清理

### 8.1 目标

让历史视图彻底只依赖：

- `LogicalBlock.turn_events`
- `HistoryTranscriptBuilder`

不再依赖：

- `clean_response` fallback

### 8.2 要处理的文件

- [`history_transcript_builder.py`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/engines/perception/history_transcript_builder.py)
- [`context_converter.py`](file:///c:/Users/29305/Projects/HiveMemory/src/hivememory/engines/perception/context_converter.py)
- 相关 perception/history 单测

### 8.3 具体清理项

#### C1. 移除 `clean_response` fallback

删除：

- `HistoryTranscriptBuilder._render_block()` 中的：
  - `elif block.clean_response`

前提：

- 所有新写入 block 均保证带 `turn_events`
- 测试与 fixture 不再构造“只有 clean_response 的新 block”

#### C2. 收紧 `context_converter`

目标：

- `context_converter` 只作为轻量委托层存在
- 不再承诺兼容旧 block 的渲染语义

#### C3. 明确 `clean_response` 的剩余职责

一旦它不再参与历史视图：

- 需要判断是否仍保留：
  - token 估算
  - relay 展示
  - block 完整性判断

若以上逻辑都已迁移，则应继续推进到 Phase 4D 删除。

### 8.4 测试要求

必补：

- builder 在无 `turn_events` 时不再默默 fallback 的测试
- 主链路 block 始终具备 `turn_events` 的测试

推荐回归：

- `tests/unit/engines/perception/test_history_transcript_builder.py`
- 聊天主链路回归

---

## 9. Phase 4D：模型字段与 legacy 结构收敛

### 9.1 目标

清理跨阶段累计形成的模型冗余，使领域模型重新变清晰。

### 9.2 重点对象

#### D1. `LogicalBlock`

需要决定每个字段的最终命运：

- 保留：
  - `user_query`
  - `assistant_final_text`
  - `turn_events`
  - `semantic_traces`
  - `identity`
  - `write_focus`
  - `update_focus`
- 待删除或降级：
  - `clean_response`
  - `raw_response`
  - `user_block`
  - `execution_chain`
  - `response_block`

#### D2. `is_complete`

当前 `is_complete` 同时兼容：

- legacy 三元组 block
- `user_query + clean_response`

Phase 4 需要重新定义为：

- 主路径按 `user_query + turn_events/assistant_final_text` 判断
- legacy 兼容逻辑迁出主模型，或仅保留在迁移层

#### D3. `to_stream_messages()`

当前它仍是明确的 legacy/兼容壳。

Phase 4 建议：

- 若 generation 和 history 已全部脱离它
- 则删除该方法

#### D4. `raw_response` / `clean_response`

需要明确两者是否还需要在运行时保留：

- 若只剩调试价值，应迁移到调试日志或离线转换器
- 若仍有部分组件使用，应先完成该组件迁移

### 9.3 不建议本阶段直接做的事

虽然长期可以考虑：

- `LogicalBlock(turn=TurnRecord, ...)`

但这不建议作为本阶段强制动作。理由：

- 会扩大改动面
- 容易让“清理兼容层”与“再做一轮模型重构”混杂

Phase 4 更适合做减法，而不是再次引入大结构变化。

---

## 10. 推荐实施顺序

推荐顺序如下：

1. `4A generation 清理`
2. `4B 感知入口清理`
3. `4C 历史视图 fallback 清理`
4. `4D 模型字段收敛`

推荐原因：

- generation 已经有完整新路径，最容易先清理
- 感知入口是核心链路，需要在 generation 稳定后再删旧字段
- 历史视图 fallback 清理依赖于上游 block 写入已完全结构化
- 模型字段收敛改动面最大，放最后风险最低

---

## 11. 每个子阶段的完成标准

### 11.1 Phase 4A 完成标准

- `GenerationRequest.context_messages` 已删除
- `MemoryGenerationEngine` 不再接受 `context_messages`
- `_blocks_to_messages()` 已删除
- generation 单测全部改为 `GenerationContext`

### 11.2 Phase 4B 完成标准

- `InteractionPayload.assistant_message` 已删除
- `_reconstruct_raw_assistant_text()` 已删除
- `SemanticFlowPerceptionLayer` 不再依赖 `MTPLogParser` 处理运行时 payload

### 11.3 Phase 4C 完成标准

- `HistoryTranscriptBuilder` 不再 fallback 到 `clean_response`
- 所有历史视图测试均基于 `turn_events`
- `context_converter` 只保留轻量委托职责

### 11.4 Phase 4D 完成标准

- `LogicalBlock` 的主字段职责已重新定义
- 已删除至少一批明确废弃字段/方法
- README / 测试 /注释不再混用 legacy 术语和新术语

---

## 12. 风险与注意事项

### 12.1 测试夹具风险

很多单测仍手工构造 legacy block 或旧请求模型。

Phase 4 中最容易出现的问题不是运行时代码崩，而是：

- 测试夹具仍在构造旧字段
- 结果导致大量非功能性测试失败

建议：

- 每删一个兼容字段，就同步批量清理 fixture

### 12.2 README / 文档漂移

当前已有以下文档/说明可能漂移：

- perception README
- generation README
- 本次双视图重构文档

建议：

- 每个子阶段都同步更新 README
- 不要把文档清理拖到最后一次性处理

### 12.3 调试可观测性

删除 `raw_response` / `assistant_message` 前，需要确认：

- 是否仍有必要保留“原始轮次快照”的调试能力

若需要：

- 建议迁移到专门的 debug dump / trace log
- 不要继续挂在领域模型主字段上

---

## 13. 建议的测试清单

### 13.1 核心单测

- `tests/unit/generation/test_generation_transcript_builder.py`
- `tests/unit/patchouli/kernel/test_librarian_core.py`
- `tests/unit/engines/perception/test_history_transcript_builder.py`
- `tests/unit/engines/perception/test_dual_path_ingestion.py`
- `tests/unit/system/test_chat_logic.py`

### 13.2 推荐补充测试

- `GenerationTranscriptBuilder` 的 legacy identity fallback 测试
- `InteractionPayload` 新签名测试
- `SemanticFlowPerceptionLayer` 仅结构化 ingest 测试
- 删除 `clean_response` fallback 后的历史 builder 行为测试

---

## 14. 推荐提交策略

不建议将整个 Phase 4 合并为一个大提交。

建议最少拆为：

1. `feat/refactor`: generation 兼容层清理
2. `feat/refactor`: 感知入口兼容层清理
3. `feat/refactor`: 历史视图 fallback 清理
4. `feat/refactor`: LogicalBlock / 旧字段收敛

每一批提交都应满足：

- 对应子阶段单测通过
- README / 文档同步
- 无“半删半留”状态

---

## 15. 一句话总结

Phase 1 ~ 3 解决的是“新结构如何建立”。

Phase 4 要解决的是：

- 哪些兼容层已经完成使命，可以删掉
- 哪些旧字段不再代表真实语义，必须收束
- 如何让代码库从“新旧并存的过渡态”，真正进入“结构清晰的稳定态”

建议从 `generation` 兼容层开始，逐步向 `perception` 和 `LogicalBlock` 收敛，而不是一口气清理全部旧代码。
