---
title: Topic Content Emptiness and Manual Lifecycle Semantics
status: todo
owner: patchouli-system
scope: topic-content-emptiness-and-manual-lifecycle-semantics
related_docs:
  - docs/patchouli/perception.md
  - docs/plans/v0.6.2-workspace-mvp.md
  - docs/todo/page-folding-cross-ingress-follow-ups.md
  - docs/todo/topic-compact-command-ingress.md
  - docs/ideas/PatchouliPageFoldingRawEvidenceDesign.md
  - docs/governance/testing/test-design-standards.md
last_reviewed: 2026-08-22
---

# Topic 内容判空与手动生命周期语义修复

## 当前结论

该事项应在 `v0.6.2 W0-P5` 的 `TopicAssetBinding` 实现前独立完成。它修复现有 Topic 内容判空和手动生命周期语义，不引入 WorkspaceAsset、binding、Context Compiler 或新的 Workspace 命名分区，也不得修改 memory generation controller 的 `wait_all` 语义。

本事项冻结以下产品语义：

- Topic 是否为空由 `blocks` 与 `state_summary` 共同决定；只有两者都没有有效内容时，buffer 才真正为空；
- binding 不参与 Topic 内容判空。P5 的正常绑定只允许发生在至少完成一轮对话之后；
- TriggerManager 的三个原子动作是 `settle / compact / evict`，不是 `settle / delete / evict`；
- “删除话题”不触发记忆写入，但必须结束 Topic 生命周期；
- “结算话题”必须触发记忆资产生成，并在结算材料被可靠接纳后结束 Topic 生命周期；
- 手动 compact 是独立操作，只压缩工作集，不结算、不驱逐 Topic；它不再与手动 settle 混用；
- 所有 compact 路径都必须保证 `retain_recent_blocks >= 1`，当前 generation 至少拥有一个近期 block；
- Topic 领域不再使用“归档/archive”指代 settle；`archive` 保留给中期记忆进入长期记忆库的既有操作。

## 问题与证据

### 1. 当前判空只检查 blocks

`TopicData.is_empty` 当前等价于 `not blocks`。ShortTermMemoryStore 的 `include_empty=False`、`discard_if_empty`、TriggerManager、manual settle 和 shutdown flush 也存在直接或间接的同类判断。

这会把以下合法状态错误识别为空 Topic：

```text
blocks = []
state_summary = "已经折叠的历史内容"
```

该状态表示 Topic 刚完成压缩、尚未产生新的近期对话；它仍然保存可用于继续对话的历史工作集，不应从活跃话题列表消失，也不应被空 Topic 清理误删。

需要审计的现有入口至少包括：

- `src/hivememory/core/models/topic.py` 的 `TopicData.is_empty`；
- `src/hivememory/patchouli/memory_library/stores.py` 的 `list_topic_data(..., include_empty=False)` 与 `get_buffer_info()`；
- `src/hivememory/engines/perception/semantic_flow_perception_layer.py` 的 `discard_if_empty()`；
- `src/hivememory/engines/perception/trigger_manager.py` 的空 Topic 提前返回；
- `src/hivememory/patchouli/services/perception.py` 的 manual settle 与 shutdown flush；
- `src/hivememory/patchouli/memory_library/buffer.py` 的 `get_topic_summary()` 展示语义。

### 2. 当前 MANUAL 同时执行 settle 与 compact，却不 evict

当前决策矩阵将 `FlushReason.MANUAL` 定义为：

```text
settle = true
compact = true
evict = false
```

其实际效果是：先把全部 blocks 汇总为 `state_summary`，再因 settle 清空 blocks，但将 Topic 留在活跃池中。这既制造了 summary-only buffer，也与前端“结算话题”的产品语义冲突：前端会乐观地从列表移除 Topic，但刷新后后端仍可能再次返回它。

后端现有 `POST /topics/{topic_id}/settle`、`settle_topic` 和 `manual_settle_topic` 方向正确；前端仍使用 `archiveTopic()`、Archive 图标、英文 aria label 及“归档失败”等旧称。这里的漂移应由前端收敛到 settle 术语解决，不应反向把后端接口改成 archive。

### 3. 真正空 Topic 的 evict 也会被提前跳过

TriggerManager 当前在读取不到 Topic 或 `topic_data.is_empty` 时统一提前返回。这使内容为空但需要结束生命周期的 Topic 无法执行 `evict=True`，影响 IDLE、LRU、SHUTDOWN 和 manual settle 行为。

“没有可 settle/compact 的内容”不能等价为“没有需要执行的生命周期动作”。Topic 存在时必须先解析决策矩阵；即使没有 settlement payload，`evict=True` 仍要移除 buffer。

## 内容判空契约

规范内容判断为：

```text
has_blocks = len(blocks) > 0
has_state_summary = bool(state_summary.strip())
has_content = has_blocks OR has_state_summary
is_empty = NOT has_content
```

状态矩阵如下：

| blocks | state_summary | `is_empty` | 说明 |
|:---:|:---:|:---:|:---|
| 无 | 无 | 是 | 真正空 Topic |
| 有 | 无 | 否 | 仅有原始近期对话 |
| 无 | 有 | 否 | 仅有压缩历史 |
| 有 | 有 | 否 | 压缩历史与近期工作集并存 |

约束：

- 空白字符串不构成有效 `state_summary`；
- `has_blocks()` 如继续存在，必须保持“是否有原始 block”的窄语义，不能静默改成 `has_content()`；
- 所有列表、discard、settlement admission 和 shutdown 判断必须消费同一内容语义，不能继续分散实现 `if buf.blocks`；
- binding 不作为内容。P5 的正常 binding 发生在至少一轮对话之后，理论上不会形成 `binding-only` Topic；本 TODO 通过删除无生产用例的 `reset_topic_content()`，避免人为制造“blocks 与 summary 均空、仅保留 binding”的矛盾状态。

## 决策矩阵与用户操作

三个原子动作固定为：

```text
settle
compact
evict
```

目标矩阵为：

| 触发原因 | Settle | Compact | Evict | 语义 |
|:---|:---:|:---:|:---:|:---|
| `TOKEN_OVERFLOW` | 否 | 是 | 否 | 自动压缩后继续对话 |
| `IDLE_TIMEOUT` | 是 | 否 | 是 | 空闲结算并结束 Topic |
| `LRU_EVICTION` | 是 | 否 | 是 | 为新 Topic 腾出容量前结算 |
| `SHUTDOWN` | 是 | 否 | 是 | 停机前结算并清空活跃池 |
| `MANUAL_SETTLE` | 是 | 否 | 是 | 用户结算为记忆资产并结束 Topic |
| `MANUAL_COMPACT` | 否 | 是 | 否 | 用户压缩后继续对话 |
| `MANUAL_DELETE` | 否 | 否 | 是 | 用户丢弃 Topic，不写记忆 |

`MANUAL_DELETE` 可以作为正式 FlushReason 进入统一 resolver，也可以继续由公开 delete 用例调用窄 evict 原语；无论采用哪种实现，最终可观察行为必须与矩阵一致。

### 删除话题

目标行为：

```text
不构造 settlement
不生成 state_summary
从活跃池移除 Topic
```

当前 `DELETE /topics/{topic_id}` 已接近该语义。实现时应保留 delete 与 settle 的明确区分，不能把“是否生成记忆”隐藏在一个布尔参数中。

### 结算话题

目标行为：

```text
冻结当前 settlement 材料
可靠接纳 memory generation task（存在可写材料时）
结束 Topic 生命周期
不为后续连续对话执行 compact
```

公开协议与用例继续使用 settle：

```text
POST /topics/{topic_id}/settle
```

`settle_topic()` 正是该 Topic 生命周期用例：将当前内容冻结为记忆生成材料，可靠提交后结束 Topic，而不是原地封存 Topic。前端 service/store、组件图标、aria label 和失败文案必须从 archive/归档改为 settle/结算；不新增 `/archive` 路由。

settle 成功不能只由 `MemoryGenerationTask | None` 决定。真正空 Topic、所有 blocks 均被 `worth_saving=False` 过滤等情况都可能没有任务；只要 Topic 生命周期按契约结束，应用层仍应返回成功，并以可选 `task_id` 或 `generation_submitted` 表达是否建立后台任务。

应按领域审计 `MemoryGenerationSource.ARCHIVE`、settlement coordinator 的 `source=...ARCHIVE` 以及 Artifact/source intent 中与 Topic settlement 有关的旧命名，只修改“Topic 内容形成记忆资产”这一语义。真正表示“中期记忆进入长期记忆库”的 archive 命名不在此次重命名范围内。

### 手动 compact

目标行为：

```text
不构造 settlement
不触发 memory generation
合并 previous state_summary 与可折叠旧前缀
裁剪已经进入摘要的旧 blocks
保留最近工作集和 Topic
```

不能直接复用当前 `retain_recent_blocks=None` 的 MANUAL compact。该路径会总结全部 blocks，却依赖后续 settle 才清空 blocks；拆成 pure compact 后会让同一内容同时出现在 `state_summary` 与完整 blocks 中。

手动 compact 应显式使用已配置的 `fold_retain_recent_blocks`，像 TOKEN_OVERFLOW 一样调用带裁剪与 token 重算的 `apply_compaction()`。若 blocks 数量不大于保留数，应返回可观察的 no-op 结果，不生成重复摘要。

建议为未来系统命令预留语义明确的入口，例如：

```text
POST /topics/{topic_id}/compact
```

本 TODO 不要求同时增加新的前端按钮或命令解析器，但内部生命周期能力和公共协议名称不得继续把 compact 与 settle 混为一体。`/compact` 的前端用户入口、当前 Topic 上下文传播与 Gateway 聚合由独立的 `topic-compact-command-ingress.md` 跟踪，不阻塞本事项完成。

## 实现前必须处理的边界

### 1. Summary-only 与 compact 最小保留量

把 summary-only Topic 判定为非空，不等于当前 Generation 已能单独结算它：

- TriggerManager 当前从最后一个可结算 block 获得 actor identity；无 block 时无法构造相同 provenance；
- `_build_settle_payload()` 当前在没有可结算 block 时返回 `None`；
- MemoryGenerationCoordinator 当前在 `GenerationContext.turns` 为空时跳过提交，即使 `state_summary` 非空；
- summary-only 的 InteractionArtifact、原始证据和重复结算语义仍未定义。

本事项采用最小边界：summary-only 参与 Topic 列表、路由、idle 与生命周期判断，但不在本修复中新增独立的 summary-only memory/artifact 生成能力。当前 generation 路径继续以 `state_summary + 至少一个 recent block` 作为可结算材料。为避免 compact 主动制造无法按现有 provenance 结算的 summary-only Topic，冻结以下跨层约束：

- manual settle 不再 compact，并在接纳成功后 evict；
- 所有公开配置、schema、命名方法和内部 compact 路径都必须满足 `retain_recent_blocks >= 1`；
- 传入 `0` 或负数必须在明确的输入边界以具体异常拒绝，不能静默提升为 1，也不能只依赖某个上层默认值；
- manual compact 与 TOKEN_OVERFLOW 等路径均至少保留一个最新 block，并保持配置指定的更大保留量；
- 完整 summary-only settlement、identity 与 raw evidence 语义继续由 `page-folding-cross-ingress-follow-ups.md` 跟踪。

该约束与 `docs/ideas/PatchouliPageFoldingRawEvidenceDesign.md` 的后续方向兼容：未来 folded blocks 可写入独立的 append-only raw evidence，默认 active context 仍保持 `state_summary + recent_blocks`，高保真 generation 再按 evidence refs 读取完整历史。当前阶段不实现 raw evidence，也不假定 active buffer 永久保留全部 blocks；待该设计落地后，再单独评估是否需要放宽“至少保留一个 recent block”的临时规避约束。

### 2. settle 任务接纳与 evict 顺序

当前 TriggerManager 在返回 settlement payload 前已经清 blocks 并 pop buffer，而 PerceptionFamiliar 随后才通过 local bus 提交 generation task。若任务接纳抛出异常，用户会收到 settle 失败，但 Topic 已经丢失，无法重试。

用户主动 settle 的提交顺序冻结为：

```text
冻结 settlement payload
  -> generation admission 成功
  -> evict Topic
```

generation admission 失败必须向用户抛出受控错误，并保持 Topic 仍可见，且 buffer、blocks 与 `state_summary` 完整不变，以便用户重试；任何 clear/pop 都不得发生在 admission 成功之前。任务被接纳后，冻结 payload/spec 已取得材料所有权；后台生成后续失败不要求恢复 Topic。若没有可提交材料，settle 仍可直接 evict 并返回成功。

实现可以为 manual settle 增加窄的 prepare/admit/evict 协调，也可以重构通用 settlement primitive，但不得为此改变 controller `wait_all`、引入跨组件 service locator，或在未验证的情况下顺带改变 IDLE/LRU/SHUTDOWN 的失败语义。

### 3. 删除 `reset_topic_content()`

`reset_topic_content()` 当前只有测试调用，正式代码没有消费者；产品逻辑也不存在“完全重置已有 Topic 后继续复用”的真实用例。需要清空当前会话时应结束旧 Topic 并新建 Topic，而不是保留一个身份不变、内容被抹除的壳。

因此实现时应优先删除 `src/hivememory/patchouli/memory_library/stores.py` 中的 `reset_topic_content()`：

- 同步删除仅为该生产方法存在的测试，不保留测试专用生产 API；
- 若测试需要构造特定状态，使用 `clear_blocks()`、`update_summary()`、`apply_compaction()` 等真实命名能力的组合，或直接建立边界清晰的 fixture；
- 不得删除仍保护真实业务行为的测试；必要覆盖应迁移到对应真实能力的行为测试；
- 删除后再次审计生产与测试引用，确保没有死 API、兼容别名或仅为旧方法保留的 mock。

移除该方法后，P5 无需再为“reset 后保留 binding 是否形成 binding-only Topic”设计额外状态：正常 binding 不会早于首轮对话，已有 Topic 也没有原地彻底重置的合法转换。

## 预期改动范围

内容判空修复主要涉及：

- `src/hivememory/core/models/topic.py`；
- `src/hivememory/patchouli/memory_library/buffer.py`；
- `src/hivememory/patchouli/memory_library/stores.py`；
- `src/hivememory/engines/perception/semantic_flow_perception_layer.py`；
- `src/hivememory/engines/perception/trigger_manager.py`；
- `src/hivememory/patchouli/services/perception.py`。

手动操作语义拆分还可能涉及：

- `FlushReason`、Perception 接口与 Patchouli local/public routes；
- Patchouli/System Topic application service；
- Server Topic router 与 response model；
- `frontend/src/services/topicApi.ts`、Topic store 与 TopicCard 的 settle 命名、图标及用户文案；
- Topic settlement 使用的 `MemoryGenerationSource`、Artifact/source intent 等领域命名；
- 当前 Perception 文档和 `v0.6.2-workspace-mvp.md` 中的矩阵描述。

移除测试专用重置能力还涉及：

- `src/hivememory/patchouli/memory_library/stores.py`；
- 当前调用 `reset_topic_content()` 的 memory library 与 perception layer unit tests。

明确不在范围内：

- WorkspaceAssetStore、TopicAssetBinding 的实际实现；
- cache、queue、scheduler、registry、EventBus 的 Workspace 分区；
- controller `wait_all` 语义；
- summary-only Artifact/provenance 与 raw evidence 的完整实现；
- 上传、Context Compiler 或 Artifact promotion。

## 测试计划

所有测试遵循 `docs/governance/testing/test-design-standards.md`，以可观察结果证明状态迁移，不能只断言矩阵常量、mock 调用次数或 mock 返回值。异步时序使用确定性 fake、event/barrier 或有界轮询，不使用固定 sleep。

### Unit

| 建议位置 | 场景 | 必须捕获的缺陷 |
|:---|:---|:---|
| `tests/unit/core/models/` | `blocks/state_summary` 四种组合的内容判空 | summary-only 再次被标为空，或空白 summary 被误作内容 |
| `tests/unit/patchouli/memory_library/` | `include_empty=False` 包含 summary-only、排除真正空 Topic | Store 继续直接按 `buf.blocks` 过滤 |
| `tests/unit/engines/perception/` | 三轴决策映射、pure compact 与 `retain_recent_blocks >= 1` | manual settle 继续 compact/keep，manual compact 总结后不裁剪旧 blocks，或 0 被静默接受 |
| `tests/unit/server/routers/`、`tests/unit/system/application/` | settle/delete/compact 的公开响应和缺失 Topic 错误 | settle 继续保留 Topic，或用 task 是否存在误判生命周期成功 |
| `tests/unit/patchouli/memory_library/`、`tests/unit/engines/perception/` | 删除 reset 专用测试并由真实命名能力覆盖状态迁移 | 为测试继续保留无生产消费者的 API，或误删真实行为覆盖 |

矩阵常量断言只能作为补充；主要断言必须落到 Topic 是否仍存在、state summary/blocks 如何变化、是否产生可查询 task 等行为结果。

### Integration

扩展 `tests/integration/patchouli/test_perception_flush_chain.py`，使用真实 `ShortTermMemoryStore + SemanticFlowPerceptionLayer + TriggerManager + PerceptionFamiliar`，只将 Relay/LLM 与 generation admission 边界替换为确定性 fake：

- summary-only Topic 仍出现在非空活跃 Topic 列表；
- `discard_if_empty` 不删除 summary-only Topic；
- truly-empty Topic 在 IDLE/SHUTDOWN/manual settle 下仍被 evict；
- manual settle 接纳 generation task 后才从 Topic 池移除；
- manual settle admission 失败时抛出受控错误，Topic 仍存在且 blocks、state summary 完整不变，可再次重试；
- manual delete 从 Topic 池移除且不产生 generation task；
- manual compact 更新 state summary、裁剪旧前缀、保留最近 blocks 与 Topic，且不产生 generation task；
- manual compact 不让同一旧前缀同时残留在 summary 与 blocks 中；
- 所有可配置 compact 入口对 `retain_recent_blocks=0` 或负数抛出具体异常，至少保留一个最新 block。

这些测试不需要 `real_infra`、`live_llm` 或 `slow` 标记。

### 回归与文档

- 更新当前依赖 `FlushReason.MANUAL` 旧语义的 unit、integration 与 deterministic e2e 用例；
- 保持 TOKEN_OVERFLOW、IDLE、LRU、SHUTDOWN 既有可观察语义，除“真正空 Topic 仍需按矩阵 evict”外不扩大修改；
- 修正 `docs/plans/v0.6.2-workspace-mvp.md` 中 `settle/delete/evict` 和 `settle/delete 矩阵` 的笔误；
- 拆分 P5 生命周期表中的 `manual settle / compact`，分别记录 manual settle、manual compact 和 manual delete；
- 更新 `docs/patchouli/perception.md` 中 MANUAL、summary-only 和手动 settle/evict 的旧描述。

## 完成条件

- Topic 内容判空统一为 `blocks OR non-blank state_summary`，所有已识别入口不再自行用 `blocks` 替代内容判断；
- summary-only Topic 可被列出、继续路由并免于空 Topic 误删；
- 真正空 Topic 在 `evict=True` 的触发下仍能结束生命周期；
- 决策矩阵及文档统一使用 `settle / compact / evict` 三轴；
- delete、settle、manual compact 成为三个命名明确、行为互不混杂的用例，Topic settlement 不再错误使用 archive 术语；
- manual settle 不 compact，仅在可靠接纳结算材料后 evict；admission 失败向用户报错且 Topic 内容完整保留，无任务时的 settle 结果不被错误报告为生命周期失败；
- manual delete 不提交记忆生成并最终 evict；
- manual compact 不提交记忆生成、不 evict、不重复承载已折叠内容，并保证 `retain_recent_blocks >= 1`；
- 所有 compact 配置与内部入口拒绝小于 1 的 retain 值，相关行为测试证明至少保留一个最新 block；
- `reset_topic_content()` 及仅为它存在的测试被移除，必要覆盖已迁移到真实命名能力且没有生产引用或死兼容层残留；
- 真正的中期记忆到长期记忆 archive 术语保持不变；只清理 Topic settlement 领域的旧 archive 命名；
- P5 计划中的 binding 生命周期矩阵与上述 Topic 生命周期语义一致；
- 新增或修改的测试符合测试设计规范，窄回归、相关 unit/integration 集合及配置要求的 PR 快速集通过。
