---
title: 文档迁移第 7 节逐篇审计：Patchouli 与 Engines
status: archived
owner: project
scope: documentation-migration-audit-section-7
archived_at: 2026-07-29
superseded_by:
  - docs/archive/plans/documentation-migration-inventory.md
  - docs/DOCUMENTATION.md
source_inventory: docs/archive/plans/documentation-migration-inventory.md
---

# 第 7 节逐篇审计：Patchouli 与 Engines

本记录是归档动作的门禁，而不是新的 Patchouli 设计入口。它逐一说明清单第 7 节的旧文档：哪些事实已经进入当前文档、哪些设计理念仍应保留、哪些历史主张明确拒绝继承，以及原文完成物理迁移后的路径。初始范围为清单中的十一篇文档；随后按用户追加范围纳入 `src/hivememory/engines/` 下四篇零散 README，但仍不延伸到第 8 节及以后、`docs/mod/` 或其他源码目录。

## 1. 审计口径

审计同时检查三种内容：

1. **事实**：以当前代码、配置、测试和跨子系统契约为准，确认旧文档中的模型、调用顺序、失败边界和限制是否仍然成立；
2. **理念**：保留能够帮助读者理解取舍的隐喻、问题背景和边界理由，但把它们放入当前文档对应章节，而不是让历史计划继续充当规范；
3. **未来承诺**：尚未实现的能力降级为限制、Todo 或 Idea；不能因为旧文档写得完整，就把它们升级成当前能力。

本节的当前真相源是 [Patchouli 总览](../../patchouli/README.md)及其七篇模块文档。`src/hivememory/engines/` 仍然保存算法实现，但 `docs/engines/` 不再与 Patchouli 并列作为子系统文档树。

## 2. 逐篇审计结果

### 2.1 `docs/patchouli/README.md`

- **分类与动作**：`current`，保留并核验；本轮补审时间为 2026-07-29。
- **已承接内容**：保留“大图书馆”作为知识所有权隐喻，明确 Patchouli 拥有短期话题、MemoryAtom、Artifacts、后台生成、检索、生命周期和 MemoryCompiler；同时明确 Gateway、Alice、System 的非职责。入口还列出了模块、代码路径、契约和真实限制。
- **需要拒绝的旧口径**：不再把 Patchouli 等同于一个拥有所有运行时职责的 `LibrarianCore`，也不把 `engines/` 的目录布局当作子系统边界。
- **结论**：正文已经是当前总览；物理迁移时仅移动同目录下的三篇旧设计，不移动本文件。

### 2.2 `ActiveMemoryGenerationDecouplingDesign.md`

- **分类与动作**：`merge`，合并后归档。
- **已承接内容**：`patchouli/generation.md` 已记录主动 WRITE/UPDATE 从 Perception 脱离、Mode B/C、finalize 顺序、focus 只作为生成意图、PendingAtom settlement 以及任务控制边界。
- **补充的设计理由**：旧链路把 MTP WRITE/UPDATE 当作话题结算，执行 `archive + compact + clear`，导致连续主动写只能看到被压扁的 summary，并在每次主动写时重复丢失原始细节。当前直接复用最近 topic context 与 `WriteFocus`/`UpdateFocus`，主动写与被动归档保持两条正交流；重复由 deduplicator 处理，主动 task 独立记录 InteractionArtifact 是已知 provenance 成本。
- **明确不继承**：`FlushReason.MTP_WRITE/MTP_UPDATE`、URGENT flush、感知层 focus/pending 字段及“主动写即清空 buffer”不再是当前语义。
- **替代入口**：[记忆生成](../../patchouli/generation.md)、[感知与短期话题](../../patchouli/perception.md)。

### 2.3 `PatchouliPassiveIngestRefactorPlan.md`

- **分类与动作**：`merge`，合并后归档。
- **已承接内容**：System Passive Ingress 文档承担 session、idle、事件提交、seal/retry 和降级；Patchouli Perception 文档承担 `InteractionPayload`、TurnRecord/TurnEvent、Action/Trace reducer、话题 buffer、settlement 与 Page Folding。
- **补充的设计理由**：`MessageTurnBuffer` 接收 `user`、`assistant`、`tool_call`、`tool_result`；只有自然语言 assistant 段落写入 `assistant_final_text`，工具过程必须保留为 TurnEvent。`target_topic` 是 user 到达时的 Gateway route 决策，Buffer 在此时绑定并在下一轮 user 到来前先 flush 旧轮，避免下一轮 gaze 污染上一轮归属。System Passive Ingress 拥有被动会话与提交编排，TheEye/Gateway 不重新拥有 observer buffer。
- **明确不继承**：旧 `system.ingest()` 的扁平 `user/assistant` 入口、`ObserverSessionBuffer`、`assistant_message` fallback 和 TheEye 私有被动主链均已退出。
- **替代入口**：[被动摄入](../../system/passive-ingress.md)、[感知与短期话题](../../patchouli/perception.md)。

### 2.4 `PatchouliTranscriptDualViewRefactor.md`

- **分类与动作**：`merge`，合并后归档。
- **已承接内容**：`TurnRecord` 保存事实，`LogicalBlock` 增加 Perception 元数据，Generation 从 blocks 构造 `GenerationContext`，Artifact 保存交互来源；当前文档也说明工具事件不能从拼接文本逆向恢复。
- **保留的设计理念**：同一轮交互应有“事实视图”和面向任务的“工作视图”；上下文压缩是工作集维护，不是原始证据的替代品；长期保存通过 settlement/artifact 链完成。
- **明确不继承**：`context_messages`、`assistant_message` fallback、完整 transcript 字符串作为所有消费者共同输入，以及“当前文档即唯一权威”的旧状态声明。
- **替代入口**：[感知与短期话题](../../patchouli/perception.md)、[记忆生成](../../patchouli/generation.md)、[Artifacts](../../patchouli/artifacts.md)。

### 2.5 `docs/engines/README.md`

- **分类与动作**：`merge`，退役归档。
- **已承接内容**：其中的模块列表、引擎职责和入口关系已经拆入 Patchouli 总览及 Perception/Generation/Retrieval/Lifecycle/MemoryCompiler 文档；Gateway 引擎内容由 Gateway 当前索引承接。
- **明确不继承**：`engines/` 不再是独立子系统真相源；算法目录的存在不代表它拥有跨模块的控制面、存储所有权或契约定义。
- **替代入口**：[Patchouli](../../patchouli/README.md)、[Gateway](../../gateway/README.md)。

### 2.6 `docs/engines/perception.md`

- **分类与动作**：`merge`，合并后归档。
- **已承接内容**：当前 Perception 文档已按 `SemanticFlowPerceptionLayer`、TriggerManager、RelayController、ShortTermMemoryStore 和 Page Folding 的真实代码重建职责、结算矩阵、LRU/idle/shutdown 触发和限制。
- **保留的设计理念**：感知层是短期语义 MMU/工作集，不是长期记忆仓库；summary 用于接力，不应伪装成原始证据；结算、压缩和驱逐是不同动作。
- **明确不继承**：旧文档中的固定容量/三级存储目标、过时 MMU 所有权、自动长期保存承诺，以及任何由损坏正文推导出的接口结论。原文包含明显 U+FFFD 编码损坏，当前事实未从损坏段落直接复制。
- **替代入口**：[感知与短期话题](../../patchouli/perception.md)、[MemoryLibrary](../../patchouli/memory-library.md)。

### 2.7 `docs/engines/generation.md`

- **分类与动作**：`merge`，合并后归档。
- **已承接内容**：三种生成模式、extractor/deduplicator、CREATE/UPDATE/TOUCH/DISCARD、Artifact 与中期库写入、任务状态机和失败隔离均已进入 `patchouli/generation.md` 与 `patchouli/artifacts.md`。
- **保留的设计理念**：记忆生成是冷路径；显式保存意图必须有 fallback；去重决定演化动作而不是机械创建；计算、持久化和任务控制应保持分层。
- **明确不继承**：早期“完整渲染器/固定阈值/自动归档”等目标，以及把 GenerationEngine 当作事件发布器或存储拥有者的描述。
- **替代入口**：[记忆生成](../../patchouli/generation.md)、[Artifacts](../../patchouli/artifacts.md)。

### 2.8 `docs/engines/retrieval.md`

- **分类与动作**：`merge`，合并后归档。
- **已承接内容**：Dense/Sparse/Fusion/Rerank 管线、查询变体、缓存、top-k、identity 约束、Retrieval 与 MemoryCompiler 的分工，以及 tags/time_range/keywords 的实现缺口已进入当前文档。
- **保留的设计理念**：检索是候选发现而非事实裁定；先召回再编译；top-k、分数与权限在 Retrieval 侧决定，Compiler 只生成任务视图。
- **明确不继承**：旧 renderer 作为 RetrievalEngine 内部万能组件、检索 miss 自动 revive 长期记忆、完整 XML/Markdown renderer 和“所有过滤器已接线”的口径。
- **替代入口**：[记忆检索](../../patchouli/retrieval.md)、[MemoryCompiler](../../patchouli/memory-compiler.md)。

### 2.9 `docs/engines/lifecycle.md`

- **分类与动作**：`merge`，合并后归档。
- **已承接内容**：MemoryLibrary 的 formal/pending/archive/trash 角色、vitality、reinforcement、gardening、archive/revive 以及事件与失败边界已进入当前 Lifecycle 与 MemoryLibrary 文档。
- **保留的设计理念**：生命力是维护排序信号而非事实真相；归档与唤醒是显式状态转移；后台 gardening 必须可解释、可取消并保留审计线索。
- **明确不继承**：固定 confidence 参与 vitality 公式、固定阈值和“用户输入永远 1.0 immutable”等旧目标；archive/revive 非事务，不能描述成原子迁移；不会因 retrieval miss 自动 revive。
- **替代入口**：[记忆生命周期](../../patchouli/lifecycle.md)、[MemoryLibrary](../../patchouli/memory-library.md)。

### 2.10 `docs/engines/memory_compiler/README.md`

- **分类与动作**：`merge`，退役归档。
- **已承接内容**：MemoryCompiler 当前入口、source/IR/target 分层、unit 与 envelope target、调用点和 RUNNABLE_TOOL 的 reserved 状态均已集中到 `patchouli/memory-compiler.md`。
- **明确不继承**：MemoryCompiler 不拥有 Retrieval、权限、Agent frame、bus、store 或工具执行；`engines/memory_compiler` 不再作为独立文档入口。
- **替代入口**：[MemoryCompiler 当前设计](../../patchouli/memory-compiler.md)。

### 2.11 `Phase1RenderConvergenceDesign.md`

- **分类与动作**：`merge`，合并后归档。
- **已承接内容**：MemoryAtom/PendingAtom/ResolveResult 的统一 IR、`PROMPT_FULL`/`PROMPT_INDEX`/`MTP_READ`/embedding targets、retrieval context 的 Full/Cascade/Compact 策略、i18n 与预算限制已逐项核验并进入当前 Compiler 文档。
- **补充的设计理由**：单条记忆的 unit 表达与 retrieval/READ/shared context 的 envelope 包装是两种责任；拆开后才能避免 renderer 同时污染检索预算、权限策略和 Agent-facing 文案。
- **明确不继承**：MTP `RUN` 可执行编译、完整多格式 renderer、懒加载已稳定化等未来能力；当前 `RUNNABLE_TOOL` 仍显式抛出 reserved 错误，Full 与 Cascade/Compact 的预算口径也仍不一致。
- **替代入口**：[MemoryCompiler 当前设计](../../patchouli/memory-compiler.md)。

## 3. 源码目录 README 补充审计

### 3.1 `src/hivememory/engines/perception/README.md`

- **分类与动作**：平行真相源，合并后归档。
- **已承接内容**：LogicalBlock 保持一次 Query、工具过程和最终响应的结构边界；Page Folding 负责工作集接力；idle/LRU/shutdown 由上层维护用例触发结算。这些理念和当前限制已经进入 Perception 与 System runtime 文档。
- **明确不继承**：Embedding“语义吸附”、`StreamParser`、私有 `IdleTimeoutMonitor`、`on_flush_callback`、`MTP_WRITE/MTP_UPDATE` URGENT flush、`assistant_message` fallback 与 `hivememory.perception` 旧导入路径均不再是当前实现。
- **替代入口**：[感知与短期话题](../../patchouli/perception.md)、[System runtime](../../system/runtime-and-bus.md)。

### 3.2 `src/hivememory/engines/generation/README.md`

- **分类与动作**：平行真相源，合并后归档。
- **已承接内容**：LLM 用于跨表达方式的价值判断和结构化提取，deduplicator 决定 CREATE/UPDATE/TOUCH/DISCARD，Engine 只返回纯计算 outcome，持久化与 settlement 由 Familiar/控制面负责。
- **补充的设计理由**：规则触发难以表达长期价值和记忆结构，LLM 更灵活但引入延迟、成本与不确定性，所以生成位于后台冷路径，显式 WRITE 保留 fallback，LLM 不拥有最终承诺。
- **明确不继承**：`ConversationBuffer`、`triggers.py`、消息数/idle/semantic trigger、旧 storage 注入方式、固定相似度矩阵、未经验证的性能指标与旧 `hivememory.generation` 导入路径。
- **替代入口**：[记忆生成](../../patchouli/generation.md)。

### 3.3 `src/hivememory/engines/retrieval/README.md`

- **分类与动作**：平行真相源，直接归档。
- **已承接内容**：Retrieval 返回 atoms 和检索元信息，Agent-facing context 由 MemoryCompiler 编译；这一边界已是当前 Retrieval 文档的核心原则。
- **明确不继承**：`query.py`、`router.py`、`searcher.py` 的旧文件清单，“Stage 2 全部完成”、时间解析/类型识别均已接线，以及旧 `RetrievalResult` 示例不能继续作为当前 API。
- **替代入口**：[记忆检索](../../patchouli/retrieval.md)、[MemoryCompiler](../../patchouli/memory-compiler.md)。

### 3.4 `src/hivememory/engines/lifecycle/README.md`

- **分类与动作**：平行真相源，合并后归档。
- **已承接内容**：vitality 使用 0～100、Calculator 保持纯计算、读取可用 `persist=False` 临时刷新、gardening 先持久刷新再交给 Collector、事件入口和 best-effort 强化边界均已进入当前 Lifecycle 文档。
- **补充的设计理由**：Collector 不持有 Calculator 或存储读取，避免与计算器/存储形成循环依赖，并使 LifecycleEngine 保持唯一编排者；同时记录当前没有跨会话的逐用户反馈状态。
- **明确不继承**：`PatchouliService.finalize_agent_run()`、`LibrarianCore.run_gardening_once()`、成功用户记忆 RUN、`resurrect_memory()` 和旧公共 API 片段已与当前 Familiar/MemoryLibrary 边界不符。
- **替代入口**：[记忆生命周期](../../patchouli/lifecycle.md)、[MemoryLibrary](../../patchouli/memory-library.md)。

## 4. 物理迁移结果

审计通过后，旧文档按下列结构移动，保留文件名与核心原文以便追溯：

```text
docs/archive/legacy-docs/patchouli/
  ActiveMemoryGenerationDecouplingDesign.md
  PatchouliPassiveIngestRefactorPlan.md
  PatchouliTranscriptDualViewRefactor.md
docs/archive/legacy-docs/engines/
  README.md
  perception.md
  generation.md
  retrieval.md
  lifecycle.md
  memory_compiler/README.md
  memory_compiler/Phase1RenderConvergenceDesign.md
docs/archive/legacy-docs/source-readmes/engines/
  perception/README.md
  generation/README.md
  retrieval/README.md
  lifecycle/README.md
```

归档后的相对链接统一指向当前 Patchouli、System、Gateway、mod、agent_runtime 和源码入口；归档目录本身不再被任何当前文档作为规范引用。`engines/perception.md` 的损坏正文按原样保留，仅修正迁移后的替代入口链接，未以损坏段落补写当前设计。

## 5. 验证门禁

- [x] 11 篇清单文档与 4 篇追加源码 README 均给出承接位置、保留理念和拒绝继承项；
- [x] 当前 Patchouli 文档补齐主动生成解耦、被动结构化摄入、`target_topic` 绑定和 unit/envelope 分层理由；
- [x] 物理目标均位于 `docs/archive/legacy-docs/` 内；
- [x] 移动后已完成 Markdown 相对链接、旧路径残留、严格 UTF-8 与 `git diff --check` 复核；唯一 U+FFFD 文件是已知损坏的归档 `engines/perception.md`。
- [x] Patchouli/Engines 定向测试共 386 项通过。
