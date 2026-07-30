---
title: 文档迁移最终收口审计
status: archived
owner: project
scope: documentation-migration-finalization
archived_at: 2026-07-29
superseded_by:
  - docs/DOCUMENTATION.md
  - docs/PROJECT.md
source_inventory: docs/plans/documentation-migration-inventory.md
---

# 文档迁移最终收口审计

本记录关闭文档迁移清单的最后一个 P2 批次。它不建立新的后端设计，而是为此前刻意留到最后的四类材料给出最终判断：开放 Ideas 是否仍与当前系统相容、仓库是否还存在平行源码 README、迁移前遗留的 `archive/mod/` 应归入哪里，以及全库入口与链接是否已经能够只指向唯一真相源。

## 1. 审计口径

收口仍遵守[文档治理规范](../../DOCUMENTATION.md)的逐篇迁移门禁：先辨别事实、理念、未来假设与失效主张，再决定保留、迁移或退役。一个模型、配置字段或接口形状已经存在，只能证明它可作为基础，不能证明围绕它的完整 Idea 已经实现；同样，Ideas 没有进入近期 Roadmap，也不意味着其中的问题背景和设计警告可以丢弃。

本批对照的主要当前入口包括 Patchouli Perception、Artifacts、Generation、Lifecycle、MemoryLibrary、Retrieval，Alice Agent Runtime 与 Orchestration，System Observability，以及 VISION 与数据模型。代码核验覆盖 Page Folding、ArtifactType、TopicMaterializeTask、MemoryAtom/RelationLayer、vitality/reinforcement/GC、RuntimeEvent、TurnEvent/AgentAction 和 CALL 编排。

## 2. Ideas 逐篇结论

### 2.1 Page Folding Raw Evidence

**分类与动作：** 继续保留为 Idea，不升级为 Plan。

**当前承接：** [感知与短期话题](../../patchouli/perception.md)已经如实记录 overflow 只 compact、不 settle、清空 blocks 且 `fold_retain_recent_blocks` 未接线；[Artifacts](../../patchouli/artifacts.md)描述当前四类 artifact 与 best-effort provenance；[记忆生成](../../patchouli/generation.md)描述真正的 settlement 输入和异步任务边界。

**保留理念：** Agent 工作集、不可变原始证据与记忆生成是三条不同语义流。保存原文不应让全部历史回流 active context，高保真后处理也不应阻塞 Agent 热路径。

**拒绝当作当前事实：** 没有 `FoldResult`、folded evidence artifact/raw turn store、`folded_artifact_refs` 或 high-fidelity settlement；现有 InteractionArtifact 不会自动捕获 overflow 前被清空的 blocks。

**升级门槛：** 需要先证明真实质量或审计损失，并定义保留/删除、用户隔离、容量、写入失败、去重和延迟基线，再建立跨 Perception/Artifacts/Generation 的独立 Plan。

### 2.2 Agent 轨迹与多 Agent TDA

**分类与动作：** 继续保留为研究 Idea，不进入 Alice 当前能力或 Roadmap。

**当前承接：** Agent Runtime 已有 sequence/action id 对齐的 TurnEvent、AgentAction 与 TraceItem；System 有带 trace/frame/run/task 关联的 RuntimeEvent；Alice 有单层、串行 CALL 及子 Agent 流事件。

**保留理念：** 应优先使用框架的确定性事件构图，TDA 初期只作为离线 monitor；H1 只能说明循环结构，不能脱离事件类型和任务结果判断好坏。

**拒绝当作当前事实：** 当前没有持久化 trajectory graph、依赖边、filtration、persistent homology、Topology Monitor 或拓扑控制器；RuntimeEvent 也不保证持久、全量或跨进程连续，Alice 不具备动态 DAG 和自治多 Agent 网络。

**升级门槛：** 先建立可脱敏重放数据集与普通计数基线，用成功、成本、失败类型和人工质量标签证明拓扑特征有稳定增益；只有离线证据成立后才讨论运行时干预。

### 2.3 Memory-Centric Agent TDA

**分类与动作：** 继续保留为研究 Idea，不进入 Patchouli 当前能力。

**当前承接：** MemoryAtom 有七种正式 MemoryType、identity/visibility/lifecycle 元数据和轻量关系预留；Creation/Version Artifact 可记录 source memory refs；Retrieval 已拥有 dense、sparse、vitality 与时间相关基础信号。

**保留理念：** 记忆是带类型、来源、关系和生命周期的可演化资产，而不是平面 chunks；语义、使用、依赖、冲突与时间应被视为不同视图，不宜提前压成单一边权。

**拒绝当作当前事实：** RelationLayer 不是图数据库或关系一致性契约；没有图索引、反向边、共同检索耐久历史、冲突分类器、TDA analyzer 或 topology-aware retrieval。Idea 中的 ProjectMemory/WorkflowMemory 等是研究词汇，不能扩张当前 MemoryType；示例字段也不能反写进 schema。

**升级门槛：** 先用当前 schema 构造只读最小投影，区分确定性边与推断边，再与 flat hybrid retrieval 和普通 graph metrics 比较；只有结构信号稳定、可行动且遵守 identity/provenance 时才建立 Plan。

### 2.4 Vitality 长期演进

**分类与动作：** 继续保留为 Idea，不把五个候选方向拆成 Todo 或伪排期。

**当前承接：** [记忆生命周期](../../patchouli/lifecycle.md)拥有当前三段式公式、四种强化事件、gardening、低水位 archive 和显式 revive；[MemoryLibrary](../../patchouli/memory-library.md)拥有实际冷热状态转移。

**保留理念：** 当前启发式公式需要真实使用数据校准；被反复使用、用户明确保护、复习历史和相似记忆干扰都可能成为未来信号，但“信号更多”不天然意味着检索质量更好。

**纠正的设计矛盾：** ShortTerm 是 topic buffer，不是 MemoryAtom 初级层；MemoryAtom 生成后进入 MidTerm；LongTerm 是退出普通检索的 archive store，不是高价值成熟层。因此“短→中→长晋升”不能建立在当前三层命名上。Consolidation 若要继续，必须先定义合并、版本、Artifact 或新状态中的哪一种语义，并保持 Lifecycle 提供信号、MemoryLibrary 管状态、Generation/专用组件处理内容的所有权。

**升级门槛：** 先持久化并校准真实事件数据，每次只引入一个状态，给出 schema migration、回滚和旧 atom 默认值；review signal 不得替代 query relevance，vitality 也不得被解释为事实正确性。

## 3. 残余 README 与零散材料

全仓复核发现，`src/` 以及 frontend、applications、scripts、tests 等源码与工具目录没有仍在维护的 Markdown/README。此前位于 `src/hivememory/engines/{perception,generation,retrieval,lifecycle}/README.md` 的四篇设计型说明已经逐篇承接并迁入 `archive/legacy-docs/source-readmes/engines/`，其索引继续保留原来源路径。

仓库根 `README.md` 与 `README_EN.md` 是当前项目入口，不属于源码局部说明，也不应为消除所有 README 而迁入 docs。归档区中的 README 则承担历史分类索引；它们不与当前模块设计竞争，因此继续保留。

## 4. `archive/mod/` 退役

迁移前位于 `docs/archive/mod/` 的 `EnableLifecycleMaintenanceDesign.md` 是已经完成的生命周期维护接入稿。其全局 scheduler、HIT/CITATION/feedback、gardening 与 archive/revive 理念已经由当前 Lifecycle 和 System Runtime 文档承接；正文中的阶段、旧路径、未来 API 与“主要剩余工作”只保留为历史上下文。

该文件现迁入[历史实施计划](./implementation/enable-lifecycle-maintenance.md)。原 `archive/mod/README.md` 只有一个指针，已经被 Archived Plans 索引完整取代，没有独立历史价值，因此退役；含义模糊的 `archive/mod/` 分类不再保留。

## 5. 最终入口与有意保留的例外

- 当前事实统一从 PROJECT、Architecture、Contracts、System、Gateway、Patchouli、Alice 与 Frontend 进入；
- 未来工作只从 Plans、Ideas、Todo、Applications 和 Roadmap 进入；
- 历史实现、旧架构和其他替代文档只从 Archive 进入；
- 原 `docs/protocols/i18n/MemoryCompilerI18nMigrationPlan.md` 当前位于 `docs/archive/legacy-docs/protocols/i18n/MemoryCompilerI18nMigrationPlan.md`，其既有编码损坏已在第 4～6 节审计中采用原字节保留。本批继续把它作为唯一已知 UTF-8 例外，不从损坏正文提取事实，也不以批量格式化改写。

## 6. 最终验证门禁

收口完成前执行以下检查：

1. 对除上述已知损坏归档外的全部 `docs/**/*.md` 做严格 UTF-8 解码；
2. 解析相对 Markdown 链接并验证目标存在；
3. 搜索 `docs/mod/`、`docs/archive/mod/`、旧 Engines/源码 README 等退役路径；
4. 检查 frontmatter 状态、主索引和迁移清单是否仍声称存在后续批次；
5. 运行 `git diff --check`，并以 Perception、Artifacts、Lifecycle、Alice Runtime 与 System Runtime 定向测试确认审计所依据的行为仍成立。

最终结果：

- `docs/` 共 138 篇 Markdown；已知损坏归档按原字节跳过，其余 137 篇严格 UTF-8 解码通过；
- 共检查 785 个相对 Markdown 链接，没有缺失目标；外部 URL、页内 anchor、`file://` 历史链接和已知损坏归档不计入相对目标数；
- `docs/mod/` 与 `docs/archive/mod/` 均已退役，当前索引没有指向退役路径的链接；归档正文中的旧路径只作为来源和演化证据保留；
- PROJECT、ROADMAP、Ideas、Archive、Archived Plans 与迁移清单均已改为完成口径，迁移清单状态为 `completed`；
- `git diff --check` 通过；
- Perception、Artifacts、Lifecycle、MemoryLibrary、Patchouli services、Agent Runtime、Alice Runtime 与 System Runtime 定向测试共 333 项通过。

至此，迁移清单的最后一个 P2 批次关闭。后续发现新的旧文档或设计偏差时，应按常规文档治理流程逐篇处理，不重新开启含义模糊的总迁移目录。
