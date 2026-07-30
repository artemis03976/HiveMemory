---
title: 文档迁移第 8 节逐篇审计：Alice 与 Agent Runtime
status: archived
owner: project
scope: documentation-migration-audit-section-8
archived_at: 2026-07-29
superseded_by:
  - docs/archive/plans/documentation-migration-inventory.md
  - docs/DOCUMENTATION.md
source_inventory: docs/archive/plans/documentation-migration-inventory.md
---

# 第 8 节逐篇审计：Alice 与 Agent Runtime

本记录是归档动作的门禁，而不是新的 Alice 设计入口。它逐一说明清单第 8 节的十篇文档：哪些事实已经进入当前文档、哪些设计理念仍应保留、哪些历史主张明确拒绝继承，以及原文完成物理迁移后的路径。本批不延伸到第 9 节 Gateway、`docs/mod/`、其他源码 README 或后续迁移项。

## 1. 审计口径

审计同时检查四种内容：

1. **事实**：以当前代码、配置、测试和跨子系统契约为准，确认 frame、CALL、权限、PendingAtom 状态、alias 解析和物化交接是否仍然成立；
2. **理念**：保留能够解释人格、权限、控制面、瞬态子帧和 store buffer 取舍的背景，但把它们写入当前文档，而不是让历史计划继续充当规范；
3. **边界**：区分 Alice 控制面、Agent Runtime 执行层与 Patchouli 记忆域，避免旧 Kernel、Cache 或 Engine 布局重新成为子系统所有权；
4. **未来承诺**：尚未实现的持久化账本、TTL、递归调用、强安全沙箱或自组织网络只能保留为历史设想或当前限制，不能升级成现有能力。

本节的当前真相源是 [Alice 总览](../../alice/README.md)、[Agent Runtime](../../alice/agent-runtime.md)、[多 Agent 编排](../../alice/orchestration.md)、[PendingAtom](../../alice/pending-atom.md)与 [MTP Runtime](../../alice/mtp-runtime.md)。

## 2. 逐篇审计结果

### 2.1 `docs/alice/README.md`

- **分类与动作**：`current`，保留并核验；本轮补审时间为 2026-07-29。
- **已承接内容**：入口已经明确 Alice 的“人偶使”定位、控制面所有权、Agent Runtime 执行层、Profile 图纸、主 run、CALL、PendingAtom、启停与观测边界，并链接四篇当前模块文档。
- **保留的设计理念**：人格、模型偏好与结构化权限必须分离；Agent Profile 是 Patchouli 管理的记忆资产，Alice 只把它解析成一次运行的图纸；主 Agent 对用户任务与最终回复负责。
- **需要拒绝的旧口径**：Alice 不再只是未来预留概念，也不通过修改话题上的 `current_agent_id` 原地变身；Patchouli 不拥有 Agent 调度、权限执行和 frame 生命周期。
- **结论**：正文已经是当前总览；物理迁移时仅移动 `phases/` 下三篇历史材料，不移动本文件。

### 2.2 `docs/alice/phases/README.md`

- **分类与动作**：`archive`，直接归档。
- **已承接内容**：Phase 1 的 Profile/Runtime 与 Phase 2 的 CALL/IPC 导航已经由 Alice 当前总览及四篇模块文档接管。
- **独立价值判断**：本文件只有阶段索引，没有未被承接的接口、理念或限制；保留它仅用于追溯最初的两阶段拆分。
- **替代入口**：[Alice 总览](../../alice/README.md)、[Agent Runtime](../../alice/agent-runtime.md)、[多 Agent 编排](../../alice/orchestration.md)。

### 2.3 `docs/alice/phases/Phase1.md`

- **分类与动作**：`merge`，合并后归档。
- **已承接内容**：Agent Profile 的 persona/model/permissions、Profile 服务发现、PromptAssembler、MTP 双层权限、公共话题多角色历史、`[From: agent]` 来源标识、Omni-Doll fallback，以及 Alice 控制面与共享 Agent Runtime 的物理分层均已进入当前文档。
- **保留的设计理念**：“万物皆记忆”让 Agent 图纸复用 MemoryAtom 的存储、检索和演化能力；人格与权限分开，避免自然语言 persona 被误当作安全授权；Prompt 隐藏未授权动作只减少误用，Runtime 拦截才执行硬边界。
- **明确不继承**：不再由 `Patchouli Kernel` 同时拥有 Agent 调度、Perception 和长期记忆；不通过 `TopicSegment.current_agent_id` 原地切换 Agent；不假定所有 Agent 默认写入 GLOBAL 记忆；Omni-Doll 的当前 fail-open 行为是已记录风险，不是理想安全模型。
- **替代入口**：[Alice 总览](../../alice/README.md)、[Agent Runtime](../../alice/agent-runtime.md)、[MTP Runtime](../../alice/mtp-runtime.md)。

### 2.4 `docs/alice/phases/Phase2.md`

- **分类与动作**：`merge`，合并后归档。
- **已承接内容**：CALL trap、Profile 服务发现、`context_refs`、瞬态子帧、黑盒隔离、单层星型拓扑、自然语言 reply、PendingAtom alias 收割与 run 级 materialize task 分流均已进入当前编排与 PendingAtom 文档。
- **补充的设计理由**：CALL 不增加模型生成的 `RETURN` 动词；子 frame 自然完成就是控制流返回，自然语言 reply 与 PendingAtom aliases 共同构成 response。这样避免在已有 frame 终态之外再制造语法和 formatter 失败路径，但 Orchestrator 仍必须区分完成、取消、预算耗尽和意外挂起。
- **明确不继承**：`context_refs` 不是“零开销、无损共享内存”；子 Agent WRITE 不会立即生成正式 MemoryAtom；Orchestrator 不从已落库 alias 收割本轮副作用；子 Agent 不能继续 CALL 孙 Agent；“无限横向扩展”与自组织涌现网络不是当前能力，Profile/RUN/瞬态 frame 也不构成强安全沙箱。
- **替代入口**：[多 Agent 编排](../../alice/orchestration.md)、[PendingAtom](../../alice/pending-atom.md)、[MTP 契约](../../contracts/mtp.md)。

### 2.5 `docs/agent_runtime/README.md`

- **分类与动作**：`merge`，退役归档。
- **已承接内容**：单 Agent loop、Koakuma ISA、PendingAtom 与执行层边界已经分别进入 Alice 的 Agent Runtime、MTP Runtime 与 PendingAtom 当前文档。
- **明确不继承**：`agent_runtime/` 是共享实现目录，不是与 Alice、Patchouli、Gateway 并列的业务子系统，也不单独拥有跨子系统契约或控制面。
- **替代入口**：[Agent Runtime](../../alice/agent-runtime.md)、[MTP Runtime](../../alice/mtp-runtime.md)、[PendingAtom](../../alice/pending-atom.md)。

### 2.6 `docs/agent_runtime/pending_atom/README.md`

- **分类与动作**：`merge`，退役归档。
- **已承接内容**：四篇分阶段设计稿中的数据模型、状态机、物化交接、alias 解析和回收语义已经由一篇当前 PendingAtom 文档统一承接。
- **独立价值判断**：本文件只保留设计演进导航，不再作为 PendingAtom 当前入口。
- **替代入口**：[Alice PendingAtom](../../alice/pending-atom.md)。

### 2.7 `PendingAtomCacheDesign.md`

- **分类与动作**：`merge`，拆分承接后归档。
- **已承接内容**：runtime shadow memory、WRITE/UPDATE 临时句柄、intent 关联、三级 alias 解析、canonical redirect、settlement 与回收窗口已经进入 PendingAtom、MTP Runtime 和 Patchouli Generation 当前文档。
- **保留的设计理念**：PendingAtom 是 write-back/store buffer，而不是未分配 UUID 的 MemoryAtom；pending alias 像运行时 handle，允许 Agent 在物化前继续寻址；intent_id 负责异步相关性；ACK、正式持久化和 completion 必须分开表达。
- **明确不继承**：当前没有 durable ledger/WAL、墙钟 TTL、事件重放、跨进程恢复或完整 completion queue；三级缓存不等于权限沙箱；所有 Agent 的写入不自动升级为 GLOBAL；Patchouli 不反向读取 Alice 的可变 PendingAtom。
- **替代入口**：[Alice PendingAtom](../../alice/pending-atom.md)、[MTP Runtime](../../alice/mtp-runtime.md)、[记忆生成](../../patchouli/generation.md)。

### 2.8 `PendingAtomMaterializeTaskDesign.md`

- **分类与动作**：`merge`，合并后归档。
- **已承接内容**：Focus 作为纯参数 DTO、`PendingAtomMaterializeTask` 作为不可变投影、Task/Settlement 请求应答对偶、Alice 按 run 收尾组装任务，以及 `AgentRunResult.materialize_tasks` 边界已经进入 Alice 与 Patchouli 当前文档。
- **保留的设计理念**：执行层的可变状态不应穿透到记忆域；跨边界只传本次物化所需的最小冻结请求，Patchouli 返回显式 settlement。子帧 IPC alias 收割服务主 Agent 当前认知，run 级 task 投影服务跨子系统交接，两者正交。
- **明确不继承**：`ChatResult.write_focus/update_focus/pending_aliases` 不再是主结果边界；loop executor 不维护另一套 Focus 累积器；Patchouli finalize 不向 Alice 查询可变对象；旧文件中的代码行号和目录布局只用于追溯当时的重构背景。
- **替代入口**：[Alice PendingAtom](../../alice/pending-atom.md)、[Agent Runtime](../../alice/agent-runtime.md)、[记忆生成](../../patchouli/generation.md)、[子系统公共契约](../../contracts/subsystem-contracts.md)。

### 2.9 `PendingAtomRuntimeDesign.md`

- **分类与动作**：`merge`，合并后归档。
- **已承接内容**：统一 Runtime 外观、内部 store、register/claim/settle/fail/cancel/collect 命令、模型上移到 `core.models.pending`、snapshot、反查索引和回收行为已经进入当前 PendingAtom 文档。
- **保留的设计理念**：行为入口应集中经过状态机闸门；store 只负责保存对象和反查索引；PendingAtom 自身的 `status/settlement` 是业务真相，snapshot 与 resolver 是派生视图。
- **补充的设计理由**：alias、intent 与 canonical 映射只回答“怎样找到对象”，不能平行保存 resolution 或 redirect 状态。否则 settlement、READ、snapshot 与 GC 在更新顺序不一致时会分别读到不同生命周期。
- **明确不继承**：不再维护独立 `_resolution` / `_redirects` 作为业务真相；文中计划 API 不能覆盖当前代码的合法迁移和回收语义；进程内 facade 不代表已有 durable runtime、幂等事件账本或跨进程恢复。
- **替代入口**：[Alice PendingAtom](../../alice/pending-atom.md)。

### 2.10 `PendingAtomStatusUnificationDesign.md`

- **分类与动作**：`merge`，合并后归档。
- **已承接内容**：生命周期 `status`、结算 `resolution` 与对象 `kind/source_verb` 的正交语义，`MATERIALIZING`、Task/Settlement 强类型边界、canonical 不变量和 resolver 派生原则已经进入当前 PendingAtom 文档与核心模型。
- **保留的设计理念**：生命周期阶段和结算分类回答不同问题，不应继续用 `COMMITTED/MERGED/TOUCHED` 一组字符串同时承担；跨模块先传强类型，只有面向 prompt、事件或持久化边界时再序列化。
- **明确不继承**：旧稿提议的平行 `_status/_resolution` 持久态不是当前设计；当前 settlement 保存在 PendingAtom 上，索引只做寻址；旧稿中把多个状态直接视为不可再迁移 terminal 的图示也不能替代当前允许 `SETTLED/FAILED/CANCELLED -> EXPIRED` 的状态机。
- **替代入口**：[Alice PendingAtom](../../alice/pending-atom.md)、[核心数据模型](../../architecture/data-model.md)。

## 3. 物理迁移结果

审计通过后，九篇旧文档按下列结构移动，保留文件名、阶段关系和核心原文以便追溯；`docs/alice/README.md` 继续留在当前入口：

```text
docs/archive/legacy-docs/alice/phases/
  README.md
  Phase1.md
  Phase2.md
docs/archive/legacy-docs/agent_runtime/
  README.md
  pending_atom/
    README.md
    PendingAtomCacheDesign.md
    PendingAtomMaterializeTaskDesign.md
    PendingAtomRuntimeDesign.md
    PendingAtomStatusUnificationDesign.md
```

归档后，Phase 文档的替代入口统一指向当前 `docs/alice/` 与 Contracts；PendingAtom 历史稿统一指向当前 Alice、Patchouli、Contracts、`docs/mod/` 和源码位置。其他已归档历史稿或尚待审计的 `docs/mod/` 文件如果引用这些设计过程，也改指本批 Archive 路径，避免用失效旧路径维持演进关系。

## 4. 验证门禁

- [x] 10 篇清单文档均给出承接位置、保留理念和拒绝继承项；
- [x] 五篇 Alice 当前文档已复核并更新日期，编排与 PendingAtom 补齐缺失的设计理由；
- [x] 九篇旧文档的物理目标均位于 `docs/archive/legacy-docs/` 内，原路径不再存在；
- [x] 已完成 Markdown 相对链接、严格 UTF-8、旧路径残留和 `git diff --check` 复核；
- [x] Alice/Agent Runtime 定向测试共 413 项通过，35 项按项目标记排除。
