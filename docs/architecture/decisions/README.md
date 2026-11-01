---
title: Architecture Decision Records
status: current
owner: project
scope: architecture-decisions
last_reviewed: 2026-10-31
---

# Architecture Decision Records

本目录保存具有长期解释价值的架构决策。ADR 使用 `NNNN-kebab-case-title.md` 命名，并包含 Context、Decision、Consequences、Alternatives、Status 和相关文档。

## 编写规则

适合写成 ADR 的是已经做出、在较长时间内仍有解释价值的选择：子系统边界为何形成，为什么选择某种数据所有权或通信模式，为什么拒绝一个重要备选方案，哪些约束导致当前设计，以及什么条件变化后可以重新评估。ADR 保持简短。

- **只叙述已成立的内容**：已经做出的决定、理由与已成立的事实。ADR 不得反向依赖计划：不以活动计划为依据，也不把未完成事项、替代条件或后续收口交由计划承担；需要追溯实施过程时，只链接已归档计划。
- **状态**：已接受且仍有效为 `accepted`。被替代的 ADR 不删除，状态改为 `superseded` 并链接新的 ADR；决策不再成立且没有替代时改为 `deprecated`，注明失效日期与原因，同样不删除，当前设计文档不再以它为依据。
- **与当前文档的关系**：当前设计文档可以链接 ADR 解释“为什么”，但仍应独立说明“现在是什么”。

## 决策索引

当前决策：

- [ADR-0001：按语义选择可变性，跨边界使用只读投影](./0001-data-model-mutability-and-boundary-projection.md)
- [ADR-0002：全局唯一身份与按需并发保护](./0002-unique-identities-and-minimal-concurrency.md)
- [ADR-0003：Memory-as-a-Tool 与 MTP RUN 的边界语义](./0003-memory-as-a-tool-and-mtp-run-boundary.md)
- [ADR-0006：记忆库的归属判据与独立工作契约](./0006-memory-library-custody-criteria-and-independence-contract.md)

已失效决策（`deprecated`，没有替代 ADR，仅供追溯，不作为设计依据）：

- [ADR-0004：执行路径派生缓存的所有权与键控](./0004-execution-path-derived-caches.md)（2026-09-27 失效）
- [ADR-0005：统一 Actor 认证网关与 Workspace 持有的授权](./0005-unified-actor-authentication-and-workspace-authorization.md)（2026-09-27 失效）

后续从历史设计中提炼 ADR 时，优先评估以下主题：

- Gateway 独立为顶层子系统；
- System / Service / Runtime 分层；
- Agent Runtime 与 MTP Runtime 解耦；
- MemoryCompiler 作为统一记忆表达边界；
- 纯异步总线与全局维护调度；
