---
title: 身份与访问体系
status: idea
horizon: candidate
owner: project
scope: identity-and-access-open-directions-and-cross-workspace-access
code_paths:
  - src/hivememory/core/models/identity.py
  - src/hivememory/core/access.py
  - src/hivememory/core/memory_access.py
  - src/hivememory/workspace/authentication.py
  - src/hivememory/workspace/authorization.py
related_docs:
  - docs/ideas/fifth-generation-architecture.md
  - docs/architecture/workspace.md
  - docs/ideas/execution-unit-thread-and-environment.md
  - docs/ideas/ae2-hivememory-architecture-analogy.md
  - docs/ideas/pending-intent-migration.md
  - docs/todo/workspace-asset-ownership-identity-split.md
last_reviewed: 2026-10-13
---

# 身份与访问体系

## 0. 文档性质

本文收集身份与访问体系尚未决定的方向，不描述当前行为，也不代表排期。理念见[第五代架构 Idea](./fifth-generation-architecture.md)第 3.5、3.6 节；当前的身份模型（三个身份模型、两阶段认证与两阶段授权、身份数据的形态与流向、访问 context 的密封与生命周期、必须持续成立的不变量）见 [Workspace 架构](../architecture/workspace.md)第 3、4 节，资源归属与发起者的交接见同文第 8 节。问题沿用原编号（I-x）。

本文于 2026-10-13 精简，删除了已实施的决定（I-1–I-11）、分批记录与现状快照；删除前版本见 commit `a384688`，两批的实施历史见 [A1 访问边界返工](../archive/plans/v0.7.0-a1-access-boundary-rework.md)与[身份与访问体系第二批](../archive/plans/v0.7.0-identity-access-batch-2.md)（均为归档计划）。

## 1. 受限穿透访问

owner（2026-10-03）：`IdentityScope` 由操作的发起者与目标 Workspace 组成，正是为了以后 actor 的一次操作可以跨过它所在的 Workspace，对其他 Workspace 做有限访问。actor 进入 Workspace 与 actor 真正进行操作，两者的 Workspace 可以不同：coder 在 WA 中执行任务，创建任务进程时确认 coder 能进入 WA；执行过程中 coder 想去 WB 查找资源，这次操作的发起者是 coder，目标是 WB，与 coder 驻留在 WA 无关。

现在第 3 阶段只接受目标等于驻留 Workspace；授权点接口已经显式接收目标 Workspace，实现受限穿透访问时只需放宽这条规则，不必改动各授权点的签名。

需要回答：

- **授予模型**：谁能对哪个非驻留 Workspace 做什么。分析（2026-10-10）：一次跨 Workspace 的操作要同时满足 agent 自身的白名单与目标一侧的授予；
- **agent 的身份**：Agent Profile 是某个 Workspace 中的记忆原子，`agent_id` 即其 alias，只在该 Workspace 内唯一，不同 Workspace 中的同名 agent 是两条独立的原子。“同一 agent 在不同 Workspace 有不同能力”只会出现在穿透访问中，那时 actor 身份需要带上驻留 Workspace（分析，2026-10-10）；
- **目标的表达**：执行一侧如何给出目标，见[执行单元 Idea](./execution-unit-thread-and-environment.md) T-5b；
- **资源边界**：资源 owner 仍按目标 Workspace 校验归属；同一进程内会同时出现多个目标 Workspace 的资源，派生缓存按派生源坐标键控，不能从 scope 的存在推导隔离；
- **与子网的关系**：Mount、Bridge 与能力子网是更完整的形态（[AE2 Idea](./ae2-hivememory-architecture-analogy.md)），穿透访问的授予模型应能向它扩展。

## 2. 资源授权的扩展

现在只有 MemoryAtom 带资源 policy（PUBLIC、PRIVATE、TEAM，只表达读取）；Topic、Artifact、WorkspaceAsset 与记忆任务只按归属授权。是否把资源策略扩展到修改与执行，见第五代架构 Idea P-8。

约束（2026-10-04，由“资源上的来源字段不参与授权”得出）：与创建者相关的权限，例如“只有创建者能修改”，应在资源创建时写入资源 policy，授权时不读取来源；否则来源会重新成为访问条件。

写入意图的可见性与 policy 见[写入意图后续方向 Idea](./pending-intent-migration.md)第 5 节。

## 3. 访问 context 可能增加的内容

访问 context 的授予内容只包含 actor、驻留 Workspace、principal 与运行绑定。以下方向可能改变这一点，各自随相应问题决定：

- **adapter**：context 暂存签发时的 principal（I-2），adapter 是否一并保存未决，随第五代架构 Idea P-1b（请求与进程的归属校验）一并决定；
- **本次运行的操作上限**：只有 P-4a 采用“进程持有白名单的子集”时，context 才需要携带它；
- **有效期**：context 没有固定有效期，随绑定的运行结束失效；plugin 模式需要长连接时，再随其设计决定是否需要有效期。

## 4. 已登记的暂缓与例外

- WorkspaceAsset 的 Store、解析服务与旧端口仍接收 `IdentityScope`，归属拆分暂缓：[WorkspaceAsset 归属拆分 Todo](../todo/workspace-asset-ownership-identity-split.md)；
- `/ingest` 在认证前组装 `IdentityScope`，是访问模型的已知例外：[Import Bus Idea](./import-bus.md) D-8a；
- chat 请求体中 `session_id` 的未来语义：[外部会话与 Topic 投影 Idea](./external-session-and-topic-projection.md)。

## 5. 相关文档

| 文档 | 关系 |
|:---|:---|
| [第五代架构 Idea](./fifth-generation-architecture.md) | 3.5、3.6 的理念；架构级开放问题 P-1、P-4a、P-7、P-8、P-9 |
| [Workspace 架构](../architecture/workspace.md) | 身份模型与访问边界的当前事实 |
| [执行单元 Idea](./execution-unit-thread-and-environment.md) | 跨 Workspace 目标在执行一侧的表达（T-5b） |
| [AE2 Idea](./ae2-hivememory-architecture-analogy.md) | 子网与跨 Workspace 的更完整形态 |
| [写入意图后续方向 Idea](./pending-intent-migration.md) | 意图可见性与 policy |
