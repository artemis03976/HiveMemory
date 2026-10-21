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
  - docs/workspace/README.md
  - docs/archive/ideas/execution-thread-definition.md
  - docs/ideas/execution-environment.md
  - docs/ideas/ae2-hivememory-architecture-analogy.md
  - docs/ideas/pending-intent-migration.md
  - docs/todo/workspace-asset-ownership-identity-split.md
last_reviewed: 2026-10-20
---

# 身份与访问体系

## 0. 文档性质

本文收集身份与访问体系尚未决定的方向，不描述当前行为，也不代表排期。理念见[第五代架构 Idea](./fifth-generation-architecture.md)第 3.5、3.6 节；当前的身份模型（三个身份模型、两阶段认证与两阶段授权、身份数据的形态与流向、访问 context 的密封与生命周期、必须持续成立的不变量）见 [身份与授权](../workspace/identity-and-authorization.md)与[认证与访问登记](../workspace/authentication.md)，资源归属与发起者的交接见同文第 8 节。问题沿用原编号（I-x）。

本文于 2026-10-13 精简，删除了已实施的决定（I-1–I-11）、分批记录与现状快照；删除前版本见 commit `a384688`，两批的实施历史见 [A1 访问边界返工](../archive/plans/v0.7.0-a1-access-boundary-rework.md)与[身份与访问体系第二批](../archive/plans/v0.7.0-identity-access-batch-2.md)（均为归档计划）。

## 1. 受限穿透访问

owner（2026-10-03）：`IdentityScope` 由操作的发起者与目标 Workspace 组成，正是为了以后 actor 的一次操作可以跨过它所在的 Workspace，对其他 Workspace 做有限访问。actor 进入 Workspace 与 actor 真正进行操作，两者的 Workspace 可以不同：coder 在 WA 中执行任务，创建任务进程时确认 coder 能进入 WA；执行过程中 coder 想去 WB 查找资源，这次操作的发起者是 coder，目标是 WB，与 coder 驻留在 WA 无关。

现在第 3 阶段只接受目标等于驻留 Workspace；授权点接口已经显式接收目标 Workspace，实现受限穿透访问时只需放宽这条规则，不必改动各授权点的签名。

需要回答：

- **授予模型**：谁能对哪个非驻留 Workspace 做什么。分析（2026-10-10）：一次跨 Workspace 的操作要同时满足 agent 自身的白名单与目标一侧的授予；
- **agent 的身份**：Agent Profile 是某个 Workspace 中的记忆原子，`agent_id` 即其 alias，只在该 Workspace 内唯一，不同 Workspace 中的同名 agent 是两条独立的原子。“同一 agent 在不同 Workspace 有不同能力”只会出现在穿透访问中，那时 actor 身份需要带上驻留 Workspace（分析，2026-10-10）；
- **目标的表达**：执行一侧如何给出目标，见本节末尾的 T-5b；
- **资源边界**：资源 owner 仍按目标 Workspace 校验归属；同一进程内会同时出现多个目标 Workspace 的资源，派生缓存按派生源坐标键控，不能从 scope 的存在推导隔离；
- **与子网的关系**：Mount、Bridge 与能力子网是更完整的形态（[AE2 Idea](./ae2-hivememory-architecture-analogy.md)），穿透访问的授予模型应能向它扩展。

以下前提与问题于 2026-10-19 从原《执行单元、执行线程与执行环境》Idea（现[执行环境 Idea](./execution-environment.md)）移入，沿用原编号。

**前提 2**（owner，2026-10-05）：一次操作的发起者与目标分开看待。

- **发起者**是客观事实，由执行单元一侧补全，agent 不在工具调用中声明；当前由发起操作的执行线程给出，见 [执行线程](../workspace/execution-threads.md)。
- **目标 Workspace** 是 agent 的意图：现在由补全的一层给出，只接受驻留 Workspace；实现跨 Workspace 的受限穿透访问后，agent 需要能够自己带上操作目标（T-5b）。

### T-5b 跨 Workspace 目标的表达形式

受限穿透访问不在 v0.7.0；授权点已显式接收目标 Workspace（见本节开头）。

- **选项**：按指令给出目标，一条指令一个目标 / 按引用限定，alias 带上 Workspace 限定 / 其他。
- **影响**：
  - alias 按 (Workspace, alias) 分区，跨 Workspace 的引用必须带限定；
  - agent 只能给出 Workspace key 一类名称，`WorkspaceIdentity` 中的 owner 由补全的一层解析，解析失败与无权访问需要对 agent 表现一致；
  - （分析）MTP 文本不能指定 Workspace 的现有约束，随穿透访问设计另行评审；发起者不能由工具调用指定（前提 2）长期成立。

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
- `/ingest` 在认证前组装 `IdentityScope`，是访问模型的已知例外：[Import Bus Idea](./import-bus.md) D-8a（2026-10-20 决定经统一认证网关，v0.7.1 首份计划随 Import Bus 重设计消除）；
- chat 请求体中 `session_id` 的未来语义：[外部会话与 Topic 投影 Idea](./external-session-and-topic-projection.md)。

**root 的声明范围**（owner，2026-10-20）：能以保留的 `system` 进入 root API 的只有用户经 HTTP 入口的管理访问，其他接入来源不能声明 `system`。现状是认证网关只限制 principal 可代表的用户，任何已登记的来源都能声明 `system`（[身份与授权](../workspace/identity-and-authorization.md#8-已知限制)第 8 节）；Profile 注入脚本（`hivememory:profile-injector`）不在这条规则的范围内（owner，2026-10-20）：它用于向默认 Workspace 快速注入预设 Agent，理想情况下只在项目初次部署时使用，不应经过权限系统；现状仍作为登记来源经网关以 `system` 进入 root API，项目接入部署流程时再调整，不影响当前流程。候选设计（分析）：在接入登记中为每个 principal 显式声明能否声明 `system`，默认不能；登记第一个外部 principal 之前落实。

## 5. 相关文档

| 文档 | 关系 |
|:---|:---|
| [第五代架构 Idea](./fifth-generation-architecture.md) | 3.5、3.6 的理念；架构级开放问题 P-1、P-4a、P-7、P-8、P-9 |
| [身份与授权](../workspace/identity-and-authorization.md) | 身份模型与访问边界的当前事实 |
| [执行线程](../workspace/execution-threads.md)（已归档的[执行线程 Idea](../archive/ideas/execution-thread-definition.md)） | 操作的发起者由执行线程给出；线程的目标现为驻留 Workspace，跨 Workspace 目标的表达见第 1 节 T-5b |
| [执行环境 Idea](./execution-environment.md) | 前提 2 与 T-5b 的原所在（2026-10-19 移入本文） |
| [AE2 Idea](./ae2-hivememory-architecture-analogy.md) | 子网与跨 Workspace 的更完整形态 |
| [写入意图后续方向 Idea](./pending-intent-migration.md) | 意图可见性与 policy |
