---
title: 执行单元、执行线程与执行环境
status: idea
horizon: candidate
owner: project
scope: execution-environment-selection-and-cross-workspace-target
code_paths:
  - src/hivememory/workspace/contracts/
  - src/hivememory/workspace/process/
  - src/hivememory/alice/application/cpu.py
  - src/hivememory/server/models/chat.py
related_docs:
  - docs/ideas/fifth-generation-architecture.md
  - docs/ideas/identity-and-access-model.md
  - docs/ideas/task-process-table-and-registration-entry.md
  - docs/ideas/external-actor-registration-and-runtime-access.md
  - docs/ideas/external-session-and-topic-projection.md
last_reviewed: 2026-10-13
---

# 执行单元、执行线程与执行环境

## 0. 文档性质

本文收集执行一侧尚未决定的方向：agent 与执行环境的分别选择，以及执行线程表达跨 Workspace 目标的方式。不描述当前行为，也不代表排期。理念见[第五代架构 Idea](./fifth-generation-architecture.md)第 3.3 节“执行者可以替换”，执行一侧的用语（CPU、CPU 驱动、harness 实例、执行线程、操作适配器、请求入口）见同文第 2 节；当前设计见[子系统公共契约](../contracts/subsystem-contracts.md)第 4 节与 [Workspace 架构](../architecture/workspace.md)第 4.3、5.4 节。前提与问题沿用原编号。

本文于 2026-10-13 精简，删除了已实施的决定（执行线程的身份与访问 context、操作请求与单一操作入口、子线程的派生与撤销等）、实施记录与代码快照；删除前版本见 commit `a384688`。

## 1. 前提（owner 提出）

**前提 2**（2026-10-05）：一次操作的发起者与目标分开看待。

- **发起者**是客观事实，由执行单元一侧补全，agent 不在工具调用中声明；
- **目标 Workspace** 是 agent 的意图：现在由补全的一层给出（只接受驻留 Workspace）；实现跨 Workspace 的受限穿透访问后，agent 需要能够自己带上操作目标（T-5b）。

**前提 9**（2026-10-06）：**agent 与执行环境解耦**。外部 harness 最初被视为一个单独的 agent，现在它是一个执行环境（执行单元的 backend）。理论上可以在外部 harness 之上再应用 Agent Profile（例如作为系统提示词），外部协议是否支持另当别论。用户在前端或任何交互环境中分别选择要使用的 agent 与执行环境。

- 分析：agent 的身份、访问登记、记忆可见性与权限都只跟随 agent，由 HiveMemory 强制执行，与执行环境无关；更换执行环境不改变 agent 能看到什么、能做什么。
- 分析：Profile 中的执行参数（persona、`model_name`、`temperature`、`top_p`、`language`）能否生效取决于执行环境，对外部 harness 只能尽力应用；应用结果的承载见第五代架构 Idea P-10a。按 ACP 仓库文档（2026-10-06 核对），`session/new` 的参数为 `cwd`、`mcpServers` 等，未见系统提示词字段；模型只能在 harness 自己声明的配置项（`configOptions` 中 `category` 为 `model` 的选项）中经 `session/set_config_option` 选择。persona 只能作为首轮 prompt 内容或各 harness 私有的扩展注入，约束力弱于系统提示词。
- 分析：Alice 成为可选的执行环境之一；同一个 Agent Profile 可以在不同执行环境中运行，便于对照评估。交互记录或记忆来源可以记录所用的执行环境，作为观测与评估信息；来源不参与授权。
- 这一条决定了[外部 Actor Idea](./external-actor-registration-and-runtime-access.md#e-6-actor-与执行单元的对应) E-6a。

**前提 10**（2026-10-06）：**执行环境的选择**。只有用户能选择执行环境，并且只能在对话开始前选择。

- 选择执行环境不是 workspace 的能力：不设对应的 actor 操作，能力层也不提供这一方法，actor 无法在执行中选择或更换执行环境；
- owner 倾向：它大概率成为系统层级的控制，在 CPU 分配时决定实际使用的 backend（T-8）；
- 分析：执行环境在对话内固定，CALL 派生的子线程无法为自己选择另一个执行环境；外部 Actor Idea E-6b 据此决定不允许 CALL 的目标位于另一个执行单元（owner，2026-10-13），一个任务进程只对应一个执行单元。

## 2. 开放问题

每个问题只列出选项及其影响，选项顺序不代表倾向。T-7、T-8 不在 v0.7.0（owner，2026-10-06）：只有 Alice 一个执行环境时，选择机制没有可验收的对象；前提 9、10 已确定执行环境与 agent 的关系，之后接入外部执行环境不需要再改动拓扑。

### T-5b 跨 Workspace 目标的表达形式

受限穿透访问不在 v0.7.0；授权点已显式接收目标 Workspace（[身份与访问体系 Idea](./identity-and-access-model.md)第 1 节）。

- 选项：按指令给出目标，一条指令一个目标 / 按引用限定，alias 带上 Workspace 限定 / 其他。
- 影响：alias 按 (Workspace, alias) 分区，跨 Workspace 的引用必须带限定；agent 只能给出 Workspace key 一类名称，`WorkspaceIdentity` 中的 owner 由补全的一层解析，解析失败与无权访问需要对 agent 表现一致。
- 影响（分析）：MTP 文本不能指定 Workspace 的现有约束，随穿透访问设计另行评审；发起者不能由工具调用指定（前提 2）长期成立。

### T-7 agent 与执行环境组合的有效性

按前提 9，用户分别选择 agent 与执行环境，并非每个组合都有效。例如 Profile 指定的模型不在 harness 声明的模型配置项中；harness 无法注入 persona；执行环境要求的工作目录等条件不满足。

- **T-7a 校验时点**：注册入口创建进程之前 / CPU 分配时，在 Patchouli prepare 之前（与 Profile 解析当前的时点相同） / 执行开始后由 CPU 驱动报告 / 其他。影响：Profile 提前到 prepare 之前解析，是因为 prepare 可能新建 Topic 或按 LRU 结算已有话题，这些副作用不可逆（[任务进程 Idea](./task-process-table-and-registration-entry.md) 2.2）；组合在 prepare 之后才被判定无效，会留下同样的副作用。
- **T-7b 无法完全应用时的处理**：拒绝请求 / 降级运行并报告未生效的项（第五代架构 Idea P-10a） / 按 Profile 字段区分：模型不可用时拒绝，persona 无法注入时降级 / 其他。

### T-8 执行环境选择的承载与系统层级的控制

前提 10 已决定只有用户能选择执行环境、只能在对话开始前选择，且不是 workspace 的能力；owner 倾向由系统层级控制。现在 chat 请求只携带 `agent_id`，组合根只注入 Alice 一个 CPU 驱动。

- **T-8a 选择的承载位置**：作为会话的属性在对话开始时确定，此后各任务进程从会话取得 / 每次请求都携带，由入口校验它与对话开始时的选择一致 / 其他。影响：会话模型见[外部会话与 Topic 投影 Idea](./external-session-and-topic-projection.md#01-会话模型与-topic-池owner2026-09-28) 0.1；无论哪一项，选择都要作为任务参数到达 CPU 分配，并记入进程记录（[任务进程 Idea](./task-process-table-and-registration-entry.md) 2.3）。
- **T-8b 默认值**：系统默认的执行环境（当前为 Alice） / Agent Profile 中声明的偏好 / 用户设置 / 其他。影响：由 Profile 声明偏好会让 agent 与执行环境重新部分耦合（前提 9）。
- **T-8c 系统层级控制的形态**：哪些执行环境可用、对哪些用户可用：由 System 在启动时随 harness 登记的执行侧面装载（[外部 Actor Idea](./external-actor-registration-and-runtime-access.md) 1.2、E-1），CPU 分配按用户的选择取得驱动 / 按用户登记可用的执行环境 / 其他。影响：向前端列出可选执行环境的接口属于 System 层面的 API，不经 workspace（[Workspace 架构](../architecture/workspace.md)第 4.5 节）。

## 3. 相关文档

| 文档 | 关系 |
|:---|:---|
| [第五代架构 Idea](./fifth-generation-architecture.md) | 3.3 的理念、第 2 节的执行一侧用语；P-10a 与 T-7b 相互约束 |
| [身份与访问体系 Idea](./identity-and-access-model.md) | 受限穿透访问的授权模型，T-5b 是它在执行一侧的表达 |
| [任务进程 Idea](./task-process-table-and-registration-entry.md) | Profile 的解析时点（2.2）与 CPU 分配记入进程记录（2.3） |
| [外部 Actor Idea](./external-actor-registration-and-runtime-access.md) | harness 登记的两个侧面（1.2）；外部执行单元不识别子线程（E-5）；外部 harness 是执行环境（E-6a） |
| [外部会话与 Topic 投影 Idea](./external-session-and-topic-projection.md) | 会话模型，T-8a 的承载位置之一 |
