---
title: 任务进程表与任务请求唯一注册入口
status: idea
horizon: candidate
owner: project
scope: task-process-model-open-directions
code_paths:
  - src/hivememory/workspace/process/
related_docs:
  - docs/ideas/fifth-generation-architecture.md
  - docs/system/application-services.md
  - docs/ideas/external-session-and-topic-projection.md
  - docs/ideas/external-actor-registration-and-runtime-access.md
  - docs/ideas/execution-unit-thread-and-environment.md
  - docs/ideas/import-bus.md
last_reviewed: 2026-10-13
---

# 任务进程表与任务请求唯一注册入口

## 0. 文档性质

本文收集任务进程模型尚未决定或尚未落地的方向，不描述当前行为，也不代表排期。理念见[第五代架构 Idea](./fifth-generation-architecture.md)第 3.2 节“一切任务都是进程”；当前设计见 [System 应用服务](../system/application-services.md)第 3、4 节、[子系统公共契约](../contracts/subsystem-contracts.md)第 4 节与 [Workspace 架构](../architecture/workspace.md)第 4.3、5.4 节。问题沿用原编号（Q-x）。

本文于 2026-10-13 精简，删除了已实施的决定、实施记录与迁移前的代码快照；删除前版本见 commit `a384688`。

## 1. 请求方的分类（owner，2026-09-27）

请求方只有两类：

| 类别 | 定义 | 例子 |
|:---|:---|:---|
| 主动请求 | 用户发出的指令主动且即时地驱动接下来的任务进程 | 经 Alice 的 chat 链路；以 controller 模式接入的外部 harness |
| 被动请求 | 由条件触发；用户设定之后不再主动发出请求 | 定时任务、队列任务 |

- **区分标准是“即时”**：进程在用户指令到达的那一刻创建，即为主动请求，否则为被动请求。队列任务虽由用户提交，但进程要等条件满足时才创建，属于被动请求。
- “被动（Passive）”一词留给被动请求体系；现有的 Passive Ingress 链路不是请求方，现称 Import Bus（[Import Bus Idea](./import-bus.md)）。
- 请求方与执行者是两个维度：controller 模式下，请求方是在 HiveMemory 入口发出指令的用户，外部 harness 是执行单元。

以下对象不是请求方：

| 对象 | 定位 |
|:---|:---|
| CALL 派生的子执行线程 | 在父 Agent 的任务进程内执行，不能请求新进程 |
| root 直接通道 | 直接执行管理操作，不经注册入口、不建进程 |
| Patchouli 的记忆任务 | 后台系统的任务，不暴露给 Agent |
| plugin 模式下的外部 harness | 对话由外部 harness 管理，不经注册入口，以不建进程的方式经能力层访问 |

## 2. 进程模型的后续方向

### 2.1 交互被接纳后即关闭进程（Q-1）

owner（2026-09-28）决定：收尾阶段等到交互被提交队列接纳，进程即可关闭，不再等待 applied。附件租借因此在接纳时随进程关闭释放；Artifact promotion 在生成时按 `binding.asset_ref` 重新取得内容（[Chat 附件链路](../system/attachments.md)第 4 节），不依赖本轮的租借（分析）。

owner（2026-10-13）决定这一项暂时不实施，与话题是否预建一起随外部会话与流式同步的设计决定（[外部会话 Idea](./external-session-and-topic-projection.md)；容器形态见[长时间运行 Agent Idea](./long-running-agent-intra-turn-context-folding.md)开头的方向）。

**取舍**：曾考虑两阶段关闭（运行中 → 结算中 → 关闭），它需要在进程表中保留不占用执行单元的进程，并定义保留窗口、容量与停机处理；写入意图由 workspace 登记、与进程解耦之后，这些都不再需要。

### 2.2 Profile 解析回到 CPU 分配

Profile 属于 CPU 分配，但现在提前到 prepare 之前解析，因为 prepare 可能新建 Topic、按 LRU 结算已有话题，这些副作用不可逆。Topic 的新建与驱逐改到交互提交之后，Profile 才能回到 prepare 之后的 CPU 分配步骤；它与 2.1 一样取决于话题是否预建。

### 2.3 进程记录与工作集的剩余槽位

owner（2026-09-28）为进程记录与工作集定下的内容中，以下部分尚未成为槽位：

- 进程记录：请求方式（主动或被动）、CPU 分配（所用的执行环境，见[执行单元 Idea](./execution-unit-thread-and-environment.md) T-8）；
- 工作集中的值：GatewayDecision、执行记录。

约束：槽位可以动态，值必须有类型，每个槽位都有声明的类型与所有者，不使用 `dict[str, Any]` 这类无类型载体；工作集中的资源在进程关闭时必须释放或补偿，无论从哪个阶段结束。

### 2.4 编译预算由执行单元声明

记忆与附件由进程编译，编译的预算与目标将来由执行单元在 CPU 分配时声明；只有 Alice 一个执行单元时沿用现有配置。

### 2.5 四阶段骨架的失效条件（分析）

controller 模式下所有任务进程共用 Gateway 分析 → 预检索 → Actor 执行 → 结算四个阶段，骨架内部由执行者在线决定（Q-4）。出现装不进四阶段的任务时需要重新讨论任务类型，例如 Deep Research 需要多轮检索与执行、长期保留研究状态，或被动请求保存的指令无法作为用户消息交给 Gateway 分析。显式执行计划（任务图）需要计划表示、规划器与计划失败语义，接近 [AE2 Idea](./ae2-hivememory-architecture-analogy.md) 中的 Job Graph，ROADMAP 中通用 workflow / DAG 为 Unscheduled。

### 2.6 被动请求（Q-6）

owner（2026-10-01）决定这一阶段先不考虑被动请求，避免现在做好的接口被后续设计推翻；入口的形状以后可能需要调整。被动请求只存在于 controller 模式，触发器的认证见[第五代架构 Idea](./fifth-generation-architecture.md) P-5。

## 3. 开放问题

每个问题只列出选项及其影响，选项顺序不代表倾向。

### Q-3b 唯一入口与传输入口

| 选项 | 内容 | 影响 |
|:---|:---|:---|
| A | 唯一入口只指唯一的注册点 | HTTP、外部协议、触发器可以各自有对外端点，最终都调用同一个注册入口 |
| B | 同时要求唯一的传输入口 | 所有任务请求经同一个对外端点，外部协议与触发器需要适配到该端点 |

与外部 Actor 的 controller 模式接入相关。

### Q-5a 命令在进程中何时、由谁运行

命令所在的请求注册为进程，Gateway 只解析命令；命令系统后置，内置命令暂时不可用。命令的运行位置随命令系统一并决定，候选尚未列出。[会话 `/compact` 指令 Todo](../todo/conversation-compact-command.md) 依赖这一决定。

### Q-5b `PASSIVE_MEMORY` 模式的去留

Gateway 的 `PASSIVE_MEMORY` 模式只服务于 Import Bus，它的去留取决于 [Import Bus Idea](./import-bus.md) 的 Q-11 与 Q-12。

### Q-7 进程表是否收录后台任务

记忆任务不暴露给 Agent。剩下的问题是进程表是否收录这类后台任务（收录时同样对 Agent 不可见）。

| 选项 | 内容 | 影响 |
|:---|:---|:---|
| A | 不进入进程表，保留在 Patchouli 内部队列与维护调度 | 进程表只包含由执行单元执行的任务 |
| B | 全部进入统一进程表 | 进程表成为全局任务表，承担调度职责 |
| C | 部分进入（例如由模型驱动的生成任务进入，纯维护不进入） | 需要明确划分标准 |

### Q-8 外部执行单元的进程

Alice 在进程内运行，进程能取消它，也知道它何时结束。外部 harness 不受网络控制，更接近 AE2 中“处理样板 → 外部机器”：网络送出材料，等待产物回流。本问题只涉及 controller 模式，plugin 模式不建进程（[外部 Actor Idea](./external-actor-registration-and-runtime-access.md#11-两种接入模式owner2026-09-27) 1.1）。

- **Q-8a 回收方式**：请求方显式关闭 / TTL / 心跳 / 组合；
- **Q-8b 取消语义**：网络对外部执行单元的取消能做到什么程度（例如吊销访问、通知外部方），各项分别是否纳入；
- **Q-8c 粒度**：外部 harness 的一轮对应一个进程 / 一个外部会话对应一个进程 / 由接入适配层决定。

### plugin 模式与进程模型的交界

随 plugin 模式设计一并决定：

- plugin 模式不建进程，它提交的 `InteractionPayload` 中 `interaction_id` 如何取值（任务进程以 `process_id` 为值）；
- 同一 harness 同时以 plugin 模式上报对话、又以 controller 模式运行任务进程时，交互记录如何避免重复。

## 4. 相关文档

| 文档 | 关系 |
|:---|:---|
| [第五代架构 Idea](./fifth-generation-architecture.md) | 3.2 的理念与检验问题；架构级开放问题 P-4a、P-5、P-7、P-9d |
| [执行单元 Idea](./execution-unit-thread-and-environment.md) | 执行环境的选择（T-8）决定 CPU 分配记入进程记录的内容 |
| [外部会话与 Topic 投影 Idea](./external-session-and-topic-projection.md) | 会话承载对话连续性；2.1、2.2 随它与流式同步的设计决定 |
| [外部 Actor Idea](./external-actor-registration-and-runtime-access.md) | controller 模式使用本文的进程模型（Q-3b、Q-8），plugin 模式不建进程 |
| [Import Bus Idea](./import-bus.md) | Q-5b 的前提 |
| [Chat Run 生命周期后续候选](./chat-run-lifecycle-follow-ups.md) | 进程生命周期的其他候选改进 |
