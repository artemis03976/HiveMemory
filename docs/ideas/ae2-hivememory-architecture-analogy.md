---
title: AE2 与 HiveMemory 的架构同构性
status: idea
owner: project
scope: ae2-inspired-architecture-and-aios-research
code_paths:
  - src/hivememory/patchouli/
  - src/hivememory/alice/
  - src/hivememory/agent_runtime/
  - src/hivememory/core/models/
related_contracts:
  - docs/contracts/mtp.md
  - docs/architecture/boundaries.md
related_documents:
  - docs/VISION.md
  - docs/architecture/overview.md
  - docs/architecture/workspace.md
  - docs/patchouli/memory-library.md
  - docs/alice/orchestration.md
last_reviewed: 2026-09-01
---

# AE2 与 HiveMemory 的架构同构性

本文记录一个来自 Applied Energistics 2（AE2）的架构类比，以及它对 HiveMemory 未来设计可能产生的启发。本文属于开放设想，不代表路线图承诺，也不能用于推断当前系统已经实现了 Workspace 子网、软件级 Capability Subnet、持久化工作流图或强隔离执行环境。

## 1. 结论摘要

AE2 与 HiveMemory 的目标不同：AE2 处理确定性的物品物流与自动合成，HiveMemory 处理带有语义不确定性的知识、能力和 Agent 行动。但二者共享一组深层结构：

> 把分散资源转化为可寻址资产，通过声明式契约把资产交给隔离的执行器，再由具有任务状态的调度层追踪中间产物、失败与最终回流。

因此，这个类比不是“存储磁盘像数据库、CPU 像调度器”的表面映射，而是一种可以帮助审查系统边界的架构语言。

类比最成立的部分有四个：

1. 持久化资产与可寻址存储；
2. Pattern/Function Schema 一类的声明式能力契约；
3. 具有中间状态和恢复位置的 Job/Frame 执行；
4. 通过网络、接口和过滤器组织资源可见性的子网结构，以及将子网进一步封装为软件能力单元的可能性。

类比最需要保留的差异也有四个：

1. AE2 的配方图通常是确定性的，Agent 任务是在线推理和反馈控制；
2. Memory 不是天然的 Workflow，只有部分类型的记忆资产具备可执行契约；
3. AE2 的过滤主要是物流隔离，不等同于面向恶意调用者的授权与安全边界；
4. HiveMemory 当前已经有存储平面和有限执行编排，但还没有完整的 Workspace 子网和持久化 Job Graph。

## 2. AE2 中可迁移的结构

本文参考 AE2 官方 1.21.1 指南中的以下机制：

- [Autocrafting](https://guide.appliedenergistics.org/1.21.1/ae2-mechanics/autocrafting)：自动合成由请求来源、Crafting CPU 和 Pattern Provider 共同完成；CPU 计算依赖并保存中间材料，Pattern Provider 向 Molecular Assembler 或外部机器投料，结果必须重新进入网络；
- [Subnetworks](https://guide.appliedenergistics.org/1.21.1/ae2-mechanics/subnetworks)：子网可以限制设备访问的存储、节省 Channel、封装复杂工厂并在内部实现并行；
- [Interface](https://guide.appliedenergistics.org/1.21.1/items-blocks-machines/interface)：Interface 可以把一个网络的存储暴露给另一个网络，未配置的 Interface 与 Storage Bus 形成“把另一个网络看成一个大箱子”的交互；
- [Storage Bus](https://guide.appliedenergistics.org/1.21.1/items-blocks-machines/storage_bus)：Storage Bus 可以将外部库存纳入网络存储，并按物品过滤、设置插入/取出方向和优先级；
- [Storage Cells](https://guide.appliedenergistics.org/1.21.1/items-blocks-machines/storage_cells)：Storage Cell 负责容量与类型承载，Drive/Chest 则提供其在网络中的挂载位置。

其中最值得迁移的不是某个方块的名称，而是以下三个设计习惯：

### 2.1 声明式黑盒契约

Processing Pattern 并不要求主网络理解外部工厂内部经过哪些步骤。它只声明输入和预期输出，外部机器可以是一台熔炉，也可以是完整的生产线。网络只依赖契约与最终回流，不依赖机器内部对象。

这与工具 Schema、Function Calling、MTP handler 和未来的 Workflow Memory 有直接对应关系：上层声明“需要什么能力”，执行器在边界内完成工作，并通过结构化结果回流。

### 2.2 作业内中间状态

Crafting CPU 保存一次 Job 的中间材料，后续步骤消费这些材料，而不是把每一步都重新解释成一个独立的用户请求。这个结构对应 Agent 运行中的 Frame、PendingAtom、Tool Result、子 Agent alias 和 materialize task。

### 2.3 子网作为封装与能力边界

AE2 子网不只是“另一个网络”。它可以拥有自己的存储、设备、路由和并行机器，并只通过显式接口把一部分能力暴露给主网。主网看到的是一个稳定端口，而不是子网内部所有设备。

这正是 Workspace 设计比“把所有工具和权限绑定到 Agent”更有扩展性的原因：Agent 是临时执行者，Workspace 才是资产、能力和策略的稳定容器。

## 3. HiveMemory 的精确映射

| AE2 概念 | HiveMemory 中更精确的对应物 | 当前状态 | 需要保留的限定 |
|:---|:---|:---|:---|
| 物品/流体栈 | `MemoryAtom`、Artifact、执行产物 | 已有基础 | 不同 MemoryType 不能被同质化处理 |
| Storage Cell | Mid/Long-term Store 或特定存储后端 | 已有基础 | 容量、类型、归档和一致性仍由 Store 决定 |
| Drive / Network Storage | `MemoryLibrary` 与 Patchouli 存储事实核心 | 已有基础 | Patchouli 还承担索引、生成、生命周期与 provenance |
| 合成请求来源 | 用户请求、Gateway Decision、System Application Service | 已有基础 | Gateway 形成入口决策，但不执行记忆或 Agent 逻辑 |
| Pattern | Tool Schema、Capability Contract、Workflow/Recipe Memory | 部分成立 | 当前没有正式的通用 Workflow Memory 类型 |
| Pattern Provider | Koakuma MTP 分发、公开 Route、能力解析器 | 部分成立 | MTP 是承载协议，不等于所有 Pattern 的完整定义 |
| Molecular Assembler / 外部机器 | Syscall、代码执行器、外部工具、子 Agent | 部分成立 | `RUN` 仍不是强安全沙箱 |
| Crafting CPU | Alice Orchestrator、`ExecutionFrame`、运行时状态 | 部分成立 | 当前没有完整 DAG、并行 specialist、持久化 Job Queue |
| CPU 中间材料 | `ExecutionProgress`、PendingAtom、materialize task | 已有基础 | 许多状态仍是进程内的，重启后不能恢复 |
| ME Controller / Network | `GlobalSystemBus`、公共 Route/Contract、System 组合根 | 已有基础 | 当前是单进程异步总线，不承诺分布式投递 |
| Subnetwork | 拥有独立资产、工具和环境的 Workspace | Workspace 已有初步资源网络边界；完整 Subnetwork 尚未实现 | 当前 Workspace ownership 不等于完整子网；历史 `WORKSPACE` visibility 只是兼容期的 team 可见性 |
| Interface + Storage Bus | 显式 Workspace Mount/Bridge | 尚未实现 | 需要独立的读取、修改、执行和导出权限 |
| Channel / Coprocessor / Crafting Storage | 并发额度、队列容量、token/cost 预算、执行槽位 | 尚未系统化 | 配额不等于授权，必须分开建模 |

### 3.1 Patchouli 更像存储平面，而不是一张磁盘

如果逐层对应，`MemoryAtom` 更像被数字化的物品栈，Store 更像具体存储元件，`MemoryLibrary` 更像存储介质协调层，而 Patchouli 则接近整个 Network Storage 加上索引、生成、版本、生命周期和证据治理系统。

Patchouli 当前已经拥有短期话题工作台、中期 Qdrant 事实库、长期文件冷藏库和 Artifact 证据旁路，详见 [MemoryLibrary 与存储层](../patchouli/memory-library.md)。它同时负责长期记忆的唯一事实所有权，因此不能被简化成“被动保存数据的磁盘”。

这与 HiveMemory 的 Memory-Native 命题一致：持久化资产承担跨执行周期的主状态，Context 是针对当前任务编译出的临时工作视图，详见 [项目愿景](../VISION.md)。

### 3.2 Pattern 对应的是可应用的记忆子集

一条原始证据、用户偏好或历史事实不是工作流。只有具备明确输入、输出、前置条件、执行方式和完成判据的资产，才真正像 AE2 Pattern。

当前 `MemoryType` 包括 `FACT`、`CODE_SNIPPET`、`REFLECTION`、`USER_PROFILE`、`WORK_IN_PROGRESS` 和 `AGENT_PROFILE`，但还没有正式的 `WORKFLOW` 或 `CAPABILITY_RECIPE` 类型，详见 [`core/models/memory.py`](../../src/hivememory/core/models/memory.py)。现阶段最接近 Pattern 的是：

- `CODE_SNIPPET`：可以由 MTP `RUN` 执行；
- `AGENT_PROFILE`：可以被发现、解析并通过 `CALL` 实例化为子 Agent；
- 某些 `WORK_IN_PROGRESS`：可以承载任务状态，但还不是完整的工作流定义。

因此，更准确的未来分层应是：

```text
Workflow / Capability Memory
    = 声明式 Pattern

MTP / Function Calling
    = Pattern 的承载与调用协议

Koakuma / Capability Resolver
    = Pattern Provider

Tool Runtime / Child Agent / External Harness
    = Molecular Assembler 或外部处理机器
```

Function Call 的 Schema 对应 Pattern；某次 Function Call 的参数则对应一次具体的“投料”，而不是 Pattern 本身。

### 3.3 Alice 与 Crafting CPU

Alice 的映射是目前最强的一组：

- `AgentRuntime` 是单个计算核心；
- `ExecutionFrame` 是任务 PCB；
- Alice Orchestrator 是当前调度器；
- PendingAtom 是作业内的延迟写入与中间产物；
- Patchouli finalize/settlement 是结果重新进入长期事实系统的提交边界。

`ExecutionFrame` 保存 `run_id`、`frame_id`、父子关系、Identity、working history、事件序列和迭代进度，使 `CALL` 后能够恢复同一个主帧，详见 [Agent Runtime](../alice/agent-runtime.md)。PendingAtom 则允许当前 run 读取自己的写入意图，却不会让半完成执行直接污染正式记忆，详见 [PendingAtom](../alice/pending-atom.md)。

但 Alice 当前还不能完整等同于 Crafting CPU。AE2 CPU 会基于确定配方计算依赖图；当前 Alice 主要进行在线的“生成—行动—观察—再生成”闭环。它目前只有根 Agent 串行调用一层子 Agent，没有持久化 DAG、并行 specialist、独立子任务重试、配额、backpressure 或完整故障恢复，详见 [多 Agent 编排](../alice/orchestration.md)。

因此，Agent Loop 更像在线反馈控制器，而不是完整的自动合成计划器。只有当持久化 Job Graph、资源预留、并行执行和恢复机制逐步形成后，Alice 才能在更完整的意义上成为 Crafting CPU。

## 4. 子网与 Workspace：最有潜力，也最不能过度宣称

AE2 子网的两个核心用途是限制设备能访问哪些存储，以及把复杂的内部生产线封装成一个对主网可见的能力。HiveMemory 可以把这一思想扩展为：

当前代码已经通过 Workspace 建立了这一方向的最小资源网络边界：`IdentityScope` 提供
不可变的访问坐标，Topic、Memory、Artifact 和 WorkspaceAsset 在各自 Store 中按
Workspace ownership 进行最终寻址，System 的共享 runtime 则继续作为公共骨架。这个
初步的“ME 网络”现状与具体代码入口见[Workspace 架构](../architecture/workspace.md)。
这里的“ME 网络”仍是架构隐喻，不表示当前已经存在独立的网络控制器、节点发现或子网
运行时；本节以下内容继续讨论尚未落地的完整主网/子网方向。

> Workspace 不是一个简单的用户标签，而是一片拥有资产、执行器、策略、配额、队列和审计边界的能力网络。

一个完整 Workspace 未来可以拥有：

- Memory namespace 与 Artifact namespace；
- 可用 Agent Profile 集合；
- Tool/Capability Registry；
- 文件、代码、网络和外部服务执行环境；
- 模型与 Provider 策略；
- token、成本、并发和时间预算；
- Job Queue 与恢复状态；
- 允许导入和导出的资源边界；
- 审计、citation 与 provenance；
- 对父 Workspace 暴露的稳定能力接口。

Agent 进入 Workspace 时，不应永久拥有这些东西，而应获得一次作用域化的执行能力：

```text
Effective Capability
  = User Identity
  ∩ Workspace Policy
  ∩ Agent Profile Allowlist
  ∩ Run-local Restrictions
```

跨 Workspace 的资源访问不应是隐式的全局 `PUBLIC` 查找，而应成为显式 Mount：

```text
WorkspaceMount
  source_workspace
  target_workspace
  resource_selector
  mode: READ | REFERENCE | WRITE | EXECUTE
  priority
  version / consistency policy
  audit policy
```

这会允许一个前端 Workspace：

- 只读挂载主 Workspace 的项目规范和公共记忆；
- 拥有自己的前端工具、Node 环境和组件库；
- 只向主 Workspace 导出构建产物、测试结果和已确认记忆；
- 对主网表现成一个稳定的 `frontend capability`，而不暴露内部 Agent 拓扑。

当前实现还没有上述完整能力容器。已落地的 Workspace 只提供身份坐标、资源 ownership、
hard boundary、传播和部分生命周期基础，不拥有独立工具、执行环境、队列、配额或审计
系统。Memory v2 的 `PUBLIC / PRIVATE / TEAM` actor read policy 只在 owning Workspace
内部生效；历史 `WORKSPACE` visibility 是按 `team_id` 解释的兼容语义，不等于当前
WorkspaceIdentity 或子网。当前文件工具也使用 Alice 配置中的单一 `workspace_path`，
不能解释成每个 Workspace 都有独立执行环境。当前事实见[Workspace 架构](../architecture/workspace.md)，
相关缺口和风险见[检索身份过滤](../patchouli/retrieval.md)与[身份隔离与执行安全治理](../governance/security/identity-and-execution-safety.md)。

此外，AE2 的过滤首先是物流边界，不是面向恶意调用者的安全授权。HiveMemory 必须由真正的资源所有者在读取、修改、执行、缓存命中、重试和后台恢复时重新校验 Identity，不能仅复制查询过滤器。

## 5. 从 Workspace 子网到软件子网与 AIOS

Workspace 主/子网还可以进一步延伸：子网不只是资源隔离域，也可以是一种向外提供特定能力的软件单元。它在内部拥有自己的记忆、工具、Agent、确定性服务、执行环境和生命周期，只通过稳定接口向主网暴露能力。

这使“软件即子网”成为一个比“Agent 即软件”更稳定的候选抽象：

- 软件的长期身份属于 Workspace/Subnet，而不是某个模型或 Agent 实例；
- Agent 是子网内部可替换的执行进程，而不是资产与权限的永久容器；
- 主网依赖 Capability Contract，不依赖子网内部使用哪个 Harness、模型或拓扑；
- 子网可以组合 Agent、传统代码、数据库、媒体引擎和外部服务，而不要求所有能力都由 LLM 实现；
- UI 是软件的一种客户端或状态投影，不是软件或 AIOS 的本体。

### 5.1 Harness-to-Harness 是主/子网调用的一种实现

社区中已经出现一个 Agent Harness 调用另一个 Agent Harness 的做法，例如由 Codex 发配任务给 Claude Code，或反向由另一个 Harness 调用 Codex。按主/子网模型，可以将其表达为：

```text
Parent Workspace / Harness
  -> TaskEnvelope
  -> limited mounts + delegated capability
  -> Child Harness Subnet
       -> plan and execute internally
       -> use its own tools and context
       -> produce Result / Artifact
  -> explicit result re-entry
  -> parent accepts, rejects or continues
```

主/子关系描述的是本次 Job 的控制拓扑，而不是 Harness 的永久等级。某次运行中 Codex 可以是主网、Claude Code 是子网；另一次运行可以完全相反。真正需要固定的是：谁拥有当前 Job、预算、取消权、最终验收和长期状态提交权。

一个外部 Harness 只有具备以下边界时，才构成真正的软件子网，而不只是“父 Agent 启动了另一个命令行程序”：

- 明确且可版本化的能力清单；
- 独立的启动、进度、挂起、取消和终止生命周期；
- 受限的 Workspace、工具、网络和文件权限；
- 显式输入与 Mount，而不是默认继承父 Harness 的完整上下文；
- 结构化 Result、Artifact、citation、warning 和资源使用信息；
- 不把内部 Runtime、缓存或可变对象直接暴露给主网；
- 对超时、部分失败、重试和已发生副作用提供稳定终态。

因此，Harness-to-Harness 是 Capability Subnet 的一个重要 Adapter，但不是子网抽象本身。子网内部也可以是普通服务、媒体引擎、数据库工作流或 Agent 与确定性组件的混合系统。

### 5.2 真正的 AIOS 不应以动态 GUI 为核心定义

部分 AIOS 构想把“由 AI 根据请求动态生成软件 GUI”视作操作系统级能力。这可以形成自适应 Shell 或交互层，却没有触及操作系统最核心的资源与生命周期问题。

一个更接近底层的 AIOS 至少需要回答：

- 资源、资产、设备和能力如何命名、发现与挂载；
- Task/Process 如何创建、调度、挂起、恢复、取消和终止；
- 文件、网络、模型、工具和外部服务如何授权；
- 用户、Workspace、Agent 与子任务的权限如何继承和缩小；
- 并发任务如何隔离，预算、配额和 backpressure 如何执行；
- 失败后哪些状态仍然可信，哪些副作用需要 reconciliation；
- 软件如何通过稳定 Capability Contract 组合另一个软件；
- 长期状态如何独立于模型、UI 和单次对话存在。

按这个标准，HiveMemory 当前组件可以形成以下候选对应：

| AIOS 角色 | HiveMemory 当前基础 | 长期可能方向 |
|:---|:---|:---|
| 持久化状态与知识文件系统 | Patchouli、MemoryAtom、Artifact | Workspace namespace、Mount、版本与恢复 |
| Process / Execution Frame | Alice、`ExecutionFrame` | Durable Job/Frame 与多运行隔离 |
| Syscall / Capability Protocol | MTP、公共 Route | 结构化 Capability Contract 与多承载协议 |
| Process Scheduler | Alice Orchestrator | DAG、并行、配额、恢复和 backpressure |
| Program/Profile Definition | Agent Profile | Workflow/Capability Memory 与软件 Manifest |
| Ingress / Shell | Gateway、Server、Frontend | 多入口意图与稳定应用 API |
| Kernel Composition / Lifecycle | HiveMemory System | Workspace/Subnet 生命周期与资源管理 |
| Application / Container Boundary | 当前可见性与单一文件 workspace 基础 | 独立 Capability Subnet |

在这个定义中，AIOS 更接近：

> 以持久化资产为主状态、以 Capability 为系统接口、以 Agent 与确定性执行器为进程、以 Workspace 为应用隔离域的运行环境。

动态 GUI 可以作为 AIOS Shell 的一种能力，但不能替代身份、调度、隔离、持久化、设备接口和恢复语义。

### 5.3 软件即子网，不等于软件内部全部 Agent 化

“软件就是一个子网体系”不应被解释为“软件的每个功能都由 Agent 完成”。确定、实时、已有成熟实现的领域能力仍应由确定性组件承担，Agent 负责语义理解、策略选择、组合和异常处理。

例如音乐软件中的音频解码、播放时钟、设备控制和格式转换具有严格的正确性与延迟要求，不适合由 LLM 临时生成执行逻辑；音乐推荐、自然语言检索、歌单策展和偏好解释则适合使用 Agent。

因此，一个软件子网应允许以下内部组合：

```text
Application Subnet
  -> deterministic domain services
  -> agentic services
  -> memory and artifact stores
  -> external service adapters
  -> resource and security policy
  -> stable public capabilities
  -> optional human-facing UI
```

主网不需要知道一个 Capability 最终由 Agent、固定算法还是混合执行器完成。只要输入、输出、权限、副作用和失败终态保持稳定，子网内部实现就可以独立演化。

### 5.4 音乐软件子网示例

一个 Music Workspace 可以包含：

```text
Music Workspace
  ├─ Deterministic Services
  │    ├─ audio decoding
  │    ├─ playback queue and clock
  │    ├─ file/library indexing
  │    ├─ device and volume control
  │    └─ format conversion
  ├─ Agentic Services
  │    ├─ recommendation
  │    ├─ natural-language retrieval
  │    ├─ playlist curation
  │    ├─ scene understanding
  │    └─ preference explanation/correction
  ├─ Memory
  │    ├─ explicit user preferences
  │    ├─ playback and skip evidence
  │    ├─ scene-specific preferences
  │    ├─ negative feedback
  │    └─ derived preference hypotheses
  └─ Public Capabilities
       ├─ music.play
       ├─ music.search
       ├─ music.recommend
       ├─ music.create_playlist
       └─ music.explain_recommendation
```

音乐记忆还需要区分不同事实层次：

- 全局用户资产：语言、无障碍、隐私和通用交互偏好；
- Music Workspace 资产：歌手、风格、年代和播放策略；
- 场景记忆：工作、运动、通勤和睡眠时的不同偏好；
- 原始证据：播放、跳过、收藏、搜索和手动删除；
- 推断记忆：例如“用户可能喜欢后摇”，必须保留来源、置信度和纠正入口；
- 临时运行状态：当前队列、设备和本次场景，不应全部物化为长期偏好。

一次跳过并不能自动证明用户不喜欢某首歌。Workspace 应把行为证据与偏好推断分开，让 Agent 可以提出、验证和修正假设，而不是把短期行为直接写成永久用户事实。

跨软件也不应默认共享全部用户记忆。主 Workspace 可以通过受控 Mount 向 Music Workspace 提供语言、日程或运动场景，但音乐子网不应因此获得用户的项目文件、开发历史或其他应用私有资产。

### 5.5 子网对外暴露 Capability，而不是具体 Agent

主网不应要求“调用 `music_recommendation_agent_v3`”，而应请求稳定能力：

```text
capability: music.recommend
inputs: scene, duration, explicit constraints
budget: time, cost, network policy
expected output: playlist artifact
```

具体由哪个 Agent、模型、算法或外部服务处理，应由音乐子网自行决定。这样子网可以更换内部模型、增加 review Agent 或改用确定性推荐算法，而不破坏主网契约。

一个候选交接模型可以包含：

```text
SubnetManifest
  id
  version
  capabilities
  accepted_inputs
  output_types
  required_mounts
  declared_side_effects
  resource_requirements

TaskEnvelope
  task_id
  caller_scope
  capability
  inputs
  mounts
  budget
  deadline

TaskResult
  status
  outputs
  artifacts
  citations
  proposed_memory_updates
  usage
  warnings
```

`proposed_memory_updates` 应保持提议语义：子网可以建议主网更新共享资产，但不能因为完成了一次调用就直接改写父网记忆。父网或资产所有者仍需执行权限、来源、冲突和生命周期判断。

### 5.6 软件子网组合的核心风险

软件子网真正困难的部分不是启动另一个 Harness，而是保证组合后仍然可控：

- 递归委派可能形成循环，需要深度限制、循环检测和总预算；
- 多个 Harness 可能同时维护同一任务状态，需要唯一 Job 所有者；
- 父子系统的 alias、文件路径和 Memory ID 需要 namespace；
- 权限只能继承或缩小，不能因为进入新 Harness 自动扩大；
- 取消、超时、预算和 trace correlation 必须沿调用链传播；
- 子网产生的文件、网络写入和其他副作用需要审计与 reconciliation；
- 自然语言回复不能作为唯一成功判据；
- Capability、输入输出和版本需要结构化契约；
- 长期状态必须明确归属于全局用户、应用 Workspace、具体 Agent 还是本次 Job；
- 子网内部观测可以向主网投影，但不能让主网直接控制内部可变状态。

只有这些边界逐步成立，Harness-to-Harness 才能从“Agent 启动另一个 Agent”升级成真正的软件组合机制。

## 6. 从 AE2 提炼出的设计原则

### 6.1 结果必须通过显式回流成为正式事实

AE2 不把“物品被放在某个箱子里”自动视为合成完成，而要求结果重新进入网络。HiveMemory 也应区分“执行器声称产生了结果”和“结果已经进入系统事实边界”。Result、Artifact、Finalize、Settlement 和 Citation 应成为可观察的提交点。

这与当前 `prepare -> run -> finalize` 和 PendingAtom 设计一致：WRITE/UPDATE 的 ACK 只是登记意图，不是正式持久化成功。

### 6.2 一个 Job 应绑定一个隔离的 CPU 状态域

AE2 一个 Crafting CPU 处理一个 Job；并行 Job 需要独立 CPU 或明确的并行容量。HiveMemory 应逐步让每个 run 拥有独立的 scheduler state、alias view、cancel scope、预算账本和 PendingAtom 归属，避免进程级共享状态相互污染。

### 6.3 黑盒可组合，内部复杂度不泄漏

Processing Pattern 不要求主网知道外部工厂内部细节。对应到 HiveMemory，父 Agent 不应获得子 Agent 的完整内部历史或执行器对象，而应接收结构化结果、可寻址 Artifact 和必要的 trace。当前 CALL 的 reply、tool result 和 alias 回填方向是合理基础。

### 6.4 容量与授权必须分开

AE2 的 Channel、Crafting Storage 和 Coprocessor 影响连接容量、并行度与吞吐；它们不是安全授权。HiveMemory 的 token、成本、并发、队列长度和模型额度同样是资源控制，而 Identity、Workspace Policy、Agent Profile 和 capability 则是授权控制。两组约束都应显式存在，不能把“没有资源”表达成“没有权限”，也不能把“有权限”误解成“系统一定有容量执行”。

### 6.5 统一资产接口，但保留类型语义

AE2 可以统一搬运多种材料，但输入、配方、存储和机器仍然有不同规则。HiveMemory 也可以用统一的可寻址资产接口承载事实、证据、代码、Profile、工作流和运行状态，但必须保留不同的真实性、修改方式、权限、生命周期和执行风险。

## 7. 当前实现的事实边界

以下内容可以作为当前已成立的对应关系：

- Patchouli 拥有长期记忆、检索、生成、生命周期和 MemoryCompiler；
- Alice 拥有 Agent run、Frame、MTP Runtime、PendingAtom 运行时视图和有限 CALL 编排；
- MTP 提供 `SEARCH / READ / RUN / WRITE / UPDATE / CALL` 六类主动能力；
- `WRITE / UPDATE` 通过 PendingAtom 延迟物化；
- `ExecutionFrame` 保存可恢复的单帧运行状态；
- WorkspaceIdentity、复合资源键和 IdentityScope 传播已经形成初步的“ME 网络”资源边界；
- Memory v2 的 `PUBLIC / PRIVATE / TEAM` actor read policy 在 Workspace ownership hard
  filter 之后执行；历史 `WORKSPACE` visibility 只作为兼容值解释；
- System、Gateway、Patchouli、Alice 通过公开 Route 和公共模型进行交接。

以下内容不能被本文当作已经实现：

- 通用 Workflow Memory 或 Capability Recipe；
- 可持久化、可恢复的 Agent DAG；
- 并行 specialist、review loop、配额和 backpressure；
- 拥有独立工具和执行环境的 Workspace；
- 作为稳定软件单元存在的 Capability Subnet；
- 结构化 `SubnetManifest / TaskEnvelope / TaskResult` 契约；
- 通用 Harness-to-Harness 生命周期和权限传播；
- 跨 Workspace 的显式 Mount/Bridge；
- 面向不受信任代码的强安全沙箱；
- 完整的跨用户、跨 Workspace 的 run-local 状态隔离。

这些边界与当前项目定位保持一致：HiveMemory 当前仍是单进程、异步、面向个人开发与实验验证的系统，详见 [PROJECT.md](../PROJECT.md) 与 [系统架构概览](../architecture/overview.md)。

## 8. 最小验证路径

如果未来要把该方向从 Idea 升级为 Plan，不应先实现完整的“AE2 for Agents”或通用 AIOS，而应先验证一个窄场景。候选纵向实验包括：

- Harness-to-Harness：主 Harness 通过结构化 TaskEnvelope 委派一次只读代码审查或受限实现任务给子 Harness；
- Application Subnet：主 Workspace 调用一个拥有独立工具链和记忆边界的 Frontend Workspace；
- Music Subnet：音乐 Workspace 使用确定性播放引擎与 Agent 推荐能力，并验证偏好记忆是否改善真实体验。

建议验证顺序如下：

1. 明确 `Workspace`、`Mount`、`Capability`、`Run` 和 `Artifact` 的所有权，不立即新增大量运行时类；
2. 定义最小 `SubnetManifest / TaskEnvelope / TaskResult`，并明确它们只是候选契约；
3. 为一个 Workspace 建立独立的工具注册表与文件根目录；
4. 让主 Workspace 通过只读 Mount 暴露少量公共 Memory/Artifact；
5. 让子 Workspace 或外部 Harness 通过公开 Capability 执行一次任务，并只返回结构化产物；
6. 验证 Identity、缓存、取消、失败、重试、预算和结果回流不会跨 Workspace 串扰；
7. 测量相比普通单 Agent CALL，子网封装是否真的改善了权限可解释性、工具复用、任务恢复、跨 Harness 组合或长期用户体验。

只有当该闭环能产生真实使用价值，且身份、失败和所有权语义能够稳定测试，才适合建立独立 Plan 或 ADR。

## 9. 开放问题

1. `team_id` 是否应演化为显式 `workspace_id`，还是 Workspace 与 Team 需要分开建模？
2. Workspace 之间应采用严格树状父子关系，还是允许经过权限检查的有向 Mount 图？
3. Memory alias 在不同 Workspace 中是局部命名，还是允许带 namespace 的全局引用？
4. Workspace 的工具环境、模型策略和文件根目录是否需要可持久化并可恢复？
5. 子 Workspace 产生的结果应该以 Artifact、MemoryAtom、Event 还是 Job Result 形式导出？
6. `EXECUTE` 能力的来源、信任级别、审批和资源限制由谁维护？
7. 跨 Workspace 的写入是否允许直接修改目标资产，还是必须通过 proposal/review/settlement？
8. 多 Agent 并发时，资源配额与权限作用域如何同时传播到子帧、后台任务和重试？
9. Harness-to-Harness 的共同最小契约应由 MTP 扩展、MCP、进程协议还是独立 Capability API 承载？
10. 主网如何验证子网“完成了任务”，而不把自然语言声明当成成功证明？
11. 软件子网是否需要统一 Manifest，还是只要求符合公共 Capability Contract？
12. 应用级 Workspace Memory 与全局用户 Memory 如何分层、挂载和撤销？
13. UI 是由子网稳定维护、由 Shell 动态投影，还是允许两种模式并存？

## 10. 升级为 Plan 的条件

本 Idea 只有在满足以下条件后才应进入 `docs/plans/`：

1. 存在真实场景或可复现实验，证明当前单层 CALL/共享 Workspace 路径确实产生了稳定问题；
2. Workspace、Mount、Capability Subnet 和 Harness 委派的所有权、权限、生命周期和失败语义已经形成公共契约草案；
3. 能够定义最小纵向闭环、迁移策略、回滚边界和验收指标；
4. 能够解决或明确隔离缓存污染、并发串扰、越权访问和执行资产逃逸；
5. 能说明需要同步更新的当前文档、MTP/Capability 契约、路由、事件、配置和帮助文档；
6. 该方向不会只是为了追求架构完整而扩大 Alice，且符合 [VISION.md](../VISION.md) 中“真实使用优先于架构完整”的门槛。

本文的最终用途不是证明 HiveMemory 已经是 AE2，而是提供一套架构检查语言：面对任何新设计，可以追问它究竟是存储资产、Pattern 契约、执行机器、Job CPU、网络端口还是子网边界。如果一个组件同时承担其中三四种角色，通常意味着所有权、状态寿命或权限边界正在重新变得模糊。
