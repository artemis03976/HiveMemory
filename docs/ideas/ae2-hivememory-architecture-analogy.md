---
title: AE2 与 HiveMemory 的架构同构性
status: idea
owner: project
scope: ae2-inspired-architecture-research
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
  - docs/patchouli/memory-library.md
  - docs/alice/orchestration.md
last_reviewed: 2026-07-31
---

# AE2 与 HiveMemory 的架构同构性

本文记录一个来自 Applied Energistics 2（AE2）的架构类比，以及它对 HiveMemory 未来设计可能产生的启发。本文属于开放设想，不代表路线图承诺，也不能用于推断当前系统已经实现了 Workspace 子网、持久化工作流图或强隔离执行环境。

## 1. 结论摘要

AE2 与 HiveMemory 的目标不同：AE2 处理确定性的物品物流与自动合成，HiveMemory 处理带有语义不确定性的知识、能力和 Agent 行动。但二者共享一组深层结构：

> 把分散资源转化为可寻址资产，通过声明式契约把资产交给隔离的执行器，再由具有任务状态的调度层追踪中间产物、失败与最终回流。

因此，这个类比不是“存储磁盘像数据库、CPU 像调度器”的表面映射，而是一种可以帮助审查系统边界的架构语言。

类比最成立的部分有四个：

1. 持久化资产与可寻址存储；
2. Pattern/Function Schema 一类的声明式能力契约；
3. 具有中间状态和恢复位置的 Job/Frame 执行；
4. 通过网络、接口和过滤器组织资源可见性的子网结构。

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
| Subnetwork | 拥有独立资产、工具和环境的 Workspace | 尚未实现 | `WORKSPACE` 当前主要是 `team_id` 可见性过滤 |
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

当前实现还没有这样的资源容器。`PUBLIC / WORKSPACE / PRIVATE` 是同一用户内的可见性模型：`PUBLIC` 仍受 `user_id` 硬过滤，`WORKSPACE` 依赖 `team_id`，`PRIVATE` 依赖来源 Agent。当前文件工具也使用 Alice 配置中的单一 `workspace_path`，不能解释成每个 Workspace 都有独立执行环境。相关缺口和风险见 [检索身份过滤](../patchouli/retrieval.md) 与 [身份隔离与执行安全计划](../plans/identity-isolation-and-execution-safety.md)。

此外，AE2 的过滤首先是物流边界，不是面向恶意调用者的安全授权。HiveMemory 必须由真正的资源所有者在读取、修改、执行、缓存命中、重试和后台恢复时重新校验 Identity，不能仅复制查询过滤器。

## 5. 从 AE2 提炼出的设计原则

### 5.1 结果必须通过显式回流成为正式事实

AE2 不把“物品被放在某个箱子里”自动视为合成完成，而要求结果重新进入网络。HiveMemory 也应区分“执行器声称产生了结果”和“结果已经进入系统事实边界”。Result、Artifact、Finalize、Settlement 和 Citation 应成为可观察的提交点。

这与当前 `prepare -> run -> finalize` 和 PendingAtom 设计一致：WRITE/UPDATE 的 ACK 只是登记意图，不是正式持久化成功。

### 5.2 一个 Job 应绑定一个隔离的 CPU 状态域

AE2 一个 Crafting CPU 处理一个 Job；并行 Job 需要独立 CPU 或明确的并行容量。HiveMemory 应逐步让每个 run 拥有独立的 scheduler state、alias view、cancel scope、预算账本和 PendingAtom 归属，避免进程级共享状态相互污染。

### 5.3 黑盒可组合，内部复杂度不泄漏

Processing Pattern 不要求主网知道外部工厂内部细节。对应到 HiveMemory，父 Agent 不应获得子 Agent 的完整内部历史或执行器对象，而应接收结构化结果、可寻址 Artifact 和必要的 trace。当前 CALL 的 reply、tool result 和 alias 回填方向是合理基础。

### 5.4 容量与授权必须分开

AE2 的 Channel、Crafting Storage 和 Coprocessor 影响连接容量、并行度与吞吐；它们不是安全授权。HiveMemory 的 token、成本、并发、队列长度和模型额度同样是资源控制，而 Identity、Workspace Policy、Agent Profile 和 capability 则是授权控制。两组约束都应显式存在，不能把“没有资源”表达成“没有权限”，也不能把“有权限”误解成“系统一定有容量执行”。

### 5.5 统一资产接口，但保留类型语义

AE2 可以统一搬运多种材料，但输入、配方、存储和机器仍然有不同规则。HiveMemory 也可以用统一的可寻址资产接口承载事实、证据、代码、Profile、工作流和运行状态，但必须保留不同的真实性、修改方式、权限、生命周期和执行风险。

## 6. 当前实现的事实边界

以下内容可以作为当前已成立的对应关系：

- Patchouli 拥有长期记忆、检索、生成、生命周期和 MemoryCompiler；
- Alice 拥有 Agent run、Frame、MTP Runtime、PendingAtom 运行时视图和有限 CALL 编排；
- MTP 提供 `SEARCH / READ / RUN / WRITE / UPDATE / CALL` 六类主动能力；
- `WRITE / UPDATE` 通过 PendingAtom 延迟物化；
- `ExecutionFrame` 保存可恢复的单帧运行状态；
- `PUBLIC / WORKSPACE / PRIVATE` 已形成基础可见性过滤；
- System、Gateway、Patchouli、Alice 通过公开 Route 和公共模型进行交接。

以下内容不能被本文当作已经实现：

- 通用 Workflow Memory 或 Capability Recipe；
- 可持久化、可恢复的 Agent DAG；
- 并行 specialist、review loop、配额和 backpressure；
- 拥有独立工具和执行环境的 Workspace；
- 跨 Workspace 的显式 Mount/Bridge；
- 面向不受信任代码的强安全沙箱；
- 完整的跨用户、跨 Workspace 的 run-local 状态隔离。

这些边界与当前项目定位保持一致：HiveMemory 当前仍是单进程、异步、面向个人开发与实验验证的系统，详见 [PROJECT.md](../PROJECT.md) 与 [系统架构概览](../architecture/overview.md)。

## 7. 最小验证路径

如果未来要把该方向从 Idea 升级为 Plan，不应先实现完整的“AE2 for Agents”，而应先验证一个窄场景，例如“主 Workspace 调用一个拥有独立前端工具链的 Frontend Workspace”。建议验证顺序如下：

1. 明确 `Workspace`、`Mount`、`Capability`、`Run` 和 `Artifact` 的所有权，不立即新增大量运行时类；
2. 为一个 Workspace 建立独立的工具注册表与文件根目录；
3. 让主 Workspace 通过只读 Mount 暴露少量公共 Memory/Artifact；
4. 让子 Workspace 通过公开能力契约执行一次前端任务，并只返回结构化产物；
5. 验证 Identity、缓存、取消、失败、重试和结果回流不会跨 Workspace 串扰；
6. 测量相比普通单 Agent CALL，Workspace 封装是否真的改善了权限可解释性、工具复用、任务恢复或并行能力。

只有当该闭环能产生真实使用价值，且身份、失败和所有权语义能够稳定测试，才适合建立独立 Plan 或 ADR。

## 8. 开放问题

1. `team_id` 是否应演化为显式 `workspace_id`，还是 Workspace 与 Team 需要分开建模？
2. Workspace 之间应采用严格树状父子关系，还是允许经过权限检查的有向 Mount 图？
3. Memory alias 在不同 Workspace 中是局部命名，还是允许带 namespace 的全局引用？
4. Workspace 的工具环境、模型策略和文件根目录是否需要可持久化并可恢复？
5. 子 Workspace 产生的结果应该以 Artifact、MemoryAtom、Event 还是 Job Result 形式导出？
6. `EXECUTE` 能力的来源、信任级别、审批和资源限制由谁维护？
7. 跨 Workspace 的写入是否允许直接修改目标资产，还是必须通过 proposal/review/settlement？
8. 多 Agent 并发时，资源配额与权限作用域如何同时传播到子帧、后台任务和重试？

## 9. 升级为 Plan 的条件

本 Idea 只有在满足以下条件后才应进入 `docs/plans/`：

1. 存在真实场景或可复现实验，证明当前单层 CALL/共享 Workspace 路径确实产生了稳定问题；
2. Workspace 与 Mount 的所有权、权限、生命周期和失败语义已经形成公共契约草案；
3. 能够定义最小纵向闭环、迁移策略、回滚边界和验收指标；
4. 能够解决或明确隔离缓存污染、并发串扰、越权访问和执行资产逃逸；
5. 能说明需要同步更新的当前文档、MTP 契约、路由、事件、配置和帮助文档；
6. 该方向不会只是为了追求架构完整而扩大 Alice，且符合 [VISION.md](../VISION.md) 中“真实使用优先于架构完整”的门槛。

本文的最终用途不是证明 HiveMemory 已经是 AE2，而是提供一套架构检查语言：面对任何新设计，可以追问它究竟是存储资产、Pattern 契约、执行机器、Job CPU、网络端口还是子网边界。如果一个组件同时承担其中三四种角色，通常意味着所有权、状态寿命或权限边界正在重新变得模糊。
