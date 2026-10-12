---
title: 执行单元、执行线程与执行环境
status: idea
horizon: current
serves_version: v0.7.0
owner: project
scope: execution-unit-thread-environment-cpu-driver-and-operation-adapter
code_paths:
  - src/hivememory/workspace/contracts/
  - src/hivememory/workspace/process/
  - src/hivememory/workspace/capability/
  - src/hivememory/alice/application/cpu.py
  - src/hivememory/alice/orchestration/
  - src/hivememory/agent_runtime/mtp/runtime.py
  - src/hivememory/core/access.py
  - src/hivememory/server/models/chat.py
related_docs:
  - docs/ideas/workspace-network-task-process-architecture.md
  - docs/ideas/identity-and-access-model.md
  - docs/ideas/task-process-table-and-registration-entry.md
  - docs/ideas/external-actor-registration-and-runtime-access.md
  - docs/ideas/pending-intent-migration.md
last_reviewed: 2026-10-11
---

# 执行单元、执行线程与执行环境

**文档状态**：Idea；T-9 操作请求与执行凭据已于 2026-10-09 实施验收，取代 T-4a/T-4c 的主线程通道；2026-10-11 决定以线程 context 取代执行凭据（T-2），同日实施验收（主线程）；同日决定子线程的派生、身份与撤销（第 4 节），由[执行线程层计划](../plans/v0.7.0-execution-thread-layer.md)实施；执行环境方向仍在讨论。实施证据见 [Alice 迁移归档计划](../archive/plans/v0.7.0-alice-capability-migration.md)与[线程 context 归档计划](../archive/plans/v0.7.0-thread-context.md)，当前事实见 [Workspace 架构](../architecture/workspace.md)。
**记录日期**：2026-10-05 起；2026-10-06 由总 Idea 第四部分拆出

## 0. 文档性质

本文讨论任务进程内“谁在执行、谁发起操作”的分层：执行单元（CPU）与执行线程（actor）、CPU 驱动与 harness 实例、操作适配器与请求入口，以及 agent 与执行环境的关系。

- 2026-10-06 由[总 Idea](./workspace-network-task-process-architecture.md)第四部分拆出；前提编号（1–10）与问题编号（T-1–T-8）不变，总 Idea 与其他文档提到“第四部分”及这些编号时，均指本文。拆出时只把章节号改为本文的编号、把指向总 Idea 的引用改写为“总 Idea X”，并补充同日的两项决定：T-7、T-8 不在 v0.7.0（第 5 节），以及写入意图迁移第 1 步对回调通道的依赖（T-4）。
- 总 Idea 的章节在本文中写作“总 Idea X”，例如“总 Idea 15.5”；身份数据的界定以[身份与访问体系 Idea](./identity-and-access-model.md)为准，外部 harness 的接入以[外部 Actor Idea](./external-actor-registration-and-runtime-access.md)为准。
- 版本归属：T-1a、T-1b、T-2a、T-3、T-4、T-5a、T-9 服务于 v0.7.0 的 Alice 迁移（总 Idea 15.5）；T-7、T-8（执行环境的选择）不在 v0.7.0，见第 5 节。
- 前提是 owner 提出的出发点；已完成的问题注明决定日期与实施状态；未完成的问题只列出选项及其影响，不替 owner 作出选择，选项顺序不代表倾向；分析均标注为分析。

## 1. 前提（owner 提出）

> 2026-10-05、2026-10-06：本文讨论任务进程内“谁在执行、谁发起操作”的分层，以及 CPU 驱动、操作适配器与请求入口的划分，由 Alice 的能力层调用迁移（总 Idea 15.5）的讨论引出。它与总 Idea 第三部分的认证与授权流程、[身份与访问体系 Idea](./identity-and-access-model.md) 中的访问 context 直接相关；身份数据的界定仍以该 Idea 为准。

1. （2026-10-05）actor 不感知能力层如何调用：agent 不在工具调用中自报身份；能力层需要的访问 context 不来自 actor，由 actor 之外的某一层提供。
2. （2026-10-05）一次操作的发起者与目标分开看待：
   - **发起者**是客观事实，由某一层补全，agent 不在工具调用中声明；
   - **目标 workspace** 是 agent 的意图：当前由补全的一层给出（第 3 阶段只接受驻留 workspace，见身份 Idea 第 3 节）；实现跨 workspace 的受限穿透访问后，agent 需要能够自己带上操作目标。
3. （2026-10-06）**执行单元 = CPU**。一个执行单元内部可以包含多个执行者（actor），即**执行线程**：发起任务的 agent 是主线程，CALL 派生的每个子 agent 是一个子线程。
   - 这修订了总 Idea 第 2.1 节“合成 CPU = 任意 Actor”的映射：最初按 AE2 类比把 CPU 与 actor 画等号；分为执行单元与执行线程两层后，一个 CPU 内可以同时存在多个 actor。
   - 总 Idea 与[任务进程 Idea](./task-process-table-and-registration-entry.md) 此前所说的“CALL 派生的子执行单元”，按本条即子执行线程；两文中的定义处已改称。
4. （2026-10-06）执行单元认识其全部执行线程的身份：任一执行线程发出的工具调用，由执行单元补上请求的发起人，再交给上一层。
5. （2026-10-06）CPU 是执行流程中的顶层抽象：任务进程不认识各个 CPU 的形态，只经统一的 CPU 端口调用。CPU 内部可以有多种 backend，即各个 agent harness 的实现（harness 实例）。
   - CPU 端口的实现称 **CPU 驱动**：把输入清单与控制（开始、取消、关闭）翻译成该 harness 的协议，并把输出翻译回事件流与终态结果。驱动按协议编写，一个驱动可以覆盖支持同一协议的多个 harness（外部 Actor Idea 1.2）。
   - 一个运行中的执行单元即 CPU 驱动加上它驱动的 harness 实例。现有的 `AliceCPU` 是 Alice 的 CPU 驱动；测试用的 `ScriptedCPU` 是没有真实 harness 的驱动。
6. （2026-10-06）adapter 称**操作适配器**（operation adapter）：把任意外部操作请求归一化为 workspace 统一能力层 API 的一次调用。例如 MTP READ 与将来经 MCP 提供的记忆读取，都归一化为 `MemoryApplicationService.read`；管理员操作同样经 HTTP 操作适配器归一化，只是因身份与操作意图不同而映射到不同的方法。定义与边界见[外部 Actor Idea](./external-actor-registration-and-runtime-access.md#34-操作适配器的定义与边界) 3.4。
   - 操作适配器属于接入侧面，不在 CPU 驱动之内：plugin 模式只经操作适配器访问能力层，没有进程，也没有驱动。controller 模式下两者如何连接（驱动把绑定进程的操作适配器端点交给 harness，工具调用经它回到 workspace），见 T-4。
   - 驱动与操作适配器分别对应外部 Actor Idea 1.2 中 harness 登记的执行侧面与接入侧面。
   - （owner，2026-10-10）管理员（唯一的 root 角色 `system`）经 HTTP 操作适配器调用的是独立的 root API，不与 actor 共用能力层方法；本条“只是因身份与操作意图不同而映射到不同的方法”相应改为“映射到不同的 API 面”。见总 Idea 15.12。
   - （owner，2026-10-09）操作适配器不直接调用能力层 API：它把协议请求解析为固定类型的操作请求，交回 CPU，由 CPU 调用对应的能力层 API；只有 CPU 持有能力层的入口。请求与能力方法一一对应，见 T-9。
7. （2026-10-06）HTTP server 承担两个角色：对任务请求，它是**请求入口**，`POST /chat` 进入唯一注册入口、创建任务进程，这不是能力层的操作（总 Idea P-4b 已决定不设“创建任务”类 operation）；对管理操作，它是 HTTP 操作适配器。
8. （2026-10-06）“kernel”是早期架构遗留的命名，不再作为概念使用；harness 实例不称 kernel。代码与文档中的剩余使用点见 [kernel 遗留命名清理 Todo](../todo/kernel-legacy-naming.md)。
9. （2026-10-06）**agent 与执行环境解耦**：外部 harness 最初被视为一个单独的 agent；现在它是一个执行环境（CPU 的 backend）。理论上可以在外部 harness 之上再应用 Agent Profile（例如作为系统提示词），外部协议是否支持另当别论。用户在前端或任何交互环境中分别选择要使用的 agent 与执行环境，外部 harness 本身与单个 agent 的概念解耦。这一条决定了[外部 Actor Idea](./external-actor-registration-and-runtime-access.md#e-6-actor-与执行单元的对应) E-6a。
   - 分析：agent 的身份、访问登记、记忆可见性与（迁入能力层 operation 控制之后的）权限都只跟随 agent，由 HiveMemory 强制执行，与执行环境无关；更换执行环境不改变 agent 能看到什么、能做什么。Profile 的权限字段并入能力层的 operation 控制（总 Idea 15.4）是这一点成立的前提：现有的 MTP 动词检查只在 Alice 内部生效。
   - 分析：Profile 中的执行参数（persona、`model_name`、`temperature`、`top_p`、`language`）能否生效取决于执行环境，对外部 harness 只能尽力应用；应用结果的承载见总 Idea P-10a。按 ACP 仓库文档（2026-10-06 核对），`session/new` 的参数为 `cwd`、`mcpServers` 等，未见系统提示词字段；模型只能在 harness 自己声明的配置项（`configOptions` 中 `category` 为 `model` 的选项）中经 `session/set_config_option` 选择。persona 只能作为首轮 prompt 内容或各 harness 私有的扩展注入，约束力弱于系统提示词。
   - 分析：Alice 成为可选的执行环境之一；同一个 Agent Profile 可以在不同执行环境中运行，便于对照评估。交互记录或记忆来源可以记录执行所用的执行环境，作为观测与评估信息；按身份 Idea 的原则，来源不参与授权。
10. （2026-10-06）**执行环境的选择**：只有用户能选择执行环境，并且只能在对话开始前选择。
    - 选择执行环境不是 workspace 的能力：operation 目录中不设对应的 operation，能力层也不提供这一方法，actor 无法在执行中选择或更换执行环境。
    - owner 倾向：它大概率成为系统层级的控制，在 CPU 分配时决定实际使用的 backend（任务进程 Idea 1.2）。具体形态见 T-8。
    - 分析：执行环境在对话内固定，CALL 派生的子线程无法为自己选择另一个执行环境；外部 Actor Idea E-6b 中“允许一个进程内有多个执行单元”的选项与本条冲突。

## 2. 现状

### 2.1 术语在现有文档中的用法（文档核对，2026-10-06）

前提 3 之前，以下术语在不同文档中跨层使用：

| 术语 | 用法 | 出处 |
|:---|:---|:---|
| Actor | 执行系统，等同于 CPU | 总 Idea 第 2.1 节（修订前）、4.1 图（修订前）、7.1.1 “run/frame/MTP 归 Actor（Alice 等）”、[AGENTS.md](../../AGENTS.md) 第 4 节“Actor 执行（经 CPU 端口）” |
| Actor | 身份：`ActorIdentity`（user、agent、team），访问登记按 (user, agent) 记录 | 身份 Idea 第 2 节；[Workspace 架构](../architecture/workspace.md) |
| Actor | 一次操作的发起者 | 总 Idea 第 12 节前提 4“actor 的任何主动操作请求都导向能力层”；能力层的“actor 可见读取” |
| CPU | 执行侧面：CPU 端口的实现，由任务进程驱动 | 任务进程 Idea 1.2；[子系统公共契约](../contracts/subsystem-contracts.md) |
| CPU | 接入侧面：能力层的调用方，包括管理员直接通道与 plugin 模式的 harness | 总 Idea 第 12 节前提 7 的注；[外部 Actor Idea](./external-actor-registration-and-runtime-access.md) 1.2 |
| 执行者 | CPU 端口的实现 | 子系统公共契约 |
| 执行者 | “Agent、一次 Chat Run、子 Frame 和后台任务” | [Workspace 架构](../architecture/workspace.md) 第 1 节 |
| adapter | 入站：把外部协议的请求归一化后调用能力层 | 总 Idea 7.1.6、15.1“HTTP 入口作为 system actor 的 adapter”；外部 Actor Idea 3.4；[`configs/system_principals.yaml`](../../configs/system_principals.yaml) 的 `adapters` 字段 |
| adapter | 泛指接口转换，与接入无关 | 代码中的 `QdrantStorageAdapter`、`AgentRunStreamAdapter`、`QueueTaskAdapter` 等 |
| kernel | 早期架构遗留：MTP 的内核工具、提示词中的“memory kernel”、前端的 `stores/kernel/` | 见 [kernel 遗留命名清理 Todo](../todo/kernel-legacy-naming.md) |

按前提 3、5–10 统一后的用语：

| 用语 | 含义 |
|:---|:---|
| CPU（执行单元） | 执行流程的顶层抽象；运行时即 CPU 驱动加上它驱动的 harness 实例 |
| CPU 端口 | 任务进程调用 CPU 的统一接口 |
| CPU 驱动 | CPU 端口的实现，按协议编写；对应 harness 登记的执行侧面 |
| harness 实例 | CPU 内部的 backend，即某个 agent harness 的实现；面向用户称**执行环境**，与 agent 分别选择（前提 9、10） |
| 执行线程（actor） | 执行单元内的执行者：主线程与 CALL 派生的子线程；对应一个 `ActorIdentity` |
| 操作适配器 | 把外部操作请求归一化为能力层 API 的一次调用；对应 harness 登记的接入侧面 |
| 请求入口 | 任务请求进入唯一注册入口的传输入口，例如 HTTP 的 `POST /chat` |

- 按前提 3，Actor 的三种用法合为一层：一个执行线程对应一个 `ActorIdentity`，也是它所发起操作的发起者；CPU 只指执行单元，只有接入侧面的参与者不称 CPU（T-6）。
- 代码中泛指接口转换的“Adapter”类名不属于操作适配器，不需要随之改名。
- 事实文档（AGENTS.md、契约、Workspace 架构、Alice 文档）描述的是现有实现，措辞在相关设计实施并晋升时随之统一，不在讨论阶段修改。
- 例：总 Idea D-9 写“Alice 作为 CPU 既要实现 CPU 端口，又要调用能力层”，前提 4 写“actor 的主动操作请求导向能力层”，两句的主语按前提 3 分属执行单元与执行线程。

### 2.2 代码现状（2026-10-07 核对）

> 本节保留建立 T-9 时的背景快照。2026-10-09 操作通道已替换为执行凭据与提交函数，SEARCH、引用记录及 CALL Profile 已经能力层，输入清单只含观测标签；子帧仍沿用主线程凭据。实际接口以 [子系统契约](../contracts/subsystem-contracts.md)为准，独立线程身份尚未实施。

- **执行单元一侧已有线程表**：Alice 每次 run 的 [`RunSession`](../../src/hivememory/alice/orchestration/run_session.py) 持有 frame 注册表，`register_root_frame` 登记唯一的根 frame（主线程），`register_callee_frame` 连同 `CallRecord` 登记 callee frame（子线程）。
- **派生在 Alice 内部完成**：MTP CALL 必须指定目标 agent alias（`MTPCallRequest.target_alias`，[`core/mtp/models.py`](../../src/hivememory/core/mtp/models.py)）；Koakuma 处理 CALL 时返回 SUSPEND 并携带调用请求（[`agent_runtime/mtp/runtime.py`](../../src/hivememory/agent_runtime/mtp/runtime.py)），`RunExecutor` 交给 `CallCoordinator` 解析目标 Profile 并创建 callee frame（[`run_executor.py`](../../src/hivememory/alice/orchestration/run_executor.py)）。整个派生过程不经过任务进程。
- **子线程沿用主线程的身份与通道**：callee frame 继承 caller 的 operations 端口，`RuntimeScope` 直接取 caller frame 的 `IdentityScope`（[`call_coordinator.py`](../../src/hivememory/alice/orchestration/sub_agent/call_coordinator.py)），被调用方的 Profile 也以调用方的身份解析（[`call_context_provider.py`](../../src/hivememory/alice/orchestration/sub_agent/call_context_provider.py)）。因此子线程发起的读取、写入意图与引用记录，在授权与记录上都算作主线程的 actor；[Alice 文档](../alice/README.md#9-当前限制与设计张力)第 9 节记录了子帧的 PendingAtom 来源会记成父 Agent。
- **CPU 驱动**：[`AliceCPU`](../../src/hivememory/alice/application/cpu.py) 经全局总线请求 Alice 的统一执行路由，把 `done` 事件转换为 `CPUExecutionResult`，其余交互事件原样转交。组合根只注入这一个驱动（[`system/assembler.py`](../../src/hivememory/system/assembler.py)），没有选择 CPU 的步骤；`CPUAllocator` 名为“CPU 分配”，实际只为 CPU 准备输入（Profile 解析、附件租借、记忆与附件编译、输入清单），进程记录也不登记所用的 CPU（任务进程 Idea 1.2）。chat 请求体 `ChatRequest`（[`server/models/chat.py`](../../src/hivememory/server/models/chat.py)）必须显式给出 `agent_id`，没有选择执行环境的字段。
- **操作适配器**：HTTP 仍是唯一已登记的外部 adapter（`system_principals.yaml` 中 `adapters: ["http"]`）。MTP 的 WRITE、UPDATE 与共同引用解析已在 Alice 内转换为 `ProcessOperations` 端口调用，由 workspace 的绑定进程通道进入能力层；SEARCH、引用记录、CALL 目标 Profile 仍走直接路由，MCP 尚未接入。
- **任务进程一侧没有线程层**：进程记录只持有一份访问 context（身份 Idea I-8）；context 的运行绑定 `RunBinding`（[`core/access.py`](../../src/hivememory/core/access.py)）只有运行类型（任务进程或请求）与运行标识。
- **CPU 已有主线程回调通道**：进程调用 `CPUPort.execute` 时把 operations 作为独立参数交给 CPU，执行单元仍交回事件流与唯一终态结果（[`workspace/contracts/cpu.py`](../../src/hivememory/workspace/contracts/cpu.py)、[`operations.py`](../../src/hivememory/workspace/contracts/operations.py)）。`ProcessOperationChannel` 绑定访问 context、注册目标与 process_id，执行者不提交这些身份参数；关闭同步使通道失效并取消在途操作，防止迟到 UPDATE 冷读再登记新意图。输入清单仍携带过渡 `IdentityScope`（I-9），用于 SEARCH、引用记录与 CALL 目标 Profile 的剩余直接调用。
- **目标 workspace**：授权点显式接收目标 workspace（身份 Idea I-4）；任务进程的阶段调用与 operations 通道均以任务注册时声明并通过认证的 workspace 为目标（I-8）。[MTP 契约](../contracts/mtp.md#2-执行位置)第 2 节继续禁止 MTP 文本、alias 或进程级缓存自行指定或推导 Workspace；已迁移的操作由进程通道绑定身份，剩余直接路由仍使用过渡 `RuntimeScope`。
- **Profile 的默认可见性**：管理入口创建的 Agent Profile 使用 `MemoryAccessPolicy.public()`（[`workspace/capability/root/agent_profiles.py`](../../src/hivememory/workspace/capability/root/agent_profiles.py)）；`AgentProfile.agent_id` 取自源原子的 `index.alias`。

## 3. 流程图

实线是现状：主线程提交与引用解析已有绑定通道，子线程当前共用它；虚线表示独立子线程身份与派生准入等第 5 节的未完成问题。SEARCH、引用记录与 CALL 目标 Profile 的剩余直连路径未在图中展开。

```mermaid
flowchart LR
    subgraph WS["workspace 一侧（控制：谁、能做什么）"]
        PR["任务进程<br/>进程记录：访问 context、阶段、终态"]
        TM["主线程的身份<br/>注册时认证的 actor"]
        TC["子线程的身份与 context<br/>派生时由被调用方确定（T-1、T-2；授权见 T-3）"]
        OA["操作适配器<br/>MTP / MCP：归一化为能力层调用"]
        FILL["绑定主线程<br/>发起者与注册目标<br/>独立线程绑定待 T-1b"]
        CAP["能力层<br/>第 3 阶段：操作授权"]
    end
    subgraph EU["执行单元 = CPU（执行：怎么做）"]
        DRV["CPU 驱动<br/>Alice：AliceCPU"]
        RUN["harness 实例<br/>Alice：一次 run（RunSession）"]
        TH0["主线程<br/>根 frame"]
        TH1["子线程<br/>callee frame"]
    end
    PR -- "CPU 端口：execute" --> DRV
    DRV -- "harness 的协议" --> RUN
    RUN --- TH0
    RUN --- TH1
    PR --- TM
    PR -.- TC
    TH0 -- "提交与引用解析：绑定通道" --> OA
    TH1 -- "当前共用主线程通道" --> OA
    TH0 -. "派生调用：目标 alias（T-1a、T-3）" .-> TC
    OA --> FILL
    FILL --> CAP
    CAP --> RES["资源 owner<br/>第 4 阶段：资源授权"]
```

图中操作适配器与补全分开画，表示归一化（对应哪个方法）与补全（谁发起、目标是哪里）是两件事；两者如何组合、操作适配器如何绑定进程，见 T-4c。

图中是第 1 步实施后的现状。T-9 决定的形态尚未实施：操作适配器只产出操作请求，由 CPU 附上发出调用的线程的执行凭据，提交到 workspace 的操作入口，再由入口授权并调用能力方法。

分析（2026-10-06）：任务进程与执行单元是同一个任务的两面，前者回答“谁、能做什么”，后者回答“怎么做”；两面在任务与线程两个粒度上对应。

| 粒度 | workspace 一侧 | 执行单元一侧 | 现状 |
|:---|:---|:---|:---|
| 任务 | 任务进程：进程记录与工作集 | 一次执行：Alice 的 `RunSession` | 两侧都已实现，经 CPU 端口对接 |
| 线程 | 每个线程一份访问 context（T-2，已决定） | 执行线程：Alice 的 `ExecutionFrame` | 主线程 context 与回调通道已实现；子线程共用主线程通道，尚无独立登记与 context；外部执行单元不识别子线程（T-1） |

按操作系统的直觉，线程属于进程而不属于 CPU；按两面分开看，线程的执行状态在执行单元内，线程的身份与授权在进程一侧，两种说法并不冲突。

## 4. 已完成的问题

### T-1 执行线程身份的确定方式

**状态**：已完成（方向）。2026-10-06 决定；主线程部分随 Alice 的能力层调用迁移与线程 context 实施，子线程由[执行线程层计划](../plans/v0.7.0-execution-thread-layer.md)实施。T-1a 与 T-1b 见本节后文。

**问题**：前提 4 由执行单元为工具调用补上发起人。“某个线程是哪个 actor”以什么为准：执行单元在工具调用上直接标注 actor，还是经进程确定？

**设计**：

- **子线程的身份在派生时由派生调用确定**（owner）：父线程派生子 agent 的工具调用必须指明目标，解析这次调用即可得到接下来的执行线程以哪个 actor 运行。Alice 的 MTP CALL 必须指定目标 agent alias（2.2），这一步就确定了子线程的身份；这也是取得子线程身份最早、最可靠的时机。
- **派生必须到达进程一侧**（分析，owner 认可）：每个执行线程有自己的访问 context（T-2），而 context 只能由 workspace 一侧签发（身份 Idea I-1）。因此派生要送到进程：进程以被调用方做 Workspace 准入（principal 继承自主线程）、签发子线程的 context，并在此判断是否允许派生（T-3）。执行单元负责认出“是哪个线程”，“是谁”由进程确定，与“身份只在授权点取得”一致。
- **外部执行单元不识别子线程**（owner，最坏打算）：外部 harness 的子 agent 派生行为不纳入，一个进程内的所有 actor 共享主线程的 context，所发起的操作都记在主线程的 actor 名下。依据与影响见[外部 Actor Idea](./external-actor-registration-and-runtime-access.md#e-5-外部执行单元的执行线程) E-5。

**取舍**：原先列出的选项中，“执行单元在工具调用上直接标注 actor、上一层照此授权”使执行单元成为身份的来源，外部执行单元可以标注任意 agent，等于自报身份（前提 1）；“进程内的执行单元直接标注、外部执行单元经进程”需要按执行单元区分信任程度。外部执行单元不识别子线程后，后一种区分不再需要：只有 Alice 存在子线程，它的派生同样经过进程。

### T-2 执行线程的访问 context

**状态**：已完成。2026-10-06 决定；2026-10-11 补充进程 context 与线程 context 的分工，并以线程 context 取代执行凭据；主线程部分同日实施验收，子线程各自的 context 属于执行线程层。T-2a 见本节后文。

**问题**：身份 Idea 第 2 节规定一份 context 只属于一个 actor 和一次运行；I-8 规定进程记录只持有访问 context；任务进程 Idea Q-10 规定父进程的访问上下文需要容纳被调用方的身份与权限。一个进程内有多个执行线程时，context 采用什么形态？

**设计**（owner）：每个执行线程一份 context，因为各线程的 agent 身份不同。

- 主线程使用注册时签发的 context；子线程在派生到达进程时另签一份（T-1）；（2026-10-11 修订：注册时签发的是进程 context，主线程在进入 CPU 执行阶段时另得一份线程 context，见下文）
- 进程记录持有本进程全部线程的 context；context 的运行绑定在进程之外再带上线程标识；
- 第 3 阶段组装的 `IdentityScope` 的发起者就是发起调用的线程的 actor。
- 影响（分析）：身份 Idea 第 2 节“一份 context 只属于一个 actor 和一次运行”保持不变，运行的粒度细化到线程；I-8“进程记录只持有访问 context”与 `RunBinding`（2.2）在实施时修订；任务进程 Idea Q-10 的“父进程的访问上下文容纳被调用方的身份与权限”由进程持有的子线程 context 满足。
- 外部执行单元不识别子线程（T-1），在 HiveMemory 看来只有主线程一个 actor，仍是一份 context 对应一个 actor。

**进程 context 与线程 context**（owner，2026-10-11）：

- **两类 context 的用途与寿命**：进程 context 服务于任务进程的全部四个阶段，用于进程自己的阶段调用（系统操作，总 Idea 15.12）；线程 context 只服务于 CPU 执行阶段，用于该线程发起的 actor 操作，CPU 执行结束即撤销。进程 context 在进程关闭时最后撤销，cleanup 等收尾的阶段调用仍可授权。
- **线程身份的令牌就是线程 context**，不另设执行凭据：CPU 持有其执行线程的 context，提交操作请求时附上发出调用的线程的 context（前提 4“由执行单元补上请求的发起人”）；context 是密封凭据，CPU 读不出身份（身份 Idea I-1）。CPU 不持有进程 context。
- **操作入口只按请求类型分派**：workspace 的操作入口把请求分派到对应的 actor API，由 actor API 用附上的线程 context 授权；入口不认识进程、线程或凭据。
- 进程记录同时持有本进程的线程 context，以便在 CPU 执行结束时撤销；执行线程层之前子线程沿用主线程的线程 context（T-3 的 2026-10-10 决定）。外部 harness 只有主线程，操作适配器持有“会话 → 线程 context”的映射。
- 本条取代 T-9 的“绑定改为执行凭据”与计划前补充决定中的执行凭据、兑现位置，T-1b 的“每个线程一份执行凭据”，以及身份 Idea I-9“删除之后”的第 2 点。
- 分析：两类 context 正好对应 2026-10-10 划分的两类操作——进程 context 用于系统操作（阶段授权），线程 context 用于 actor 操作（操作授权）。
- （owner，2026-10-11）主线程的线程 context 从进程 context 派生：主线程与进程是同一个 actor，不重新认证。子线程的 context 以被调用方准入后同样从进程 context 派生（2026-10-11，见本节后文“子线程的派生、身份与撤销”）。
- 实施：[线程 context 取代执行凭据](../archive/plans/v0.7.0-thread-context.md)（2026-10-11 实施验收并归档）。

**取舍**：另一种做法是进程只持有主线程的一份 context，授权时另行传入发起线程的 actor；它使一份 context 对应多个 actor，需要修订身份 Idea 第 2 节，第 3 阶段还要另行校验该 actor 属于本进程登记的线程。

### T-4a、T-4c 主线程的回调通道

**状态**：已完成（主线程部分）。2026-10-06 决定，2026-10-07 实施验收；历史记录见[写入意图登记与读取缓存失效归档计划](../archive/plans/v0.7.0-intent-registry-and-read-cache.md)。T-4b 与子线程的独立身份见本节后文“子线程的派生、身份与撤销”。

**问题**：写入意图迁移第 1 步需要 Alice 经能力层提交与读回意图（总 Idea 15.11），通道怎样交给 Alice、操作适配器怎样绑定进程？

**设计**（owner 接受的默认决定 W2、W3）：

- **T-4a**：作为 `CPUPort.execute` 的独立参数交给 CPU，协议定义在 `workspace.contracts`，不放进输入清单。输入清单是冻结的 DTO，访问 context 及持有它的对象不能进入 DTO（身份 Idea 不变量 2）。
- **T-4c**：每个进程一个绑定实例，持有本进程主线程的访问 context 与目标 workspace，对外方法不带这两个参数，进程关闭即失效；MTP 到这些方法的翻译放在 Alice 一侧。
- 影响（分析）：执行线程层（T-1b）可以为每个线程各发一个实例，沿用同一结构。

**取舍**：放进输入清单会让持有 context 的对象进入 DTO；由执行单元在执行开始时向进程取得，需要额外的查找入口。不绑定进程、每次调用附带标记，需要在归一化之后另做补全。

**演进**（2026-10-09）：T-4c 的通道形态由 T-9 取代，绑定进程的通道对象改为每个线程的执行凭据与 workspace 的单一操作入口。T-4a 中“不放进输入清单、作为 `execute` 的独立参数”针对的是持有访问 context 的对象；T-9 之后每个进程交给 CPU 的是主线程的执行凭据，它的传递方式留给计划决定。

### T-6 只有接入侧面的参与者是否称 CPU

**状态**：已完成。2026-10-06 决定；属于讨论层面的用语，无需代码实施。

**问题**：前提 3 起 CPU 专指执行单元。总 Idea 第 12 节前提 7 把管理员操作称为“CPU 的一种接入”，并注明此处的“CPU”指能力层的调用方（外部 Actor Idea 1.2 的接入侧面）；总 Idea 4.1 图曾把管理员直接通道与 plugin 模式的 harness 画在 CPU 一栏。它们没有执行单元，只调用能力层。是保留“接入侧面的 CPU”的称谓，还是 CPU 只指执行单元？

**设计**：CPU 只指执行单元（前提 3、5）。只有接入侧面的参与者不称 CPU，它们经操作适配器调用能力层（前提 6）：管理员直接通道是经 HTTP 操作适配器、以请求级 context 发起操作的 actor（保留的 `system`），没有执行单元，也没有进程；plugin 模式的 harness 经 MCP 等操作适配器访问，同样不建进程（外部 Actor Idea 1.1）。总 Idea 第 12 节前提 7 的注、4.1 图与外部 Actor Idea 1.2 已按此修订。

**取舍**：保留“接入侧面的 CPU”会让 CPU 端口的 CPU 与能力层调用方的 CPU 继续同名。

**补充**（owner，2026-10-10）：管理员直接通道的 actor `system` 是唯一的 root 角色，调用 root API；plugin 模式的 harness 是普通 actor，只能使用 actor API（总 Idea 15.12）。

### T-9 操作请求与 workspace 的操作入口

**状态**：已完成并实施（2026-10-09）；2026-10-11 执行凭据由线程 context 取代（T-2），同日实施。2026-10-08、2026-10-09 的决定先作为 Alice 迁移阶段一落地，随后完成 SEARCH、引用记录与 CALL Profile，以及过渡身份删除；[归档计划](../archive/plans/v0.7.0-alice-capability-migration.md)保留三阶段验收。每个子线程独立签发 context/凭据仍属于后续执行线程层，本次子帧共用主线程提交函数。

**问题**（owner，2026-10-08）：第 1 步实施的进程操作通道（T-4c，2.2）有两个结构问题：

- 通道由 `TaskProcessRunner` 在每个进程中临时创建、注入 `execute`，而能力层的 API 本身是稳定的；runner 为了创建通道，必须持有能力服务（`_memory_service`），能力服务不应进入 runner；
- 通道是一个集成的方法集合：每增加一种能力，`ProcessOperations` 与通道都要增加方法，最终回到早期 Patchouli 一个门面代理全部 API 的状态，破坏按分类实现的能力层。

分析：两个问题出自同一处：按进程创建的对象把**绑定**（谁发起、属于哪个进程、目标 workspace，随进程和线程变化）与**调用面**（可调用哪些能力方法，稳定，由能力层各分类所有）放在一起，因此必须在每个进程中重新声明一遍调用面。同一原因还使目标 workspace 在创建时固定（与前提 2 的跨 workspace 目标不符），关闭时也只能取消调用方的整个任务，因为副作用发生前无法再检查一次绑定。

**设计**（owner）：

- **不改走总线路由**：Patchouli 从大门面改为总线结构，是因为其内部方法的使用方复杂，随意传递端口会造成严重的管理与可读性问题；workspace 的流向比较清楚，能力层的调用方只有两类，CPU 一侧（agent 发出的请求）与管理员一侧（HTTP 及其操作适配器），两者最终都归一化到唯一的能力层调用，不需要架设总线（总 Idea 7.1.6 保持不变）。应用服务本身已经接近端口的形态，不再为每个方法另行定义端口。
- **操作适配器只产出操作请求**：适配器不直接调用能力层 API，而是把协议请求解析为固定类型的操作请求，交回 CPU，由 CPU 调用对应的能力层 API；只有 CPU 持有能力层的入口（前提 4、6）。
- **请求与能力方法一一对应**，因此：
  1. 对 actor 暴露的可用操作、解析出的操作请求、能力层的 API 与对应的 `WorkspaceOperation` 是同一套操作列表；之后可能重写其分类与命名，使之更正式；
  2. 将来重构 agent 可用操作的白名单（总 Idea 15.4）时，以这张列表为基础；
  3. HTTP 管理侧理论上也应采用这种映射形态；管理员操作相对独立，暂不迁移。
     - （owner，2026-10-10，取代本点）管理员能力属于独立的 root API，不进入这张操作列表；操作请求与操作列表只描述 actor API，root API 由 HTTP 管理入口直接调用（总 Idea 15.12）。

**设计**（owner 接受的形态，2026-10-09）：

- **单一操作入口**：workspace 一侧提供一个执行操作请求的入口，只有一个泛型方法，接收操作请求与调用方，按请求类型分派到能力方法。映射只在这里做一次，不由每个 CPU 驱动各自实现；入口的接口不随能力增加而增长，增长的是按能力分类组织的请求类型。
- **请求模型**：放在 `workspace.contracts`（只依赖 core）；每个请求类型声明自己的结果类型，保留静态类型；能力方法可以直接以请求模型为参数，避免请求与方法签名两处重复。
- **多步操作经提交函数往返**：CALL 要先解析 `context_refs` 与目标 Profile 再派生，RUN 先读取记忆再在本地执行，READ 在解析后还要编译为 Agent 可读文本，适配器需要拿到结果才能继续。因此“交回 CPU”实现为：CPU 交给适配器一个只有一个方法的提交函数，绑定发出调用的线程的执行凭据；适配器只认识请求模型，不持有能力层，也不持有身份。
- **绑定改为执行凭据**：进程为每个执行线程签发不透明的执行凭据，CPU 提交请求时附上发出调用的线程的凭据，workspace 在授权点由凭据还原访问 context；进程关闭时同步吊销凭据，产生副作用的路径在副作用前检查凭据是否仍有效，不再取消调用方的任务。（2026-10-11 取代：线程身份的令牌改为线程 context，CPU 执行结束即撤销，见 T-2）
- **装配**：runner 只签发与吊销凭据，不持有能力服务；能力服务由组合根装配一次，交给 System 门面（HTTP）与操作入口。
- **分层规则不变**：Alice 只依赖 `workspace.contracts` 中的请求模型与入口协议，不需要允许其他子系统导入 `workspace.capability`。

影响（分析）：

- T-4c 由本条取代；T-1b 的方向由本条确定：每个线程一份执行凭据，派生时签发，由 CPU 在每次提交时附上。（2026-10-11 修订为每个线程一份线程 context，见 T-2）
- 现有 `WorkspaceOperation` 的粒度比能力方法粗：`resource.read` 覆盖点读、alias 批读与引用解析，`memory_intent.submit` 覆盖 WRITE 与 UPDATE。合为同一套列表时，需要拆分条目或合并方法，随分类与命名的重写处理。
- MTP 动词是协议语法，与操作不是一一对应：READ、WRITE、UPDATE 各对应一个请求；RUN 由引用解析与本地执行组成，CALL 组合引用解析、目标 Profile 与派生。白名单按操作还是按动词授权，属于总 Idea 15.4“统一 operation 与 MTP 动词、系统工具的粒度”。
- 外部执行单元的操作适配器（ACP 会话提供的 MCP 等）产出同一套请求，经同一个入口提交。入口的调用方因此有两种形态：CPU 附上的执行凭据，以及请求级访问 context 的持有者（暂不迁移的 HTTP 管理侧；plugin 模式随其设计另定）。

留给计划的细节：入口调用方参数的形态（联合类型或两个入口）；凭据表的位置（CPU 分配已使 `workspace.process` 依赖能力层，凭据表放在授权一侧可避免循环）；组合根的装配顺序（Alice 目前在能力服务之前构建）；主线程凭据交给 CPU 的方式（T-4a）；目标 workspace 的携带方式（当前默认注册目标，显式目标的表达见 T-5b）。

计划前的补充决定（owner，2026-10-09）：

- **执行凭据**：进程内的不透明对象，按对象身份判定有效（与身份 Idea I-8 的进程句柄同一做法），绑定访问 context、目标 workspace 与 process_id；进程关闭时同步吊销。需要跨传输的外部 harness 由操作适配器持有“会话 → 凭据”的映射（外部 Actor Idea 3.4），凭据本身不跨传输。（2026-10-11 取代：不另设执行凭据，见 T-2）
- **兑现位置**：操作入口是唯一认识凭据的地方，先把凭据还原为访问 context，再调用能力方法；能力方法的签名不变，仍只接收访问 context，“调用方参数的形态”因此不再需要选择。（2026-10-11 取代：CPU 直接附上线程 context，入口只按请求类型分派，见 T-2）
- **操作列表**：请求类型暂与能力方法一一对应，多对一地映射到现有 `WorkspaceOperation`；正式列表随 Profile 权限与白名单重构完成（总 Idea 15.4）。
  - （owner，2026-10-10）actor 操作为 READ（含 Agent Profile）、SEARCH、RUN、WRITE、UPDATE 与只有 Alice 使用的 CALL，附件读取以后可能加入；进程阶段调用与 root 操作都不在列表中。命名在能力层分层的计划中确定（总 Idea 15.12）；2026-10-10 实施后为 `memory.read`、`memory.search`、`memory.run`、`memory.write`、`memory.update` 与 `agent.call`，当前事实见 [Workspace 架构](../architecture/workspace.md)第 4.5 节。
- **引用记录**：不是操作，也不设请求类型，是能力层引用读取的自动副作用（总 Idea P-12b）。
- 其余细节（凭据表位置、装配顺序、凭据交给 CPU 的方式、目标 workspace）由[计划](../archive/plans/v0.7.0-alice-capability-migration.md)给出。

**取舍**：

- 在 `workspace.contracts` 按能力分类声明端口：能力服务可以直接实现，但方法签名要在两处维护；
- 允许 Alice 导入 `workspace.capability`：不需要端口，但要为分层规则开例外，并让子系统持有对方的服务；
- 改走总线路由：天然解耦，但推翻总 Idea 7.1.6，并失去静态类型；
- 由各 CPU 驱动把请求映射到能力方法：每个驱动重复一张映射表，Alice 的驱动仍要导入能力层。

### T-1b 后续调用与执行线程的对应

**状态**：已完成（方向）。2026-10-09 随 T-9 确定，2026-10-11 随 T-2 修订；实施随执行线程层。

**问题**：派生只确定“出现了一个以某个 actor 运行的线程”；该线程此后的每次工具调用都要对应到它自己的 context（T-2）。外部执行单元只有主线程（T-1），本问题只涉及 Alice。

**设计**：每个执行线程一份执行凭据，派生时由进程签发；CPU 在每次提交操作请求时附上发出调用的线程的凭据（T-9）。（2026-10-11 修订：凭据改为线程 context，CPU 提交时附上发出调用的线程的 context，见 T-2）

**取舍**：原选项为“派生时签发绑定该线程 context 的句柄或端点，线程的调用都经它发出 / 每次调用附带线程标记，由进程按登记查表 / 其他”。T-9 采用前一项的凭据形式，但凭据由 CPU 附上，线程本身不持有调用端点。

### T-1a、T-2a、T-3、T-4b、T-5a 子线程的派生、身份与撤销

**状态**：已完成。2026-10-11 决定；由[执行线程层计划](../plans/v0.7.0-execution-thread-layer.md)实施。

**问题**：T-1、T-2 已确定子线程的身份由派生调用确定、派生必须到达进程、每个线程一份 context。剩下的是派生经什么途径到达进程（T-1a、T-4b）、子线程的认证与权限从哪里来（T-3，原总 Idea P-5a）、子线程 context 何时失效（T-2a），以及子线程的默认目标（T-5a）。执行线程层之前，子线程沿用主线程的线程 context 与操作集合（T-3 的 2026-10-10 决定）。

**设计**：

- **子线程的权限与可见性**（owner，T-3b）：子线程以 CALL 目标 agent 的身份运行，按它自己的操作白名单与记忆可见性授权；第 3 阶段组装的 `IdentityScope` 的发起者是被调用方，写入意图的发起者也是被调用方，进程关联不变。
- **撤销**（owner，T-2a）：子线程 context 与主线程的线程 context 一起在 CPU 执行阶段结束时撤销，关闭流程兜底。
- 以下由已有决定推出（分析，owner 认可）：
  - **派生途径**（T-1a、T-4b）：CALL 本就是操作列表中的 `agent.call`（T-9、总 Idea 15.12），派生就是这条请求：经操作入口到 actor API，授权后交给进程，由进程完成准入、签发与登记；
  - **派生授权**（T-3c）：调用方的 `agent.call`；
  - **子线程的认证**（T-3a）：与主线程一样从进程 context 派生，actor 换为被调用方，派生时做被调用方的 Workspace 准入，principal 沿用进程的；
  - **默认目标**（T-5a）：驻留 Workspace，与主线程相同。
- 只有主线程能派生：子线程不能 CALL（只有根 frame 发起 CALL），进程也只接受主线程发起的派生。
- CPU 驱动把子线程 context 绑定为子帧的提交函数，Alice 的运行层不接触 context（T-2）。

**风险**（分析，owner 接受）：调用方持有 `agent.call` 就能让被调用方读取它自己的 PRIVATE 记忆，并经 CALL 的返回值交给调用方；闸门是 `agent.call` 与被调用方的准入。“谁能调用谁”的更细策略不在本条讨论。

**取舍**：T-3b 的其余选项中，按主线程的白名单与可见性（执行线程层之前的做法）使子线程的身份只是名义上的；取交集需要每次授权检查两份登记，并让子线程 context 另外记住调用方。T-2a 中 CALL 返回时撤销更严格，但需要另设“结束线程”的路径，而 Alice 是进程内的可信代码，外部执行单元不识别子线程（T-1）。

## 5. 未完成的问题

每个问题只列出选项及其影响，不作选择；选项顺序不代表倾向。T-1b 已于 2026-10-09 随 T-9 确定方向；T-1a、T-2a、T-3、T-4b、T-5a 已于 2026-10-11 决定，均移至第 4 节。

### T-5b 跨 workspace 目标的表达形式

**背景**：前提 2；授权点已显式接收目标 workspace（2.2）。

- **T-5b 将来的表达形式**（受限穿透访问不在 v0.7.0）：按指令给出目标，一条指令一个目标 / 按引用限定，alias 带上 workspace 限定 / 其他。影响：alias 目前按 (workspace, alias) 分区，跨 workspace 的引用必须带限定；agent 只能给出 workspace key 一类名称，`WorkspaceIdentity` 中的 owner 由补全的一层解析，解析失败与无权访问需要对 agent 表现一致。
- 影响（分析）：MTP 文本不能指定 Workspace 的现有约束，随穿透访问设计另行评审；发起者不能由工具调用指定（前提 1、2）长期成立，已经落地的进程绑定通道继续满足这一边界。

T-7、T-8 不在 v0.7.0（owner，2026-10-06）：v0.7.0 只有 Alice 一个执行环境，选择机制没有可验收的对象；它随后续版本实施，不一定是 v0.7.1。前提 9、10 已确定执行环境与 agent 的关系，v0.7.0 之后接入外部执行环境不需要再改动拓扑。

### T-7 agent 与执行环境组合的有效性

**背景**：按前提 9，用户分别选择 agent 与执行环境，并非每个组合都有效。例如 Profile 指定的模型不在 harness 声明的模型配置项中；harness 无法注入 persona；执行环境要求的工作目录等条件不满足。

- **T-7a 校验时点**：注册入口创建进程之前 / CPU 分配时，在 Patchouli prepare 之前（与 Profile 解析当前的时点相同） / 执行开始后由 CPU 驱动报告 / 其他。影响：Profile 目前提前到 prepare 之前解析，是因为 prepare 可能新建 Topic 或按 LRU 结算已有话题，这些副作用不可逆（任务进程 Idea 1.2“解析时点的中间态”）；组合在 prepare 之后才被判定无效，会留下同样的副作用。
- **T-7b 无法完全应用时的处理**：拒绝请求 / 降级运行并报告未生效的项（总 Idea P-10a） / 按 Profile 字段区分：模型不可用时拒绝，persona 无法注入时降级 / 其他。

### T-8 执行环境选择的承载与系统层级的控制

**背景**：前提 10 已决定只有用户能选择执行环境、只能在对话开始前选择，且不是 workspace 的能力；owner 倾向由系统层级控制，在 CPU 分配时决定实际使用的 backend。现状：chat 请求只携带 `agent_id`，组合根只注入 Alice 一个 CPU 驱动（2.2）。

- **T-8a 选择的承载位置**：作为会话的属性在对话开始时确定，此后各任务进程从会话取得 / 每次请求都携带，由入口校验它与对话开始时的选择一致 / 其他。影响：ConversationSession 尚未实现，会话模型见[外部会话与 Topic 投影 Idea](./external-session-and-topic-projection.md#01-会话模型与-topic-池owner2026-09-28) 0.1；无论哪一项，选择都要作为任务参数到达 CPU 分配，并记入进程记录（任务进程 Idea 1.2）。
- **T-8b 默认值**：系统默认的执行环境（当前为 Alice） / Agent Profile 中声明的偏好 / 用户设置 / 其他。影响：由 Profile 声明偏好会让 agent 与执行环境重新部分耦合（前提 9）。
- **T-8c 系统层级控制的形态**：哪些执行环境可用、对哪些用户可用：由 System 在启动时随 harness 登记的执行侧面装载（外部 Actor Idea 1.2、E-1），CPU 分配按用户的选择取得驱动 / 按用户登记可用的执行环境 / 其他。影响：向前端列出可选执行环境的接口属于 System 层面的 API，不经 workspace（总 Idea 15.3）。

### 与其他问题的关联

| 本文问题 | 相关问题 |
|:---|:---|
| T-1a、T-1b、T-2a、T-3、T-4b、T-5a | T-1、T-2、T-9；总 Idea 15.4、15.12、P-5a（并入 T-3）；身份 Idea I-8；任务进程 Idea Q-10；外部 Actor Idea E-5 |
| T-5b | 身份 Idea I-4 与第 3 节（受限穿透访问） |
| T-7 | 前提 9；总 Idea P-10a；任务进程 Idea 1.2（Profile 的解析时点） |
| T-8 | 前提 10；总 Idea 15.3；外部 Actor Idea 1.2、E-1、E-6；外部会话 Idea 0.1；任务进程 Idea 1.2（CPU 分配记入进程记录） |
| T-9 | T-4、T-1b；总 Idea 7.1.6、15.4、15.5；外部 Actor Idea 3.4 |

分析（2026-10-07）：按总 Idea 15.5 的重新划分，写入意图迁移第 1 步已完成 T-4 中主线程的提交与共同引用解析部分，收窄后的 Alice 能力层调用迁移完成 T-4（2026-10-09 起以 T-9 为第一步），执行线程层实现 T-1、T-2 的决定并给出 T-1a、T-1b、T-2a、T-3 的一种实现；外部 Actor 的 controller 模式（v0.7.1）沿用同一结构，但只有主线程（T-1、外部 Actor Idea E-5），actor 与执行单元的对应见外部 Actor Idea E-6。


## 6. 与其他文档的关系

| 文档 | 关系 |
|:---|:---|
| [总 Idea](./workspace-network-task-process-architecture.md) | 本文原为其第四部分；Alice 的能力层调用迁移（15.5）、Profile 权限并入 operation 控制（15.4）、与写入意图迁移的先后（P-11）与能力层为 MTP 补齐的接口（P-12）在总 Idea 维护 |
| [身份与访问体系 Idea](./identity-and-access-model.md) | 访问 context 的界定；T-2 使运行绑定细化到线程，I-8 与 `RunBinding` 在实施时修订；T-4 决定 I-9 的删除 |
| [任务进程 Idea](./task-process-table-and-registration-entry.md) | CPU 端口与 CPU 分配（1.2）；Q-10 由 T-1、T-2 满足 |
| [外部 Actor Idea](./external-actor-registration-and-runtime-access.md) | 操作适配器的定义（3.4，2026-10-09 起只产出操作请求，见 T-9）；harness 登记的两个侧面（1.2）；外部执行单元不识别子线程（E-5）；外部 harness 是执行环境（E-6a） |
| [写入意图迁移 Idea](./pending-intent-migration.md) | 第 1 步先于 Alice 的能力层调用迁移（总 Idea P-11），需要主线程的回调通道（T-4） |

## 7. 形成计划的条件

- 满足 [Ideas 升级规则](./README.md#升级规则)与[文档治理规范](../DOCUMENTATION.md)第 8.3 节的计划约束；
- 本文不单独形成计划：其决定由 v0.7.0 中消费它们的方向实施，所需的前置决定见 [Plans 导航](../plans/README.md#v070)；
- 已完成的问题在对应计划实施、验收并晋升为事实之后，按晋升门禁写入当前事实文档。
