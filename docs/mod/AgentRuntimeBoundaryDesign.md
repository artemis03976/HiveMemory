# Agent 执行引擎边界裁定与目录结构设计

**文档状态**: Draft (草案)
**适用范围**: `alice/` 子系统整体定位、`alice/runtime/` 下三个 Runtime（AgentRuntime / KoakumaRuntime / PendingAtomRuntime）及其周边执行组件、拟新建的 top-level `agent_runtime/` 目录
**核心目标**: 为 v4 架构演进遗留的"alice 子系统职责压平"问题做一次性边界裁定，明确单智能体执行引擎与多智能体编排两层的归属，确立 `agent_runtime/` 作为 top-level 共享层的定位与目标结构。本期只裁定边界、不强制走结构变动。

---

## 1. 文档目标

本文不是又一份功能开发规划，而是一次**边界宪法**的制定。它回答 PendingAtom 系列工作推进困难时暴露出的根本问题：

> alice 子系统到底是否应该承担 AgentRuntime、KoakumaRuntime、PendingAtomRuntime 三者？如果承担，三者在 alice 中的定位是什么？如果不承担，三者应划归何处？

[PendingAtomRuntimeDesign](../agent_runtime/pending_atom/PendingAtomRuntimeDesign.md) 在收尾处已经预判到这一点，并明确把"alice 子系统的物理目录拆分"留给了后续讨论。本文就是那次讨论的结论。

设计目标：

- 裁定 alice 子系统的真实身份：它是**多智能体编排域**，不是"agent 相关代码的大筐"。
- 裁定三个 Runtime 的归属：它们共同构成**单智能体执行引擎**，是一个独立的层，而非编排域的内部细节。
- 确立 top-level `agent_runtime/` 目录作为该执行引擎的物理载体，与 `engines/` 对称，被 alice 消费。
- 给出一条可复用的**三选一边界判定规则**，作为以后任何新增能力的归属裁决依据，防止"边界再次被压平"。
- 明确本期**只裁定边界、不强制结构变动**：迁移顺序作为附录给出，但不在本期承诺落地。

---

## 2. 问题的根因：被压平的两层

### 2.1 v4 演进中真正明确的与含糊的

v4 架构演进（见 [SystemArchitecture_v4.0](../architecture/evolution/SystemArchitecture_v4.0.md)）中，有两项工作的边界是完全清晰的：

1. **patchouli 收敛为纯记忆域**：Agent 生成循环、多智能体调用等非记忆职责被全部逐出，patchouli 现在专注记忆域操作并通过总线暴露能力。状态理想。
2. **顶层 HiveMemorySystem 成立**：消除了 patchouli 作为 god system 的历史包袱，作为 facade 管理对外 API、子系统装配与全局组件。状态理想。

唯独 **alice 子系统的引入不够明确**。它在原始规划里本应是多智能体编排子系统（多智能体第三阶段工作）。但 v4 演进时，没有任何系统划分能够承载 Agent 运行循环控制（LoopExecutor），于是 alice 被提前引入，**先作为其底层运行时支持而存在**。

### 2.2 "一个 alice" 实际上是"两个层"

问题的本质在于：alice 被当成"一个东西"（多智能体编排）来设计，但它实际承载了**两个职责层**，且两层被压平在同一个扁平命名空间里：

| 层 | 职责 | 当前物理位置 |
| :--- | :--- | :--- |
| 单智能体执行引擎 | 把一个 agent 的一轮跑到收敛：生成循环、工具执行、本轮未落库的瞬态写状态 | `alice/runtime/`（与编排混在一起） |
| 多智能体编排 | 协调多个 agent：谁跑、何时跑、CALL 派生、权限、结果回流、profile —— alice 的**真身** | `alice/runtime/agent/`（与引擎混在一起） |

三个 Runtime 不是"不伦不类"，它们恰好是**前一层的全部**。把它们形容为"更硬件、像体系结构/组成原理"是准确的直觉——它们就是 CPU。当前的缺陷不是"放错了子系统"，而是**两层之间那道缝没有被命名**，于是每次 PendingAtom 这种横跨缝隙的能力进来，就无处安放，只能被抹在 loop_executor / koakuma / runtime 三处，留下兼容债。

### 2.3 OS / 体系结构映射

把整套系统按操作系统视角对齐，缺失的那一层就显形了：

| HiveMemory | OS / 体系结构对应 | 职责 |
| :--- | :--- | :--- |
| patchouli | 存储子系统 + 文件系统 | 持久化记忆、检索、生成、生命周期（IO 域） |
| **AgentRuntime / LoopExecutor** | **CPU 取指-译码-执行循环** | 把一个 agent 的一轮跑到收敛 |
| **KoakumaRuntime** | **ISA + 系统调用陷入** | 解释 MTP 指令、syscall 分发、陷入记忆域 |
| **PendingAtomRuntime** | **store buffer / 写缓冲** | run 期间未落库写入的瞬态一致性视图 |
| alice（真身） | **OS 进程调度器 + IPC** | 哪个 agent 跑、何时跑、CALL=fork、结果如何回流 |

三个 Runtime 合起来是**处理器**（单智能体执行引擎），alice 的真身是跑在处理器之上的 **OS**。两者天然耦合——OS 必须通过 CPU 暴露的中断/陷入机制驱动它——所以它们装在同一个子系统疆域内是对的；但它们**必须是两个被明确命名的层**，而不是混在一坨。

---

## 3. 边界裁定

### 3.1 三选一判定规则（边界宪法）

任何新增能力进来时，按下列顺序套用，命中即归属，不再争论：

> **① 它是在"跑一个 agent 的一轮"吗？**
> （生成循环、工具执行、本轮未落库的瞬态写状态、别名解析/缓存）
> → **执行引擎层**（`agent_runtime/`）
> **不变量：这一层必须 agent-数量无关。它绝不能出现 `sub-agent` / `topology` / `下一个该调谁` 这类词汇。**
>
> **② 它是在"协调多个 agent，或管理 agent 的身份/生命周期/策略"吗？**
> （谁跑、何时跑、CALL 派生、权限策略、结果回流、profile 解析、别名收割、IPC 组装）
> → **编排层**（`alice/`，alice 的真身）
>
> **③ 它是在"管理持久化记忆"吗？**
> （存储、检索、生成、查重、生命周期）
> → **patchouli**

### 3.2 三个 Runtime 的裁定结果

**AgentRuntime / KernelLoopExecutor → 执行引擎（①）**
取指-执行循环本身是引擎职责。但当前 [loop_executor.py](../../src/hivememory/alice/runtime/agent/loop_executor.py) 中的 `_execute_call` / `_assemble_ipc_return` / `_try_harvest_alias` / `_fetch_context_refs_content`、以及 SUSPEND 分支里 `suspend_frame` / `fork_sub_frame` / 递归跑子帧 / `resume_frame` 这一整套，**是编排职责长在了引擎循环里**。这是病灶所在，留待 §5 处理。

**KoakumaRuntime → 执行引擎（①）**
MTP 解析、权限校验、syscall/tool 执行、响应格式化，是单 agent 一步动作的指令集解释器，agent-数量无关。后期 RUN 能力扩大（更多 syscall、更重的工具沙箱）只会让这一层更厚，进一步佐证它需要独立的物理空间。

**PendingAtomRuntime → 执行引擎（①）**
这是最易纠结、但裁定最干净的一个。它的三个特征恰好全部指向执行引擎：

| 观察 | 含义 |
| :--- | :--- |
| 与多智能体**无关**（单 agent 也需要写缓冲） | 命中"agent-数量无关"不变量 → 引擎，**不是**编排 |
| 与 Agent/Koakuma **耦合极深**，无法独立 | 三者同属执行引擎同一层 → 耦合是同层内聚，不是设计缺陷 |
| run 作用域、主帧+子帧共享同一实例、写入互相可见 | 正是"一个进程多线程共享同一 store buffer" |
| settlement 从 patchouli 生成流水线回填 reconcile | 正是"写缓冲最终与主存对账"；落库前 agent 能 READ 自己的 `draft_` 别名，主存看不到 |

因此 **`PendingAtom` / `Settlement` / `Focus` 等数据模型上移到 [core/models/pending.py](../../src/hivememory/core/models/pending.py) 是对的**（它们是 alice↔patchouli 缝隙上的共享词汇），但 **`PendingAtomRuntime` 这个运行时不该继续上移到 core，也不该下沉到 patchouli**——它就是执行引擎的写缓冲单元，归 `agent_runtime/`。

### 3.3 名义上不拆为顶层子系统

执行引擎与编排通过**共享可变状态**（同一个 PendingAtomRuntime / AtomCache 实例）+ **同步递归**（CALL→suspend→在同一引擎跑子帧→resume）紧耦合。在两者之间架设 `SubsystemProtocol` + `GlobalSystemBus` 契约会是过度工程——等于把一个进程内的递归调用强行切成跨总线 RPC。

**裁定：用子系统内部的层边界（top-level 共享目录），而不是子系统边界。** 见 §4。

---

## 4. 目录结构裁定：top-level `agent_runtime/`

### 4.1 依据：top-level ≠ 子系统

本项目里 top-level 目录早已分为两类，惯例已成立：

- **子系统**：`patchouli/`、`alice/`（+ 宿主层 `system/`）——实现 `SubsystemProtocol`、注册公开路由、有独立生命周期。
- **层 / 共享库**：`core/`、`engines/`、`prompts/`、`infrastructure/`、`i18n/`——被子系统消费，本身不是子系统。

`agent_runtime/` 属于**后者**。它与 `engines/` 形成精确对称：

> `engines/`（记忆引擎：感知/生成/检索/生命周期）之于 `patchouli/`
> ＝
> `agent_runtime/`（单智能体执行引擎：Agent loop / Koakuma ISA / PendingAtom 写缓冲）之于 `alice/`

patchouli 用 `engines/` 的积木装配记忆域；alice 用 `agent_runtime/` 的积木装配编排域。升级目录层级的依据是**它是一个被消费的独立层**，而不是它的体量。体量大只是附带的佐证（且后期 Koakuma RUN 能力扩大会让它更大）。

### 4.2 护栏（破则前功尽弃）

`agent_runtime/` 是库，不是子系统。以下约束是硬性的：

- **不**实现 `SubsystemProtocol`；
- **不**注册 `GlobalSystemBus` 公开路由；
- **不**持有自己的 `start()` / `stop()` / `health()`；
- 它依赖的 bus（AliceBus）、config 等一律由 alice **注入**；
- 依赖方向**严格单向**：`alice → agent_runtime`，反向**一行 import 都不允许**。

一旦这条护栏被破（例如让 `agent_runtime/` 反向 import alice 的 frame_scheduler），它就会偷偷退化回子系统，本文全部裁定失效。

### 4.3 命名：采用 `agent_runtime/`，否决 `foundation/`

`foundation/` 正是把 alice 拖垮的那类名字——语义模糊、无边界、谁都能往里塞。本次重构的全部动机就是逃离"模糊大筐"，新建一个 foundation 大筐是原地踏步，**否决**。

采用 **`agent_runtime/`**：精确命名职责（运行单个 agent 的运行时）。它与现有 `AgentRuntime` 类名的轻微重叠，反而提示包内需要一个明确的聚合根（见 §4.4）。

### 4.4 目标结构

```text
agent_runtime/                 # top-level 共享层：单智能体执行引擎（CPU）—— agent-数量无关
  engine.py                    #   引擎聚合根：持有 Koakuma+Pending+Cache+Resolver+WorkerAgent，
                               #   对外暴露 run_frame(frame) -> Result | Suspend
  loop_executor.py             #   纯取指-执行循环：命中 SUSPEND 时交还控制，不自我编排
  koakuma.py                   #   MTP ISA：解析 / 权限 / syscall 分发 / 响应格式化
  syscalls/                    #   内核 syscall 集 + 用户态工具沙箱
  pending_atom/                #   run 作用域写缓冲（PendingAtomRuntime + 私有 store）
  cache.py                     #   L1 原子缓存
  resolver.py                  #   三级别名解析
  worker_agent.py              #   LLM 调用封装
  mtp_executor.py              #   Koakuma 的 MTPExecutor adapter
  models.py                    #   ExecutionFrame / MTPExecutionContext / GenerationResult / StreamChunk

alice/                         # 子系统：多智能体编排（OS 调度器）—— alice 的真身
  system.py                    #   SubsystemProtocol 宿主
  service.py
  runtime/
    core.py                    #   AliceRuntime：组合根，注入 config/bus 装配 engine，持有编排组件
    bus.py
    orchestration/             #   编排域
      frame_scheduler.py       #     帧栈 / 拓扑
      sub_agent_dispatcher.py  #     CALL→fork→在引擎上跑子帧→resume→IPC（从 loop_executor 抽出）
      profile_resolver.py      #     agent profile 解析
      harvester.py             #     子帧别名收割
```

### 4.5 边界货币：ExecutionFrame

`ExecutionFrame` 是跨缝流动的**边界货币**：

- 编排层**造帧**（create_main_frame / fork_sub_frame）；
- 引擎层**消费帧**（run_frame 只读取 frame、推进 working_history，不决定派生）。

`frame_scheduler`、CALL 派发、`_try_harvest_alias`、`_assemble_ipc_return` 全部归编排，留在 alice；引擎只保留纯粹的 fetch-execute 循环。`ExecutionFrame` 本身定义在 `agent_runtime/models.py`（引擎侧），编排作为上游消费者引用它——这与依赖方向 `alice → agent_runtime` 一致。

---

## 5. 两个 `run` 不是一个：引擎与编排的接口契约

裁定时必须解开一个表面矛盾。两句期望当前无法同时成立：

1. "alice 回归编排与调度能力域"；
2. "alice 只需调用 `run_agent` 就能发配一个 Agent 去干活，单智能体执行对 alice 透明"。

如果 `run_agent` 把**整棵递归树**（含 CALL 派生子 agent、帧栈、收割、IPC）全包在引擎里跑完，alice 调一次就结束——**alice 就没有编排可做了**，第 1 句话变假。这正是现在 [loop_executor.py](../../src/hivememory/alice/runtime/agent/loop_executor.py) 的状态：编排逻辑长在引擎循环里。

解法是承认**两层各有一个 run，粒度不同**：

| 层 | 接口 | 语义 |
| :--- | :--- | :--- |
| 引擎（agent_runtime） | `run_frame(frame) -> Result \| Suspend` | 把**一个** agent 的一帧跑到收敛**或陷入**：命中 CALL 就返回 suspend 信号，自己不 fork。可重入，**不感知拓扑** |
| 编排（alice） | `run_agent(...)` | 驱动引擎；收到 suspend 就 fork 子帧、再调引擎跑子帧、resume、收割别名、组装 IPC |

两句话于是同时成立：引擎对 alice 透明（alice 不看 `run_frame` 内部），编排确实留在 alice（处理 suspend / 帧栈 / 子 agent 派发）。对应 OS 模型：CALL = syscall 陷入，控制权交还调度器（alice），调度器决定派生哪个进程在同一 CPU（引擎）上跑。

---

## 6. 本期范围：只裁边界，不强制结构变动

本次重构的重点是**完全解耦、彻底划清系统边界**，因此**不急于走结构变动**。本文交付的是裁定本身：

- §3 的三选一判定规则即刻生效，作为后续所有归属争议的裁决依据；
- §4 的 `agent_runtime/` 目标结构作为既定方向被冻结，但**不在本期承诺迁移落地**；
- 任何新增能力（含暂缓的 PendingAtomRuntime 后续工作）从现在起按 §3.1 归位，不再往压平的命名空间里塞。

### 6.1 迁移顺序（附录，非本期承诺）

当未来决定走结构变动时，必须遵循一条铁律：**逻辑先，目录后**。原因正是要逃离的兼容债——

当前 `loop_executor` 依赖 `frame_scheduler`（create/fork/suspend/resume）。若**先**把 `loop_executor` 移进 `agent_runtime/`、`frame_scheduler` 留在 alice，就会凭空产生 `agent_runtime → alice` 的反向依赖，只能靠加兼容 shim（正是要逃的债）或把 `frame_scheduler` 也拖进引擎（把编排错标成引擎）收场。**先移目录反而破坏了移目录的理由。**

按实际依赖分类（已核对源码）：

**可干净下沉**（只依赖 core / engines / prompts / system.config，无编排依赖、无 alice 反向依赖）：
`koakuma.py`、`syscalls/`、`pending_atom/`、`cache.py`、`resolver.py`、`agent/worker_agent.py`、`agent/mtp_executor.py`、`models.py`

**仍纠缠、必须先解缝再移**：
`agent/loop_executor.py`（循环=引擎，CALL 那坨=编排）、`agent/frame_scheduler.py`（纯编排，留 alice）、`agent/profile_resolver.py`（纯编排，留 alice）、`agent/runtime.py`（引擎入口与编排装配混在一起）

推荐顺序：

1. **CALL 反转**：引擎循环命中 SUSPEND 时返回结构化信号给调用方，`_execute_call` / 收割 / IPC 上移到 alice 编排组件（见 §4.5、§5）。
2. **结晶引擎聚合根**：让引擎聚合根持有 Koakuma+Pending+Cache+Resolver+WorkerAgent 并暴露 `run_frame`；把现散在 [AliceRuntime.__init__](../../src/hivememory/alice/runtime/core.py) 里的引擎装配收进去，AliceRuntime 只持有这一个引擎对象 + 编排组件。顺带把 `_on_pending_atom_settled` / `_refresh_l1_cache_for_settlement`（写缓冲 reconcile，属引擎）从 AliceRuntime 挪进引擎。
3. **整体迁移**：引擎自洽后一次性移到 `agent_runtime/`。

可选折中：第 0 步先把"可干净下沉"那批叶子移过去（依赖已朝下，零反转、零债），纠缠三件套留到第 1–2 步做完再移。代价是引擎短暂被劈成两处，略别扭但无债。

---

## 7. 裁定速查

| 对象 | 归属 | 层 |
| :--- | :--- | :--- |
| AgentRuntime / LoopExecutor 的循环 | `agent_runtime/` | 执行引擎 |
| KoakumaRuntime / syscalls | `agent_runtime/` | 执行引擎 |
| PendingAtomRuntime + store | `agent_runtime/pending_atom/` | 执行引擎（写缓冲） |
| AtomCache / AliasResolver / WorkerAgent | `agent_runtime/` | 执行引擎 |
| `PendingAtom` / `Settlement` / `Focus` 数据模型 | `core/models/` | 共享词汇（已落地） |
| FrameScheduler / 帧栈 / 拓扑 | `alice/` | 编排 |
| CALL 派发 / IPC 组装 / 别名收割 | `alice/` | 编排 |
| ProfileResolver | `alice/` | 编排 |
| 存储 / 检索 / 生成 / 查重 / 生命周期 | `patchouli/` + `engines/` | 记忆域 |

**两条不变量**（违反即说明归错层）：

- 执行引擎层不得出现 `sub-agent` / `topology` / `下一个该调谁` 词汇。
- `agent_runtime/` 不得反向 import `alice/`。

