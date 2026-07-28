---
title: Legacy Alice Phase 2 Design
status: superseded
owner: alice
scope: completed-sub-agent-call-and-pending-atom-phase
archived_at: 2026-07-28
superseded_by:
  - docs/alice/orchestration.md
  - docs/alice/pending-atom.md
  - docs/contracts/mtp.md
---

> 本文保留 CALL、子帧 IPC 与 PendingAtom 的阶段性设计过程，已停止维护。当前行为以[多 Agent 编排](../orchestration.md)、[PendingAtom](../pending-atom.md)和 [MTP 契约](../../contracts/mtp.md)为准。

# HiveMemory 多智能体子系统设计文档
## Phase 2: 子代理调用与 IPC (Sub-Agent Call & IPC)

**文档状态**: Draft (草案)
**核心负责模块**: Patchouli Kernel / KoakumaRuntime
**架构隐喻**: 进程 Fork、系统调用与进程间通信 (IPC)

---

### 1. 设计动机与阶段目标 (Motivation & Objectives)

本章阐述 HiveMemory 多智能体架构在 Phase 2 阶段的设计动机。本阶段标志着系统正式从“被动多态”向“主动协同”跨越，是构建高级复杂任务编排（如 Plan & Execute 模式）的基础底座。

#### 1.1 架构演进背景 (Architectural Evolution Background)

在 Phase 1（人偶图纸与运行时）中，我们成功实现了 Worker Agent 的高度配置化与热插拔能力。通过将 Agent 身份（Identity）与感知层的话题缓存（Topic Buffer）解耦，系统允许用户在单一话题中“手动换将”（多进程分时复用）。

然而，Phase 1 架构存在一个本质的局限性：**人偶之间是平行的孤岛。** 它们依赖于外部（用户或网关）的驱动进行上下文交接，无法在内部形成自动化的分工协作链条。

为了让 Agent 能够独立解决需要跨领域专业知识的复杂任务（例如：编写代码 -> 运行测试 -> 编写技术文档），系统必须赋予主 Agent **主动唤起子 Agent（Nested Process Execution）** 的能力。这在操作系统语境下，等同于实现了 **进程派生（Fork）** 与 **进程间通信（IPC, Inter-Process Communication）**。

#### 1.2 Phase 2 核心目标 (Core Objectives)

Phase 2 的核心使命是打通 Agent 之间的调用链路与状态同步，具体包含以下四个工程目标：

1. **MTP 协议的进程通信扩展 (IPC via MTP)**
   * 扩展原有的 MTP (Memory Tool Protocol) 指令集，引入 `CALL` 指令。
   * 允许主 Agent 通过自然语言定义任务（Task）并传递上下文指针（Context Refs），原生驱动子代理执行任务。

2. **去中心化的动态服务发现 (Dynamic Service Discovery)**
   * 践行“万物皆记忆 (Everything is a Memory)”的架构哲学，将 Agent Profile（人偶图纸）视同标准的知识型记忆原子。
   * 允许主 Agent 通过网关预检索菜单（Pre-fetching）或主动的 `SEARCH` 指令，在运行时动态发现并加载所需的专业子代理。

3. **瞬态沙盒运行时 (Transient Sandbox Runtime)**
   * 彻底解决多智能体系统的“上下文污染”问题。
   * 为被唤起的子 Agent 开辟独立、无头（Headless）、阅后即焚的 `ExecutionFrame`（运行时上下文帧），确保子代理的思维链（CoT）及试错过程不污染主话题的 Buffer。

4. **无缝的自动收割与结果回填 (Auto-Harvesting & Injection)**
   * 由 Patchouli Kernel 承担“守护进程”与“记账员”的角色。
   * 自动捕获子 Agent 在生命周期内产生的副作用（如触发 `WRITE` 生成的记忆原子别名），并将其与子 Agent 的自然语言回复混合打包，透明地注入回主 Agent 的上下文中。

#### 1.3 非目标 (Non-Goals for Phase 2)

为了防止过度设计（Over-engineering），控制系统复杂度并保障核心逻辑的极简性，我们在本阶段明确排除以下目标：

1. **不支持深层嵌套或网状通信拓扑 (Strict Star Topology)**
   * **限制**：强制系统处于调用深度为 1 的“星型拓扑”中（即：主 Agent 可以调用多个子 Agent，但子 Agent 被禁止再次发起 `CALL` 指令）。
   * **理由**：避免“递归黑洞”导致的 Token 爆炸与不可控的幻觉链式崩塌，确保系统在当前大模型能力下的绝对稳定性。

2. **不引入显式的 `RETURN` 协议指令**
   * **限制**：MTP 协议中不增加专门用于返回结果的指令。
   * **理由**：保持 MTP 仅用于“操作底层存储与执行环境”的纯洁性。依托 Kernel 拦截 LLM 生成的自然停止（`finish_reason == "stop"`），将自然语言输出隐式视为结束信号。

3. **不包含爱丽丝 (Alice) 顶层编排器**
   * 本阶段仅打通“Agent 调 Agent”的底层管道。由超级管理员（Alice）基于此管道进行复杂的“动态捏人”或 DAG 任务图规划，将作为 Phase 3/4 的任务。

---

### 2. MTP 协议扩展：`CALL` 指令 (The IPC Protocol)

为了支撑多智能体间的协同工作，我们将 Memory Tool Protocol (MTP) 的边界从“存储与外围工具调用”正式扩展至 **进程间通信 (IPC, Inter-Process Communication)**。

我们在本阶段引入且仅引入一条全新的核心指令：`CALL`。设计严格遵循“如无必要，勿增实体”的奥卡姆剃刀原则，摒弃了繁杂的系统级约束参数，将智能调度的重任交还给自然语言本身。

#### 2.1 指令语法标准 (Standard Syntax)

`CALL` 指令遵循 MTP v1.0 的三段式标准管道格式：

```text
⟪ CALL | target_agent_alias | task="..." context_refs=["alias1", "alias2"] ⟫
```

*   **`VERB` (动作)**: `CALL`。触发系统挂起当前进程，并派生（Fork）子代理进程。
*   **`TARGET` (目标)**: `target_agent_alias`。对应系统中预置或动态发现的 `AGENT_PROFILE` 虚拟原子的别名（如 `backend_doll`, `data_analyst`）。
*   **`ARGS` (参数)**: 仅支持 `task` 与 `context_refs` 两个参数。

#### 2.2 参数规范与自然语言重载 (Natural Language Overloading)

在传统的 Agent 框架中，调用子 Agent 通常需要传递 `output_format`、`max_retries`、`temperature` 等大量结构化参数。MTP 协议彻底抛弃了这种做法，转而采用 **自然语言重载 (Natural Language Overloading)**。

*   **`task` (必填 - 任务描述)**: 
    *   **定义**：子 Agent 被唤醒后接收到的首条 User Message。
    *   **用法**：主 Agent 将所有的业务背景、格式约束、边界条件直接以自然语言的形式“打包”在 `task` 中。
    *   *示例*：`task="实现登录接口的 JWT 校验逻辑。只要 Python 纯代码，不需要解释。如果依赖缺失请通过 MTP 报错。"`
    *   *优势*：极大地降低了 LLM 生成指令时的语法错误率（JSON 拼写错误），同时保持了语言模型最擅长的思维链（CoT）连贯性。

*   **`context_refs` (选填 - 共享指针)**: 
    *   **定义**：主 Agent 传递给子 Agent 的“共享内存指针”列表。
    *   **用法**：填入已经存在于主 Agent 上下文中的记忆别名（如 `["mem_api_spec_v2"]`）。

#### 2.3 零开销上下文继承：指针传递 (Pass-by-Reference)

`context_refs` 参数是 MTP 实现高效 IPC 的核心魔法，它完美模拟了操作系统中的**“共享内存 (Shared Memory)”**机制。

*   **痛点**：如果主 Agent 仅仅在 `task` 里用文字告诉子 Agent “请参考登录接口文档”，子 Agent 醒来后不得不自行花费 Token 和时间去调用 `⟪ SEARCH ⟫` 和 `⟪ READ ⟫` 重新获取文档。
*   **指针传递优化**：
    1. 当主 Agent 传入 `context_refs=["mem_api_spec_v2"]` 时，Patchouli Kernel 在拦截到 `CALL` 指令后，会直接从数据库/缓存中提取该原子的完整内容。
    2. Kernel 在为子 Agent 组装初始的 System Prompt 时，**强行将这些记忆内容塞入子 Agent 的预检索注入区 (RAG Menu)**。
    3. 子 Agent 醒来的第一秒，无需进行任何检索操作，它的“大脑”中就已经无损装载了主 Agent 提供的核心背景知识。
*   **收益**：消除了冗余的向量检索开销，降低了 Token 消耗，确保了父子进程间“知识背景”的绝对一致性。

#### 2.4 隐式的 RETURN 机制 (Implicit Return)

为了保持 MTP 协议的纯粹性，我们**拒绝引入显式的 `⟪ RETURN ⟫` 指令**。

*   **设计原则**：`CALL` 指令的结束，完全依赖于底层 LLM 调用的自然终止（即 `finish_reason == "stop"` 且未触发其他 MTP 定界符）。
*   **运作机制**：当子 Agent 认为任务已完成，输出一段不再包含 MTP 符号的自然语言总结（例如：“接口代码已编写完毕，通过了静态检查。”）并停止生成时，Patchouli Kernel 自动视其为任务完成信号。
*   （*注：关于 Kernel 如何在这一刻自动收割子 Agent 产生的新记忆并回填给主 Agent，将在第 5 章“结果回填与轨迹折叠”中详细阐述。*）

#### 2.5 基于 Profile 的静态权限隔离 (Static Permission Isolation)

`CALL` 指令的安全性，由被调用方（Target Agent）自身的图纸（Profile）属性决定，而非调用方。

即使主 Agent 在 `task` 中恶意命令子 Agent 去删除文件或修改核心记忆，只要子 Agent 图纸中的 `artifacts.permissions` 不包含 `sys_delete_file` 或 `WRITE` 权限，底层小恶魔（Koakuma）执行器就会严格实施硬拦截。这种**基于静态角色图纸的 RBAC 权限控制**，彻底阻断了多智能体协作中常见的“越权提权”漏洞。

---

### 3. 代理服务发现机制 (Service Discovery)

在多智能体系统中，主 Agent 如何知道系统中有哪些“帮手”可用，是协同工作的先决条件。借鉴 Unix 哲学中“万物皆文件（Everything is a file）”的思想，HiveMemory 确立了 **“万物皆记忆 (Everything is a Memory)”** 的服务发现基调。

我们不设立专门的 Agent 注册表或中心化的控制节点，而是将 `AGENT_PROFILE`（人偶图纸）视同为普通的知识型记忆原子。这一设计使得代理的服务发现完美融入了现有的 RAG 与 MTP 检索体系。

#### 3.1 虚拟原子的“可执行”隐喻 (The Executable Metaphor)

在操作系统的语境下：
*   **系统调用 (`sys_`)**：等同于 Shell 的内置命令（Built-in commands），如 `cd`, `echo`，无需寻址，随时可用。
*   **人偶图纸 (`AGENT_PROFILE`)**：等同于存放在 `/usr/bin` 目录下的可执行文件（Executables）。图纸原子中的 `alias` 就是执行文件名，`summary` 就是它的说明手册（man page）。

主 Agent 必须通过“环境变量 ($PATH)”或者“主动查找 (find)”，才能获取并调用这些人偶。HiveMemory 提供了以下两种互补的服务发现途径：

#### 3.2 途径 A：预检索感知 (Passive Pre-fetching)

这是系统默认且最高效的服务发现方式，利用网关的全局视野在对话起始阶段“顺手”完成加载。

*   **工作流**：
    1.  当用户输入初始需求（如：“帮我写一份详细的行业研报并翻译成英文”）时，网关（The Eye）生成重写后的 Query。
    2.  `Retrieval Familiar` (检索使魔) 使用该 Query 扫描全量记忆库。由于向量检索的天然语义关联性，不仅会召回相关的行业研报数据，还会精准召回带有“信息收集”、“英文翻译”标签的 `researcher_doll` 和 `translator_doll` 两个图纸原子。
    3.  Patchouli Kernel 在为主 Agent 拼装初始 System Prompt 时，将召回的代理原子单独归类，渲染为**可选调用的服务菜单**。
*   **Prompt 注入形态示例**：
    ```markdown
    <memory_index>[Relevant Facts & Codes]
    1. [ID: fact_industry_data] 行业基础数据集...
    
    [Available Sub-Agents (Ready to CALL)]
    2. [ID: researcher_doll] "Data Researcher" - 擅长信息收集与数据清洗
    3. [ID: translator_doll] "EN Translator" - 中英互译专家
    </memory_index>
    ```
*   **优势**：主 Agent 一“醒来”，视野中就已经备好了趁手的兵器和副手，可以直接发出 `CALL` 指令，实现了“零交互开销”的智能匹配。

#### 3.3 途径 B：主动寻址 (Active Discovery)

当任务在执行中途发生偏转，或者主 Agent 遇到超纲问题时（预检索未覆盖的盲区），主 Agent 可以利用 MTP 协议主动在系统中“摇人”。

*   **工作流**：
    1.  主 Agent 在编写完代码后，发现自己缺乏编写单元测试的信心。
    2.  它主动发出 MTP 搜索指令，限定过滤条件为代理配置：
        `⟪ SEARCH | * | query="擅长写单元测试的 Agent" filter="type:AGENT_PROFILE" ⟫`
    3.  Kernel 拦截后调用检索层，返回匹配的人偶菜单：
        `<mtp_response>[Menu]: 1. qa_tester_doll (QA Engineer) - 专注于单元测试与集成测试编写。</mtp_response>`
    4.  主 Agent 获知该 `alias`，随即发起通信：
        `⟪ CALL | qa_tester_doll | task="为上述代码编写 pytest 单元测试" ⟫`

#### 3.4 架构收益：去中心化的自组织网络 (Self-Organizing Network)

通过将服务发现机制下沉到统一的记忆检索通道，HiveMemory 获得了巨大的架构红利：

1.  **彻底消灭硬编码团队 (No Hardcoded Teams)**：开发者不需要在代码逻辑中规定“主流程 Agent 只能调用特定的 3 个子 Agent”。系统的协作拓扑结构是解耦的、动态的。
2.  **无限的横向扩展性 (Infinite Scalability)**：如果需要为系统增加一个新能力（例如“数据库优化专家”），开发者（或未来的超级管理员爱丽丝）只需向数据库中写入一个新的 `AGENT_PROFILE` 记忆原子即可。该能力会立即通过 `SEARCH` 或预检索，自动向全系统的其他 Agent 开放，无需重启服务或修改代码。
3.  **涌现协作 (Emergent Collaboration)**：Agent A 找到 Agent B，Agent B 在自己的执行帧中又找到了 Agent C，由此形成基于任务上下文自动演化的动态协作网络，这是通往高级通用人工智能（AGI）基础设施的必由之路。

---

### 4. 内核调度与瞬态沙盒 (Kernel Scheduler & Transient Sandbox)

在单核时代，Patchouli Kernel 仅需要维护一个当前正在交互的 Agent 状态。当引入 `CALL` 指令后，系统必须具备挂起（Suspend）父进程、拉起子进程（Fork & Run），并在子进程自然结束后恢复（Resume）父进程的能力。

为了防止多智能体在嵌套调用时引发灾难性的“状态污染（State Pollution）”与上下文死锁，我们重构了 Kernel 的运行机制，引入了 **运行时帧 (ExecutionFrame)** 与 **瞬态沙盒 (Transient Sandbox)**。

#### 4.1 运行时帧 (ExecutionFrame) 架构

在重构后的 Kernel 看来，不存在所谓的“主 Agent”或“子 Agent”，Kernel 是无状态的函数执行引擎。它只认 **运行时帧 (`ExecutionFrame`)**。

`ExecutionFrame` 等同于 CPU 的寄存器快照或进程控制块（PCB），它封装了 LLM 单次生成循环所需的全部局部状态：

```python
@dataclass
class ExecutionFrame:
    process_id: str             # 进程唯一 ID (如: "pid_001")
    agent_profile: AgentProfile # 当前装载的人偶图纸 (身份与权限)
    working_history: List[dict] # 传给 LLM API 的实际对话数组 (role+content)
    depth: int                  # 进程调用栈深度 (主 Agent = 0, 子 Agent = 1)
    topic_id: Optional[str]     # 挂载的感知层 Buffer ID (子进程通常为 None)
```
通过将所有会话状态从 Kernel 实例的 `self` 属性中剥离进独立游离的 `ExecutionFrame` 对象，Python 强大的 `asyncio` 协程机制天然地为我们保障了不同 Agent 堆栈的绝对隔离。

#### 4.2 装载与卸载生命周期 (Hydration & Dehydration)

主 Agent（Depth=0）与 子 Agent（Depth=1）执行相同的 MTP 递归解析循环，二者的唯一区别在于其 **生命周期的装载（进入 Kernel）与卸载（离开 Kernel）方式不同**：

*   **主 Agent 的生命周期（持久化进程）**：
    *   **装载 (Hydration)**：从感知层 (MMU) 申请分配 `TopicBuffer`，将其翻译为 `working_history` 数组并装入 Frame。
    *   **卸载 (Dehydration)**：生成彻底结束后，将新增的执行轨迹和回复组装为 `InteractionPayload`，重新 `ingest` 回感知层存档。
*   **子 Agent 的生命周期（瞬态进程）**：
    *   **装载**：不经过感知层。当 Kernel 捕获到 `CALL` 时，在内存中直接无中生有构造 Frame，其初始 `working_history` 仅包含 `[SystemPrompt, {"role":"user", "content": task}]`。
    *   **卸载**：生成结束后，提取最终的文本结果作为返回值。**整个 Frame 随之被垃圾回收 (GC) 直接销毁**，不触碰感知层的任何硬盘存档。

#### 4.3 瞬态沙盒与黑盒隔离原则 (Transient Sandbox & Black-Box Isolation)

在很多早期的开源框架中，子 Agent 的思考过程会被粗暴地拼接到主 Agent 的上下文（Context）中，导致严重的“分形嵌套陷阱（Fractal Nesting）”——主 Agent 会被子 Agent 冗长的试错日志和思维链绕晕。

HiveMemory 采用 **“黑盒原则 (Black-Box Principle)”**：
1.  **无头话题 (Headless Topic)**：子 Agent 运行在一个纯内存的“瞬态沙盒”中。它在运行期间发出的所有 `⟪ READ ⟫`、`⟪ RUN ⟫` MTP 指令及试错痕迹，都被严格限制在自己独立的 `working_history` 内。
2.  **主进程屏蔽**：主 Agent 的感知层 `LogicalBlock` 绝不记录子 Agent 的执行细节。主 Agent 只能看到一个极其干净的 MTP 调用结果：`[调用了 Backend_Doll，任务完成，返回了结果...]`。
3.  **防污染**：这种隔离确保了主话题 Buffer 的语义纯净度，让 Librarian (帕秋莉) 能够在未来提炼出极高质量的中期记忆原子，而不受噪音干扰。

#### 4.4 星型拓扑与安全防爆机制 (Star Topology Constraints)

为了防止多智能体之间陷入“无限相互召唤”的死循环黑洞，Phase 2 阶段强制执行 **调用深度为 1 的星型拓扑（Star Topology）** 限制。子代理绝不允许再次派生孙代理。

系统通过“软硬结合”的方式落地该安全限制：

1.  **软限制：动态 Prompt 裁剪 (Prompt Stripping)**
    *   在为 `ExecutionFrame` 组装 System Prompt 时，检测 `frame.depth >= 1`。
    *   若为子代理，Kernel 在渲染 MTP 权限列表时，**动态剔除 `CALL` 动词的教学**，并且**拒绝向其注入 `[Available Sub-Agents]` 服务菜单**。通过“遮蔽其双眼”，从源头降低其发起调用的概率。
2.  **硬限制：执行器兜底拦截 (Runtime Interception)**
    *   即便子 Agent 产生幻觉，硬行输出了 `⟪ CALL | xxx ⟫` 指令，底层的 Koakuma (小恶魔) 执行器在校验指令时会检查 `current_frame.depth`。
    *   一旦检测到越界，执行器瞬间阻断，并向该子 Agent 抛出标准错误注入：`<mtp_response status="error">Permission Denied: Sub-agents are not allowed to invoke CALL.</mtp_response>`，强制其退回自然语言生成状态。

---

### 5. 结果回填与轨迹折叠 (Return Mechanism & Trajectory Folding)

在操作系统中，子进程执行完毕后会通过退出状态码（Exit Code）或进程间管道（Pipes）将结果返回给父进程。在 HiveMemory 中，由于子 Agent 的生命周期被完全限制在“瞬态沙盒”中，我们必须设计一套既符合自然语言直觉，又能精准传递结构化数据的返回机制。

#### 5.1 隐式返回机制 (Implicit Return Mechanism)

为了坚守 MTP 协议的极简美学，系统**拒绝引入显式的 `⟪ RETURN ⟫` 指令**。

*   **生态位替代**：大语言模型（LLM）的自然结束（`finish_reason == "stop"` 且未输出协议定界符）即被视为最完美的 Return 信号。
*   **执行逻辑**：当子 Agent 认为分配的 `task` 已经完成，它会输出一段总结性的自然语言（例如：“登录接口的代码已编写完毕，并通过了参数边界测试。”）。Patchouli Kernel 捕获到这一自然停止事件后，自动将该段文本提取为子进程的**标准标准输出 (STDOUT)**。

#### 5.2 内核自动收割 (Kernel Auto-Harvesting of Side-Effects)

仅仅返回自然语言是不够的。子 Agent 在沙盒中运行期间，极大概率会调用 `⟪ WRITE ⟫` 或 `⟪ UPDATE ⟫` 指令，将关键代码或事实固化为记忆原子（Memory Atoms）。这些新生成的记忆，主 Agent 并不知情。

为了建立父子进程间的强关联指针，Kernel 担当起了**“记账员 (Bookkeeper)”**的角色：

1.  在子 Agent 的 Sub-Loop 启动时，Kernel 为其初始化一个空的收割队列：`harvested_pointers =[]`。
2.  当子 Agent 触发 `WRITE/UPDATE` 且小恶魔 (Koakuma) 成功将数据存入数据库并返回别名（如 `mem_login_api`）时，Kernel 默默将该别名压入 `harvested_pointers` 队列。
3.  子进程结束销毁时，Kernel 成功截获所有在瞬态生命周期内产生的“状态副作用 (Side-Effects)”。

#### 5.3 混合回填结构 (Hybrid Return Payload Assembly)

Kernel 将收集到的 **“自然语言回复”** 与 **“自动收割的记忆指针”** 进行缝合，包装成标准的 `<mtp_response>` XML 结构，一次性唤醒并回填给主 Agent。

**主 Agent 视角下接收到的 Prompt 注入形态示例**：
```markdown
⟪ CALL | backend_doll | task="实现登录接口的逻辑，保存到记忆中" ⟫
<mtp_response status="success" type="ipc_return">
[Sub-Agent Reply]:
我已经完成了登录接口的开发和静态测试，鉴权采用了跨域安全的 JWT 方案，代码已存档。

[Artifacts Generated / Updated]:
- mem_login_api_spec (API 接口逻辑代码)
- tool_jwt_validator (JWT 校验工具函数)
</mtp_response>
(主 Agent 被唤醒，结合上述结果继续生成...)
```

**架构红利**：
主 Agent 醒来后，不仅获得了详细的任务汇报，还能在 `[Artifacts]` 区域看到极其结构化的**强指针**。它立刻就知道：“如果我接下来要测试，我可以直接 `⟪ RUN | tool_jwt_validator ⟫`”。这使得智能体之间的“能力与知识传递”彻底免去了二次检索的开销。

#### 5.4 轨迹折叠与认知减负 (Trajectory Folding & Cognitive Offloading)

在主 Agent 的感知层（MMU）视角中，子 Agent 冗长复杂的试错过程（如反复 SEARCH、多次 RUN 代码失败重试）必须是一个**绝对的黑盒 (Black Box)**。

1.  **Semantic Trace 折叠**：在感知层的 `LogicalBlock` 中，这一轮极其复杂的 IPC 交互被高度压缩、折叠为一条极其简单的单行轨迹：
    `{"action": "CALL", "target": "backend_doll", "task": "...", "status": "success"}`
2.  **Librarian 减负**：当包含该 `CALL` 动作的话题超时，被 Flush 给帕秋莉 (Librarian) 进行长期记忆提炼时，帕秋莉只会看到“主 Agent 委派了任务，子 Agent 交回了成果”。这种结果导向 (Result-Oriented) 的摘要逻辑，极大地降低了 Librarian 的认知过载，避免了摘要模型陷入无意义的分形细节泥潭中。

---

### 6. 端到端执行时序流 (End-to-End Workflow)

1.  **[主 Agent]** 发出指令：`⟪ CALL | coder | task="写个爬虫" ⟫` -> **API Stop**。
2.  **[Kernel]** 捕获 `CALL`，**挂起** 主 Agent 的 ExecutionFrame。
3.  **[Kernel]** 从 DB 拉取 `coder` 图纸，构建新的 ExecutionFrame（Depth=1），**启动** Sub-Loop。
4.  **[子 Agent]** 醒来，看到 task。它可能会发 `⟪ READ ⟫`、`⟪ RUN ⟫`、`⟪ WRITE ⟫`。
5.  **[Kernel]** 处理子 Agent 的请求，**记录**其 `WRITE` 产生的 `mem_spider_code`。
6.  **[子 Agent]** 输出：“爬虫写好了” -> **API Stop (自然结束)**。
7.  **[Kernel]** 销毁子 Agent 内存帧。将最后一条总结性消息“爬虫写好了”与产生的记忆 `mem_spider_code` 缝合为 XML。
8.  **[Kernel]** **唤醒** 主 Agent，将 XML 追加到其 History 中，恢复主生成循环。
