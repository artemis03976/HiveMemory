# HiveMemory 子系统设计文档
## Memory Tool Protocol (MTP) Specification v1.0

**文档状态**: Draft (草案)
**适用阶段**: Phase 4 (Kernel Re-architecture)
**核心负责人**: Patchouli System Architect

---

### 1. 设计动机与概述 (Design Motivation & Overview)

本章旨在阐述 **Memory Tool Protocol (MTP)** 的设计初衷。MTP 不仅仅是对现有 Function Calling 机制的替代，更是 HiveMemory 从“被动检索系统”向 **“AI 操作系统 (AIOS)”** 演进的关键战略支点。它定义了 Worker Agent（用户态）与 Patchouli Kernel（内核态）之间的标准通信总线。

#### 1.1 背景与痛点 (Background & Problem Statement)

随着 HiveMemory 进入 Phase 3，帕秋莉已具备完善的读写能力。然而，在将系统接入真实 Agent 工作流时，现有的业界标准交互模式 —— **Function Calling (FC) / Tool Use** —— 暴露出了严重的系统性缺陷，阻碍了 Agent 智能上限的提升：

1.  **思维断层与心流破坏 (Cognitive Break & Flow Disruption)**
    *   LLM 的本质是基于概率的文本生成引擎，其推理过程依赖于连贯的“心流” (Chain of Thought)。
    *   标准的 FC 强制 LLM 停止自然语言生成，转而输出严格的结构化数据（JSON）。这种 **模式切换 (Mode Switch)** 类似于让人类在进行深度写作时突然停下来填写一张复杂的 Excel 表格，严重抑制了模型的发散性思维和逻辑连贯性。
    *   **MTP 的解答**：我们需要一种 **Inline Action (行内行动)** 机制，允许 Agent 在思考过程中像引用文献或输入 Shell 指令一样，自然地穿插系统调用。

2.  **Token 冗余与延迟 (Token Overhead)**
    *   JSON Schema 的定义和输出极其冗长。一个简单的读取操作可能消耗 50+ Tokens，且不支持原生的并发描述（需依赖复杂的列表结构）。
    *   **MTP 的解答**：通过极简的 **自定义协议符号**（如 `⟪ READ | id ⟫`），将交互开销降低至个位数 Token。

3.  **静态工具的局限性 (Static Tooling Limitations)**
    *   现有工具通常是硬编码在系统中的 Python 函数。Agent 无法在运行时“学习”新工具，也无法将记忆中的代码直接转化为工具。
    *   **MTP 的解答**：我们需要支持 **"Memory as a Tool"**，即允许 Agent 从记忆库中动态加载代码并执行，实现能力的无限扩展。

4.  **无状态 API 的交互困境 (Stateless Interaction Constraints)**
    *   现代 LLM 服务（OpenAI/Claude API）均为无状态流式输出，不支持原生的“暂停-插入-恢复”机制。MTP 的设计必须在这一物理限制下，通过 **Stop Sequence** 和 **Prompt Prefilling** 技术实现逻辑上的系统中断。

#### 1.2 设计目标 (Design Objectives)

MTP 的设计遵循以下三个核心原则：

1.  **心流保持 (Flow Preservation)**
    *   协议必须是对自然语言的**轻量级增强**，而非替代。Agent 应能在使用协议的同时保持自然语言的推理逻辑。协议的语法应直观、类人类思维（Action -> Target -> Args）。

2.  **OS 化演进 (Evolution to AIOS)**
    *   确立 **Kernel (内核)** 与 **User Space (用户态)** 的边界。
    *   HiveMemory 不再仅仅是一个数据库，而是承担起 **IO 调度**、**权限管理** 和 **运行时环境 (Runtime)** 的职责。MTP 即为 Agent 调用内核能力的 **Syscall (系统调用)** 接口。

3.  **动态扩展 (Dynamic Extensibility)**
    *   实现 **"Everything is a Memory"**。工具、知识、事实均以 Memory Atom 的形式存在。
    *   MTP 必须支持统一的接口来访问这些异构数据，使得 Agent 可以通过简单的指令组合，实现复杂的动态任务编排。

#### 1.3 核心术语定义 (Core Terminology)

为了明确系统边界，本文档使用以下术语：

*   **MTP (Memory Tool Protocol)**
    *   定义：Worker Agent 与 HiveMemory 系统交互的专用文本协议。
    *   特征：使用特殊定界符 `⟪...⟫` 包裹，支持流式解析。

*   **Kernel Space (内核态 / Patchouli Kernel)**
    *   定义：由帕秋莉（Librarian/Familiar）托管的高权限环境。
    *   包含：直接的向量数据库访问权限、文件系统 IO 权限、硬编码的系统工具 (`sys_` 工具集)。
    *   特性：对 Worker Agent 不可见，仅通过 MTP 指令触发。

*   **User Space (用户态 / Worker Context)**
    *   定义：Worker Agent 的运行环境，即 LLM 的上下文窗口 (Context Window)。
    *   包含：Agent 的短期记忆、注入的记忆摘要、以及沙箱化的动态工具运行结果。
    *   特性：受限环境，无法直接操作底层资源。

*   **Memory as a Tool (MaaT)**
    *   定义：一种特殊的机制，允许将存储在记忆库中的 `CODE` 类型原子，作为可执行函数被加载和运行。
    *   意义：标志着记忆系统从“只读”向“可执行”的质变。

--- 

### 2. 协议语法与格式 (Syntax & Format Specification)

本章定义 MTP 的“物理形态”。设计遵循 **“Token 极简原则”** 与 **“抗干扰原则”**，确保协议符号在自然语言和代码块中具有极高的辨识度，同时符合 LLM 的逻辑生成直觉。

#### 2.1 基础语法结构 (Basic Syntax Structure)

MTP 采用 **三段式管道结构**，由特殊的数学双角符包裹。

**标准 BNF 定义**:
```bnf
<MTP_Command> ::= "⟪" <Space> <Verb> <Space> "|" <Space> <Target> <Space> ["|" <Space> <Args>] "⟫"
```

**组件详解**:

1.  **定界符 (Delimiters)**:
    *   **左定界符**: `⟪` (U+27EA, MATHEMATICAL LEFT DOUBLE ANGLE BRACKET)
    *   **右定界符**: `⟫` (U+27EB, MATHEMATICAL RIGHT DOUBLE ANGLE BRACKET)
    *   *设计理由*: 占用 Token 少，且在 Python/JS/HTML 等常见代码语法中极少出现，避免了解析时的转义冲突。

2.  **分隔符 (Separator)**:
    *   符号: `|` (Vertical Line)
    *   *解析规则*: 解析器仅识别指令中的 **前两个** `|` 作为段落分割。`Args` 内部出现的 `|` 视为内容的一部分。

3.  **三段式逻辑**:
    *   **VERB (动作)**: 要做什么？ (e.g., `READ`, `RUN`)
    *   **TARGET (对象)**: 对谁做？ (e.g., `mem_login_script`, `*`)
    *   **ARGS (参数 - 可选)**: 具体细节。支持两种格式：
        *   *Key-Value*: `key="value"` (用于简单配置)
        *   *Raw Content*: `` `content` `` (使用反引号包裹，用于传递代码或长文本)

---

#### 2.2 指令集定义 (Instruction Set)

指令集分为 **同步原子操作**（直接返回结果）和 **异步特权信号**（触发后台流程）。

**A. 核心能力 (Core Capabilities - Sync)**

| 指令 | 说明 | 目标 (Target) 示例 | 参数 (Args) | 典型用途 |
| :--- | :--- | :--- | :--- | :--- |
| **SEARCH** | **发现**。进行模糊检索，返回 Index 菜单。 | `*` (全局) 或 `global` | `query="..."`, `filter="..."` | 未知信息的探索。不返回具体内容。 |
| **READ** | **查阅**。获取记忆原子的 Payload 内容。 | `alias` 或 `[id1, id2]` | (通常为空) | 获取已知记忆的详细代码或文档。**支持列表并行。** |
| **RUN** | **执行**。调用内核工具或记忆中的代码。 | `sys_tool` 或 `mem_tool` | `key="val"` | 执行副作用操作（读写文件、联网）。 |

**B. 特权信号 (Privileged Signals - Async)**

| 指令 | 说明 | 目标 (Target) 示例 | 参数 (Args) | 典型用途 |
| :--- | :--- | :--- | :--- | :--- |
| **WRITE** | **记录**。向帕秋莉发送高优先级保存信号。 | `*` | `title="..."`, `` content=`...` `` | 手动提交重要代码或结论。 |
| **UPDATE** | **修正**。请求更新已有记忆。 | `alias` | `` patch=`...` `` | 修正已存在的错误记忆。 |

---

#### 2.3 别名系统 (The Alias System)

为了增强语义可读性，MTP 摒弃裸 UUID，采用 **语义化别名 (Semantic Alias)** 进行寻址。

**2.3.1 命名规范**
*   格式: `snake_case` (蛇形命名法)
*   结构建议: `category` + `action/subject`
*   示例:
    *   `sys_read_file` (系统内核工具)
    *   `tool_deploy_k8s` (用户态工具原子)
    *   `fact_project_env` (事实类原子)

**2.3.2 解析机制：双层路由 (Two-Tier Resolution)**
解析器在处理 `TARGET` 字段时，遵循以下优先级：

1.  **Level 1: 上下文热映射 (Context Hot-Map)**
    *   *来源*: 当前 System Prompt 中已注入的 Index 列表。
    *   *逻辑*: 如果 Retrieval Familiar 已经检索到了某条记忆并给出了 Alias，直接使用内存中的 UUID 映射。
    *   *性能*: O(1)，100% 准确。

2.  **Level 2: 全局冷检索 (Global Cold-Lookup)**
    *   *来源*: 向量数据库全量索引。
    *   *逻辑*: 如果 Agent 凭“印象”调用了一个未在 Context 中的别名（如 `tool_old_script`），系统将其作为 Keyword 触发一次精确匹配检索。
    *   *结果*: 命中则加载；未命中则报错，提示使用 SEARCH。

---

#### 2.4 协议示例集 (The Spellbook)

以下是 MTP 在不同场景下的标准形态，用于构建 Few-Shot 教学数据。

**场景 1: 并行查阅 (Parallel Reading)**
> *Agent 希望同时查看两个相关文档以节省交互轮次。*
```text
⟪ READ | [fact_api_spec, tool_db_connector] | ⟫
```

**场景 2: 执行系统工具 (System Call)**
> *Agent 调用内核能力读取本地文件。*
```text
⟪ RUN | sys_read_file | path="./config/settings.yaml" ⟫
```

**场景 3: 记忆即工具 (Memory as a Tool)**
> *Agent 检索到了一个部署脚本，并直接执行它，传入参数。*
```text
⟪ RUN | tool_deploy_script | env="production" retries=3 ⟫
```

**场景 4: 主动写入 (Explicit Write)**
> *Agent 完成了一段核心逻辑，要求系统永久保存，内容包含复杂代码。*
```text
⟪ WRITE | * | title="Login Auth Logic" content=`
def login(user):
    # Core logic here
    pass
` ⟫
```

**场景 5: 探索未知 (Discovery)**
> *Agent 不知道有没有关于 "Redis" 的记忆。*
```text
⟪ SEARCH | * | query="Redis configuration" filter="type:CODE" ⟫
```

---

### 3. 拦截与执行机制 (Interception & Execution Mechanism)

本章定义 Worker Agent 发出 MTP 指令后，Patchouli Kernel 如何接管控制权、执行指令并将结果回填至上下文闭环中。

#### 3.1 拦截架构 (The Interception Architecture)

由于现代 LLM 推理服务（如 OpenAI/Claude）不支持原生的“暂停-恢复”流式控制，MTP 采用 **“主动中断 (Active Interruption)”** 策略。

**3.1.1 停止序列 (Stop Sequence)**
*   **配置**: 在调用 LLM API 时，强制设置 `stop=["⟫"]`（即协议的右定界符）。
*   **原理**: 一旦 LLM 生成完一个完整的指令（例如 `⟪ READ | mem_01 |`），API 会立即停止输出。此时，服务端获得的文本是以未闭合的管道符结尾的。

**3.1.2 拦截流程 (Interception Workflow)**
当 API 请求结束（Finish Reason = "stop"）时，Kernel 执行以下逻辑：
1.  **捕获 (Capture)**: 获取最后生成的未闭合文本片段。
2.  **补全 (Completion)**: 自动追加缺失的 `⟫`，构成完整的指令字符串。
3.  **解析 (Parsing)**: 使用正则或字符串分割，提取 `VERB`, `TARGET`, `ARGS`。
4.  **挂起 (Suspend)**: 此时 Worker Agent 的生成过程逻辑上被挂起，进入内核态。

---

#### 3.2 并行与递归策略 (Parallelism & Recursion)

为了解决“多指令执行效率”与“API 调用成本”的矛盾，MTP 实现了两种维度的并发策略。

**3.2.1 显式并行 (Explicit Parallelism - Syntax Level)**
针对 **数据读取 (READ)** 类操作，协议支持列表语法。这是最高效的并发模式。
*   **场景**: Agent 需要同时查阅 3 个文档。
*   **指令**: `⟪ READ | [doc_a, doc_b, doc_c] | ⟫`
*   **内核行为**: Patchouli 启动 `asyncio.gather`，并发发起 3 次向量库查询或内容提取，一次性聚合结果回填。

**3.2.2 递归中断 (Recursive Interrupt - Logic Level)**
针对 **异构指令序列**（例如：先 Search，根据结果再 Read，最后 Run），采用递归循环机制。
*   **场景**: Agent 输出 `⟪ SEARCH... ⟫`（被截断） -> Kernel 执行 -> 回填结果 -> **立即发起新的 API 请求** -> Agent 继续输出 `⟪ READ... ⟫`（再次被截断）。
*   **优势**: 保持了 Agent 的思维链（Chain of Thought）不中断。
*   **成本优化**: 依赖 LLM 厂商的 **Prompt Caching (上下文缓存)** 技术，后续的递归请求仅需计算新增 Token 的 Attention，大幅降低延迟与费用。

---

#### 3.3 注入与回填策略 (Injection & Patching)

执行完毕后，Kernel 必须将结果返回给 User Space。为了维持 Agent 的 **自我一致性 (Self-Consistency)**，我们采用 **Prompt Prefilling (预填充)** 技术。

**3.3.1 伪造助手历史 (Fake Assistant History)**
系统不将工具结果作为 System Message 插入，而是将其**伪装**成 Agent 自己刚刚生成的内容。
*   *逻辑*: 将 `指令` + `执行结果` 拼接到对话历史的 `assistant` 角色消息中，然后让 LLM 继续续写（Resume）。

**3.3.2 响应容器 (Response Container)**
使用 `<mtp_response>` XML 标签包裹内核返回的内容，起到命名空间隔离的作用。

*   **Schema**:
    ```xml
    <mtp_response status="success|error" time="ms">
        ... Payload ...
    </mtp_response>
    ```

**3.3.3 语义化回填 (Semantic Patching)**
根据指令类型，回填不同风格的内容：

*   **Type A: 数据类 (READ/SEARCH)**
    *   **策略**: 直接展示 Markdown 数据。
    *   **示例**:
        ```markdown
        ⟪ READ | [mem_01] | ⟫
        <mtp_response status="success">
        [mem_01]: def login(): ... (code content)
        </mtp_response>
        (Agent继续生成: "根据这段代码，我发现...")
        ```

*   **Type B: 动作类 (RUN/WRITE)**
    *   **策略**: 使用 **自然语言状态描述**。数字状态码对 LLM 缺乏语义引导性，自然语言能更好地触发后续生成。
    *   **示例**:
        ```markdown
        ⟪ RUN | sys_write_file | path="config.py" ⟫
        <mtp_response status="success">
        File "config.py" has been written successfully.
        </mtp_response>
        (Agent继续生成: "现在配置文件已就绪，我开始部署...")
        ```

---

### 4. 系统内核设计：工具与沙箱 (Kernel System Design)

本章定义 Patchouli Kernel 的运行时架构。为了兼顾执行效率与系统安全性，我们借鉴现代操作系统的设计理念，构建了分层级的工具执行环境。

#### 4.1 架构分层 (Layered Architecture)

系统将所有可执行能力划分为两个特权层级：**Level 0 (内核态)** 和 **Level 1 (用户态)**。

**4.1.1 Level 0: 内核系统调用 (Kernel Syscalls)**
*   **定义**: 硬编码在 Patchouli 后端 Python 代码中的原子函数。
*   **命名空间**: 强制使用 `sys_` 前缀（如 `sys_read_file`）。
*   **特性**:
    *   **常驻内存**: 随系统启动加载，无需检索，**Zero Latency**。
    *   **高权限**: 拥有直接访问宿主机文件系统、网络或数据库的特权。
    *   **不可变性**: Worker Agent 无法修改或覆盖这些逻辑。
*   **典型工具**:
    *   `sys_clock`: 获取当前时间。
    *   `sys_io_read/write`: 基础文件操作。
    *   `sys_web_search`: 联网搜索接口。

**4.1.2 Level 1: 用户态应用 (Userland Apps)**
*   **定义**: 存储在 HiveMemory 向量数据库中的 `CODE` 类型记忆原子。即 **"Memory as a Tool"**。
*   **命名空间**: 使用 `mem_` 或 `tool_` 前缀（如 `tool_data_analysis`）。
*   **特性**:
    *   **动态加载**: 需通过 Alias 或 UUID 从数据库检索加载 (Lazy Loading)。
    *   **受限运行**: 必须在隔离沙箱中执行。
    *   **可演进**: Agent 可以通过 `WRITE` 指令创造新工具，或通过 `UPDATE` 修正现有工具。

---

#### 4.2 调度与分发机制 (Dispatching & Routing)

当拦截器捕获到 `RUN` 指令时，**Dispatcher (分发器)** 负责将请求路由至正确的执行单元。

**4.2.1 快速路径分发 (Fast-Path Dispatching)**
Dispatcher 维护一个内存级的 `KERNEL_REGISTRY` 字典。
*   **逻辑**:
    1.  检查 `TARGET` 是否存在于 `KERNEL_REGISTRY`。
    2.  **Hit**: 直接调用对应的 Python 函数，参数透传。这是**快速路径**，耗时 < 1ms。
    3.  **Miss**: 进入慢速路径。

**4.2.2 慢速路径与缓存 (Slow-Path & Hot Cache)**
当目标为用户态工具时，系统需执行检索加载。为了优化性能，引入 **LRU Cache**。
*   **逻辑**:
    1.  检查 `USER_TOOL_CACHE` (内存 LRU)。
    2.  **Cache Hit**: 直接获取缓存的可执行代码对象，送入沙箱。
    3.  **Cache Miss**:
        *   向 Qdrant 发起检索 (ByID 或 Alias)。
        *   获取 Memory Atom，校验 `type == "CODE_SNIPPET"`。
        *   提取 `payload.content`。
        *   送入沙箱执行，并将代码缓存入 LRU。

---

#### 4.3 安全沙箱设计 (Security Sandbox Design)

由于用户态工具（L1）可能由 Agent 自主编写或来自不可信的历史记忆，直接 `exec()` 是极度危险的。系统强制实施沙箱隔离。

**4.3.1 沙箱选型策略**
*   **MVP 阶段**: **RestrictedPython / PyPy Sandbox**
    *   *原理*: 在 Python 解释器层面限制 `globals` 和 `builtins`。
    *   *限制*: 禁用 `import os`, `open`, `subprocess` 等危险模块。仅允许纯逻辑计算和数据处理。
*   **Production 阶段**: **Ephemeral Docker Containers (瞬时容器)**
    *   *原理*: 为每次 `RUN` 启动一个轻量级 Docker 容器（或复用热备容器池）。
    *   *优势*: 真正的 OS 级隔离，支持安装 pip 依赖（如 pandas, numpy）。

**4.3.2 权限控制 (Permission Control)**
*   **网络隔离**: 默认禁止 L1 工具访问外网。如需联网，必须通过 HMP 调用 `sys_web_search`（内核代理由）。
*   **文件系统**:
    *   L1 工具仅能访问挂载的 `/workspace` 临时目录。
    *   禁止访问 `/etc`, `/proc` 或 Patchouli 的核心代码库。

**4.3.3 资源配额 (Resource Quotas)**
*   **超时熔断**: 单个 `RUN` 指令执行时间不得超过 30秒。
*   **内存限制**: 单个沙箱最大内存 512MB。
*   **循环检测**: 防止 Agent 写出 `while True:` 导致资源耗尽。

---

#### 4.4 错误处理与反馈 (Error Handling)

内核不仅负责执行，还负责将执行结果“翻译”给 Agent。

*   **执行成功**:
    *   返回 `status="success"`。
    *   内容: 工具的 `STDOUT` 输出或函数 `return` 值。
*   **运行时错误 (Runtime Error)**:
    *   返回 `status="error"`。
    *   内容: 捕获 Python Traceback 的最后 3 行（避免过长），并附上自然语言提示："Tool execution failed. Please check your arguments or logic."
*   **未找到工具 (404 Not Found)**:
    *   返回 `status="error"`。
    *   内容: "Tool alias not found. Did you forget to SEARCH first?"

---

### 5. 代理行为与 Prompt 工程 (Agent Behavior & Prompting)

本章定义如何通过 **System Prompt** 和 **Few-Shot Learning** 技术，构建 Worker Agent 的“使用说明书”，使其能够准确、适时、高效地调用内核能力。

#### 5.1 System Prompt 设计策略 (System Prompt Design)

为了在有限的 Context Window 内实现最高的指令遵循度，我们采用 **“规格说明书 (Spec-based)”** 结合 **“高密度单样本 (One-Shot High-Density)”** 的教学策略。

**5.1.1 核心组成模块**

System Prompt 应包含以下四个强制模块：

1.  **角色定义 (Role Definition)**:
    *   将 Agent 定义为 **"HiveOS Operator"** 而非单纯的 Chatbot。
    *   强调“严谨性”：不依赖内部训练数据回答事实性问题，必须查阅 HiveMemory。

2.  **协议规格定义 (Protocol Specification)**:
    *   摒弃自然语言描述，使用 **类代码 (Pseudo-code)** 定义语法。LLM 对结构化定义的理解能力远强于长文本。
    *   **重点**: 明确 `READ` 的列表语法，暗示并行操作。

3.  **负面约束 (Negative Constraints)**:
    *   **NO JSON**: 明确禁止输出原生 Function Calling 格式。
    *   **NO HALLUCINATION**: 禁止臆造 `mem_` ID。

4.  **高密度演示 (The Dense Demo)**:
    *   使用**一个**涵盖全流程（搜索 -> 并行读取 -> 执行）的对话示例，替代多个简单示例。这既节省 Token，又展示了多指令间的逻辑流转。

**5.1.2 The "Golden Prompt" 模板**

```markdown
### HIVE MEMORY KERNEL CONTEXT ###

You are an intelligent Agent running on HiveOS. You have access to a persistent memory kernel via the Memory Tool Protocol (MTP).

[PROTOCOL RULES]
1. **INTERACTION**: Do NOT use JSON/Function Calling. Use the MTP syntax directly in your text flow.
2. **SYNTAX**: `⟪ VERB | TARGET | ARGS ⟫`
   - Delimiters: `⟪` and `⟫` (Double Angle Brackets).
   - Separator: `|` (Pipe).
3. **VERBS**:
   - `SEARCH`: Find unknown memories. Target=`*`. Arg `query="..."`.
   - `READ`: Fetch content. Target=`[id1, id2]` (Use LIST for batching!).
   - `RUN`: Execute tools. Target=`alias`. Arg `key="val"`.
   - `WRITE`: Save insights. Target=`*`. Arg `content="..."`.

[BEHAVIORAL GUIDELINES]
- **Verify First**: If asked about specific facts/code, SEARCH/READ memory first. Do not guess.
- **Batch Operations**: Always group multiple READ requests into one list `[a, b, c]` to save IO cycles.
- **Inline Flow**: Don't stop to ask for permission. Execute the protocol commands as part of your thought process.

[ONE-SHOT DEMONSTRATION]
User: "Deploy the login module using the standard config."
Assistant: I need to find the login deployment script and the standard config.
⟪ SEARCH | * | query="login deploy script" filter="type:CODE" ⟫
<mtp_response>
[Menu]: 1. mem_deploy_login (Alias) 2. fact_std_config (Alias)
</mtp_response>
Found them. I will read both the script and the config to ensure compatibility.
⟪ READ | [mem_deploy_login, fact_std_config] | ⟫
<mtp_response>
[mem_deploy_login]: def deploy(env): ...
[fact_std_config]: ENV_VARS = {...}
</mtp_response>
The data is loaded. Now executing the deployment.
⟪ RUN | mem_deploy_login | env="prod" config_ref="fact_std_config" ⟫
```

---

#### 5.2 触发时机控制 (Triggering Logic)

为了防止 Agent 滥用协议（例如在闲聊中不断 Search），需要在 Prompt 中植入 **“认知阈值 (Cognitive Thresholds)”**。

**5.2.1 触发原则**

*   **信息缺口 (Information Gap) -> 触发 SEARCH/READ**
    *   判据：当用户询问具体事实（“上周的会议记录”）、技术细节（“utils.py 的代码”）且 Context 中不存在时。
    *   *指令*: "When you lack context to answer truthfully, use `SEARCH`."

*   **能力缺口 (Capability Gap) -> 触发 RUN**
    *   判据：当用户要求产生副作用（“保存文件”、“运行代码”、“联网搜索”）时。
    *   *指令*: "When actions are required (IO, Execution), use `RUN`."

*   **高价值沉淀 (Insight) -> 触发 WRITE**
    *   判据：当生成了高质量代码、解决方案或达成了共识时。
    *   *指令*: "If the generated content is valuable for future reuse, use `WRITE` to save it."

**5.2.2 角色微调 (Role Tuning)**
根据 Agent 的类型，调整 System Prompt 的首句来控制触发倾向：
*   **Coder Agent**: "你是严谨的工程师。**必须**先查阅已有代码库再编写新代码。"（激进查阅）
*   **Chat Agent**: "你是得力的助手。仅在必要时查阅记忆。"（保守查阅）

---

#### 5.3 错误处理与自我纠正 (Error Handling & Self-Correction)

Agent 可能会犯错（如幻觉 ID、语法错误）。我们需要利用 LLM 的 **In-Context Learning** 能力，让其通过内核反馈进行自我纠正。

**5.3.1 常见错误与反馈回路**

| 错误类型 | 内核反馈 (Injected via `<mtp_response>`) | 期望的 Agent 修正行为 |
| :--- | :--- | :--- |
| **ID 幻觉** (调用了不存在的 `mem_xyz`) | `Error: Alias 'mem_xyz' not found in context. Did you mean to use SEARCH?` | Agent 停止瞎猜，转而发出 `⟪ SEARCH... ⟫` 指令。 |
| **语法错误** (忘记闭合 `⟫`) | `Error: Protocol syntax error. Expected '⟫'.` | Agent 重新输出正确的完整指令。 |
| **参数错误** (Run 缺少必填参) | `Error: Missing argument 'path' for tool 'sys_read'.` | Agent 检查上文逻辑，补全参数后再次 `RUN`。 |

**5.3.2 纠偏 Prompt**
在 System Prompt 底部增加一行兜底指令：
> "If you receive an `<mtp_response status="error">`, analyze the error message and retry with the corrected command immediately."

---

### 6. 附录：示例与用例 (Appendix: Examples)

本章提供 MTP 在不同业务场景下的标准交互日志。
**图例说明**:
*   `[User]`: 用户输入。
*   `[Agent]`: Worker Agent 生成的文本（在 `⟫` 处会被 API 截断）。
*   `[Kernel]`: Patchouli 内核执行后回填的 System 注入内容。
*   `[Resume]`: 拼接历史后，Agent 继续生成的文本。

---

#### Use Case 1: 启动自省与环境恢复 (Startup & Introspection)
**场景描述**: 用户希望继续上次未完成的工作，但 Agent 的 Context Window 已被重置。Agent 需要通过 `SEARCH` 找回上下文。

```text
[User]
我们继续上次关于 "Prometheus" 项目的讨论，我要加个新监控项。

[Agent]
(Thinking: Context is empty. I need to find what "Prometheus project" refers to.)
收到。为了确保上下文准确，我需要先检索一下我们之前的项目记录。
⟪ SEARCH | * | query="Prometheus project status" filter="type:FACT" ⟫

[Kernel] (Interruption -> Execution -> Injection)
<mtp_response>
[Menu]:
1. fact_prometheus_env (Alias) - "Prometheus config path and server IP."
2. fact_monitor_list (Alias) - "List of current active monitors."
</mtp_response>

[Resume] (Agent continues)
找到了项目环境配置和当前的监控列表。我将读取它们以获取详细信息。
⟪ READ | [fact_prometheus_env, fact_monitor_list] | ⟫

[Kernel]
<mtp_response>
[fact_prometheus_env]: Server: 10.0.0.5, Config: /etc/prometheus/prometheus.yml
[fact_monitor_list]: - CPU Usage - RAM Usage - Disk I/O
</mtp_response>

[Resume]
环境已确认。服务器 IP 为 10.0.0.5。您想添加什么类型的监控项？
```

---

#### Use Case 2: 记忆即工具 —— 动态加载与执行 (Memory as a Tool)
**场景描述**: 用户要求部署服务。Agent 发现记忆库中已经存储了一个经过验证的部署脚本（Memory Atom），于是直接作为工具调用它。

```text
[User]
把当前的修改部署到测试环境。

[Agent]
好的。我检索一下是否有现成的部署脚本。
⟪ SEARCH | * | query="deploy script test env" filter="type:CODE" ⟫

[Kernel]
<mtp_response>
[Menu]:
1. tool_deploy_test (Alias) - "Python script to deploy to staging via SSH."
   > Args: target_host (str), restart (bool)
</mtp_response>

[Resume]
发现 `tool_deploy_test` 工具。我将直接执行它进行部署。
⟪ RUN | tool_deploy_test | target_host="192.168.1.100" restart=True ⟫

[Kernel] (Sandbox Execution)
<mtp_response status="success">
[STDOUT]:
Connecting to 192.168.1.100...
Uploading files...
Restarting service...
Deployment Successful. Service is UP.
</mtp_response>

[Resume]
部署已成功完成，服务已重启。需要我检查日志吗？
```

---

#### Use Case 3: 混合流与主动记忆 (Mixed Flow & Active Write)
**场景描述**: 这是一个复杂的 **"Search -> Read -> Write"** 链条。Agent 遇到报错，查阅文档解决后，主动将这个解决方案保存为新的记忆，以便未来复用。

```text
[User]
运行的时候报错了：`Error 503: Service Unavailable`。

[Agent]
这可能是负载均衡配置的问题。我查一下相关文档。
⟪ SEARCH | * | query="Error 503 load balancer" ⟫

[Kernel]
<mtp_response>
[Menu]: 1. doc_nginx_config (Alias)
</mtp_response>

[Resume]
我读取一下 Nginx 配置。
⟪ READ | [doc_nginx_config] | ⟫

[Kernel]
<mtp_response>
[doc_nginx_config]: ... upstream backend { server 127.0.0.1:8080; } ...
</mtp_response>

[Resume]
(Thinking: The port is wrong. Should be 9090.)
发现问题了，端口配置错误。我将使用系统工具修正文件。
⟪ RUN | sys_write_file | path="/etc/nginx.conf" content=`...server 127.0.0.1:9090;...` ⟫

[Kernel]
<mtp_response status="success">File updated.</mtp_response>

[Resume]
配置已修复。这个问题很典型，我将把这个解决方案（端口 8080 -> 9090）记录到记忆库中，防止下次再犯。
⟪ WRITE | * | title="Fix for Error 503" content="The upstream port must be 9090, not 8080." ⟫

[Kernel] (Async Signal)
<mtp_response status="ack">Memory saved as 'fact_fix_error_503'.</mtp_response>

[Resume]
已记录该修复方案。您现在可以尝试重启服务了。
```

---

# 7. 系统重构与运行时架构 (System Refactoring & Runtime Architecture)

为了支撑引入的 **MTP (Memory Tool Protocol)**，系统必须从早期的“单向线性处理”模式，转型为支持“多轮递归中断”的 **AIOS (AI Operating System)** 架构。本章定义了 PatchouliSystem v3.0 的核心拓扑结构。

## 7.1 架构演进理念 (Architectural Philosophy)

我们将 HiveMemory 的运行模式重定义为 **“帕秋莉 OS (Patchouli OS)”** 模型：

*   **从“流水线”到“事件循环”**：系统不再只是处理一次性的 Request/Response，而是维护一个持续的 **Kernel Loop**，负责在 Worker Agent 生成过程中处理多次 MTP 中断与恢复。
*   **从“三位一体”到“一核三使”**：
    *   原有的分身将作为独立的微服务（Microservices）。
    *   引入 **Patchouli Kernel** 作为总线与调度器。
    *   引入 **Koakuma (小恶魔)** 作为专职的 MTP 处理器。

## 7.2 系统拓扑图 (System Topology)

v3.0 架构采用了星形拓扑，以 Kernel 为中心，连接网关、微服务与数据层。

```mermaid
graph TD
    UserClient[用户 / Worker Agent] <--> API_Interface
    
    subgraph "PatchouliSystem (The Facility)"
        API_Interface <--> TheEye[The Eye / 真理之眼 \n(Ingress Gateway)]
        
        TheEye <--> Kernel[Patchouli Kernel \n(State & Scheduler)]
        
        subgraph "Microservices Layer (The Staff)"
            Kernel <--> Retrieval[Retrieval Familiar \n(Read-Only Service)]
            Kernel <--> Koakuma[Koakuma / 小恶魔 \n(MTP Runtime Service)]
            Kernel -.->|Async Log| Librarian[Librarian Core \n(Write/Manage Service)]
        end
        
        subgraph "Data & Runtime Layer"
            Retrieval <--> Qdrant[(Vector DB)]
            Librarian <--> Qdrant
            Koakuma <--> Sandbox[Docker/RestrictedEnv]
            Librarian <--> SqlDB[(Meta DB)]
        end
    end
```

## 7.3 核心组件定义 (Core Component Definitions)

### 7.3.1 守门人：真理之眼 (The Eye / Gateway)
*   **定位**：系统的 **Ingress Controller**，独立于 Kernel 之外。
*   **职责**：
    *   **流量清洗**：拦截无效请求、系统指令（如 `/clear`）。
    *   **意图识别**：L1拦截层负责，判断请求是 `CHAT`（直接转发）还是 `WORK`（需 Kernel 介入）。
    *   **查询重写**：L2语义分析层负责，将用户的模糊 Query 转化为精准的 Semantic Query。
*   **交互**：作为 Kernel 的 Client，向 Kernel 发送标准化的 `JobRequest`。

### 7.3.2 操作系统：帕秋莉内核 (Patchouli Kernel)
*   **定位**：系统的 **Orchestrator (编排器)** 与 **State Manager (状态管理器)**。
*   **职责**：
    *   **Session State**：维护对话历史、上下文缓存 (Prompt Caching) 和临时变量。
    *   **LLM IO**：持有 Worker Agent 的 API Client，负责发送请求并处理 `stop=["⟫"]` 中断信号。
    *   **调度总线**：根据当前状态，决定调用哪个微服务（Retrieval, Koakuma, 或 Librarian）。

### 7.3.3 执行层：三大微服务 (The Triad Services)

1.  **Retrieval Familiar (检索使魔)**
    *   **功能**：只读服务。负责 RAG 检索、混合排序、上下文渲染。
    *   **服务对象**：既服务于 Kernel（开场注入），也通过内部 API 服务于 Koakuma（处理 `READ` 指令）。

2.  **Koakuma (小恶魔 / MTP Executor)**
    *   **[新增组件]**
    *   **功能**：无状态计算服务。负责 MTP 协议的 **解析 (Parsing)**、**路由 (Routing)** 和 **执行 (Execution)**。
    *   **能力**：持有沙箱环境的控制权，负责运行 `sys_` 工具和 `mem_` 代码。

3.  **Librarian Core (大图书馆本体)**
    *   **功能**：写入与管理服务。负责记忆生成、去重、演化和 GC。
    *   **特性**：**纯异步 (Async)**。从 Kernel 接收对话日志副本，不阻塞前台响应。

## 7.4 运行时数据流：内核递归循环 (The Kernel Recursive Loop)

引入 MTP 后，一次用户请求的处理流程变为一个递归循环：

1.  **初始化 (Initialization)**
    *   **Eye** 处理请求，转发给 **Kernel**。
    *   **Kernel** 呼叫 **Retrieval Familiar**，获取 Index Menu，组装初始 System Prompt，解决 Worker Agent 的冷启动问题，确保其面对新 Query 时有合适的 Memory 背景。

2.  **生成循环 (The Generation Loop)**
    *   **Phase A (Request)**: Kernel 向 LLM 发起生成请求（带 `stop` 参数）。
    *   **Phase B (Decision)**:
        *   *分支 1 (Finished)*: LLM 正常结束生成 -> Kernel 将结果返回给用户 -> 异步投递日志给 **Librarian**。
        *   *分支 2 (Interrupted)*: 捕获到 MTP 信号 -> Kernel 暂停，提取指令 Buffer。
    *   **Phase C (Execution)**: Kernel 将 Buffer 发送给 **Koakuma**。
        *   **Koakuma** 解析指令，执行工具/检索，返回 XML 格式结果。
    *   **Phase D (Resume)**: Kernel 将 XML 结果追加到 History，**跳转回 Phase A**（发起新一轮续写）。

## 7.5 容错与服务降级 (Fault Tolerance)

本架构实现了“能力分层”，确保核心功能的可用性。即便任意一个分身离线，其余分身的核心功能也不会受到严重影响或降级处理：

*   **Koakuma 离线**：
    *   *现象*：MTP 指令执行超时或失败。
    *   *降级*：Kernel 向 Worker Agent 注入 `<mtp_response status="error">System Offline</mtp_response>`。
    *   *后果*：Agent 失去“手”（执行能力），退化为普通 Chatbot，但依然拥有“脑”（记忆能力）。
*   **Retrieval 离线**：
    *   *现象*：开场检索失败。
    *   *降级*：Kernel 注入空 Context 启动对话。
    *   *后果*：Agent 暂时失忆，但仍能对话。
*   **Librarian 离线**：
    *   *现象*：日志投递失败。
    *   *策略*：Kernel 将日志暂存至 Redis 队列，等待重试。
    *   *后果*：用户无感知，仅记忆生成延迟。
