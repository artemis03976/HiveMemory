# HiveMemory 系统开发路线图

**文档状态**: Living Roadmap  
**更新时间**: 2026-07-02  
**版本口径**: `v0.x.0` 表示一个可独立验收的系统能力阶段，不等同于所有内部设计文档的 phase 编号。

---

## 1. 路线图定位

HiveMemory 的演进不是从一个普通聊天应用逐步叠功能，而是围绕“记忆系统作为 Agent Runtime 的底层操作系统”展开。

当前路线图按真实演进顺序重新整理：

| 版本 | 阶段主题 | 状态 | 核心价值 |
| :--- | :--- | :--- | :--- |
| v0.1.0 | 三级记忆系统与前端 MVP | 已完成 | 建立 Patchouli 记忆闭环、生命周期管理、API 与可视化前端 |
| v0.2.0 | 多 Agent 隔离与 Agent Profile 虚拟记忆原子 | 已完成 | 将 Agent 身份配置化、记忆化，并支持隔离的多 Agent 运行 |
| v0.3.0 | 多 Agent CALL、PendingAtom、Alice Orchestrator、MemoryCompiler | 已完成 | 打通 Agent 间调用、临时记忆结算、编排边界与记忆编译入口 |
| v0.4.0 | Runtime Control 与系统事件观测 | 已完成 / 稳定化 | 将 chat run 与 memory task 建模为可取消、可观测、可审计的运行时对象 |
| v0.5.0 | Data Durability & Artifact 体系 | 已完成 | 将原始交互与记忆生成 provenance 固化为可追溯的冷资产；建立 artifact 数据底座 |
| v0.5.1 | 基础设施清理 | 已完成 | Config 重分层（Patchouli/Alice 独立 config）、NoOp 组件补全、cancel_event 传入 MTP executor |
| v0.5.2 | Async-Native Adaptation | 已完成 | 存储层切换 AsyncQdrantClient，generation/retrieval 全链路原生 async，并补齐 lifecycle / runtime health 的异步调用 |
| v0.5.3 | Patchouli 架构重构 | 已完成 | LibrarianCore 解构，建立 MemoryLibrary + 业务能力拆分，扩展 RetrievalFamiliar 为全层检索 |
| v0.6.0 | System Gateway & Commands | 下一阶段 | Gateway 上移、系统指令、复合意图拆分、Gateway 降级与自定义拦截规则 |
| v0.6.1 | Runtime Job Queue | 候选规划 | Agent 任务队列、定时任务、hook 触发任务、任务 outcome artifact |
| v0.6.2 | Chat Attachments | 候选规划 | 对话页面附件上传、解析、artifact 化与对话上下文增强 |
| v0.6.3 | Frontend Experience Quick Wins | 候选规划 | 浅色主题、自定义背景上传、对话阻塞动态标语 |
| v0.7.0 | Document Ingestion | 候选规划 | 外源长文档解析为复数记忆资产，支持来源追溯与人工确认 |
| v0.7.1 | MTP READ Provenance | 候选规划 | 优先推进历史、provenance、artifact 来源读取；MTP RUN 沙箱后移 |
| v0.7.2 | Deep Research MVP | 候选规划 | 基于记忆库的多步搜索、证据沉淀与研究报告 artifact |
| v0.7.3 | Conversation Branching MVP | 可选规划 | 对话气泡编辑与安全分叉重放；完整记忆回档后移 |
| v0.8.0+ | Alice Phase 3 与高级编排 | 后置规划 | 完整任务图、复杂协作策略、review loop 与跨子系统 orchestration |
| v0.9.0+ | 高级生命周期与可逆记忆 | 后置规划 | 记忆 split/merge、完整 rollback、L3 冷存储复活、MTP RUN 沙箱 |

---

## 2. 已完成阶段

### v0.1.0: 三级记忆系统与前端 MVP

> 原路线图中的“基础设施准备、MVP Ingestion、Retrieval & Injection、Lifecycle & Evolution、Frontend & API”统一归档为 v0.1.0。

v0.1.0 的目标是让系统从零形成可用的记忆闭环：

```text
用户对话
  -> 记忆提取
  -> MemoryAtom 入库
  -> 检索
  -> 上下文注入
  -> 前端可视化与人工管理
```

核心交付：

1. **基础设施与核心模型**
   - 建立 Python / FastAPI / Qdrant / LLM provider 的基础运行环境。
   - 定义 `MemoryAtom` 的冰山模型：`meta`、`index`、`payload`、`artifacts`。
   - 跑通结构化对象写入与读取。

2. **Patchouli 记忆写入原型**
   - 建立对话日志到结构化记忆的 ingestion 链路。
   - 实现早期 Librarian / generation chain。
   - 支持从对话中抽取可落库的长期记忆。

3. **Retrieval 与上下文注入**
   - 建立基础向量检索与元数据过滤。
   - 实现 memory context renderer。
   - 将检索结果注入 Worker Agent prompt，形成对话记忆闭环。

4. **Lifecycle 与演化**
   - 引入 search-before-write。
   - 支持 insert / update / merge 的初步判断。
   - 建立版本历史、活跃度、归档等生命周期概念。

5. **API 与前端**
   - 使用 FastAPI 暴露 chat / memories 等核心接口。
   - 建立前端聊天、记忆列表、记忆可视化与人工 CRUD。
   - 让系统从 CLI 原型进入可交互产品雏形。

v0.1.0 的意义：HiveMemory 第一次具备完整 MVP 形态，Patchouli 从“记录员”进入“可检索、可维护的记忆管理员”阶段。

### v0.2.0: 多 Agent 隔离与 Agent Profile 虚拟记忆原子

> 对应多智能体子系统 Phase 1：Agent Profile & Runtime。

v0.2.0 的目标是让系统不再绑定单一 Worker Agent，而是支持隔离的多 Agent 身份。

核心交付：

1. **Agent Profile 虚拟记忆原子**
   - 将 Agent 配置建模为特殊 `MemoryAtom`。
   - 使用 `AGENT_PROFILE` 类记忆保存 persona、system prompt、工具权限、默认行为。
   - 复用现有存储、检索、版本演化能力管理 Agent Profile。

2. **Agent 身份与 Topic 解耦**
   - Topic / conversation 不再天然绑定单一 Agent。
   - 同一话题中可以切换 active agent。
   - 记忆写入和检索带上 agent identity 与 visibility。

3. **权限沙箱**
   - 根据 Agent Profile 动态拼装 MTP 工具菜单。
   - 在运行时执行层拦截越权工具调用。
   - 为后续 Agent 间调用和 Alice 编排提供安全边界。

v0.2.0 的意义：HiveMemory 开始从单 Agent 记忆系统转向多 Agent runtime，Agent 本身也成为可存储、可检索、可演化的记忆对象。

### v0.3.0: 多 Agent CALL、PendingAtom、Alice Orchestrator、MemoryCompiler

> 对应多智能体子系统 Phase 2 及随后围绕 PendingAtom、编排边界、MemoryCompiler 的系统整理。

v0.3.0 的目标是让 Agent 不只是并列存在，而是可以通过系统协议主动协作。

核心交付：

1. **MTP CALL 与子 Agent 运行帧**
   - 扩展 MTP，引入 `CALL` 指令。
   - 主 Agent 可以通过 alias 唤起子 Agent。
   - 子 Agent 在独立 `ExecutionFrame` 中运行，避免污染主话题上下文。
   - 子 Agent 结果通过 IPC 回填给主 Agent。

2. **Agent Runtime 边界整理**
   - 将单 Agent 执行循环收敛到 `agent_runtime/`。
   - 明确 `AgentLoopExecutor` 的 generate -> MTP -> return/suspend 职责。
   - 将多 Agent 相关编排职责从执行循环中拆出。

3. **Alice Orchestrator 初步落地**
   - `AgentOrchestrator` 承担 CALL、子 Agent frame、IPC 结果组装。
   - Alice 从概念调度器进入实际 runtime 边界。
   - 为后续顶层策略编排保留入口。

4. **PendingAtom 与 materialize task**
   - WRITE / UPDATE 不再必须同步落库为正式记忆。
   - 引入 `PendingAtom` 作为运行时临时记忆句柄。
   - Agent run 结束后收集 pending materialize tasks。
   - Patchouli finalize 触发主动记忆生成并结算 pending atom。

5. **MemoryCompiler 收敛**
   - 收敛 retrieval、MTP READ、shared context 等记忆渲染入口。
   - 建立 target-first handler 和 envelope 分层。
   - 为未来 Memory IR、MTP READ 历史编译、MTP RUN executable asset 做准备。

v0.3.0 的意义：系统从“可切换多 Agent”进入“可协作多 Agent”，并建立了运行时临时记忆与长期记忆之间的结算边界。

### v0.4.0: Runtime Control 与系统事件观测

> 详细规划见 [V0.4.0RuntimeControlAndObservabilityPlan.md](mod/V0.4.0RuntimeControlAndObservabilityPlan.md)。

v0.4.0 不继续扩展复杂智能行为，而是补齐运行时控制面。

核心交付：

1. **完整取消语义**
   - 前端 Stop、SSE abort、浏览器断连、后端 `/chat/stop` 收敛到统一取消控制面。
   - 取消信号穿透 ChatApplicationService、Alice Orchestrator、AgentLoopExecutor、WorkerAgent、sub-agent frame、MTP 执行边界、Patchouli finalize。
   - 被取消的 run 不继续生成误导性最终回答，也不默认触发长期记忆生成。

2. **主动记忆生成独立运行时**
   - `run_active_generation()` 从 finalize 的直接子过程升级为可查询、可取消、可观测的 memory generation task。
   - 前端增加后台记忆生成状态窗口。
   - memory task 可单独取消，不影响已经完成的对话回答。

3. **系统事件观测流 v1**
   - 从 logging namespace 捕获升级为 logging + domain event 双通道。
   - 建立 `RuntimeEvent` / `RuntimeEventBus` / `/runtime-events/stream`。
   - 修复生产环境下观测面板无法稳定显示的问题。

4. **运行时契约测试**
   - 覆盖 cancel、memory task、event stream 的单元测试与集成测试。
   - 建立 v0.4.0 的验收清单。

v0.4.0 的意义：一次用户交互及其派生的记忆生成任务，被建模为可取消、可观测、可审计、可清理的 runtime run/task。

---

## 3. 当前规划阶段

### v0.5.0: Data Durability 与 Async Cold Path

> 详细规划见 [V0.5.0DataDurabilityAndAsyncColdPathPlan.md](mod/V0.5.0DataDurabilityAndAsyncColdPathPlan.md)。

v0.5.0 的目标不是增加新的智能行为，而是补齐数据耐久性与异步冷路径契约。

核心问题：

> 系统生成的每条长期记忆，是否能稳定追溯到它来自哪一次原始交互、哪一批 runtime facts、哪一次 generation task、哪一组检索证据，以及当时采用了怎样的生成视图？

核心交付：

1. **ArtifactStore v1**
   - 新增轻量本地 artifact store 抽象。
   - 支持 JSON artifact 写入、读取、hash 校验和按 id 查询。
   - 后续可替换为数据库或对象存储。

2. **原始交互冷保存**
   - 每次完成的 chat run 保存 `RawInteractionArtifact`。
   - cancelled / failed run 可配置保存 partial artifact。
   - artifact 绑定 `generation_id`、`trace_id`、`topic_id`、`agent_id`、`user_id`、`memory_task_ids`。

3. **双视图 transcript artifact**
   - 保存 raw facts：`InteractionPayload`、`turn_events`、`mtp_traces`、retrieval refs。
   - 保存 history view：可重放给 Agent 的消息视图。
   - 保存 generation view：用于记忆生成的 `GenerationContext` 与渲染文本。

4. **MemoryAtom provenance**
   - 新建、更新、合并、触碰、丢弃的 generation result 均能指向来源 artifact。
   - `MemoryAtom.payload.artifacts` 只保存稳定 refs，不直接塞大块 raw transcript。
   - `full_history` 的每次版本变更附带来源 artifact、task、run。

5. **Memory task outcome**
   - memory task 运行结果落 artifact。
   - 进程内 task 句柄之外，保留可审计的完成记录。

6. **Retrieval evidence artifact**
   - 检索请求具备 request id、timeout、cancel、failure policy。
   - 关键检索证据、渲染上下文和 memory refs 可被保存。
   - 长期记忆可追溯到当时使用过的检索证据。

7. **Async cold path runner**
   - generation task 和 retrieval request 收敛到统一异步任务契约。
   - 支持状态、超时、取消、重试、事件发布和 artifact output。
   - 减少“同步函数外包一层 `to_thread`”的临时形态。

v0.5.0 的意义：把“对话事实”和“记忆事实”从运行时缓存提升为可持久化、可引用、可审计、可异步重放的数据资产。

---

## 4. 后续版本重排

v0.5.x 的基础设施工作完成后，后续版本不再只按原先的 v0.7/v0.8/v0.9 大块推进，而是按“能力链”拆成更短的可验收阶段：

```text
System Gateway
  -> Runtime Job Queue
  -> Attachments / Document Artifacts
  -> Document Ingestion
  -> MTP READ Provenance
  -> Deep Research
  -> Conversation Branching / Advanced Lifecycle
```

这样安排的目标是：在剩余全职开发窗口内，优先交付对产品能力最有放大作用、且能复用现有 artifact / runtime / memory library 基建的功能；将 MTP RUN 沙箱、完整 Alice Phase 3、完整记忆回档这类高风险能力后移。

### v0.6.0: System Gateway 与系统指令

v0.6.0 的规划大体保持不变，不额外塞入附件、研究、长文档入库等新功能。它的重点是把 Gateway 变成系统级入口，并为后续任务队列和 Deep Research 预留统一命令与 task spec 出口。

目标：

- 将 Gateway / TheEye 正式上移到 System 级。
- 支持 `/clear` 等系统指令的统一解析与执行。
- 增加复合意图识别与任务拆分。
- 让 Gateway 的拆分结果能被 Patchouli、Alice 与下游 task runner 消费。
- **完善 Gateway 自定义拦截规则**：重新引入 `RuleInterceptor` 的自定义规则注入机制（`custom_system_patterns` / `custom_chat_patterns`），与系统指令注册表打通，支持运行时动态扩展拦截规则。
- **Gateway LLM 降级与重试**：为 `LLMAnalyzer` 增加重试策略（指数退避）、超时控制与降级路径，LLM 调用失败时自动回退到 `NoOpSemanticAnalyzer` 的保守结果，而非将异常上抛给 TheEye。
- 为后续 `/ingest`、`/research`、定时任务、hook 任务预留命令注册与 task dispatch 边界，但不在 v0.6.0 内完成这些业务能力。

依赖：

- v0.4.0 的 runtime run/task 控制。
- v0.5.0 的 raw interaction artifact 与 task provenance。
- v0.5.x 的 Patchouli / async-native / storage 基建。

验收口径：

- Gateway 不再只是 chat 前置过滤器，而是 System 级意图入口。
- 系统指令、普通聊天、复合意图拆分有清晰的 contract。
- LLM analyzer 失败不会破坏 chat 主流程。

### v0.6.1: Runtime Job Queue、定时任务与 Hook 触发

这部分建议提前，因为它是长文档解析、Deep Research、后台 Agent 工作流的共同底座。没有统一任务队列，后续功能容易退化成各自维护一套后台状态。

目标：

- 建立统一 `Job` / `Task` 抽象，覆盖即时任务、定时任务和 hook-triggered 任务。
- 支持任务列表、状态查询、取消、重试、超时、失败原因与 outcome artifact。
- 复用 v0.4.0 runtime events 与 v0.5.0 artifact provenance。
- 提供最小前端任务列表，用于观察后台任务与失败结果。
- 为 document ingestion、Deep Research、定时记忆维护任务保留任务类型扩展点。

建议边界：

- 先做单机内置队列或轻量持久化队列，不急于引入分布式任务系统。
- 任务图编排暂不做，只保证单任务生命周期稳定。

### v0.6.2: 对话页面附件上传与解析

附件能力优先服务对话体验与 artifact 数据底座，不应一开始就自动写入长期记忆。更稳妥的路径是：上传文件先成为 artifact，再由 chat context 或后续 ingestion job 消费。

目标：

- 支持对话页面上传附件，并将原始文件保存为 artifact。
- 解析附件文本，生成 parsed text artifact 与 parse metadata。
- 在当前对话中将附件作为上下文增强输入。
- 前端展示附件 chip、上传状态、解析状态与解析失败原因。
- 优先支持 `txt` / `md` / `pdf`，随后补 `docx` 等格式。

建议边界：

- 不在本阶段自动拆分为长期记忆。
- 不在本阶段实现复杂文件库管理。
- 大文件解析交给 v0.6.1 job queue，避免阻塞 chat stream。

### v0.6.3: 前端体验快赢

这部分实现风险低、用户感知强，适合穿插在 v0.6.1-v0.6.2 期间完成。

目标：

- 支持浅色主题。
- 支持自定义背景上传与本地持久化配置。
- 对话阻塞、等待模型、等待工具调用时显示动态闪烁标语。

建议边界：

- 主题系统先覆盖主要 chat / memory / settings 界面，不追求一次性完成所有边缘组件。
- 自定义背景应有可读性保护，例如遮罩、模糊、透明度调节。
- 阻塞标语只表达运行状态，不承诺不可观测的内部推理内容。

### v0.7.0: 外源长文档解析为记忆资产

长文档解析建议独立于对话页面，建模为 `Document Ingestion Job`。它应消费 document artifact，输出候选记忆资产，并保留来源段落、页码或 chunk 引用。

目标：

- 支持从上传文档 artifact 创建 document ingestion job。
- 将长文档解析为 chunk，并保存 chunk artifact / document parse artifact。
- 基于 chunk 生成复数候选记忆资产。
- 候选记忆先进入 PendingAtom 或 reviewable draft，不默认直接污染正式记忆库。
- 每条候选记忆都能追溯到来源文档、chunk、解析任务与生成任务。

依赖：

- v0.6.1 的 job queue。
- v0.6.2 的文件 artifact 与解析能力。
- v0.5.0 的 artifact / provenance。
- v0.5.3 的 MemoryLibrary / ingestion pipeline。

### v0.7.1: MTP READ Provenance 与历史读取

原 v0.7.0 中的 MTP READ 和 MTP RUN 应拆开。MTP READ 的收益更高、风险更低，应提前；MTP RUN executable asset、Docker 沙箱与真取消应后移。

目标：

- MTP READ 从简单读取 payload 升级为读取版本历史、provenance、artifact 来源。
- 支持 Agent 查询某条记忆来自哪些 raw interaction、document chunk、retrieval evidence 或 generation task。
- 支持将 provenance 编译为可读上下文，供 Deep Research、审计与人工确认使用。
- 补齐 READ 结果中的权限边界，避免越权读取不可见来源。

后移内容：

- MTP RUN 从记忆原子编译 executable asset。
- Docker 或等效隔离沙箱。
- MTP RUN 真取消与进程级 kill。

### v0.7.2: Deep Research MVP

Deep Research 是近期最有产品放大作用的高级能力，但应建立在 job queue、文档入库、MTP READ provenance 之上。第一版不需要完整 Alice Phase 3，只需要可观测、可取消、可沉淀的单任务循环。

目标：

- 支持通过 Gateway 命令或前端入口创建 research job。
- 研究任务执行 `plan -> memory retrieval -> evidence artifact -> interim notes / pending memories -> next query -> report artifact` 循环。
- 每轮检索都保存 retrieval evidence artifact。
- 中间结论可沉淀为 pending memories 或 research notes。
- 最终输出 task report artifact，并引用 memory refs / document refs / retrieval evidence refs。
- 前端展示研究任务状态、阶段进度、关键引用与最终报告。

建议边界：

- 不在 MVP 中实现复杂多 Agent 并行协作。
- 不在 MVP 中实现自动网络深搜，除非已有稳定外部搜索工具和权限模型。
- 不自动把所有中间结论写入正式长期记忆，默认进入可审核状态。

### v0.7.3 可选: 对话编辑与安全分叉重放

对话气泡编辑与任意距离回档会直接触碰 memory provenance 和可逆性。短期建议先做“分叉/重放”，不要承诺完整记忆撤销。

目标：

- 支持编辑某条用户消息，并从该 turn 创建新的对话分支继续运行。
- 旧分支后续消息标记为 superseded，而不是物理删除。
- 新分支生成独立 raw interaction artifact 与 generation provenance。
- 若旧分支曾产生长期记忆，前端提示这些记忆来自已废弃分支，并提供人工归档/撤销入口。

暂不做：

- 自动回滚已经写入的长期记忆。
- 自动拆解由旧分支参与 merge / update 的复杂记忆历史。
- 任意 memory graph 级别的反向撤销。

### v0.8.0+ 后置: Alice Phase 3 顶层编排

目标：

- Alice 正式成为顶层 orchestrator。
- 管理不同编排策略：single-agent、parallel specialists、plan-and-execute、review loop 等。
- 将复杂任务拆成可观测、可取消、可追踪的任务图。
- 统一控制 Patchouli、AgentRuntime、Gateway 与 MemoryCompiler 的协作。

后置原因：

- 该阶段依赖稳定的 Gateway task spec、job queue、task outcome artifact 和若干真实后台任务类型。
- 在 Deep Research MVP 之前做完整编排，容易先完成框架而缺少可验证工作负载。

### v0.9.0+ 后置: 高级生命周期、完整回档与 MTP RUN 沙箱

目标：

- 初步实现记忆 split / merge 行为。
- 支持将混杂记忆拆成多个可追溯记忆。
- 支持多条相近记忆合并，同时保留来源与版本历史。
- 建立完整 memory rollback / branch invalidation / provenance reverse index。
- MTP RUN 从执行 `payload.content` 升级为从记忆原子编译 executable asset。
- 引入隔离沙箱执行环境、资源限制、权限快照、运行前检查与执行结果 artifact。
- 基于隔离沙箱能力实现 MTP RUN 真取消。

后置原因：

- split/merge 与完整 rollback 都依赖稳定 provenance、历史读取和反向索引。
- MTP RUN 沙箱对安全性、资源限制、取消语义和部署环境要求较高，不适合挤占近期产品能力窗口。

---

## 5. 长期方向

HiveMemory 的长期目标可以分为四条主线。

1. **记忆系统主线**
   - 从 MemoryAtom CRUD 走向可审计、可回放、可演化的记忆图谱。
   - 强化版本历史、provenance、artifact、split/merge、权限与可解释性。
   - 支持 document memory、research memory、conversation branch memory 等更多记忆生成来源。

2. **运行时主线**
   - 从单次 chat stream 走向 run/task/job graph。
   - 强化 cancel、timeout、retry、event、task outcome、backpressure。
   - 支持定时任务、hook 任务、后台 Agent job 与可观察的长任务生命周期。

3. **多智能体主线**
   - 从 Agent Profile 隔离走向 Alice 顶层编排。
   - 支持多策略、多角色、多阶段、可回溯的协作流程。
   - 让 Deep Research 流程、长文档入库流程成为 Alice 编排的真实工作负载。

4. **编译系统主线**
   - 从 prompt renderer 走向 MemoryCompiler IR。
   - 支持 READ、RUN、embedding、shared context、agent profile、document chunk、research report、executable asset 等多目标编译。

---

## 6. 当前优先级

当前最高优先级是推进 v0.6.0，并为后续两个月左右的全职开发窗口保留清晰的取舍。

近期优先级：

1. 完成 v0.6.0 System Gateway 与系统指令，不扩大 scope。
2. 完成 v0.6.1 Runtime Job Queue，统一后台任务生命周期。
3. 完成 v0.6.2 对话附件上传与解析，所有文件先 artifact 化。
4. 穿插完成 v0.6.3 前端体验快赢：浅色主题、自定义背景、阻塞动态标语。
5. 完成 v0.7.0 Document Ingestion，将外源长文档解析为可审核记忆资产。
6. 完成 v0.7.1 MTP READ provenance，为 Deep Research 和回档语义打底。
7. 冲刺 v0.7.2 Deep Research MVP，形成可观测、可取消、可追溯的研究任务与报告 artifact。
8. 时间允许时完成 v0.7.3 对话编辑与安全分叉重放 MVP。

明确后移：

1. MTP RUN executable asset、Docker 沙箱与真取消。
2. 完整 Alice Phase 3 任务图编排。
3. 完整任意距离记忆回档。
4. 高级 split / merge 生命周期维护。
5. L3 冷存储复活与复杂 memory graph 反向索引。
