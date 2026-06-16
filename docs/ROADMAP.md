# HiveMemory 系统开发路线图

**文档状态**: Living Roadmap  
**更新时间**: 2026-06-13  
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
| v0.5.0 | Data Durability & Artifact 体系 | 进行中 | 将原始交互与记忆生成 provenance 固化为可追溯的冷资产；建立 artifact 数据底座 |
| v0.5.1 | 基础设施清理 | 规划中 | Config 重分层（Patchouli/Alice 独立 config）、NoOp 组件补全、cancel_event 传入 MTP executor |
| v0.5.2 | Async-Native Adaptation | 规划中 | 存储层切换 AsyncQdrantClient，generation/retrieval 全链路原生 async |
| v0.5.3 | Patchouli 架构重构 | 规划中 | LibrarianCore 解构，建立 MemoryLibrary + MemoryIngestionPipeline，扩展 RetrievalFamiliar 为全层检索 |
| v0.6.0 | System Gateway & Commands | 候选规划 | Gateway 上移、复合意图拆分、MTP READ/RUN、Alice Phase 3 |
| v0.7.0+ | 记忆 split/merge、L3 冷存储复活 | 候选规划 | 在稳定数据底座上扩展高级生命周期行为 |

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

## 4. 后续候选版本

以下内容不是强制顺序，但建议在 v0.5.0 的数据底座完成后再进入。

### v0.6.0 候选: System Gateway 与系统指令

目标：

- 将 Gateway / TheEye 正式上移到 System 级。
- 支持 `/clear` 等系统指令的统一解析与执行。
- 增加复合意图识别与任务拆分。
- 让 Gateway 的拆分结果能被 Patchouli、Alice 与下游 task runner 消费。

依赖：

- v0.4.0 的 runtime run/task 控制。
- v0.5.0 的 raw interaction artifact 与 task provenance。

### v0.7.0 候选: MTP READ 历史编译与 MTP RUN executable asset

目标：

- MTP READ 从简单读取 payload 升级为读取版本历史、provenance、artifact 来源。
- MTP RUN 从执行 `payload.content` 升级为从记忆原子编译 executable asset。
- 引入权限快照、构建过程、运行前检查和执行结果 artifact。

依赖：

- MemoryCompiler IR 继续推进。
- v0.5.0 的 artifact/provenance。
- v0.4.0 的 cancel 与 runtime event。

### v0.8.0 候选: Alice Phase 3 顶层编排

目标：

- Alice 正式成为顶层 orchestrator。
- 管理不同编排策略：single-agent、parallel specialists、plan-and-execute、review loop 等。
- 将复杂任务拆成可观测、可取消、可追踪的任务图。
- 统一控制 Patchouli、AgentRuntime、Gateway 与 MemoryCompiler 的协作。

依赖：

- v0.3.0 的 CALL / frame / IPC 基础。
- v0.4.0 的 runtime control。
- v0.5.0 的 task artifact 与 provenance。
- v0.6.0 的 Gateway 任务拆分能力。

### v0.9.0 候选: 记忆分裂、合并与高级生命周期

目标：

- 初步实现记忆 split / merge 行为。
- 支持将混杂记忆拆成多个可追溯记忆。
- 支持多条相近记忆合并，同时保留来源与版本历史。
- 生命周期维护从归档/评分扩展到结构性整理。

依赖：

- v0.5.0 的 raw transcript、history、provenance。
- MemoryCompiler READ 能读取历史和来源。
- Retrieval evidence 可解释为什么某些记忆被合并或拆分。

---

## 5. 长期方向

HiveMemory 的长期目标可以分为四条主线。

1. **记忆系统主线**
   - 从 MemoryAtom CRUD 走向可审计、可回放、可演化的记忆图谱。
   - 强化版本历史、provenance、artifact、split/merge、权限与可解释性。

2. **运行时主线**
   - 从单次 chat stream 走向 run/task/job graph。
   - 强化 cancel、timeout、retry、event、task outcome、backpressure。

3. **多智能体主线**
   - 从 Agent Profile 隔离走向 Alice 顶层编排。
   - 支持多策略、多角色、多阶段、可回溯的协作流程。

4. **编译系统主线**
   - 从 prompt renderer 走向 MemoryCompiler IR。
   - 支持 READ、RUN、embedding、shared context、agent profile、executable asset 等多目标编译。

---

## 6. 当前优先级

当前最高优先级是完成 v0.5.0：

1. `ArtifactKind` / `ArtifactRef` / artifact payload models。
2. `ArtifactStore` v1 与本地文件系统实现。
3. chat run raw interaction artifact。
4. history / generation transcript artifact。
5. memory generation provenance 写入。
6. memory task outcome artifact。
7. retrieval evidence artifact。
8. async cold path task runner。
9. artifact read API 与前端 provenance 入口。
10. 兼容旧数据与现有路由。
