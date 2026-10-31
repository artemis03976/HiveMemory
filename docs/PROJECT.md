---
title: HiveMemory Project
status: current
owner: project
scope: project-overview-and-index
code_paths:
  - src/hivememory/system/
  - src/hivememory/server/
  - src/hivememory/mcp_server/
  - integrations/claude-code/
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/routes-and-events.md
  - docs/contracts/mtp.md
  - docs/contracts/mcp.md
last_reviewed: 2026-10-31
---

# HiveMemory 项目总览

本文既是 HiveMemory 后端设计文档的全局入口，也是理解项目缘起与核心设计理念的第一站。它首先回答“为什么需要 HiveMemory、我们试图解决什么长期问题”，然后再说明“项目今天已经走到哪里、系统如何划分、从哪里继续阅读”。

更完整的长期命题与可证伪假设见[项目愿景](./VISION.md)，未来排期见[开发路线图](./ROADMAP.md)，具体可执行行为则以当前架构和契约文档为准。

## 1. 为什么要做 HiveMemory

### 1.1 蜂巢记忆的构想

HiveMemory（蜂巢记忆系统）最初来自一个朴素但顽固的问题：今天的语言模型可以在单次对话中表现得非常聪明，却很难自然拥有跨越会话、模型和执行环境的长期连续性。

正如蜂巢以稳定结构保存、组织并反复利用采集到的资源，HiveMemory 希望把 Agent 运行中不断流过的对话、决策、代码、资料和经验，从易逝的上下文转化为拥有身份、来源和生命周期的记忆资产。它们不只是“被保存过的文本”，而应当可以被再次寻址、验证、修正、组合和使用。

这也是项目名称中 “Hive” 的真正含义：系统追求的不是无限堆积信息，而是让分散在不同会话和不同 Agent 中的知识，逐渐形成可共享、可追溯、仍然由用户拥有的长期结构。

### 1.2 四类长期问题

#### 上下文遗忘与注意力衰减

更长的 context window 能延后遗忘，却不能自动解决状态治理。重要约束可能被埋在长对话中间，窗口裁剪或会话重置更会使偏好、决策和工具经验整体消失。Agent 因而反复询问已经回答过的问题，或一再重做已经完成的工作。

#### 跨会话知识隔离

在 Session A 中调试完成的代码、确认过的架构决策或踩过的坑，通常不会自然进入 Session B。没有稳定的知识资产层，Agent 只能依赖用户重新描述，或重新生成一份质量不确定的答案。对长期项目而言，这种隔离会持续侵蚀一致性，也违背知识复用的基本原则。

#### 外源信息昂贵而易逝

Web Search、Deep Research、长文档和工具输出往往包含高价值证据，但在普通 Agent 中，它们常常只活在当前上下文里。会话结束后再次使用这些信息，意味着重新搜索、重新解析，并承担来源变化和结果漂移的风险。真正有价值的外源材料需要保留证据、出处和当时形成结论的关系，而不只是留下一段摘要。

#### 多 Agent 协作中的信息孤岛

Coder、Reviewer、Researcher 或 PM Agent 如果只靠转发 prompt 协作，信息会在反复压缩中失真，责任和版本也难以追踪。多 Agent 的价值不应建立在“消息传得更多”，而应建立在它们能够围绕同一组可寻址知识资产工作，并清楚区分谁读取、谁修改、谁产生了新的证据。

### 1.3 为什么普通 RAG 仍然不够

传统 RAG 缓解了“找不到旧信息”，却通常仍把检索结果视为一次性的 prompt 片段。它很少回答：这条信息是否已经过时？它来自哪里？本次回答是否修改了它？不同 Agent 是否有权读取或执行它？会话结束后产生的新结论如何回到知识库？

因此，HiveMemory 关注的不只是 retrieval，而是一个完整闭环：

```text
原始交互与外部证据
  -> 保守识别值得保留的内容
  -> 形成可寻址、可追溯的记忆资产
  -> 为当前任务编译工作上下文
  -> Agent 主动读取、使用或提出修正
  -> 新的执行事实再次进入记忆演化
```

## 2. 项目希望建立什么

### 2.1 核心成功标准

- **长期连续性**：重要偏好、约束、事实和工作成果能够跨会话继续生效，而不是依赖用户反复提醒；
- **知识复用**：面对相似任务时，系统优先发现并利用已经验证的历史资产，而不是无条件重新生成；
- **可解释的记忆演化**：用户能够理解系统为什么记得、引用了什么、何时发生更新，以及某条知识为何被降权或归档；
- **多 Agent 共享共识**：不同 Agent 围绕同一事实核心协作，同时保持身份、权限和责任边界；
- **渐进式自主维护**：系统能够辅助写入、更新、合并和遗忘，但所有自主行为都必须建立在可观察、可追溯和可纠正的基础上。

这些是需要持续验证的目标，而不是当前版本已经完全兑现的宣传承诺。

### 2.2 核心设计理念

#### 记忆是资产，不是日志堆积

并非所有 token 都值得成为长期记忆。原始证据、候选记忆和正式记忆需要不同的身份与生命周期；保存一切会迅速把记忆库变成另一种不可检索的聊天记录。

#### Context 是针对任务编译出的工作视图

持久化资产才承担跨执行周期的长期状态，context window 则是由当前任务、Agent Profile、权限、检索结果和 token 预算共同编译出的临时视图。MemoryCompiler 的存在正是为了避免各模块随意拼接 prompt，最终形成彼此不一致的“上下文真相”。

#### 记忆既可以被注入，也必须能够被主动使用

预检索适合低延迟地提供背景，但 Agent 还需要在运行中决定何时 SEARCH、READ、WRITE、UPDATE 或使用某项工具。MTP 是当前对 “Memory as a Tool” 的协议化表达，它的价值在双向交互边界，而不只在特殊语法。

#### 热路径与冷路径必须分离

用户希望立即获得相关记忆，高质量整理却可能需要更慢的感知、去重、生成和生命周期判断。HiveMemory 因而把当前响应所需的 Gateway 决策、检索和上下文准备放在热路径，把可延迟的记忆沉淀与维护放入冷路径。后台整理不能成为前台交互的强制等待项。

#### 记忆、入口决策与 Agent 执行是三种不同责任

Patchouli 关心什么值得保留以及知识如何演化；Gateway 关心一条输入应以何种方式进入系统；Alice 关心模型、工具和子 Agent 如何完成一次运行。把三者放进同一个“聪明的总管”会制造循环依赖和模糊所有权，因此当前架构将它们拆为同级子系统，再由 System 应用层编排协作。

#### 原始证据与保守演化优先

错误记忆往往比遗忘更危险。摘要不能替代不可恢复的来源，未知不能被包装成确定事实，更新也应尽量修正既有资产而不是不断追加冲突副本。Artifact、provenance、PendingAtom 和结构化错误模型都是这一原则在不同层面的工程结果。

### 2.3 评估问题

项目是否有价值，最终不能只以功能数量判断。更关键的问题包括：

- 在真实重复任务中，相关记忆能否被稳定召回并实际复用？
- 与只依赖原生上下文相比，事实一致性和长期约束遵循是否改善？
- 一条记忆被引用、修改或判定无效时，是否能够追溯到证据与运行过程？
- 系统产生的错误记忆和重复记忆是否会随时间收敛，而不是不断放大？
- 用户需要手动修正、删除或重新解释同一事项的频率是否下降？
- Alice 的 memory-native 运行方式是否能在至少一个真实场景中产生普通外挂 RAG 难以获得的优势？

### 2.4 非目标

- HiveMemory 不追求成为通用 AGI，也不宣称当前系统能够无监督地持续自我进化；
- 它不以替代所有现有 Agent harness 为前提，Patchouli 应当能够独立服务外部系统；
- 它不是只有向量搜索与 prompt 注入的 RAG 包装层；
- 当前版本不是分布式任务平台，也没有可安全执行不受信任代码的强隔离沙箱；
- 多模态、完整任务图、任意距离回档等方向只有在形成真实需求和可靠前置能力后才会进入当前设计。

## 3. 今天的项目定位

HiveMemory 是面向 LLM Agent 的持久化记忆系统与实验性 Memory-Native Agent Runtime。它把值得长期保留的信息建模为可寻址、可检索、可更新、可追溯的记忆资产，并让 Agent 在生成过程中主动使用这些资产。

项目当前同时验证两条互补路径：

- **记忆基础设施**：通过 API 和 Import Bus 接收外部 harness 已经发生的交互，经 MCP 让外部 harness 检索与读取记忆，提供记忆存储、检索、话题、生成和生命周期能力；
- **原生 Agent Runtime**：以 Alice 执行 Agent，以 MTP 访问记忆与工具，验证记忆作为运行时状态和控制输入的价值。

HiveMemory 不是通用 AGI，也不是已经完成的分布式 Agent 平台。当前实现是单进程、异步、面向个人开发与实验验证的系统。

## 4. 版本状态

| 口径 | 当前值 | 含义 |
|:---|:---|:---|
| 最近已发布基线 | `v0.7.1` | Git tag 创建于 2026-10-30；plugin 模式接入 Claude Code：对话经 Import Bus 回流进入记忆生成，Claude Code 经 MCP 工具检索与读取记忆；另有推理模型的空闲卸载与工具解析链路重写 |
| 当前开发版本 | `v0.7.2` | Planned：版本目标为生成质量，见 [ROADMAP](./ROADMAP.md) |
| 规范代码版本 | `0.7.1` | 由 `src/hivememory/_version.py` 唯一声明，Python 构建、运行时、HTTP API 与前端包清单保持一致 |

版本号与发布状态是两个不同事实。`pyproject.toml` 通过 setuptools dynamic metadata 读取规范代码版本，FastAPI/OpenAPI 与 `/health` 直接复用运行时版本，前端清单由 CI 一致性检查约束。当前代码和前后端清单为 0.7.1，与最近已发布标签 v0.7.1 对应；v0.7.2 开发期间合入的内容在版本收尾时升版。版本目标判定与核对记录见 [v0.7.1 收尾审计](./archive/plans/v0.7.1-release-closeout-audit.md)，上一个版本见 [v0.7.0 收尾审计](./archive/plans/v0.7.0-release-closeout-audit.md)。Python PEP 440 与 npm SemVer 对预发布后缀的规范化方式不同，因此当前门禁不发布预发布包；未来若需要 rc/beta，必须先为两种生态补充显式映射和构建产物校验，不能绕过一致性检查。

发布信息以 Git tag 为准，开发中的设计状态以本文、[当前架构](./architecture/overview.md)和[路线图](./ROADMAP.md)为准。

## 5. 当前已具备的能力

### 5.1 入口与对话

- 任务请求的唯一注册入口：chat 与命令经两阶段认证登记为任务进程，以 `process_id` 标识；进程持有自己的 task 与输出通道，用户停止、交付方断连与系统关闭都由进程在自己的 task 中收尾；
- 主动 chat：Gateway 决策后执行 Patchouli prepare（话题与检索）、任务进程 CPU 分配（Profile、附件与记忆编译）、经 CPU 端口执行 Actor（当前为 Alice run）、有输出时任务进程封口并追加会话、Patchouli finalize；
- ConversationSession 的创建、列表、重命名、删除与历史恢复；会话属于 Workspace，逐轮选择 Agent，Alice 从全量原文构建带工具减重与接力摘要的预算视图，Gateway 使用最近 N 轮；原文与压缩检查点仅在进程内保存；
- SSE 流式 Agent run；
- 全局系统指令的注册与解析：命令以命令终态结束、不进入普通 chat，当前不执行内置命令（[Gateway 全局命令](./gateway/commands.md)）；
- Import Bus：外部 harness 一轮已结束交互的整轮导入（`POST /api/v1/import/turns`），经统一认证与导入授权、Gateway 话题落位、共用封口后提交 Patchouli 感知，交互应用完成后才确认，不检索记忆、不返回记忆上下文；仓库附带 Claude Code 插件（`integrations/claude-code/`），其中的连接器在轮次结束与会话开始、结束时把会话记录中已结束的轮次导入，并把插件自身记忆工具的调用标为 SEARCH、READ，避免取回的记忆被再提取；
- MCP 记忆访问：外部 harness 经 `POST /mcp` 的两个只读工具 `memory_search` 与 `memory_read` 检索与读取记忆；每次工具调用以请求头中的接入身份与凭据经统一认证网关签发一份请求级 context，经操作入口的请求级提交进入 actor API、按访问登记逐次授权，不建进程；Claude Code 插件以 `.mcp.json` 声明这个 MCP server（[MCP 记忆访问](./contracts/mcp.md)）；
- 外部 harness 的接入身份：每个外部 harness 在接入登记中有自己的 principal 与接入凭据（Claude Code 为 `harness:claude-code`），导入与 MCP 共用；能否声明保留的 `system` 由登记逐个限定；
- Gateway 的话题路由、查询分析、请求取消、总超时和局部保守降级。

### 5.2 记忆与知识

- MemoryAtom 的创建、查询、更新、删除、引用和反馈；
- 基于 Qdrant 的持久化存储与 Dense + Sparse 混合检索；
- 活跃话题、语义缓冲、感知、生成和生命周期维护；
- Raw Interaction、Transcript、Retrieval Evidence、Memory Version 等 artifact/provenance 基础；
- MemoryCompiler 的统一检索上下文与 MTP READ 渲染；
- 可查询、可取消、可观测的记忆生成任务。

Workspace 身份隔离与附件链路已在 v0.6.2 落地：TXT/Markdown/DOCX 经上传、确定性解析、READY 选择和 AttachmentCompiler 进入 Chat；成功 Interaction 才形成 TopicAssetBinding，后续 Memory CREATE/UPDATE 按需生成来源 Artifact。WorkspaceAsset 仍是进程内 working set，事实入口见 [Workspace](./workspace/README.md) 与 [Chat 附件](./system/attachments.md)。

### 5.3 Agent Runtime

- Agent Profile 驱动的模型、采样参数与语言；Agent 的操作权限只由 Workspace 访问登记的 actor 操作白名单表达；
- Agent loop、结构化 turn events、取消与运行终态；
- MTP `SEARCH / READ / RUN / WRITE / UPDATE / CALL`；
- workspace 共享的 PendingAtom 登记、操作内实时派发与队列接纳、alias redirect 和结算通知；
- MTP 引用读取统一经 workspace 读取视图：L0 意图按 Workspace 回读，L1/L2 正式记忆逐次检查 actor policy；原子与 Profile 解析缓存由 workspace 持有，canonical 变更同步失效；
- 根 Agent 对子 Agent 的单层 CALL 编排：子 Agent 作为执行线程以被调用方身份运行，按其白名单与记忆可见性授权。

### 5.4 运行与观测

- `GlobalSystemBus` 公开 RPC / PubSub 和子系统 local bus；
- Local Work Queue Runtime 的 lane、状态机、backpressure、retry、cancel、timeout 与 shutdown drain；
- chat、Gateway、Agent、memory task、maintenance、Import Bus、system lifecycle 等 RuntimeEvent；
- 有界事件回放与 stream gap 表达；
- 全局维护调度器、模型 warmup、health/readiness；
- Provider / Model 注册表和单次 generation options 覆盖。

## 6. 当前系统结构

```text
HiveMemorySystem（组合根与门面）
  ├─ 门面提供的服务
  │    ├─ TaskProcessService      任务进程表与 chat 任务进程编排（workspace.process）
  │    ├─ ImportBusService        外部交互导入（System 的 Import Bus）
  │    ├─ WorkspaceOperationEntry 操作入口（只供 MCP 入口使用其请求级提交）
  │    └─ Memory / Task / Profile / Topic / Asset 管理服务（workspace.capability 的 root API，只供 system）
  ├─ 共享运行时（components）：GlobalSystemBus / RuntimeEventBus / Scheduler / Local Work Queue
  ├─ Workspace 设施：认证网关、认证一侧与操作授权者、写入意图登记、读取视图与失效订阅、WorkspaceAssetStore
  ├─ GatewaySystem   入口决策与命令
  ├─ PatchouliSystem 记忆与知识平面
  └─ AliceSystem     Agent 执行与控制平面
```

代码按层组织（core/config → components → engines/infrastructure → workspace/gateway/patchouli/alice → system → server/mcp_server），依赖只能指向同层或更低层。详细组件图、分层规则、主动链路与 Import Bus 的数据流和启停顺序见[系统架构概览](./architecture/overview.md)。职责和状态所有权见[系统边界](./architecture/boundaries.md)。顶层装配、应用服务、配置与 i18n 的内部设计从 [System 当前文档](./system/README.md)进入；总线、调度器、work queue 与运行时事件等机制从 [Components 当前文档](./components/README.md)进入。

## 7. 子系统

### 7.1 Gateway

Gateway 把原始入口消息投影为命令终态或稳定 `GatewayDecision`。它负责入口拦截、命令、候选话题、话题路由、查询分析和保守降级，不执行检索或回复生成。

代码入口：`src/hivememory/gateway/system.py`、`runtime/`、`workflow/`、`commands/`、`analysis/`。

当前设计入口：[Gateway 总览](./gateway/README.md)、[固定工作流](./gateway/workflow.md)、[话题与查询分析](./gateway/analysis.md)、[全局命令](./gateway/commands.md)。

### 7.2 Patchouli

Patchouli 拥有长期记忆、话题、Agent Profile、检索、感知、生成、生命周期和记忆任务。主动链路中它负责 prepare/finalize，MTP 所需的长期记忆能力也由它通过公开路由提供。

代码入口：`src/hivememory/patchouli/system.py`、`runtime/`、`application/`、`services/`、`control/`、`memory_library/`，以及由它拥有的 `src/hivememory/engines/{perception,generation,retrieval,lifecycle,artifacts,memory_compiler}/`。

当前设计入口：[Patchouli 总览](./patchouli/README.md)、[MemoryLibrary](./patchouli/memory-library.md)、[Artifacts](./patchouli/artifacts.md)、[感知](./patchouli/perception.md)、[生成](./patchouli/generation.md)、[检索](./patchouli/retrieval.md)、[生命周期](./patchouli/lifecycle.md)与 [MemoryCompiler](./patchouli/memory-compiler.md)。

### 7.3 Alice

Alice 以 CPU 端口的实现接入任务进程，消费任务进程组装的输入清单（`CPUInputManifest`）执行 Agent run，拥有 frame、Agent loop、Koakuma MTP runtime、frame ACK alias 集合和 CALL 编排。写入意图登记与引用解析由 workspace 持有，Alice 经线程 context 绑定的提交函数发出操作请求。它不直接拥有长期记忆存储。会话压缩算法与逐帧输入守卫见 [Agent Runtime](./alice/agent-runtime.md#31-会话上下文视图)，检查点由 workspace 会话资源接受和保存。

代码入口：`src/hivememory/alice/system.py`、`runtime/`、`src/hivememory/agent_runtime/`。

当前设计入口：[Alice 总览](./alice/README.md)、[Agent Runtime](./alice/agent-runtime.md)、[多 Agent 编排](./alice/orchestration.md)、[PendingAtom](./alice/pending-atom.md)与 [MTP Runtime](./alice/mtp-runtime.md)。

### 7.4 Workspace

Workspace 是与上述三个子系统同层的包，没有独立的生命周期宿主，由组合根装配。它负责认证入口与访问检查、能力层（只供 `system` 的 root API 与只经操作入口供 actor 使用的 actor API）、任务进程表与 chat 任务进程编排（含 Actor 阶段内的执行线程）、会话资源、读取视图与写入意图登记，以及 WorkspaceAsset。Topic、Memory 与 Artifact 的领域所有权仍在 Patchouli。

代码入口：`src/hivememory/workspace/`。

当前设计入口：[Workspace 总览](./workspace/README.md)、[身份与授权](./workspace/identity-and-authorization.md)、[认证与访问登记](./workspace/authentication.md)、[能力层](./workspace/capability.md)、[任务进程](./workspace/task-process.md)、[执行线程](./workspace/execution-threads.md)、[会话资源](./workspace/sessions.md)、[读取视图与写入意图](./workspace/read-view-and-write-intents.md)与 [WorkspaceAsset](./workspace/assets.md)。

## 8. 关键协作流程

### 8.1 主动模式

```text
message
  -> Gateway PROCESS
  -> [命令终态（只解析，不执行）] 或 GatewayDecision
  -> Patchouli prepare（话题与检索）
  -> 任务进程 CPU 分配（Profile、附件租借与编译、记忆编译、输入清单）
  -> Actor 执行：CPU 端口（当前为 Alice run）
  -> 有输出时封口交互记录（InteractionPayload，含结局）并追加会话
  -> Patchouli finalize
  -> response + background memory tasks
```

有输出的 completed/cancelled/failed 均提交交互；无输出、进程异常与 finalize 前断连不追加会话、不提交。finalize 任务在交出 finalizing 事件前创建，交付关闭也等待提交结束；尚未交接的 prepare 结果才请求 cleanup，补偿仅限新建空 Topic。附件租借在进程结束时释放，已接纳的显式写入继续生成。

### 8.2 Import Bus

```text
外部 harness 一轮已结束的交互（例如 Claude Code 连接器）
  -> POST /api/v1/import/turns
  -> 统一认证网关 + 导入授权
  -> Import Bus 编排（按外部会话串行，查询交互进度）
  -> Gateway PROCESS(PASSIVE_MEMORY)：只取话题落位
  -> 共用封口函数 seal_interaction
  -> Patchouli interaction.submit -> 感知队列
  -> 等待应用完成后确认（accepted / duplicate / skipped）
```

Import Bus 不运行 Alice、MTP、命令或回复生成，不检索记忆，也不返回记忆上下文。完整设计见 [Import Bus](./system/import-bus.md)。

### 8.3 MTP

Agent 使用 `⟪ VERB | TARGET | ARGS ⟫` 在生成中主动检索、读取、执行、写入、更新记忆或调用子 Agent。资源操作统一经绑定线程 context 的操作请求进入 workspace 的 actor API，每个动词检查访问登记中对应的一项 actor 操作，Alice 只持有只读观测标签；READ 与 RUN 的引用交付统一记录 `workspace.reference_read` citation，CALL 目标 Profile 包括内置目标都要求 `agent.call`。WRITE/UPDATE 在操作内登记并实时派发，ACK 表示生成队列已接纳，正式持久化由 Patchouli 生成链处理。停止被接受或 Actor 阶段结束时全部执行线程结束、线程 context 撤销，此后的操作请求以线程已结束拒绝；CPU 失败、取消、进程关闭或 CALL 子帧失败不撤回已提交写入。

完整规范见[MTP 契约](./contracts/mtp.md)。

## 9. 设计真相源

### 9.1 治理与方向

- [文档治理与维护规范](./DOCUMENTATION.md)：分类、状态、目录和 PR 闭环；
- [项目愿景](./VISION.md)：长期定位、假设和取舍原则；
- [开发路线图](./ROADMAP.md)：发布历史、当前阶段和未来计划；
- [文档迁移清单](./archive/plans/documentation-migration-inventory.md)：已经完成的旧文档迁移动作与批次记录。

### 9.2 当前架构

- [系统架构概览](./architecture/overview.md)
- [系统边界与所有权](./architecture/boundaries.md)
- [数据模型与可变性边界](./architecture/data-model.md)
- [Architecture 索引](./architecture/README.md)
- [架构决策记录](./architecture/decisions/README.md)

### 9.3 当前契约

- [子系统公共契约](./contracts/subsystem-contracts.md)
- [公开路由与事件](./contracts/routes-and-events.md)
- [Memory Tool Protocol](./contracts/mtp.md)
- [MCP 记忆访问](./contracts/mcp.md)
- [跨边界错误模型](./contracts/error-model.md)

### 9.4 子系统与系统模块

各子系统与系统模块的当前文档入口：

- [System](./system/README.md)：组合根、应用服务、Import Bus、配置与 i18n；
- [Components](./components/README.md)：总线、维护调度器、Local Work Queue、串行门与运行时事件等进程内机制；
- [Gateway](./gateway/README.md)：固定 workflow、话题/查询分析与全局命令；
- [Patchouli](./patchouli/README.md)：MemoryLibrary、Artifacts、Perception、Generation、Retrieval、Lifecycle 与 MemoryCompiler；
- [Alice](./alice/README.md)：Agent Runtime、CALL 编排、Agent Profile、写入意图端口的消费与 MTP Runtime；
- [Workspace](./workspace/README.md)：身份与授权、认证与访问登记、能力层、任务进程与执行线程、会话资源、读取视图与写入意图、WorkspaceAsset。

### 9.5 其他文档类型

- [Governance](./governance/README.md)：跨版本可靠性、安全和数据模型治理主题及其阶段性基线；
- [Research](./research/README.md)：外部 Memory、Agent 系统、论文与技术方案的调研需求、证据快照、设计比较和适配分析；
- [Plans](./plans/README.md)：已经绑定版本或里程碑、可独立验收的功能与迁移；
- [Ideas](./ideas/README.md)：未形成承诺的开放探索；
- [Todo](./todo/README.md)：小范围缺陷和技术债；
- [Help](./help/README.md)：Docker/本地安装、配置来源、Provider/Model 管理与排障；
- [Applications](./applications/README.md)：产品规格、试验假设与验证证据；
- [Frontend](./frontend/README.md)：应用壳、Chat、管理页面、状态所有权与传输；
- [Archive](./archive/README.md)：已被替代或完成的历史材料。

本轮旧文档迁移已经关闭：主干、Ideas、历史实施稿与源码 README 均已逐篇确定责任和最终位置，旧混合目录不再承担入口职责。分类依据和全库门禁可从[文档迁移最终收口审计](./archive/plans/documentation-migration-finalization-audit.md)追溯。

## 10. 当前已知限制

- 复合意图目前只有分类信号与不对外提交的私有 `sub_intents`，公共 envelope、多分支消费和合并语义尚未实现；相关[复合意图分解 Idea](./ideas/composite-intent-decomposition.md)当前未排期。自定义入口拦截规则同样尚未形成明确接入需求与独立 Plan；
- RuntimeEvent、memory task、PendingAtom、Agent frame 和若干恢复状态主要是进程内能力，统一成熟度与恢复门槛见[耐久性与故障恢复治理](./governance/reliability/durability-and-recovery.md)；
- 重试、重复投递和跨存储副作用尚未形成统一业务边界，见[跨子系统幂等性与重试治理](./governance/reliability/idempotency-and-retry.md)；
- MTP RUN 不能作为执行不受信任代码的安全沙箱；身份传播、缓存隔离和执行安全见[身份隔离与执行安全治理](./governance/security/identity-and-execution-safety.md)；
- Document Ingestion、Deep Research、完整对话分叉和高级记忆回档仍是未来工作；
- 前端仍是面向个人开发的实验工作台：没有聊天历史恢复、账户边界和完整 Settings 配置结构对齐，mock fallback 也尚未统一标识。

## 11. 修改入口

改变后端设计的 PR 应先确定语义所有者，然后更新对应当前文档：

| 变化 | 必须检查的文档 |
|:---|:---|
| 子系统职责、状态所有权或依赖方向 | Architecture overview / boundaries |
| route、event、模型、错误或 MTP | Contracts 对应文档 |
| System / Gateway / Patchouli / Alice 内部设计 | 对应子系统当前文档 |
| Frontend 状态、页面、传输或控制面 | Frontend 对应当前文档；用户操作变化同时检查 Help |
| 安装、配置、端口或排障方式 | Help 对应指南与根 README |
| 应用目标、假设或验收证据 | Applications 对应规格；不得反写为系统当前事实 |
| 尚未实现的新功能 | Plans，不写入当前能力 |
| 版本阶段或排期 | ROADMAP |
| 已完成计划或被替代设计 | 合并当前事实后进入 Archive |

具体评审清单见[DOCUMENTATION.md](./DOCUMENTATION.md)。
