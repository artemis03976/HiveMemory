---
title: HiveMemory Project
status: current
owner: project
scope: project-overview-and-index
code_paths:
  - src/hivememory/system/
  - src/hivememory/server/
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/routes-and-events.md
  - docs/contracts/mtp.md
last_reviewed: 2026-07-28
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

- **记忆基础设施**：通过 API 和 Passive Ingress 接收外部信息流，提供记忆存储、检索、话题、生成和生命周期能力；
- **原生 Agent Runtime**：以 Alice 执行 Agent，以 MTP 访问记忆与工具，验证记忆作为运行时状态和控制输入的价值。

HiveMemory 不是通用 AGI，也不是已经完成的分布式 Agent 平台。当前实现是单进程、异步、面向个人开发与实验验证的系统。

## 4. 版本状态

| 口径 | 当前值 | 含义 |
|:---|:---|:---|
| 最新已发布标签 | `v0.5.0` | 最近一次可由 Git tag 指认的发布基线 |
| 当前开发基线 | `v0.6.0` | 已合并 Gateway、Commands、Workflow、Passive Ingress，尚未发布 |
| Python 包元数据 | `0.1.0-beta` | `pyproject.toml` 中尚待统一的历史值，不代表当前发布状态 |
| 运行时代码版本 | `0.6.0` | `src/hivememory/__init__.py` 的开发版本声明 |

版本状态不得只引用其中一个字段。发布信息以 Git tag 为准，开发中的设计状态以本文、[当前架构](./architecture/overview.md)和[路线图](./ROADMAP.md)为准。

## 5. 当前已具备的能力

### 5.1 入口与对话

- 主动 chat：Gateway 决策后执行 Patchouli prepare、Alice run、Patchouli finalize；
- SSE 流式与非流式 Agent run；
- 全局系统指令注册、解析、分发和 chat 短路；
- Passive Ingress：外部离散事件去重、顺序缓冲、封口提交和失败重试；
- Gateway 的话题路由、查询分析、请求取消、总超时和局部保守降级。

### 5.2 记忆与知识

- MemoryAtom 的创建、查询、更新、删除、引用和反馈；
- 基于 Qdrant 的持久化存储与 Dense + Sparse 混合检索；
- 活跃话题、语义缓冲、感知、生成和生命周期维护；
- Raw Interaction、Transcript、Retrieval Evidence、Memory Version 等 artifact/provenance 基础；
- MemoryCompiler 的统一检索上下文与 MTP READ 渲染；
- 可查询、可取消、可观测的记忆生成任务。

### 5.3 Agent Runtime

- Agent Profile 驱动的模型、采样参数、语言和权限；
- Agent loop、结构化 turn events、取消与运行终态；
- MTP `SEARCH / READ / RUN / WRITE / UPDATE / CALL`；
- PendingAtom 延迟物化、alias redirect 和结算通知；
- 根 Agent 对子 Agent 的有限深度 CALL 编排。

### 5.4 运行与观测

- `GlobalSystemBus` 公开 RPC / PubSub 和子系统 local bus；
- chat、Gateway、Agent、memory task、maintenance、passive ingress、system lifecycle 等 RuntimeEvent；
- 有界事件回放与 stream gap 表达；
- 全局维护调度器、模型 warmup、health/readiness；
- Provider / Model 注册表和单次 generation options 覆盖。

## 6. 当前系统结构

```text
HiveMemorySystem
  ├─ System application services
  │    ├─ ChatApplicationService
  │    └─ PassiveIngressService
  ├─ GlobalSystemBus / RuntimeEventBus / Scheduler
  ├─ GatewaySystem   入口决策与命令
  ├─ PatchouliSystem 记忆与知识平面
  └─ AliceSystem     Agent 执行与控制平面
```

详细组件图、主动/被动数据流和启停顺序见[系统架构概览](./architecture/overview.md)。职责和状态所有权见[系统边界](./architecture/boundaries.md)。顶层装配、应用服务、运行时、配置、可观测性与 i18n 的内部设计从 [System 当前文档](./system/README.md)进入。

## 7. 三个子系统

### 7.1 Gateway

Gateway 把原始入口消息投影为命令终态或稳定 `GatewayDecision`。它负责入口拦截、命令、候选话题、话题路由、查询分析和保守降级，不执行检索或回复生成。

代码入口：`src/hivememory/gateway/system.py`、`runtime/`、`workflow/`、`commands/`、`analysis/`。

当前设计入口：[Gateway 总览](./gateway/README.md)、[固定工作流](./gateway/workflow.md)、[话题与查询分析](./gateway/analysis.md)、[全局命令](./gateway/commands.md)。

### 7.2 Patchouli

Patchouli 拥有长期记忆、话题、Agent Profile、检索、感知、生成、生命周期和记忆任务。主动链路中它负责 prepare/finalize，MTP 所需的长期记忆能力也由它通过公开路由提供。

代码入口：`src/hivememory/patchouli/system.py`、`src/hivememory/patchouli/runtime/`、`application/`、`memory/`，以及共享的 `src/hivememory/engines/`。

### 7.3 Alice

Alice 消费 `AgentRunContext` 执行 Agent run，拥有 frame、Agent loop、Koakuma MTP runtime、PendingAtom 运行时视图和 CALL 编排。它不直接拥有长期记忆存储。

代码入口：`src/hivememory/alice/system.py`、`runtime/`、`src/hivememory/agent_runtime/`。

## 8. 关键协作流程

### 8.1 主动模式

```text
message
  -> Gateway PROCESS
  -> [command short-circuit] 或 GatewayDecision
  -> Patchouli prepare
  -> Alice run / run_stream
  -> Patchouli finalize
  -> response + background memory tasks
```

Prepare 成功而 finalize 未完成时，System 请求 Patchouli cleanup。取消或失败的 Agent run 不默认触发长期记忆生成。

### 8.2 被动模式

```text
external event
  -> PassiveIngressService
  -> Gateway PROCESS(PASSIVE_MEMORY)
  -> optional memory context
  -> turn buffer / seal
  -> Patchouli submit interaction
```

被动模式不运行 Alice、MTP、命令或回复生成。

### 8.3 MTP

Agent 使用 `⟪ VERB | TARGET | ARGS ⟫` 在生成中主动检索、读取、执行、写入、更新记忆或调用子 Agent。WRITE/UPDATE 返回 ACK 只表示 PendingAtom 已登记，正式持久化由 Patchouli finalize 后续处理。

完整规范见[MTP 契约](./contracts/mtp.md)。

## 9. 设计真相源

### 9.1 治理与方向

- [文档治理与维护规范](./DOCUMENTATION.md)：分类、状态、目录和 PR 闭环；
- [项目愿景](./VISION.md)：长期定位、假设和取舍原则；
- [开发路线图](./ROADMAP.md)：发布历史、当前阶段和未来计划；
- [文档迁移清单](./plans/documentation-migration-inventory.md)：旧文档迁移动作与批次。

### 9.2 当前架构

- [系统架构概览](./architecture/overview.md)
- [系统边界与所有权](./architecture/boundaries.md)
- [Architecture 索引](./architecture/README.md)
- [架构决策记录](./architecture/decisions/README.md)

### 9.3 当前契约

- [子系统公共契约](./contracts/subsystem-contracts.md)
- [公开路由与事件](./contracts/routes-and-events.md)
- [Memory Tool Protocol](./contracts/mtp.md)
- [跨边界错误模型](./contracts/error-model.md)

### 9.4 子系统与系统模块

System 与 Gateway 已完成本轮事实核验和当前文档重建；Patchouli 与 Alice 仍处于 P1 迁移阶段，使用其 README 时应同时核对代码和 P0 契约：

- [System](./system/README.md)：组合根、应用服务、Passive Ingress、runtime/bus、配置、可观测性与 i18n；
- [Gateway](./gateway/README.md)：固定 workflow、话题/查询分析与全局命令；
- [Patchouli](./patchouli/README.md)
- [Alice](./alice/README.md)

### 9.5 其他文档类型

- [Plans](./plans/README.md)：明确但尚未完全实现的功能与迁移；
- [Ideas](./ideas/README.md)：未形成承诺的开放探索；
- [Todo](./todo/README.md)：小范围缺陷和技术债；
- [Help](./help/README.md)：安装、配置、使用和排障；
- [Applications](./applications/README.md)：产品规格；
- [Frontend](./frontend/README.md)：前端当前设计与迁移入口；
- [Archive](./archive/README.md)：已被替代或完成的历史材料。

## 10. 当前已知限制

- `v0.6.0` 的复合意图下游消费和自定义入口规则尚未完整落地；
- 包元数据、README 历史版本文字和发布流程仍需继续统一；
- RuntimeEvent 与当前 memory task 状态主要是进程内能力，通用持久化 Job Queue 尚未实现；
- MTP RUN 不能作为执行不受信任代码的安全沙箱；
- 附件、Document Ingestion、Deep Research、完整对话分叉和高级记忆回档仍是未来工作；
- Patchouli、Alice 与 P2 文档迁移尚未完成，`docs/mod/`、`docs/engines/` 和部分旧 README 仍只可作为待核验或历史材料；本批已迁移的 System/Gateway 旧设计已标记为 `superseded`。

## 11. 修改入口

改变后端设计的 PR 应先确定语义所有者，然后更新对应当前文档：

| 变化 | 必须检查的文档 |
|:---|:---|
| 子系统职责、状态所有权或依赖方向 | Architecture overview / boundaries |
| route、event、模型、错误或 MTP | Contracts 对应文档 |
| System / Gateway / Patchouli / Alice 内部设计 | 对应子系统当前文档 |
| 尚未实现的新功能 | Plans，不写入当前能力 |
| 版本阶段或排期 | ROADMAP |
| 已完成计划或被替代设计 | 合并当前事实后进入 Archive |

具体评审清单见[DOCUMENTATION.md](./DOCUMENTATION.md)。
