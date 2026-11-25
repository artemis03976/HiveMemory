---
title: HiveMemory Roadmap
status: current
owner: project
scope: releases-and-planned-capabilities
updates:
  - docs/PROJECT.md
  - docs/plans/
  - docs/governance/
  - docs/ideas/
  - docs/todo/
  - docs/archive/plans/
last_reviewed: 2026-11-24
---

# HiveMemory 开发路线图

本文维护版本阶段、当前完成度、依赖关系和相关入口：当前版本围绕什么功能、为什么是它，以及哪些方向仍是候选。已经生效的详细设计进入 Architecture、System 或子系统文档；跨版本质量目标进入 Governance；尚未实施的设计进入 Ideas；绑定版本的实施方案进入 Plans；完成的计划在收尾时删除，只在 [Archived Plans](./archive/plans/README.md) 登记。

版本功能优先由真实使用中观察到的问题决定（[VISION](./VISION.md) 第 10.10 节）。计划之间不预先排定顺序：一份计划只判断能否立即执行，能够立即执行时绑定当前版本建立；不能立即实施的方向只留在 Idea，不预先分配版本号（第 4.1 节）。

## 1. 状态口径与当前状态

| 状态 | 含义 |
|:---|:---|
| Released | 已有对应 Git tag 的发布版本 |
| Current Development | 当前开发版本，主体可能已合并但尚未发布 |
| Completed | 版本或事项内容已收尾；是否已发布仍以 Git tag 为准 |
| Candidate | 候选方向，范围仍可调整，不绑定版本 |
| Unscheduled | 已确认问题或方向，但尚未进入具体版本承诺 |
| Partially Landed | 一部分已进入当前实现，其余仍未完成 |
| Deferred | 已明确后置，不属于近期承诺 |

- **最近已发布**：`v0.7.3`（Git tag，2026-11-04），主动写入与被动提取的收敛，作为过渡措施。
- **当前内容基线**：`v0.8.0`，生成核心的升级，2026-11-21 收尾（第 3 节，[收尾审计](./archive/plans/v0.8.0-release-closeout-audit.md)），发布标签 `v0.8.0` 在合并后创建；下一个版本 `v0.8.1` 围绕短期记忆，开发中（第 3.4 节）。
- **规范代码版本**：`0.8.0`，由 `src/hivememory/_version.py` 唯一声明，Python 包、前端清单与锁文件保持一致。
- **v1.0 已确定的范围**：桌面客户端，以及 Claude Code、DeepSeek Harness、Codex 与 pi 四个 harness 的 plugin 模式接入。从 v1.0 起，每个版本的 GitHub Release 附上本机打的桌面安装包；此前的版本照常只发布工作流生成的 Python 包与前端产物（owner，2026-11-21）。

## 2. 发布历史

| 版本 | 核心结果 | 当前依据 |
|:---|:---|:---|
| `v0.1.0-beta` / `v0.1.1` | 记忆 MVP、API、基础前端与早期冷热路径 | Git tag；后续设计已演进 |
| `v0.2.0` | 多 Agent 隔离、Agent Profile 记忆化 | Git tag |
| `v0.3.0` | CALL、PendingAtom、Alice Orchestrator、MemoryCompiler | [Alice](./alice/README.md)、[MTP](./contracts/mtp.md) |
| `v0.4.0` | chat run / memory task 取消控制与 RuntimeEvent | [路由与事件](./contracts/routes-and-events.md) |
| `v0.5.0` | artifact/provenance、MemoryLibrary、async-native、模型注册 | [System](./system/README.md)、[Patchouli](./patchouli/README.md) |
| `v0.6.0` | System Gateway、全局命令、主动/被动入口契约 | [Gateway](./gateway/README.md)、[Import Bus](./system/import-bus.md) |
| `v0.6.1` | Local Work Queue Runtime、交互提交统一接入、记忆生成队列 | [Runtime](./components/runtime-and-bus.md#3-local-work-queue-runtime) |
| `v0.6.2` | Workspace MVP、端到端 scope、Chat Attachments、V1 Memory 迁移 | [Workspace](./workspace/README.md)、[收尾审计](./archive/plans/v0.6.2-release-closeout-audit.md) |
| `v0.7.0` | Workspace 网络与任务进程：唯一注册入口、两阶段认证与授权、能力层、执行线程、会话资源、写入意图实时派发、记忆内容版本 | [Workspace](./workspace/README.md)、[收尾审计](./archive/plans/v0.7.0-release-closeout-audit.md) |
| `v0.7.1` | plugin 模式接入 Claude Code：Import Bus 对话回流与 MCP 记忆访问 | [Import Bus](./system/import-bus.md)、[MCP 记忆访问](./contracts/mcp.md)、[收尾审计](./archive/plans/v0.7.1-release-closeout-audit.md) |
| `v0.7.2` | 生成质量：生成材料不再按单轮截断、一个话题提取多条记忆、查重命中后先落库再合并、Claude Code 经 MCP 写入与更新记忆 | [记忆生成](./patchouli/generation.md)、[收尾审计](./archive/plans/v0.7.2-release-closeout-audit.md) |
| `v0.7.3` | 主动写入与被动提取的收敛（过渡措施）：话题内主动写入优先、生成失败带原因码、主动写入沿用 agent 的标题 | [记忆生成](./patchouli/generation.md#42-本话题的写入)、[收尾审计](./archive/plans/v0.7.3-release-closeout-audit.md) |
| `v0.8.0` | 生成核心的升级：记忆类型的收敛、AgentProfileStore、生成的记忆指向交互轮次、馆长 agent 与写入时的编辑集修订；试验场与第一次 HaluMem 评测 | [记忆生成](./patchouli/generation.md)、[收尾审计](./archive/plans/v0.8.0-release-closeout-audit.md) |

各版本的计划见 [Archived Plans](./archive/plans/README.md)。过去文档中的 `v0.5.1` 至 `v0.5.3` 是 v0.5 开发期的内部批次，不是发布标签；旧文档中出现的、后来撤销或改排的版本号（例如 `v0.7.2` Deep Research、`v0.8.x` 产品化窗口）以本文为准，不重写历史。

## 3. v0.8：记忆核心（v0.8.0 已收尾，v0.8.1 开发中）

### 3.1 v0.8.0 的版本目标

生成核心的升级：类型的更迭与 Patchouli 馆长 agent。owner 于 2026-11-21 决定以已经完成的方向收尾，被动生成的“写什么”不在 v0.8.0 中等待。v0.8 重新设计记忆从短期到中期的流转，不为现有症状打补丁；顺序是生成核心的升级、被动生成、主动写入，馆长 agent 是被动生成体系中的一环。一个版本围绕一个功能，由几份能各自验收的计划完成。设计见 [v0.8 总纲](./ideas/memory-flow-short-to-mid-term.md)及其子文档。

### 3.2 v0.8.0 已完成的方向

| 方向 | 结果 | PR |
|:---|:---|:---|
| 记忆类型的收敛 | 去掉 WORK_IN_PROGRESS；CODE_SNIPPET 与 URL_RESOURCE 合并为资源记忆；关闭 RUN 的用户工具路径；已有记忆按 owner 逐条决定迁移 | #225 |
| AgentProfileStore | Agent Profile 移出中期库，以文件保存并热重载；可用 agent 的列表常驻提示词 | #226 |
| 生成的记忆指向交互轮次 | 事件记下来自本轮第几次模型响应，段由事件派生，记忆的版本记录引用段 | #228 |
| 馆长 agent | 话题结算可选馆长：先看本话题的写入与库中相关记忆再写；与一次性生成并存为用户选项 | #231 |
| 试验场第一版 | 隔离的试验实例与 HaluMem 的驱动、评分器与报告；在 Medium 的 5 个用户上对照一次性生成与馆长；评测之前修复英文提取提示词中的中文 Schema 说明 | #234 |
| 写入时的更新 | 馆长可以修订读过的记忆；馆长的修订、主动 UPDATE 与查重提名的合并改用编辑集，索引的修改幅度由编辑集的结构限制；三条路径提交前按同一条规则重读与重放；合并按 Workspace 依次执行、沿吸收关系改指 | #236 |

当前事实见[记忆生成](./patchouli/generation.md)、[MemoryLibrary 与存储层](./patchouli/memory-library.md)，计划见 [Archived Plans](./archive/plans/README.md#v080)。复核的固定流程在分支上实现后不合并。本地提示词工作台（#232）用于在导出的结算材料上对照馆长与一次性生成的提示词，不进入产品。试验场是 v0.8 评测的承载（[孵化场、试验场与垃圾回收场](./ideas/experiment-field.md)第 6 节），第一次评测的结果见[评测集](./ideas/memory-evaluation-benchmarks.md)第 6 节，用法见[在 HaluMem 上评测记忆生成](./help/halumem-evaluation.md)。

### 3.3 v0.8.0 未纳入的事项与验收

- 未纳入 v0.8.0 的事项见[收尾审计](./archive/plans/v0.8.0-release-closeout-audit.md)第 4 节：被动生成写什么与短期记忆的定义仍在孵化，主动写入的生成材料、时空图索引与读的一侧没有开始；各设计领域的开放问题见 [v0.8 总纲](./ideas/memory-flow-short-to-mid-term.md)第 4.1 节。
- 验收方式仍是候选：单轮不交记忆税，长程看演化质量与人工介入（[v0.8 总纲](./ideas/memory-flow-short-to-mid-term.md)第 4.2 节）；留存率与人工介入量没有在新机制上复测。

### 3.4 v0.8.1：短期记忆（开发中）

版本目标：短期记忆以话题为载体，话题的工作状态回答“做到哪了”。现在话题只写不读，只被 Patchouli 用来生成记忆；空闲结算又把连续的对话切成几个话题，后台的话题反而比前台的会话更不连贯。设计要点：

- 话题的工作状态由已整理的部分（只增的工作记录，每一条指回原文）与未整理的原始交互记录组成；话题停下时写一条，一轮之内什么时候写暂缓。
- 结算拆成沉淀与退役：空闲只触发沉淀，退役只由容量与明确的结束触发。
- 折叠只为材料服务，折叠摘要结构化。
- 两个使用时点都由系统推送：另一个会话被路由回已有话题时推送工作状态，会话压缩之后只推送工作记录的索引。

设计与开放问题见[短期记忆：话题的工作状态及其维护](./ideas/short-term-memory-and-context-compaction.md)；结算拆成沉淀与退役（[感知与短期话题](./patchouli/perception.md#4-触发原因与具名用例)）与话题跨重启保存（同一文档[第 7 节](./patchouli/perception.md#7-维护与关闭)）已完成，工作状态落地之前的短期记忆代码整理也已完成，计划见 [Archived Plans](./archive/plans/README.md#v081)；话题的存储介质改为每个话题一条只追加的日志，计划见[话题的存储介质](./plans/v0.8.1-topic-append-only-log.md)；工作记录建立在它之上，其余部分还没有计划。

## 4. 版本之外的工作与候选方向

### 4.1 计划的准入

- 一份计划只在能够立即执行时建立：阻塞问题已经决定、依赖已经成为事实，就绑定当前版本建立；同一时间只有一份活动计划。
- 不能立即实施的方向只留在 Idea，不预先分配版本号，本文也不为它们排先后。
- 一个版本围绕一个功能，由多份计划依次完成，不以单份计划完成整个版本。

计划的写法与准入见 [Plans](./plans/README.md)。

### 4.2 目标总览

| 方向 | 状态 | 目标结果 | 入口 |
|:---|:---:|:---|:---|
| `v0.8.0` 生成核心的升级 | Completed（2026-11-21 收尾） | 类型的更迭、馆长 agent 与写入时的编辑集修订；被动生成与主动写入的版本未定 | 第 3 节 |
| `v0.8.1` 短期记忆 | Current Development（结算拆成沉淀与退役、话题跨重启保存与短期记忆代码整理已完成，话题的存储介质已有计划） | 话题成为前台可读的工作记录：工作状态、沉淀与退役、折叠与推送 | 第 3.4 节；[短期记忆 Idea](./ideas/short-term-memory-and-context-compaction.md) |
| 身份与访问体系 | Candidate / 单独主线 | 发布前打好地基：调用方证明、登记生效时点等仍未决定的项 | 第 4.3 节；[身份与访问体系 Idea](./ideas/identity-and-access-model.md) |
| 观测与日志 | Unscheduled / 版本之外 | 存量日志收敛的剩余部分 | [日志与运行时事件](./governance/observability/logs-and-events.md) |
| 冷启动、历史导入与外源文档资产化 | Candidate | 冷启动种子、历史对话导入、文档解析与来源保全 | 第 4.4 节 |
| MTP READ 专项编译与来源表达 | Candidate | 按资源类型、版本、定位与预算编译 READ 输出，给出可核验引用 | 第 4.5 节 |
| Deep Research MVP | Candidate | 研究状态、来源、证据、发现与报告闭环，执行提供者可替换 | 第 4.6 节 |
| 记忆价值体系重新设计 | Candidate | owner 重新设计价值体系，不在现有信号上增补 | 第 4.7 节 |
| 前端可靠性与资源体验 | Partially Landed | 状态可信性、资源操作、来源与导入体验 | 第 4.8 节 |
| controller 模式的外部 harness 接入 | Candidate | 外部 harness 作为执行环境运行用户选择的 agent | 第 5.6 节 |
| Alice 高级编排、高级记忆生命周期、执行基座与沙箱 | Deferred | 见各节 | 第 5.2 至 5.4 节 |

### 4.3 版本之外的工作

- **身份与访问体系**：单独的主线，最好在发布前打好地基。统一身份授权入口、Import Bus 的授权进入能力层、WorkspaceAsset 的归属拆分、交互提交收束与话题按需创建、管理页修改记忆可见性均已完成；HTTP 管理接口要求管理凭据，后端只监听本机，管理凭据只留在桌面客户端与后端的内存中。仍未决定的项见[身份与访问体系 Idea](./ideas/identity-and-access-model.md)。
- **观测与日志**：运行时事件写入单独的事件记录文件、失败事件带稳定原因码、日志带 trace、事件生产端收敛均已完成；存量日志收敛（W5）只完成了一部分（[日志与运行时事件](./governance/observability/logs-and-events.md)）。
- **HiveMemory 使用手册 skill**：跨 harness 通用，随插件加载，在使用中逐步完善（中英文两份 skill 在 [`integrations/hivememory/skill-languages/`](../integrations/hivememory/skill-languages/zh/SKILL.md)，后端启动时按语言设置写入插件目录）。
- **已完成**：DeepSeek Harness、Codex 与 pi 的 plugin 模式接入、Electron 桌面客户端与桌面端安装插件、日常实例与开发实例分开、写入句柄的可靠性、统一语言设置、前端 UI 优化与空入口接线、主配置的个人覆盖与热更新、访问登记的运行时管理、轻量化的第一批等，见 [Archived Plans](./archive/plans/README.md#版本之外)。

### 4.4 冷启动、历史导入与外源文档资产化（候选）

- **冷启动与历史对话导入**：解决“空库如何首次产生价值”和“既有会话如何保真迁入”。冷启动只让用户显式选择少量事实、偏好或资料作为种子，不把默认示例当作用户事实；历史导入保留来源、会话与轮次标识、发生时间、说话者及分支关系，助手的推测或旧结论不能直接升级成用户当前事实。harness 插件安装之前的会话从未导入。
- **导入过程**支持预览、来源与 Workspace 选择、稳定导入标识、去重、进度与失败报告，以及只保留证据、稍后提炼记忆；不按实时顺序把全部历史重放进当前活动话题。承诺重启续跑之前，先交付导入 checkpoint 与所需的任务持久化。
- **外源文档摄入**：先形成原始 Artifact，再产生 representation、chunk、locator 与 evidence，最后按策略提炼记忆；本地文档的解析不依赖 Alice 或通用代码执行。
- **依赖**：批量自动物化依赖记忆价值体系（第 4.7 节）；历史导入承接外部 Actor 接入契约的身份、来源与提交语义。验收至少包含重复导入、部分失败、来源回查、跨 Workspace 隔离、旧信息与当前事实冲突，以及中断后的可解释处理。

### 4.5 MTP READ 专项编译与来源表达（候选）

在完整记忆历史与外源文档的来源、表示与版本契约之上，扩展 MemoryCompiler 对文档、代码、历史证据等资源的定向读取、片段定位、token 预算和引用呈现。结果应说明读到哪个资源与版本、选择了哪个片段、引用如何回到原始证据，以及截断、缺失来源与无权限如何表达；不在文本末尾追加一个链接了事，也不建立第二套 provenance。READ 截断正文的缺陷已修为交付全文，按 token 预算截断并给出读取其余内容的方式仍属本方向。

### 4.6 Deep Research MVP（候选）

研究领域持有 request、source、evidence、finding 与 report；协调器负责推进、重试与进度，执行 adapter 提供搜索、抓取、文件读取或分析，研究状态不依赖 Alice 的 frame。首个 MVP 选择一个有预算和停止条件的真实研究任务，交付来源采集、证据定位、发现记录、报告 Artifact、取消与失败披露；只为验证过的研究流程增加最小协调状态，不预先建设通用 DAG 平台。跨进程恢复的范围在计划中冻结，在此之前只承诺进程内的暂停与继续。

### 4.7 记忆价值体系：重新设计（候选）

owner 重新设计价值体系，不在现有信号上增补。现状：入口不再产生价值信号，`worth_saving` 已移除，价值由写入者判定；正式记忆的生命周期评分另有依据（[Lifecycle](./patchouli/lifecycle.md)）。约束：入口信号不能替代 Patchouli 的最终决定，也不单独成为删除来源证据的依据；新体系不追溯性批量删除旧记忆，任何存量重评或迁移另有预览、审计和回滚范围。“什么值得写成记忆”是 v0.8 写入链路中与之相关的开放问题。

### 4.8 前端与 Todo

前端的可靠性与资源体验随受影响的 API 并行交付：统一身份状态、真实/过期/错误的区分与失败披露；记忆库的语义检索、可见性策略、附件与来源检查、导入预览与失败重试。界面、主题与空入口已随前端 UI 优化完成，动效、响应速度与自适应布局等细节见 [Todo](./todo/README.md)。Todo 排期按核对后的状态与实际依赖吸收，跨系统范围扩大时建立 Plan，不借“清理 Todo”引入未评审的新状态机。

### 4.9 依赖与验收门槛

```text
v0.7.3（已发布）
   |
   +-> v0.8.0（已收尾）生成核心的升级：类型 / AgentProfileStore / 轮次指向 / 馆长 / 编辑集修订；试验场第一版承载评测
          -> v0.8.1（开发中）短期记忆：话题的工作状态 / 沉淀与退役 / 折叠 / 推送
          -> 被动生成 -> 主动写入（版本未定）

候选方向之间的依赖（判断一份计划能否立即执行，不表示排期）：
外部 Actor 的身份 / 来源 / 提交语义 -> 历史导入契约与批次能力
记忆价值体系 -> 冷启动与历史导入的批量自动物化
完整记忆历史 + 文档来源 / 版本 -> MTP READ 专项编译
可靠执行提供者 + 证据 / READ + 研究状态恢复契约 -> Deep Research
任务进程表与唯一注册入口 + 外部 Actor 接入契约 + 身份与访问体系的地基 -> controller 模式
前端可靠性与资源体验 -> 随受影响的 API 并行交付
```

每项候选进入实施前必须冻结范围、权威状态、数据来源、失败/取消/恢复承诺、幂等边界和观察指标。任何跨重启恢复、强隔离或 Actor 替换承诺都必须有对应实现证据；仍未具备的能力不能通过 UI、accepted 响应或事件日志伪装成完成。

## 5. 产品化与后置方向

### 5.1 Electron 桌面客户端

Completed，当前事实见[桌面客户端](./frontend/desktop-client.md)：托盘常驻的安装版持有本机后端与 Qdrant，只有桌面窗口能管理；不签名、不自动升级、不做备份，删除记忆数据需要二次确认；任何正常退出都保存话题与待生成的工作，下次启动后继续。安装包随 v1.0 发布。崩溃或强制结束时的丢失不在已完成的范围内，按[耐久性与故障恢复](./governance/reliability/durability-and-recovery.md)另行处理。

### 5.2 Alice 高级编排

Deferred。plan-and-execute、多角色协作、parallel specialists、review loop 和任务图需经过 VISION 中 Alice 存在价值的判定，并与“外部 harness + 同一 Patchouli 记忆库”比较；优先验证能证明记忆作为状态与控制输入有价值的能力。

### 5.3 高级记忆生命周期

Deferred。记忆的分裂与合并、完整 rollback、branch invalidation、provenance reverse index 和 L3 复活，等待真实演化与恢复需求；落库之后的演化质量记在[生命力分数长期演进](./ideas/VitalityScoringLongTermEvolutionIdeas.md)方向 6。记忆价值体系的重新设计不等于提前实现全部生命周期能力。

### 5.4 执行基座与不可信代码沙箱

Deferred。目标是统一执行 provider 的环境绑定、输出、退出码、结构化结果、超时、取消和清理语义：Workspace 声明资源与可用能力，执行 provider 兑现文件、网络、进程与资源限制，Actor 选择调用。受信任执行、工具 API 限制与 OS 级强隔离必须区分，工作目录、prompt 或 Python 包装层不足以限制任意代码；第一版无法实施相应隔离时，不接纳要求该隔离等级的任务。RUN 的用户工具路径已关闭，可运行这条轴见[类型体系](./ideas/memory-type-system.md)第 6.2 节。

### 5.5 未排期的其他工作包

| 事项 | 状态与进入条件 | 入口 |
|:---|:---|:---|
| 自定义入口拦截规则 | Unscheduled；需真实外部接入场景、配置所有者和验收样本 | 待建立 |
| 领域持久化与恢复 | 导入、Research、桌面化按各自承诺提取最小切片，通用 checkpoint/saga 平台不提前展开 | [Durability Governance](./governance/reliability/durability-and-recovery.md) |
| 领域幂等与 reconciliation | 新的提交/导入入口先明确所需的业务幂等，再独立设计存量领域操作的补偿 | [Idempotency Governance](./governance/reliability/idempotency-and-retry.md) |
| 数据模型可变性治理 | 按具体 DTO/聚合风险分批 | [Mutability Governance](./governance/data-model/mutability.md) |
| 跨用户认证及权限治理 | 外部 connector 先落实所需的身份映射，不能把 scope 字段当作认证 | [Identity Governance](./governance/security/identity-and-execution-safety.md)、[身份与访问体系 Idea](./ideas/identity-and-access-model.md) |
| 通用长期 workflow / DAG | Research 先验证最小真实流程，再决定是否提炼通用机制 | [VISION](./VISION.md) |
| Conversation Branching | 依赖来源、生命周期和真实编辑需求 | [原始资料保存 Idea](./ideas/PatchouliPageFoldingRawEvidenceDesign.md#8-与编辑回退和分支的关系)第 8 节、[Issue #6](https://github.com/artemis03976/HiveMemory/issues/6) |

### 5.6 controller 模式的外部 harness 接入

Candidate。controller 模式把 Codex、Claude Code、pi 一类成熟的 harness 作为执行环境纳入：用户在对话开始前分别选择 agent 与执行环境，请求经唯一注册入口登记为任务进程，交给该 harness 执行；HiveMemory 控制任务边界（actor 选择、输入、生命周期与取消、执行轨迹回流），不控制 harness 内部的 loop、工具与上下文压缩。步骤分析与阻塞问题见[外部 Actor Idea](./ideas/external-actor-registration-and-runtime-access.md)，执行环境的选择见[执行环境 Idea](./ideas/execution-environment.md)，接口资料见[接口调研快照](./research/harness-interface-survey-snapshot.md)。验收必须形成真实闭环：外部 harness 经能力层检索已有记忆并实际用于任务，写入意图经能力层提交，交互记录进入记忆库，下一次会话能召回新知识并定位来源；仅转发请求、采集日志或返回 memory context 不足以证明接入完成。

## 6. 长期方向与评估

长期取舍遵循 [VISION](./VISION.md)：先证明开放记忆基础设施的实际价值，再验证 Alice 的 memory-native 运行能否带来额外收益。模型与 harness 的更换不应迫使用户重建长期知识资产；Actor 可替换也不表示其执行过程或能力可以完全等价迁移。评估逐步在同一任务和记忆库上比较无记忆、普通检索、外部 harness + Patchouli、Alice + Patchouli，记录跨会话复用、来源可查、错误与重复记忆、用户修正成本、执行可靠性、延迟和资源开销；新增功能、抽象层数和目录重排本身不作为成功指标。

## 7. 路线图维护规则

1. 只有 Git tag 对应的版本标为 Released。
2. 当前版本写明目标、方向与验收方向；计划只在能够立即执行时绑定当前版本建立，候选方向不预先分配版本号或先后；详细设计只维护一份。
3. 只有完整完成实现、测试、迁移与审查的内容晋升到当前事实文档，完成的计划在收尾时删除；本文中完成的项目链接 PR、归档索引或发布记录，不复述实施过程。
4. 部分落地保持 Partially Landed，Deferred 项只有满足触发条件才启动。
5. PROJECT、Architecture、System 与子系统索引的行为说明按文档治理收口门禁更新；调整候选排期不提前向事实层推广设计。
6. 依据真实使用、导入样本和运行故障定期复核候选方向与下一个版本的功能；范围变化显式记录，不把前置能力缺口隐藏在上层功能中。
