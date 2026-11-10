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
last_reviewed: 2026-11-09
---

# HiveMemory 开发路线图

本文只维护版本阶段、当前完成度、依赖关系和相关入口。已经生效的详细设计进入 Architecture、System 或子系统文档；跨版本质量目标进入 Governance；未承诺方向进入 Ideas；绑定版本的实施方案进入 Plans；完成后的实施文档进入 Archive。

路线图回答当前版本围绕什么功能、为什么是它，以及哪些方向仍是候选。自 v0.7.0 之后，计划之间不预先排定顺序：一份计划只判断能否立即执行，能够立即执行时绑定当前版本建立；不能立即实施的方向只留在 Idea 候选池，不预先分配版本号（owner，2026-10-30，第 4.1 节）。版本功能优先由真实使用中观察到的问题决定（[VISION](./VISION.md) 第 10.10 节）。能力之间的依赖仍然成立：异步任务、文件、文档、来源和研究如果各自建立一套临时状态与失败语义，规模越大越难保持记忆闭环，依赖未成为事实的计划不能立即执行（第 4.9 节）。

## 1. 状态口径

| 状态 | 含义 |
|:---|:---|
| Released | 已有对应 Git tag 的发布版本 |
| Current Development | 当前开发基线，主体可能已合并但尚未发布 |
| Completed | 版本或事项内容已收尾；是否已发布仍以 Git tag 为准 |
| Planned | 已形成目标和大致边界，尚未成为当前事实 |
| Candidate | 候选方向，范围仍可调整；v0.7.0 之后不绑定版本 |
| Unscheduled | 已确认问题或方向，但尚未进入具体版本承诺 |
| Partially Landed | 阶段中的一部分已进入当前实现，其余仍未完成 |
| Deferred | 已明确后置，不属于近期承诺 |

当前版本事实如下：

- 最近已发布基线：`v0.7.2`（Git tag，2026-11-01）；[收尾审计](./archive/plans/v0.7.2-release-closeout-audit.md)记录版本目标判定、内容核对与验证结果；
- 当前内容基线：`v0.7.3`，主动写入与被动提取的收敛，作为过渡措施（owner，2026-11-03），2026-11-04 收尾：话题内主动写入优先、生成任务失败带原因码与主动写入沿用 agent 的标题三份计划均已完成，发布标签 `v0.7.3` 在合并后创建；[收尾审计](./archive/plans/v0.7.3-release-closeout-audit.md)记录版本目标判定、内容核对与验证结果（第 4.3 节）。v0.7.3 之后的版本目标尚未确定；
- 版本之外：观测与日志的其余工作包随时可以开始（owner，2026-11-03，第 4.2 节），本机 HTTP 管理面已以管理凭据完成防误用（2026-11-05）；HiveMemory 使用手册 skill 已基本稳定，在使用中逐步完善（owner，2026-11-03，第 4.2 节）。

当前规范代码版本为 `0.7.3`，由 `src/hivememory/_version.py` 唯一声明并供构建与运行时复用。Python 包、前端清单与锁文件保持一致；本次发布标签为 `v0.7.3`，待合并后创建，最近已发布基线为 `v0.7.2`。

## 2. 发布历史

| 版本 | 状态 | 核心结果 | 当前依据 |
|:---|:---:|:---|:---|
| `v0.1.0-beta` / `v0.1.1` | Released | 记忆 MVP、API、基础前端与早期冷热路径 | Git tag；后续设计已演进 |
| `v0.2.0` | Released | 多 Agent 隔离、Agent Profile 记忆化 | Git tag；[Patchouli 当前设计](./patchouli/README.md)；[Alice 当前设计](./alice/README.md) |
| `v0.3.0` | Released | CALL、PendingAtom、Alice Orchestrator、MemoryCompiler | Git tag；[Alice 当前设计](./alice/README.md)；[MTP](./contracts/mtp.md) |
| `v0.4.0` | Released | chat run / memory task 取消控制与 RuntimeEvent | Git tag；[路由与事件](./contracts/routes-and-events.md) |
| `v0.5.0` | Released | artifact/provenance、MemoryLibrary、async-native、模型注册 | Git tag；[System 当前设计](./system/README.md)；[Patchouli 当前设计](./patchouli/README.md) |
| `v0.6.0` | Released | System Gateway、全局命令、主动/被动入口契约、Passive Ingress 串行与 outbox | Git tag；[Gateway 当前设计](./gateway/README.md)；[Import Bus 当前设计](./system/import-bus.md)（2026-10-21 取代 Passive Ingress） |
| `v0.6.1` | Released | Local Work Queue Runtime、Active/Passive Interaction Submission 统一接入、Memory Generation queue 与进程内可靠生命周期 | Git tag；[System Runtime 当前设计](components/runtime-and-bus.md#3-local-work-queue-runtime)；[归档实施计划](./archive/plans/v0.6.1-local-work-queue-runtime.md) |
| `v0.6.2` | Released | Workspace MVP（WorkspaceIdentity、端到端 scope、双 Workspace 隔离、进程级 WorkspaceAssetStore）、Identity 投影收敛、V1 Memory 迁移、Chat Attachments、MTP scope 修复与 Topic/短期存储边界整理 | Git tag；[Workspace](./workspace/README.md)；[附件链路](./system/attachments.md)；[收尾审计](./archive/plans/v0.6.2-release-closeout-audit.md) |
| `v0.7.0` | Released | Workspace 网络与任务进程：唯一注册入口与任务进程、两阶段认证与授权、能力层与 Agent 操作白名单、执行线程、会话资源与自动压缩、写入意图实时派发、记忆内容版本 | Git tag；[Workspace](./workspace/README.md)；[收尾审计](./archive/plans/v0.7.0-release-closeout-audit.md) |
| `v0.7.1` | Released | plugin 模式接入 Claude Code：Import Bus 对话回流与 MCP 记忆访问；推理模型空闲卸载；`TurnEvent` 统一的工具解析链路 | Git tag；[Import Bus](./system/import-bus.md)；[MCP 记忆访问](./contracts/mcp.md)；[收尾审计](./archive/plans/v0.7.1-release-closeout-audit.md) |
| `v0.7.2` | Released | 生成质量：生成材料不再按单轮截断，Topic 之后的原始证据可以回查；一个话题提取多条记忆；查重命中后先落库再合并；Claude Code 经 MCP 写入与更新记忆 | Git tag；[Patchouli 生成](./patchouli/generation.md)；[MCP 记忆访问](./contracts/mcp.md)；[收尾审计](./archive/plans/v0.7.2-release-closeout-audit.md) |

过去文档中的 `v0.5.1`、`v0.5.2`、`v0.5.3` 是 v0.5 开发期的内部工作批次，不是当前仓库中的独立发布标签。它们的已实现事实应按模块并入当前文档，而不是继续作为平行版本入口。

## 3. 最近已发布基线：v0.7.2

主题：**生成质量**。Git tag `v0.7.2` 创建于 2026-11-01。逐项内容、版本目标判定、未纳入的事项与验证结果见[收尾审计](./archive/plans/v0.7.2-release-closeout-audit.md)，各计划见 [Archived Plans](./archive/plans/README.md#v072)。

### 3.1 版本目标

- **版本目标**（owner，2026-10-30）：生成质量。v0.7.1 让 Claude Code 的日常使用开始产生记忆，真实使用随即暴露出生成链路的问题：生成视图只剩导入原文的 8%，折叠裁剪的原文没有留存，一个话题至多产出一条记忆，查重判出的 UPDATE 整体覆盖了已有记忆，plugin 模式没有接回 agent 的写入与更新。v0.7.2 由这些问题驱动 Patchouli 核心机制的改进。
- **计划的准入**（owner，2026-10-30）：计划之间不预设先后，各方向在阻塞问题决定、能够立即执行时建立计划（第 4.1 节）。
- **验收**：以 Claude Code 的真实会话验收，不只依靠离线测试。显式写入与更新经真实会话实测；生成材料不再按单轮截断有真实结算样本；多条提取与先落库再合并经离线回放验收，真实会话中的样本随日常使用观察（生成材料修正与查重与更新按 owner 决定，2026-10-31；多条提取按其计划第 9.3 节）。收尾审计第 1 节逐条判定。

### 3.2 发布内容

| 方向 | 结果 | 当前事实 |
|:---|:---|:---|
| 生成材料修正与原始证据保存 | 生成视图只限制单次工具调用的结果，一轮交互不再被单轮、思考与参数上限截断；`worth_saving` 移除；Topic 按顺序保存全部交互记录与折叠记录，工作视图由二者派生；结算冻结全部记录，来源 InteractionArtifact 覆盖已折叠的轮次并带折叠记录 | [感知与短期话题](./patchouli/perception.md)、[Patchouli 生成](./patchouli/generation.md)、[Artifacts](./patchouli/artifacts.md) |
| plugin 模式开放 WRITE 与 UPDATE | Claude Code 经 MCP 的 `memory_write`、`memory_update` 进入 actor API，派发到 Patchouli 的 Mode B、Mode C，与主动链路共用意图登记与生成链路；写入时 Import Bus 为进行中的一轮就地确定话题，这一轮回流时沿用该话题；记忆编译的命名与文案不再按 MTP 编写 | [MCP 记忆访问](./contracts/mcp.md)、[Import Bus](./system/import-bus.md#47-写入时确定话题) |
| 查重与更新：先落库再合并 | 查重落在合并区间时，草稿先作为新记忆落库，再由生成队列中的合并任务把两条记忆融合为一条，不再整体覆盖已有记忆；写入句柄随合并改指保留的记忆；判定信号不变 | [Patchouli 生成](./patchouli/generation.md#4-查重与合并)、[读取视图与写入意图](./workspace/read-view-and-write-intents.md) |
| 一个话题提取多条记忆 | 话题结算一次调用输出草稿列表，逐条查重与写入，同批别名互不冲突；输出被截断时任务失败、不再静默记为没有产出；Librarian 的输出上限提高到 32768 | [Patchouli 生成](./patchouli/generation.md#31-mode-a被动观察--archive) |
| 文档治理 | 归档与工作区索引每条一行、只列活动事项；文档治理规范删去已完成的迁移内容与内部重复；DeepSeek Harness 插件接入的调研需求与快照 | [文档治理规范](./DOCUMENTATION.md)、[调研需求与报告](./research/deepseek-harness-plugin-survey.md) |

### 3.3 发布范围边界

- **不在 v0.7.2**：查重的判定信号暂不改动（owner，2026-10-31，[查重 Todo](./todo/generation-dedup-decision-signal.md)）；记忆价值体系的重新设计（第 4.7 节）；冷启动与历史导入（第 4.4 节）；重排模型 int8 与推理模型重新加载的停顿（Todo，未排期）；正文进入检索索引（搁置，限制见 [Patchouli 生成](./patchouli/generation.md#8-当前限制)第 8 节）；交到 Topic 之前的原始资料保存仍是候选（[Raw Evidence Idea](./ideas/PatchouliPageFoldingRawEvidenceDesign.md)）。

上一个发布版本 v0.7.1（plugin 模式接入 Claude Code）的内容见第 2 节与[收尾审计](./archive/plans/v0.7.1-release-closeout-audit.md)，各计划见 [Archived Plans](./archive/plans/README.md#v071)。

## 4. 近期计划

### 4.1 排序原则与版本调整

本轮规划依据 [VISION](./VISION.md) 的双轨策略：Workspace/Patchouli 提供可独立使用的资源与记忆基础设施；Alice 是 memory-native reference runtime，外部 harness 可以通过适配器成为其他 Actor。先验证资源可复用、执行可替换，再推进文档、研究和桌面产品化。

“脱离 Actor”在此指资源身份、授权、证据、结果和任务状态不依赖某个特定 Actor。确定性解析可由普通服务完成；搜索、浏览或代码执行可以调用工具 worker 或 Actor。Deep Research 的执行仍需要执行者，但研究状态和结果应在执行者离开后继续存在。外部 harness 无需采用 Alice 的 loop、run/frame 工作集或完整 MTP 语法；主动写入的 Pending 读取与结算属于共同资源能力。

**计划的准入**（owner，2026-10-30）：

- v0.7.0 之后，计划之间没有预先排定的顺序约束。一份计划只判断能否立即执行：阻塞问题已经决定、依赖已经成为事实，就绑定当前版本建立；
- 不能立即实施的方向只留在 Idea 候选池，不预先分配版本号，路线图也不为它们排先后；
- 一个版本围绕一个功能，由多份计划完成，不以单份计划完成整个版本。

版本号的调整如下；这些调整不改变规范代码版本、Git tag 或已完成阶段的归属：

| 原排期 | 现状 | 调整原因 |
|:---|:---|:---|
| `v0.6.2` Workspace 资源体系与 Agent 执行边界 | `v0.7.0`，已发布（第 2 节） | 独立于 W0/W1 收口；2026-09-27 起按 Workspace 网络与任务进程架构重新规划 |
| `v0.7.0` Document Ingestion & Provenance | 2026-09-27 改排 `v0.7.2`（冷启动、历史导入与文档资产化）；2026-10-30 撤销版本号，改为候选方向（第 4.4 节） | v0.7.2 改以生成质量为版本目标；候选方向不预先分配版本 |
| `v0.7.1` MTP READ Provenance | 2026-09-27 改排 `v0.7.3`；2026-10-30 撤销版本号（第 4.5 节） | 同上 |
| `v0.7.2` Deep Research MVP | 2026-09-27 改排 `v0.7.4`；2026-10-30 撤销版本号（第 4.6 节） | 同上 |
| 整体后置的 Executable Asset Sandbox | 曾前置为 `v0.7.1` 执行基座；2026-10-20 起暂不考虑（第 5.4 节） | 原意是先解决 MTP RUN 可复现的可靠性问题和可兑现的能力边界；v0.7.1 改以外部 harness 接入为目标 |
| Electron `v0.8.x` 产品化窗口、Alice 高级编排 `v0.8.0+`、高级记忆生命周期 `v0.9.0+` | 2026-10-30 撤销版本窗口（第 5.1–5.3 节） | 候选方向不预先分配版本 |

旧 Idea、W1 实施记录及历史材料中出现的上述旧版本号，按本表解释；不重写历史使其看起来曾采用新顺序。新建 Plan 必须使用当前版本。

### 4.2 目标总览

| 版本或候选方向 | 状态 | 目标结果 | 依赖与入口 |
|:---|:---:|:---|:---|
| `v0.7.3` 主动写入与被动提取的收敛 | Completed（2026-11-04 收尾）/ 版本目标 owner，2026-11-03：三份计划均已完成 | 在统一的记忆架构成形之前止住两种生成方式互相改写（过渡措施）：被动提取不再改写本话题的主动写入，主动写入沿用 agent 的标题，生成失败可诊断 | 第 4.3 节；[收尾审计](./archive/plans/v0.7.3-release-closeout-audit.md) |
| 第二个外部 harness：DeepSeek Harness 的 plugin 模式接入 | Completed（2026-11-01 收尾）/ 版本之外的工作（owner，2026-10-31）；接入路径为纯原生插件（owner，2026-11-01）；owner 决定收尾，真实模型下的完整验收在日常使用中完成；调研中发现子 Agent 会话独立导入属于实现漂移（owner，2026-11-02），已于 2026-11-02 修复：只导入顶层会话，子 Agent 的结算通知进入父会话这一轮的生成材料 | dsh 经对话回流与原生记忆工具接入 HiveMemory，Windows 桌面端可以连接 WSL 中的 HiveMemory；之后经它进行 agent memory 与 harness 项目的调研，同时作为压测与数据积累 | [调研需求与报告](./research/deepseek-harness-plugin-survey.md)、[归档计划](./archive/plans/dsh-plugin-integration.md)、[接入手册](./help/deepseek-harness-plugin.md)、[子 Agent 会话的归档计划](./archive/plans/dsh-subagent-sessions.md) |
| Codex 的 plugin 模式接入 | Completed（2026-11-08 收尾，PR #199）/ v1.0 范围（owner，2026-11-04）；接入沿用 Claude Code 的形态、与 Claude Code 合用一个插件目录（owner，2026-11-08）；接入登记沿用配置文件，运行时登记之后另做（owner，2026-11-08）；以真实模型与本机后端的验收在日常使用中完成 | Codex 中的 agent 经 MCP 检索、读取、写入与更新记忆，写入由 hook 补上可信的轮次信息；对话回流 Import Bus；只支持 Codex 与 HiveMemory 都在 Windows 上原生运行 | [归档计划](./archive/plans/codex-plugin-integration.md)、[调研](./research/codex-plugin-survey.md)、[接入 Codex](./help/codex-plugin.md)、[Import Bus](./system/import-bus.md#10-codex-插件) |
| pi 的 plugin 模式接入 | Completed（2026-11-08 收尾，PR #201）/ v1.0 范围（owner，2026-11-04）；接入方式与 dsh 相同，扩展自带记忆工具、只允许模型直接调用，放进统一插件目录；社区子 Agent 扩展第一版不支持（owner，2026-11-08）；以真实模型与本机后端的验收已完成（2026-11-08）；dsh 插件已于同日并入同一目录，与 pi 共用 Node 公共部分（[归档计划](./archive/plans/dsh-unified-plugin-dir.md)） | pi 中的 agent 经扩展注册的记忆工具检索、读取、写入与更新记忆，写入由扩展补上可信的轮次信息；pi 没有持久的轮次标识，扩展为每一轮在会话记录中写下结束标记，据此回流 Import Bus；只支持 pi 与 HiveMemory 都在 Windows 上原生运行 | [归档计划](./archive/plans/pi-plugin-integration.md)、[调研](./research/pi-plugin-survey.md)、[接入 pi](./help/pi-plugin.md)、[Import Bus](./system/import-bus.md#11-pi-扩展) |
| 本机 HTTP 接口的密钥与访问控制 | Completed（防误用，2026-11-05）/ 防刻意随桌面客户端（第 5.1 节）：`/api/v1/config` 响应打码（PR #157）；密钥移出主配置与 `config.yaml`，提供商凭证收归 Provider 注册表（2026-11-01，PR #159，[归档 Todo](./archive/todo/config-secrets-written-to-yaml.md)）；模型定义不再保存连接凭证（2026-11-01，PR #166，[归档 Todo](./archive/todo/model-registry-api-key-override.md)）；Gateway 与 Librarian 只经模型与 Provider 注册表选择模型，主配置不再有 LLM 密钥（2026-11-01，PR #167，[归档计划](./archive/plans/llm-config-retirement.md)）；HTTP 管理面要求管理凭据、后端默认只监听本机地址，先防误用，防刻意随桌面客户端（owner，2026-11-05；[归档计划](./archive/plans/http-management-credential.md)） | HTTP 管理接口证明调用方，harness 中的 agent 不能绕过 Workspace 访问登记 | [认证与访问登记](./workspace/authentication.md#1-两阶段认证)第 1 节；[归档 Todo](./archive/todo/local-http-management-access.md)；防刻意见[身份与访问体系 Idea](./ideas/identity-and-access-model.md#http-管理入口的调用方证明)第 4 节 |
| 身份与访问体系 | Candidate / 不绑定版本；owner（2026-11-05）定为单独的主线方向，最好在发布前打好地基。统一身份授权入口已完成（2026-11-06，[归档计划](./archive/plans/unified-identity-authorization-entry.md)）：认证得到工作 actor，plugin 请求与任务进程共用同一个认证网关与两个授权方法（actor 执行期间的操作授权、系统级操作的 root 授权），交互的参与者显式传递；Import Bus 的服务只认证、需要授权的步骤经 root API（2026-11-06，[归档计划](./archive/plans/import-bus-authorization-in-capability-layer.md)）。WorkspaceAsset 的归属拆分已完成（2026-11-06，[归档计划](./archive/plans/workspace-asset-ownership-split.md)），`IdentityScope` 只由操作授权者构造。交互提交收束与话题按需创建已完成（2026-11-06，[归档计划](./archive/plans/interaction-submission-and-topic-on-demand.md)）：controller 与 Import Bus 的交互提交统一经 root API，话题在第一次写入或交互应用时建立，prepare 与 finalize 拆为能力层调用 | 从 principal 的登记与调用方证明，到进入 Workspace 的身份、操作调用的授权与资源可见性，收敛 v0.7.0 以来多轮返工留下的例外与补丁；controller 与 plugin 两种接入都以它为基础。仍未决定的有登记生效时点、controller 回调的调用方证明、进程控制中取消以外的授权主体与资源可见性 | [身份与访问体系 Idea](./ideas/identity-and-access-model.md)、[身份与授权](./workspace/identity-and-authorization.md#4-操作授权)、[现状调查](./governance/baselines/identity-access-current-state-2026-11-05.md)（实施前）、[身份治理](./governance/security/identity-and-execution-safety.md) |
| 观测与日志 | Unscheduled / 版本之外的工作（owner，2026-11-03）；W1～W3 已完成（2026-11-03），W4、W6 已完成、W5 部分完成（2026-11-07），其余未排期 | 排查一次失败只需要后端日志与事件文件：运行时事件写入单独的事件记录文件并可按关联标识排时间线、生成任务之外的失败事件带稳定原因码、维护任务空转不发事件，均已完成；日志格式带 trace 已完成；存量日志收敛的剩余部分（W5）见治理主题；事件生产端收敛（W6）随计划 [Runtime Event 体系完善](./archive/plans/runtime-event-system.md) 完成，事件按一次运行归组，补齐了 actor 操作与话题的事件 | [日志与运行时事件](./governance/observability/logs-and-events.md) |
| HiveMemory 使用手册 skill | Landed（2026-11-01，PR #162）：owner 决定做成跨 harness 通用的 skill，不只服务 DeepSeek Harness（2026-11-01）；已基本稳定，不再单独定稿，在使用中逐步完善（owner，2026-11-03） | 引导各 harness 中的 agent 主动检索、写入与修正记忆，并与 harness 自己的记忆共存，Claude Code、Codex、pi 与 dsh 都随插件加载它；skill 与 MCP 写入说明给出不分运行模式的写入判据，不以对话会归档作为少写的理由（owner，2026-11-09） | [skill](../integrations/hivememory/skills/hivememory/SKILL.md)、[接入 Claude Code](./help/claude-code-plugin.md)、[接入 DeepSeek Harness](./help/deepseek-harness-plugin.md) |
| 冷启动、历史导入与外源文档资产化 | Candidate / 不绑定版本 | 冷启动种子、历史对话导入、文档解析、证据与候选记忆分层，冻结 provenance | 第 4.4 节；[Artifacts](./patchouli/artifacts.md)、[附件链路](./system/attachments.md) |
| MTP READ 专项编译与来源表达 | Candidate / 不绑定版本 | 按资源类型、版本、定位和预算编译 READ 输出，给出可核验引用 | 第 4.5 节；[MTP](./contracts/mtp.md)、[MemoryCompiler](./patchouli/memory-compiler.md) |
| Deep Research MVP | Candidate / 不绑定版本 | 研究状态、来源、证据、发现和报告闭环，执行提供者可替换 | 第 4.6 节 |
| 统一的交互记忆：轨迹、断言与整理 | Candidate / 不绑定版本（owner，2026-11-03） | 同一套架构把对话与长程 agent 的交互都整理成记忆资产：候选记忆是带归属、证据位置与发生时间的断言，唯一写入者按断言的属性整理；对照断言清单标注已做离线回放 | [统一的交互记忆](./ideas/unified-interaction-memory.md) |
| 外部 harness 的 controller 模式 | Candidate / 不绑定版本（2026-10-20 由 v0.7.1 版本目标退回） | 外部 harness 作为执行环境运行用户选择的 agent，请求登记为任务进程并交给该 harness 执行 | 第 5.6 节；[接口调研快照](./research/harness-interface-survey-snapshot.md) |
| 记忆价值体系重新设计 | Candidate / 不绑定版本 | owner 重新设计价值体系，不在现有信号上增补 | 第 4.7 节 |
| Frontend Reliability & Resource UX | Partially Landed / 后续 Candidate | 先修身份、数据来源和状态可信性，再完善资源操作、来源和导入体验 | 随受影响的 API 并行；[Frontend](./frontend/README.md)与第 4.8 节 Todo |
| Electron 桌面客户端 | Candidate / 不绑定版本 | 单一客户端管理本地服务、数据目录、连接、升级与诊断 | 第 5.1 节；[状态与传输](./frontend/state-and-transports.md)、[配置](./system/configuration.md) |

### 4.3 v0.7.3：主动写入与被动提取的收敛

状态：Completed（2026-11-04；版本目标 owner，2026-11-03）。版本内的三份计划均已归档，规范版本已调整为 0.7.3，本次发布标签为 v0.7.3（合并后创建）。逐项内容、版本目标判定、未纳入的事项与验证结果见[收尾审计](./archive/plans/v0.7.3-release-closeout-audit.md)，各计划见 [Archived Plans](./archive/plans/README.md#v073)；证据、离线回放与决定见已归档的[收敛 Idea](./archive/ideas/generation-source-convergence.md)。

- **版本目标**（owner，2026-11-03）：在统一的记忆架构成形之前，先止住两种生成方式互相改写。这是过渡措施，不代表主动写入优先是更好的选择；之后的方向见[统一的交互记忆](./ideas/unified-interaction-memory.md)。DeepSeek Harness 接入后（2026-11-01），owner 经它调研了四个仓库，这是第一批以主动写入为主的真实样本：被动提取复现并改写了主动写入，同一轮里 agent 有意分开写的记忆被查重合并，主动写入的标题在生成时被丢弃，合并失败只能凭时序推断原因。
- **验收**：各计划以离线回放验收；日常使用中的实测依赖自然发生的结算、写入与失败，不作为归档条件。收尾审计第 1 节逐条判定。

| 方向 | 结果 | 当前事实 |
|:---|:---|:---|
| 话题内主动写入优先（过渡措施） | 被动草稿查重命中本话题主动写入结算到的记忆时丢弃，同一话题的主动写入之间不再提名合并；记录只在进程内 | [Patchouli 生成](./patchouli/generation.md#42-本话题的主动写入) |
| 生成任务失败带原因码 | 失败分类按异常给出稳定原因码，进入工作记录、`work.failed`、`memory.task.failed` 与分类处的一行日志；任务事件载荷带 `error_class` | [Patchouli 生成](./patchouli/generation.md#61-失败原因码)、[事件契约](./contracts/routes-and-events.md#44-失败与降级观测) |
| 主动写入沿用 agent 的标题 | Mode B 把 agent 拟定的标题交给模型并要求沿用，只在与内容不符或缺少检索名词时修改；MCP 写入工具说明与使用手册 skill 同步 | [Patchouli 生成](./patchouli/generation.md#32-mode-b主动-write)、[MCP 记忆访问](./contracts/mcp.md#3-工具) |

不在 v0.7.3：合并遇到版本冲突时是否重做（[合并版本冲突 Todo](./todo/merge-version-conflict.md)）；多条提取后被动提取的膨胀与低留存（[被动提取膨胀 Todo](./todo/passive-extraction-volume-and-retention.md)）；两种生成方式在统一架构中的兼容（[统一的交互记忆](./ideas/unified-interaction-memory.md)，候选方向）；查重的判定信号（[查重 Todo](./todo/generation-dedup-decision-signal.md)）；导入轮次的跨会话话题路由（[跨会话话题路由 Todo](./todo/cross-conversation-topic-placement.md)）；本机 HTTP 管理接口证明调用方（2026-11-05 以管理凭据完成防误用，[归档 Todo](./archive/todo/local-http-management-access.md)）。

### 4.4 冷启动、历史导入与外源文档资产化（候选）

状态：Candidate，不绑定版本（2026-10-30 撤销原 `v0.7.2` 的版本号）。建议拆成两个可独立验收的 Plan，共享资源归属、来源和物化策略；确定性解析可以提前并行验证。

**冷启动与历史对话导入**解决“空库如何首次产生价值”和“既有会话如何保真迁入”。冷启动允许用户显式选择少量项目事实、偏好、Profile 或资料作为种子；不得自动把默认示例当作用户事实。历史导入保留 source、conversation/turn 标识、发生时间、说话者及分支/编辑关系；助手的推测或旧结论不能直接升级成用户当前事实。Claude Code 插件安装之前的会话从未导入，MCP 记忆访问的第一个检索落空样本即由此而来（[已归档的观察 Todo](./archive/todo/import-bus-real-use-observation.md)）。

导入过程支持预览、来源与 Workspace 选择、稳定导入标识、去重、进度/失败报告，以及只保留证据、稍后提炼记忆。它承接外部 Actor 接入契约的身份、来源和领域提交语义及历史样例缺口，另行定义批次、历史时间、冲突和恢复语义；复用 Patchouli 的物化入口，不简单按实时顺序把全部历史文本重放进当前活动话题，避免改写当前偏好或丢失历史时间关系。

**外源文档摄入**先形成 RAW/Artifact，再产生 representation、chunk、locator 和 evidence，最后按策略提炼候选与正式 Memory。复用 W1 的附件与 provenance 基础，并明确进程内 WorkspaceAsset 与持久化 Artifact 的区别。来源获取、确定性解析、模型辅助理解和记忆物化各自有成功/失败边界。

本地已给定文档的解析不依赖 Alice 或通用代码执行；需要网页获取、本地目录浏览或复杂转换时使用相应 provider。语义提炼可以调用模型或 Actor，但不能让文档注册、证据保存和读取依赖 Alice 的 run。

验收至少包含重复导入、部分失败、来源回查、跨 Workspace 隔离、旧信息与当前事实冲突，以及中断后的可解释处理。若承诺重启续跑，先交付导入 checkpoint 和所需任务持久化切片；现有进程内队列不足以承担该承诺。批量自动物化依赖第 4.7 节的价值体系，避免一次导入大量低质量或相互冲突的记忆。

### 4.5 MTP READ 专项编译与来源表达（候选）

状态：Candidate，不绑定版本（2026-10-30 撤销原 `v0.7.3` 的版本号）。基于 [v0.7.0 A2-P](archive/plans/v0.7.0-a2-pre-memory-version-and-lifecycle.md)的完整记忆历史，以及外源文档摄入（第 4.4 节）的来源、文档表示与版本契约，扩展 MemoryCompiler 对文档、代码、历史证据等资源的定向读取、片段定位、token 预算和引用呈现。先冻结实际需要的 READ 模式与错误语义，再扩展协议。完整历史存储、内容版本与 lifecycle 更新边界由 A2-P 提前交付，本方向不再反向定义记忆写入格式。

结果应说明读到哪个资源和版本、选择了哪个片段、引用如何回到原始证据，以及截断、缺失来源、旧裁剪历史和无权限如何表达。历史记录中的 lifecycle 是捕获时状态，额外展示当前评估时必须区分时点，不能用当前原子补造历史。不能仅在现有文本末尾追加一个链接，也不能建立第二套 provenance 或检索状态。其编译能力可由外部 adapter 复用，MTP 负责自己的协议呈现。READ 截断正式记忆正文的缺陷已先行修复为交付全文（2026-11-01，[已归档的 Todo](./archive/todo/mtp-read-content-truncation.md)）；按 token 预算截断并给出读取其余内容的方式仍属本方向。

### 4.6 Deep Research MVP（候选）

状态：Candidate，不绑定版本（2026-10-30 撤销原 `v0.7.4` 的版本号）。研究领域持有 request、source、evidence、finding 和 report；协调器负责推进、重试与进度；执行 adapter 提供搜索、抓取、文件读取或分析。资源和研究状态不依赖 Alice 的 frame，Actor 可替换也不表示各 harness 的工具与策略完全等价。

首个 MVP 选择一个有预算和停止条件的真实研究任务，交付来源采集、证据定位、发现记录、报告 Artifact、取消和失败披露。以第 4.4、4.5 节的证据与读取结果作为报告依据，并复用既有队列；仅为验证过的研究流程增加最小协调状态，不预先建设通用 DAG 平台。

可先使用一个可靠的外部 harness 或工具 provider。若该路径不运行不可信本地代码，不把完整本地沙箱当作前置条件；若需下载并执行未知代码，先满足相应隔离门槛。执行提供者缺失时任务应明确等待或失败，不阻止已有研究资产读取。

研究状态跨 Actor 存续与跨进程恢复是两个承诺。恢复范围必须在正式 Plan 中冻结：若要在应用重启后继续，先持久化来源、进度与 checkpoint，并定义重放和重复副作用；在此之前只承诺实际支持的进程内暂停/继续。

### 4.7 记忆价值体系：重新设计（候选）

状态：Candidate，不绑定版本。owner（2026-10-30）将重新设计价值体系，而不是在现有信号上增补。

现状：入口不再产生价值信号。Gateway 的 WRITE/SKIP/UNKNOWN 预判与由它派生的 `worth_saving` 已随 v0.7.2 的生成材料修正移除（第 3 节），`PASSIVE_MEMORY` 也不再运行查询分析。此前 Perception settlement 曾按 `worth_saving` 整轮筛选材料，Claude Code 真实导入中这一预判只看到用户消息，把“请提交”“继续”背后的整轮实现丢出了生成，2026-10-29 起不再筛选（证据见已归档的[生成材料 Todo](./archive/todo/generation-material-filtering-and-truncation.md)）。正式 Memory 的生命周期评分另有依据，事实入口见 [Gateway Analysis](./gateway/analysis.md)、[Perception](./patchouli/perception.md)和 [Lifecycle](./patchouli/lifecycle.md)。

已有的约束：入口信号不能替代 Patchouli 的最终决定，也不应单独成为删除来源证据的依据；新体系不追溯性批量删除旧记忆，任何存量重评或迁移另有预览、审计和回滚范围。正式的 Idea 与 Plan 待建立。

### 4.8 前端与 Todo 并行工作流

Frontend 已部分落地，后续优化按可用性和资源操作体验推进，详细事实见 [Frontend](./frontend/README.md)。本轨不阻塞后端基础重构，但与受影响 API 一起验收。

- **近期可靠性**：统一身份状态，明确真实/stale/error，修复失败披露和乐观更新，覆盖请求终态；HTTP、SSE 与 WebSocket 的连接目标已统一由 `VITE_BACKEND_ORIGIN` 配置（2026-11-07）。
- **资源操作体验**：Memory Garden 真实语义检索、可见性策略、附件/来源检查、导入预览与失败重试。Workspace 选择 UI 只在对应后端能力真实可用后开放。
- **交互和视觉完善**：布局、导航、窄屏、可访问性、空状态及主题；避免将未接线的入口或研究页面显示成已有功能。Terminal 入口已是全局运行页（[归档计划](./archive/plans/run-view.md)）。

Todo 排期按已核对状态和实际依赖吸收，不能把目录中所有事项视为新需求：

| Todo | 排期方式 | 范围约束 |
|:---|:---|:---|
| [MTP 缓存作用域重验（归档）](./archive/todo/mtp-cache-scope-revalidation.md) | Completed / Archived；保留为 v0.7.0 回归基线 | 已修复的 L0/L1/L2 隔离语义不重排为待开发功能 |
| [总线路由类型化与 mypy 门禁（归档）](./archive/plans/v0.7.0-typed-bus-routes.md) | Completed / Archived（2026-10-12）：三阶段实现、验收与审查完成，`mypy src` 清零；属于 v0.7.0 内容 | 提供方 contracts 声明类型化 route；运行时参数检查保留，CI 按 `uv.lock` 检查源码和类型回归样例 |
| [Alice 健康探针](./todo/alice-health-probes.md) | v0.7.0 收尾时搁置（2026-10-19），未绑定版本 | 反映执行器实际就绪/故障；不令资源 readiness 依赖 Alice 在线 |
| [日志与运行时事件](./governance/observability/logs-and-events.md)（原 RuntimeEvent 生产端迁移 Todo，2026-11-03 转入治理主题 W6） | 版本之外的工作（owner，2026-11-03），随时可以开始 | 保持事件只做观测，不以事件投递成功控制业务状态 |
| [Mock fallback 披露（归档）](./archive/todo/frontend-mock-fallback-disclosure.md) | Completed / Archived（2026-11-07）：owner 决定 mock 数据只在 debug 模式下启用 | 默认读取失败如实显示，不以 mock 数据冒充真实结果 |
| [Memory 管理页列表与检索（归档）](./archive/todo/frontend-memory-library-listing.md) | Completed / Archived（2026-11-02）：列表跟随游标取全，语义检索请求后端 | 列表取全并显示真实总量；语义检索复用真实后端能力，保留失败和空结果 |
| [Memory visibility policy UI](./todo/memory-visibility-policy-ui.md) | 后端契约稳定后进入资源体验 | UI 编辑策略不能替代后端授权 |
| [会话 /compact 指令](./todo/conversation-compact-command.md) | 会话与 Alice 自动压缩已完成，仍依赖命令运行位置与命令系统接回 | compact 只作用于 CPU 的会话上下文，不改写 Session 记录与 Topic；后台 Topic 的 page folding 是另一个问题 |
| [Work Queue 多 lane 拓扑](./todo/work-queue-runtime-lane-topology.md) | 保持 Deferred | 只有共享 Store、跨 lane 调度或可复现故障触发才重评 |

上述排期摘要不代替 Todo 本体的完成条件。实现时先复核其是否已解决；跨系统范围扩大则建立 Plan，不通过“清理 Todo”引入未评审的新状态机。

### 4.9 依赖与验收门槛

```text
v0.7.1（已发布）：plugin 模式接入 Claude Code，真实使用开始产生样本
        |
        +-> v0.7.2（已发布）生成质量：生成材料与原始证据 / plugin 模式的写入与更新 / 多条提取 / 查重与更新
        |
        +-> v0.7.3（已收尾）主动写入与被动提取的收敛（过渡措施）：话题内主动写入优先 / 生成失败原因码 / 沿用 agent 的标题

候选方向之间的依赖（用于判断一份计划能否立即执行，不表示排期）：
外部 Actor 的身份 / 来源 / 提交语义 -> 历史导入契约与批次能力
记忆价值体系 -> 冷启动与历史导入的批量自动物化
v0.7.0 A2-P 完整记忆历史 + 文档来源 / 版本 -> MTP READ 专项编译
可靠执行提供者 + 证据 / READ + 研究状态恢复契约 -> Deep Research
任务进程表与唯一注册入口 + 外部 Actor 接入契约 -> controller 模式
身份与访问体系的地基（调用方证明、登记生效时点等） -> controller 模式；更多外部 harness 的 plugin 接入与桌面化
服务生命周期 / 传输 / 数据升级稳定 -> Electron 产品化
前端可靠性与资源体验 -> 随受影响的 API 并行交付
```

当前版本的计划见 [Plans 导航](./plans/README.md)。确定性导入可以提前并行，依赖复杂获取或转换的切片需等对应 provider。

每项候选进入实施前必须冻结范围、权威状态、数据来源、失败/取消/恢复承诺、幂等边界和观察指标。任何跨重启恢复、强隔离或 Actor 替换承诺都必须有对应实现证据。仍未具备的能力不能通过 UI、accepted 响应或事件日志伪装成完成。

## 5. 产品化与后置方向

### 5.1 Electron 桌面客户端

状态：Candidate，不绑定版本（2026-10-30 撤销原 `v0.8.x` 产品化窗口），正式 Plan 待建立；不是 Deep Research 或 Alice 高级编排的强制下游。完整发布之前可以做有限启动/连接原型，完整发布以资源生命周期、前端传输和数据管理稳定为门槛。

桌面化覆盖的工作多于 Web 页面外壳：本地 Python 服务及存储依赖的分发/连接、启动就绪、端口冲突、关闭 drain、数据目录、备份与升级、凭据、日志与崩溃诊断。HTTP 管理凭据的防刻意随桌面客户端完成：主进程在内存中持有凭据，不落盘（[身份与访问体系 Idea](./ideas/identity-and-access-model.md#http-管理入口的调用方证明)第 4 节）。HTTP、SSE、WebSocket 的连接目标已经统一配置（2026-11-07），直连时的管理凭据与 CORS 由桌面客户端处理。平台先选实际使用的一种完成闭环，再扩展安装包和签名发布。

Electron renderer 使用公开资源 API，main process 只持有必要的原生桥接与服务管理权限；远程内容和 Agent 代码不在特权主进程执行。Electron 的 renderer sandbox 不能替代 Agent 执行沙箱。

验收是可安装、启动、使用、关闭、重启和升级，用户知识资产持续可读；卸载/删除数据必须有明确语义。桌面生命周期引出的最小持久化需求按 [恢复治理](./governance/reliability/durability-and-recovery.md) 拆出，不依靠 localStorage 承担权威资源恢复。

### 5.2 Alice 高级编排

状态：Deferred，不绑定版本（2026-10-30 撤销原 `v0.8.0+` 后置窗口），不是已承诺的单一版本主题。plan-and-execute、多角色协作、parallel specialists、review loop 和任务图需经过 VISION 中 Alice 存在价值的判定，并与“外部 harness + 同一 Patchouli 记忆库”比较。

优先验证能证明记忆作为状态与控制输入有价值的能力，不因外部 Actor 接入而复制一套通用 harness 功能清单。

### 5.3 高级记忆生命周期

状态：Deferred，不绑定版本（2026-10-30 撤销原 `v0.9.0+` 方向）。Memory split/merge、完整 rollback、branch invalidation、provenance reverse index 和 L3 复活等待真实演化与恢复需求。记忆价值体系的重新设计不等于提前实现全部生命周期能力。

### 5.4 执行基座与不可信代码沙箱

状态：Deferred。执行基座与沙箱基线原排期在 v0.7.1，2026-10-20 起暂不考虑（owner）；完整不可信代码沙箱是其后的剩余工作包。

**执行基座与沙箱基线**：用户实测 MTP RUN 经常失败是启动调查的依据，正式 Plan 首先收集复现样例，区分协议解析、alias/权限、参数编译、环境、依赖、工具进程与取消链路的问题；不预设所有故障都由缺少沙箱引起。

目标是统一执行 provider 的环境绑定、stdout/stderr、退出码、结构化结果、超时、取消和清理语义。Workspace 声明资源和可用能力，执行 provider 兑现文件、网络、进程与资源限制，Actor 选择调用；工具执行不能反向占有资源系统。

验收覆盖正常结束、工具异常、启动失败、超时、取消、子进程清理和不同 Workspace 环境隔离。受信任执行、工具 API 限制和 OS 级强隔离必须明确区分：工作目录、prompt 或 Python 包装层不足以限制任意代码。若第一版无法实施相应隔离，不接纳要求该隔离等级的任务。完整不可信代码沙箱保留独立的实现与验证门槛。

**完整不可信代码沙箱**：按实际威胁与负载选择隔离机制，验证宿主文件、网络、进程、资源配额、子进程终止和跨 Workspace 逃逸边界。不以目录配置或工具包装宣称强隔离。

### 5.5 未排期的其他工作包

| 事项 | 状态与进入条件 | 分类入口 |
|:---|:---|:---|
| 复合意图分解 | Unscheduled；先有 C0 指标和真实样本，再冻结 composite envelope | [Idea](./ideas/composite-intent-decomposition.md) |
| 自定义入口拦截规则 | Unscheduled；需真实外部接入场景、配置所有者和验收样本 | 正式 Idea/Plan 待建立 |
| 领域持久化与恢复 | Unscheduled 剩余范围；导入、Research、桌面化按各自承诺提取最小切片，通用 checkpoint/saga 平台不提前展开 | [Durability Governance](./governance/reliability/durability-and-recovery.md) |
| 领域幂等与 reconciliation | Unscheduled 剩余范围；新的提交/导入入口先明确所需业务幂等，再独立设计存量领域操作补偿 | [Idempotency Governance](./governance/reliability/idempotency-and-retry.md) |
| 数据模型可变性治理 | Unscheduled；按具体 DTO/聚合风险分批，不混入完整 PendingAtom 重写 | [Mutability Governance](./governance/data-model/mutability.md) |
| 跨用户认证及权限治理 | Unscheduled 的完整范围；外部 connector 先落实所需身份映射，不能把 scope 字段当作认证 | [Identity Governance](./governance/security/identity-and-execution-safety.md)；设计方向见[身份与访问体系 Idea](./ideas/identity-and-access-model.md) |
| 通用长期 workflow / DAG | Unscheduled；Research 先验证最小真实流程，再决定是否提炼通用机制 | [VISION](./VISION.md)；独立 Plan 待建立 |
| Conversation Branching | Unscheduled；依赖来源、生命周期和真实编辑需求 | [Raw Evidence Idea](./ideas/PatchouliPageFoldingRawEvidenceDesign.md#8-与编辑回退和分支的关系)第 8 节；[Issue #6](https://github.com/artemis03976/HiveMemory/issues/6)（对话气泡编辑与回档） |
| 配置热更新 | 接入登记与操作白名单的运行时编辑已完成（2026-11-09，含前端开关、接入凭据的生成与轮换、principal 的 Agent 限定）；其他配置文件（主配置）的持久化与热更新仍为 Unscheduled | [访问登记的运行时管理（归档）](./archive/plans/runtime-access-registration.md)；[认证与访问登记](./workspace/authentication.md#2-两类访问登记) |
| 其他历史存储转换 | Unscheduled；若发现仍有未迁移存量，先列证据再建迁移 Plan | 已完成的 [V1 迁移历史](./archive/plans/v0.6.2-v1-memory-legacy-migration.md)不重新列为待办 |

### 5.6 controller 模式的外部 harness 接入

状态：Candidate，不绑定版本。2026-10-20 曾定为 v0.7.1 的版本目标，同日改为 plugin 模式（[v0.7.1 收尾审计](./archive/plans/v0.7.1-release-closeout-audit.md)）；首份计划的方向（首个外部驱动的纯文本闭环）随之撤回，计划未建立。步骤分析与阻塞问题保留在[外部 Actor Idea](./ideas/external-actor-registration-and-runtime-access.md#controller-模式的步骤分析退回候选方向)，Codex、Claude Code、pi 与 ACP 的接口资料见[接口调研快照](./research/harness-interface-survey-snapshot.md)（2026-10-20，模型实测待验证）；重新排期时先复核其中的未知项与待实测项。

controller 模式把 Codex、Claude Code、pi 一类成熟的 agent harness 作为执行环境与 CPU 类型纳入兼容范围：用户在对话开始前分别选择 agent 与执行环境（[执行环境 Idea](./ideas/execution-environment.md)前提 9、10；外部 Actor Idea E-6a），外部 harness 作为执行环境运行所选 agent，请求经唯一注册入口登记为任务进程，处理后交给该 harness 执行。它建立在 v0.7.0 的任务进程表与注册入口之上，消费外部 Actor 的接入与运行时访问契约（原计划 B），不强依赖 HiveMemory 本地沙箱；外部 harness 继续使用自己的工具和执行环境。采用 ACP 一类的通用协议还是各 harness 的 SDK 或非交互 CLI，见 Idea 的 E-2a。

验收必须形成真实闭环：请求登记为任务进程并由外部 harness 执行；外部 harness 经能力层检索已有记忆并实际用于任务，写入/修订意图经能力层提交；进程的交互记录进入记忆库（去向见 [任务进程](./workspace/task-process.md)）；下一次会话能召回新形成的知识并定位来源。取消、失败与进程回收按外部执行单元的进程问题（[任务进程 Idea](./ideas/task-process-table-and-registration-entry.md#q-8-外部执行单元的进程) Q-8）的决定呈现。仅转发请求、仅采集日志、仅发送事件或仅返回 memory context 均不足以证明 Actor 接入完成。

controller 模式下 HiveMemory 控制任务边界（actor 选择、输入、生命周期与取消、执行轨迹回流），不控制外部 harness 内部的 loop、工具与上下文压缩；外部 harness 默认保有自己的 prompt history，HiveMemory 不隐式接管上下文压缩。

## 6. 长期方向与评估

长期取舍遵循 [VISION](./VISION.md)：先证明开放记忆基础设施的实际价值，再验证 Alice 的 memory-native 运行能否带来额外收益。模型与 harness 的更换不应迫使用户重建长期知识资产；Actor 可替换性也不表示其执行过程或能力可以完全等价迁移。

评估逐步使用同一任务和记忆库比较无记忆、普通检索、外部 harness + Patchouli、Alice + Patchouli。记录跨会话复用、来源可查、错误与重复记忆、用户修正成本、执行可靠性、延迟和资源开销。新增功能、抽象层数和目录重排本身不作为成功指标。

## 7. 路线图维护规则

1. 只有 Git tag 对应版本标为 Released；本次重排不修改规范代码版本或构建产物。
2. 当前版本的功能在 Roadmap 写明目标、方向与验收方向；计划只在能够立即执行时绑定当前版本建立，不能立即实施的方向留在 Idea 候选池，不预先分配版本号或先后（owner，2026-10-30）；详细设计只维护一份。
3. 仅已完整完成实现、测试、迁移与审查的内容晋升到当前事实文档，并归档 Plan。
4. 部分落地保持 Partially Landed，已完成 Todo 不重新排期，Deferred 项只有满足触发条件才启动。
5. 排期变化同步 Plan 的 target、文件名、索引和反向引用；已完成历史文档保留原版本，当前路线提供迁移映射。
6. PROJECT、Architecture、System 和子系统索引的行为说明按文档治理收口门禁更新；单纯调整候选排期不提前向事实层推广设计。
7. 定期依据真实使用、导入样本和运行故障复核候选方向与下一个版本的功能；范围变化显式记录，不把前置能力缺口隐藏在上层功能中。
