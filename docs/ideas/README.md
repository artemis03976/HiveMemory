---
title: Ideas
status: current
owner: project
scope: uncommitted-exploration
last_reviewed: 2026-10-09
---

# Ideas

本目录保存尚未形成项目承诺的开放设想、研究假设和候选方向。Idea 可以解释一项机会为什么值得探索，也可以保留暂时无法收敛的多种路径；它不能用未来类名、配置草案或阶段编号替代当前设计，更不能因为已有少量基础设施就被解释为功能已经排期。

## 分类

Idea 以 frontmatter 的 `horizon` 字段分为三类，规则见[文档治理规范](../DOCUMENTATION.md)第 8.4 节。类别变化只修改该字段与本索引，不移动文件。

| horizon | 含义 |
|:---|:---|
| `current` | 当前版本方向的设计讨论，直接为下一份 Plan 供料；以 `serves_version` 写明服务的版本，可以记录 owner 已作出的决定 |
| `candidate` | 问题具体、尚未排期的功能或技术方向 |
| `long-term` | 长期方向、研究假设或概念性指导，不绑定版本 |

## 当前版本的设计讨论（`current`）

服务于 v0.7.0，版本内的计划顺序见 [Plans 导航](../plans/README.md#v070)。

| Idea | 当前已经具备的基础 | 仍需验证的核心问题 |
|:---|:---|:---|
| [Workspace 网络与任务进程架构](./workspace-network-task-process-architecture.md) | 包分层（第二部分，已实施）；两阶段认证与两阶段授权（A1 访问边界返工）；统一交互提交队列；写入意图第 1 步与读取缓存失效（2026-10-07 已实施） | 已完成（第 5、9、15 节）：迁移方式与 v0.7.0 范围（M-1–M-3、M-5–M-7，含四条版本目标）、请求方分类、Import Bus 移出现有系统；system 包边界 D-1–D-9；管理员直接通道（方案 C）、每次请求重新校验身份、System 层面的 API 不经 workspace、Profile 权限并入 operation 控制（2026-10-06 改为单独建立计划）、Alice 的能力层调用迁移（2026-10-06 按先后重新划分为读取缓存失效、写入意图迁移第 1 步、收窄后的 Alice 迁移与执行线程层）、访问登记与 context 生命周期、读取按视角区分、阶段授权、取消与进程控制、写入意图迁移第 1 步先于 Alice 迁移（P-11）。第四部分于 2026-10-06 拆出为[执行单元、执行线程与执行环境](./execution-unit-thread-and-environment.md)。未完成（第 6、10、16 节）：M-4；Import Bus 的 Q-11–Q-13（不在 v0.7.0）；D-8a、engines 的既有向上导入、core 的内容整理；P-1b/c、P-3、P-4a、P-5（P-5a 并入执行单元 Idea T-3）、P-7（其余控制操作）、P-8、P-9a/c/d/f、P-10a、P-12b（引用记录）；P-11a 缓存维护与 P-12a 中立引用解析已于 2026-10-07 实施 |
| [执行单元、执行线程与执行环境](./execution-unit-thread-and-environment.md) | 2026-10-06 由总 Idea 第四部分拆出；CPU 端口与 `AliceCPU`、Alice 的 `RunSession` frame 注册表；T-4a/T-4c 主线程操作通道（2026-10-07 已实施，通道形态已由 T-9 取代） | 前提（2026-10-05、2026-10-06）：执行单元 = CPU = CPU 驱动 + harness 实例，其中的 actor 是执行线程，执行单元为线程的工具调用补全发起者；外部操作请求经操作适配器归一化为操作请求、由 CPU 经 workspace 的操作入口调用能力层，任务请求经请求入口进入注册入口；agent 与执行环境解耦，用户在对话开始前分别选择，选择执行环境不是 workspace 的能力。已完成：T-1（子线程身份在派生时由派生调用确定、派生到达进程，外部执行单元不识别子线程）、T-2（每个线程一份 context）、T-6（只有接入侧面的参与者不称 CPU）、T-9（操作适配器只产出操作请求，与能力方法、`WorkspaceOperation` 同一套操作列表；workspace 单一操作入口，每个线程一份执行凭据）及随之确定的 T-1b。未完成：T-1a（派生到达进程的途径）、T-2a（子线程 context 的失效时点）、T-3（派生授权）、T-4b 及子线程独立身份的回调通道与进程绑定（T-4a/T-4c 主线程部分已实施）、T-5（目标 workspace）；T-7、T-8（执行环境选择）不在 v0.7.0 |
| [身份与访问体系](./identity-and-access-model.md) | 第一、二批均于 2026-10-04 实施归档：密封的访问 context、两阶段认证与两阶段授权；Patchouli 内部和后台记录分开携带归属与发起者；SETTLE 使用 system；actor 不含 session_id。当前事实见 [Workspace 架构](../architecture/workspace.md)，本地阶段证据见 [第二批归档计划](../archive/plans/v0.7.0-identity-access-batch-2.md) | I-1–I-11（含 I-6a）均已决定并实施，I-9 保留 CPU 过渡身份，随 Alice 能力层调用迁移删除；[WorkspaceAsset 内部拆分](../todo/workspace-asset-ownership-identity-split.md)由用户明确暂缓；受限跨 Workspace 访问不在 v0.7.0 |
| [任务进程表与任务请求唯一注册入口](./task-process-table-and-registration-entry.md) | v0.7.0 内的五个批次已实施归档：`workspace.process` 的注册入口、进程表、执行器 `TaskProcessRunner` 与 CPU 分配，CPU 端口，命令只解析不执行 | 已完成（1.1、1.2、4.1）：请求方分类、任务进程的结构、Q-1–Q-6、Q-7（记忆任务不暴露给 Agent）、Q-9、Q-10、Q-14–Q-16；其中 Topic 按需创建、写入意图实时提交与“接纳后即退出”归其他方向，尚未实施。未完成（4.2）：Q-3b 传输入口、Q-5a 命令的运行位置（随命令系统后置）、Q-5b、Q-7 进程表是否收录后台任务、Q-8 外部 CPU 的进程（v0.7.1） |
| [外部会话消息的接收与 Topic 投影](./external-session-and-topic-projection.md) | `InteractionPayload`、`TurnEvent`、`InteractionApplyRecord` 与 Passive turn buffer | 原 A3 计划退回，v0.7.0 内完成、Alice 为第一个使用者：Topic 体系如何接收外部 Actor 的会话消息；任务进程 Idea Q-9 已选 B：实际使用的对话上下文由 `ConversationSession` 提供，Topic 与 Gateway 话题路由只为记忆生成服务；会话操作（新建、恢复、压缩）、workspace 共享 Topic 池与前端回归 session 模型已决定（前端与新建、恢复在 v0.7.0，Alice 的压缩约在 v0.7.1）；模型字段与 Topic 生命周期事项待定 |
| [写入意图（PendingAtom）体系的迁移](./pending-intent-migration.md) | workspace `WriteIntentRegistry`、共同引用解析与读取缓存失效（第 1 步于 2026-10-07 实施，[历史计划](../archive/plans/v0.7.0-intent-registry-and-read-cache.md)）；`PendingAtomMaterializeTask` 与主动生成仍在 finalize 交接 | 原 A4 计划退回，v0.7.0 内分两步完成：第 1 步已迁出 Alice 登记与原子解析、提供能力层提交/全 Workspace 回读并接结算事件；第 2 步的实时物化派发、移除 interaction 隐式物化字段、Topic 全资料、结算后句柄寿命及未送达对账仍待实施或决定 |

## 候选方向（`candidate`）

| Idea | 当前已经具备的基础 | 仍需验证的核心问题 |
|:---|:---|:---|
| [外部 Actor 的接入登记与运行时访问](./external-actor-registration-and-runtime-access.md) | 两类接入登记（配置装载）、统一认证网关（无生产调用方）、ingest HTTP 入口与能力层 | 原计划 B 退回：外部 Actor 分为 plugin 与 controller 两种接入模式（controller 模式为 v0.7.1 的首个真实接入，plugin 模式在其后的 v0.7.x 完善）；外部 Actor 如何登记进系统、运行时如何访问系统，以及操作适配器的接口边界（E-1 已决定维持配置装载；E-3a 已决定以归一化定义操作适配器；E-5 已决定外部执行单元不识别子线程、所有 actor 共享主线程的 context；E-6a 已决定外部 harness 是执行环境、与 agent 分别选择；E-2、E-3b、E-4、E-6b 待决） |
| [长时间运行 Agent 的 Turn 内上下文折叠](./long-running-agent-intra-turn-context-folding.md) | TurnEvent、LogicalBlock、Agent runtime、topic Page Folding 与 passive event ingress | 如何在一个 turn 内多次 compact，同时保持执行连续性、记忆生成语义、原始证据和跨入口契约 |
| [Page Folding Raw Evidence](./PatchouliPageFoldingRawEvidenceDesign.md) | `state_summary` 折叠、InteractionArtifact 与异步 Generation | 后台 Topic 累计消息过长时，如何保证输入记忆生成的资料不会导致上下文爆炸；保存原始折叠页是否值得引入新的耐久性、隐私与去重成本（2026-10-01 定位为后台问题，前台会话压缩归 CPU） |
| [Chat Run 生命周期后续候选](./chat-run-lifecycle-follow-ups.md) | 已完成的取消最小闭环、SSE 与 run registry | 哪些候选具有独立收益，是否值得分别立项，而不是实施一次性大重构 |
| [复合意图分解](./composite-intent-decomposition.md) | `COMPOSITE` 分类信号与私有 `sub_intents` | 真实样本能否证明单主意图路径存在稳定缺口，以及 envelope、消费所有权与 fallback 如何冻结 |
| [全项目时间使用统一规范](./project-wide-time-semantics-standardization.md) | A2-P 已统一 Memory 域时间（UTC-aware、四字段职责、局部 now 注入）；附录 D 记录的域外调用清单 | monotonic 域统一、topic 域字段分类、耗时统计与展示 allowlist 的治理路径与 owner |

## 长期方向与概念（`long-term`）

| Idea | 当前已经具备的基础 | 仍需验证的核心问题 |
|:---|:---|:---|
| [AE2 与 HiveMemory 的架构同构性](./ae2-hivememory-architecture-analogy.md) | Patchouli 存储平面、Alice Frame/编排、MTP 能力契约与当前可见性过滤 | Workspace/软件子网、Harness-to-Harness、Workflow Memory、Job Graph、Mount/Bridge、AIOS 资源抽象与隔离执行是否值得进入真实验证 |
| [MaaT 与 Skill/Plugin 边界的资产消费模型](./memory-as-tool-skill-plugin-boundary.md) | `MemoryCompiler` target 体系、MTP `RUN` 两层分发、MemoryAtom/Artifact 分离 | 如何让 MTP RUN 编译为结构化执行意图，证明相对外部 harness + Patchouli 的结构性优势，以及 CODE_SNIPPET 的迁移边界、ToolInvocationIR 与共享 MemoryCompiler IR 的关系、执行证据回流与终端实践的晋升机制 |
| [生命力分数长期演进](./VitalityScoringLongTermEvolutionIdeas.md) | 当前 vitality 公式、强化事件、gardening 与显式 archive/revive | 哪些新状态能由真实使用数据校准，而不是继续叠加启发式参数 |
| [Agent 轨迹与多 Agent TDA](./TDA_Agent_Research_Ideas.md) | TurnEvent、AgentAction、RuntimeEvent 与单层 CALL | 拓扑特征能否比普通计数/规则更稳定地解释成功、成本和失败类型 |
| [Memory-Centric Agent TDA](./TDA_Memory_Centric_Agent_Ideas.md) | MemoryAtom 关系预留、检索信号与来源/版本 Artifact | 多视图记忆图能否形成可重复、可行动且优于平面检索的信号 |

本索引于 2026-09-27 按 `horizon` 重新分区。这里的材料均不代表路线图承诺，也不能作为当前能力引用；`current` 类中记录的 owner 决定只约束后续 Plan 的编写，不等于已经实现。已实现或不再跟踪的 Idea 移入 [Archived Ideas](../archive/ideas/README.md)，例如已随 v0.6.2 实现的 Chat Attachments 初步设计备忘。既有 Ideas 的历史分类依据见[文档迁移最终收口审计](../archive/plans/documentation-migration-finalization-audit.md)。

## 升级规则

Idea 进入实施前至少需要：

1. 有来自真实运行、可复现实验或明确产品场景的问题证据；
2. 能说明目标、非目标、受影响的所有权与稳定契约；
3. 能用基线、指标和失败样本证明方案价值，而不只是证明可以实现；
4. 有分阶段迁移、兼容、隐私/耐久性和回滚考虑；
5. 建立独立 Plan，并列出完成后必须更新的当前文档。

在这些条件满足前，Ideas 不进入 Roadmap，也不应拆成没有独立完成语义的 Todo。
