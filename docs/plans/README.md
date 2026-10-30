---
title: Plans
status: current
owner: project
scope: implementation-plan-navigation-by-version
last_reviewed: 2026-10-30
---

# Plans

本目录存放已经绑定版本或里程碑、能够独立实施和验收，但尚未成为当前事实的计划。本索引是**版本内的计划导航**：按版本列出计划的状态、顺序与链接，不记录设计决定。设计讨论见 [Ideas](../ideas/README.md)，版本规划摘要见 [ROADMAP](../ROADMAP.md)，已完成的计划见 [Archived Plans](../archive/plans/README.md)。

## v0.7.1

版本目标已决定（owner，2026-10-20）并已完成（2026-10-29，对话回流与记忆访问两份计划均已归档）：plugin 模式的真实外部 harness 接入，首个接入的 harness 为 Claude Code，按小步分别建立计划；同日先定的 controller 模式目标退回候选方向，其首份计划方向（首个外部驱动的纯文本闭环）随之撤回。目标结果与验收方向见 [ROADMAP](../ROADMAP.md) 第 4.4 节。

| 顺序 | 方向 | 状态 | 入口 |
|:---:|:---|:---|:---|
| 1 | [Import Bus 重设计：以 Claude Code 对话回流为第一条真实路径](../archive/plans/v0.7.1-import-bus-redesign.md) | Archived（2026-10-21）；当前事实见 [Import Bus](../system/import-bus.md)；真实使用验收随日常使用持续观察（[Todo](../todo/import-bus-real-use-observation.md)） | —— |
| 2 | [记忆访问：Claude Code 经 MCP 工具检索与读取记忆](../archive/plans/v0.7.1-memory-access-mcp.md) | Archived（2026-10-29）；当前事实见 [MCP 记忆访问](../contracts/mcp.md)；检索与读取的真实使用样本记入同一观察 [Todo](../todo/import-bus-real-use-observation.md) | —— |
| — | [推理模型的空闲卸载与按需加载](../archive/plans/v0.7.1-model-idle-unload.md)（plugin 模式下后端常驻时的内存开销） | Archived（2026-10-29）；当前事实见 [Patchouli](../patchouli/README.md#41-模型驻留与空闲卸载) | —— |
| 3 | [工具解析链路重写：`TurnEvent` 作为唯一的交互原子事件](../archive/plans/v0.7.1-tool-call-parsing-chain.md) | Archived（2026-10-30）；当前事实见[数据模型](../architecture/data-model.md)、[感知与短期话题](../patchouli/perception.md)与 [Patchouli 生成](../patchouli/generation.md)；并入的原 Todo《导入的 MCP 读取在轨迹中显示 alias》随之解决 | —— |

## v0.7.0

已发布（Git tag `v0.7.0`，2026-10-20）。范围（owner，2026-09-27）是完全完成项目向新架构的演进，使 workspace 体系在项目架构中稳定存在，首条迁移流程为 Alice 的 chat 链路；版本目标的逐条判定、各方向与归档计划的核对、收尾时搁置的事项和验证结果见[收尾审计](../archive/plans/v0.7.0-release-closeout-audit.md)，版本摘要见 [ROADMAP](../ROADMAP.md) 第 3 节。版本内按小批量逐次实施，同一方向同时只有一份生效计划，下表按方向列出已归档的计划。

| 方向 | 归档计划 |
|:---|:---|
| 任务进程 | [落位与进程标识](../archive/plans/v0.7.0-task-process-table.md)、[prepare 拆分与 CPU 输入清单](../archive/plans/v0.7.0-task-process-prepare-split.md)、[结算阶段的中立输入](../archive/plans/v0.7.0-task-process-finalize-neutral-input.md)、[CPU 端口与测试 CPU](../archive/plans/v0.7.0-task-process-cpu-port.md)、[命令只解析不执行](../archive/plans/v0.7.0-task-process-command-parse-only.md)、[任务进程拥有执行流](../archive/plans/v0.7.0-task-process-owned-execution.md) |
| 访问边界与身份 | [A1 访问边界](../archive/plans/v0.7.0-a1-workspace-access-boundary.md)、[A1 访问边界返工](../archive/plans/v0.7.0-a1-access-boundary-rework.md)、[身份与访问体系第二批](../archive/plans/v0.7.0-identity-access-batch-2.md) |
| 能力层 | [Alice 的能力层调用迁移](../archive/plans/v0.7.0-alice-capability-migration.md)、[能力层分层](../archive/plans/v0.7.0-capability-surface-split.md)、[Agent 操作白名单](../archive/plans/v0.7.0-agent-operation-whitelist.md) |
| 执行线程 | [线程 context 取代执行凭据](../archive/plans/v0.7.0-thread-context.md)、[执行线程层](../archive/plans/v0.7.0-execution-thread-layer.md)、[执行线程的正式定义](../archive/plans/v0.7.0-execution-thread-definition.md) |
| 写入意图与读取视图 | [登记迁入 workspace 与读取缓存失效](../archive/plans/v0.7.0-intent-registry-and-read-cache.md)、[写入意图实时派发](../archive/plans/v0.7.0-intent-inflight-dispatch.md) |
| 会话 | [会话与 Alice 迁移](../archive/plans/v0.7.0-conversation-session.md)、[Alice 会话压缩](../archive/plans/v0.7.0-alice-session-compaction.md) |
| 记忆模型 | [A2-P 记忆内容版本与 Lifecycle](../archive/plans/v0.7.0-a2-pre-memory-version-and-lifecycle.md)、[时间语义与可控时钟](../archive/plans/v0.7.0-time-semantics-and-controllable-clock.md) |
| 结构收敛 | [Alice 与 agent_runtime 结构重构](../archive/plans/v0.7.0-alice-agent-runtime-restructure.md)、[提示词编译前移](../archive/plans/v0.7.0-cpu-prompt-compilation.md)、[异常体系收敛](../archive/plans/v0.7.0-error-model-convergence.md)、[LLM 访问收敛](../archive/plans/v0.7.0-llm-access-port.md)、[总线路由类型化与 mypy 门禁](../archive/plans/v0.7.0-typed-bus-routes.md)、[代码收敛清单](../archive/todo/v0.7.0-code-convergence.md) |

版本期间删除的计划（A2 未完成部分、A5、A6、WRX-0 清单、计划 A 协调入口与边界宪章）删除前的最后版本见 commit `dda9d9d`，处置记录见 commit `6add90f` 版本的《Workspace 网络与任务进程架构》Idea 第 5.5 节。

## 未绑定版本

| 计划 | 状态 | 说明 |
|:---|:---:|:---|
| 暂无 | — | 原占位计划“话题折叠、Actor 上下文与原始证据统一改造”于 2026-10-01 退回 Idea 后删除（删除前最后版本见 commit `74b5056`）：后台部分并入 [Page Folding Raw Evidence](../ideas/PatchouliPageFoldingRawEvidenceDesign.md)，前台部分并入[长时间运行 Agent 的 Turn 内上下文折叠](../ideas/long-running-agent-intra-turn-context-folding.md) |

## 已完成的计划

已完成或被替代的计划移入 [Archived Plans](../archive/plans/README.md)，那里按时间倒序登记全部归档计划及其事实入口；各版本的逐项核对见版本收尾审计（[v0.7.0](../archive/plans/v0.7.0-release-closeout-audit.md)、[v0.6.2](../archive/plans/v0.6.2-release-closeout-audit.md)）。

## 准入规则

- 必须填写明确 `target` 或在标题/范围中绑定当前版本；
- 必须具有独立目标、非目标、迁移方式、测试和验收出口；
- 计划之间的关系满足[文档治理规范](../DOCUMENTATION.md)第 8.3 节的三条约束：同一目标方向只有一份生效计划、不设总分结构；设计依据只来自事实文档、代码、ADR、已归档计划、已冻结的 Research 报告与背景 Idea；计划之间只有整份粒度的顺序依赖；
- 本索引只做导航，不记录设计决定；
- 未排期治理主题进入 [Governance](../governance/README.md)；尚需验证价值、样本或所有权的候选进入 [Ideas](../ideas/README.md)；范围较小的缺陷和技术债进入 [Todo](../todo/README.md)；
- 已完成或被替代的实施稿进入 [Archived Plans](../archive/plans/README.md)；未完成实施即作废的计划直接删除，方向保留但前提改变的未实施计划退回 Idea，入链与遗留事项按文档治理规范第 8.4、10 节处理。
