---
title: Archived Todo
status: current
owner: project
scope: completed-todo-history
last_reviewed: 2026-11-06
---

# Archived Todo

本目录保存已完成的修复与技术债处置历史；当前行为从对应事实文档进入，活动待办见 [Todo](../../todo/README.md)。每条登记一行：归档日期、所在的 PR 与替代它的当前文档；处置过程见各事项自身。

- [WorkspaceAsset 内部归属与操作身份拆分](./workspace-asset-ownership-identity-split.md)：2026-11-06，随 [WorkspaceAsset 的归属拆分](../plans/workspace-asset-ownership-split.md)完成，PR #185；[WorkspaceAsset](../../workspace/assets.md#3-身份与归属)
- [本机 HTTP 管理接口不证明调用方](./local-http-management-access.md)：2026-11-05，PR #182；[认证与访问登记](../../workspace/authentication.md#1-两阶段认证)、[配置指南](../../help/configuration.md#51-访问登记)
- [运行时事件没有持久的记录](./runtime-events-not-persisted.md)：2026-11-03，PR #175；[可观测性](../../components/observability.md#51-事件记录文件)、[System 配置](../../system/configuration.md)
- [维护任务空转也发事件](./maintenance-event-noise.md)：2026-11-03，PR #174；[运行时机制](../../components/runtime-and-bus.md#22-运行结果与事件)、[可观测性](../../components/observability.md)
- [部分失败事件没有稳定原因码](./failure-event-reason-codes.md)：2026-11-03，PR #173；[事件契约](../../contracts/routes-and-events.md#44-失败与降级观测)、[可观测性](../../components/observability.md)
- RuntimeEvent 生产端迁移后续：2026-11-03 转入[日志与运行时事件](../../governance/observability/logs-and-events.md)治理主题的工作包 W6，文件删除，删除前最后版本在 commit `afbd7075`
- [Memory 管理页列表不全与检索](./frontend-memory-library-listing.md)：2026-11-02，PR #171；[前端管理页面](../../frontend/management-views.md#11-当前能力)、[MemoryLibrary](../../patchouli/memory-library.md)
- [模型级密钥覆盖写入受版本控制的 models.yaml](./model-registry-api-key-override.md)：2026-11-01，PR #166；[System 配置与注册表](../../system/configuration.md#4-配置所有权与兼容边界)、[配置指南](../../help/configuration.md#4-配置-model-registry)
- [MTP READ 截断正式记忆的正文](./mtp-read-content-truncation.md)：2026-11-01，PR #165；[MemoryCompiler](../../patchouli/memory-compiler.md#4-envelope-targets)、[MTP 契约](../../contracts/mtp.md#32-read)
- [生成组件测试的分类与重复覆盖](./generation-component-test-classification.md)：2026-11-01，PR #164；[Patchouli 生成](../../patchouli/generation.md#4-查重与合并)
- [主配置保存把密钥写入受版本控制的 YAML](./config-secrets-written-to-yaml.md)：2026-11-01，PR #159；[System 配置与注册表](../../system/configuration.md)、[配置指南](../../help/configuration.md#5-主配置与生效边界)
- [生成材料的筛选与截断](./generation-material-filtering-and-truncation.md)：2026-10-31，PR #150；[Patchouli 生成](../../patchouli/generation.md)、[感知与短期话题](../../patchouli/perception.md)
- [Import Bus 真实使用观察](./import-bus-real-use-observation.md)：2026-10-30，PR #148，owner 决定不再逐项跟踪；[生成查重的决策信号](../../todo/generation-dedup-decision-signal.md)
- [任务进程中无生产使用方的代码与可合并小项](./task-process-unused-paths.md)：2026-10-30，PR #145；[任务进程](../../workspace/task-process.md)
- [记忆编辑可以给非 Profile 记忆写入 agent_config](./memory-edit-agent-config-type-check.md)：2026-10-27，PR #140；[Patchouli 生成](../../patchouli/generation.md)
- [话题折叠计入交互提交的处理时限](./topic-fold-in-interaction-apply.md)：2026-10-27，PR #139；[感知与短期话题](../../patchouli/perception.md)
- [模型推理移出事件循环](./inference-off-event-loop.md)：2026-10-27，PR #138；[Patchouli](../../patchouli/README.md)
- [日志配置段的级别、格式与输出没有生效](./logging-config-not-applied.md)：2026-10-19，PR #135；[System 配置](../../system/configuration.md)
- [构造根配置会改写进程默认语言](./config-construction-resets-default-language.md)：2026-10-19，PR #135；[i18n](../../system/i18n.md)
- [v0.7.0 代码收敛清单](./v0.7.0-code-convergence.md)：2026-10-19，PR #134；[AGENTS.md](../../../AGENTS.md)
- [Profile 管理列表的可见性语义](./agent-profile-management-list-visibility.md)：2026-10-16，PR #123；[System 应用服务](../../system/application-services.md)
- [测试套件冗余收敛后续](./test-suite-redundancy-follow-ups.md)：2026-10-16，PR #123；[测试设计规范](../../governance/testing/test-design-standards.md)
- [kernel 遗留命名清理](./kernel-legacy-naming.md)：2026-10-12，PR #118；[MTP Runtime](../../alice/mtp-runtime.md)、[Chat 视图](../../frontend/chat-view.md)
- [检索分数阈值与候选截取收敛](./retrieval-hybrid-threshold-with-reranker.md)：2026-10-12，PR #117；[Patchouli 检索](../../patchouli/retrieval.md)
- [TaskProcess 容器的职责与依赖方向](./task-process-container-ownership.md)：2026-10-04，PR #106；[任务进程](../../workspace/task-process.md)
- [WorkspaceAsset 上传的认证上下文与 scope 不一致](./workspace-asset-upload-access-scope-mismatch.md)：2026-10-04，PR #106；[能力层](../../workspace/capability.md#1-三个调用面)
- [Memory Alias 重名缺陷](./memory-alias-uniqueness.md)：2026-09-27，PR #103；[MemoryLibrary](../../patchouli/memory-library.md)、[alias 后续 Todo](../../todo/memory-alias-follow-ups.md)
- [AgentProfile 模型演进](./agent-profile-model-evolution.md)：2026-09-27，PR #103；[Alice](../../alice/README.md)、[第五代架构 Idea](../../ideas/fifth-generation-architecture.md)
- [MTP 缓存命中作用域重验](./mtp-cache-scope-revalidation.md)：2026-09-14，PR #99，继续作为隔离回归基线；[MTP 契约](../../contracts/mtp.md)
- [记忆来源与作者语义](./memory-provenance-vs-authorship.md)：2026-09-06，PR #96；[Artifacts](../../patchouli/artifacts.md#25-来源与归属语义)
- [Topic 关闭时的失败隔离](./topic-shutdown-per-topic-failure-isolation.md)：2026-09-05，PR #95；[感知与短期话题](../../patchouli/perception.md)
- [ShortTermMemoryStore 边界收敛](./short-term-memory-store-boundary-cleanup.md)：2026-09-02，PR #95；[MemoryLibrary](../../patchouli/memory-library.md)

## 升级为计划、并入计划或删除的事项

这些事项没有归档文件，历史由版本控制保留：

- “Memory Garden 接入真实语义检索”：2026-11-02 并入 [Memory 管理页列表不全与检索](./frontend-memory-library-listing.md)（已归档），删除前的最后版本见 commit `46c82c8c`；
- “Gateway 与 Librarian 的旧式直配路径”：2026-11-01 升级为[模型选择统一为 ModelSelection，组件模型只经注册表解析](../plans/llm-config-retirement.md)（已归档），删除前的最后版本见 commit `190accd2`；
- “导入的 MCP 读取在轨迹中显示 alias”：2026-10-30 并入[工具解析链路重写计划](../plans/v0.7.1-tool-call-parsing-chain.md)，删除前的最后版本见 commit `051625d9`；
- “总线路由的类型化与静态签名检查”：2026-10-12 升级为[总线路由类型化与 mypy 门禁](../plans/v0.7.0-typed-bus-routes.md)，删除前的最后版本见 commit `8faa0b7`；
- “A1 访问边界返工”：2026-10-02 升级为[同名计划](../plans/v0.7.0-a1-access-boundary-rework.md)；
- “Page Folding 跨入口上下文与证据后续技术债”：2026-10-01 按前台与后台拆分后删除，删除前的最后版本见 commit `74b5056`；后台缺口并入 [Page Folding Raw Evidence Idea](../../ideas/PatchouliPageFoldingRawEvidenceDesign.md)，前台缺口并入 [Turn 内上下文折叠 Idea](../../ideas/long-running-agent-intra-turn-context-folding.md)，通用压缩集中到[短期记忆与上下文压缩](../../ideas/short-term-memory-and-context-compaction.md)。
