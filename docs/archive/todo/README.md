---
title: Archived Todo
status: current
owner: project
scope: completed-todo-history
last_reviewed: 2026-10-27
---

# Archived Todo

本目录保存已完成的修复与技术债处置历史；当前行为从对应事实文档进入，活动待办见 [Todo](../../todo/README.md)。

- [记忆编辑可以给非 Profile 记忆写入 agent_config](./memory-edit-agent-config-type-check.md)：2026-10-27 在 `fix/memory-edit-agent-config-type-check` 完成，Familiar 提交边界的类型约束拒绝非 `AGENT_PROFILE` 原子带 `agent_config`（`InvalidMemoryFieldError`，HTTP 422），检查在写版本记录之前；记忆与 Profile 继续共用编辑入口，不拆 Profile 专用编辑，存量数据无需清理；当前事实见 [Patchouli 生成](../../patchouli/generation.md)第 5 节。
- [话题折叠计入交互提交的处理时限](./topic-fold-in-interaction-apply.md)：2026-10-27 在 `fix/topic-fold-in-interaction-apply` 完成，交互写入话题即完成应用，折叠作为提交队列中与该话题提交共用排序键的独立工作项，折叠期间同一话题的后续提交在队列中等待、不计入处理时限；LLM relay 摘要新增独立时限，超时与其他回退都接在已有摘要之后；写入成为应用中最后一个可能失败的步骤，apply journal 随之移除，幂等收敛到提交队列；当前事实见[感知与短期话题](../../patchouli/perception.md)第 5 节。
- [模型推理移出事件循环](./inference-off-event-loop.md)：2026-10-27 在 `feat/model-memory-footprint` 完成，嵌入与重排模型各持有一个推理线程，懒加载与推理不再在事件循环上执行，同一模型的调用逐个进行，Patchouli 停止时在生成 lane 之后释放线程；onnxruntime 创建会话时持有 GIL，模型加载期间事件循环仍会停顿；当前事实见 [Patchouli](../../patchouli/README.md) 第 4 节。
- [日志配置段的级别、格式与输出没有生效](./logging-config-not-applied.md)：2026-10-19 在 v0.7.0 收尾分支修复，server 生命周期在装配系统前按 `logging` 段设置根 logger 的级别、格式、控制台与文件输出，两种启动方式一致，只替换入口自己装上的 handler；当前事实见 [System 配置](../../system/configuration.md)第 4 节。
- [构造根配置会改写进程默认语言](./config-construction-resets-default-language.md)：2026-10-19 在 v0.7.0 收尾分支修复，删除配置模型上的同步校验器，进程默认语言只在组合根装配运行配置时设置，配置路由构造的临时对象不再改变它；当前事实见 [i18n](../../system/i18n.md)第 2 节。
- [v0.7.0 代码收敛清单](./v0.7.0-code-convergence.md)：2026-10-19 归档。2026-10-17 对 `v0.6.2..239b1a7c` 的发布前收敛审查，共 39 条，覆盖与顶层设计不一致、结构冗余与死代码、可读性与注释卫生三类问题；38 条完成，第 2.15 条中 `PatchouliService` 的改名移位与公开路由命名按 owner 意见不处理。实现经 PR #130、#131、#133 与分支 `refactor/v0.7.0-convergence-p3` 合入，其中第 1.6、3.9 条随[异常体系收敛](../plans/v0.7.0-error-model-convergence.md)、第 3.1 条随[任务进程拥有执行流](../plans/v0.7.0-task-process-owned-execution.md)完成；现行规则见 [AGENTS.md](../../../AGENTS.md)与[文档治理规范](../../DOCUMENTATION.md)第 11.2 节。
- [Profile 管理列表的可见性语义](./agent-profile-management-list-visibility.md)：2026-10-16 修复，root Profile 列表补齐 owner-management 语义（`enforce_actor_visibility=False`），不再只列出 PUBLIC Profile；当前事实见[路由与事件契约](../../contracts/routes-and-events.md)与 [System 应用服务](../../system/application-services.md)。
- [测试套件冗余收敛后续](./test-suite-redundancy-follow-ups.md)：2026-10-16 完成，按[冻结调查基线](../../governance/baselines/test-suite-redundancy-2026-10-16.md)收敛参数化矩阵、测试随归属迁入 owner、退役类型层测试与 `_e2e` 命名，生命周期评分与强化组并入 owner 并补充重算顺序用例；现行规则见[测试设计规范](../../governance/testing/test-design-standards.md)第 4 节。
- [kernel 遗留命名清理](./kernel-legacy-naming.md)：2026-10-12 完成，系统工具保留 syscall 命名只去掉 Kernel（错误键改为 `mtp.run.tool_not_found`），提示词移除 kernel 与 HiveOS，后端注释改用当前组件名，前端观测面板改称 Workspace Vision、Chat 页面与中间栏改称 Chat View 与 ChatPane；当前事实见 [MTP Runtime](../../alice/mtp-runtime.md)第 5 节与 [Chat 视图](../../frontend/chat-view.md)。
- [检索分数阈值与候选截取收敛](./retrieval-hybrid-threshold-with-reranker.md)：2026-10-12 完成，分数阈值只作用于 Dense 召回（默认 0.5），Sparse、融合与重排不设阈值；融合不再提前截取，截取只在重排之后按请求的 `top_k`（缺省为 `retrieval.default_top_k`）进行一次；当前事实见 [Patchouli 检索](../../patchouli/retrieval.md)第 4、5、9 节。
- [TaskProcess 容器的职责与依赖方向](./task-process-container-ownership.md)：2026-10-04 完成，编排骨架从进程容器拆为所有进程共用的执行器 `TaskProcessRunner`，注册入口只持有生命周期依赖；当前事实见 [任务进程](../../workspace/task-process.md)。
- [WorkspaceAsset 上传的认证上下文与 scope 不一致](./workspace-asset-upload-access-scope-mismatch.md)：2026-10-04 随 v0.7.0 A1 访问边界返工结构性修复（上传只使用授权返回的 scope）；当前事实见 [能力层](../../workspace/capability.md#1-三个调用面)第 1 节。
- [MTP 缓存命中作用域重验](./mtp-cache-scope-revalidation.md)：v0.6.2 修复与 READ/RUN 回归入口；当前事实见 [MTP 契约](../../contracts/mtp.md)。
- [ShortTermMemoryStore 边界收敛](./short-term-memory-store-boundary-cleanup.md)。
- [记忆来源与作者语义](./memory-provenance-vs-authorship.md)。
- [Topic 关闭时的失败隔离](./topic-shutdown-per-topic-failure-isolation.md)。
- [Memory Alias 重名缺陷](./memory-alias-uniqueness.md)：同 Workspace alias 唯一性的两层修复（PR #103）；当前事实见 [MemoryLibrary](../../patchouli/memory-library.md)第 1.2 节，后续事项见 [Todo](../../todo/memory-alias-follow-ups.md)。
- [AgentProfile 模型演进](./agent-profile-model-evolution.md)：模型自持 `agent_id`（PR #103）；当前事实见 [Alice](../../alice/README.md)，开放项转为 [第五代架构 Idea](../../ideas/fifth-generation-architecture.md) P-10a。
