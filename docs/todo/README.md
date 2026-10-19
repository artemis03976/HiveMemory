---
title: Todo
status: current
owner: project
scope: small-defects-and-technical-debt
last_reviewed: 2026-10-18
---

# Todo

本目录用于范围较小、排期灵活的缺陷和技术债。跨系统功能或需要完整迁移、验收方案的工作应进入 `plans/`。

当前事项：

- [v0.7.0 代码收敛清单](./v0.7.0-code-convergence.md)（2026-10-17 对 `v0.6.2..239b1a7c` 的收敛审查，共 39 条；Alice 结构相关条目已由[归档计划](../archive/plans/v0.7.0-alice-agent-runtime-restructure.md)完成，提示词组装相关条目已由[提示词编译前移](../archive/plans/v0.7.0-cpu-prompt-compilation.md)完成，第 1.6、3.9 条已由[异常体系收敛](../archive/plans/v0.7.0-error-model-convergence.md)完成，其余按条目处理；2026-10-18 在 `4c7151a6` 复核后 8 条完成、7 条部分完成、24 条未处理；随后第 1 节 P1 与第 2.1–2.3、2.7–2.13 条完成（第 2.9 条改为 P1），现为 25 条完成、4 条部分完成、10 条未处理）；
- [日志配置段的级别、格式与输出没有生效](./logging-config-not-applied.md)（2026-10-18 清理不生效配置键时发现：`logging` 段只有 WebSocket 广播被读取，两种启动方式都不按配置设置日志）；
- [记忆编辑可以给非 Profile 记忆写入 agent_config](./memory-edit-agent-config-type-check.md)（2026-10-18 引入 `MemoryEdit` 时发现：编辑链不按类型限制 `agent_config`，普通记忆被写入无效配置并生成版本）；
- [构造根配置会改写进程默认语言](./config-construction-resets-default-language.md)（2026-10-17 提示词编译收尾审查发现：`GET /config/defaults` 与 `POST /config` 构造配置对象时重置进程默认语言）；
- [WorkspaceAsset 内部归属与操作身份拆分](./workspace-asset-ownership-identity-split.md)（身份第二批明确暂缓：Store/解析旧 scope 接口与 System 物化 reader 桥接）；
- [Memory alias 后续事项](./memory-alias-follow-ups.md)（未排期：无 alias 记忆的寻址、alias 查询索引）；
- [会话 `/compact` 指令](./conversation-compact-command.md)（2026-10-01 由原 Topic `/compact` 系统指令接入重写：compact 归属前台的 CPU 会话上下文，依赖 ConversationSession、命令运行位置与 Alice 会话压缩）；
- [Memory Garden 接入真实语义检索](./frontend-memory-semantic-search.md)；
- [建立前端身份状态所有权](./frontend-identity-ownership.md)；
- [Work Queue Runtime 多 lane 拓扑技术债](./work-queue-runtime-lane-topology.md)；
- [RuntimeEvent 生产端迁移后续](./runtime-event-producer-migration.md)；
- [统一前端 mock fallback 的状态披露](./frontend-mock-fallback-disclosure.md)；
- [Memory 可见性策略前端组件](./memory-visibility-policy-ui.md)（依赖 Memory CRUD 契约先支持 `access_policy` 读写）；
- [补齐 Alice Runtime 健康探针](./alice-health-probes.md)；
- [engines 对上层的既有导入](./engines-upward-imports.md)（包分层登记的 10 处已知例外，原记录于 Workspace 网络与任务进程架构 Idea）；
- [RUN 目标的可执行性检查](./run-target-executability-check.md)（能力层分层审查遗留：只有 `memory.run` 的 actor 可经 RUN 的错误探测可见记忆的类型与存在，不可执行目标也记录引用）；

原“Profile 管理列表的可见性语义”于 2026-10-16 修复并归档（root Profile 列表改按 owner-management 语义列出 Workspace 内全部 Profile）：[归档记录](../archive/todo/agent-profile-management-list-visibility.md)；当前事实见[路由与事件契约](../contracts/routes-and-events.md)。

原“测试套件冗余收敛后续”于 2026-10-16 完成并归档（按冻结基线收敛参数化矩阵、测试随归属迁入 owner、退役类型层测试，生命周期评分与强化组并入 owner）：[归档记录](../archive/todo/test-suite-redundancy-follow-ups.md)；现行规则见[测试设计规范](../governance/testing/test-design-standards.md)第 4 节。

原“kernel 遗留命名清理”于 2026-10-12 实施并归档（系统工具保留 syscall 命名只去掉 Kernel，提示词移除 kernel 与 HiveOS，前端观测面板改称 Workspace Vision，Chat 页面与中间栏改称 Chat View 与 ChatPane）：[归档记录](../archive/todo/kernel-legacy-naming.md)；当前事实见 [MTP Runtime](../alice/mtp-runtime.md) 与 [Chat 视图](../frontend/chat-view.md)。

原“检索分数阈值与候选截取收敛”于 2026-10-12 实施并归档（阈值只作用于 Dense 召回，融合不再提前截取，截取只在重排之后进行一次）：[归档记录](../archive/todo/retrieval-hybrid-threshold-with-reranker.md)；当前事实见 [Patchouli 检索](../patchouli/retrieval.md)。

原“总线路由的类型化与静态签名检查”涉及 core、各子系统 contracts、总线与 CI 的跨系统改动，于 2026-10-12 升级为计划：[v0.7.0 总线路由类型化与 mypy 门禁（已完成归档）](../archive/plans/v0.7.0-typed-bus-routes.md)（删除前最后版本见 commit `8faa0b7`）。

原“A1 访问边界返工”涉及 workspace、Patchouli、server 与配置的跨系统改动，于 2026-10-02 升级为计划，2026-10-04 实施归档：[v0.7.0 A1 访问边界返工](../archive/plans/v0.7.0-a1-access-boundary-rework.md)。原“WorkspaceAsset 上传的认证上下文与 scope 不一致”已随该计划修复并归档：[归档记录](../archive/todo/workspace-asset-upload-access-scope-mismatch.md)。原“TaskProcess 容器的职责与依赖方向”于 2026-10-04 完成并归档（编排骨架拆为执行器 `TaskProcessRunner`，进程只是状态容器）：[归档记录](../archive/todo/task-process-container-ownership.md)。

原“Page Folding 跨入口上下文与证据后续技术债”于 2026-10-01 按前台与后台拆分后删除（删除前最后版本见 commit `74b5056`）：后台 Topic 折叠的缺口并入 [Page Folding Raw Evidence Idea](../ideas/PatchouliPageFoldingRawEvidenceDesign.md) 第 9 节，前台上下文的缺口并入 [Turn 内上下文折叠 Idea](../ideas/long-running-agent-intra-turn-context-folding.md) 第 14 节。 2026-10-16 进一步将通用压缩与 Topic folding 优化集中到[短期记忆与上下文压缩](../ideas/short-term-memory-and-context-compaction.md)，原始资料全链路继续由 Raw Evidence 维护。

已完成的 [MTP 缓存命中作用域重验](../archive/todo/mtp-cache-scope-revalidation.md) 已归档，继续作为隔离回归基线；[Memory Alias 重名缺陷](../archive/todo/memory-alias-uniqueness.md)与 [AgentProfile 模型演进](../archive/todo/agent-profile-model-evolution.md)于 2026-09-27 归档（后者的开放项转为 Idea 问题）；其他已完成事项见 [Archived Todo](../archive/todo/README.md)。

Todo 只保存问题、证据、影响和完成条件。若事项扩展为跨系统功能或身份架构，应升级为 Plan；若已有项目 Issue，则链接 Issue，避免维护两份详细状态。
