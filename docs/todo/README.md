---
title: Todo
status: current
owner: project
scope: small-defects-and-technical-debt
last_reviewed: 2026-10-30
---

# Todo

本目录用于范围较小、排期灵活的缺陷和技术债。跨系统功能或需要完整迁移、验收方案的工作应进入 `plans/`。

当前事项：

- [推理模型重新加载时的事件循环停顿](./model-reload-event-loop-stall.md)（2026-10-30 由空闲卸载计划的实测验收登记：onnxruntime 创建会话与卸载时的 `gc.collect()` 持有 GIL，重新加载期间其他请求最长等待约 1～3.7 秒，两个模型同时加载时最长；owner 决定记为已知限制，之后单独优化，未排期）；
- [MTP READ 截断正式记忆的正文](./mtp-read-content-truncation.md)（2026-10-29 由记忆访问计划阶段三审查登记：MTP READ 取编译选项默认的 500 字符上限，超长正文被截断，提示“再次 READ”却得到同样结果；预计随 v0.7.3 MTP READ 专项编译一起修复）；
- [生成材料的筛选与截断](./generation-material-filtering-and-truncation.md)（2026-10-29 由 Import Bus 真实使用观察登记：Gateway 价值预判只看用户消息，把整轮实现丢出生成，结算已停止按 `worth_saving` 筛选；待在真实样本上校准生成视图的每轮上限，并随记忆价值策略重设计决定 `worth_saving` 的去留）；
- [生成查重的决策信号](./generation-dedup-decision-signal.md)（2026-10-29 由 Import Bus 真实使用观察登记：dense top-1 余弦在记忆对记忆的比较中分不开同一工作与同项目的其他记忆，固定阈值够不到，查重只会新建；UPDATE 为整体覆盖，不能单靠调低阈值）；
- [重排模型 int8 与 ONNX 内存池](./reranker-int8-and-onnx-memory.md)（2026-10-27 排查常驻内存时登记：重排模型 int8 先做实验评估，另有 ONNX 内存池保留推理峰值、嵌入截断上限 8192 token、配置项不生效与 Docker 预下载落空；后四项已于同日完成，int8 评估待开始）；
- [Import Bus 真实使用观察](./import-bus-real-use-observation.md)（2026-10-21 Import Bus 重设计收尾时登记：真实使用验收随日常使用持续观察，核对闭环、话题落位、生成质量与开销、凭据、导入耗时与连接器行为；2026-10-29 记忆访问计划收尾时并入 MCP 检索与读取的观察，首个样本为历史决定未导入、正文不进检索索引而未命中）；
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

原“导入的 MCP 读取在轨迹中显示 alias”于 2026-10-30 并入计划：它是工具调用进入 `TurnEvent` 时 `tool_name` 含义重叠的一个表现，随[工具解析链路重写计划](../plans/v0.7.1-tool-call-parsing-chain.md)从根上处理（删除前最后版本见 commit `051625d9`）。

原“任务进程中无生产使用方的代码与可合并小项”于 2026-10-30 完成并归档（删除只供测试使用的非流式交付、按句柄取消、诊断查询链与重复 `process_id` 预检；交付方离开的停止原因合并为 `client_disconnected`，观测标签移到进程记录，删除对 prepare 结果归属的重复比对）：[归档记录](../archive/todo/task-process-unused-paths.md)；当前事实见[任务进程](../workspace/task-process.md)。

原“记忆编辑可以给非 Profile 记忆写入 agent_config”于 2026-10-27 完成并归档（提交边界的类型约束拒绝非 `AGENT_PROFILE` 记忆带 `agent_config`，被拒绝的编辑不产生版本，Profile 编辑照常经共用编辑入口生效）：[归档记录](../archive/todo/memory-edit-agent-config-type-check.md)；当前事实见 [Patchouli 生成](../patchouli/generation.md)第 5 节。

原“话题折叠计入交互提交的处理时限”于 2026-10-27 完成并归档（折叠移出交互应用，作为提交队列中与话题提交共用排序键的独立工作项，折叠期间同话题的后续提交在队列中等待；LLM 摘要设独立时限并在回退时保留已有摘要；apply journal 随之移除，幂等收敛到提交队列）：[归档记录](../archive/todo/topic-fold-in-interaction-apply.md)；当前事实见[感知与短期话题](../patchouli/perception.md)第 5 节。

原“模型推理移出事件循环”于 2026-10-27 完成并归档（每个模型一个推理线程，加载与推理不再在事件循环上执行；onnxruntime 创建会话时持有 GIL，加载期间事件循环仍会停顿）：[归档记录](../archive/todo/inference-off-event-loop.md)；当前事实见 [Patchouli](../patchouli/README.md) 第 4 节。

原“Profile 管理列表的可见性语义”于 2026-10-16 修复并归档（root Profile 列表改按 owner-management 语义列出 Workspace 内全部 Profile）：[归档记录](../archive/todo/agent-profile-management-list-visibility.md)；当前事实见[路由与事件契约](../contracts/routes-and-events.md)。

原“测试套件冗余收敛后续”于 2026-10-16 完成并归档（按冻结基线收敛参数化矩阵、测试随归属迁入 owner、退役类型层测试，生命周期评分与强化组并入 owner）：[归档记录](../archive/todo/test-suite-redundancy-follow-ups.md)；现行规则见[测试设计规范](../governance/testing/test-design-standards.md)第 4 节。

原“kernel 遗留命名清理”于 2026-10-12 实施并归档（系统工具保留 syscall 命名只去掉 Kernel，提示词移除 kernel 与 HiveOS，前端观测面板改称 Workspace Vision，Chat 页面与中间栏改称 Chat View 与 ChatPane）：[归档记录](../archive/todo/kernel-legacy-naming.md)；当前事实见 [MTP Runtime](../alice/mtp-runtime.md) 与 [Chat 视图](../frontend/chat-view.md)。

原“检索分数阈值与候选截取收敛”于 2026-10-12 实施并归档（阈值只作用于 Dense 召回，融合不再提前截取，截取只在重排之后进行一次）：[归档记录](../archive/todo/retrieval-hybrid-threshold-with-reranker.md)；当前事实见 [Patchouli 检索](../patchouli/retrieval.md)。

原“总线路由的类型化与静态签名检查”涉及 core、各子系统 contracts、总线与 CI 的跨系统改动，于 2026-10-12 升级为计划：[v0.7.0 总线路由类型化与 mypy 门禁（已完成归档）](../archive/plans/v0.7.0-typed-bus-routes.md)（删除前最后版本见 commit `8faa0b7`）。

原“A1 访问边界返工”涉及 workspace、Patchouli、server 与配置的跨系统改动，于 2026-10-02 升级为计划，2026-10-04 实施归档：[v0.7.0 A1 访问边界返工](../archive/plans/v0.7.0-a1-access-boundary-rework.md)。原“WorkspaceAsset 上传的认证上下文与 scope 不一致”已随该计划修复并归档：[归档记录](../archive/todo/workspace-asset-upload-access-scope-mismatch.md)。原“TaskProcess 容器的职责与依赖方向”于 2026-10-04 完成并归档（编排骨架拆为执行器 `TaskProcessRunner`，进程只是状态容器）：[归档记录](../archive/todo/task-process-container-ownership.md)。

原“Page Folding 跨入口上下文与证据后续技术债”于 2026-10-01 按前台与后台拆分后删除（删除前最后版本见 commit `74b5056`）：后台 Topic 折叠的缺口并入 [Page Folding Raw Evidence Idea](../ideas/PatchouliPageFoldingRawEvidenceDesign.md) 第 9 节，前台上下文的缺口并入 [Turn 内上下文折叠 Idea](../ideas/long-running-agent-intra-turn-context-folding.md) 第 14 节。 2026-10-16 进一步将通用压缩与 Topic folding 优化集中到[短期记忆与上下文压缩](../ideas/short-term-memory-and-context-compaction.md)，原始资料全链路继续由 Raw Evidence 维护。

已完成的 [MTP 缓存命中作用域重验](../archive/todo/mtp-cache-scope-revalidation.md) 已归档，继续作为隔离回归基线；[Memory Alias 重名缺陷](../archive/todo/memory-alias-uniqueness.md)与 [AgentProfile 模型演进](../archive/todo/agent-profile-model-evolution.md)于 2026-09-27 归档（后者的开放项转为 Idea 问题）；其他已完成事项见 [Archived Todo](../archive/todo/README.md)。

Todo 只保存问题、证据、影响和完成条件。若事项扩展为跨系统功能或身份架构，应升级为 Plan；若已有项目 Issue，则链接 Issue，避免维护两份详细状态。
