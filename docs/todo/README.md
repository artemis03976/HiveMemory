---
title: Todo
status: current
owner: project
scope: small-defects-and-technical-debt-rules-and-index
last_reviewed: 2026-11-20
---

# Todo

本目录用于范围较小、排期灵活的缺陷和技术债，写法见文末的“规则”。

当前事项：

- [Memory 域之外的时间语义统一](./time-semantics-outside-memory-domain.md)：限流器、耗时统计与话题域的时间字段仍用墙钟或 naive 时间；
- [自定义背景，以及桌面端界面偏好的持久化](./custom-background-and-preference-persistence.md)：自定义背景与毛玻璃，以及重启后保留主题等界面偏好；
- [前端细节优化：动效、响应速度与自适应布局](./frontend-ui-polish.md)：前端 UI 优化之后剩下的细节；
- [保存后修改 Agent 别名，并实时更新所有引用](./agent-alias-rename.md)：改名时访问登记、接入声明等引用随之更新；
- [引用记忆面板的“相关度”](./referenced-memory-relevance.md)：相关度条需要后端交出检索分数；
- [WorkspaceAsset Store 中没有生产调用方的方法](./workspace-asset-store-unused-methods.md)：只在测试中调用的方法，逐个补上使用方或删除；
- [合并中止并留下重复：合并目标正等着被吸收，或两个合并同时写同一条记忆](./merge-version-conflict.md)：合并因目标已被删除或版本已变而中止，两条记忆都留下；
- [多条提取后被动提取的记忆膨胀与低留存](./passive-extraction-volume-and-retention.md)：被动提取新建过多、留存偏低的证据与度量；
- [导入轮次被落进另一个外部会话的话题](./cross-conversation-topic-placement.md)：外部会话的第一轮被路由进无关会话的话题；
- [Agent Profile 以 ModelSelection 表达模型选择](./agent-profile-model-selection.md)：Profile 的模型参数改用 ModelSelection；
- [推理模型重新加载时的事件循环停顿](./model-reload-event-loop-stall.md)：模型重新加载期间其他请求被阻塞数秒；
- [生成查重的决策信号](./generation-dedup-decision-signal.md)：dense top-1 余弦分不开同一工作与同项目的其他记忆；
- [重排模型 int8 与 ONNX 内存池](./reranker-int8-and-onnx-memory.md)：重排模型 int8 的质量与内存评估；
- [Memory alias 后续事项](./memory-alias-follow-ups.md)：alias 占用查询的 payload index；
- [会话 `/compact` 指令](./conversation-compact-command.md)：由 CPU 会话上下文承担的 compact 指令；
- [Work Queue Runtime 多 lane 拓扑技术债](./work-queue-runtime-lane-topology.md)：每套业务队列各自一套 Runtime 与 Store 的拓扑；
- [补齐 Alice Runtime 健康探针](./alice-health-probes.md)：健康检查只返回装配状态，不验证模型端点与运行中的 frame。

已完成的事项，以及升级为计划、并入计划或删除的事项，登记在 [Archived Todo](../archive/todo/README.md)。

## 规则

Todo 只适用于范围较小、排期灵活的缺陷或技术债；大型功能、跨系统重构、身份架构，或需要完整迁移与验收方案的工作形成 Plan（[Plans](../plans/README.md)）。Todo 只保存问题、证据、影响和完成条件，至少包含影响范围、现象或证据、完成条件和追踪链接；项目 Issue 中已有对应事项时链接 Issue，不维护两份详细状态。完成的 Todo 在收尾时删除（[文档治理规范](../DOCUMENTATION.md)第 10 节）。
