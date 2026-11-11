---
title: Todo
status: current
owner: project
scope: small-defects-and-technical-debt
last_reviewed: 2026-11-11
---

# Todo

本目录用于范围较小、排期灵活的缺陷和技术债。跨系统功能或需要完整迁移、验收方案的工作应进入 `plans/`。

当前事项：

- [保存后修改 Agent 别名，并实时更新所有引用](./agent-alias-rename.md)（2026-11-11 由前端 UI 优化计划登记：别名同时是 Profile 解析、访问登记、接入声明、对话与 CALL 中 Agent 的标识，改名时这些引用不会随之更新，Agent 页暂时只允许在草稿中修改别名；owner 决定改名时引用要实时更新，未排期）；
- [引用记忆面板的“相关度”](./referenced-memory-relevance.md)（2026-11-11 由前端 UI 优化计划登记：样稿中的相关度条没有数据来源，预检索结果经操作入口交出时已经只剩记忆本身、不带分数；owner 决定之后结合后端一起加回，未排期）；
- [WorkspaceAsset Store 中没有生产调用方的方法](./workspace-asset-store-unused-methods.md)（2026-11-06 由 WorkspaceAsset 归属拆分计划的评审登记：`resolve_asset`、`list_workspace_assets`、`create_asset`、`register_raw_representation`、`remove_asset` 只在测试中调用，REMOVED 状态因此没有生产来源；owner 决定保留，逐个检查是补上使用方还是删除，未排期）；
- [并发合并同一条记忆时，后写回的合并失败并留下重复](./merge-version-conflict.md)（2026-11-04 随 v0.7.3 收尾由收敛 Idea 第 4.4 节移入：同一次结算的两条草稿命中同一条记忆时，后写回的合并以版本冲突放弃，留下重复；话题内主动写入优先之后仍出现一次；是否重做等以原因码记录的真实失败再判断，未排期）；
- [多条提取后被动提取的记忆膨胀与低留存](./passive-extraction-volume-and-retention.md)（2026-11-03 由记忆增长统计登记：多条提取后每次结算的 Mode A 新建从 1 条变为平均 5.2 条，其中仅 52% 留存，owner 手动清理；owner 的目标是全自主记忆维护，本 Todo 保留证据与度量，可合进任一 Idea 或计划）；
- [导入轮次被落进另一个外部会话的话题](./cross-conversation-topic-placement.md)（2026-11-03 由 dsh 调研取证登记：外部会话的第一轮没有会话上下文，Gateway 凭标题中的通用词把一次 dsh 调研落进了同时进行的 Claude Code 会话的话题，结算材料混合了两段无关对话，主动写入的贡献者被污染；一个话题能否跨外部会话待 owner 决定，未排期）；
- [Agent Profile 以 ModelSelection 表达模型选择](./agent-profile-model-selection.md)（2026-11-01 由模型选择统一计划登记：Profile 仍以平铺的 `model_name`、`temperature`、`top_p` 表达模型选择，没有 `max_tokens`；改动涉及已存 Profile 的读取与 Agent 编辑页，owner 决定另行处理，未排期）；
- [推理模型重新加载时的事件循环停顿](./model-reload-event-loop-stall.md)（2026-10-30 由空闲卸载计划的实测验收登记：onnxruntime 创建会话与卸载时的 `gc.collect()` 持有 GIL，重新加载期间其他请求最长等待约 1～3.7 秒，两个模型同时加载时最长；owner 决定记为已知限制，之后单独优化，未排期）；
- [生成查重的决策信号](./generation-dedup-decision-signal.md)（2026-10-29 由 Import Bus 真实使用观察登记：dense top-1 余弦在记忆对记忆的比较中分不开同一工作与同项目的其他记忆；2026-10-30 补充：同项目记忆增多后，查重判出的 UPDATE 整体覆盖了两条记忆的主要内容；2026-10-31 owner 决定命中后先落库再融合，覆盖问题随 v0.7.2 的[查重与更新计划](../archive/plans/v0.7.2-dedup-merge.md)解决并于同日归档；判定信号暂不改动，信号部分留在本 Todo；2026-11-01 查询改为与入库的嵌入文本同形，阈值不变）；
- [重排模型 int8 与 ONNX 内存池](./reranker-int8-and-onnx-memory.md)（2026-10-27 排查常驻内存时登记：重排模型 int8 先做实验评估，另有 ONNX 内存池保留推理峰值、嵌入截断上限 8192 token、配置项不生效与 Docker 预下载落空；后四项已于同日完成；int8 评估 owner 2026-10-30 决定现在不是必要项，移出 v0.7.1，未排期）；
- [Memory alias 后续事项](./memory-alias-follow-ups.md)（未排期：alias 占用查询的 payload index；无 alias 记忆的寻址已于 2026-11-07 改为回退到 `mem_<memory id>`）；
- [会话 `/compact` 指令](./conversation-compact-command.md)（2026-10-01 由原 Topic `/compact` 系统指令接入重写：compact 归属前台的 CPU 会话上下文，依赖 ConversationSession、命令运行位置与 Alice 会话压缩）；
- [Work Queue Runtime 多 lane 拓扑技术债](./work-queue-runtime-lane-topology.md)；
- [Memory 可见性策略前端组件](./memory-visibility-policy-ui.md)（依赖 Memory CRUD 契约先支持 `access_policy` 读写）；
- [补齐 Alice Runtime 健康探针](./alice-health-probes.md)；

已完成的事项，以及升级为计划、并入计划或删除的事项，登记在 [Archived Todo](../archive/todo/README.md)。

Todo 只保存问题、证据、影响和完成条件。若事项扩展为跨系统功能或身份架构，应升级为 Plan；若已有项目 Issue，则链接 Issue，避免维护两份详细状态。
