---
title: Todo
status: current
owner: project
scope: small-defects-and-technical-debt
last_reviewed: 2026-11-01
---

# Todo

本目录用于范围较小、排期灵活的缺陷和技术债。跨系统功能或需要完整迁移、验收方案的工作应进入 `plans/`。

当前事项：

- [Gateway 与 Librarian 的旧式直配路径](./legacy-llm-direct-config.md)（2026-11-01 由密钥移出主配置登记：没有 `model_id` 时直接使用配置中的 `model`、`api_key`、`api_base`，经 `HIVEMEMORY__LLM__*__API_KEY` 等环境变量注入，绕过模型与 Provider 注册表；owner 选择修复范围时未纳入，未排期）；
- [模型级密钥覆盖写入受版本控制的 models.yaml](./model-registry-api-key-override.md)（2026-11-01 由密钥移出主配置登记：模型页面的“高级覆盖”把模型级 `api_key` 写入 `configs/models.yaml`；owner 选择修复范围时未纳入，未排期）；
- [本机 HTTP 管理接口不证明调用方](./local-http-management-access.md)（2026-11-01 由 dsh 接入试用登记：管理接口以 server 的 principal 接入或不经认证网关，本机任何进程包括 harness 中的 agent 都能读写记忆、配置与密钥；会改变 HTTP 的信任模型，实施前先形成 Idea 或计划；owner 决定下一步修复）；
- [生成组件测试的分类与重复覆盖](./generation-component-test-classification.md)（2026-10-31 查重与更新修正审查登记：组件调用位于 E2E 目录，查重集成断言同时接受多种决策；后续按归属层收敛，未排期）；
- [推理模型重新加载时的事件循环停顿](./model-reload-event-loop-stall.md)（2026-10-30 由空闲卸载计划的实测验收登记：onnxruntime 创建会话与卸载时的 `gc.collect()` 持有 GIL，重新加载期间其他请求最长等待约 1～3.7 秒，两个模型同时加载时最长；owner 决定记为已知限制，之后单独优化，未排期）；
- [MTP READ 截断正式记忆的正文](./mtp-read-content-truncation.md)（2026-10-29 由记忆访问计划阶段三审查登记：MTP READ 取编译选项默认的 500 字符上限，超长正文被截断，提示“再次 READ”却得到同样结果；随 MTP READ 专项编译一起修复，该方向 2026-10-30 起不绑定版本，未排期）；
- [生成查重的决策信号](./generation-dedup-decision-signal.md)（2026-10-29 由 Import Bus 真实使用观察登记：dense top-1 余弦在记忆对记忆的比较中分不开同一工作与同项目的其他记忆；2026-10-30 补充：同项目记忆增多后，查重判出的 UPDATE 整体覆盖了两条记忆的主要内容；2026-10-31 owner 决定命中后先落库再融合，覆盖问题随 v0.7.2 的[查重与更新计划](../archive/plans/v0.7.2-dedup-merge.md)解决并于同日归档；判定信号暂不改动，信号部分留在本 Todo；2026-11-01 查询改为与入库的嵌入文本同形，阈值不变）；
- [重排模型 int8 与 ONNX 内存池](./reranker-int8-and-onnx-memory.md)（2026-10-27 排查常驻内存时登记：重排模型 int8 先做实验评估，另有 ONNX 内存池保留推理峰值、嵌入截断上限 8192 token、配置项不生效与 Docker 预下载落空；后四项已于同日完成；int8 评估 owner 2026-10-30 决定现在不是必要项，移出 v0.7.1，未排期）；
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

已完成的事项，以及升级为计划、并入计划或删除的事项，登记在 [Archived Todo](../archive/todo/README.md)。

Todo 只保存问题、证据、影响和完成条件。若事项扩展为跨系统功能或身份架构，应升级为 Plan；若已有项目 Issue，则链接 Issue，避免维护两份详细状态。
