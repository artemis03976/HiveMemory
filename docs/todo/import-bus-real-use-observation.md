---
title: Import Bus Real-Use Observation
status: todo
owner: project
scope: import-bus-and-claude-code-connector-real-use-observation
related_docs:
  - docs/system/import-bus.md
  - docs/archive/plans/v0.7.1-import-bus-redesign.md
  - docs/patchouli/generation.md
  - docs/ROADMAP.md
last_reviewed: 2026-10-27
---

# Import Bus 真实使用观察

## 问题与证据

[Import Bus 重设计](../archive/plans/v0.7.1-import-bus-redesign.md)的实现、测试与代码审查已经完成，第 8 节第 13 项的真实使用验收按 owner 决定（2026-10-21）改为随日常使用持续观察，不作为计划的收尾条件。以下项目只能在真实的 Claude Code 使用中核对，单元、集成与离线 E2E 测试覆盖不到：

- **闭环是否形成**：日常使用后记忆库出现来自这些会话的记忆，可从记忆回查到交互 Artifact 与来源会话；中断的轮次以 `cancelled` 进入；HiveMemory 停止期间 Claude Code 正常工作，重新启动后积压的轮次按顺序补发且不重复；Alice 的新会话能召回这些记忆。
- **话题落位**：Gateway 的落位是否合理；话题路由失败时续接上一轮的兜底是否把换了话题的轮次归入旧话题。
- **生成质量与开销**：思考、工具调用与结果进入生成视图后，生成的记忆质量、生成转写的长度与截断比例、Gateway 与生成的模型调用量；据此调整 `patchouli.generation.transcript` 的上限，样本同时供 [ROADMAP](../ROADMAP.md) 第 4.8 节的价值策略校准。
- **折叠**：块的 token 统计计入动作段后，折叠的频率；Page Folding 的接力摘要不含思考与工具结果，被折叠的块在结算前失去这些材料的影响；LLM relay 下一次折叠的实际耗时、超过 `summary_timeout_seconds` 而回退为规则摘要的比例，以及同一话题下一轮因等待折叠得到 503 `apply_pending` 的次数，据此调整摘要时限。
- **凭据**：工具结果中的密钥、令牌是否出现在生成的记忆中（原文随交互 Artifact 持久化，提取提示词要求不写入记忆）。
- **导入耗时与失败**：导入请求的耗时、`apply_pending` 与 `import_bus.apply_failed` 的比例；拒绝记录中 413 的比例；交互 Artifact 是否因查重窗口或服务重启出现重复。
- **连接器**：30 分钟无新条目即按中断结束的规则是否误判长时间等待权限确认的轮次；Claude Code 本地记录中思考正文的可得性（较新的模型默认只记录签名或摘要）。
- **资源**：提交队列的终态工作记录保存完整的编码载荷，工具输出很大时的内存占用；必要时调低 `import_bus.max_request_bytes`。

## 影响

这些项目决定 Claude Code 对话回流能否作为记忆在真实路径中积累的依据，也是候选的下一步（Claude Code 经 MCP 工具访问记忆）与记忆核心改进的输入。没有观察记录时，生成转写的上限、价值信号与续接兜底都只能沿用候选默认值。

## 完成条件

- 在一段时间的真实使用后，逐项写下观察结论（可附导入轮次数、记忆条数、拒绝与失败的计数，不复制会话正文或凭据）；
- 需要改动的项目各自登记为 Todo、Idea 或计划，本 Todo 随之归档；
- 不需要改动的项目在结论中说明理由。
