---
title: Import Bus Real-Use Observation
status: archived
archived_at: 2026-10-30
implemented_by: v0.7.1 收尾（archive/plans/v0.7.1-release-closeout-audit.md）
superseded_by: docs/system/import-bus.md
owner: project
scope: import-bus-and-claude-code-connector-real-use-observation
related_docs:
  - docs/system/import-bus.md
  - docs/contracts/mcp.md
  - docs/archive/plans/v0.7.1-import-bus-redesign.md
  - docs/archive/plans/v0.7.1-memory-access-mcp.md
  - docs/archive/plans/v0.7.1-release-closeout-audit.md
  - docs/patchouli/generation.md
  - docs/ROADMAP.md
last_reviewed: 2026-10-30
---

# Import Bus 真实使用观察

> 已归档（2026-10-30，v0.7.1 收尾）。owner 决定：之后正常使用即可，不再作为待办逐项跟踪。处置见文末“归档结论”；当前事实见 [Import Bus](../../system/import-bus.md) 与 [MCP 记忆访问](../../contracts/mcp.md)，已知限制分别在两篇的“已知限制”一节。以下正文保留归档前的原文，链接已改为归档后的相对路径。

## 问题与证据

[Import Bus 重设计](../plans/v0.7.1-import-bus-redesign.md)的实现、测试与代码审查已经完成，第 8 节第 13 项的真实使用验收按 owner 决定（2026-10-21）改为随日常使用持续观察，不作为计划的收尾条件。以下项目只能在真实的 Claude Code 使用中核对，单元、集成与离线 E2E 测试覆盖不到：

- **闭环是否形成**：日常使用后记忆库出现来自这些会话的记忆，可从记忆回查到交互 Artifact 与来源会话；中断的轮次以 `cancelled` 进入；HiveMemory 停止期间 Claude Code 正常工作，重新启动后积压的轮次按顺序补发且不重复；Alice 的新会话能召回这些记忆。
- **话题落位**：Gateway 的落位是否合理；话题路由失败时续接上一轮的兜底是否把换了话题的轮次归入旧话题。
- **生成质量与开销**：思考、工具调用与结果进入生成视图后，生成的记忆质量、生成转写的长度与截断比例、Gateway 与生成的模型调用量；据此调整 `patchouli.generation.transcript` 的上限，样本同时供 [ROADMAP](../../ROADMAP.md) 第 4.8 节的价值策略校准。
- **折叠**：块的 token 统计按渲染出的整轮材料估算后，折叠的频率；LLM relay 的近期事件自 2026-10-30 起含思考与外部工具（[工具解析链路重写](../plans/v0.7.1-tool-call-parsing-chain.md)，验收中对三轮交互直接调用耗时 2.7 秒），真实折叠的实际耗时、超过 `summary_timeout_seconds` 而回退为规则摘要的比例，以及同一话题下一轮因等待折叠得到 503 `apply_pending` 的次数，据此调整摘要时限。
- **凭据**：工具结果中的密钥、令牌是否出现在生成的记忆中（原文随交互 Artifact 持久化，提取提示词要求不写入记忆）。
- **导入耗时与失败**：导入请求的耗时、`apply_pending` 与 `import_bus.apply_failed` 的比例；拒绝记录中 413 的比例；交互 Artifact 是否因查重窗口或服务重启出现重复。
- **连接器**：30 分钟无新条目即按中断结束的规则是否误判长时间等待权限确认的轮次；Claude Code 本地记录中思考正文的可得性（较新的模型默认只记录签名或摘要）。
- **资源**：提交队列的终态工作记录保存完整的编码载荷，工具输出很大时的内存占用；必要时调低 `import_bus.max_request_bytes`。
- **记忆访问（MCP）**：agent 是否在需要时检索，以及“该用而未用”与“用了但没找到”的样本（VISION 假设 H3）；检索结果的相关性与过时比例；经 MCP 读取记录的引用对排序与归档的影响；回声标记是否生效，即用过检索工具的轮次结算后没有生成重复记忆；导入的检索与读取在生成视图中显示查询与所读的 alias（2026-10-30 以验收轮次确认，真实会话中 agent 自行调用后的结果待确认）；凭据配置出错的频率。

进度（2026-10-29）：生成质量一项中，查重从未做出新建以外的决策，已登记为[生成查重的决策信号](../../todo/generation-dedup-decision-signal.md)；生成转写的长度与截断比例已对账，Gateway 价值预判整轮丢弃实现过程的问题已停用筛选，每轮上限的校准登记为[生成材料的筛选与截断](./generation-material-filtering-and-truncation.md)。

记忆访问的第一个样本（2026-10-29，记忆访问计划实测验收时，记忆库共 30 条记忆）：在 Claude Code 中问及“Import Bus 连接器遇到 4xx 时怎样处理游标、当初为什么这样定”，agent 调用了三次 `memory_search`（各返回 5、2、4 条）与两次 `memory_read`，判断没有相关记忆。原因有两个：

- 该决定是 2026-10-20 至 10-21 在 Import Bus 重设计的会话中定下的，插件 2026-10-27 才安装，那些会话从未导入；记忆库中只有一条无关记忆（话题折叠耗时的决定）在正文中顺带提到“保留游标”，标题与摘要都没有涉及。历史会话的补导属于 [ROADMAP](../../ROADMAP.md) 第 4.5 节的 v0.7.2。
- 检索的稠密与稀疏向量都只由标题、类型、标签与摘要编译（`engines/memory_compiler/handlers/embedding.py`），正文不进入索引，只写在正文中的事实召回不到。

owner 判断（2026-10-29）：根本是记忆生成的粒度问题——一次话题结算至多产出一条记忆，同一话题里的多项决定只能压进一条记忆，次要内容落在正文中；从一段话题材料中提取多条记忆比较麻烦，作为已知限制记录在 [Patchouli 生成](../../patchouli/generation.md#8-当前限制)第 8 节。

## 影响

这些项目决定 Claude Code 对话回流能否作为记忆在真实路径中积累的依据，也是 MCP 记忆访问的检索效果与记忆核心改进的输入。没有观察记录时，生成转写的上限、价值信号与续接兜底都只能沿用候选默认值。

## 完成条件

- 在一段时间的真实使用后，逐项写下观察结论（可附导入轮次数、记忆条数、拒绝与失败的计数，不复制会话正文或凭据）；
- 需要改动的项目各自登记为 Todo、Idea 或计划，本 Todo 随之归档；
- 不需要改动的项目在结论中说明理由。

## 归档结论（2026-10-30）

owner 决定在 v0.7.1 收尾时归档本 Todo，之后正常使用即可。归档时并未逐项写下观察结论，完成条件第一、三项按这一决定不再执行；各项的去向如下：

| 观察项 | 去向 |
|:---|:---|
| 生成质量与开销 | 已登记为 Todo：[生成查重的决策信号](../../todo/generation-dedup-decision-signal.md)、[生成材料的筛选与截断](./generation-material-filtering-and-truncation.md)；`worth_saving` 整轮筛选已于 2026-10-29 停用 |
| 记忆访问（MCP） | 首个样本的两项原因：历史会话未导入属于 [ROADMAP](../../ROADMAP.md) 第 4.5 节的 v0.7.2；生成粒度记为 [Patchouli 生成](../../patchouli/generation.md#8-当前限制)的已知限制 |
| 闭环、话题落位、折叠、凭据、导入耗时与失败、连接器、资源 | 没有形成需要改动的观察结论；相关风险已写在 [Import Bus](../../system/import-bus.md#10-已知限制) 第 10 节与 [MCP 记忆访问](../../contracts/mcp.md#8-已知限制) 第 8 节的已知限制中。日常使用中再发现问题时按缺陷另行登记 |

推理模型重新加载时的停顿在同一时期由空闲卸载计划的实测登记为[独立 Todo](../../todo/model-reload-event-loop-stall.md)，不属于本 Todo。
