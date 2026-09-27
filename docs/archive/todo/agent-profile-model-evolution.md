---
title: AgentProfile 模型演进
status: archived
archived_at: 2026-09-27
implemented_by: "PR #103 (dda9d9d)"
superseded_by: docs/alice/README.md
owner: patchouli
scope: agent-profile-mtp-decoupling-and-external-actor-semantics
related_docs:
  - docs/ideas/workspace-network-task-process-architecture.md
  - docs/ideas/external-actor-registration-and-runtime-access.md
last_reviewed: 2026-09-27
---

# AgentProfile 模型演进

> **归档说明（2026-09-27）**：已完成的两项（模型自持 `agent_id`、可见性不进入模型）随 PR #103（`dda9d9d`）合并，当前事实见 [Alice](../../alice/README.md)中的 Agent Profile 说明。仍开放的“能力描述与 MTP 解耦”与“外部 actor 应用语义”转为 [Workspace 网络与任务进程架构 Idea](../../ideas/workspace-network-task-process-architecture.md)的 P-10 问题。文中引用的 A2 计划已作废删除，引用改为纯文本，删除前最后版本见 commit `dda9d9d`。

## 问题与证据

`AgentProfile`（`src/hivememory/core/models/agent.py`）当前字段为 persona、model_name、temperature、top_p、allowed_mtp_verbs、allowed_sys_tools。三个已知限制：

1. **无身份字段**：agent_id（alias）只存在于外层调用参数与 HTTP 投影（`server/models/agent.py::AgentProfileResponse`），核心模型不自持身份；
2. **MTP 中心**：`allowed_mtp_verbs` / `allowed_sys_tools` 的语义绑定 MTP/System tools 体系，对外部 harness（MCP、B transport）没有定义含义；
3. **外部 actor 语义未定义**："已应用/部分支持/拒绝"的应用语义只有原则表述（A2 §2.3），没有字段或伴随契约承载。

影响：模型作为"面向所有 actor 的 Profile 定义"尚不可靠；当前仅作为 Alice/MTP 体系的运行配置是成立的。

## 已裁定（2026-09-25，随 A2-1 落地）

- **身份：模型自持 agent_id。** `AgentProfile` 补 agent_id 字段（唯一 alias），解析时从源原子的 `index.alias` 填入；profile cache 以 `(Workspace, agent_alias)` 作 key（A2 §2.3）。
- **可见性：不进模型。** 可见性真相留在源原子的 `MemoryAccessPolicy`；授权依据由 `GET_AGENT_PROFILE` backing 结果（`ResolvedAgentProfile`，A2 §8 D-3）附带、随缓存项内部保存，不进 AgentProfile、不出能力面、不经外部 wire。不做"内嵌 policy"的变体——授权内部数据不进入能力描述对象、不上外部 wire、不在 run 固定副本中残留。

## 仍开放（待归属模型演进计划）

- **能力描述与 MTP 解耦**：将 `allowed_mtp_verbs` / `allowed_sys_tools` 泛化为 harness 中性的能力/限制描述，由各 adapter 映射到自身体系；
- **外部 actor 应用语义**：为"已应用/部分支持/拒绝"定义可承载的字段或伴随契约。

与缓存/读取契约的关系：A2 的 profile cache 对解析结果不透明，key 与失效锚在 `(Workspace, agent_alias)` 与事件 alias 集上——剩余演进不破坏缓存契约；反之 A2 不因本债阻塞，两者可独立推进。**前置依赖**：agent_id 的唯一性即 alias 唯一性（C2 裁定的 `(Workspace, agent_alias)` key 以此为前提），见 [Memory Alias 重名缺陷](./memory-alias-uniqueness.md)——其第二层 store 校验是 agent_id 唯一性的保证点。

## 完成条件

- [x] 身份口径冻结：模型自持 agent_id（随 A2-1 落地）；
- [x] 可见性裁定：不进模型，policy 依据随缓存项（已落宪章 §5/A2 §2.3）；
- [ ] 能力描述字段与 MTP 解耦，或有明确的按 harness 映射契约；
- [ ] 外部 actor 的应用语义有字段/契约承载；
- 开放项归属某次模型演进计划，不与 A2 缓存交付互相阻塞。
