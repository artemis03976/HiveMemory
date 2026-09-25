---
title: AgentProfile 模型演进
status: todo
owner: patchouli
scope: agent-profile-identity-and-harness-neutral-capabilities
related_docs:
  - docs/plans/v0.7.0-plan-a-boundary-charter.md
  - docs/plans/v0.7.0-a2-workspace-resource-reads-and-caches.md
  - docs/plans/v0.7.0-external-memory-service-and-actor-interaction.md
last_reviewed: 2026-09-25
---

# AgentProfile 模型演进

## 问题与证据

`AgentProfile`（`src/hivememory/core/models/agent.py`）当前字段为 persona、model_name、temperature、top_p、allowed_mtp_verbs、allowed_sys_tools。三个已知限制：

1. **无身份字段**：agent_id（alias）只存在于外层调用参数与 HTTP 投影（`server/models/agent.py::AgentProfileResponse`），核心模型不自持身份。2026-09-25 C2 裁定以 alias 为 AgentProfile 的 actor 面向身份、profile cache 按 `(Workspace, agent_alias)` 作 key——身份口径需要在模型或契约层冻结；
2. **MTP 中心**：`allowed_mtp_verbs` / `allowed_sys_tools` 的语义绑定 MTP/System tools 体系，对外部 harness（MCP、B transport）没有定义含义；
3. **外部 actor 语义未定义**：v0.7.0 明确不承诺外部 harness 能执行这些配置（A2 §2.3），但"已应用/部分支持/拒绝"的应用语义只有原则表述，没有字段或伴随契约承载。

影响：模型作为"面向所有 actor 的 Profile 定义"尚不可靠；当前仅作为 Alice/MTP 体系的运行配置是成立的。

## 演进方向（待归属计划）

- **身份口径冻结**：模型自持 agent_id（alias），或在契约层明确"身份由解析入参承载"的稳定口径，二者取一；
- **能力描述与 MTP 解耦**：将 `allowed_*` 泛化为 harness 中性的能力/限制描述，由各 adapter 映射到自身体系；
- **外部 actor 应用语义**：为"已应用/部分支持/拒绝"定义可承载的字段或伴随契约。

与缓存/读取契约的关系：[A2](../plans/v0.7.0-a2-workspace-resource-reads-and-caches.md) §2.3 的 profile cache 对解析结果不透明，key 与失效锚在 `(Workspace, agent_alias)` 与事件 alias 集上——本债的模型演进不破坏缓存契约；反之 A2 也不因本债阻塞，两者可独立推进。

## 完成条件

- 身份口径冻结（模型自持或契约承载，二者取一）；
- 能力描述字段与 MTP 解耦，或有明确的按 harness 映射契约；
- 外部 actor 的应用语义有字段/契约承载；
- 以上归属某次模型演进计划，不与 A2 缓存交付互相阻塞。
