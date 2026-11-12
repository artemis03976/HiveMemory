---
title: Agent Alias Rename
status: todo
owner: project
scope: rename-a-saved-agent-alias-and-update-every-reference-at-once
code_paths:
  - frontend/src/components/AgentManagement.tsx
  - frontend/src/services/agentApi.ts
  - frontend/src/stores/chat/chatStore.ts
  - src/hivememory/patchouli/services/retrieval.py
  - src/hivememory/server/routers/access.py
  - src/hivememory/workspace/sessions/service.py
related_docs:
  - ../frontend/management-views.md
  - ../workspace/identity-and-authorization.md
  - ../archive/plans/frontend-ui-and-entry-wiring.md
last_reviewed: 2026-11-11
---

# 保存后修改 Agent 别名，并实时更新所有引用

## 现状与证据

Agent 的别名（`index.alias`）同时是它在系统中的标识：

- Profile 解析按别名查找图纸（[`RetrievalFamiliar.get_agent_profile`](../../src/hivememory/patchouli/services/retrieval.py) 的 `agent_alias`）；
- Workspace 的访问登记以 `agent_id` 记录 Agent 的操作白名单（[`/actors/agents/{agent_id}`](../../src/hivememory/server/routers/access.py)），外部接入的来源以 `allowed_agent_ids` 声明可以代表哪些 Agent；
- 对话请求以 `agent_id` 指定执行的 Agent，前端把当前选择持久化在 `chat-store` 的 `currentAgentId` 中；
- 会话的每一轮记录产生它的 `agent_id`（[`sessions/service.py`](../../src/hivememory/workspace/sessions/service.py)）；
- CALL 以别名指定被调用的 Agent。

后端的记忆更新接口允许改写别名，但上述引用不会随之改变：改名之后访问登记与 `allowed_agent_ids` 仍指向旧别名，对话与 CALL 用旧别名会解析失败。前端 UI 优化计划因此把已保存 Agent 的别名改为只读，只在新建的草稿中可以修改。

## 影响

已保存的 Agent 不能改名；想换别名只能新建一个 Agent，再手工迁移访问登记与外部接入的声明。

## 完成条件

- owner 2026-11-11 决定：别名保存后也要能修改，修改时引用实时更新；记为 Todo，未排期。
- [ ] 确定改名的入口与语义：由 root API 提供“改名”操作，在一次操作中改写别名并更新访问登记、接入声明与其他持久化的引用；旧别名是否短期保留为可解析的别名，需要决定。
- [ ] 确定历史数据的处理：会话中已记录的 `agent_id`、记忆来源中的 Agent 标识是保留原值（如实记录当时的身份），还是随改名更新。
- [ ] 前端在已保存 Agent 上提供改名，成功后同步更新对话页的当前选择与 Agent 列表。
- [ ] 改名的权威测试放在 root API 一层，覆盖引用一并更新与别名冲突时拒绝。
