---
title: Gateway
status: draft
owner: gateway
scope: gateway-subsystem
last_reviewed: 2026-07-27
---

# Gateway

本目录是 Gateway 子系统当前设计的目标入口。Gateway 负责入口级决策、系统指令、话题与查询分析，以及向下游投影稳定的决策结果。

计划收录的当前设计包括：

- `analysis.md`：用户查询分析、降级与决策投影；
- `commands.md`：系统指令注册、解析、分发与短路语义；
- `workflow.md`：Gateway workflow、step、不变量和失败边界；
- 子系统 README 中的职责、非职责、依赖和代码入口。

当前目录仍处于文档迁移阶段。现有 `docs/engines/gateway.md` 与 `docs/mod/V0.6.0*` 同时包含旧设计、已实现事实和未来计划，必须按[迁移清单](../plans/documentation-migration-inventory.md)拆分后才能成为当前真相源。
