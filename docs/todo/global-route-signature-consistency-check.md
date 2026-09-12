---
title: 全局路由 kwargs 与 handler 签名一致性校验
status: todo
owner: system
scope: bus-route-signature-consistency-check
related_docs:
  - docs/contracts/routes-and-events.md
  - docs/contracts/subsystem-contracts.md
  - docs/system/attachments.md
  - docs/system/runtime-and-bus.md
last_reviewed: 2026-09-12
---

# 全局路由 kwargs 与 handler 签名一致性校验

## 问题与证据

`GlobalSystemBus` 的 RPC 是直接调用：`request(route, *args, **kwargs)` 的 kwargs 名称必须与 handler 形参完全一致，否则在调用时刻以 `TypeError` 爆发。2026-09-12 的实际运行故障即属此类：Chat 附件功能中，`ChatApplicationService` 向 `PATCHOULI_PREPARE_AGENT_RUN` 传递 `attachments=`，而真实 handler `PatchouliService.prepare_agent_run` 的形参名为 `selected_attachments=`，总线直接调用在运行时抛出 `TypeError`（traceback 见分支提交记录）。

该 bug 逃过了全部既有测试，原因是测试接缝存在盲区：

- Chat 流单元测试为 `PATCHOULI_PREPARE_AGENT_RUN` 注册了 `**kwargs` 替身——任何 kwarg 名称都被静默接纳；
- prepare 层单元测试直接调用 `PatchouliService.prepare_agent_run`，使用的是正确的形参名；
- 两条链在"总线绑定真实 handler"这一接缝上没有交叉验证，形参改名（本例中 handler 侧从设计稿到实现演进出 `selected_attachments`）无法被任何测试捕获。

同类风险并不限于这一条路由：`GlobalRoutes` 上所有"调用方传 kwargs + handler 收 kwargs"的直接调用对（prepare/finalize/cleanup、gateway process、alice run、memory/topic 管理等）都依赖人工保持名称一致。

## 建议方案（待实现时细化）

建立一个集中的签名一致性校验，任选其一或组合：

1. **录制式契约测试**：以 SystemAssembler 真实装配（或相关子系统局部装配）注册全部生产 handler，再用一个校验型总线包装 `request()`——每次调用前用 `inspect.signature` 断言 kwargs ⊆ handler 形参且位置参数数量合法；随后复用既有 chat/upload 集成测试驱动流量，任何路由上的 kwargs 漂移在测试内直接失败；
2. **静态枚举对照**：对每条存在调用方的 route，在测试中枚举调用方传递的 kwargs 名称集合，断言其为 `inspect.signature(生产 handler)` 形参的子集；调用点清单需要人工维护，但无需启动真实子系统；
3. 两者结合：静态枚举保证已知调用点，录制式校验兜底未知调用路径。

## 约束

- 校验只做签名比对，不改变总线调用语义、不引入参数转换或默认值注入；
- handler 侧 `**kwargs` 形参（如有）视为豁免，不要求枚举；
- 替身测试可以继续存在，但每条涉及调用方新增 kwargs 的路由至少需要一个绑定真实 handler 的测试。

## 完成条件

- W1 附件故障场景（caller 传 `attachments`、handler 收 `selected_attachments`）在测试套件中必然失败，而非依赖真实运行发现；
- 校验覆盖至少 Chat → Patchouli 的 prepare/finalize/cleanup 三条路由，其余路由可增量接入；
- 既有测试套件在全量运行下保持通过，无新增误报。

## 触发记录

- 2026-09-12：运行时发现 `attachments` / `selected_attachments` 形参名不匹配（该 bug 已修复，回归测试见 `tests/integration/system/test_workspace_asset_chat_selection.py::test_chat_bus_route_reaches_real_prepare_with_attachments`）；本 todo 记录的是"此类接缝缺少系统性校验"的技术债本身。
