---
title: Contracts
status: current
owner: system
scope: cross-subsystem-contracts
related_docs:
  - docs/architecture/workspace.md
last_reviewed: 2026-09-01
---

# Contracts

本目录是跨子系统稳定契约的唯一文档入口。它定义跨边界可观察的输入、输出、不变量、错误和兼容语义，也解释这些交接为何采用当前形式；它不复制各子系统内部实现，也不把函数目录当成契约说明。

Workspace 是这些交接的资源归属坐标；本目录只记录契约需要携带的 scope、边界拒绝和观测标签，完整 Workspace 设计见[Workspace 架构](../architecture/workspace.md)。

## 当前契约

- [子系统公共契约](./subsystem-contracts.md)：System、Gateway、Patchouli、Alice 的生命周期、输入输出和协作顺序；
- [公开路由与事件](./routes-and-events.md)：`GlobalSystemBus` 路由、业务事件和 RuntimeEvent 语义；
- [Memory Tool Protocol](./mtp.md)：MTP 语法、动词、权限、响应和运行时边界；
- [跨边界错误模型](./error-model.md)：业务终态、结构化错误、warning、控制异常与降级。

## 使用规则

- route 字符串以 `src/hivememory/system/contracts/route_names.py` 为代码级唯一来源；
- 协议模型以 `src/hivememory/core/protocol/` 和 `src/hivememory/core/mtp/` 为执行证据；
- local bus、workflow state 和具体引擎对象不是公共契约；
- 旧 `docs/protocols/` 已在逐篇审计后移入 `archive/legacy-docs/protocols/`，不得从 Archive 反向推断当前契约。

修改公共契约时，应在同一 PR 更新实现、调用方、契约测试和本目录对应文档。
