---
title: Contracts
status: draft
owner: system
scope: cross-subsystem-contracts
last_reviewed: 2026-07-27
---

# Contracts

本目录是跨子系统稳定契约的目标入口。它只定义跨边界可观察的输入、输出、不变量、错误和兼容语义，不复制各子系统内部实现。

计划收录：

- `subsystem-contracts.md`：System、Patchouli、Alice、Gateway 的责任和调用边界；
- `routes-and-events.md`：全局/局部路由、事件信封和发布语义；
- `mtp.md`：Memory Tool Protocol；
- `error-model.md`：跨边界错误、warning 和降级表达。

现有 `docs/protocols/` 将在核对实现后迁入或合并到本目录。在迁移完成前，旧协议文档的 `Draft` 或 `Implemented` 自述不能替代代码与测试验证。
