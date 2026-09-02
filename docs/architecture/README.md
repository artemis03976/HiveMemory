---
title: Architecture
status: current
owner: system
scope: architecture-index
last_reviewed: 2026-09-01
---

# Architecture

本目录只索引当前生效的总体架构、系统边界、数据模型和架构决策。这里的文档不仅说明组件如何连接，也应保存边界为何形成、状态为何这样分配，以及评审新改动时应检查哪些设计矛盾。架构演进材料已经退出当前设计入口。

## 当前设计

- [系统架构概览](./overview.md)：组合根、三个子系统、主动/被动链路与启停顺序；
- [系统边界与所有权](./boundaries.md)：职责、状态所有权、依赖方向与禁止越界；
- [数据模型与可变性边界](./data-model.md)：MemoryAtom 语义结构、模型角色、冻结深度、受控可变与边界投影；
- [Workspace 架构](./workspace.md)：身份坐标、资源归属、WorkspaceAssetStore、Topic binding 与运行时生命周期；
- [架构决策记录](./decisions/README.md)：长期有效的设计决策及其理由。

## 相关入口

- [跨子系统契约](../contracts/README.md)
- [项目总览](../PROJECT.md)
- [开发路线图](../ROADMAP.md)
- [数据模型可变性治理](../governance/data-model/mutability.md)
- [历史架构](../archive/legacy-architecture/README.md)
