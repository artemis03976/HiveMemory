---
title: engines 对上层的既有导入
status: archived
archived_at: 2026-11-09
superseded_by: docs/architecture/overview.md
owner: project
scope: engines-known-upward-imports-in-package-layering
code_paths:
  - tests/unit/architecture/test_package_layers.py
  - src/hivememory/engines/
  - src/hivememory/memory_library/
related_docs:
  - docs/patchouli/memory-library.md
  - docs/gateway/commands.md
  - docs/ideas/fifth-generation-architecture.md
last_reviewed: 2026-11-09
---

# engines 对上层的既有导入

> 状态：Archived（2026-11-09）。owner 决定（2026-11-09）：不为这些导入新增端口协议，因为端口要多次跳转才能读懂；改为把被导入的包下移。当前事实见[系统架构概览](../../architecture/overview.md)的分层表与 [MemoryLibrary](../../patchouli/memory-library.md)。以下为登记时的内容，链接已按归档位置调整。

## 状态

**已完成。** 10 处导入都只用于类型标注，engines 实际用到的是 7 个存储方法与 2 个命令注册表方法。

- 记忆存储层从 `patchouli/memory_library` 下移为 L2 的 `hivememory.memory_library`：它只依赖 core、engines 的模型与编解码和 infrastructure，使用方只有 engines 与 Patchouli，放在 Patchouli 包内时 engines 的 9 处用法都成了向上导入。分层测试新增规则，只允许 Patchouli 与 engines 导入它，所有权仍由 Patchouli 持有。
- Gateway 命令包从 `gateway/commands` 下移到 `engines/gateway/commands`：它只依赖 core，与入口拦截放在一起。
- `KNOWN_UPWARD_IMPORTS` 删除，分层测试断言不存在任何向上导入；[AGENTS.md](../../../AGENTS.md) 第 3 节的已知例外说明改为存储层的使用方约束。

## 问题与证据

包按 L0–L5 分层，依赖只能指向同层或更低层（[AGENTS.md](../../../AGENTS.md) 第 3 节）。分层落地时没有处理 engines（L2）对上层的既有导入，它们作为已知例外登记在分层测试的 `KNOWN_UPWARD_IMPORTS` 中；测试要求实际导入与登记完全一致，规则禁止新增。现有 10 处（2026-10-13 核对）：

- `engines/artifacts`、`generation`、`lifecycle`、`retrieval` 共 9 处导入 `patchouli.memory_library`，主要是 `stores`；
- `engines/gateway` 1 处导入 `gateway.commands`。

## 影响范围

被导入的存储模型、协议与 Gateway 命令定义，以及上述 engines 模块与对应子系统之间的构造和注入关系。处理方式按每处的性质选择：

| 方式 | 内容 | 影响 |
|:---|:---|:---|
| 下移 | 把被导入的模型或协议下移到 L0/L1 | 需要区分存储实现与可以下移的协议、模型；与 core 的内容整理相互约束（[第五代架构 Idea](../../ideas/fifth-generation-architecture.md)第 5 节 C） |
| 上移 | 把相关 engines 模块上移到对应子系统 | engines 不再承载这些算法，子系统内部职责变重 |
| 端口 | 由 engines 定义端口，子系统实现并注入 | 每处需要一个端口定义 |

## 完成条件

- `KNOWN_UPWARD_IMPORTS` 为空并从分层测试中删除；
- 被处理模块的行为测试回归通过，分层规则不再需要例外；
- [AGENTS.md](../../../AGENTS.md) 第 3 节删除“engines 对上层的少量既有导入登记为已知例外”的说明。

## 追踪

- 2026-09-26：包分层实施时登记为已知例外，原记录于第五代架构 Idea 的前身（《Workspace 网络与任务进程架构》Idea）第 10 节；
- 2026-10-07：`engines/memory_compiler` 对 `agent_runtime.aliases` 的 2 处导入随引用解析迁入 workspace 删除；
- 2026-10-13：随该 Idea 重写转为本 Todo，未绑定版本或 Issue；
- 2026-11-09：存储层与 Gateway 命令包下移，例外清空，归档。
