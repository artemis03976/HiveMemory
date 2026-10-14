---
title: engines 对上层的既有导入
status: todo
owner: project
scope: engines-known-upward-imports-in-package-layering
code_paths:
  - tests/unit/architecture/test_package_layers.py
  - src/hivememory/engines/
  - src/hivememory/patchouli/memory_library/
related_docs:
  - docs/ideas/workspace-network-task-process-architecture.md
last_reviewed: 2026-10-13
---

# engines 对上层的既有导入

## 问题与证据

包按 L0–L5 分层，依赖只能指向同层或更低层（[AGENTS.md](../../AGENTS.md) 第 3 节）。分层落地时没有处理 engines（L2）对上层的既有导入，它们作为已知例外登记在分层测试的 `KNOWN_UPWARD_IMPORTS` 中；测试要求实际导入与登记完全一致，规则禁止新增。现有 10 处（2026-10-13 核对）：

- `engines/artifacts`、`generation`、`lifecycle`、`retrieval` 共 9 处导入 `patchouli.memory_library`，主要是 `stores`；
- `engines/gateway` 1 处导入 `gateway.commands`。

## 影响范围

被导入的存储模型、协议与 Gateway 命令定义，以及上述 engines 模块与对应子系统之间的构造和注入关系。处理方式按每处的性质选择：

| 方式 | 内容 | 影响 |
|:---|:---|:---|
| 下移 | 把被导入的模型或协议下移到 L0/L1 | 需要区分存储实现与可以下移的协议、模型；与 core 的内容整理相互约束（[Workspace 网络与任务进程架构 Idea](../ideas/workspace-network-task-process-architecture.md)第 10 节） |
| 上移 | 把相关 engines 模块上移到对应子系统 | engines 不再承载这些算法，子系统内部职责变重 |
| 端口 | 由 engines 定义端口，子系统实现并注入 | 每处需要一个端口定义 |

## 完成条件

- `KNOWN_UPWARD_IMPORTS` 为空并从分层测试中删除；
- 被处理模块的行为测试回归通过，分层规则不再需要例外；
- [AGENTS.md](../../AGENTS.md) 第 3 节删除“engines 对上层的少量既有导入登记为已知例外”的说明。

## 追踪

- 2026-09-26：包分层实施时登记为已知例外，原记录于 Workspace 网络与任务进程架构 Idea 第 10 节；
- 2026-10-07：`engines/memory_compiler` 对 `agent_runtime.aliases` 的 2 处导入随引用解析迁入 workspace 删除；
- 2026-10-13：随该 Idea 重写转为本 Todo，未绑定版本或 Issue。
