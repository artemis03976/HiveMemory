---
title: Legacy Engines Documentation Index
status: superseded
owner: project
scope: legacy-engine-centric-documentation-index
archived_at: 2026-07-28
superseded_by:
  - docs/patchouli/README.md
  - docs/gateway/README.md
---

> `engines/` 仍是代码中的算法目录，但不再作为与子系统并行的当前文档树。记忆引擎从 [Patchouli 当前文档](../../../patchouli/README.md)进入，入口决策从 [Gateway 当前文档](../../../gateway/README.md)进入；本索引只保留迁移前的目录背景。

# Engines

本目录用于存放 HiveMemory 现有记忆引擎的实现说明与能力边界文档。

## 当前收录

- [Gateway 当前索引](../../../gateway/README.md)：Gateway / Router / Application Service 入口链路（旧 `gateway.md` 尚待其所属批次处理）。
- [perception.md](./perception.md): 感知、摄入与记忆构建链路。
- [generation.md](./generation.md): 主动生成与写入决策链路。
- [retrieval.md](./retrieval.md): 检索、融合与上下文编译链路。
- [lifecycle.md](./lifecycle.md): 生命周期、强化、归档与回收链路。
- [memory_compiler](./memory_compiler/README.md): MemoryCompiler 表达收敛与渲染边界文档。
