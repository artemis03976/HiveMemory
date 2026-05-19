# HiveMemory 架构演进文档索引

本目录用于集中存放 HiveMemory 各次重要架构演进的规划文档，避免架构草案长期散落在 `docs/mod/` 或协议文档中，便于后续统一回顾、对比与继续演进。

## 当前收录

- [第二次架构演进：SystemArchitecture_v2.0](file:///c:/Users/29305/Projects/HiveMemory/docs/architecture_evolution/SystemArchitecture_v2.0.md)
  - 关键词：帕秋莉三位一体、冷热路径分层、记忆基础设施重构

- [第三次架构演进：SystemArchitecture_v3.0](file:///c:/Users/29305/Projects/HiveMemory/docs/architecture_evolution/SystemArchitecture_v3.0.md)
  - 关键词：Patchouli Kernel、MTP、Koakuma、递归中断运行时、Patchouli OS
  - 说明：本文件从 [`MemoryToolProtocol.md`](file:///c:/Users/29305/Projects/HiveMemory/docs/MemoryToolProtocol.md) 中与“系统重构与运行时架构”直接相关的内容独立整理而来，用于承载第三次架构演进的系统级视图；MTP 协议细节仍以原文档为准。

- [第四次架构演进：SystemArchitecture_v4_TopLevelSketch](file:///c:/Users/29305/Projects/HiveMemory/docs/architecture_evolution/SystemArchitecture_v4_TopLevelSketch.md)
  - 关键词：顶层系统层、Patchouli 子系统、Alice 子系统、系统运行时分层

- [第四次架构演进：SystemArchitecture_v4_PhaseD_ChatApplicationServiceMigration_Design](file:///c:/Users/29305/Projects/HiveMemory/docs/architecture_evolution/SystemArchitecture_v4_PhaseD_ChatApplicationServiceMigration_Design.md)
  - 关键词：Phase D、ChatApplicationService、chat 编排迁移、Alice 运行时、Patchouli 记忆域边界

- [第四次架构演进补充：SystemArchitecture_v4_RuntimeConvergence_Addendum](file:///c:/Users/29305/Projects/HiveMemory/docs/architecture_evolution/SystemArchitecture_v4_RuntimeConvergence_Addendum.md)
  - 关键词：PatchouliRuntime、AliceRuntime、System-Service-Runtime、长期收敛、运行环境宿主

- [第四次架构演进补充：SystemArchitecture_v4_AliceRuntimeConvergence_Design](file:///c:/Users/29305/Projects/HiveMemory/docs/architecture_evolution/SystemArchitecture_v4_AliceRuntimeConvergence_Design.md)
  - 关键词：AliceRuntime、AgentRuntime、KoakumaRuntime、运行时收敛、双 runtime 分层

## 约定

- 本目录优先存放“架构演进”级别文档，而不是单个组件或一次局部重构的设计草案
- 协议、组件、前后端、专题研究类文档继续保留在各自原有目录
- 若某次演进的核心内容最初写在专题文档中，建议在本目录额外整理一份独立的系统视角文档，便于纵向对比
