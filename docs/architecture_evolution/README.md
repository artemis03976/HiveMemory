# HiveMemory 架构演进文档索引

本目录用于集中存放 HiveMemory 各次重要架构演进的系统级文档，便于后续统一回顾、对比与继续演进。

## 当前收录

- [第二次架构演进：SystemArchitecture_v2.0](./SystemArchitecture_v2.0.md)
  - 关键词：Patchouli 三位一体、冷热路径分层、记忆基础设施重构

- [第三次架构演进：SystemArchitecture_v3.0](./SystemArchitecture_v3.0.md)
  - 关键词：Patchouli Kernel、MTP、Koakuma、递归中断运行时、Patchouli OS
  - 说明：本文用于承载第三次架构演进的系统级视图；MTP 协议细节仍以 [MemoryToolProtocol.md](../MemoryToolProtocol.md) 为准。

- [第四次架构演进最终总纲：SystemArchitecture_v4_TopLevelSketch](./SystemArchitecture_v4_TopLevelSketch.md)
  - 状态：Final (已收敛)
  - 说明：文件名保留 `TopLevelSketch` 仅为兼容既有链接；该文档已不再是草图。
  - 关键词：HiveMemorySystem、System-Service-Runtime、PatchouliRuntime、AliceRuntime、AgentRuntime、KoakumaRuntime、ChatApplicationService、结构化 turn events

## v4 归档参考

以下文档是第四次架构演进过程中的阶段设计记录，保留用于追溯迁移背景与取舍。若它们与 v4 最终总纲冲突，以最终总纲为准。

- [Phase B 历史方案](./SystemArchitecture_v4_PhaseB_Design.md)
- [Patchouli Subsystem Normalization 设计](./SystemArchitecture_v4_PatchouliSubsystemNormalization_Design.md)
- [Phase C Alice Runtime Foundation 设计](./SystemArchitecture_v4_PhaseC_AliceRuntimeFoundation_Design.md)
- [Phase D ChatApplicationService Migration 设计](./SystemArchitecture_v4_PhaseD_ChatApplicationServiceMigration_Design.md)
- [Runtime Convergence Addendum](./SystemArchitecture_v4_RuntimeConvergence_Addendum.md)
- [Alice Runtime Convergence 设计](./SystemArchitecture_v4_AliceRuntimeConvergence_Design.md)

## 约定

- 本目录优先存放“架构演进”级别文档，而不是单个组件或一次局部重构的设计记录。
- 协议、组件、前后端、专题研究类文档继续保留在各自原有目录。
- 每次架构演进应有一份最终总纲文档；阶段设计文档在演进完成后应标记为归档参考。
