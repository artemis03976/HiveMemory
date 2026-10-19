---
title: Governance Baselines
status: current
owner: project
scope: point-in-time-governance-evidence
last_reviewed: 2026-10-19
---

# Governance Baselines

本目录保存为跨版本治理建立的阶段性调研基线。每份基线冻结某个日期的代码入口、风险矩阵、现状分级和后续优先级；它是决策证据，不是当前设计文档，也不表示对应治理阶段已经完成。

当前基线：

- [Phase D0 持久化状态清单](./durability-d0-state-inventory.md)；
- [Phase I0 业务操作清单](./idempotency-i0-operations-inventory.md)；
- [Phase S0 身份与威胁模型清单](./identity-s0-threat-model-inventory.md)；
- [数据模型 Phase I 清单](./data-model-phase-i-inventory.md)；
- [测试冗余后续追查（2026-10-16）](./test-suite-redundancy-2026-10-16.md)（83 个参数函数与测试归属核查，待处理项见对应 Todo）；
- [v0.7.0 代码结构成因调查（2026-10-18）](./v0.7.0-code-structure-causes-2026-10-18.md)（v0.6.2 以来代码结构为何显得乱的五个成因与证据；执行流成因由对应计划处理，其余未排期）。

基线原则上冻结。代码继续演进后，由当前设计文档描述最新事实；只有严重错误、断链或追溯元数据需要修正时才修改旧基线。若治理阶段需要新的全量调查，应建立带新阶段或日期的新基线，并显式链接其替代关系。
