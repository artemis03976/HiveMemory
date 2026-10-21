---
title: Research
status: current
owner: project
scope: external-systems-papers-and-technical-research
last_reviewed: 2026-10-20
---

# 调研资料

本目录集中保存对外部 Memory、Agent 系统、harness、论文和相关技术方案的调研需求与报告。需求规定研究范围、问题、取证与交付要求；报告积累来源、版本、机制比较、实验结果及对 HiveMemory 的启发。报告中的外部能力和适配建议不代表 HiveMemory 已有能力或已作出的设计决定。

分类、取证要求、元数据与快照维护规则统一见[文档治理规范第 8.6 节](../DOCUMENTATION.md#86-research)。编写中的需求或报告使用 `draft`，有效需求使用 `current`，报告形成证据快照后使用 `baseline`；需求有效不表示实测已完成，未实测和未知项仍须保留。

## 调研需求

| 需求 | 对应报告与验证状态 |
|:---|:---|
| [外部 harness 接口调研需求](./harness-interface-survey.md) | [2026-10-20 调研快照](./harness-interface-survey-snapshot.md)已完成资料调研；模型实测待验证，随 controller 模式退回候选方向暂缓（2026-10-20）。 |

## 报告索引

| 主题 | 报告 | 证据日期 | 范围与验证边界 |
|:---|:---|:---|:---|
| Agent harness 接口与接入 | [外部 harness 程序化接口调研快照](./harness-interface-survey-snapshot.md) | 2026-10-20 | Codex、Claude Code、pi 与 ACP；接口对比、会话、取消、输入、权限、MCP 与许可。资料核对及本地版本/离线 schema 检查；模型实测待验证。调研需求见[外部 harness 接口调研需求](./harness-interface-survey.md)。 |
| Agent 上下文与记忆管理 | [Agent Harness 会话压缩技术调研](./agent-context-compaction-research.md) | 2026-10-16 | Claude Code、Codex、Hermes、pi、OpenCode V2 的固定源码/文档快照；预算、历史投影、检查点、取消与恢复。保留当时的适配建议，未做真实长会话质量对比。 |

新报告按主题加入本索引；Memory 系统设计、Agent 架构、检索与存储、论文与算法等方向均可在此积累。报告增多后再按稳定主题拆分子目录。

## 与其他目录的关系

| 内容 | 归属 |
|:---|:---|
| 外部系统或论文的实际调研、横向比较与实验记录 | 本目录 |
| 独立的调研需求、取证范围与交付要求 | 本目录，与对应报告相互链接 |
| HiveMemory 的开放假设、架构类比与候选设计 | [Ideas](../ideas/README.md) |
| 已绑定版本、可验收的实施方案 | [Plans](../plans/README.md) |
| 对 HiveMemory 自身治理问题的代码调查和风险基线 | [Governance Baselines](../governance/baselines/README.md) |
| HiveMemory 当前职责、契约与实现行为 | [项目总览](../PROJECT.md)中的当前设计入口 |

AE2 架构类比、TDA 研究假设等仍保留在 Ideas，由各自文档继续维护开放问题。
