---
title: Research
status: current
owner: project
scope: external-systems-papers-and-technical-research
last_reviewed: 2026-11-01
---

# 调研资料

本目录集中保存对外部 Memory、Agent 系统、harness、论文和相关技术方案的调研需求与报告。需求规定研究范围、问题、取证与交付要求；报告积累来源、版本、机制比较、实验结果及对 HiveMemory 的启发。报告中的外部能力和适配建议不代表 HiveMemory 已有能力或已作出的设计决定。

调研同样遵守[文档治理规范](../DOCUMENTATION.md)的事实与计划分离和晋升门禁；本目录特有的规则见下文“编写规则”一节。

## 调研需求

| 需求 | 对应报告与验证状态 |
|:---|:---|
| [外部 harness 接口调研需求](./harness-interface-survey.md) | [2026-10-20 调研快照](./harness-interface-survey-snapshot.md)已完成资料调研；模型实测待验证，随 controller 模式退回候选方向暂缓（2026-10-20）。 |
| [DeepSeek Harness 插件接入调研需求与报告](./deepseek-harness-plugin-survey.md) | 同文第 7 节已形成 2026-11-01 证据快照；27 项能力、hook 对应、路径比较与接入要求已核对，无真实凭据机制探针通过，真实模型及 HiveMemory 端到端待验证。 |

## 报告索引

| 主题 | 报告 | 证据日期 | 范围与验证边界 |
|:---|:---|:---|:---|
| Agent harness 接口与接入 | [外部 harness 程序化接口调研快照](./harness-interface-survey-snapshot.md) | 2026-10-20 | Codex、Claude Code、pi 与 ACP；接口对比、会话、取消、输入、权限、MCP 与许可。资料核对及本地版本/离线 schema 检查；模型实测待验证。调研需求见[外部 harness 接口调研需求](./harness-interface-survey.md)。 |
| Agent harness 插件接入 | [DeepSeek Harness 插件接入调研](./deepseek-harness-plugin-survey.md#7-调研结果) | 2026-11-01 | dsh 0.2.2-alpha.2 固定源码、插件/MCP/会话/skill 与 HiveMemory 适配要求；本地脚本化模型及 HTTP MCP 探针通过，真实模型和后端链路待验证，接入路径尚未选择。 |
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

## 编写规则

**状态与元数据**：

- 报告编写时使用 `draft`，形成可引用的调研快照后使用 `baseline`，并填写 `snapshot_at`、`owner`、`scope`、`last_reviewed` 和相关文档入口；不使用 Idea 的 `horizon`。`snapshot_at` 是证据日期，`last_reviewed` 可以记录后来的分类或链接维护，不代表重新核验过外部系统。引用历史 HiveMemory 代码时，以 `source_commit` 标明对应的本地代码基线。
- 调研需求编写中使用 `draft`，有效时使用 `current`。需求的 `current` 只表示范围、取证和交付要求有效，不表示报告、实测或验收已经完成；未实测和未知项仍须保留。

**每份报告至少说明**：

- 研究问题、对象、范围和取证方法；
- 来源链接、版本或固定提交、访问日期与 `snapshot_at`；
- 已确认的外部事实、比较结果与关键限制；
- 文档声明、源码核对、实际运行分别验证了什么，未实测或无法确定的项明确标出；
- 对 HiveMemory 的适配分析、可借鉴之处与不适用边界；
- 仍需验证的问题，以及承接活动设计的 Idea、Plan 或其他相关文档。

**与本项目设计的边界**：调研可以记录分析和建议，但必须与外部事实区分；它不代表路线图承诺、owner 决定或 HiveMemory 当前能力。持续演进的本项目候选设计、目标接口和实施步骤放在 Idea 或 Plan，不能借本目录绕过晋升门禁。调研需求与对应报告同存于本目录并相互链接，发起调研的 Idea、Plan 或治理文档保留背景与入口；主要内容是开放假设或架构类比的文档归属 Idea。

**命名与冻结**：报告按主题使用英文小写 kebab-case 文件名。同一主题需要重新调查时，另建带日期的快照并注明替代关系，保留旧证据的时间边界。已冻结的报告只修复事实错误、断链和追溯元数据；外部项目发布新版本或本项目完成相应功能，都不构成改写旧调研结论的理由。`baseline` 只表示冻结已取得的证据与未知项，不表示所有实测均已完成。
