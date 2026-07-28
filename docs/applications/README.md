---
title: HiveMemory Applications
status: current
owner: product
scope: product-specifications-and-validation-evidence
updates:
  - docs/PROJECT.md
  - docs/ROADMAP.md
last_reviewed: 2026-07-28
---

# HiveMemory Applications

本目录保存建立在 HiveMemory 之上的产品规格、用户场景和验证证据。应用文档回答“系统能否在一个真实问题中产生持续价值”，但不反向定义后端架构，也不因为写出了完整验收步骤就把尚未运行的试验描述成已经交付的产品。

## 当前应用

- [三餐推荐助手](./MealAssistantProductSpec.md) — `planned`。已形成 MVP 定位、Agent Profile、system prompt 与一周验证方案；仓库中尚无独立应用包、专用 UI 或真实用户验收记录。

## 应用文档的三层事实

每份规格都应明确区分：

- **当前依赖**：已经由 System、Patchouli、Alice、Gateway、Contracts 或 Frontend 当前文档证明的能力；
- **试验假设**：需要通过真实用户任务验证的产品判断，例如“跨会话记忆会让用户感到它记得我”；
- **未来依赖**：尚未落地、不能作为当前验收前提的功能。

应用中的 persona、prompt、样例和成功标准可以保持具体，因为它们是可执行试验设计；没有实现包、测试记录或用户证据时，状态仍应保持 `planned`。后端事实分别以 [Project](../PROJECT.md)、[当前架构](../architecture/overview.md)和[跨子系统契约](../contracts/README.md)为准。
