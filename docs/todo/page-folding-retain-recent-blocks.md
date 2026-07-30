---
title: Page Folding Retain Recent Blocks
status: todo
owner: patchouli
scope: perception-fold-retain-recent-blocks
related_docs:
  - docs/patchouli/perception.md
  - docs/ideas/PatchouliPageFoldingRawEvidenceDesign.md
last_reviewed: 2026-07-29
---

# Page Folding 接通保留最近 blocks 配置

## 问题与证据

`Perception` 已有 `fold_retain_recent_blocks` 配置，但当前 `TOKEN_OVERFLOW` 仍由 RelayController 生成 `state_summary` 后清空全部 blocks，配置没有进入 TriggerManager 的裁剪结果。当前行为与设计目标“折叠旧页、保留有限工作集”不一致，见[感知与短期话题](../patchouli/perception.md)第 5 节。

这项 Todo 只处理 active buffer 中保留最近 blocks 的局部语义，不等于实现 raw-evidence side-channel；后者继续留在 [Page Folding Raw Evidence Idea](../ideas/PatchouliPageFoldingRawEvidenceDesign.md)。

## 影响

- 配置调用方无法通过 `fold_retain_recent_blocks` 控制 overflow 后的短期工作集；
- overflow 前尚未 settlement 的最近 turns 也会被丢弃，只剩 `state_summary`；
- 不同配置值的运行行为没有可验证差异。

## 完成条件

- 明确 `retain_count=0`、大于 blocks 数量和负值配置的行为；
- `TOKEN_OVERFLOW` 在生成 summary 后只移除应折叠部分，保留的 blocks 继续参与下一轮 topic context；
- 不改变 IDLE/LRU/SHUTDOWN/MANUAL 的现有 settle/evict 语义；
- 增加配置、token、summary、后续 ingest 和 shutdown 的单元/E2E 测试；
- 更新 `docs/patchouli/perception.md` 的矩阵、当前限制和配置说明。
