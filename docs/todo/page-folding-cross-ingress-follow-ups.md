---
title: Page Folding Cross-Ingress Follow-ups
status: todo
owner: patchouli-system
scope: page-folding-context-ownership-and-evidence-debt
related_docs:
  - docs/patchouli/perception.md
  - docs/system/passive-ingress.md
  - docs/ideas/PatchouliPageFoldingRawEvidenceDesign.md
  - docs/ideas/long-running-agent-intra-turn-context-folding.md
last_reviewed: 2026-07-30
---

# Page Folding 跨入口上下文与证据后续技术债

## 问题与证据

Page Folding 已能在 overflow 后保留最近 blocks，但以下边界仍未形成完整契约：

- 当前只按 block 数保留后缀；单个超大 turn 或最近 blocks 本身可能继续超过 `fold_token_threshold`，因此它仍是软水位线。Turn 内多次 compact 的候选架构见[长时间运行 Agent 的 Turn 内上下文折叠](../ideas/long-running-agent-intra-turn-context-folding.md)；
- 公开配置拒绝 `fold_retain_recent_blocks=0`。底层 Store 已定义零值为“清空全部 blocks”，但 `TopicData.is_empty`、shutdown settlement、Generation identity 与 InteractionArtifact 尚不支持 summary-only topic；
- Passive Ingress 默认由外部 harness 管理 prompt history，公共响应只返回 retrieval memory。系统尚无 `external | hivememory` 这类显式上下文所有权能力，也没有带版本或覆盖游标的 compacted context 输出；
- active 与 passive 共享 `InteractionPayload` 和短期 topic，payload/block 没有稳定的 ingress origin 与 connector provenance，不能安全地按入口类型切换整个 topic 的 folding 策略；
- `TOKEN_OVERFLOW` 仍是 compact-only。被折叠旧前缀不会自动形成 settlement 或 raw evidence artifact，长时间运行的 passive conversation 尤其可能只剩有损摘要。完整旁路设想见 [Page Folding Raw Evidence Idea](../ideas/PatchouliPageFoldingRawEvidenceDesign.md)。

## 影响

- Page Folding 不能宣称严格保证模型 context 不溢出；
- 不能仅通过把保留数设为零来获得可靠的 summary-only 生命周期；
- Discord 等轻量 bot 无法直接把内部 folding 当作外部 prompt compact 服务；
- 若未来直接根据 `PASSIVE_MEMORY` 跳过 folding，Patchouli 内部 buffer 和 settlement generation 可能无界增长；
- overflow 前的高保真原始 turn 仍可能在正式记忆或 artifact 生成前被裁剪。

## 完成条件

- 决定并验证 count limit 与 token budget 的联合保留算法，包括单个超大 block 的处理和可观测事件；
- 若支持 retain zero，完整定义 summary-only topic 的非空判断、身份归属、shutdown/idle settlement、Generation 与 artifact 行为；否则持续在 schema 和当前文档中明确拒绝；
- 若支持 HiveMemory-managed bot context，形成独立 Plan，定义上下文所有权、版本/覆盖游标、返回模型及调用方替换历史的幂等契约；
- 在需要入口差异化策略前，先定义 interaction origin/provenance 及混合 topic 的冲突规则，不根据 connector 名称或单个事件临时推断；
- 为 folded prefix 选择 checkpoint settlement、append-only raw evidence 或其他耐久化路径，并定义写入失败、容量、隐私、删除与 shutdown 语义；
- 每个跨系统能力在实施前从本 Todo 拆出独立 Plan，并同步更新 Perception、Generation、Artifacts、Passive Ingress 与相关契约文档。
