# Patchouli Transcript Phase 4 Cleanup Plan

## 1. 状态说明

本文件已不再作为独立的进行中文档维护。

原本记录在这里的 Phase 4 清理计划，已经随着本轮 transcript / 消息流重构的推进基本完成，并已合并进以下权威文档：

- `docs/mod/PatchouliTranscriptDualViewRefactor.md`

---

## 2. 为什么不再单独维护

继续保留一份独立的 Phase 4 计划文档会带来两个问题：

1. 很容易继续把“已经完成的清理项”误写成待办
2. 会让设计草案、阶段计划、当前实现三者再次发生漂移

因此当前策略是：

- 设计、落地现状、已完成清理项、剩余可选收尾项
- 全部统一收口到 `PatchouliTranscriptDualViewRefactor.md`

---

## 3. 这份文件现在保留什么

这份文件只保留两个作用：

- 作为历史路径的兼容占位，避免旧链接失效
- 明确告诉后续维护者：请不要再把它当作当前实施计划继续扩写

---

## 4. 如需查看什么内容

请统一查看：

- 当前模型分层
- 当前主动 / 被动模式链路
- 已完成的兼容层清理
- 仍保留的有限兼容层
- 后续可选清理项

以上内容均已迁移到：

- `docs/mod/PatchouliTranscriptDualViewRefactor.md`

---

## 5. 结论

`PatchouliTranscriptPhase4CleanupPlan.md` 现已归档。

后续与 transcript / 消息流重构相关的说明、设计与收尾，都应只更新：

- `docs/mod/PatchouliTranscriptDualViewRefactor.md`
