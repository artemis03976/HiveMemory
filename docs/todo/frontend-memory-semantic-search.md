---
title: Frontend Memory Semantic Search
status: todo
owner: frontend
scope: memory-garden-search
related_docs:
  - docs/frontend/management-views.md
last_reviewed: 2026-07-29
---

# Memory Garden 接入真实语义检索

## 问题与证据

Memory Garden 当前所谓 semantic 模式仍是在前端已加载列表上执行本地 substring 过滤，并没有请求 Patchouli 的全局记忆检索。因此它只能找到当前页已加载且字面匹配的内容，名称会让用户误以为已经进行了向量/混合检索。

当前行为与代码入口见[管理页面设计](../frontend/management-views.md)。

## 影响

- 大型记忆库中未加载到前端的内容不可发现；
- 近义表达和摘要语义无法命中；
- UI 对检索能力的描述与真实行为不一致。

## 完成条件

- 明确调用现有 Retrieval route，或为管理视图建立受身份约束的搜索契约；
- 查询覆盖全局可见记忆库，而非只过滤当前列表；
- UI 区分语义检索、结构化筛选与本地过滤，并呈现加载/失败/空结果状态；
- 请求携带当前 Identity，结果继续遵守 MemoryLibrary 可见性边界；
- 增加前端交互测试与后端契约测试，并更新 `docs/frontend/management-views.md`。

本项只处理搜索真实性，不扩展为完整的记忆管理重构。
