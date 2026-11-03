---
title: Frontend Memory Library Listing and Search
status: todo
owner: frontend
scope: memory-library-full-listing-total-ordering-and-search
code_paths:
  - frontend/src/hooks/useMemories.ts
  - frontend/src/services/memoryApi.ts
  - frontend/src/stores/memory/viewStore.ts
  - src/hivememory/server/routers/memories.py
  - src/hivememory/workspace/capability/root/memory.py
  - src/hivememory/patchouli/application/memory_management_service.py
  - src/hivememory/patchouli/services/retrieval.py
  - src/hivememory/infrastructure/storage/vector_store.py
related_docs:
  - docs/frontend/management-views.md
  - docs/patchouli/memory-library.md
  - docs/todo/frontend-mock-fallback-disclosure.md
  - docs/todo/memory-visibility-policy-ui.md
last_reviewed: 2026-11-02
---

# Memory 管理页列表不全与检索

## 问题与证据（2026-11-02）

库内有 143 条记忆（不含 Agent Profile），Memory Library 只显示 98 条，最新的几条看不到：

- 看不到的：`user_research_prefers_mechanism`、`ref_dsh_memory_quality_review` 等；
- 能看到的最新一条是 `ref_clean_build_artifacts`。

Memory Library 目前是 owner 唯一的记忆管理入口。

原因已对照 Qdrant 与 `GET /api/v1/memories?limit=100` 的实际返回核实：

1. **只取一页**：页面在加载时只请求一次 `limit=100`（`useMemories`），路由的 `limit` 上限也是 100，没有分页或游标。
2. **这一页与时间无关**：后端的 scroll 不带 `order_by`（`vector_store._scroll`），返回 Qdrant 按 point ID 排列的第一页，并丢弃 `next_page_offset`。point ID 与创建时间无关，库超过 100 条后，页面拿到的是一个与时间无关的子集。前端按创建时间排序，排的也只是这个子集。
3. **取完再排除 Profile**：Agent Profile 在取回 100 条之后才排除（`MemoryManagementService.list_memories` 的 `exclude_types`）。这一页里有 2 条 Profile，于是只剩 98 条。
4. **没有真实总量**：`MemoryListResponse.total` 是本次返回的条数，页面头部的数字是前端过滤后的条数。两者都不是库的总量，页面无法提示还有记忆没有加载。

实测结果：

- 集合里共 145 个 point，即 143 条记忆加 2 条 Profile；
- API 返回的恰好是 scroll 第一页的 100 个 point 去掉 2 条 Profile；
- 未显示的 45 条分布在 2026-02 到 2026-11 的各个日期，其中 2026-11-01 之后创建的有 39 条。

## 一并处理的事项

- **语义检索**：原 Todo「Memory Garden 接入真实语义检索」于 2026-11-02 并入本 Todo。semantic 模式只在已加载的列表上做 substring 过滤；而列表接口已有 `query` 参数，由中期库在整个 Workspace 内做向量检索，前端没有使用。修列表与修检索改的是同一条请求链路，所以一起处理。
- **管理页面的两项遗留**（[管理页面](../frontend/management-views.md#13-当前偏差)第 1.3 节）：
  - `statusFilter` 持久化在 store 中，但没有界面，也不参与过滤；归档的记忆已移出中期库，列表接口本来就取不到。
  - 删除先在页面上移除，API 失败后不回滚。

## 做法

**最简版本（默认）**

- 后端：
  - 列表接口支持游标分页，沿用 Qdrant 按 point ID 的分页（`next_page_offset`）作为游标；
  - Agent Profile 的排除移进 Qdrant 的过滤条件，每页条数不再因事后过滤而减少；
  - `total` 改为同一过滤条件下的计数（Qdrant count）。
- 前端：首次加载和写操作之后翻页取全，在浏览器中沿用现有逻辑排序与过滤；页面头部显示库的总量。
- 暂不处理：整个库载入浏览器，记忆数到数千条时加载会变慢；排序仍在前端完成。

**可选的更完整版本**：服务端按创建时间或更新时间排序，页面按页加载。代价：

- Qdrant 的 `order_by` 要求该字段有支持 range 条件的 payload index，当前集合没有任何 payload index，需要单独安排迁移；
- 使用 `order_by` 时不返回 `next_page_offset`，分页要改用 `start_from` 加上排除已读 ID 的过滤条件。

## 完成条件

- 库内记忆超过单页上限时，页面能看到全部记忆；默认排序下，最新创建的记忆排在最前；
- 列表接口每页的条数不因 Profile 排除而减少，并返回库的真实总量；页面显示总量与已加载的数量；
- semantic 模式调用后端检索（列表接口的 `query` 或 Retrieval route），覆盖整个 Workspace；界面区分语义检索与本地过滤，并呈现加载、失败与空结果三种状态；
- `statusFilter` 删除，或接上真实的语义；删除失败时回滚并提示；
- Patchouli 的 `MEMORY_LIST` route 在参数或返回上的变化同步到契约与全部调用方，包括 Agent Profile 管理列表；
- 后端测试覆盖三项：超过一页时取全、排除 Profile 不减少页内条数、总量计数；
- 前端测试基础设施只有 `node --test` 覆盖的少数模块；交互行为没有条件写测试的，手动验证并在交付说明中写明；
- 更新[管理页面](../frontend/management-views.md)第 1 节与第 5 节。

## 非目标

- 所有页面 mock fallback 的统一披露，见[统一前端 mock fallback 的状态披露](./frontend-mock-fallback-disclosure.md)；
- 可见性策略的编辑，见 [Memory 可见性策略前端组件](./memory-visibility-policy-ui.md)；
- 归档记忆的浏览与复活。
