---
title: Frontend Memory Library Listing and Search
status: archived
archived_at: 2026-11-02
implemented_by: PR #171
superseded_by: docs/frontend/management-views.md
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

> 状态：Archived（2026-11-02，已按默认做法修复，处置见文末一节）。当前事实见[管理页面](../../frontend/management-views.md#11-当前能力)第 1 节、[MemoryLibrary](../../patchouli/memory-library.md)与[路由与事件](../../contracts/routes-and-events.md#23-patchouli-memory)第 2.3 节。以下为登记时的原文，相对链接已随归档改正。

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
- **管理页面的两项遗留**（[管理页面](../../frontend/management-views.md#13-当前偏差)第 1.3 节）：
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
- 更新[管理页面](../../frontend/management-views.md)第 1 节与第 5 节。

## 非目标

- 所有页面 mock fallback 的统一披露，见[统一前端 mock fallback 的状态披露](../../todo/frontend-mock-fallback-disclosure.md)；
- 可见性策略的编辑，见 [Memory 可见性策略前端组件](../../todo/memory-visibility-policy-ui.md)；
- 归档记忆的浏览与复活。

## 处置（2026-11-02）

按默认做法修复，PR #171。

- **后端**：
  - 列表接口 `GET /api/v1/memories` 新增 `cursor` 参数，响应新增 `next_cursor`；`total` 改为库内总数。
  - Patchouli 的 `memory.list` 与 `patchouli.public.memory.list` 改为返回 `MemoryListPage`（一页记忆、下一页游标与总数）。
  - 中期库的 scroll 跟随 Qdrant 的 `next_page_offset` 翻页，新增按同一过滤条件的计数。
  - 业务过滤新增排除类型，在 Qdrant 的 `must_not` 中执行；管理服务原先取回后排除 Agent Profile 的那一步随之删除。
  - Agent Profile 管理列表取这条路由的第一页，上限仍是 100 条。
- **前端**：
  - 加载时跟随 `next_cursor` 取全并在浏览器中排序，页面头部显示库内总数，未取全时提示已加载数。
  - 搜索分为语义检索（停止输入 400 毫秒后以 `query` 请求后端，结果保持相关度顺序，显示检索中、失败与无结果）与本地过滤（`alias:`、`tag:` 前缀）。
  - 删除失败时把记忆放回并提示；`statusFilter` 删除。
- **测试**：
  - [管理列表集成测试](../../../tests/integration/patchouli/test_memory_management_listing.py)经真实的管理服务、local bus、RetrievalFamiliar 与 Qdrant 内存模式，验证翻页取全且每条一次、总数不越过 Workspace，以及排除的类型既不列出也不计数。临时让游标不下传或去掉类型排除时，对应用例失败。
  - [前端列表客户端测试](../../../frontend/tests/memory-api.test.mjs)验证跟随游标取完全部页；只取一页时失败。
  - 路由、门面、适配器与存储层的既有测试随契约改写。
- **实测**：以新代码只读访问当时的 Qdrant，库内 157 条记忆分 100 与 57 两页取全，没有重复，不含 Profile，最新的记忆都在其中。

已知限制写在[管理页面](../../frontend/management-views.md#13-当前偏差)第 1.3 节：整库载入浏览器，记忆数到数千条时首次加载变慢；页面只在进入和创建记忆后重新加载。
