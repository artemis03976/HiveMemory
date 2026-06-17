# 记忆生命周期 (Memory Lifecycle)

该包负责处理记忆活跃度（vitality）的计算、强化（reinforcement）、归档（archival）和垃圾回收（garbage collection）。

当前的实现以 `MemoryLifecycleEngine` 为核心。除非在构建该引擎，否则调用方不会直接与垃圾回收器或计算器进行交互。

## 分数范围

- `confidence_score`（置信度）: `0.0-1.0`
- `vitality_score`（活跃度）: `0.0-100.0`

活跃度始终以 `0-100` 的范围进行存储和返回。UI 代码和存储过滤逻辑不应将活跃度再乘以 `100`。

## 核心组件

- `VitalityCalculator`：针对单个 `MemoryAtom` 的纯计算器。
- `DynamicReinforcementEngine`：应用生命周期事件并持久化强化后的记忆。
- `FileBasedArchiver`：将记忆从热存储移动到压缩的本地归档文件中。
- `PeriodicGarbageCollector`：扫描调用方提供的记忆，并将低活跃度的候选记忆进行归档。
- `MemoryLifecycleEngine`：负责编排刷新、强化、归档和垃圾回收（GC）的控制层。

## 当前事件路径

- `HIT`（命中）：在 `PatchouliService.finalize_agent_run()` 提交交互后记录。它会针对每个最终的检索结果进行去重，并使用 `source="retrieval.finalize"`。
- `CITATION`（引用）：由 MTP 的 `READ` 指令以及成功的用户记忆 `RUN` 指令通过 Patchouli 公共路由 `patchouli.public.record_memory_citation` 进行记录。
- `FEEDBACK_POSITIVE` / `FEEDBACK_NEGATIVE`（正/负反馈）：通过 HTTP 接口 `POST /api/v1/memories/{memory_id}/feedback` 进行记录。

强化机制在集成边界上采取“尽力而为（best-effort）”的策略：失败时仅会记录日志，且不应中断聊天或 MTP 响应，除非调用方明确要求严格执行。

## 活跃度刷新策略

活跃度刷新由调用方控制其作用域（caller-scoped）：

- 检索和记忆 API 的读取路径在返回记忆前会刷新它们的活跃度，通常使用 `persist=False`（不持久化）。
- `MemoryLifecycleEngine.run_garbage_collection()` 会加载所有活跃的记忆，使用 `persist=True` 刷新其活跃度，然后将刷新后的记忆集合异步传递给 `garbage_collector.collect(...)`。
- `PeriodicGarbageCollector` 本身不包含 `VitalityCalculator`；它仅处理那些 `meta.vitality_score` 已经被引擎或调用方刷新过的记忆。

这种设计避免了计算器与垃圾回收器之间的循环依赖，并确保引擎始终作为唯一的编排者。

## 垃圾回收

生命周期的定期清理（gardening）由全局维护调度器通过 `LibrarianCore.run_gardening_once()` 驱动。Patchouli 注册的任务如下：

- 归属（owner）: `patchouli`
- 任务名（name）: `memory_gardening`
- 回调（callback）: `runtime.librarian_core.run_gardening_once`

系统不再使用本地生命周期调度器、APScheduler 或 `ScheduledGarbageCollector`。

手动执行方式：

```python
archived_count = await lifecycle_engine.run_garbage_collection(force=True)
```

## 公共 API 接口

```python
lifecycle_engine.refresh_vitality(memory, persist=False)
lifecycle_engine.refresh_vitality_batch(memories, persist=False)
lifecycle_engine.record_hit(memory_id, source="retrieval.finalize")
lifecycle_engine.record_citation(memory_id, source="mtp.read")
lifecycle_engine.record_feedback(memory_id, positive=True, source="ui.memory_ref")
await lifecycle_engine.run_garbage_collection(force=False)
await lifecycle_engine.archive_memory(memory_id)
memory = await lifecycle_engine.resurrect_memory(memory_id)
```

`VitalityCalculator.calculate(memory)` 保持纯粹：它仅计算分数，不负责持久化。

## 待办事项

- 持久化或暴露每个用户的反馈状态，以便 UI 能够在跨会话中显示历史反馈。
- 添加生命周期事件历史/调试接口，以提高运维可见性。
- 在活跃度和置信度作为排序质量因子时，继续验证检索融合（retrieval fusion）的实际表现。
- 更新旧版的高层设计文档，将其中描述的计划中的生命周期行为修正为当前已实现的行为。
