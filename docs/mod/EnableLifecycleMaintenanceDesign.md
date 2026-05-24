# 启用生命周期维护设计

## 1. 目标

本文档定义了在生产路径中启用 HiveMemory 长期记忆生命周期系统的实施计划。

目标工作涵盖三个相互关联的能力：

- 由全局异步维护调度器 (asyncio maintenance scheduler) 驱动的 Librarian 定期维护模式。
- 将检索命中 (Retrieval hit) 和显式记忆使用的反馈融入生命周期活力 (vitality) 评分中。
- 调整记忆活力和置信度 (confidence) 的用户反馈路径。

首次实现应当使生命周期循环具有可观察性和可测试性，不应引入第二个调度器、独立的作业框架或对记忆生成进行大规模更改。

## 2. 现状

代码库中已经包含了大部分底层组件：

- `HiveMemorySystem` 创建了一个 `GlobalMaintenanceScheduler`。
- `PatchouliSystem.register_maintenance_tasks()` 注册了 `perception_idle_flush`。
- `LibrarianCore.run_gardening_once()` 已作为全局调度器回调入口。
- `MemoryLifecycleEngine` 暴露了 `record_hit`、`record_citation`、`record_feedback` 和 `run_garbage_collection` 方法。
- `PeriodicGarbageCollector` 扫描调用方传入的、已刷新生命力的记忆并将其归档。
- 检索过程返回最终的 `RetrievalResponse` / `RetrievalResult`，HIT 已在 `finalize_agent_run()` 中 best-effort 记录。
- 前端右侧引用记忆栏和 API 已具备单记忆反馈至生命周期的路线。

主要剩余工作是可观测性、反馈语义持久化和检索融合调优。

## 3. 设计原则

### 3.1 仅使用全局调度器

生命周期维护任务必须注册到现有的 `GlobalMaintenanceScheduler` 中。

不要重新启用 `ScheduledGarbageCollector` 或添加 APScheduler。旧的 `ScheduledGarbageCollector` 实现已经移除；生命周期维护只通过全局维护调度器驱动。

### 3.2 将业务逻辑保留在生命周期中

调度器应仅决定任务何时运行。任务回调应该调用面向生命周期的方法，例如 `LibrarianCore.run_gardening_once()`。

调度器不应了解活力阈值、归档格式、记忆评分或反馈语义。

### 3.3 区分命中 (Hit)、引用 (Citation) 和反馈 (Feedback)

系统应当区分：

- `HIT`（命中）：一条记忆被选中用于上下文注入。
- `CITATION`（引用）：Agent 通过 READ/RUN 或等效的显式引用主动使用了一条记忆。
- `FEEDBACK_POSITIVE` / `FEEDBACK_NEGATIVE`（正向反馈/负向反馈）：用户对基于记忆的响应或特定记忆进行了评判。

这样可以防止低质量的候选记忆仅仅因为出现在宽泛的召回集中就获得加分。

### 3.4 倾向于幂等且尽力而为的强化

强化过程不应阻塞聊天的热路径 (hot path)。强化失败应当被记录并可被观察，但除非调用方明确要求严格行为，否则不应导致用户响应失败。

## 4. 定期维护 (Scheduled Gardening)

### 4.1 新的 Librarian API

在 `LibrarianCore` 中添加一个方法：

```python
async def run_gardening_once(self) -> GardeningResult:
    ...
```

第一个版本应当：

1. 验证 `lifecycle_engine` 是否可用。
2. 运行 `lifecycle_engine.run_garbage_collection(force=False)`。
3. 返回包含归档数量、耗时和错误摘要的结构化结果。

建议的轻量级结果模型：

```python
class GardeningResult(BaseModel):
    success: bool
    archived_count: int = 0
    duration_ms: float = 0.0
    error: str | None = None
```

如果第一个补丁不引入模型，可以先作为一个字典 (dict) 开始。

### 4.2 调度器注册

扩展 `PatchouliSystem.register_maintenance_tasks()` 以注册第二个任务：

```python
MaintenanceTaskSpec(
    owner="patchouli",
    name="memory_gardening",
    interval_seconds=tasks_config.lifecycle_gc_interval_hours * 3600,
    enabled=tasks_config.enable_lifecycle_gc,
)
```

回调：

```python
self.runtime.librarian_core.run_gardening_once
```

现有的 `scheduler.tasks.enable_lifecycle_gc` 应作为主要启用开关。较旧的 `lifecycle.garbage_collector.enable_schedule` 不应启动任何本地调度器。

### 4.3 关机行为

在首次实现中，生命周期 GC (垃圾回收) 不应在关机时自动运行。关机应继续优先处理 observer/perception 队列的排空 (drain)。

可以稍后通过管理/调试端点添加手动维护功能。

## 5. 检索命中强化 (Retrieval Hit Reinforcement)

### 5.1 强化点

在最终记忆被选中用于上下文注入后，记录 `HIT`。

当前集成点：

- `PatchouliService.finalize_agent_run()`，在 `submit_interaction()` 完成后，对最终注入上下文的 `retrieval_result.memories` 记录 HIT。
- 单次 finalize 内按 memory id 去重。

历史候选点：

- `PatchouliService.retrieve_for_gaze()` / `prepare_agent_run()`。
- `RetrievalFamiliar.retrieve_async()`。

避免在密集/稀疏/混合检索器内部记录命中。这些层生成的是候选结果和中间分数，而不是确认的上下文使用情况。

### 5.2 错误处理

命中强化应当是尽力而为的：

```python
for memory in retrieval_result.memories:
    try:
        self._runtime.librarian_core.lifecycle_engine.record_hit(memory.id, source="retrieval.finalize")
    except Exception:
        logger.warning("Failed to record memory hit", exc_info=True)
```

如果热路径延迟变得明显，稍后可以将其移至异步队列。

### 5.3 去重

对于单个检索响应，每个记忆 ID 最多记录一次命中。

如果同一记忆在一个 Agent 运行中被检索多次，第一个实现可能会对每个检索响应进行计数。一旦运行级别的事件 ID 被持久化，就可以添加更严格的基于单次运行的去重逻辑。

## 6. 引用强化 (Citation Reinforcement)

### 6.1 什么是引用

`CITATION`（引用）应代表比检索命中更强的证据。初始引用来源：

- MTP `READ` 成功解析并返回记忆负载 (payload)。
- MTP `RUN` 成功运行一个记忆库中的记忆。
- 未来的显式答案引用标记（如果助手响应格式支持它们）。

### 6.2 集成点

首次实现点：

- 在别名 (aliases) 解析为记忆原子之后，调用 `KoakumaRuntime._execute_read`。
- 在代码记忆被解析并执行成功后，调用 `KoakumaRuntime._execute_run`。

如果 Alice 直接调用 Patchouli 生命周期会违反子系统边界，则通过现有总线路由进行传递，或添加一个 Patchouli 本地/公共生命周期路由。

### 6.3 来源元数据

使用特定的来源标签：

- `source="mtp.read"`
- `source="mtp.run"`
- `source=f"agent:{agent_id}"`（如果身份可用）

生命周期事件模型已接受元数据，因此可以添加更丰富的细节。

## 7. 用户反馈强化 (User Feedback Reinforcement)

### 7.1 API 接口

添加后端端点：

```http
POST /api/v1/memories/{memory_id}/feedback
```

请求体：

```json
{
  "positive": true,
  "source": "ui.memory_ref"
}
```

响应应包含 `ReinforcementResult` 或紧凑的数据传输对象 (DTO)：

```json
{
  "memory_id": "...",
  "event_type": "feedback_positive",
  "previous_vitality": 52.0,
  "new_vitality": 67.0,
  "previous_confidence": 0.6,
  "new_confidence": 0.6
}
```

### 7.2 响应级别反馈（后续）

聊天 UI 通常接收对助手回答的反馈，而不是对单个记忆的反馈。针对这种情况，未来可以引入第二个端点：

```http
POST /api/v1/chat/messages/{message_id}/memory-feedback
```

它应将反馈应用于附加到该回答的所有记忆引用，或者接受明确的子集。

只要聊天记忆引用对前端仍然可用，第一个实现可以仅支持直接的记忆反馈。

### 7.3 负向反馈语义

负向反馈应该：

- 应用生命周期的负面惩罚。
- 使用 `negative_confidence_multiplier` 降低 `confidence_score`。
- 不对记忆本体进行任何操作。

## 8. 配置

主要调度器配置：

```yaml
scheduler:
  tasks:
    enable_lifecycle_gc: false
    lifecycle_gc_interval_hours: 24
```

生命周期策略配置：

```yaml
lifecycle:
  garbage_collector:
    low_watermark: 20.0
    batch_size: 10
```

推荐的首次发布策略：

- 在开发配置中，默认保持 `enable_lifecycle_gc: false`，直到测试和可观测性工作完成。
- 提供明确的启用文档。
- 不要使用 `lifecycle.garbage_collector.enable_schedule` 作为运行时调度器开关，这一实现需要逐步移除。

## 9. 可观测性 (Observability)

暴露或记录以下内容：

- `patchouli.memory_gardening` 的调度器状态。
- 维护结果摘要：归档数量、耗时、成功/失败。
- 按事件类型分类的强化失败记录。
- 来自 `MemoryLifecycleEngine.get_stats()` 的生命周期统计信息。

未来的调试 API：

```http
GET /api/v1/system/maintenance
GET /api/v1/lifecycle/stats
GET /api/v1/memories/{memory_id}/events
```

## 10. 测试计划

### 10.1 单元测试

为以下内容添加或扩展测试：

- `LibrarianCore.run_gardening_once()` 调用生命周期 GC 并返回数量。
- `PatchouliSystem.register_maintenance_tasks()` 注册了 perception_idle_flush 和 memory_gardening。
- 根据所选的调度器语义，禁用的生命周期 GC 注册为禁用状态或不运行。
- 检索命中强化在 finalize 阶段对最终注入的记忆调用 `record_hit()`，并在单次 finalize 内去重。
- MTP READ/RUN 引用在成功使用时调用 `record_citation()`。
- 反馈端点调用 `record_feedback()` 并返回分数增量。

### 10.2 集成测试

覆盖一个活动的聊天流程：

1. 预埋入记忆。
2. 进行检索。
3. 验证返回的记忆得到 HIT。
4. 模拟 READ/RUN 或直接引用。
5. 验证 CITATION 调整了记忆生命力 (vitality)。
6. 提交负向反馈。
7. 验证置信度下降。

### 10.3 回归测试

确保：

- 如果强化失败，聊天不会受到影响。
- 调度器异常被隔离。
- GC 仅归档低于配置低水位的记忆。
- 现有的 perception_idle_flush 调度保持完好。

## 11. 实施顺序

1. 将 `run_gardening_once()` 添加到 `LibrarianCore`。
2. 在 `PatchouliSystem` 中注册 `patchouli.memory_gardening`。
3. 添加调度器注册和维护的单元测试。
4. 在最终的检索响应边界添加检索 HIT 强化。
5. 为 MTP READ/RUN 添加引用强化。
6. 添加记忆反馈 API。
7. 添加前端单记忆反馈连线；消息级别反馈留作后续 UX 扩展。
8. 核心循环稳定后，添加可观测性端点。

## 12. 未决问题 (Open Questions)

- 在第一个补丁中，HIT 强化应该同步运行，还是应该调度为轻量级的后台任务？
- 响应级别反馈应该应用于每个记忆引用，还是仅应用于用户选择的引用？
- 非常负面的反馈是否应标记为 `verification_status=DEPRECATED`，还是应该需要明确的用户理由（例如 `outdated`）？
- 归档的记忆是否应该对直接的别名 (alias) 查找隐藏，还是别名查找应该支持“复活” (resurrection)？

第一次实施可以在没有解决所有这些问题的情况下进行。关键是建立生命周期事件路径，并保持每个事件来源的明确性。
