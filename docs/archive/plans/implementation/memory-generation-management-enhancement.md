---
title: Legacy Memory Generation Management Enhancement Plan
status: superseded
owner: patchouli
scope: completed-generation-task-management-enhancement
archived_at: 2026-07-29
superseded_by: docs/patchouli/generation.md
---

> 本计划中的 spec 隔离、后台任务等待与 shutdown drain 已进入当前实现，本文停止维护。当前任务控制面、失败隔离、等待/取消和关闭时序以[记忆生成](../../../patchouli/generation.md)为准。

# 记忆生成管理增强计划

> 版本：v0.5.x 补丁  
> 优先级：中  
> 状态：设计评审中

---

## 一、背景与问题

本文档涵盖记忆生成管理链路中三个独立但同属一个 PR 范畴的问题。

---

### 问题一：`submit_generation_many` 串行构建 spec

**位置：** `patchouli/control/memory_generation_coordinator.py:92-100`

```python
specs = [
    await self._build_active_spec(task, ...)  # ← 串行 await
    for task in tasks
]
```

Agent 在同一轮对话中发出多条 WRITE/UPDATE 时，`submit_active` 对每个 task 串行构建 spec。
其中 UPDATE 类型在 `_build_active_spec` 内需要发出 `MEMORY_GET` 总线请求（I/O 等待），
多条 UPDATE 的等待时间完全累加，不必要地延长了主动生成链路的响应延迟。

---

### 问题二：UPDATE 目标缺失时整批任务失败

**位置：** `patchouli/control/memory_generation_coordinator.py:132-134`

```python
if existing is None:
    logger.error(...)
    raise RuntimeError(f"UPDATE target memory not found: {focus.base_uuid}")
```

`_build_active_spec` 在 UPDATE 目标不存在时抛出 `RuntimeError`。
该异常在 `submit_active` 的列表推导式中未被捕获，导致：

1. **整批任务失败**：同一次调用中的 WRITE 任务也不会被提交
2. **PendingAtom 事件泄漏**：失败的 UPDATE 任务的 `pending_alias` 不会发布 `PENDING_ATOM_FAILED` 事件，
   MTP 运行时无法得知该 PendingAtom 的最终状态，可能造成上游等待悬空

UPDATE 目标不存在是业务异常（Agent 发送了无效的 base_uuid），应当隔离为单任务失败，不影响同批次其他任务。

---

### 问题三：`flush_all_for_shutdown` 不等待生成任务完成

**位置：** `patchouli/services/perception.py:168-173`（已有 TODO 注释）

```python
# TODO: 目前仅保证任务提交完成，不等待生成执行完毕。
#       未来需在 MemoryGenerationTaskController 提供 drain_all() 接口，
#       在此调用以确保 shutdown 前所有记忆生成任务执行完毕。
```

`flush_all_for_shutdown` 在提交所有 settlement 任务后立即返回。
服务关闭时，后台生成协程可能尚未完成，进程退出时这些任务被强制终止，
导致部分记忆未能持久化到 mid-term 存储。

---

## 二、设计方案

### 2.1 并行构建 spec + 单任务失败隔离（问题一 + 问题二）

两个问题合并解决：将串行推导式改为 `asyncio.gather`，同时在 `_build_active_spec` 层捕获
UPDATE 目标缺失异常，返回 `None` 并发布 `PENDING_ATOM_FAILED` 事件，由 `submit_active`
过滤空结果后提交剩余有效 spec。

**`_build_active_spec` 签名不变，新增包装方法 `_try_build_active_spec`：**

```python
async def _try_build_active_spec(
    self,
    task: PendingAtomMaterializeTask,
    *,
    topic_id: str,
    gen_context,
    interaction_input: InteractionArtifactInput | None,
) -> MemoryGenerationTaskSpec | None:
    try:
        return await self._build_active_spec(
            task, topic_id=topic_id,
            gen_context=gen_context,
            interaction_input=interaction_input,
        )
    except RuntimeError as exc:
        logger.error(
            "Active spec build failed, skipping task: "
            "pending_alias=%s, err=%s",
            task.pending_alias, exc,
        )
        if task.pending_alias:
            await self._publish_pending_atom_failed(task.pending_alias)
        return None
```

**`submit_active` 改为并行构建并过滤失败项：**

```python
async def submit_active(
    self,
    tasks: List[PendingAtomMaterializeTask],
    topic_id: str,
) -> List[MemoryGenerationTask]:
    if not tasks:
        return []

    topic_data = await self._bus.request(PatchouliLocalRoutes.TOPIC_GET, topic_id)
    blocks = topic_data.recent_blocks(5) if topic_data is not None else []
    state_summary = topic_data.state_summary if topic_data is not None else ""
    gen_context = self._transcript_builder.build_context(blocks, state_summary=state_summary)
    interaction_input = self._build_interaction_input(
        topic_id=topic_id,
        topic_title=topic_data.topic_title if topic_data is not None else "",
        topic_summary=topic_data.topic_summary if topic_data is not None else "",
        blocks=blocks,
    )

    raw = await asyncio.gather(*[
        self._try_build_active_spec(
            task,
            topic_id=topic_id,
            gen_context=gen_context,
            interaction_input=interaction_input,
        )
        for task in tasks
    ])
    specs = [s for s in raw if s is not None]
    if not specs:
        return []
    return await self._bus.request(
        PatchouliLocalRoutes.MEMORY_TASK_SUBMIT_GENERATION_MANY,
        specs,
    )
```

`_build_active_spec` 保持原有逻辑和 `RuntimeError` 抛出行为不变，
错误处理集中在 `_try_build_active_spec` 一层，职责清晰。

需要在 `coordinator.py` 中新增 `_publish_pending_atom_failed` 辅助方法（向 local bus 发布
`PatchouliLocalEvents.PENDING_ATOM_FAILED`），或直接复用已有的 bus.publish 调用模式。

---

### 2.2 `drain_all` + shutdown 等待（问题三）

**`MemoryGenerationTaskController` 新增 `drain_all()` 方法：**

`MemoryGenerationTask._bg_task` 持有后台 `asyncio.Task` 引用，
`MemoryGenerationTaskRegistry.list_all()` 可枚举所有任务。

```python
async def drain_all(self, timeout: float | None = None) -> dict[str, int]:
    """等待所有运行中的生成任务完成，用于 shutdown 前的优雅排水。

    Args:
        timeout: 最长等待秒数，None 表示无限等待。

    Returns:
        {"drained": N, "timed_out": M}
    """
    bg_tasks = [
        mt._bg_task
        for mt in self._task_registry.list_all()
        if mt._bg_task is not None and not mt._bg_task.done()
    ]
    if not bg_tasks:
        return {"drained": 0, "timed_out": 0}

    done, pending = await asyncio.wait(bg_tasks, timeout=timeout)
    if pending:
        logger.warning(
            "drain_all timed out: %d tasks still running after %.1fs",
            len(pending), timeout,
        )
    return {"drained": len(done), "timed_out": len(pending)}
```

**`PatchouliRuntime.shutdown_drain()` 在 perception flush 后调用 drain：**

```python
async def shutdown_drain(self) -> dict[str, Any]:
    if self._shutdown_drain_started:
        ...  # 幂等保护不变

    self._shutdown_drain_started = True
    perception_result = await self.perception_familiar.flush_all_for_shutdown()

    drain_result = await self._task_controller.drain_all(
        timeout=self._patchouli_config.shutdown.drain_timeout_seconds,  # 建议默认 30s
    )

    return {
        "success": True,
        "perception": perception_result,
        "drain": drain_result,
        "reentrant": False,
    }
```

`drain_timeout_seconds` 作为新配置项加入 `PatchouliConfig.shutdown`（如该 section 不存在则新建）。

**`PerceptionFamiliar.flush_all_for_shutdown()` 删除 TODO 注释。**

---

## 三、文件变更清单

| 文件 | 变更内容 |
|------|---------|
| `patchouli/control/memory_generation_coordinator.py` | 新增 `_try_build_active_spec()`；`submit_active()` 改为 `asyncio.gather` + None 过滤；新增 `_publish_pending_atom_failed()` |
| `patchouli/control/memory_generation_tasks.py` | 新增 `drain_all()` 方法 |
| `patchouli/runtime/core.py` | `shutdown_drain()` 增加 `drain_all()` 调用，返回值含 drain 结果 |
| `patchouli/services/perception.py` | 删除 `flush_all_for_shutdown` 的 TODO 注释 |
| `system/config/patchouli.py` | `PatchouliConfig` 新增 `shutdown.drain_timeout_seconds`（默认 30） |

---

## 四、不在本 PR 范围内

- `submit_generation_many` 本身的串行任务提交（`for spec in specs: await submit_generation(spec)`）：
  当前是串行提交但各任务异步执行，延迟可接受；如需并行提交可后续单独优化
- 任务持久化（进程崩溃后恢复未完成的生成任务）
- `drain_all` 超时后强制取消剩余任务的策略
