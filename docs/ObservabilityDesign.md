---
title: Legacy Observability and Log Stream Design
status: superseded
owner: system
scope: legacy-observability-design
archived_at: 2026-07-28
superseded_by:
  - docs/system/observability.md
  - docs/contracts/routes-and-events.md
---

> 本文已停止维护，只保留可观测性方案的历史背景。当前 RuntimeEvent、operation observer、健康状态与旁路原则以 [`docs/system/observability.md`](./system/observability.md) 为准，公开事件边界以 [`docs/contracts/routes-and-events.md`](./contracts/routes-and-events.md) 为准。

# HiveMemory 可观测性与日志流设计文档

## 1. 背景与痛点

随着 HiveMemory 架构的演进，系统内部存在着多种类型的执行流，主要包括：
1. **前台交互流（Foreground）**：用户主动发起的交互，如对话（Chat）、检索（Search）。
2. **后台静默流（Background/Silent）**：系统自动触发的维护任务，如 `TriggerManager` 触发的异步话题归档（Archive）与记忆生成。

目前的日志系统存在以下痛点：
* **日志交错冲突**：传统的按时间线输出的日志，在面对并发执行的前台任务和后台静默任务时，日志输出会相互交错，导致前端终端窗口的日志难以阅读。
* **职责模糊**：如果将所有组件级别的日志（如感知层的结算日志）强行推入运行时总线，会模糊其“业务指令路由”的核心职责，且增加总线负担。
* **静默错误吞没**：像 `FlushObserver` 这种旧的观测机制与现有代码脱节，导致后台静默执行的记忆生成操作（Archive）一旦发生异常，容易被全局 `try...except` 吞没，前端和开发者无法感知。

## 2. 设计目标

1. **分离业务指令与观测数据**：坚持运行时总线负责业务消息路由，而组件级日志和运行状态通过专门的**可观测性流（Observability Stream）**旁路输出。
2. **总分结构的日志呈现**：放弃纯时间线的日志展示，采用基于任务上下文的**“总-分（折叠树）”**结构呈现。
3. **消除旧观测机制的冗余**：通过在任务入口统一注入上下文和生命周期日志，废弃老旧且脱节的 `FlushObserver` 逻辑。

## 3. 核心概念与数据结构

借鉴分布式追踪（Distributed Tracing）的思想，引入以下核心概念：

### 3.1 Trace (完整链路)
代表一个完整的、宏观的业务流。
* **示例**：“处理用户第 5 次提问”、“后台归档话题 T_02”。
* **标识**：拥有全局唯一的 `trace_id`。

### 3.2 Span (子任务)
代表 Trace 链路下的一个具体执行阶段或组件任务，也就是“总分结构”中的“总”。
* **示例**：“执行 Retrieval”、“执行 Koakuma”。
* **标识**：由 `span_name` 标识。

### 3.3 推送日志元数据契约 (SSE 载荷)
所有发送给前端的观测日志应遵循以下 JSON 格式：

```json
{
  "timestamp": "10:48:01",
  "level": "INFO",
  "trace_id": "a1b2c3d4",
  "task_type": "foreground",  // 可选值: "foreground" (前台任务) | "background" (后台任务)
  "span_name": "RetrievalFamiliar.Retrieve",
  "message": "Found 3 memory atoms."
}
```

## 4. 后端实现方案

为了在不侵入现有大量 `logger.info()` 业务代码的前提下实现该设计，采用 Python `contextvars` 结合自定义 `logging.Filter` 的轻量级拦截方案。

### 4.1 上下文变量定义
使用 `contextvars` 维护当前协程（asyncio）或线程的任务上下文：

```python
import contextvars

# 定义当前任务的上下文
current_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("current_trace_id", default="system")
current_span_name: contextvars.ContextVar[str] = contextvars.ContextVar("current_span_name", default="main")
current_task_type: contextvars.ContextVar[str] = contextvars.ContextVar("current_task_type", default="foreground")
```

### 4.2 日志拦截器 (Filter)
创建一个自定义的 `logging.Filter`，自动将上下文信息注入到每条标准日志记录中：

```python
import logging

class TraceInjectFilter(logging.Filter):
    """自动将 Trace 和 Span 信息注入到每一条日志记录中"""
    def filter(self, record):
        record.trace_id = current_trace_id.get()
        record.span_name = current_span_name.get()
        record.task_type = current_task_type.get()
        return True

# 绑定到根 Logger
logging.getLogger().addFilter(TraceInjectFilter())
```

### 4.3 独立的可观测性通道
实现一个自定义的 `logging.Handler`（如 `SSELogHandler`），专门用于拦截格式化后的日志，并通过 SSE (Server-Sent Events) 推送给前端。
* **原则**：绝对不走运行时总线，避免污染业务事件。

### 4.4 典型场景改造示例：静默归档 (Archive)

在 `TriggerManager` 或 `LibrarianCore` 触发异步归档任务前，设置上下文，从而**替代旧版的 `FlushObserver` 机制**。

```python
import uuid
import asyncio

async def _async_archive_wrapper(payload):
    # 1. 初始化独立后台 Trace
    trace_id = f"archive-{uuid.uuid4().hex[:8]}"
    
    # 2. 设置上下文
    trace_token = current_trace_id.set(trace_id)
    type_token = current_task_type.set("background")
    span_token = current_span_name.set("LibrarianCore.Archive")
    
    try:
        # 发送起始日志（显式声明 Span 开始，解决静默问题）
        logger.info(f"开始执行后台静默归档任务, 涉及 {len(payload.get('blocks', []))} 个 blocks")
        
        # 执行实际归档逻辑，内部的 logger.info 会自动带上 trace_id 和 span_name
        await self._on_generate_memory(payload)
        
        logger.info("后台静默归档任务执行成功")
    except Exception as e:
        logger.error(f"后台静默归档任务失败: {e}", exc_info=True)
    finally:
        # 恢复上下文（可选，但在长生命周期应用中推荐）
        current_trace_id.reset(trace_token)
        current_task_type.reset(type_token)
        current_span_name.reset(span_token)
```

## 5. 前端渲染策略

前端接收到结构化的日志流后，需改变传统的平铺追加渲染模式，采用基于树状结构的动态折叠面板。

### 5.1 数据聚合结构
前端在内存中维护一个类似于如下的结构：
`Map<trace_id, Map<span_name, Array<LogMessage>>>`

### 5.2 渲染规则
1. **自动建块**：当收到某 `trace_id` 下某个 `span_name` 的第一条日志时，前端自动在终端 UI 中创建一个可折叠的 UI 块。
   * 格式示例：`▶ [RetrievalFamiliar.Retrieve]`
2. **日志归属**：后续收到相同 `trace_id` 和 `span_name` 的日志，全部追加到该折叠块的内部。这样即使多个 Trace 并发，日志也不会交错。
3. **基于 task_type 的默认展示策略**：
   * **`task_type === "foreground"`**（如用户问答）：相关的 Span 折叠块**默认展开**，方便用户实时查看进度。
   * **`task_type === "background"`**（如静默归档）：相关的 Span 折叠块**默认折叠**，不打扰用户当前注意力，但保留了可追溯的能力。

## 6. 演进路线总结

1. **废除旧逻辑**：移除 `LibrarianCore` 中未使用且已脱节的 `FlushObserver` 相关接口与逻辑。
2. **基建注入**：引入 `contextvars`，实现 `TraceInjectFilter` 并配置全局 Logger。
3. **埋点改造**：在关键任务的入口处（如 System API 摄入点、TriggerManager 异步任务触发点）包裹 Context 设定。
4. **通道与前端呈现**：构建独立的 SSE 日志通道，前端根据 `trace_id` 和 `span_name` 实现分组折叠渲染。

该方案以最低的业务代码侵入性，实现了完美的任务追踪与“总分结构”呈现，彻底解决了异步静默任务的日志交错与异常吞没问题。
