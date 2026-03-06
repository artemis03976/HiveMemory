# HiveMemory 技术架构更新文档
## 主题：系统并发控制与事件总线规整 (Concurrency & System Bus Protocol)

**文档状态**: Active (执行中)
**适用阶段**: Phase 4.5 (Kernel & Perception Refactoring)
**核心重构模块**: `infrastructure.system_bus`, `engines.kernel`, `engines.perception`

---

### 1. 背景与动机 (Motivation)

随着 HiveMemory 向单进程 AIOS (AI Operating System) 演进，系统内部出现了大量的 I/O 密集型操作（LLM 请求、向量库读写）和复杂的内存状态变更。在原有的代码中，同步与异步逻辑混杂，导致了严重的控制流认知负担。

本次架构更新旨在解决以下问题：
1. **彻底摒弃多线程 (`threading`)**：统一采用 Python 的 `asyncio` 单线程并发模型，降低上下文切换开销与锁死风险。
2. **规范化事件总线 (SystemBus)**：明确 RPC (同步等待) 与 Pub/Sub (后台脱手) 的使用边界。
3. **消除竞态条件 (Race Conditions)**：在极短的交互间隔内，确保感知层 (Perception) 状态的一致性。
4. **引入 CQS 原则 (命令查询分离)**：规范状态写入与状态读取的接口设计。

---

### 2. 核心并发范式 (Core Concurrency Paradigm)

在 HiveMemory 的开发中，所有涉及到网络、磁盘读写的方法必须声明为 `async def`。在调用时，严格遵循以下两条调度铁律：

#### 2.1 热链路阻塞 (The Hot Path - `await`)
*   **定义**：当前业务流**必须依赖该操作的结果**，或者该操作**会改变后续读取的系统状态**。
*   **行为**：使用 `await` 挂起当前协程，将 CPU 控制权交还给事件循环，直到操作完成。
*   **典型场景**：
    *   Kernel 请求 LLM 生成回复。
    *   Kernel 向感知层提交 (Ingest) 对话数据（必须等待内存打扫完毕）。

#### 2.2 冷链路脱手 (The Cold Path - Fire-and-Forget)
*   **定义**：主业务流不需要该操作的返回值，且该操作耗时较长（如深度推理、写入数据库）。
*   **行为**：使用 `asyncio.create_task()` 将其扔给后台静默执行，当前代码瞬间向下流转。
*   **典型场景**：
    *   话题被驱逐时，唤醒 Librarian 生成记忆原子。

---

### 3. SystemBus 接口规范 (System Bus Specification)

为了降低认知负担，所有的 `await` 和 `Fire-and-Forget` 都被封装在 `SystemBus` 中。组件之间严禁直接通过实例化相互调用，必须通过总线通信。

#### 3.1 `bus.request(route, **kwargs)` -> RPC 模式
*   **底层实现**：等待目标协程执行完毕并返回结果。
*   **使用场景**：
    *   `await bus.request("retrieval.fetch", ...)` (需要拿到记忆用于注入)
    *   `await bus.request("perception.ingest", ...)` (需要确认状态已写入)

#### 3.2 `bus.emit(event, **kwargs)` -> Event 模式
*   **底层实现**：内部调用 `asyncio.create_task()`，瞬间返回。
*   **使用场景**：
    *   `bus.emit("librarian.generate_memory", ...)` (投递给后台，不阻塞用户交互)
    *   `bus.emit("system.log_error", ...)`

---

### 4. 关键防错设计 (Critical Safety Designs)

#### 4.1 彻底防止竞态条件 (Anti-Race Condition)
在 `Kernel` 与 `Perception` 的交互中，**绝不允许使用 `emit` 来提交数据**。

*   **错误示范**：Kernel 使用 `bus.emit("perception.ingest")` 提交轮次 N。随后用户极快地输入了轮次 N+1 的消息，Kernel 立即 `request` 获取上下文。由于 `ingest` 还在后台执行（可能正在压缩 Summary），Kernel 将拿到撕裂的脏数据。
*   **架构规范**：数据摄入（Ingest）作为状态突变操作，**必须**使用 `await bus.request` 进行硬阻塞。此时前端处于 Loading 状态，直到感知层完全清理好内存，才允许进行下一次交互。

#### 4.2 引入 CQS 原则 (Command Query Separation)
关于“感知层在 Token 溢出时是否需要直接返回 Summary 给 Kernel”，系统严格遵循 CQS 原则：

*   **Command (命令/写)**：`await bus.request("perception.ingest", payload)`。该接口仅负责改变内部状态（路由、折叠、清空），返回成功信号 `{"status": "ok"}`。不返回具体的 Context 数据。
*   **Query (查询/读)**：`await bus.request("perception.route_and_get_context", query)`。该接口仅负责读取当前干净的内存状态（包含最新生成的 Summary 和 Blocks），绝对不改变状态。

#### 4.3 MTP 递归循环的摄入时机 (Ingestion Timing Exception)
在 MTP 协议的多次打断与递归中，极易产生碎片化 Ingest 的 Bug。

*   **架构规范**：当 LLM 因触发协议被 `stop=["⟫"]` 截断时（如正在执行 `⟪ READ ⟫`），**严禁**调用 `perception.ingest`。
*   **正确时机**：Kernel 必须将中间的 MTP 轨迹暂存在局部的临时 History 中。**只有当 LLM 的 Finish Reason != "stop"（即本轮发言彻底自然结束）时**，才将包含所有痕迹的超大 Payload 一次性提交给感知层。

---

### 5. 标准代码骨架示例参考 (Code Scaffolding)

**Kernel 端的主循环片段：**

```python
async def run_interaction_loop(self, user_query):
    # 1. Query: 获取当前绝对干净的上下文
    topic_id, context = await self.bus.request("perception.route_and_get_context", query=user_query)
    
    # 2. Hot Path: 驱动生成与 MTP 递归 (在 Kernel 内存中维护)
    final_payload = await self._recursive_mtp_loop(context)
    
    # 3. Command: 同步阻塞，硬等待感知层打扫战场 (解决防竞态)
    # 感知层可能在里面耗时 2 秒进行 Compact
    await self.bus.request("perception.ingest", payload=final_payload)
    
    # 4. 解锁前端，等待用户的下一次输入
    return final_payload.clean_response
```

**Perception 端的 Trigger Manager 片段：**

```python
async def resolve_topic(self, topic_id, trigger):
    topic = self.topics[topic_id]
    
    # Cold Path (Fire-and-forget): 丢给帕秋莉，不等待
    if trigger in ["IDLE", "EVICT", "MTP_WRITE"]:
        self.bus.emit("librarian.generate_memory", data=topic.snapshot())
        
    # Hot Path (Await): 影响下一次聊天的内存压缩，必须等
    if trigger in ["OVERFLOW", "MTP_WRITE"]:
        topic.state_summary = await self._compact_blocks(topic)
        topic.blocks.clear()
```

---

### 总结

通过这份更新文档，HiveMemory 确立了以 `asyncio` 和 `SystemBus` 为核心的并发基调。开发者在编写代码时只需通过区分 **RPC (Request/Await)** 与 **Event (Emit/Background)**，即可清晰地控制系统的状态一致性与响应延迟。这将极大提升系统在长文本、高并发场景下的健壮性。