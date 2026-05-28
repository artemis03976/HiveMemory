# Router 到 Application Service 收口重构计划

**文档状态**: Draft\
**所属演进**: 第四次架构演进后续收口 / v4.1\
**文档定位**: 规划后端 HTTP router 从直接访问系统内部对象迁移到应用服务门面的重构路径。

***

## 1. 背景

第四次架构演进已经将 HiveMemory 的核心结构收敛为：

```text
HiveMemorySystem
  -> system/application/*
  -> PatchouliSystem
  -> AliceSystem
```

其中 Patchouli 与 Alice 分别作为记忆域与 Agent / MTP 运行域的同级子系统存在，系统级跨域编排由 `system/application/` 与全局公开路由承担。

但当前 HTTP router 层仍存在若干旧式访问路径，例如：

- `system.patchouli.storage`
- `system.patchouli.runtime`
- `system.patchouli.librarian_core`
- `system.patchouli.librarian_core.perception_layer.buffer_manager`

这些访问方式使 HTTP adapter 越过应用服务层，直接触碰子系统内部结构，削弱了 v4 架构中的封装边界。

***

## 2. 目标

本轮重构目标不是改动业务行为，而是收口后端 API 边界：

1. HTTP router 只负责协议适配：
   - 解析 FastAPI 请求
   - 调用应用级 use case
   - 转换 response model
   - 抛出 HTTP 层错误
2. router 不再直接访问 Patchouli / Alice 的内部对象。
3. `HiveMemorySystem` 作为组合根与服务注册入口，不膨胀为包含所有业务动作的 God Facade。
4. 新增或整理领域化 application service：
   - `MemoryApplicationService`
   - `AgentApplicationService`
   - `TopicApplicationService`
   - 继续保留现有 `ChatApplicationService`
   - 继续保留现有 `PassiveIngressService`
5. server 依赖注入从 `get_system()` 逐步过渡为 `get_memory_service()` / `get_topic_service()` 等专用入口。

***

## 3. 目标结构

推荐长期结构如下：

```text
server/
  deps.py
  routers/
    chat.py       -> ChatApplicationService
    memories.py   -> MemoryApplicationService
    agents.py     -> AgentApplicationService
    topics.py     -> TopicApplicationService
    ingest.py     -> PassiveIngressService

system/
  system.py
  application/
    chat_service.py
    memory_service.py
    agent_service.py
    topic_service.py
    passive_ingress_service.py
```

`HiveMemorySystem` 对外暴露服务入口，而不是暴露所有业务动作：

```python
class HiveMemorySystem:
    @property
    def chat_service(self) -> ChatApplicationService: ...

    @property
    def memory_service(self) -> MemoryApplicationService: ...

    @property
    def agent_service(self) -> AgentApplicationService: ...

    @property
    def topic_service(self) -> TopicApplicationService: ...

    @property
    def passive_ingress_service(self) -> PassiveIngressService: ...
```

FastAPI 依赖注入层提供更窄的服务入口：

```python
def get_memory_service() -> MemoryApplicationService:
    return get_system().memory_service
```

router 只依赖对应应用服务：

```python
@router.get("/memories")
async def list_memories(
    service: MemoryApplicationService = Depends(get_memory_service),
):
    return await service.list_memories(...)
```

***

## 4. 边界原则

### 4.1 Router 不应访问的对象

router 中不应出现以下访问：

- `system.patchouli`
- `system.alice`
- `system.patchouli.storage`
- `system.patchouli.runtime`
- `system.patchouli.librarian_core`
- `system.patchouli.librarian_core.perception_layer`
- 任意子系统 `_private` 属性

### 4.2 Application Service 可以做什么

应用服务是 HTTP / CLI / UI use case 的稳定入口，可以：

- 组合多个子系统公开能力
- 调用全局公开路由 `GlobalRoutes`
- 做 use case 层的数据转换
- 执行权限、过滤、默认参数与错误转换
- 调用子系统 service 的公开方法

应用服务不应：

- 直接操作子系统 runtime 的私有字段
- 复刻 Patchouli / Alice 内部领域逻辑
- 暴露 storage、runtime、bus 等基础设施对象给 router

### 4.3 HiveMemorySystem 的角色

`HiveMemorySystem` 是 composition root，负责：

- 构建对象图
- 管理生命周期
- 持有全局总线与调度器
- 暴露应用服务入口
- 提供少量系统级健康检查 / readiness / warmup 门面

它不应长期承载所有业务动作方法。

***

## 5. 当前穿透点盘点

### 5.1 memories router

文件：`src/hivememory/server/routers/memories.py`

当前问题：

- 直接通过 `system.patchouli.storage` 做 CRUD。
- 通过 `runtime._engines` 或 `librarian_core.lifecycle_engine` 获取 lifecycle engine。
- router 内部承担 vitality refresh 与 feedback 业务。

目标：

- 新增 `MemoryApplicationService`。
- 将 memory CRUD、search、feedback、vitality refresh 收入 service。
- router 只负责 request / response model 转换。

### 5.2 agents router

文件：`src/hivememory/server/routers/agents.py`

当前问题：

- 直接构造 `MemoryAtom` 并写入 `system.patchouli.storage`。
- Agent Profile 的存储细节暴露在 HTTP 层。

目标：

- 新增 `AgentApplicationService`。
- 将 Agent Profile 的创建与列表查询迁入 service。
- 后续可进一步迁入 Patchouli 的 profile 管理公开能力。

### 5.3 topics router

文件：`src/hivememory/server/routers/topics.py`

当前问题：

- 直接访问 `librarian_core.get_active_topics_snapshots()`。
- 直接访问 `perception_layer.buffer_manager.pop_buffer(topic_id)` 驱逐话题。

目标：

- 新增 `TopicApplicationService`。
- 将 `list_active_topics()`、`archive_topic()`、`evict_topic()` 收入 service。
- Patchouli 侧应提供公开 route 或 service 方法执行话题管理操作。

### 5.4 app readiness / warmup

文件：`src/hivememory/server/app.py`

当前问题：

- lifespan 中直接调用 `system.patchouli.runtime.warmup_models()`。
- readiness 中直接调用 `system.patchouli.runtime.is_models_ready()`。

目标：

- 在 `HiveMemorySystem` 提供系统级门面：
  - `warmup_models()`
  - `is_models_ready()`
  - 或 `readiness()`
- server 层不再感知 Patchouli runtime。

***

## 6. 迁移步骤

### Phase 1: 增加系统级服务入口

1. 在 `system/application/` 下新增：
   - `memory_service.py`
   - `agent_service.py`
   - `topic_service.py`
2. 在 `HiveMemorySystem.build()` 中装配这些 service。
3. 在 `HiveMemorySystem` 上暴露服务属性：
   - `memory_service`
   - `agent_service`
   - `topic_service`
4. 在 `server/deps.py` 中新增：
   - `get_memory_service()`
   - `get_agent_service()`
   - `get_topic_service()`

验收：

- 现有 router 仍可运行。
- 新 service 可被单元测试直接调用。

### Phase 2: 迁移 memories router

1. 将 memory CRUD 与搜索迁入 `MemoryApplicationService`。
2. 将 lifecycle feedback 与 vitality refresh 迁入 `MemoryApplicationService`。
3. router 改为依赖 `get_memory_service()`。
4. 保持 HTTP response model 不变。

验收：

- `server/routers/memories.py` 不再出现 `system.patchouli`。
- memory API 行为与返回结构保持不变。

### Phase 3: 迁移 agents router

1. 将 Agent Profile 创建和列表查询迁入 `AgentApplicationService`。
2. router 改为依赖 `get_agent_service()`。
3. 后续根据需要将 Agent Profile CRUD 下沉为 Patchouli 公开契约。

验收：

- `server/routers/agents.py` 不再出现 `system.patchouli`。
- Agent Profile API 行为保持不变。

### Phase 4: 迁移 topics router

1. 将活跃话题查询迁入 `TopicApplicationService`。
2. 将手动归档迁入 `TopicApplicationService`。
3. 为“从活跃池驱逐话题”建立明确 use case：`evict_topic(topic_id)`。
4. Patchouli 侧补齐公开 route 或公开 service 方法，避免 service 触碰 `perception_layer.buffer_manager`。

验收：

- `server/routers/topics.py` 不再出现 `system.patchouli`。
- topic archive / delete 行为保持不变。

### Phase 5: readiness / warmup 收口

1. 在 `HiveMemorySystem` 增加 `warmup_models()` 与 `is_models_ready()`。
2. `server/app.py` 改为调用系统级门面。
3. 后续可将 readiness 返回结构集中为 `SystemReadiness` 模型。

验收：

- `server/app.py` 不再访问 `system.patchouli.runtime`。

### Phase 6: 兼容访问器清理

1. 统计 `system.patchouli` / `system.alice` 在 server 层的剩余使用点。
2. server 层完全清理后，将 `HiveMemorySystem.patchouli` / `alice` 标记为内部兼容访问器。
3. 测试与脚本若仍需使用内部对象，应迁移到专用测试 fixture 或明确的 debug API。

验收：

- `src/hivememory/server` 中不再出现 `system.patchouli` / `system.alice`。
- 生产 HTTP 层只依赖应用服务。

***

## 7. 测试计划

### 单元测试

- `MemoryApplicationService`
  - create / list / get / update / delete
  - search filters
  - feedback
  - lifecycle unavailable fallback
- `AgentApplicationService`
  - create profile
  - list profile
- `TopicApplicationService`
  - list active topics
  - archive topic
  - evict topic

### Router 测试

- 保持现有 HTTP response schema 不变。
- 使用 mock application service 替代完整 `HiveMemorySystem`。
- 验证 router 不依赖子系统内部对象。

### 集成测试

- 启动 FastAPI lifespan。
- 验证：
  - `/health/ready`
  - `/memories`
  - `/agents`
  - `/topics`
  - `/chat/stream`

***

## 8. 非目标

本轮不处理：

- Patchouli / Alice 内部 runtime 再拆分。
- memory storage 的持久化实现替换。
- 前端 API schema 大规模改名。
- 全量权限系统设计。
- CLI / desktop app 的入口整理。

***

## 9. 风险与注意事项

1. 不要把所有 API 方法直接堆到 `HiveMemorySystem` 上。
   - 短期可以用少量系统级门面过渡。
   - 长期应通过 application service 属性暴露能力组。
2. 不要让 application service 重新变成新的“万能 runtime”。
   - service 应按 use case 组织，避免保存复杂运行状态。
3. 不要在 router 中保留 fallback 私有访问路径。
   - 一旦出现 `getattr(system.patchouli, "_xxx")`，说明边界又被打开。
4. 保持 HTTP response model 稳定。
   - 本轮是架构收口，不应引入前端破坏性变更。

***

## 10. 完成判定

完成后应满足：

- `src/hivememory/server/routers/*` 不直接访问 Patchouli / Alice 内部对象。
- `server/deps.py` 提供面向 router 的窄依赖入口。
- `HiveMemorySystem` 是组合根与服务注册入口，而非 God Facade。
- memory / agent / topic API 均通过 `system/application/` 层进入系统。
- 现有 HTTP API 行为保持兼容。
