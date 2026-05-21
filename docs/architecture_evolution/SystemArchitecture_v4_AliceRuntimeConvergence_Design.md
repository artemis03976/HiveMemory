# HiveMemory 第四次架构演进 Alice Runtime 收敛设计

> **归档说明**: 本文是 v4 演进过程中的阶段设计记录，保留用于追溯迁移背景与取舍。v4 当前最终结构、术语与实现准则已统一收敛到 [SystemArchitecture_v4_TopLevelSketch.md](./SystemArchitecture_v4_TopLevelSketch.md)。如本文与最终总纲冲突，以最终总纲为准。

**文档状态**: Archived (阶段设计记录)\
**所属演进**: 第四次架构演进\
**建议定位**: Phase C 后续收尾 / Alice Runtime Convergence\
**阶段目标**: 在 `Patchouli` 子系统完成 `Runtime` 语义收敛后，对 `Alice` 子系统进行对应的结构收尾：以 `AliceRuntime` 取代当前 `AgentRuntimeHost`，并在其内部显式区分 `AgentRuntime` 与 `KoakumaRuntime` 两类运行时职责。\
**配套文档**:

- [SystemArchitecture_v4_TopLevelSketch.md](./SystemArchitecture_v4_TopLevelSketch.md)
- [SystemArchitecture_v4_PhaseC_AliceRuntimeFoundation_Design.md](./SystemArchitecture_v4_PhaseC_AliceRuntimeFoundation_Design.md)
- [SystemArchitecture_v4_PhaseD_ChatApplicationServiceMigration_Design.md](./SystemArchitecture_v4_PhaseD_ChatApplicationServiceMigration_Design.md)
- [SystemArchitecture_v4_RuntimeConvergence_Addendum.md](./SystemArchitecture_v4_RuntimeConvergence_Addendum.md)

***

## 1. 文档定位

这份文档解决的不是 “Alice 是否需要 runtime” 这个问题。

这个问题在当前代码状态下其实已经有答案：

- `alice/runtime/host.py` 中的 `AgentRuntimeHost` 已经承担了运行时对象聚合职责
- `KernelLoopExecutor` 已经承担 Agent 执行循环
- `FrameScheduler` 已经承担帧栈与子 Agent 上下文调度
- `KoakumaRuntime` 已经承担 MTP / Tool / syscall 执行环境

因此，这次设计要回答的是另一个更精确的问题：

> 既然 Alice 已经事实性拥有 runtime，如何把它进一步收敛为与 v4 长期规划一致的 `AliceRuntime`，并把内部职责边界表达清楚？

本文的核心结论是：

- `AgentRuntimeHost` 不应继续作为长期命名保留
- `AliceRuntime` 应成为 Alice 子系统唯一公开的 runtime 聚合根
- `AliceRuntime` 内部应显式持有两个子运行时：
  - `AgentRuntime`：负责 Agent 执行循环与帧栈调度
  - `KoakumaRuntime`：负责 MTP / Tool / syscall 执行环境

这意味着本次收敛的目标不是 “重写执行引擎”，而是：

> **在不破坏既有行为的前提下，完成 Alice 子系统的运行时命名收敛、职责显式化与结构定型。**

***

## 2. 背景与现状

### 2.1 当前代码中的事实结构

当前 Alice 子系统实际上已经形成如下结构：

```text
AliceSystem
  -> AliceService
  -> AgentRuntimeHost
       -> KoakumaRuntime
       -> FrameScheduler
       -> WorkerAgentService
       -> KernelLoopExecutor
       -> AgentProfileCache
```

从职责上看：

- `AliceSystem` 已经是子系统宿主
- `AliceService` 已经是对外门面
- `AgentRuntimeHost` 已经是内部运行时聚合器

也就是说，Alice 当前并不是 “没有 runtime”，而是 “runtime 还没有完成最终命名和结构表达”。

### 2.2 当前结构的主要问题

虽然现状已经接近目标，但仍存在 4 个问题。

#### 问题 1：`AgentRuntimeHost` 不是长期语义

`Host` 这个命名更像过渡期术语，它表达了 “收容对象” 的事实，但没有表达清楚：

- 它是 Alice 的正式 runtime
- 它持有运行时状态
- 它是未来对齐 `PatchouliRuntime` 的结构落点

长期看，这层应被正式命名为 `AliceRuntime`。

#### 问题 2：执行 runtime 与工具 runtime 仍混在同一层级表达

当前 `AgentRuntimeHost` 同时持有两类不同性质的对象：

- Agent 执行相关：
  - `KernelLoopExecutor`
  - `FrameScheduler`
  - `WorkerAgentService`
- 工具环境相关：
  - `KoakumaRuntime`
  - syscall/tool registry
  - interaction state

这会使 “谁负责执行循环” 和 “谁负责工具调用环境” 的边界仍然不够显式。

#### 问题 3：runtime 内部层次尚未清晰表达

当前读代码时，容易把 `KernelLoopExecutor` 误读为 Alice 的主 runtime，把 `KoakumaRuntime` 误读为其附属 helper。

但更合理的理解应是：

- `KernelLoopExecutor` 属于执行态 runtime
- `KoakumaRuntime` 属于工具态 runtime
- 它们都只是 `AliceRuntime` 的内部组成部分

#### 问题 4：System / Runtime 的边界还可以继续收口

当前 `AliceSystem` 仍直接负责本地 routes 的注册与注销。

而在 `Patchouli` 的最终形态中，本地 routes 已经由 `PatchouliRuntime` 自持有并挂载，`System` 只负责生命周期委托。

Alice 最终也应向这个方向收敛。

***

## 3. 设计目标

本设计只做 5 件事。

### 3.1 以 `AliceRuntime` 替代 `AgentRuntimeHost`

明确 Alice 的运行时聚合根，并与 `PatchouliRuntime` 在结构表达上对齐。

### 3.2 在 `AliceRuntime` 内部显式区分两类运行时

将当前运行时对象图明确分为：

- `AgentRuntime`
- `KoakumaRuntime`

使执行态和工具态不再混在同一个语义层次上。

### 3.3 保持 `AliceService` 作为对外能力门面

`AliceService` 继续只暴露：

- `run_agent(...)`
- `run_agent_stream(...)`
- `register_preretrieval_aliases(...)`
- `get_interaction_state(...)`

不让其重新膨胀为运行时容器。

### 3.4 保持 `AliceSystem` 作为子系统宿主

`AliceSystem` 继续只负责：

- runtime / service 装配
- public route 注册
- 生命周期桥接

不直接理解内部执行引擎细节。

### 3.5 保持顶层取消控制边界不下沉

`cancel_generation` 的事件注册表继续保留在 `ChatApplicationService`。

`AliceRuntime` 负责消费 `cancel_event`，但不重新持有 generation registry。

这与当前顶层应用服务的职责边界保持一致，也符合既有工程约束。

***

## 4. 核心设计结论

### 4.1 总体结构

目标结构如下：

```text
AliceSystem
  -> AliceRuntime
      -> AgentRuntime
          -> KernelLoopExecutor
          -> FrameScheduler
          -> WorkerAgentService
      -> KoakumaRuntime
      -> AgentProfileCache
      -> AliceBus
  -> AliceService
```

### 4.2 一句话解释

- `AliceRuntime` 是 Alice 的总 runtime 聚合根
- `AgentRuntime` 是执行 runtime
- `KoakumaRuntime` 是工具 runtime

这种结构表达的是：

> **Alice 是一个 Control Runtime，而这个 Control Runtime 内部又天然包含执行环境与工具环境两类子运行时。**

### 4.3 为什么不是直接让 `KoakumaRuntime` 并入 `AgentRuntime`

不建议把 `KoakumaRuntime` 合并进 `AgentRuntime`，原因有三点：

- `KoakumaRuntime` 的职责边界本身已经足够清晰，且明显不同于执行循环
- 它持有 interaction state、tool registry、syscall 语义、权限与缓存，不属于单纯的 loop 子组件
- 将其并入 `AgentRuntime` 会再次模糊 “执行调度” 与 “工具环境” 的边界

因此，更合适的结构是：

- `AgentRuntime` 管执行
- `KoakumaRuntime` 管工具环境
- `AliceRuntime` 统一持有两者

***

## 5. 角色定义

### 5.1 `AliceRuntime`

#### 角色

- Alice 子系统的正式 runtime 聚合根
- Alice 内部运行时状态与对象图的唯一宿主
- `AliceService` 的唯一 runtime 依赖

#### 持有内容

- `AliceBus`
- `KoakumaRuntime`
- `AgentRuntime`
- `AgentProfileCache`
- runtime-owned health / status
- local route registration state

#### 应负责

- 创建并接线 `AgentRuntime` 与 `KoakumaRuntime`
- 为 `AliceService` 提供统一 runtime 级 API
- 提供 interaction state 导出入口
- 提供预检索记忆别名注入入口
- 持有并挂载 Alice 私有 local routes
- 汇总 runtime health/status

#### 不应负责

- 顶层 chat 编排
- Patchouli prepare/finalize 逻辑
- generation cancel registry

### 5.2 `AgentRuntime`

#### 角色

- Agent 执行态 runtime
- 承载一次或多次 Agent 运行的执行循环与帧栈调度能力

#### 应持有

- `KernelLoopExecutor`
- `FrameScheduler`
- `WorkerAgentService`

#### 应负责

- `run_agent(...)`
- `run_agent_stream(...)`
- 主帧 / 子帧创建与推进
- CALL 派生与恢复
- 流式与非流式执行主循环
- `cancel_event` 消费

#### 不应负责

- tool / syscall registry
- MTP interaction state 的长期持有
- 记忆域 prepare/finalize

### 5.3 `KoakumaRuntime`

#### 角色

- MTP / Tool Runtime
- Agent 执行环境中的工具语义解释器与 syscall 执行环境

#### 应继续负责

- MTP 指令解析
- tool / syscall registry
- 权限沙箱
- WRITE / UPDATE 延迟捕获
- interaction traces
- `write_focus` / `update_focus`
- alias / atom cache

#### 不应负责

- 主执行循环
- 帧栈调度
- 顶层对外门面

### 5.4 `AliceService`

#### 角色

- Alice 对外用例门面

#### 应负责

- 将对外请求转发给 `AliceRuntime`
- 保持稳定的公开能力语义
- 不暴露内部双 runtime 结构

#### 不应负责

- 持有 runtime 对象图
- 维护执行态状态
- 直接依赖 `AgentRuntime` 与 `KoakumaRuntime` 两者

### 5.5 `AliceSystem`

#### 角色

- Alice 子系统宿主

#### 应负责

- 创建 `AliceRuntime`
- 创建 `AliceService`
- 注册 public routes
- 在 `start()/stop()` 中委托 runtime 挂载与卸载 local routes

#### 不应负责

- 直接操作 loop executor / frame scheduler / koakuma
- 理解内部执行态细节

***

## 6. 依赖方向

为了避免双重聚合层重新退化为 “上帝对象”，本设计要求依赖方向固定如下：

```text
AliceSystem
  -> AliceRuntime
  -> AliceService

AliceService
  -> AliceRuntime

AliceRuntime
  -> AgentRuntime
  -> KoakumaRuntime
```

关键约束如下：

- `AliceService` 只能依赖 `AliceRuntime`
- `AliceSystem` 只能持有 `AliceRuntime` 与 `AliceService`
- `AgentRuntime` 可以使用 `KoakumaRuntime`，但不反向成为其宿主
- `KoakumaRuntime` 不应依赖 `AgentRuntime`

一句话说就是：

> **执行 runtime 可以调用工具 runtime，但工具 runtime 不反向接管执行 runtime。**

***

## 7. 建议 API 边界

### 7.1 `AliceService` 对外 API

建议保持现有对外语义不变：

```python
class AliceService:
    async def run_agent(...)
    async def run_agent_stream(...)
    async def register_preretrieval_aliases(...)
    async def get_interaction_state(...)
```

### 7.2 `AliceRuntime` 对内 API

建议由 `AliceRuntime` 提供统一 runtime facade：

```python
class AliceRuntime:
    async def run_agent(...)
    async def run_agent_stream(...)
    def register_preretrieval_aliases(...)
    def export_interaction_state(...)
    def mount_local_routes(service: "AliceService") -> None
    def unmount_local_routes() -> None
    def health() -> dict[str, Any]
```

说明：

- `AliceService.run_agent(...)` 仍是对外门面
- 实际执行委托给 `AliceRuntime.run_agent(...)`
- `AliceRuntime` 再进一步委托给 `AgentRuntime`

### 7.3 `AgentRuntime` 对内 API

建议聚焦执行态能力：

```python
class AgentRuntime:
    async def run_agent(...)
    async def run_agent_stream(...)
    def health() -> dict[str, Any]
```

其内部再持有：

- `KernelLoopExecutor`
- `FrameScheduler`
- `WorkerAgentService`

这里不建议把 `ExecutionFrame` 创建细节直接暴露给 `AliceService` 或 `AliceSystem`。

### 7.4 `KoakumaRuntime` API

`KoakumaRuntime` 继续保留既有协议与工具语义接口，不要求在本阶段做大改。

当前更重要的是明确其角色，而不是重写其 API 面。

***

## 8. 交互状态与控制边界

### 8.1 interaction state 归属

interaction state 继续由 `KoakumaRuntime` 持有，包括：

- `mtp_traces`
- `write_focus`
- `update_focus`

但导出入口由 `AliceRuntime` 统一包装：

```text
AliceService.get_interaction_state()
  -> AliceRuntime.export_interaction_state()
      -> KoakumaRuntime.get_*
```

这样可以保证：

- 对外只有 `AliceService`
- 对内只有 `AliceRuntime` 直接了解双 runtime 结构

### 8.2 取消控制边界

取消控制继续保持当前模式：

```text
ChatApplicationService
  -> generation_id -> asyncio.Event registry
  -> AliceService.run_agent_stream(cancel_event=...)
  -> AliceRuntime / AgentRuntime 消费 cancel_event
```

本设计明确不把 generation registry 下沉到 `AliceRuntime`。

理由是：

- 取消控制属于顶层应用服务语义
- Alice 只负责执行过程中的响应
- 继续下沉会重新耦合顶层控制面与子系统 runtime

***

## 9. 路由与生命周期收敛

### 9.1 local routes

建议 Alice 参考 Patchouli 的最终形态，将 local routes 的挂载下沉到 `AliceRuntime`：

- `AliceSystem.start()` 调用 `AliceRuntime.mount_local_routes(self._service)`
- `AliceSystem.stop()` 调用 `AliceRuntime.unmount_local_routes()`

这样 `AliceSystem` 不再直接理解 local route 细节。

### 9.2 public routes

public routes 仍由 `AliceSystem` 注册到全局 bus。

原因是：

- public routes 本身属于子系统宿主的公开契约
- 这部分和 Patchouli 当前模式一致

### 9.3 health/status

建议 `AliceRuntime.health()` 返回结构化健康状态，而不是简单的 `"ok"` 字符串集合。

示意：

```python
{
    "local_routes_registered": True,
    "agent_runtime": {
        "loop_executor": "ok",
        "frame_scheduler": "ok",
        "worker_agent": "ok",
    },
    "koakuma_runtime": {
        "status": "ok",
        "interaction_state_loaded": True,
    },
}
```

这里不要求一次到位做复杂监控，但建议至少完成：

- 组件分组
- runtime 层级表达
- route mount 状态表达

***

## 10. 文件布局建议

建议在不大面积移动文件的前提下，先完成语义收敛。

### 10.1 推荐目录形态

```text
alice/
  system.py
  service.py
  runtime/
    __init__.py
    runtime.py           # AliceRuntime
    agent_runtime.py     # AgentRuntime
    koakuma.py
    loop_executor.py
    frame_scheduler.py
    worker_agent.py
    execution_frame.py
    cache.py
    bus.py
    syscalls/
```

### 10.2 过渡期兼容策略

建议按以下顺序收敛：

1. 新增 `runtime.py`，引入 `AliceRuntime`
2. 新增 `agent_runtime.py`，承接执行态组合
3. 将 `host.py` 保留为短期兼容壳，最终删除
4. 更新 `system.py` / `service.py` / `runtime/__init__.py` 的导入与注释

如果希望减少过渡期文件数，也可以直接将 `host.py` 重命名为 `runtime.py`。

但无论采用哪种文件级策略，长期类语义应收敛为：

- `AliceRuntime`
- `AgentRuntime`
- `KoakumaRuntime`

而不是：

- `AgentRuntimeHost`
- `AliceRuntimeHost`

***

## 11. 迁移步骤建议

### Step 1

引入 `AliceRuntime`，以其替代 `AgentRuntimeHost` 作为系统持有的 runtime 根对象。

此步骤优先解决命名与依赖入口问题，不要求立即改变内部行为。

### Step 2

从当前 host 结构中抽出 `AgentRuntime`，将以下内容内聚进去：

- `KernelLoopExecutor`
- `FrameScheduler`
- `WorkerAgentService`
- 执行相关 health

### Step 3

让 `AliceRuntime` 持有：

- `AgentRuntime`
- `KoakumaRuntime`
- `AgentProfileCache`
- local route mount state

并通过 `AliceRuntime` 暴露统一 runtime facade。

### Step 4

将 `AliceSystem` 中的 local route 注册逻辑下沉到 `AliceRuntime`。

### Step 5

将 `AliceService.get_interaction_state()` 改为委托 `AliceRuntime.export_interaction_state()`，完成状态导出边界统一。

### Step 6

清理命名、注释与导出残留，包括：

- `runtime/__init__.py`
- 文档注释中的 `Host` 残留
- 类型标注中的旧类名

***

## 12. 非目标

本设计明确不做以下事情：

- 不重写 `KoakumaRuntime` 的 MTP 协议实现
- 不修改 Phase D 顶层 `prepare -> run -> finalize` 主链路
- 不把 generation registry 从 `ChatApplicationService` 下沉到 Alice
- 不重新引入 `Bootstrap`、`LifecycleManager`、独立 `RuntimeHost` 抽象
- 不为了形式对称，把 Alice 硬拆成和 Patchouli 完全同构的内部对象图

一句话说：

> **本次演进是运行时边界收敛，不是行为重写。**

***

## 13. 风险与规避

### 风险 1：`AgentRuntime` 变成空壳

如果 `AgentRuntime` 只是简单转发 `KernelLoopExecutor`，这层就会失去存在价值。

规避方式：

- 让 `AgentRuntime` 明确持有执行态对象图
- 让 `run_agent(...)` / `run_agent_stream(...)` 成为其正式边界

### 风险 2：双 runtime 反向依赖

如果 `KoakumaRuntime` 开始回调或持有 `AgentRuntime`，容易形成结构回环。

规避方式：

- 固定依赖方向：`AgentRuntime -> KoakumaRuntime`
- 禁止 `KoakumaRuntime -> AgentRuntime`

### 风险 3：`AliceRuntime` 重新膨胀成上帝对象

如果所有逻辑又重新回到 `AliceRuntime`，那么内部拆分将失去意义。

规避方式：

- `AliceRuntime` 只做聚合与统一 facade
- 执行逻辑留在 `AgentRuntime`
- 工具语义留在 `KoakumaRuntime`

### 风险 4：系统层重新理解内部细节

如果 `AliceSystem` 继续直接接触内部 routes 或 loop 组件，会削弱收敛效果。

规避方式：

- `System` 只依赖 `AliceRuntime`
- local routes 下沉到 runtime

***

## 14. 验收标准

当以下条件成立时，可认为 Alice Runtime 收敛已基本落地：

- `AgentRuntimeHost` 已不再作为长期类名保留
- `AliceSystem` 明确持有 `AliceRuntime`
- `AliceService` 只依赖 `AliceRuntime`
- `AliceRuntime` 明确持有 `AgentRuntime` 与 `KoakumaRuntime`
- `AgentRuntime` 已承担执行态入口与相关 health 边界
- `KoakumaRuntime` 继续承担 MTP / Tool Runtime 边界
- interaction state 已通过 `AliceRuntime` 统一导出
- local routes 已由 `AliceRuntime` 挂载与卸载
- 顶层 `ChatApplicationService` 的取消 registry 仍保持不变

***

## 15. 最终建议

本设计建议采用如下判断作为 Alice 收尾工作的主导原则：

> **不要把这次改造理解为“再造一个新 runtime”，而应理解为“把已经存在的 Alice 运行时正式命名、内部拆层，并收敛到长期可维护结构”。**

对应到实现上，就是：

- 用 `AliceRuntime` 替换 `AgentRuntimeHost`
- 用 `AgentRuntime` 显式承接执行态 runtime
- 保留 `KoakumaRuntime` 作为独立的 MTP / Tool Runtime
- 让 `AliceService` 与 `AliceSystem` 继续保持薄边界

这样既能承接当前代码现实，又能与 v4 的长期 `System -> Service -> Runtime` 收敛目标保持一致。
