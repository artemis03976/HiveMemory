# HiveMemory 第四次架构演进补充设计

> **归档说明**: 本文是 v4 演进过程中的阶段设计记录，保留用于追溯迁移背景与取舍。v4 当前最终结构、术语与实现准则已统一收敛到 [SystemArchitecture_v4_TopLevelSketch.md](./SystemArchitecture_v4_TopLevelSketch.md)。如本文与最终总纲冲突，以最终总纲为准。

**文档状态**: Archived (阶段设计记录)\
**所属演进**: 第四次架构演进补充部分\
**文档定位**: 在 `Phase B / Phase C / Phase D` 已基本完成的基础上，补充说明 `PatchouliKernel -> PatchouliRuntime` 的长期收敛方向，以及 `Alice` 子系统向 `System = Runtime + Service` 对称结构继续规整的目标。

**前置文档**:

- [SystemArchitecture\_v4\_TopLevelSketch.md](file:///c:/Users/29305/Projects/HiveMemory/docs/architecture_evolution/SystemArchitecture_v4_TopLevelSketch.md)
- [SystemArchitecture\_v4\_PatchouliSubsystemNormalization\_Design.md](file:///c:/Users/29305/Projects/HiveMemory/docs/architecture_evolution/SystemArchitecture_v4_PatchouliSubsystemNormalization_Design.md)
- [SystemArchitecture\_v4\_PhaseC\_AliceRuntimeFoundation\_Design.md](file:///c:/Users/29305/Projects/HiveMemory/docs/architecture_evolution/SystemArchitecture_v4_PhaseC_AliceRuntimeFoundation_Design.md)
- [SystemArchitecture\_v4\_PhaseD\_ChatApplicationServiceMigration\_Design.md](file:///c:/Users/29305/Projects/HiveMemory/docs/architecture_evolution/SystemArchitecture_v4_PhaseD_ChatApplicationServiceMigration_Design.md)

***

## 1. 文档定位

这份补充文档回答的核心问题不是“`PatchouliKernel` 还要不要继续删”，而是：

> **在 v4 主体演进已经完成后，Patchouli 与 Alice 两个子系统的长期稳定形态应该是什么？**

当前代码已经完成了几项关键收敛：

- `ChatApplicationService` 已成为顶层主动 chat 编排入口
- `PatchouliService` 已收缩为记忆准备 / interaction 提交门面
- `PatchouliKernel` 内部历史总线回环和大量纯直通方法已被清理
- Alice 已通过 runtime host 形式承接执行循环、Koakuma 与 WorkerAgent 相关运行环境

但这同时引出了新的问题：

- `PatchouliKernel` 这个名字已经越来越不贴近当前职责
- 当前 `PatchouliKernel` 更像“组件装配容器”，而不是传统意义上的 kernel
- 若直接改名为 `PatchouliRuntime`，当前职责又还不够完整，容易形成新的语义错位
- Alice 虽然已具备 runtime host 结构，但对外分层命名与 Patchouli 还不够完全对称

因此，这份文档将 `Phase D` 之后的长期方向明确为：

> **把 Patchouli 从“历史 Kernel 残影”继续演化为真正的 Capability Runtime，并让 Alice 与 Patchouli 在更高层次上收敛到统一的** **`System -> Service -> Runtime`** **结构。**

***

## 2. 长期目标

长期目标不是重新引入 `RuntimeHost`、`Bootstrap`、`LifecycleManager` 一类额外抽象，而是：

- 继续保持当前 v4 已形成的低抽象、显式装配风格
- 让 `Runtime` 自身成为子系统内部运行环境宿主
- 让 `System` 退回到真正的子系统宿主职责
- 让 `Service` 退回到用例编排职责

目标结构如下：

```text
HiveMemorySystem
  -> PatchouliSystem
      -> PatchouliRuntime
      -> PatchouliService
  -> AliceSystem
      -> AliceRuntime
      -> AliceService
```

其中：

- `System` 负责子系统宿主、顶层生命周期与总线接入
- `Service` 负责对外用例编排和边界输入输出
- `Runtime` 负责子系统内部组件图、运行期状态、内部能力边界与运行环境管理

***

## 3. Patchouli 的长期目标形态

### 3.1 当前问题

当前 `PatchouliKernel` 已经不再是旧架构中的星型微服务调度中心。

它现在主要承担：

- storage / librarian llm / reranker 初始化
- retrieval / perception / generation / lifecycle 引擎构建
- `RetrievalFamiliar` / `LibrarianCore` 组件持有
- 模型 warmup / readiness / health
- 少量残余能力方法，如检索聚合与 agent profile 加载

这意味着当前类已经更接近“运行环境容器”，但仍缺少几项真正属于 runtime 的职责：

- local bus 的归属
- 内部 route mount / unmount
- shutdown drain
- runtime 生命周期状态
- 内部能力边界的统一导出

### 3.2 长期建议

将 `PatchouliKernel` 重命名并演化为 `PatchouliRuntime`，使其成为：

> **Patchouli 子系统的能力运行环境宿主。**

长期职责建议如下。

### 3.3 PatchouliRuntime 应持有的职责

#### a. 组件图持有

`PatchouliRuntime` 直接持有：

- storage
- librarian llm
- reranker
- retrieval engine
- perception layer
- generation engine
- lifecycle engine
- `RetrievalFamiliar`
- `LibrarianCore`

这部分已经基本存在，应继续保留。

#### b. local bus 归属

当前 `PatchouliSystem` 中的 `_local_bus` 应长期下沉到 `PatchouliRuntime`。

原因：

- local bus 代表 Patchouli 内部能力边界
- 它服务的是 runtime 内部对象图
- 它不属于子系统宿主层的职责

长期上应由 `PatchouliRuntime` 自己持有并管理：

- `mount_local_routes()`
- `unmount_local_routes()`
- `list_local_routes()`

#### c. 运行期状态

`PatchouliRuntime` 应统一持有运行期状态，而不是只保留模型 readiness：

- created
- starting
- started
- warming\_up
- stopping
- drained
- stopped

这使得 `health()` 不再只是模型是否 ready，而是运行环境整体状态。

#### d. warmup / health / drain

以下能力应长期统一归到 `PatchouliRuntime`：

- `warmup_models()`
- `health()`
- `shutdown_drain()`

尤其 `shutdown_drain()` 当前直接操作 perception/librarian 内部状态，本质上更像 runtime 行为，而不是 subsystem host 行为。

### 3.4 对当前残余方法的建议

#### `handle_hot()`

当前 `handle_hot()` 的真实语义已经不再是“处理热路径”，而更接近：

- 从 `EyeGazeResult` 构建 `RetrievalRequest`
- 执行 retrieval
- 将结果包装成统一的结果模型

长期建议：

- 保留这项能力，但改名
- 让名字反映“检索聚合”而不是“热路径处理”

建议名称：

- `retrieve_for_gaze()`
- 或 `resolve_memory_context()`

推荐 `retrieve_for_gaze()`，语义最直接。

#### `load_agent_profile()`

当前 `load_agent_profile()` 本质是 `storage.get_memory_by_alias()` 的领域化包装。

长期有两种合理方案：

- 保留在 `PatchouliRuntime`，作为 runtime 提供的一项基础能力
- 或抽成独立的 `AgentProfileLoader` / `AgentProfileRepository`，由 `PatchouliRuntime` 持有

推荐第二种：

- `PatchouliRuntime.agent_profiles.load(alias)`

这样 runtime 自身保持清晰，不继续堆积杂项 helper。

### 3.5 PatchouliService 的长期边界

`PatchouliService` 不应回收 runtime 职责，而应继续收敛为用例编排门面。

长期保持以下职责即可：

- `prepare_agent_run(...)`
- `finalize_agent_run(...)`
- `cleanup_prepared_agent_run(...)`
- `manual_archive_topic(...)`
- `analyze_and_retrieve(...)`

### 3.6 PatchouliSystem 的长期边界

`PatchouliSystem` 应收缩为真正的 subsystem host。

长期职责建议只保留：

- 持有 `PatchouliRuntime`
- 持有 `PatchouliService`
- 将 public routes 接入 `GlobalSystemBus`
- 将 maintenance scheduler 与 runtime 对接
- 实现 `start/stop/health`

不再长期持有：

- local bus
- local route mount 细节
- shutdown drain 的底层逻辑

***

## 4. Alice 的长期对称化目标

### 4.1 长期目标

Alice 的长期方向不是大改边界，而是：

> **把当前已基本成立的 runtime host 形式进一步规整到与 Patchouli 对称的** **`System -> Service -> Runtime`** **结构。**

目标结构：

```text
AliceSystem
  -> AliceRuntime
  -> AliceService
```

### 4.2 AliceRuntime 应长期持有的职责

`AliceRuntime` 长期应显式成为 Agent 运行环境宿主，持有：

- loop executor
- worker agent service
- Koakuma runtime
- tool / syscall registry
- interaction state
- runtime-owned health / status

### 4.3 AliceService 的长期边界

`AliceService` 长期继续承接对外能力门面：

- `run_agent(...)`
- `run_agent_stream(...)`
- runtime interaction state 导出
- cancel / control 相关对外边界

也就是说，Alice 的对外语义应逐渐收敛为：

- `Service` 是运行时能力的公开用例门面
- `Runtime` 是执行环境本体

### 4.4 与 Patchouli 的对称关系

两者不需要“做相同的事”，但需要“采用相同的层次结构”。

差异是允许且合理的：

- Alice 是 Control Runtime
- Patchouli 是 Capability Runtime

它们不要求内部对象图相同，但长期应共享相同的宿主结构：

```text
System = subsystem host
Service = use-case facade
Runtime = internal runtime environment
```

***

## 5. 非目标

本补充设计明确不做以下事情：

- 不重新引入 `RuntimeHost` 独立壳层
- 不重新引入独立 `Bootstrap` 抽象
- 不重新引入独立 `LifecycleManager` 抽象
- 不让 `PatchouliService` 重新膨胀为运行环境宿主

也就是说，长期方案的关键词是：

> **职责回收，但不回退到高抽象化。**

***

## 6. 建议迁移顺序

### Step 1

完成 `PatchouliKernel -> PatchouliRuntime` 的命名收敛，同时保留兼容导出，先解决语义不一致问题。

### Step 2

将 `PatchouliSystem` 中的 `_local_bus`、`_register_local_routes()`、`_unregister_local_routes()` 下沉到 `PatchouliRuntime`。

### Step 3

将 `shutdown_drain()` 的 perception flush 逻辑下沉到 `PatchouliRuntime`，让 `PatchouliSystem.stop()` 只调用 runtime 生命周期动作。

### Step 4

将 `handle_hot()` 改名为 `retrieve_for_gaze()`，并明确其为 runtime 能力入口，而不是“热路径主处理器”。

### Step 5

将 `load_agent_profile()` 从 runtime 主类中拆出为独立 loader/repository，并由 runtime 持有。

### Step 6

在 Alice 侧完成命名与结构规整：

- `AliceRuntime`
- `AliceService`
- `AliceSystem`

并统一其内部健康、控制与状态边界。

***

## 7. 验收标准

当以下条件成立时，可认为长期 runtime 收敛方向已经落地：

- `PatchouliKernel` 已不再作为长期命名保留，而由 `PatchouliRuntime` 替代
- `PatchouliRuntime` 自己持有 local bus、route mount、warmup、health、shutdown drain
- `PatchouliSystem` 不再理解 Patchouli 内部 local route 细节
- `PatchouliService` 仍保持为用例编排门面，而不是运行时容器
- `handle_hot()` 已收敛为更贴切的检索能力命名，或被等价替代
- `load_agent_profile()` 已从 runtime 主类中拆出或明确归位
- Alice 侧已能明确表达 `System / Service / Runtime` 三层结构

***
