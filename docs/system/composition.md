---
title: System Composition and Lifecycle
status: current
owner: system
scope: composition-root-and-lifecycle
code_paths:
  - src/hivememory/system/assembler.py
  - src/hivememory/system/system.py
  - src/hivememory/system/contracts/subsystem.py
related_contracts:
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/routes-and-events.md
  - docs/architecture/boundaries.md
last_reviewed: 2026-07-28
---

# System 组合根与生命周期

`HiveMemorySystem` 是进程内顶层宿主。它把 Gateway、Patchouli、Alice、应用服务和全局运行时放到同一条可推理的生命周期里，但不替任何子系统执行领域工作。

这个边界是项目从旧对象图中抽离出来的关键结果。若 HTTP router 或某个子系统自己装配依赖，就会出现多套总线、不同的取消状态和不一致的启停顺序；若 System 进一步拥有记忆算法、Gateway workflow 或 Agent loop，又会重新成为一个无法替换的全能总管。因此 System 只拥有组合关系、进程级状态和跨子系统用例。

## 1. 组合结构

`SystemAssembler.assemble()` 按四层生成中间产物，最后交给 `HiveMemorySystem`：

```text
HiveMemorySystem.build(config)
  -> SystemAssembler
       -> runtime bundle
            GlobalSystemBus
            GlobalMaintenanceScheduler
            RuntimeEventBus / NullRuntimeEventSink
       -> registries bundle
            ProviderRegistry
            ModelRegistry
       -> subsystem bundle
            GatewaySystem
            PatchouliSystem
            AliceSystem
       -> application-service bundle
            Chat / PassiveIngress / Memory / MemoryTask
            Agent / Topic / Readiness services
```

四个 Bundle 是装配器的私有交接对象，不是公共协议。它们的作用是让依赖顺序显式可读：运行时先存在，注册表再解析模型配置，子系统共享全局基础设施，应用服务最后只拿到公共总线和必要配置。

### 1.1 Runtime bundle

- `GlobalSystemBus`：跨子系统 public route 的 RPC/Pub/Sub 交接面；
- `GlobalMaintenanceScheduler`：在当前主 `asyncio` loop 调度维护任务；
- `RuntimeEventBus`：启用时保存有界观测事件和订阅队列；
- `NullRuntimeEventSink`：观测关闭时的无副作用替代实现。

观测设施和业务总线在装配阶段就分开，是为了让 RuntimeEvent 的失败不会阻塞一次正常业务调用。

### 1.2 Registry bundle

`ProviderRegistry` 保存 provider 凭证与环境配置的合并结果；`ModelRegistry` 根据模型引用解析 Gateway/Librarian 的 LLM 配置。System 在子系统构建前完成这一步，使 Gateway 和 Patchouli 使用已经解析过的配置，而不是在请求路径里各自重新读取凭证。

注册表是 System 的全局资源，但它不拥有模型调用，也不决定某次请求使用哪条业务流程。具体运行时仍由 Gateway、Patchouli 或 Alice 按各自契约消费。

### 1.3 Subsystem bundle

三个子系统以平级宿主装配：

| 子系统 | System 负责的部分 | 子系统自己负责的部分 |
|:---|:---|:---|
| Gateway | 注入配置、全局总线和观测 sink | GatewayRuntime、命令、上下文、workflow 与公共 process route |
| Patchouli | 注入配置、全局总线、维护调度器和观测 sink | 记忆、话题、检索、感知、生成任务与 prepare/finalize |
| Alice | 注入配置、全局总线、模型注册表和观测 sink | Agent run、frame、MTP、工具和 PendingAtom 运行时 |

System 不通过这些宿主的具体 Runtime 互相串联；跨边界链路由应用服务通过 `GlobalSystemBus` 发起。

## 2. 启动顺序

当前 `HiveMemorySystem.start()` 的有效顺序是：

```text
Gateway.start
  -> Patchouli.start
  -> Alice.start
  -> GlobalMaintenanceScheduler.start
  -> PassiveIngressService.start
  -> SYSTEM_READY
```

顺序背后的理由是：先挂载三个子系统的 route，使应用服务有可用的交接面；再启动调度器；最后让 Passive Ingress 注册自己的 idle flush 任务。`start()` 要求在运行中的 `asyncio` loop 内执行，因为维护调度器不创建隐藏 event loop。

重复 `start()` 不再重复挂载 route 或重启 scheduler，而是发布一次带 `already_started=true` 的生命周期观测并返回。启动失败会发布 `system.start_failed`，保留已完成步骤和第一个未完成步骤后继续抛出原异常。

当前实现没有在 `start()` 失败后自动逆序回滚已启动的子系统；调用方不能把失败事件误读为已经完成清理。这个限制需要在后续生命周期增强中单独处理，不通过隐式补偿掩盖。

## 3. 停止顺序

当前 `stop()` 的有效顺序是：

```text
GlobalMaintenanceScheduler.stop
  -> PassiveIngressService.shutdown_drain
  -> Alice.stop
  -> Patchouli.stop
  -> Gateway.stop
  -> SYSTEM_STOPPED
```

先停调度器是为了阻止新的维护 tick；随后 Passive Ingress 封口并尽力提交仍在 outbox 中的 turn。只有被动摄入完成 shutdown drain 后，才关闭 Alice、Patchouli 和 Gateway，避免在仍有 sealed turn 待提交时撤掉 Patchouli route。

重复 `stop()` 会保持幂等：scheduler 已停止时不重复等待，未启动的系统仍会执行必要的被动 drain 并发布 `already_stopped=true`。任一步骤失败都会发布 `system.stop_failed`，记录已完成步骤、scheduler 状态和被动 drain 摘要后抛出异常。

## 4. 健康状态与公共入口

`HiveMemorySystem.health()` 汇总：

- 当前 System 是否处于 started 状态；
- Gateway、Patchouli、Alice 各自的 `health()`；
- Patchouli 模型是否 ready。

健康状态是观测和管理入口，不替代业务契约。模型尚未 ready 不等于所有 route 都不存在；反过来，健康返回 `ok` 也不保证一次具体检索或生成调用一定成功。

System 对外暴露的是应用服务属性和 registry/sink 查询，例如 `chat_service`、`ingress_service`、`memory_service`、`runtime_events`、`model_registry`。这些属性方便 HTTP 或其他 adapter 注入依赖，但 adapter 仍应调用应用服务，不应从属性继续下钻到子系统 Runtime。

## 5. 生命周期不变量与矛盾检查

- 所有 public route 的挂载和撤销必须由对应子系统宿主完成；
- 应用服务不直接持有 Gateway/Patchouli/Alice 实例；
- 维护任务必须在 scheduler 注册，不能由业务组件偷偷创建第二个 interval loop；
- `SYSTEM_READY` 只在所有启动步骤完成后发布，RuntimeEvent 失败不能改变这个判断；
- `SYSTEM_STOPPED` 的观测摘要不等于所有 outbox 已持久化，必须查看 `passive_shutdown_drain`；
- registry 解析失败、子系统启停失败和业务请求失败不能被统一降级成健康 `ok`。

评审新的组合代码时，优先检查是否出现第二个 GlobalSystemBus、应用层直连子系统 Runtime、启动失败后仍接受请求，或把 RuntimeEvent 当作控制信号的情况。

## 6. 验证入口

- `src/hivememory/system/assembler.py`
- `src/hivememory/system/system.py`
- `tests/unit/system/test_hivememory_system.py`
- `tests/unit/system/test_lifecycle.py`
- `tests/unit/system/contracts/test_contracts.py`
