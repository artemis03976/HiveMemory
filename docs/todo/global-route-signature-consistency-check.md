---
title: 总线路由的类型化与静态签名检查
status: todo
owner: system
scope: typed-bus-routes-and-static-call-site-checking
related_docs:
  - docs/components/runtime-and-bus.md
  - docs/contracts/routes-and-events.md
  - docs/contracts/subsystem-contracts.md
  - docs/system/attachments.md
  - docs/archive/plans/v0.7.0-identity-access-batch-2.md
last_reviewed: 2026-10-04
---

# 总线路由的类型化与静态签名检查

## 问题与证据

`AsyncSystemBus.request(route, *args, **kwargs)` 以 `handler(*args, **kwargs)` 直接调用；route 是字符串常量，调用点与 handler 签名之间没有静态类型约束，全靠人工保持一致。`GlobalSystemBus` 与各子系统的本地总线（如 `PatchouliBus`、Gateway 的本地总线）都继承这一调用方式。

- **2026-09-12（W1 附件）**：`ChatApplicationService` 向 `PATCHOULI_PREPARE_AGENT_RUN` 传 `attachments=`，handler 的形参名为 `selected_attachments=`，运行时才以 `TypeError` 暴露。Chat 流单元测试为该路由注册了 `**kwargs` 替身，prepare 层测试直接调用 handler，两条链在“总线绑定真实 handler”这一接缝上没有交叉验证（回归测试：`tests/integration/system/test_workspace_asset_chat_selection.py::test_chat_bus_route_reaches_real_prepare_with_attachments`）。
- **2026-10-04（身份第二批）**：Patchouli 本地路由的参数从 `identity_scope` 整体改为 `belong_to` / `from_actor`。其中 7 个调用点按位置传入归属；按位置传入错误类型（例如仍传 `IdentityScope`）时不会报错，handler 可能把它当作字典键或比较对象静默使用（见[已归档的第二批计划](../archive/plans/v0.7.0-identity-access-batch-2.md)）。

## 已有的运行时检查

2026-10-04 起，总线在调用 handler 前按其签名与类型标注只校验、不转换参数（[运行时机制](../components/runtime-and-bus.md)第 1 节“参数检查”）：名称、数量与可用 `isinstance` 表达的类型不符时抛出 `RouteArgumentError`。它覆盖所有总线，但只在请求实际发生时生效，因此：

- 只有被测试或运行实际走到的调用点才会被检查，绑定 `**kwargs` 替身的测试仍会放过漂移；
- handler 签名改变后，旧调用点要到运行时才失败，不能在提交前发现全部受影响的调用点。

本 Todo 处理剩余的静态部分。

## 方案：类型化 route 与 mypy

把 route 常量从字符串改为携带签名的类型化对象，由 mypy 检查注册与全部调用点：

```python
P = ParamSpec("P")
R = TypeVar("R")

class Route(Generic[P, R]):
    name: str

class AsyncSystemBus:
    def register(self, route: Route[P, R], handler: Callable[P, Awaitable[R]]) -> None: ...
    async def request(self, route: Route[P, R], *args: P.args, **kwargs: P.kwargs) -> R: ...
```

- 每条 route 的签名在契约处声明一次（例如以只声明签名的协议函数构造 `Route`）；`register` 要求 handler 与之匹配，`request` 按其检查参数名、数量、类型与返回值。
- 2026-10-04 以项目的 mypy 验证过这一形状：handler 签名漂移、按位置传入错误类型（`IdentityScope` 传给 `WorkspaceIdentity`）与关键字拼错三种情况都在静态检查时报错，正确调用通过。
- 规模（2026-10-04）：全局 route 29 条、Patchouli 本地 route 35 条，另有 Gateway 本地 route；`request` 调用点约 76 处。
- 总线的运行时调用语义不变（仍是直接调用、不转换参数）；运行时参数检查保留，作为 mypy 之外的兜底。

## 前置条件

- **mypy 门禁**：CI 目前只运行 pytest、ruff 与 black，不运行 mypy，`mypy src` 有约 209 个既有错误。没有门禁时类型化 route 不提供任何保护。owner 于 2026-10-04 决定：本事项与 mypy 错误修复分支一同完成；门禁至少覆盖 route 契约模块与全部 `request` 调用点所在模块。
- handler 与调用点模块的类型标注需要能被 mypy 解析（只在 `TYPE_CHECKING` 下导入的名称对 mypy 可见，不影响静态检查）。

## 约束

- 不改变总线的调用语义，不引入参数转换或默认值注入；
- handler 侧的 `(*args, **kwargs)` 代理（如 Alice 的本地到全局路由代理）以 `Route[..., R]` 豁免，被代理的 route 在全局总线一侧检查；
- 替身测试可以继续存在，但替身也要满足 route 的签名类型。

## 完成条件

- 全局总线与各子系统本地总线的 route 都是类型化对象，`register` 与 `request` 只接受类型化 route；
- W1 附件场景（调用方传 `attachments`、handler 收 `selected_attachments`）与身份第二批场景（按位置把 `IdentityScope` 传给 `WorkspaceIdentity` 形参）都在 CI 的 mypy 检查中失败；
- CI 运行覆盖上述模块的 mypy 门禁，既有测试套件全量通过。
