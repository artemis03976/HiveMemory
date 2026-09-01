---
title: MTP 缓存命中作用域重验
status: todo
owner: alice
scope: mtp-alias-cache-scope-revalidation
code_paths:
  - src/hivememory/agent_runtime/aliases/resolver.py
  - src/hivememory/agent_runtime/pending_atom/runtime.py
  - src/hivememory/agent_runtime/aliases/cache.py
related_docs:
  - docs/contracts/mtp.md
  - docs/alice/mtp-runtime.md
  - docs/alice/pending-atom.md
  - docs/architecture/workspace.md
last_reviewed: 2026-09-01
---

# MTP 缓存命中作用域重验

## 问题与证据

`RuntimeAliasResolver` 的 L0 路径命中 `PendingAtomRuntime` 后，会直接把 pending 交给 `_resolve_pending_hit()`，当前没有比较 pending 自身 `runtime_scope.identity_scope` 与本次 `MTPExecutionContext.identity_scope`。`PendingAtomRuntime` 由 `AliceRuntime` 进程级共享，因此只要调用方知道另一个执行者的 pending alias，就可能在 READ/RUN 的解析链中看到不属于自己的临时写入意图、状态或 redirect 信息。

L1 `KoakumaAtomCache` 命中当前已经通过 `memory_is_readable()` 重验 Memory 的 Workspace ownership 与 actor policy；L2 冷查询也会由最终 Memory owner 和 resolver 防御性校验。因此本事项的直接缺口是 L0 pending 命中，不能退化为重新按 Workspace 拆分缓存或复制一套 alias store。

## 影响

- 不同 Workspace 之间可能通过共享的 pending alias 解析路径产生临时记忆信息泄漏；
- 同一 Workspace 内不同 actor 的 pending 写入意图也可能被错误解析，破坏 MTP alias 的执行者边界；
- pending 的 `failed`、`expired`、`discarded` 或 canonical redirect 状态可能被越权调用方观察到；
- 该问题会使 MTP 契约要求的“记忆访问使用调用方 `IdentityScope`”在 L0 路径上失效。

## 修复边界

在 `RuntimeAliasResolver` 的 L0 命中边界执行 scope 重验：只有 pending 的 `runtime_scope.identity_scope` 与当前 `MTPExecutionContext.identity_scope` 完全一致时，才允许继续解析 pending、redirect 或其终态。作用域不匹配时应按 alias 不可见处理，不泄漏 pending 的内容、状态、所属 Workspace 或是否存在。

修复只增加最终解析边界的校验，不改变以下既有架构：

- `PendingAtomRuntime`、`KoakumaAtomCache` 和 alias resolver 继续由 AliceRuntime 进程级共享；
- 不新增按 Workspace 复制的 cache、controller、命名域或额外协调器；
- L1/L2 的现有 Memory ownership 与 actor policy 重验继续保留；
- 不改变 pending 生命周期、settlement、回收或 `wait_all()` 的通用语义。

## 测试与完成条件

- 同一 scope 注册的 pending alias 仍可正常解析为 pending、redirect、discarded、failed 或 expired 结果；
- 相同 actor 但不同 Workspace 的调用方无法解析该 alias，且结果不泄漏 alias 所属状态或内容；
- 相同 Workspace 但不同 actor 的调用方同样无法解析该 alias；
- 越权 L0 命中不会继续触发 canonical atom 的读取、citation 或 RUN 执行；
- L1 shared atom cache 在跨 Workspace、跨 actor 场景下继续执行现有 ownership/actor policy 重验；
- 增加 resolver 链路的 unit/integration 回归测试，并覆盖 READ 与 RUN 两类消费者；
- 修复不引入新的缓存分区、持久化记录或跨 Store 控制组件。

实现完成后，应同步更新 [MTP 契约](../contracts/mtp.md)、[MTP Runtime](../alice/mtp-runtime.md) 和 [PendingAtom](../alice/pending-atom.md) 中的已知限制描述，并从本 Todo 链接到对应的测试入口。
